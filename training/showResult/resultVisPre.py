#coding:utf-8
import torch
import torch.nn as nn #专门为神经网络设计的模块化接口
from utils.data_utils import DataOrderScaner
from ..models import EncoderDecoder_without_dropout
import time, os, logging
from cluster.ClusteringModule import ClusterModule
from cluster import ClusterTool
import numpy as np
from sklearn import metrics


# 查找字典中指定的value的key
def get_keys(d, value):
    return [k for k, v in d.items() if v == value]

# 计算错分数量
def calculate_error(k, labels_dict, ori_labels_dict):
    count_sum = [0] * k
    count_correct = [0] * k
    count_error = [0] * k
    count_label = [0] * k
    res = []  # 记录错误的数据
    right = []
    error_sort = []
    for i in range(k):
        data_id = get_keys(labels_dict, i)  # 聚类结果中第i个簇（簇号为i+1）对应的数据的ID
        # data_id = get_keys(labels_dict, i)  # 聚类结果中第i个簇（簇号为i+1）对应的数据的ID
        count = [0] * k  # 新建一个k长的列表，每个值都是0，统计每个label出现的次数
        for j in data_id:
            # count[ori_labels_dict[j]] += 1
            count[ori_labels_dict[j]] = count[ori_labels_dict[j]] + 1

        count_most = count.index(max(count))  # 聚类结果中第i个簇（簇号为i+1）对应的数据的 原始的簇号 中 出现频率最高的

        # 找出错误的数据
        for p in data_id:
            if ori_labels_dict[p] != count_most:
                res.append(p)
                right.append(ori_labels_dict[p])
                error_sort.append(count_most)

        count_sum[i] = len(data_id)
        count_correct[i] = count[count_most]
        count_error[i] = count_sum[i] - count_correct[i]
        # count_label[i] = count_most + 1
        count_label[i] = count_most

    error_total = sum(count_error)  # 错分的总数
    return error_total,res,right, error_sort

def showPic(args):
    print("**************** use model to generate label and show result with picture************")

    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.device)  # 指定第几块显卡
    # 初始化需要评估的数据集
    scaner = DataOrderScaner(os.path.join(args.data_dir, "test.ori"), 
                             os.path.join(args.data_dir, "test.label"), 
                             args.batch)
    scaner.load()
    # 模型框架
    m0 = EncoderDecoder_without_dropout(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        m0.cuda()
        m1.cuda()
    # 加载模型参数
    assert os.path.isfile(args.best_cluster_model), f"load {args.best_cluster_model} failed, please do joint-training first"

    logging.info(f"=> loading best_model {args.best_model} @ {time.ctime()}")
    if args.cuda and torch.cuda.is_available():
        best_model = torch.load(args.best_model, map_location=f'cuda:{args.device}')
    else:
        best_model = torch.load(args.best_model, map_location=torch.device('cpu'))

    m0.load_state_dict(best_model["m0"])
    m1.load_state_dict(best_model["m1"])


    centroid, error_total, nmi, ari, labels_dict, ori_labels_dict = get_cluster_centroid(args,
                                                                                               args.best_model,
                                                                                               scaner,
                                                                                               scaner.get_data_num(),
                                                                                               save=args.save,
                                                                                               e=args.expId)

    print(
        "cluster error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{}******".format(
            error_total, round(nmi, 4), round(ari, 4)))
    centroid = torch.FloatTensor(centroid)
    if args.cuda and torch.cuda.is_available():
        centroid = centroid.cuda()

    labels = {}
    with open("../../data/train_"+str(args.expId)+".csv", 'w') as wf:
        for i in range(len(ori_labels_dict)):
            labels[i] = ori_labels_dict[i]
            wf.write(str(i+1)+","+str(ori_labels_dict[i])+","+str(labels_dict[i])+"\n")

    error_total,res,right, error_sort = calculate_error(args.clusterNum, labels_dict, labels)
    nmi = calculate_nmi(list(labels_dict.values()), list(labels.values()))
    ari = calculate_ari(list(labels_dict.values()), list(labels.values()))

    print(
        "cluster error_total:{} nmi:{} ari:{} ******".format(
            error_total, round(nmi, 4), round(ari, 4)))

# 计算nmi
def calculate_nmi(y_pred, y_true):
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return nmi

def calculate_ari(y_pred, y_true):
    score = metrics.adjusted_rand_score(y_pred, y_true)
    return score

def get_cluster_centroid(args, model, data, data_num, center=None, feature=None, save=False,e=-1):
    if feature is None:
        # 定义模型
        m0_2 = EncoderDecoder_without_dropout(args.vocab_size,
                                              args.embedding_size,
                                              args.hidden_size,
                                              args.num_layers,
                                              args.bidirectional)
        if args.cuda and torch.cuda.is_available():
            m0_2.cuda()
        m0_2_optimizer = torch.optim.Adam(m0_2.parameters(), lr=args.learning_rate)
        # 加载模型

        if args.cuda and torch.cuda.is_available():
            best_model = torch.load(model, map_location=f'cuda:{args.device}')
        else:
            best_model = torch.load(model, map_location=torch.device('cpu'))
        m0_2.load_state_dict(best_model["m0"])
        m0_2_optimizer.load_state_dict(best_model["m0_optimizer"])

        # 神经网络前向传播，获得降维后的特征
        data.start = 0  # 每一个epoch计算qij时，先让scaner的start指针归零
        m0_2.eval()
        feature = []
        labels = []
        i = 0
        while True:
            i = i + 1
            src, lengths, invp, label = data.getbatch_scaner()  # src[12,64] invp是反向索引
            if src is None: break
            if args.cuda and torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            # 计算encoder学习到的轨迹表示
            h, _ = m0_2.encoder(src, lengths)
            h = m0_2.encoder_hn2decoder_h0(h) # (num_layers, batch, hidden_size * num_directions)
            h = h.transpose(0, 1).contiguous()   # (batch, num_layers, hidden_size * num_directions)，例如 [64, 3, 256]
            h2 = h[invp]
            size = h2.size()
            # 使用三层特征拼接的特征
            h2 = h2.view(size[0], size[1]*size[2])
            feature.append(h2.cpu().data)
            labels.extend(label)
        feature = torch.cat(feature)
        feature = feature[data.shuffle_invp]
        labels = np.array(labels)[data.shuffle_invp]
    if not args.hasLabel:
        labels = None

    # k-means聚类，获取初始簇中心
    centroids, error_total, nmi, ari, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(feature.cpu().data, args.sourcedata, k=args.clusterNum,
                                                                     n=data_num, center=center, save=save,epoch=e,tsne_filename="pre",labels=labels)

    return centroids, error_total, nmi, ari, labels_dict, labels