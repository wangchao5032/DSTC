#coding:utf-8
import torch
import torch.nn as nn #专门为神经网络设计的模块化接口
from utils.data_utils import DataOrderScaner
from ..models import EncoderDecoder_without_dropout
import time, os, logging
from cluster.ClusteringModule import ClusterModule
from cluster import ClusterTool
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
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
    scaner = DataOrderScaner(os.path.join(args.data, "train.ori"), os.path.join(args.data, "train.label"), args.t2vec_batch)
    # or use test ori with test label
    scaner.load()
    # 模型框架
    m0 = EncoderDecoder_without_dropout(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),  # Sequential:按顺序构建网络
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        m0.cuda()
        m1.cuda()
    # 加载模型参数
    if not os.path.isfile(args.best_cluster_model):  # 如果完成了预训练，但是没有进行联合训练
        raise Exception
    else:
        print("=> loading best_cluster_model '{}'".format(args.best_cluster_model))
        logging.info("loading best_cluster_model @ {}".format(time.ctime()))
        best_cluster_model = torch.load(args.best_cluster_model)
        features = best_cluster_model["features"]  # 加载上一次保存的模型中的特征
        labels = best_cluster_model["labels"] if args.hasLabel else None  # 加载上一次保存的模型中的特征
        centroid, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(None,
                                                                                                            args.sourcedata,
                                                                                                            k=args.clusterNum,
                                                                                                            n=
                                                                                                            features.shape[
                                                                                                                0],
                                                                                                            center=None,
                                                                                                            feature=features.cpu().data,
                                                                                                            save=False,
                                                                                                            epoch=-1,
                                                                                                            labels=labels)

        centroid = torch.FloatTensor(centroid)
        if args.cuda and torch.cuda.is_available():
            centroid = centroid.cuda()
        # (3) 定义m2
        m2 = ClusterModule(centroid)
        if args.cuda and torch.cuda.is_available():
            m2.cuda()
        # (4) 加载m0,m1,m2
        args.start_iteration = best_cluster_model["epoch"]
        m0.load_state_dict(best_cluster_model["m0"])
        m1.load_state_dict(best_cluster_model["m1"])
        m2.load_state_dict(best_cluster_model["m2"])
        
    # 遍历数据并绘图
    scaner.start = 0
    features = []
    labels = []
    with torch.no_grad():
        while True:
            try:
                input, lengths, invp, label = scaner.getbatch_scaner()
                # 表示所有数据遍历完毕
                if input is None:
                    break
                if args.cuda and torch.cuda.is_available():
                    input, lengths, invp = input.cuda(), lengths.cuda(), invp.cuda()
                output, h = m0(input, lengths, input)  # decoder_h0(层数，batch,hiddensize*2)(3,128,256)
                # 转置，并强制拷贝一份tensor
                h = h.transpose(0, 1).contiguous()  # h[batch size, num_layers, 256]
                # 回到轨迹的原始顺序
                h = h[invp]  # 获得原始数据顺序对应的特征
                size = h.size()
                # 也就是使用三层特征，拼接成一个特征
                h = h.view(size[0], size[1] * size[2])  # [batch size, 3*256]
                features.append(h.cpu())
                labels.extend(label)
            except KeyboardInterrupt as e:
                print(e)
                break
    features = torch.cat(features)
    features = features[scaner.shuffle_invp]
    labels = np.array(labels)[scaner.shuffle_invp]
    if not args.hasLabel:
        labels = None
    
    _, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(None,
                                                                                                 args.sourcedata,
                                                                                                 k=args.clusterNum,
                                                                                                 n=scaner.get_data_num(),
                                                                                                 center=m2.get_centroid().cpu().data,
                                                                                                 feature=features.cpu().data,
                                                                                                 save=args.save,
                                                                                                 epoch=args.expId,
                                                                                                 tsne_filename="post",
                                                                                                 labels = labels)

    print(
        "cluster error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{}******".format(
            error_total, round(nmi, 4), round(ari, 4), round(inertia_start, 4), round(inertia_end, 4), n_iter))

    distA = pdist(features, metric='euclidean')
    distB = squareform(distA)
    ind = np.argsort(distB, 1)
    ind = ind[:, :400]
    np.savetxt("../../data/dist_2_" + str(args.expId) + ".txt", ind)


    with open("../../data/train_"+str(args.expId)+".csv", 'w') as wf:
        ori_labels_dict = {}
        for i in range(len(labels)):
            label = labels[i]
            ori_labels_dict[i] = label
            wf.write(str(i + 1) + "," + str(ori_labels_dict[i]) + "," + str(labels_dict[i]) + "\n")
            
    error_total,res,right, error_sort = calculate_error(args.clusterNum, labels_dict, ori_labels_dict)
    nmi = calculate_nmi(list(labels_dict.values()), list(ori_labels_dict.values()))
    ari = calculate_ari(list(labels_dict.values()), list(ori_labels_dict.values()))

    print(
        "cluster error_total:{} nmi:{} ari:{} ******".format(
            error_total, round(nmi, 4), round(ari, 4)))

# 计算nmi
def calculate_nmi(y_pred, y_true):
    # nmi = metrics.normalized_mutual_info_score(y_true, y_pred)  # 37服务器要把average_method参数隐藏掉
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return nmi

def calculate_ari(y_pred, y_true):
    score = metrics.adjusted_rand_score(y_pred, y_true)
    return score