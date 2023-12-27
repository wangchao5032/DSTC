# coding:utf-8
import torch
import torch.nn as nn  # 专门为神经网络设计的模块化接口
from torch.nn.utils import clip_grad_norm_
from models import EncoderDecoder, EncoderDecoder_without_dropout
from utils.data_utils import DataLoader, DataOrderScaner
import os, logging, h5py
from utils import constants
from cluster.ClusteringModule import ClusterModule, calculate_center_dist, calculate_neighbor_dist
from cluster import ClusterTool
# from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import datetime, time

# region 损失函数
def NLLcriterion(vocab_size):
    "construct NLL criterion 原论文有的"
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    ## The first dimension is not batch, thus we need
    ## to average over the batch manually
    # criterion = nn.NLLLoss(weight, size_average=False)
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion


def KLDIVcriterion(vocab_size):
    "construct KLDIV criterion 原论文有的"
    # weight = torch.ones(vocab_size)
    # weight[constants.PAD] = 0
    # ## The first dimension is not batch, thus we need
    # ## to average over the batch manually
    # criterion = nn.KLDivLoss(weight, size_average=False)
    criterion = nn.KLDivLoss(reduction='sum')
    return criterion


def KLDIVloss(output, target, criterion, V, D):
    """
    KL 散度损失函数 原论文有的
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    ## (batch, k) index in vocab_size dimension
    ## k-nearest neighbors for target
    # 词汇表是vocabsize * k大小，因此是取出target的k邻居
    indices = torch.index_select(V, 0, target)
    # 取出这k个邻居的输出
    # todo： 看输出是什么 ############
    ## (batch, k) gather along vocab_size dimension
    # output 包含每个样本预判出来词汇表的值的softmax结果
    # indices 是目标值的邻居
    # 取出预判结果中的邻居个的结果
    outputk = torch.gather(output, 1, indices)
    ## (batch, k) index in vocab_size dimension
    # 取出k个邻居的权重
    targetk = torch.index_select(D, 0, target)
    return criterion(outputk, targetk)


def KLDIVloss2(output, target, criterion, V, D):
    """
    constructing full target distribution, expensive 原论文有的
    """
    indices = torch.index_select(V, 0, target)
    targetk = torch.index_select(D, 0, target)
    fulltarget = torch.zeros(output.size()).scatter_(1, indices, targetk)
    ## here: need Variable(fulltarget).cuda() if use gpu
    fulltarget = fulltarget.cuda()
    return criterion(output, fulltarget)


def dist2weight(D, dist_decay_speed=0.8):
    "原论文有的"
    # D = D.div(10)# 0.1)
    # D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    # D = D + 1
    # D = 1/D
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    ## The PAD should not contribute to the decoding loss

    D[constants.PAD, :] = 0.0
    return D


def KMeansCriterion(weighted_dist):
    "k-means损失函数 ---wangchao"
    # loss = torch.mean(min_dist, dim=0)
    loss = torch.sum(weighted_dist, dim=1)
    loss = torch.mean(loss)
    return loss


def KLDIV_KMeans_loss(feature, output, target, criterion, criterion_cluster, V, D):
    "神经网络使用KLDIV+KMeans聚类损失函数--暂时没用 wangchao"
    # 计算神经网络损失函数
    loss1 = KLDIVloss(output, target, criterion, V, D)
    # 计算聚类损失函数
    loss2 = criterion_cluster(feature)
    return loss1 + loss2


def NLL_KMeans_loss(feature, output, target, criterion, criterion_cluster, alpha, lambda_val):
    "神经网络使用NLL+KMeans聚类损失函数----暂时没用 wangchao"
    # 计算神经网络损失函数
    loss1 = criterion(output, target)  # eg: output(1792,18866) target(1792)
    # 计算聚类损失函数
    loss2 = criterion_cluster(feature, alpha)
    loss = loss1 + lambda_val * loss2
    return loss


def ClusterCriterion(Q, P):
    "聚类KL散度损失函数 ---待修改---暂时没用 wangchao"
    log_arg = P / Q
    log_exp = torch.log(log_arg)
    sum_arg = P * log_exp
    loss = sum_arg.sum(1).sum(0)
    return loss


def batchloss(output, target, generator, lossF, generator_batch):
    """
    One batch loss

    Input:
    output (seq_len, batch, hidden_size): the output of EncoderDecoder 。是decoder解码后的结果
    target (seq_len, batch): target tensor 原轨迹
    generator: map the output of EncoderDecoder into the vocabulary space and do
        log transform     。是m1，把decoder编码后的结果投影到词空间
    lossF: loss function
    generator_batch: the maximum number of words to generate each step
    ---
    Output:
    loss
    """
    batch = output.size(1)
    loss = 0
    ## we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    for o, t in zip(output.split(generator_batch),
                    target.split(generator_batch)):
        ## (seq_len, generator_batch, hidden_size) =>
        ## (seq_len*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = generator(o)
        ## (seq_len*generator_batch,)
        t = t.view(-1)
        loss += lossF(o, t)

    return loss.div(batch)


# endregion

def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def savecheckpoint(state, filename="best_model.pt"):
    # torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'best_model.pt')
    torch.save(state, filename)


def save_best_cluster_model(state, filename="best_cluster_model.pt"):
    '''
    保存神经网络+聚类联合训练后的模型"
    :param state: 模型参数
    :param filename: 文件名
    :return:
    '''
    torch.save(state, filename)


def validate_cluster(valData, model, lossF, args):
    """
    valData (DataLoader)
    """
    m0, m1, m2 = model
    m0.eval()
    m1.eval()
    m2.eval()
    centroid, error_total, nmi, ari, inertia_start, inertia_end, n_iter, features, cluster_data_neighbors, labels = get_cluster_centroid_with_model(args,
                                                                                               m0,
                                                                                               valData,
                                                                                               valData.get_data_num(), center=m2.get_centroid().cpu().data,save=False,
                                                                                               e=-1)
    valData.set_neighbor_data(cluster_data_neighbors)
    valData.start = 0
    ## switch to evaluation mode
    # lambda_val = args.lambda_val
    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1
    
    features = []
    labels = []
    total_loss = 0
    for iteration in range(num_iteration):
        # input, lengths, invp, label = valData.getbatch_scaner()
        (input, lengths, invp, label), (input_neighbor, lengths_neighbor, invp_neighbor, _) = valData.getbatch_scaner(
            True)

        if input is None:
            break
        if args.cuda and torch.cuda.is_available():
            # input, lengths, invp = input.cuda(), lengths.cuda(), invp.cuda()
            input, lengths, invp, input_neighbor, lengths_neighbor, invp_neighbor = input.cuda(), lengths.cuda(), invp.cuda(), \
                                                                                    input_neighbor.cuda(), lengths_neighbor.cuda(), invp_neighbor.cuda()
                # 自编码器的前向传播
        output, h = m0(input, lengths, input)
        h = h.transpose(0, 1).contiguous()
        h = h[invp]
        size = h.size()
        h = h.view(size[0], size[1] * size[2])
        features.append(h.cpu().data)
        labels.extend(label)
        qij_batch, weighted_dist_batch = m2(h)
        # 计算自编码器的损失函数
        loss_ed = batchloss(output, input, m1, lossF, args.generator_batch)
        # 邻居
        output_neighbor, h_neighbor = m0(input_neighbor, lengths_neighbor, input_neighbor)
        h_neighbor = h_neighbor.transpose(0, 1).contiguous()
        h_neighbor = h_neighbor[invp_neighbor]
        size_neighbor = h_neighbor.size()
        h_neighbor = h_neighbor.view(size_neighbor[0], size_neighbor[1] * size_neighbor[2])

        loss_neighbor = calculate_neighbor_dist(h, h_neighbor)
        # loss_assignment_batch = assignment_loss(qij_batch.log(), target) / qij_batch.shape[0]
        # tot_loss_assignment += loss_neighbor
        
        # 聚类前向传播+聚类损失函数
        loss_cluster = args.kmeans * KMeansCriterion(weighted_dist_batch)
        weight = (qij_batch ** 2) / torch.sum(qij_batch, 0)
        target = (weight.t() / torch.sum(weight, 1)).t().detach()  # pij
        loss_cluster += (1 - args.kmeans) * ClusterCriterion(qij_batch, target) / h.size(0)
        
        loss_dist_batch = calculate_center_dist(m2.get_centroid())
        
        # 计算总的损失函数
        loss = args.alpha * loss_ed.item() + \
               args.beta * loss_cluster.item() + args.gamma * loss_dist_batch.item() + args.delta * loss_neighbor

        total_loss += loss# * output.size(1)
    features = torch.cat(features)
    features = features[valData.shuffle_invp]
    labels = np.array(labels)[valData.shuffle_invp]
    _, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(None,
                                                                                                 args.sourcedata,
                                                                                                 k=args.clusterNum,
                                                                                                 n=valData.get_data_num(),
                                                                                                 center=m2.get_centroid().cpu().data,
                                                                                                 feature=features.cpu().data,
                                                                                                 save=False,
                                                                                                 epoch=-1,
                                                                                                 labels=labels)
    ## switch back to training mode
    m0.train()
    m1.train()
    m2.train()
    print(
        "val sta: error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{} @{}".format(
            error_total,
            round(nmi, 4),
            round(ari, 4),
            round(inertia_start, 4),
            round(inertia_end, 4),
            n_iter,
            time.ctime()))
    return total_loss / num_iteration, nmi, ari

def genLoss(gendata, m0, m1, lossF, args):
    """
    One batch loss
    Input:
    gendata: a named tuple contains
        gendata.src (seq_len1, batch): input tensor
        gendata.lengths (1, batch): lengths of source sequences
        gendata.trg (seq_len2, batch): target tensor.
    m0: map input to output.
    m1: map the output of EncoderDecoder into the vocabulary space and do
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    """
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg
    if args.cuda and torch.cuda.is_available():
        input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
    ## (seq_len2, batch, hidden_size)
    output = m0(input, lengths, target)

    batch = output.size(1)
    loss = 0
    ## we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    for o, t in zip(output.split(args.generator_batch),
                    target.split(args.generator_batch)):
        ## (seq_len, generator_batch, hidden_size) =>
        ## (seq_len*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = m1(o)
        ## (seq_len*generator_batch,)
        t = t.view(-1)
        loss += lossF(o, t)

    return loss.div(batch)

def validate(valData, model, lossF, args):
    """
    valData (DataLoader)
    """
    m0, m1 = model
    ## switch to evaluation mode
    m0.eval()
    m1.eval()
    # lambda_val = args.lambda_val
    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_loss = 0
    for iteration in range(num_iteration):
        input, lengths, target = valData.getbatch_loader()
        if args.cuda and torch.cuda.is_available():
            input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
        # 自编码器的前向传播
        output, decoder_h0 = m0(input, lengths, target)
        # 计算自编码器的损失函数
        loss_ed = batchloss(output, target, m1, lossF, args.generator_batch)
        # loss_cluster = 0
        # # 计算总的损失函数
        loss = loss_ed #+ lambda_val * loss_cluster

        total_loss += loss.item() #* output.size(1)
    ## switch back to training mode
    m0.train()
    m1.train()
    return total_loss / valData.size


def train(args):
    '''
    the phase of learning preliminary trajectory representation is based on t2vec method(Deep Representation Learning for Trajectory Similarity Computation)
    '''
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.device)  # 指定第几块显卡（从0开始）
    # 输出运行日志
    logging.basicConfig(filename="models/training_"+str(args.expId)+".log", level=logging.INFO)
    ## ---------------------- 1、加载数据 ----------------------
    traintrg = os.path.join(args.data, "train.trg")  # 训练集：原轨迹*20
    trainsrc = os.path.join(args.data, "train.src")  # 训练集：带噪声的轨迹*20
    # 初始化数据集类
    trainData = DataLoader(trainsrc, traintrg, args.t2vec_batch, args.bucketsize)
    print("Reading training data...")
    # 加载train.src和train.trg，是训练数据，分为了8份 20000000
    trainData.load(args.max_num_line)
    print("Allocation: {}".format(trainData.allocation))
    print("Percent: {}".format(trainData.p))

    valsrc = os.path.join(args.data, "val.src")  # 验证集:带噪声的轨迹*20
    valtrg = os.path.join(args.data, "val.trg")  # 验证集:原轨迹*20
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        # 同上
        valData = DataLoader(valsrc, valtrg, args.t2vec_batch, args.bucketsize, True)
        print("Reading validation data...")
        # 加载val.src和val.trg，是验证数据，不分8份
        valData.load()
        assert valData.size > 0, "Validation data size must be greater than 0"
        print("Loaded validation data size {}".format(valData.size))
    else:
        print("No validation data found, training without validating...")

    ## ---------------------- 2、创建损失函数、模型、优化器 create criterion, model, optimizer ----------------------
    if args.criterion_name == "NLL":
        # 使用NLL损失函数
        criterion = NLLcriterion(args.vocab_size)
        lossF = lambda o, t: criterion(o, t)  # criterion是NLLcriterion的返回值，是一个函数，该函数要求输入o和t
    elif args.criterion_name == "KLDIV":
        # 使用KLD散度作为损失函数
        # 首先看k近邻是否存在
        assert os.path.isfile(args.knearestvocabs), \
            "{} does not exist".format(args.knearestvocabs)
        print("Loading vocab distance file {}...".format(args.knearestvocabs))
        # 获取词汇表每个词和他的k近邻信息
        with h5py.File(args.knearestvocabs) as f:  # 加载词汇表
            # k近邻以及距离
            V, D = f["V"][...], f["D"][...]
            V, D = torch.LongTensor(V), torch.FloatTensor(D)
        # 引入空间邻近性，使用距离计算权重
        D = dist2weight(D, args.dist_decay_speed)  # 把距离转换为权重（计算公式论文里有）
        if args.cuda and torch.cuda.is_available():  # variable放在GPU上运行
            V, D = V.cuda(), D.cuda()
        # todo-----------------
        criterion = KLDIVcriterion(args.vocab_size)
        lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)

    m0 = EncoderDecoder(args.vocab_size,  # 词汇表大小
                        args.embedding_size,  # 嵌入层输出大小
                        args.hidden_size,  # 隐藏层
                        args.num_layers,  # 层数
                        args.dropout,  # dropout值，防止过拟合
                        args.bidirectional)  # 决定是否采用双RNN
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),  # Sequential:按顺序构建网络
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        m1.cuda()
        criterion.cuda()
    else:
        print("=> training with CPU")
    # 更新策略
    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)
    start_epoch = 0
    start_batch = 0
    ## ---------------------- 3、加载模型状态和优化器状态 load model state and optmizer state ----------------------
    if os.path.isfile(args.best_model):
        print("=> loading best_model '{}'".format(args.best_model))
        logging.info("Restore training @ {}".format(time.ctime()))
        best_model = torch.load(args.best_model)
        # args.start_iteration = best_model["iteration"]
        start_epoch = best_model["epoch"]
        start_batch = best_model["batch"]
        trainData.index = best_model["data_idx"]
        best_avg_loss = best_model["best_avg_loss"]
        best_prec_loss = best_model["best_prec_loss"]
        m0.load_state_dict(best_model["m0"])
        m1.load_state_dict(best_model["m1"])
        m0_optimizer.load_state_dict(best_model["m0_optimizer"])
        m1_optimizer.load_state_dict(best_model["m1_optimizer"])
        es = 0
    else:
        print("=> no best_model found at '{}'".format(args.best_model))
        logging.info("Start training @ {}".format(time.ctime()))
        best_avg_loss = float('inf')  # 记录最好的训练误差
        best_prec_loss = float('inf')
        es = 0

    ## ---------------------- 4、模型训练 ----------------------
    num_iteration = args.epochs * sum(
        trainData.allocation) // args.t2vec_batch  # 共sum(trainData.allocation)个数据，每次批处理batch个，整个数据轮epochs次
    print("Iteration starts at {} and will end at {}".format(args.start_iteration, num_iteration - 1))
    num_batchs = sum(trainData.allocation) // args.t2vec_batch
    # writer = SummaryWriter('runs/pre_loss')
    ## training
    # for iteration in range(args.start_iteration, num_iteration):
    isEnd = False
    
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)
    for epoch in np.arange(start_epoch, args.epochs):
        trainData.shuffle()
        for batch in np.arange(start_batch, num_batchs):
            try:
                # 4.1 加载数据
                # input：训练集中batch个带噪声的轨迹，大小为这batch数据最大的长度*batch，例如50*128
                # length：每条轨迹的真实长度，大小为1*batch
                # target：训练集中batch个原轨迹，是和input对应的target数据，最大长度不一定和input一样，例如可能是70*128
                input, lengths, target = trainData.getbatch_loader()
                if args.cuda and torch.cuda.is_available():
                    input, lengths, target = \
                        input.cuda(), lengths.cuda(), target.cuda()

                # 4.2 清空上一步的残余更新参数值
                m0_optimizer.zero_grad()
                m1_optimizer.zero_grad()

                ## 4.3 forward computation 前向传播！！！
                # (1) 自编码器前向传播
                output, decoder_h0 = m0(input, lengths, target)  # decoder_h0(层数，batch,hiddensize*2)(3,128,256)
                
                # todo
                # 三重负样本损失
                # decoder_h0 = decoder_h0.transpose(0, 1).contiguous()
                # # 也就是使用三层特征，拼接成一个特征
                # decoder_h0 = decoder_h0.view(decoder_h0.size()[0], decoder_h0.size()[1] * decoder_h0.size()[2])  # [batch size, 3*256]
                # 
                # 
                # pos_output, pos_h0 = m0(target, lengths, target)  # decoder_h0(层数，batch,hiddensize*2)(3,128,256)
                # pos_h0 = pos_h0.transpose(0, 1).contiguous()
                # # 也就是使用三层特征，拼接成一个特征
                # pos_h0 = pos_h0.view(pos_h0.size()[0], pos_h0.size()[1] * pos_h0.size()[2])  # [batch size, 3*256]
                # 
                # neg_output, neg_h0 = m0(neg, neg_lengths, neg_target)
                # neg_h0 = neg_h0.transpose(0, 1).contiguous()
                # # 也就是使用三层特征，拼接成一个特征
                # neg_h0 = neg_h0.view(neg_h0.size()[0], neg_h0.size()[1] * neg_h0.size()[2])  # [batch size, 3*256]
                # 
                # # 三重
                # pos = torch.sqrt(torch.sum(torch.square(decoder_h0 - pos_h0), dim=-1, keepdim=False))
                # neg = torch.sqrt(torch.sum(torch.square(decoder_h0 - neg_h0), dim=-1, keepdim=False))
                # margin = 0.5
                # dist = pos - neg + margin
                # dist = torch.maximum(dist, torch.zeros_like(dist))
                # negsample_loss = torch.mean(dist)

                # neg_dist = torch.sort(neg_dist).values
                # neg_dist_max = neg_dist[len(neg_dist) - 1]
                # neg_dist = neg_dist / neg_dist_max
                # 
                # neg_dist_mid = neg_dist[len(neg_dist) // 2 + 1].item()
                # 
                # # dist_loss = 0
                # neg_dist = torch.exp(neg_dist_mid - neg_dist)
                # # 有些距离相等的不做考虑，界限是1
                # neg_dist_loss_num = torch.sum(neg_dist > 1) # 距离小于mid的数量
                # neg_dist_loss = torch.sum(neg_dist[neg_dist > 1])
                # neg_dist_loss /= (neg_dist_loss_num // 2)

                # (2) 计算自编码器损失函数
                loss = batchloss(output, target, m1, lossF, args.generator_batch)
                total_loss = loss# + negsample_loss
                # todo

                ## 4.4 compute the gradients 计算梯度,反向传播！！！
                total_loss.backward()
                ## clip the gradients
                clip_grad_norm_(m0.parameters(), args.max_grad_norm)
                clip_grad_norm_(m1.parameters(), args.max_grad_norm)
                ## one step optimization  #参数更新值
                m0_optimizer.step()
                m1_optimizer.step()

                ## 4.5 average loss for one word
                avg_loss = loss.item() / target.size(0)# target.size(0)是句子的长度

                if batch % args.print_freq == 0:
                    # print("epoch: {} batch: {}\tLoss: {}, neg_loss: {}".format(epoch, batch, avg_loss, negsample_loss))
                    print("epoch: {} batch: {}\tLoss: {}".format(epoch, batch, avg_loss))#, negsample_loss))

                # writer.add_scalar('ed', avg_loss, iteration)

                if (epoch * num_batchs + batch) % args.save_freq == 0:
                    prec_loss = validate(valData, (m0, m1), lossF, args)
                    if batch % args.print_freq == 0:
                        print("epoch: {} batch: {}\tval Loss: {}".format(epoch, batch, prec_loss))
                    if prec_loss < best_prec_loss:
                        best_prec_loss = prec_loss
                        logging.info("Best model with loss {} at epoch {} @ {}" \
                                     .format(best_prec_loss, epoch, time.ctime()))
                        es = 0
                        print(
                            "Saving the model at epoch {} batch {} validation loss {}".format(epoch, batch, prec_loss))
                        savecheckpoint({
                            "epoch": epoch,
                            "batch": batch,
                            "data_idx": trainData.index,
                            "best_avg_loss": best_avg_loss,
                            "best_prec_loss": best_prec_loss,
                            "m0": m0.state_dict(),  # 保存学习到的参数
                            "m1": m1.state_dict(),
                            "m0_optimizer": m0_optimizer.state_dict(),
                            "m1_optimizer": m1_optimizer.state_dict()
                        }, args.best_model)
                        if best_prec_loss < 0:
                            isEnd = True
                            break
                    else:
                        es += 1
                        print("Counter {} of 5".format(es))
            
                        if es > 4:
                            print("Early stopping with best_loss: ", best_prec_loss, "and val_loss for this epoch: ", prec_loss, "...")
                            isEnd = True
                            break
                
                    # # if math.fabs(avg_loss) < math.fabs(best_avg_loss):
                    # #     best_avg_loss = avg_loss
                    # #     logging.info(
                    # #         "Best model with loss {} at epoch {} batch {}@ {}".format(best_avg_loss, epoch, batch, time.ctime()))
                    # #     is_best = True
                    # # else:
                    # #     is_best = False
                    # 
                    # if is_best:  # 如果是最好的模型再保存，否则不保存，加快运行速度
                    #     print("Saving the model at epoch {} batch {} validation loss {}".format(epoch, batch, prec_loss))
                    #     savecheckpoint({
                    #         "epoch": epoch,
                    #         "batch": batch,
                    #         "data_idx" : trainData.index,
                    #         "best_avg_loss": best_avg_loss,
                    #         "best_prec_loss": best_prec_loss,
                    #         "m0": m0.state_dict(),  # 保存学习到的参数
                    #         "m1": m1.state_dict(),
                    #         "m0_optimizer": m0_optimizer.state_dict(),
                    #         "m1_optimizer": m1_optimizer.state_dict()
                    #     }, is_best, args.best_model)
                    #     if avg_loss < 0:
                    #         break
            except KeyboardInterrupt as e:
                print(e)
                break
        if isEnd:
            end_dt = datetime.datetime.now()
            end_t = time.time()
            print("END DATETIME")
            print(end_dt)
            print("Total time: " + str(end_t - start_t))
            break
    logging.info("End training with loss {}  @ {}".format(avg_loss, time.ctime()))
    # writer.close()


def get_cluster_centroid(args, model, data, data_num, center=None, feature=None, save=False, e=-1):
    path = None
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
        best_model = torch.load(model)
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
            h = m0_2.encoder_hn2decoder_h0(h)  # (num_layers, batch, hidden_size * num_directions)
            h = h.transpose(0, 1).contiguous()  # (batch, num_layers, hidden_size * num_directions)，例如 [64, 3, 256]
            h2 = h[invp]
            size = h2.size()
            # 使用三层特征拼接的特征
            h2 = h2.view(size[0], size[1] * size[2])
            feature.append(h2.cpu().data)
            labels.extend(label)
        feature = torch.cat(feature)
        feature = feature[data.shuffle_invp]
        labels = np.array(labels)[data.shuffle_invp]
    if not args.hasLabel:
        None
        
    # k-means聚类，获取初始簇中心
    centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(path,
                                                                                                         args.sourcedata,
                                                                                                         k=args.clusterNum,
                                                                                                         n=data_num,
                                                                                                         center=center,
                                                                                                         feature=feature.cpu().data,
                                                                                                         save=save,
                                                                                                         epoch=e,
                                                                                                         labels = labels)
    return centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, feature, cluster_data_neighbors, labels


def get_cluster_centroid_with_model(args, m0, data, data_num, center=None, feature=None, save=False, e=-1):
    path = None
    if feature is None:
        # 定义模型
        m0_2 = EncoderDecoder_without_dropout(args.vocab_size,
                                              args.embedding_size,
                                              args.hidden_size,
                                              args.num_layers,
                                              args.bidirectional)
        if args.cuda and torch.cuda.is_available():
            m0_2.cuda()
        # 加载模型
        m0_2.load_state_dict(m0.state_dict())

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
            h = m0_2.encoder_hn2decoder_h0(h)  # (num_layers, batch, hidden_size * num_directions)
            h = h.transpose(0, 1).contiguous()  # (batch, num_layers, hidden_size * num_directions)，例如 [64, 3, 256]
            h2 = h[invp]
            size = h2.size()
            # 使用三层特征拼接的特征
            h2 = h2.view(size[0], size[1] * size[2])
            feature.append(h2.cpu().data)
            labels.extend(label)
        feature = torch.cat(feature)
        feature = feature[data.shuffle_invp]
        labels = np.array(labels)[data.shuffle_invp]
    if not args.hasLabel:
        None

    # k-means聚类，获取初始簇中心
    centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(
        path,
        args.sourcedata,
        k=args.clusterNum,
        n=data_num,
        center=center,
        feature=feature.cpu().data,
        save=save,
        epoch=e,
        labels=labels)
    return centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, feature, cluster_data_neighbors, labels

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs，暂时没用"""
    # lr *= (0.1 ** (epoch // 2))
    decay_rate = 1
    # lr = 1/(1+decay_rate*epoch)*init_lr
    if epoch < 20:
        lr = init_lr
    elif epoch < 40:
        lr = init_lr * 0.9
    elif epoch < 60:
        lr = init_lr * 0.8
    else:
        lr = init_lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_cluster(args):
    '''
    Joint training optimization
    '''

    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.device)  # 指定第几块显卡

    logging.basicConfig(filename="models/training_" + str(args.expId) + ".log", level=logging.INFO)
    logging.info("clustering---------------------------- @ {}".format(time.ctime()))
    # ----------------------- 1 加载数据---------------------------
    scaner = DataOrderScaner(os.path.join(args.data, "train.ori"), os.path.join(args.data, "train.label"),args.batch)
    scaner.load()
    val_scaner = DataOrderScaner(os.path.join(args.data, "val.ori"), os.path.join(args.data, "val.label"),args.batch)
    val_scaner.load()
    # 获取原始轨迹数量
    data_num = scaner.get_data_num()
    # assignment_loss = nn.KLDivLoss(size_average=False)

    # -------------------- 2 创建 模型、优化器 -------------
    # 2.1 创建神经网络损失函数
    if args.criterion_name == "NLL_cluster":
        criterion_ed = NLLcriterion(args.vocab_size)
        lossF = lambda o, t: criterion_ed(o, t)
    elif args.criterion_name == "KLDIV_cluster":
        assert os.path.isfile(args.knearestvocabs), \
            "{} does not exist".format(args.knearestvocabs)
        print("Loading vocab distance file {}...".format(args.knearestvocabs))
        with h5py.File(args.knearestvocabs) as f:  # 加载词汇表
            # k近邻以及距离
            V, D = f["V"][...], f["D"][...]
            V, D = torch.LongTensor(V), torch.FloatTensor(D)
        # 引入空间邻近性，使用距离计算权重, 把距离转换为权重, dist_decay_speed是论文中theta
        D = dist2weight(D, args.dist_decay_speed)
        if args.cuda and torch.cuda.is_available():  # variable放在GPU上运行
            V, D = V.cuda(), D.cuda()
        criterion_ed = KLDIVcriterion(args.vocab_size)
        lossF = lambda o, t: KLDIVloss(o, t, criterion_ed, V, D)
    else:
        print("please input correct criterion")
        return

    # 2.2 定义神经网络模型、优化器
    # 与单独训练差不多去掉dropout层
    m0 = EncoderDecoder_without_dropout(args.vocab_size,
                                        args.embedding_size,
                                        args.hidden_size,
                                        args.num_layers,
                                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),  # Sequential:按顺序构建网络, LogSoftmax方便kl loss计算
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        m1.cuda()
        criterion_ed.cuda()
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)

    # 2.3 加载模型状态和优化器状态
    if not os.path.isfile(args.best_model):  # 如果没有完成预训练
        raise Exception
    if not os.path.isfile(args.cluster_model):  # 如果完成了预训练，但是没有进行联合训练
        print("=> loading best_model '{}'".format(args.best_model))
        logging.info("loading best_model @ {}".format(time.ctime()))
        # (1) 加载 神经网络 模型状态和优化器状态
        # best_model = torch.load(args.best_model, map_location='cuda:0') # 如果在实验室服务器上第2个GPU训练的预训练，那么在只有1个GPU的服务器上加载时会出错
        best_model = torch.load(args.best_model)
        m0.load_state_dict(best_model["m0"])
        m1.load_state_dict(best_model["m1"])
        m0_optimizer.load_state_dict(best_model["m0_optimizer"])
        m1_optimizer.load_state_dict(best_model["m1_optimizer"])
        start_epoch=0
        # (2) 创建 聚类 模型、优化器
        # 根据m0encoder部分计算轨迹特征表示，获得初始聚类中心centroid, 以及计算指标， 其中默认初始的簇数为args.clusterNum
        # TODO(HJH): 看
        centroid, error_total, nmi, ari, inertia_start, inertia_end, n_iter, feature, cluster_data_neighbors, _ = get_cluster_centroid(args,
                                                                                                   args.best_model,
                                                                                                   scaner,
                                                                                                   data_num, save=False,
                                                                                                   e=-1)
        # scaner.set_neighbor_data(cluster_data_neighbors)
        # centroid.shape : 簇数, feature, 此时feature 是 三层feature的拼接
        centroid = torch.FloatTensor(centroid)
        if args.cuda and torch.cuda.is_available():
            centroid = centroid.cuda()

        print(
            "*******Start******* cluster error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{}******".format(
                error_total, round(nmi, 4), round(ari, 4), round(inertia_start, 4), round(inertia_end, 4), n_iter))
        logging.info(
            "*******Start*******cluster error_total:{}\t nmi:{}\t ari:{}\t inertia_start:{}\t inertia_end:{}\t n_iter:{}".format(
                error_total,
                round(nmi, 4), round(ari, 4),
                round(inertia_start, 4),
                round(inertia_end, 4), n_iter))
        # 定义聚类模型, 后续簇中心损失函数仅仅更新簇中心坐标，而不会对编码器改进
        m2 = ClusterModule(centroid)
        m2_optimizer = torch.optim.Adam(m2.parameters(), lr=args.m2_learning_rate)
        if args.cuda and torch.cuda.is_available():
            m2.cuda()
    # TODO(hjh): 理解该部分
    else:  # 如果进行了一部分的联合训练
        # (1) 加载模型
        print("=> loading cluster_model '{}'".format(args.cluster_model))
        logging.info("loading cluster_model @ {}".format(time.ctime()))
        cluster_model = torch.load(args.cluster_model)
        # (2) 获得初始聚类中心。其实可以随便定义一个centroid，因为目的只是在创建m2时不报错，在第(4)步会对m2中centroid的值进行更新
        features = cluster_model["features"]  # 加载上一次保存的模型中的特征
        labels = cluster_model["labels"] if args.hasLabel else None
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
        # scaner.set_neighbor_data(cluster_data_neighbors)
        centroid = torch.FloatTensor(centroid)
        if args.cuda and torch.cuda.is_available():
            centroid = centroid.cuda()
        # (3) 定义m2
        m2 = ClusterModule(centroid)
        m2_optimizer = torch.optim.Adam(m2.parameters(), lr=0.008)
        if args.cuda and torch.cuda.is_available():
            m2.cuda()
        m2_lr_init = m2_optimizer.param_groups[0]['lr']

        # (4) 加载m0,m1,m2
        start_epoch = cluster_model["epoch"]
        m0.load_state_dict(cluster_model["m0"])
        m1.load_state_dict(cluster_model["m1"])
        m2.load_state_dict(cluster_model["m2"])

        m0_lr_init = m0_optimizer.param_groups[0]['lr']
        m0_optimizer.load_state_dict(cluster_model["m0_optimizer"])
        m1_lr_init = m1_optimizer.param_groups[0]['lr']
        m1_optimizer.load_state_dict(cluster_model["m1_optimizer"])
        m2_lr_init = m2_optimizer.param_groups[0]['lr']
        m2_optimizer.load_state_dict(cluster_model["m2_optimizer"])

    # -------------------------------3 模型训练-------------------------------
    best_loss = float('inf')  # 记录最好的训练误差
    best_val_loss = float('inf')
    best_val_nmi = 0
    es = 0
    logging.info(args)
    lossE = []
    lossK = []
    lossA = []
    lossD = []
    lossT = []
    lossN = []
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)
    for epoch in range(start_epoch, args.epochs):
        # -------------------- 3.1 用上一次编码器的结果计算所有数据的q,p,weighted_dist --------------------
        # 一个epoch中p和q是不变的。Get the current distribution。
        val_scaner.start = 0
        scaner.start = 0  # 每一个epoch计算qij时，先让scaner的start指针归零
        scaner.set_neighbor_data(cluster_data_neighbors)
        # 分别是自编码器、线形层、聚类层
        m0_optimizer.zero_grad()
        m1_optimizer.zero_grad()
        m2_optimizer.zero_grad()

        # --------------------3.2 神经网络+聚类联合训练--------------------
        i = 0
        scaner.start = 0
        tot_loss_ed = 0
        tot_loss_kmeans = 0
        tot_loss_assignment = 0
        tot_loss_dist = 0
        tot_loss_neighbor = 0
        LOSS_ED = []
        LOSS_KMEANS = []
        LOSS_ASSIGNMENT = []
        LOSS_DIST = []
        LOSS_NEIGHBOR = []
        TOTAL_LOSS = []
        while True:
            try:
                # （1） 加载数据
                (input, lengths, invp, _), (input_neighbor, lengths_neighbor, invp_neighbor, _) = scaner.getbatch_scaner(True)  # 这里要变成shuffle！！！！！！
                # 表示所有数据遍历完毕
                if input is None:
                    break
                if args.cuda and torch.cuda.is_available():
                    input, lengths, invp, input_neighbor, lengths_neighbor, invp_neighbor = input.cuda(), lengths.cuda(), invp.cuda(), \
                                                                                            input_neighbor.cuda(), lengths_neighbor.cuda(), invp_neighbor.cuda()

                # （2） 清空上一步的残余更新参数值
                m0_optimizer.zero_grad()  # 如果加入了m0会有问题
                m1_optimizer.zero_grad()
                m2_optimizer.zero_grad()
                # 神经网络前向传播，获得降维后的特征
                # (3) forward computation 前向传播, h是encoder的特征表示结果, output是decoder的输出
                # decoder_h0(层数，batch,hiddensize*2)
                output, h = m0(input, lengths, input)
                # 转置，并强制拷贝一份tensor h[batch size, num_layers, 256]
                h = h.transpose(0, 1).contiguous()
                # 获得原始数据顺序对应的特征
                h = h[invp]
                size = h.size()
                # 也就是使用三层特征，拼接成一个特征
                h = h.view(size[0], size[1] * size[2])  # [batch size, 3*256]
                # 聚类前向传播, 计算出簇间距离，分配到一个最适合的簇，利用softmax让其可导，参与到梯度传播
                qij_batch, weighted_dist_batch = m2(h)

                # (4) 计算损失函数+反向传播
                # a.自编码器损失函数
                # TODO(hjh): 看
                loss_ed_batch = batchloss(output, input, m1, lossF, args.generator_batch)
                # 损失值和比例因子结合，并除以轨迹长度求下平均
                # args.alpha是比例因子1（4比较合适），input.size(0)是句子的长度 average loss for one word
                loss_ed_batch = loss_ed_batch / input.size(0)
                loss_ed_batch = args.alpha * loss_ed_batch
                tot_loss_ed += loss_ed_batch.cpu().data.numpy()
                loss_ed = loss_ed_batch.cpu().data.numpy()

                # 邻居的 todo
                output_neighbor, h_neighbor = m0(input_neighbor, lengths_neighbor, input_neighbor)
                h_neighbor = h_neighbor.transpose(0, 1).contiguous()
                h_neighbor = h_neighbor[invp_neighbor]
                size_neighbor = h_neighbor.size()
                h_neighbor = h_neighbor.view(size_neighbor[0], size_neighbor[1] * size_neighbor[2])
                
                loss_neighbor = calculate_neighbor_dist(h, h_neighbor)
                # loss_assignment_batch = assignment_loss(qij_batch.log(), target) / qij_batch.shape[0]
                loss_neighbor_batch = args.delta * loss_neighbor
                tot_loss_neighbor += loss_neighbor_batch.cpu().data.numpy()
                loss_neighbor = loss_neighbor_batch.cpu().data.numpy()
                # tot_loss_assignment += loss_neighbor
                
                # # b.聚类损失函数
                # todo
                # k-means损失函数
                loss_kmeans_batch = KMeansCriterion(weighted_dist_batch)
                loss_kmeans_batch = args.kmeans * args.beta * loss_kmeans_batch # args.beta是比例因子2（0.01比较合适）
                loss_kmeans = loss_kmeans_batch.cpu().data.numpy()
                tot_loss_kmeans += loss_kmeans

                # 软分配损失
                # todo
                weight = (qij_batch ** 2) / torch.sum(qij_batch, 0)
                target = (weight.t() / torch.sum(weight, 1)).t().detach() # pij
                loss_assignment_batch = ClusterCriterion(qij_batch, target)
                loss_assignment_batch = loss_assignment_batch / h.size(0)
                # loss_assignment_batch = assignment_loss(qij_batch.log(), target) / qij_batch.shape[0]
                loss_assignment_batch = (1 - args.kmeans) * args.beta * loss_assignment_batch
                loss_assignment = loss_assignment_batch.cpu().data.numpy()
                tot_loss_assignment += loss_assignment

                # 簇间距离损失函数
                loss_dist_batch = calculate_center_dist(m2.get_centroid())
                loss_dist_batch = args.gamma * loss_dist_batch  # args.gamma是比例银子3（0.1比较合适）
                loss_dist = loss_dist_batch.cpu().data.numpy()
                tot_loss_dist = tot_loss_dist + loss_dist

                # loss = args.alpha * (loss_ed_batch + args.beta * ((loss_kmeans_batch) + args.gamma * loss_dist_batch))
                # loss = loss_ed_batch + loss_kmeans_batch + loss_dist_batch + loss_neighbor_batch
                loss = loss_ed_batch + loss_assignment_batch + loss_dist_batch + loss_neighbor_batch + loss_kmeans_batch
                loss.backward()

                if i % 10 == 9:
                    # loss_ed = loss_ed_batch.item()
                    # loss_kmeans = loss_kmeans_batch.item()
                    # loss_dist = loss_dist_batch.item()
                    LOSS_ED.append(loss_ed)
                    LOSS_KMEANS.append(loss_kmeans)
                    LOSS_ASSIGNMENT.append(loss_assignment)
                    LOSS_DIST.append(loss_dist)
                    LOSS_NEIGHBOR.append(loss_neighbor)
                    # TOTAL_LOSS.append(loss_ed + loss_dist + loss_kmeans)
                    TOTAL_LOSS.append(loss_ed + loss_dist + loss_assignment + loss_neighbor + loss_kmeans)
                    print("Epoch:" + str(epoch) + ", Iteration :" + str(i) + ", loss_ed = " + str(loss_ed_batch.cpu().data.numpy())
                          + ", loss_kmeans = " + str(loss_kmeans)
                          + ", loss_assignment = " + str(loss_assignment_batch.cpu().data.numpy())
                          + ", loss_dist = " + str(loss_dist_batch.cpu().data.numpy())
                          + ", loss_neighbor = " + str(loss_neighbor_batch.cpu().data.numpy())
                          + ", tot_loss = " + str(loss.cpu().data.numpy()))

                # (5) clip the gradients 梯度裁剪
                clip_grad_norm_(m0.parameters(), args.max_grad_norm)
                clip_grad_norm_(m1.parameters(), args.max_grad_norm)
                clip_grad_norm_(m2.parameters(), args.max_grad_norm)

                # (6) one step optimization  #参数更新值
                m0_optimizer.step()
                m1_optimizer.step()
                m2_optimizer.step()

                i += 1

            except KeyboardInterrupt as e:
                print(e)
                break

        lossE += list(LOSS_ED)
        lossK += list(LOSS_KMEANS)
        lossA += list(LOSS_ASSIGNMENT)
        lossD += list(LOSS_DIST)
        lossN += list(LOSS_NEIGHBOR)
        lossT += list(TOTAL_LOSS)
        # --------------------4 保存模型--------------------
        tot_loss = tot_loss_ed + tot_loss_kmeans + tot_loss_dist + tot_loss_assignment + tot_loss_neighbor
        centroid, error_total, nmi, ari, inertia_start, inertia_end, n_iter, features, cluster_data_neighbors, labels = get_cluster_centroid_with_model(args,
                                                                                                   m0,
                                                                                                   scaner,
                                                                                                   data_num, center=m2.get_centroid().cpu().data,save=False,
                                                                                                   e=-1)
        # _, error_total, nmi, ari, inertia_start, inertia_end, n_iter, cluster_data_neighbors, labels_dict = ClusterTool.DoKMeansWithError(None,
        #                                                                                              args.sourcedata,
        #                                                                                              k=args.clusterNum,
        #                                                                                              n=data_num,
        #                                                                                              center=m2.get_centroid().cpu().data,
        #                                                                                              feature=features.cpu().data,
        #                                                                                              save=False,
        #                                                                                              epoch=epoch,
        #                                                                                              labels=labels)
        
        val_loss, val_nmi, val_ari = validate_cluster(val_scaner, [m0,m1,m2], lossF, args)
        print("val loss:{}".format(val_loss))
        
        # save_best_cluster_model({
        #     "epoch": epoch,
        #     "m0": m0.state_dict(),
        #     "m1": m1.state_dict(),
        #     "m2": m2.state_dict(),
        #     "m0_optimizer": m0_optimizer.state_dict(),
        #     "m1_optimizer": m1_optimizer.state_dict(),
        #     "m2_optimizer": m2_optimizer.state_dict(),
        #     "features": features,
        #     "labels":labels
        # }, args.cluster_model)
        # _, error_total, nmi, ari, inertia_start, inertia_end, n_iter = get_cluster_centroid(args, args.cluster_model,
        #                                                                                     scaner, data_num,
        #                                                                                     center=m2.get_centroid().cpu().data,
        #                                                                                     # feature=features.cpu().data,
        #                                                                                     save=False, e=epoch)
        print(
            "epoch:{} ed:{} kmeans:{} dist:{} total:{} error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{} @{}".format(
                epoch,
                round(tot_loss_ed, 4),
                round(tot_loss_kmeans, 4),
                round(tot_loss_dist, 4),
                round(tot_loss, 4),
                error_total,
                round(nmi, 4),
                round(ari, 4),
                round(inertia_start, 4),
                round(inertia_end, 4),
                n_iter,
                time.ctime()))
        logging.info(
            "epoch:{} ed:{} kmeans:{} dist:{} total:{} error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{} @{}".format(
                epoch,
                round(tot_loss_ed, 4),
                round(tot_loss_kmeans, 4),
                round(tot_loss_dist, 4),
                round(tot_loss, 4),
                error_total,
                round(nmi, 4),
                round(ari, 4),
                round(inertia_start, 4),
                round(inertia_end, 4),
                n_iter,
                time.ctime()))

        # TODO(hjh) : best_loss在训练一半的情况
        # if tot_loss < best_loss:
        #     best_loss = tot_loss
        #     save_best_cluster_model({
        #         "epoch": epoch,
        #         "m0": m0.state_dict(),
        #         "m1": m1.state_dict(),
        #         "m2": m2.state_dict(),
        #         "m0_optimizer": m0_optimizer.state_dict(),
        #         "m1_optimizer": m1_optimizer.state_dict(),
        #         "m2_optimizer": m2_optimizer.state_dict(),
        #         "features": features,
        #         "labels":labels
        #     }, args.best_cluster_model)
        #     print("save epoch:{} loss:{} error_total:{} nmi:{} ari:{}".format(epoch, round(tot_loss, 4), error_total,
        #                                                                       round(nmi, 4), round(ari, 4)))
        #     logging.info(
        #         "save epoch:{} loss:{} error_total:{} nmi:{} ari:{}".format(epoch, round(tot_loss, 4), error_total,
        #                                                                     round(nmi, 4), round(ari, 4)))
        

        if val_nmi > best_val_nmi:
            best_val_nmi = val_nmi
            es = 0
            save_best_cluster_model({
                "epoch": epoch,
                "m0": m0.state_dict(),
                "m1": m1.state_dict(),
                "m2": m2.state_dict(),
                "m0_optimizer": m0_optimizer.state_dict(),
                "m1_optimizer": m1_optimizer.state_dict(),
                "m2_optimizer": m2_optimizer.state_dict(),
                "features": features,
                "labels": labels
            }, args.best_cluster_model)
            print("save epoch:{} loss:{} error_total:{} nmi:{} ari:{}".format(epoch, round(tot_loss, 4), error_total,
                                                                              round(nmi, 4), round(ari, 4)))
            logging.info(
                "save epoch:{} loss:{} error_total:{} nmi:{} ari:{}".format(epoch, round(tot_loss, 4), error_total,
                                                                            round(nmi, 4), round(ari, 4)))
        else:
            es += 1
            print("Counter {} of 5".format(es))

            if es > 4:
                print("Early stopping with best_nmi: ", best_val_nmi, "and val_nmi for this epoch: ", val_nmi, "...")
                break
        
        if epoch % 1 == 0:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            fig = plt.figure(figsize=(20, 10))
            axes = fig.subplots(nrows=2, ncols=2)
            x = range(0, len(lossE))
            axes[0, 0].plot(x, lossE, '.-')
            plt_title = 'ed loss' # '重构损失'
            axes[0, 0].set_title(plt_title)
            axes[0, 0].set_xlabel('epoch')
            axes[0, 0].set_ylabel('LOSS')

            # todo
            # axes[0, 1].plot(x, lossK, '.-')
            # plt_title = 'cluster loss' # '聚类损失'
            # axes[0, 1].set_title(plt_title)
            # axes[0, 1].set_xlabel('epoch')
            # axes[0, 1].set_ylabel('LOSS')

            # todo
            axes[0, 1].plot(x, lossA, '.-')
            plt_title = 'cluster assignment loss' # '聚类损失'
            axes[0, 1].set_title(plt_title)
            axes[0, 1].set_xlabel('epoch')
            axes[0, 1].set_ylabel('LOSS')

            axes[1, 0].plot(x, lossD, '.-')
            plt_title = 'dist loss' # '簇间损失'
            axes[1, 0].set_title(plt_title)
            axes[1, 0].set_xlabel('epoch')
            axes[1, 0].set_ylabel('LOSS')

            axes[1, 1].plot(x, lossT, '.-')
            plt_title = 'total loss' # '总损失'
            axes[1, 1].set_title(plt_title)
            axes[1, 1].set_xlabel('epoch')
            axes[1, 1].set_ylabel('LOSS')
            plt.savefig("./models/loss_" + str(args.expId) + ".png")
            print("本次loss图保存成功")
            # plt.show()
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))

