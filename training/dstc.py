#coding:utf-8

import argparse
import os
from train import train, train_cluster
import datetime, time
import json

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="../data/",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="models/checkpoint.pt",
    help="The saved checkpoint (only neural network)")

parser.add_argument("-best_model", default="models/best_model.pt",
    help="The saved best_model (only neural network)")

parser.add_argument("-best_cluster_model", default="models/best_cluster_model.pt",
    help="The saved best model combined neural network with cluster model")

parser.add_argument("-cluster_model", default="models/cluster_model.pt",
    help="The saved last model combined neural network with cluster model")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=True,
    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.1,
    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-m2_learning_rate", type=float, default=0.008)

parser.add_argument("-batch", type=int, default=128, # 256
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=32,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=128, # 256
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=40,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=40,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=True,
    help="True if we use GPU to train the model")

parser.add_argument("-criterion_name", default="KLDIV",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default="data/hangzhou-vocab-dist-cell75-1.h5", # None
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-isPreTrain", type=bool, default=True,
    help="is pre train")

parser.add_argument("-vocab_size", type=int, default=11483,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(30, 30), (50, 50)],
    help="Bucket size for training")

parser.add_argument("-clusterNum", type=int, default=19*24, # 聚类簇的数目
                    help="cluster number of KMeans algorithm")

parser.add_argument("-alpha", type=int, default=2, # 4
                    help="coefficient of reconstruction loss")

parser.add_argument("-beta", type=int, default=0.01,
                    help="coefficient of clustering loss")

parser.add_argument("-gamma", type=int, default=0.1,
                    help="coefficient of distance loss between centroids")

parser.add_argument("-delta", type=int, default=0.1,
                    help="coefficient of neighbor loss between datas")

parser.add_argument("-sourcedata", default='preprocessing/make_data/20200101_jianggan.h5',
                    help="source data and label")

parser.add_argument("-devices", default=[0, 1, 2, 3, 4, 5, 6, 7],
# parser.add_argument("-devices", default=[0, 1, 2, 3],
# parser.add_argument("-devices", default=[4, 5, 7],
    help="Bucket size for training")

parser.add_argument("-expId", default=1,
    help="Bucket size for training")

parser.add_argument("-device", default=-1,
    help="Bucket size for training")

parser.add_argument("-save", default=True,
    help="tsne png save")

parser.add_argument("-hasLabel", default=True,
    help="是否有标签数据")

parser.add_argument("-kmeans", default=0,
    help="是否使用KmeansLoss")

args = parser.parse_args()

## __main__
args.bucketsize = [(100000, 100000)]  # 把src和trg的数据长度都不超过100000的数据存在同一个列表中
args.bucketsize = [(100000, 100000)]  # 把src和trg的数据长度都不超过100000的数据存在同一个列表中


if __name__ == "__main__":
    # 时间计算
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)

    ## 1, 初始化
    with open("conf/train_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        args.isPreTrain = json_data["isPreTrain"]
        args.expId = json_data["expId"]
        checkPointName = str.split(args.checkpoint, '.')
        args.checkpoint = checkPointName[0] + '_' + str(args.expId) + "." + checkPointName[1]
        bestModelName = str.split(args.best_model, '.')
        args.best_model = bestModelName[0] + '_' + str(args.expId) + "." + bestModelName[1]
        bestClusterName = str.split(args.best_cluster_model, '.')
        args.best_cluster_model = bestClusterName[0] + '_' + str(args.expId) + "." + bestClusterName[1]
        clusetName = str.split(args.cluster_model, '.')
        args.cluster_model = clusetName[0] + '_' + str(args.expId) + "." + clusetName[1]
        args.device = args.devices[(args.expId - 1) % 8]
        args.vocab_size = json_data["vocabSize"]
        args.clusterNum = json_data["clusterNum"]
        args.hasLabel = json_data["hasLabel"]
        args.embedding_size = json_data["embeddingSize"] # 嵌入层输出 128
        args.hidden_size = json_data["hiddenSize"] # featur维度 64.
        args.batch = json_data["batch"] # 32可
        args.t2vec_batch = json_data["t2vecBatch"] # 32
        if not args.isPreTrain:
            args.save = json_data["needSave"]
            if args.save:
                args.save_freq = json_data["saveFreq"]
            args.criterion_name = "KLDIV_cluster"
        args.dropout = json_data["dropout"] # 0.3
        args.learning_rate = json_data["learningRate"]
        args.m2_learning_rate = json_data["m2LearningRate"]
        args.dist_decay_speed = json_data["distDecaySpeed"] # 对远距离cell的惩罚值，也就是theta的倒数，越大惩罚越大
        args.alpha = json_data["alpha"]# 10 # 控制重构
        args.beta = json_data["beta"] # 控制聚类损失函数
        args.gamma = json_data["gamma"] # 簇间
        args.delta = json_data["delta"]# 邻居
        args.kmeans = json_data["kmeans"]
        args.sourcedata = json_data["sourceData"]
        args.epochs = json_data["epochs"]
        args.dataName = json_data["dataName"]
        args.knearestvocabs = "../data/" + args.dataName + "-knearestvocabs.h5";

    for dirpath,dirnames,filenames in os.walk(args.data):
        for file in filenames:
            file_type = file.split('.')[-1]
            if (file_type == "h5"):
                args.knearestvocabs = os.path.join(dirpath, file)  # 文件全名
                break

    print("execute exp of {} file name of best_model is {} ".format(args.expId, args.best_model))

    print(args)

    if args.isPreTrain:
        train(args)
    else:
        # 联合训练
        if os.path.exists(args.cluster_model):
            os.remove(args.cluster_model)
            os.remove(args.best_cluster_model)
        train_cluster(args)