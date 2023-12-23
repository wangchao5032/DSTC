#coding:utf-8
import sys
sys.path.append('../..')
sys.path.append('..')
import argparse
from training.showResult.resultVisPre import showPic as showPrePic
from training.showResult.resultVisDstc import showPic as showDstcPic
import json

def insert_exp_id(file_path, exp_id):
    i = file_path.rfind('.')
    return f'{file_path[:i]}_{exp_id}{file_path[i:]}'

def UpdateArgs(args, json_data):
    if json_data["mode"] == "pretraining":
        args.isPreTrain = True
    elif json_data["mode"] == "joint-training":
        args.isPreTrain = False
    else:
        print('Mode must be "pretraining" or "joint-training"')
        exit(0)

    # rename file
    args.expId = json_data["expId"]
    args.checkpoint = insert_exp_id(args.checkpoint, args.expId)
    args.best_model = insert_exp_id(args.best_model, args.expId)
    args.best_cluster_model = insert_exp_id(args.best_cluster_model, args.expId)
    args.cluster_model = insert_exp_id(args.cluster_model, args.expId)

    # parameters
    args.vocab_size = json_data["vocabSize"]
    args.clusterNum = json_data["clusterNum"]
    args.hasLabel = json_data["hasLabel"]
    args.embedding_size = json_data["embeddingSize"]
    args.hidden_size = json_data["hiddenSize"]
    args.batch = json_data["batch"]
    args.t2vec_batch = json_data["t2vecBatch"]
    if not args.isPreTrain:
        args.save = json_data["needSave"]
        if args.save:
            args.save_freq = json_data["saveFreq"]
        args.criterion_name = "KLDIV_cluster"
    args.dropout = json_data["dropout"]
    args.learning_rate = json_data["learningRate"]
    args.m2_learning_rate = json_data["m2LearningRate"]
    args.dist_decay_speed = json_data["distDecaySpeed"] # 对远距离cell的惩罚值，也就是theta的倒数，越大惩罚越大
    args.alpha = json_data["alpha"] # 控制重构
    args.beta = json_data["beta"]   # 控制聚类损失函数
    args.gamma = json_data["gamma"] # 簇间
    args.delta = json_data["delta"] # 邻居
    args.kmeans = json_data["kmeans"]
    args.sourcedata = json_data["sourceData"]
    args.epochs = json_data["epochs"]
    args.dataName = json_data["dataName"]
    args.knearestvocabs = args.data_dir + args.dataName + "-knearestvocabs.h5";


parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data_dir", default="../../data/",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="../models/checkpoint.pt",
    help="The saved checkpoint (only neural network)")

parser.add_argument("-best_model", default="../models/best_model.pt",
    help="The saved best_model (only neural network)")

parser.add_argument("-best_cluster_model", default="../models/best_cluster_model.pt",
    help="The saved best model combined neural network with cluster model")

parser.add_argument("-cluster_model", default="../models/cluster_model.pt",
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

parser.add_argument("-expId", default=1,
    help="exp id")

parser.add_argument("-device", default=-1,
    help="device id")

parser.add_argument("-save", default=True,
    help="tsne png save")

parser.add_argument("-hasLabel", default=True,
    help="是否有标签数据")

parser.add_argument("-kmeans", default=0,
    help="是否使用KmeansLoss")

args = parser.parse_args()

args.bucketsize = [(100000, 100000)]  # 把src和trg的数据长度都不超过100000的数据存在同一个列表中


if __name__ == "__main__":
    ## 1, 初始化
    with open("../conf/train_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        UpdateArgs(args, json_data)

    print("execute exp of {} file name of best_model is {} ".format(args.expId, args.best_model))
    print(args)

    if args.isPreTrain:
        showPrePic(args)
    else:
        showDstcPic(args)
