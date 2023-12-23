"FuzzyCMeans前向传播"
import torch
import torch.nn as nn #专门为神经网络设计的模块化接口
from torch.autograd import Variable


class ClusterModule(nn.Module):
    def __init__(self, centroid):
        super(ClusterModule, self).__init__()
        # 簇中心
        self.centroid = centroid
        self.centroid = nn.Parameter(self.centroid)  # 网络要训练的参数，添加之后可以利用m2.state_dict()查看网络参数

    def forward(self, z):
        # 簇数
        # [num_clusters, 3*256]
        num_clusters = self.centroid.size()[0]
        # z : [batch size, 3*256]
        # 数据数量，也就是batch size
        num_features = z.size()[0]

        # 1. 计算qij
        # 在第二个维度增加一维度，并扩充num_clusters倍个，其中num_clusters是簇数，方便后续计算
        xx = z.unsqueeze(1).expand(z.data.size(0), num_clusters, z.data.size(1))
        # 对于簇数据，也增加num_features倍数个，方便计算
        yy = self.centroid.unsqueeze(0).expand(z.data.size(0), num_clusters, z.data.size(1))
        # batchsize, 簇数, 3*256
        dd = xx - yy
        # 各个数据到簇中心距离的平方 squeeze 从数组的形状中删除单维度条目 [batchsize, 簇数]
        dist = torch.pow(dd, 2).sum(2).squeeze()  # 每个数据到k个簇中心的距离

        # alpha参数没有，也没有pow指数得到qij 此时认为alpha是1吧
        qij_numerator = torch.pow(1 + dist, -1)  # 分子
        # sum(1) [batch size] unsqueeze(1) [batch size, 1]
        normalizer_q = (qij_numerator.sum(1)).unsqueeze(1)  # 分母
        # 归一化
        qij = qij_numerator / normalizer_q


        # 2.计算weighted_dist
        # 计算一个带权重的误差，让轨迹数据到各个簇是按分布规律分配
        alpha = 1000 #2
        # # 2.1 查找隐藏特征到最近的簇中心的距离和簇号
        # 1维度就是和各个簇的距离，求这些距离的最小
        # torch.min 返回的是实际最小值，以及最小值所在索引，也就是簇号
        min_dist, min_label = torch.min(dist, 1)

        # 2.2 计算指数 compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes
        # min_dist = min_dist.expand(x.data.size(0),y.data.size(0)) # 把min_dist从（128,1）扩展到（128,3）才可以与dist进行相减运算
        # 再次扩充，恢复簇的维度，方便计算
        min_dist = min_dist.view(min_dist.size(0), 1).expand(num_features, num_clusters)
        # -alpha（数据到所有簇的距离-最小距离）
        stack_exp = torch.exp(-alpha * (dist - min_dist))  # stack_exp(128,3)是所有数据到k个簇中心的距离的指数
        # 所有指数求和
        sum_exponentials = torch.sum(stack_exp, dim=1)  # sum_exponentials（128,1）是所有数据到k个簇中心的距离的指数的和

        # 2.3 compute softmaxes and the embedding/representative distances weighted by softmax
        # sum_exponentials = sum_exponentials.expand(x.data.size(0),y.data.size(0)) # 把sum_exponentials从(128,1)扩展到（128,3）才能进行下式除法运算
        # 再次扩充，恢复簇的维度，方便计算
        sum_exponentials = sum_exponentials.view(sum_exponentials.size(0), 1).expand(num_features, num_clusters)  # 把sum_exponentials从(128,1)扩展到（128,3）才能进行下式除法运算
        # 求一个softmax，stack_exp算各个值的指数，sum_exponentials是求一个和
        # [batch size, 簇数]
        softmax = stack_exp / sum_exponentials  # softmax2（128,3）
        # weighted_dist是每个点到各个簇的损失矩阵，之后KMeansCriterion会对每个点到各个簇的损失求和，dim取1，如何求平均
        weighted_dist = dist * softmax
        return qij, weighted_dist

    def get_centroid(self):
        return self.centroid

def calculateP(q):
    # Function to calculate the desired distribution Q^2, for more details refer to DEC paper
    q_2 = torch.pow(q, 2)
    q_sum = q.sum(0)
    pij_numerator = q_2 / q_sum  # 分子
    normalizer_p = pij_numerator.sum(1).unsqueeze(1)  # 分母
    P = pij_numerator / normalizer_p
    return P

def calculate_center_dist(center):
    dist = torch.pdist(center, 2) # 求行和行之间L2范数
    dist = torch.sort(dist).values
    dist_max = dist[len(dist)-1]
    dist = dist / dist_max

    dist_mid = dist[len(dist) // 2 + 1].item()

    # dist_loss = 0
    dist = torch.exp(dist_mid - dist)
    # 有些距离相等的不做考虑，界限是1
    loss_num = torch.sum(dist > 1)
    dist_loss = torch.sum(dist[dist > 1])
    dist_loss /= (loss_num)
    return dist_loss

def calculate_neighbor_dist(h, h_neighbor):
    num_neighbor = len(h_neighbor) // len(h)
    xx = h_neighbor.view(len(h), num_neighbor, h_neighbor.shape[1])
    yy = h.unsqueeze(1).expand(h.shape[0], num_neighbor, h.shape[1])
    dd = xx - yy
    dist = torch.pow(dd, 2).sum(2).squeeze()  # 每个数据到邻居的距离
    
    dist = torch.sort(dist).values
    dist_max = dist[:,num_neighbor-1]
    dist_max = dist_max.unsqueeze(1).expand(dist_max.shape[0], num_neighbor)
    dist_max = dist_max.clamp(min=1)
    dist = dist / dist_max

    return torch.mean(dist) #/ h_neighbor.size(0)
