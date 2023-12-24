import math
import sys
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import spatial 

def Dist(data, i, j, tWindow):
    """
    计算两点之间的距离（暂时没用）
    :param data: 数据集
    :param i: 数据集中的第i个对象
    :param j: 数据集中的第j个对象
    :param tWindow: 时间窗口
    :return: 数据集中的第i个对象和第j个对象之间的距离
    """
    dist_i_j = float("inf")
    if i == j:
        return dist_i_j
    # 提取第i和j个对象的时间、经度、维度
    t1, t2 = data.values[i][0], data.values[j][0]
    x1, x2 = data.values[i][1], data.values[j][1]
    y1, y2 = data.values[i][2], data.values[j][2]

    if abs(t1 - t2) <= tWindow:
        dist_i_j = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return dist_i_j


def GetKN(data, k, t_window):
    """
    计算所有点的k近邻距离，以及K近邻
    :param data: 数据集
    :param k: k
    :param t_window: 时间窗口
    :return:
    """
    rows = data.shape[0]  # 行数

    dist_matrix_topK = np.zeros(rows)  # 所有点的k近邻矩阵：第0列是不考虑类型的k距离，第1列是第1种类型的k距离，以此类推
    k_neighbor_idx = np.zeros((rows, k))  # 所有点的k近邻编号
    kdtree = spatial.KDTree(data[['lon', 'lat']])
    # 计算距离矩阵 查找每个点的k近邻及距离
    print("正在计算距离矩阵...")
    for i in range(rows):
        if i % 10000 == 0:
            print("计算距离矩阵:正在处理第{0}个轨迹点".format(i))
        [topk_dist, topk_id] = kdtree.query(data.loc[i][['lon', 'lat']], k + 1)
        dist_matrix_topK[i] = sorted(topk_dist[1:])[k-1]
        k_neighbor_idx[i] = topk_id[1:]
    k_neighbor_idx = k_neighbor_idx.astype(int)  # 把编号转为int类型
    return dist_matrix_topK, k_neighbor_idx


def GetRN(topK_neighbor, kt):
    """
    计算每个点的共享近邻
    :param topK_neighbor:所有点的k近邻编号
    :param kt:kt
    :return: 每个点的共享近邻列表
    """
    rows = topK_neighbor.shape[0]  # 数据集中点的数目
    rstn = []  # 每个点的共享近邻列表   [[] for _ in range(rows)]
    print("正在计算共享近邻...")
    for i in range(rows):
        if i % 10000 == 0:
            print("计算共享近邻:正在处理第{0}个轨迹点".format(i))
        rstn_i = []  # 第i个点的共享近邻列表
        neighbor_i = topK_neighbor[i]  # 第i个点的k近邻
        for j in neighbor_i:  # 第i个点的j邻居
            neighbor_j = topK_neighbor[int(j)]  # 第j个点的k近邻
            neighbor_i_j = list(set(neighbor_i).intersection(set(neighbor_j)))  # 第i个点和第j个点的k近邻交集
            # 如果i和j的共享近邻个数大于kt，且i也是j的k近邻
            if len(neighbor_i_j) >= kt and (i in neighbor_j):
                rstn_i.append(j)
        rstn.append(rstn_i)
    return rstn


# def STME(data, k, kt, t_window, min_pts, distK_sigma_multi, distK_sigma_times, spatiotemoral=False):
def STME(data, k, kt, t_window, min_pts, spatiotemoral=False):
    n_data = data.shape[0]  # 行数
    n_types = max(data['type']) + 1# max(data['type']) + 1  # 点的类型数
    print("共{0}个轨迹点,{1}种类型".format(n_data, n_types))

    # 1. 计算每个点的k近邻距离，以及最近的K个邻居
    dist_topK, topK_neighbor = GetKN(data, k, t_window)

    # 2. 计算聚类优先级（每个点第k个邻居的升序排序后的距离）
    print("正在计算聚类优先级...")
    # dist_topk_alltype = dist_topK[:, 0]  # 所有数据的k近邻距离
    dist_topk_alltype = dist_topK.copy()  # 所有数据的k近邻距离
    priority = np.argsort(dist_topk_alltype)

    # 3. 计算每个点的refined neighbors
    RSTN = GetRN(topK_neighbor, kt)

    # 4. 聚类
    label = 0  # 初始化簇标号
    clusters_density = [[] for _ in range(n_types+1)]  # 簇有效密度列表：第0列是不考虑类型的有效密度，第1列是第1种类型的有效密度，以此类推
    clusters_num = [[] for _ in range(n_types+1)]  # 簇中点数：第0列存的是簇中点的总数，第1列是簇中第1种类型点的数目，以此类推
    print("开始聚类...")
    for p in priority:  # 按照第k个邻居距离的升序进行枚举，也就是第p个点一定是目前所有点里k邻域范围最小的
        # 判断第p个点是否已经被处理过了
        if data.iloc[p].at['cluster'] != 0:
            continue

        # 判断第p个点是否是噪声
        r_neighbor = RSTN[p]  # 第p个点的refined neighbors / 直接可达点
        if len(r_neighbor) < min_pts:  # 如果第p个点是噪声，不是核心点
            data.iloc[p, 3] = -1
            continue

        # 判断refined neighbors中free_pts的数目
        r_neighbor_free = data.iloc[r_neighbor].query('cluster==0 | cluster==-1')  # 查找第p个点的refined neighbors中标签为0或-1的点
        normed_density = data.iloc[r_neighbor]['cluster'].values.tolist() # 第p个点的refined neighbors中的标签
        r_neighbor_most_label = max(normed_density, key=normed_density.count)  # 第p个点的refined neighbors中出现最多的类型
        if len(r_neighbor_free) < min_pts and r_neighbor_most_label != 0:  # 第p个点的refined neighbors中标签为0或-1的点的数目小于MinPts，且出现最多的类型不是0 
            data.iloc[p, 3] = r_neighbor_most_label# 那么类型赋值给最多的label
            continue

        # 第p个点是核心点，开始一个新的簇！ 否则就是全新的一个点
        label += 1
        print("cluster:{0}".format(label))
        # 构造初始队列
        queue = [p] + r_neighbor  # 将第p个点及其RSTN加入队列
  
        # 遍历队列
        while len(queue) > 0:
            cur_pt = queue[0]  # 取出队列中的第一个点作为当前点
            queue.remove(queue[0])  # 把当前点移出队列
            if data.iloc[cur_pt].at['cluster'] > 0:  # 如果当前点被标记过，这个点只可能是作为共享近邻被加入队列了
                continue
            data.iloc[cur_pt, 3] = label

            # (3) 遍历当前点的k邻居，加入队列
            cur_topK_neighbor = topK_neighbor[cur_pt]  # 队列中当前点的k个邻居
            cur_RSTN = RSTN[cur_pt]  # 队列中当前点的共享邻居/直接可达点
            if len(cur_RSTN) >= min_pts:  # 如果当前点是核心点
                # type_following.append(data.iloc[cur_pt].at['type'])
                for neigh in cur_topK_neighbor:
                    # 如果当前点的邻居:a) 是噪声或还没有被标记过; b)不在对列中;
                    # c) 在只考虑有影响力的类型下，当前点邻居的考虑类型的k距离 和 初始簇考虑类型的k距离均值 的差值distK_diff(type_valid)，若差值在初始簇的3倍标准差以内
                    if (data.iloc[neigh].at['cluster'] == -1 or data.iloc[neigh].at['cluster'] == 0) \
                            and (not (neigh in queue)):
                            # and all(distK_diff[type_valid] < distK_sigma[type_valid] * distK_sigma_times):
                        queue.append(neigh)
        # 计算当前簇的点数
        cluster = data[data['cluster'] == label]  # 当前簇
        clusters_num[0].append(len(cluster))  # 当前簇的点的总数
        for i in range(n_types):
            clusters_num[i+1].append(len(cluster[cluster['type'] == i]))

        # 计算当前簇有效密度
        if spatiotemoral==False or all(cluster['t'].values == np.zeros(len(cluster))):  # 如果计算的是二维凸包或时间维度一列都是0
            hull = ConvexHull(cluster[['lat', 'lon']].values)  # 当前簇的最小凸包
        else:
            hull = ConvexHull(cluster[['timestamp', 'lat', 'lon']].values)
        V = hull.volume
        clusters_density[0].append(clusters_num[0][0] / V)  # 当前簇的密度=当前簇的数量/当前簇的体积
        for i in range(n_types):  # 当前簇的第i类点的密度 = 当前簇中该类点的数量/当前簇的体积
            clusters_density[i+1].append(clusters_num[i+1][0] / V)

    # 5. 计算簇的归一化混合密度
    print("开始计算簇的归一化混合密度...")
    cluster_normed_density = []  # 簇的归一化混合密度：第0列是第1个簇的归一化密度，以此类推
    nClusters = len(clusters_density[0])
    if nClusters > 1:
        for i in range(nClusters):
            normed_density = (clusters_density[0][i] - min(clusters_density[0])) / (max(clusters_density[0]) - min(clusters_density[0]))
            cluster_normed_density.append(normed_density)
    else:
        cluster_normed_density.append(1)

    return data, cluster_normed_density, label
