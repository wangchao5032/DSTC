import os
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from collections import OrderedDict
from sklearn.manifold import TSNE

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

        count_sum[i] = len(data_id)
        count_correct[i] = count[count_most]
        count_error[i] = count_sum[i] - count_correct[i]
        # count_label[i] = count_most + 1
        count_label[i] = count_most

    error_total = sum(count_error)  # 错分的总数
    return error_total,res

# 计算nmi
def calculate_nmi(y_pred, y_true):
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    return nmi

def calculate_ari(y_pred, y_true):
    score = metrics.adjusted_rand_score(y_pred, y_true)
    return score

# 保存图片
def save_png(file_original_path, file_png, k, labels,epoch=-1, error=None):
    if not os.path.exists(file_png):
        os.makedirs(file_png)

    colors = ['red', 'green', 'yellow', 'c', 'darkred', 'lightcoral', 'tan', 'blue', 'brown', 'wheat', 'navy',
              'springgreen', 'orange', 'turquoise', 'slateblue', 'teal', 'plum', 'skyblue', 'hotpink', 'pink',
              'lightgreen']
    f = h5py.File(file_original_path, 'r')  # 读取原始数据
    for i in range(k):
        temp = np.where(labels == i)  # 筛选第i个簇的所有样本
        print("保存第{0}个簇的图片".format(str(i+1)))
        for j in temp[0]:
            data_ori = f['trips']['%d' % (1 + j)]  # 原始轨迹
            long = data_ori[:, 0]  # 轨迹的经度
            lat = data_ori[:, 1]  # 轨迹的纬度
            # if j not in error:
            #     plt.plot(long, lat, color=colors[i])
            #     # plt.scatter(long[0], lat[0], s=100, color='black')
            # else:
            #     plt.plot(long, lat, color=colors[i])
            #     # plt.plot(long, lat, color='black')
            #     # plt.scatter(long[0], lat[0], s=20, color='red')
            plt.plot(long, lat, linewidth=5)

            plt.axis([119.97, 120.25, 30.21, 30.34])
            # plt.show()
            # plt.savefig(os.path.join(file_png, 'cluster:%d.png' % (i + 1)))
            plt.savefig(os.path.join(file_png, 'cluster:%d-%d.png' % (i + 1, j)))
            # plt.savefig(os.path.join(file_png, 'epoch:%d-cluster:%d-%d.png' % (epoch, i + 1, j)))
            plt.cla()
            plt.clf()
            plt.close()

    f.close()

def calculate_feature_center_dist(feature, center):
    num_feature = len(feature)
    num_center = len(center)
    res = 0
    for i in range(num_feature):
        tmp=[]
        for j in range(num_center):
            dist = ((feature[i] - center[j]) ** 2).sum(0)
            tmp.append(dist)
        min_dist = min(tmp)
        res += min_dist
    return res

def save_cluster_h5(file_original_path, file_cluster_path, cluster_labels):
    f = h5py.File(file_original_path)  # 原始数据的路径
    n = f.attrs.get('num')[0]  # 文件中轨迹的数目

    if os.path.exists(file_cluster_path):  # 若聚类结果文件已存在，则覆盖
        os.remove(file_cluster_path)
    f2 = h5py.File(file_cluster_path)  # 生成的聚类结果的路径
    # 为f2创建group
    labels = f2.create_group("labels")  # 存储轨迹的标签
    timestamps = f2.create_group("timestamps")  # 存储轨迹的时间戳
    trips = f2.create_group("trips")  # 存储轨迹的xy坐标
    sources = f2.create_group("sources")  # 存储轨迹第一个点的来源：来自哪个csv文件+第几行

    for i in range(n):
        # 从聚类结果中获得新的标签
        if i<n-1:
            label = cluster_labels[i]
        else:
            label = -1  # train.ori中只有n-1个数据
        # 从原始数据中获得时间戳、轨迹、来源信息, h5文件中轨迹的编号从1开始
        timestamp = f['timestamps']['%d' % (i + 1)]
        trip = f['trips']['%d' % (i + 1)]
        source = f['sources']['%d' % (i + 1)]

        # 写入到新的文件中
        labels.create_dataset(str(i + 1), data = label)
        timestamps.create_dataset(str(i + 1), data = timestamp)
        trips.create_dataset(str(i + 1), data = trip)
        sources.create_dataset(str(i + 1), data = source)
    # 在f2中写入属性
    f2.attrs.create('num',[i+1],dtype=int)

    f.close()
    f2.close()

# 计算向量之间的欧氏距离
def eucl_dist(a,b):
    dist = np.linalg.norm(a - b)
    return dist

# 挑选距离【簇中心】的距离比较近的轨迹作为热点轨迹，比较远的轨迹为异常轨迹
def find_hot_abnormal(data, k, centroids, labels):
    for i in range(k):
        # 计算每个簇中对象距离簇中心的距离
        centroid_i = centroids[i]  # 第i个簇中心的特征
        d_list = []
        temp = np.where(labels == i)  # 筛选第i个簇的所有样本
        for j in temp[0]:  # j是在原数据中的顺序
            d = eucl_dist(centroid_i, data[j].numpy())
            d_list.append(d)
        # 筛选热点和异常轨迹
        d_list_sort = sorted(d_list)
        n = len(d_list_sort)  # 第i个簇的样本数目
        d_threshold_hot = d_list_sort[int(n * 0.05)]  # 距离簇中心的距离最近的10%对象的距离阈值
        d_threshold_ab = d_list_sort[-int(n * 0.05)]  # 距离簇中心的距离最远的10%对象的距离阈值
        res = []
        for j in range(len(d_list)):
            if d_list[j] < d_threshold_hot:  # 筛选热点轨迹
                index = temp[0][j]  # 在原数据中的顺序
                labels[index] = "{0}".format(100 + i)  # 例如第3个簇中热点轨迹的簇标号为103
            elif d_list[j] > d_threshold_ab: # 筛选异常轨迹
                index = temp[0][j]  # 在原数据中的顺序
                labels[index] = "{0}".format(200 + i)  # 例如第3个簇中异常轨迹的簇标号为203

def DoKMeansWithError(feature, file_original_path, k=10, n=3999, center=None, save=False, epoch=-1, tsne_filename="-1", labels=None):
    '''
    读取h5格式的数据，进行kmeans聚类
    :param feature_path: 降维后的特征的路径（h5格式的数据 trj.h5）
    :param k: 簇的个数
    :param n: 只读取前n条轨迹
    :return:
    '''
    # -----1、读取降维后的数据
    if type(feature) is str:
        f = h5py.File(feature, 'r')
        data = f['layer3'][0:n]
        f.close()
    else:
        data = feature

    # -----2、读取真值数据
    ori_labels_dict = {}
    if labels is None:
        f = h5py.File(file_original_path, 'r') # 读取原始数据（h5文件）
        # 遍历每个轨迹数据获取其标签值
        for i in range(n):
            label = f['labels']['%d' % (1 + i)]  # 原始标签
            ori_labels_dict[i] = int(label[()])
        f.close()
    else:
        for i, label in enumerate(labels):
            ori_labels_dict[i] = label

    # -----3、kmeans聚类
    # (1)聚类
    if center is None:
        inertia_start = 0
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(data)
    else:
        inertia_start = calculate_feature_center_dist(data.cpu().data.numpy(),center.numpy())
        kmeans = KMeans(n_clusters=k, init=center.numpy(), n_init=1, tol=1e-5, max_iter=10000).fit(data)
    # 得到簇中心
    centroids = kmeans.cluster_centers_
    # 计算每个轨迹分配到最近的簇中心后求出其距离和
    inertia_end = calculate_feature_center_dist(data.cpu().data.numpy(), centroids)
    n_iter = kmeans.n_iter_
    labels = kmeans.labels_

    # (2) 修改簇标签
    labels_dict = {}
    for i in range(len(labels)):
        # 获取每个轨迹的聚类后label
        labels_dict[i] = labels[i]

    # -----4、计算指标
    # 计算聚类结果的指标
    error_total, error_tri = calculate_error(k, labels_dict, ori_labels_dict)  # 错分概率
    nmi = calculate_nmi(list(labels_dict.values()), list(ori_labels_dict.values()))
    ari = calculate_ari(list(labels_dict.values()), list(ori_labels_dict.values()))
    # 计算轮廓系数
    score = silhouette_score(data, kmeans.labels_, metric='euclidean')

    # -----5、绘制聚类结果并保存
    # save_cluster_h5(file_original_path, file_cluster_path, labels)
    if save:
        # (1) 生成聚类结果cluster.h5文件，结构和original.h5一样，label是聚类后的标签
        # print("=> saving cluster result into {}".format(file_cluster_path))
    #     save_cluster_h5(file_original_path, file_cluster_path, labels)
        # (2) 保存图片
        # save_png(file_original_path, 'cluster/png_new_1000_3_bestclustermodel/', k, labels)
        DoTSNE(data, 2, k, labels_dict, ori_labels_dict, centroids, 'cluster_png/predict_tsne_'+ tsne_filename +'/',epoch=epoch)

    cluster_data_neighbors = {}
    for i in range(k):
        data_ids = get_keys(labels_dict, i)
        for data_id in data_ids:
            neighbor = np.array(data_ids)[np.random.randint(len(data_ids)/2, len(data_ids), 3)]
            cluster_data_neighbors[data_id] = neighbor
            
    return centroids, error_total, nmi, ari, cluster_data_neighbors, labels_dict

# 计算validity
#《Determination of Number of Clusters in K-Means Clustering and Application in Colour Image Segmentation》
def cal_validity(features, centroids, labels):
    # 计算intra
    n = features.shape[0]
    k = centroids.shape[0]
    d_tot = 0
    for i in range(k):
        centroid_i = centroids[i]  # 第i个簇中心的特征
        temp = np.where(labels == i)  # 筛选第i个簇的所有特征
        for j in temp[0]:
            feature_j = features[j]
            d = sum(np.power((centroid_i - feature_j.numpy()), 2))
            d_tot += d
    intra = d_tot / n
    # 计算inter
    min_d = float("inf")
    for i in range(k):
        for j in range(k):
            if i==j:
                continue
            d = sum(np.power((centroids[i] - centroids[j]), 2))
            min_d = min(min_d,d)
    inter = min_d
    score = intra/inter
    return score


def CalClusterNum(feature):
    scores = []
    start,end = 2,25
    for i in range(start,end+1):
        # 构建并训练模型
        kmeans = KMeans(n_clusters=i, random_state=0, init='k-means++').fit(feature)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        # 计算轮廓系数
        # score = silhouette_score(feature, kmeans.labels_, metric='euclidean')
        # 计算Calinski-Harabasz指数
        # score = calinski_harabasz_score(feature, kmeans.labels_)
        # 计算validity
        score = cal_validity(feature, centroids, labels)
        scores.append(score)
        print('数据聚%d类calinski_harabaz指数为：%f' % (i, score))

    # 绘制图表
    X = list(range(start, end + 1))
    plt.plot(X, scores)
    # plt.show()
    plt.savefig("score.png")
    plt.cla()
    plt.clf()
    plt.close()

def DoDBSCANWithError(feature_path, file_original_path, k=19, n=1899, center=None, feature=None, save=False,epoch=-1):
    '''
    读取h5格式的数据进行kmeans聚类
    :param feature_path: 降维后的特征的路径（h5格式的数据 trj.h5）
    :param k: 簇的个数
    :param n: 只读取前n条轨迹
    :return:
    '''

    # -----1、读取降维后的数据
    if feature is None:
        f = h5py.File(feature_path, 'r')  # 打开h5文件
        data = f['layer3'][0:n]  # 取出主键为layer3的前n个数据
        f.close()
    else:
        data = feature

    # -----2、读取真值数据
    f = h5py.File(file_original_path, 'r') # 读取原始数据（h5文件）
    ori_labels_dict = {}
    for i in range(n):
        label = f['labels']['%d' % (1 + i)]  # 原始标签
        ori_labels_dict[i] = int(label[()])
        # ori_labels_dict[i] = int(label[0][0])
    f.close()

    # -----3、kmeans聚类
    dbscan = OPTICS().fit(data)
    labels = dbscan.labels_

    labels_dict = {}
    for i in range(len(labels)):
        labels_dict[i] = labels[i]  # +1让聚类结果的簇号从1-19，和真值保持一致
        # labels_dict[i] = labels[i] + 1  # +1让聚类结果的簇号从1-19，和真值保持一致

    # -----3、计算错分概率
    error_total, error_tri = calculate_error(k, labels_dict, ori_labels_dict)
    nmi = calculate_nmi(list(labels_dict.values()), list(ori_labels_dict.values()))

    # -----4、绘制聚类结果并保存
    if save == True:
        save_png(file_original_path, 'cluster/png_new/', k, labels, epoch=epoch,error=error_tri)
    centroids=None
    inertia_start=None
    inertia_end=None
    n_iter=None
    return centroids, error_total, nmi, inertia_start, inertia_end, n_iter

def DoPCA(data, component_num):
    '''
    PCA降维
    :param data: 保留的主成分个数
    :param component_num:
    :return:
    '''
    pca = PCA(n_components=component_num)
    feature = pca.fit_transform(data)
    print("component_num:{} \t sum:{}\t".format(component_num, sum(pca.explained_variance_ratio_)))
    return feature

def DoTSNE(features, n_components, k, labels_dict, ori_labels_dict, centers,file_png,epoch=-1):
    '''
    TSNE降维
    :param data:
    :param n_components:
    :return:
    '''
    data = np.vstack((features, centers))
    # n_components 嵌入空间的维度
    data_embedded = TSNE(n_components=n_components, init='pca', perplexity=10).fit_transform(data)

    features_embedded = data_embedded[:len(features)]
    centers_embedded = data_embedded[len(features):]

    colors = ['red', 'green', 'yellow', 'c', 'darkred', 'lightcoral', 'tan', 'blue', 'orange','lightgreen',
              'skyblue', 'wheat', 'navy','pink',
              'springgreen',  '#C76DA2', 'slateblue', 'teal', 'plum', 'skyblue', 'hotpink',"fuchsia", "aqua"
              ]
    # colors = ['green', 'yellow', 'c', 'darkred', 'lightcoral', 'tan', 'blue', 'lime', 'fuchsia', 'aqua',]
    if not os.path.exists(file_png):
        os.makedirs(file_png)
    # ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    plt.figure(dpi=400)
    for i in range(k):
    # for i in [0,15]:
        # 先绘制簇中的每个数据
        data_id = get_keys(labels_dict, i)  # 聚类结果中第i个簇（簇号为i+1）对应的数据的ID
        for j in data_id:
            # ax.scatter(features_embedded[j][0], features_embedded[j][1], features_embedded[j][2], s=1, alpha=1,color=colors[i], label=str(i) + "/" +colors[i])
            plt.scatter(features_embedded[j][0], features_embedded[j][1], s=1, alpha=1,color=colors[i], label=str(ori_labels_dict[j])+"_"+str(i) + "/" +colors[i])
        # 再绘制第i个簇中心
        plt.scatter(centers_embedded[i][0], centers_embedded[i][1], s=20, color='hotpink')
        # ax.scatter(centers_embedded[i][0], centers_embedded[i][1], centers_embedded[i][2], s=100, color='hotpink')

        # plt.axis([-8.735152, -8.156309, 40.953673, 41.307945])
        # plt.savefig(os.path.join(file_png, '%d-%d.png' % (epoch,i)))
        # plt.cla()
        # plt.clf()
        # plt.close()
    plt.gcf().subplots_adjust(right = 0.8)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="truth_clusterlabel/color", markerscale=10., scatterpoints=1, loc=(1.1, 0.5))
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(file_png, 'epoch-%d.png' % (epoch)), bbox_inches='tight')
    # plt.savefig(os.path.join(file_png, 'epoch-%d.png' % (epoch)))
    plt.cla()
    plt.clf()
    plt.close()

    plt.figure(dpi=400)
    for i in range(k):
    # for i in [0,15]:
        # 先绘制簇中的每个数据
        data_id = get_keys(labels_dict, i)  # 聚类结果中第i个簇（簇号为i+1）对应的数据的ID
        for j in data_id:
            # ax.scatter(features_embedded[j][0], features_embedded[j][1], features_embedded[j][2], s=1, alpha=1,color=colors[i], label=str(i) + "/" +colors[i])
            plt.scatter(features_embedded[j][0], features_embedded[j][1], s=1, alpha=1,color=colors[i])
        # 再绘制第i个簇中心
        # plt.scatter(centers_embedded[i][0], centers_embedded[i][1], s=20, color='hotpink')
        # ax.scatter(centers_embedded[i][0], centers_embedded[i][1], centers_embedded[i][2], s=100, color='hotpink')

        # plt.axis([-8.735152, -8.156309, 40.953673, 41.307945])
        # plt.savefig(os.path.join(file_png, '%d-%d.png' % (epoch,i)))
        # plt.cla()
        # plt.clf()
        # plt.close()
    # plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(file_png, 'nonelabel_epoch-%d.png' % (epoch)))
    # plt.savefig(os.path.join(file_png, 'epoch-%d.png' % (epoch)))

def DoKMeans(feature_path, k=19, n=1899):
    '''
    读取h5格式的数据，进行kmeans聚类
    :param feature_path: 降维后的特征的路径（h5格式的数据 trj.h5）
    :param k: 簇的个数
    :param n: 只读取前n条轨迹
    :return:
    '''

    # -----1、读取降维后的数据
    f = h5py.File(feature_path, 'r')  # 打开h5文件
    data = f['layer3'][0:n]  # 取出主键为layer3的前n个数据
    f.close()

    # -----2、kmeans聚类
    kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(data)
    # kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    return centroids

#%%


