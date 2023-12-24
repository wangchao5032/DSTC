import h5py, json
from sklearn.cluster import DBSCAN
import os
import pandas as pd
from shapely.geometry import Point, MultiPoint, GeometryCollection
import numpy as np
from stme_module import STME
import pickle
import datetime, time

def stme(trjfiles, k, kt, min_pts):
    ## 纬度在前，经度在后 [latitude, longitude]
    coords = []
    coords_set = set()
    for trjfile in trjfiles:
        print("文件 ： " + trjfile)
        # if trjfile.startswith("2020"):
        #     continue
        with h5py.File(trjfile, "r") as f:
            num = f.attrs.get("num")[0]  # 该文件下轨迹数量
            print("轨迹数量: " + str(num))
            for i in range(1, num):
                trip = f["trips"][str(i)]  # 第 i 条路径
                for (lon, lat) in trip:
                    if not str(lon) + "_" + str(lat) in coords_set:
                        coords.append(np.array([lat, lon]))
                        coords_set.add(str(lon) + "_" + str(lat))
    coords = np.array(coords)
    df = pd.DataFrame({'lat': coords[:, 0], 'lon': coords[:, 1], 'type': -1, 'cluster': 0})

    print("参数: k： " + str(k) + " kt: " + str(kt) + " min_pts: " + str(min_pts))
    df, clusterDensityList_nor, num_clusters = STME(df, k=k, kt=kt, t_window=86400, min_pts=min_pts) # 0.0381 751
    # earth's radius in km
    cluster_labels = df['cluster']
    raito = len(cluster_labels[cluster_labels[:] == -1]) / len(cluster_labels)  # 计算噪声点个数占总数的比例
    print('rattio:' + str(raito))
    print('Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')
    # df.to_csv("./spatio_splits/lab_spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    df.to_csv("./spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    
    # 所有簇的点组成的面
    hulls = []
    for n in range(num_clusters):
        points = [Point(i, j) for i, j in coords[cluster_labels == n+1]]
        multipoints = MultiPoint(points)
        hulls.append(multipoints.convex_hull)
    return hulls
    # 做交集，变成独立的面
    # curr_hulls = []
    # for n in range(num_clusters):
    #     if hulls[n].is_empty:
    #         continue
    #     if (n == num_clusters - 1):
    #         curr_hulls.append(hulls[n])
    #     else:
    #         if hulls[n].disjoint(GeometryCollection(hulls[n + 1:])):
    #             curr_hulls.append(hulls[n])
    #         else:
    #             # 不相交的地方自行一个
    #             differ = hulls[n].difference(GeometryCollection(hulls[n + 1:]))
    #             if not differ.is_empty:
    #                 curr_hulls.append(differ)
    #             # 相交部分需要提取出来单独一个
    #             # 其他的剪掉该相交部分
    #             curr_hulls.append(hulls[n].intersection(GeometryCollection(hulls[n + 1:])))
    #             for i in range(n + 1, num_clusters):
    #                 if hulls[i].is_empty:
    #                     continue
    #                 hulls[i] -= curr_hulls[-1]
    # return curr_hulls
    

def dbscan(trjfiles):
    ## 纬度在前，经度在后 [latitude, longitude]
    coords = []
    coords_set = set()
    for trjfile in trjfiles:
        with h5py.File(trjfile, "r") as f:
            num = f.attrs.get("num")[0]  # 该文件下轨迹数量
            for i in range(1, num):
                trip = f["trips"][str(i)]  # 第 i 条路径
                for (lon, lat) in trip:
                    if not str(lon) + "_" + str(lat) in coords_set:
                        coords.append(np.array([lat, lon]))
                        coords_set.add(str(lon) + "_" + str(lat))
    coords = np.array(coords)

    # earth's radius in km
    kms_per_radian = 6378.1370 # 6371.0086
    # define epsilon as 0.5 kilometers, converted to radians for use by haversine
    # This uses the 'haversine' formula to calculate the great-circle distance between two points
    # that is, the shortest distance over the earth's surface
    # http://www.movable-type.co.uk/scripts/latlong.html
    epsilon = 1 / kms_per_radian # todo and min_samples
    # epsilon = 15 / kms_per_radian # todo and min_samples

    # radians() Convert angles from degrees to radians
    # db = DBSCAN(eps=epsilon, min_samples=15, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    # db = DBSCAN(eps=epsilon, min_samples=15).fit(coords)
    db = DBSCAN(eps=epsilon, min_samples=25, metric='haversine').fit(coords)
    cluster_labels = db.labels_
    raito = len(cluster_labels[cluster_labels[:] == -1]) / len(cluster_labels)  # 计算噪声点个数占总数的比例
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # 获取分簇的数目
    # score = metrics.silhouette_score(coords, cluster_labels)
    
    
    # get the number of clusters (ignore noisy samples which are given the label -1)
    # num_clusters = len(set(cluster_labels) - set([-1]))
    print('rattio:' + str(raito))
    print('Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')
    df = pd.DataFrame({'lat':coords[:,0], 'lon': coords[:,1]})
    df['label'] = cluster_labels
    # 所有簇的点组成的面
    hulls = []
    for n in range(num_clusters):
        points = [Point(i, j) for i, j in coords[cluster_labels == n]]
        multipoints = MultiPoint(points)
        hulls.append(multipoints.convex_hull)
    # 做交集，变成独立的面
    curr_hulls = []
    for n in range(num_clusters):
        if hulls[n].is_empty:
            continue
        if (n == num_clusters - 1):
            curr_hulls.append(hulls[n])
        else:
            if hulls[n].disjoint(GeometryCollection(hulls[n + 1:])):
                curr_hulls.append(hulls[n])
            else:
                # 不相交的地方自行一个
                differ = hulls[n].difference(GeometryCollection(hulls[n + 1:]))
                if not differ.is_empty:
                    curr_hulls.append(differ)
                # 相交部分需要提取出来单独一个
                # 其他的剪掉该相交部分
                curr_hulls.append(hulls[n].intersection(GeometryCollection(hulls[n + 1:])))
                for i in range(n + 1, num_clusters):
                    if hulls[i].is_empty:
                        continue
                    hulls[i] -= curr_hulls[-1]
    return curr_hulls
    # sns.lmplot('lat', 'lon', df, hue='label', fit_reg=False)
    # plt.show()

    # kmeans = KMeans(n_clusters=1, n_init=1, max_iter=20, random_state=20)
    # cluster_centers = {}
    # for n in range(num_clusters):
    #     # print('Cluster ', n, ' all samples:')
    #     one_cluster = coords[cluster_labels == n]
    #     # print(one_cluster[:1])
    #     # clist = one_cluster.tolist()
    #     # print(clist[0])
    #     kk = kmeans.fit(one_cluster)
    #     cluster_centers[n] = kk.cluster_centers_
    
    # return df, num_clusters , cluster_centers
    
def splitSpatioBasedDensity():
    with open("conf/preprocess_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        # 读取该目录所有文件 也就是需要解析的所有文件、
        k = json_data["stme_k"]
        kt = json_data["stme_kt"]
        min_pts = json_data["stme_min_pts"]
    # 完善文件路径
    h5_files = os.listdir("./originData")
    stme_file = "./spatio_splits/stme_obj"
    for i in range(len(h5_files)):
        # if h5_files[i].startswith("2020"):
        #     continue
        h5_files[i] = os.path.join("./originData", h5_files[i])
    # points, spatio_num, spatio_pos = dbscan(h5_files)
    
    # 空间划分以及时间计算
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)
    hulls = stme(h5_files, k, kt, min_pts)
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))
    
    with open(stme_file + "_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".pkl", 'wb') as f:  # open file with write-mode
        picklestring = pickle.dumps(hulls)
        f.write(picklestring)
    return hulls
