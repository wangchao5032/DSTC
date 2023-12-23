import h5py, json
from sklearn.cluster import DBSCAN
import os
import pandas as pd
from shapely.geometry import Point, MultiPoint, GeometryCollection
import numpy as np
from stme_module import STME

def stme(trjfile, k, kt, min_pts):
    # coords = [latitude, longitude]
    coords = []
    with h5py.File(trjfile, "r") as f:
        num = f.attrs.get("num")[0]  # 该文件下轨迹数量
        print("轨迹数量: " + str(num))
        for i in range(1, num):
            trip = f["trips"][str(i)]  # 第 i 条路径
            for (lon, lat) in trip:
                coords.append(np.array([lat, lon]))
    coords = np.array(coords)
    df = pd.DataFrame({'lat': coords[:, 0], 'lon': coords[:, 1], 'type': -1, 'cluster': 0})

    print("参数: k： " + str(k) + " kt: " + str(kt) + " min_pts: " + str(min_pts))
    df, _, num_clusters = STME(df, k=k, kt=kt, t_window=86400, min_pts=min_pts) 
    cluster_labels = df['cluster']
    raito = len(cluster_labels[cluster_labels[:] == -1]) / len(cluster_labels)  # 计算噪声点个数占总数的比例
    print('rattio:' + str(raito))
    print('Clustered ' + str(len(coords)) + ' points to ' + str(num_clusters) + ' clusters')
    df.to_csv("./spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    
    # 所有簇的点组成的面
    hulls = []
    for n in range(num_clusters):
        points = [Point(i, j) for i, j in coords[cluster_labels == n+1]]
        multipoints = MultiPoint(points)
        hulls.append(multipoints.convex_hull)
    return hulls


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
    epsilon = 1 / kms_per_radian 

    db = DBSCAN(eps=epsilon, min_samples=25, metric='haversine').fit(coords)
    cluster_labels = db.labels_
    raito = len(cluster_labels[cluster_labels[:] == -1]) / len(cluster_labels)  # 计算噪声点个数占总数的比例
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # 获取分簇的数目
    
    # get the number of clusters (ignore noisy samples which are given the label -1)
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

    
def divideByDensity(h5_file, k, kt, min_pts):
    # 空间划分以及时间计算
    hulls = stme(h5_file, k, kt, min_pts)
    # hulls = dbscan(h5_file, k, kt, min_pts)
    
    # with open(stme_file + "_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".pkl", 'wb') as f:
    #     picklestring = pickle.dumps(hulls)
    #     f.write(picklestring)
    return hulls
