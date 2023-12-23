import json, pickle
import h5py
import numpy as np
import pandas as pd
import random
from shapely.geometry import Point, MultiPoint
from utils.SpatialRegionTools import lonlat2meters, coord2cell, SpatialRegion

from osgeo import ogr, osr

driver = ogr.GetDriverByName("ESRI Shapefile")
data_source_point = driver.CreateDataSource("./shp/avg_split_moni_final_split_point-500.shp")
data_source_mian = driver.CreateDataSource("./shp/avg_split_moni_final_split_mian-500.shp")
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def read_data(data_file):
    ## 纬度在前，经度在后 [latitude, longitude]
    coords = []
    coords_set = set()
    with h5py.File(data_file, "r") as f:  
        num = f.attrs.get("num")[0]  # 该文件下轨迹数量
        for i in range(1, num):
            trip = f["trips"][str(i)]  # 第 i 条路径
            for lon, lat in trip:
                if not str(lon) + "_" + str(lat) in coords_set:
                    coords.append(np.array([lat, lon]))
                    coords_set.add(str(lon) + "_" + str(lat))
    coords = np.array(coords)
    df = pd.DataFrame({'lat': coords[:, 0], 'lon': coords[:, 1], 'type': -1, 'cluster': 0})
    return coords, df


colors = []
for i in range(3000):
    colors.append(randomcolor())
colors = np.array(colors)
mapbox_access_token = 'pk.eyJ1Ijoiamg5NzkwMyIsImEiOiJjbDNxdThmNHgwamFrM2xxZXIzN2Q1a2xtIn0.Z5zaksGuC7Yp8qim88ghyg'

k = 20  # 20
kt = 10  # 10
min_pts = 12  # 12


if __name__ == '__main__':
    with open("conf/preprocess_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        # 读取配置文件
        needTime = False
        cellsize = json_data["distcellsize"]
        cityname = json_data["cityname"]
        paramfile = json_data["paramfile"]  # 存放研究区域的参数
        stme_file = json_data["spatiosplitobj"] + "_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".pkl"
        use_grid = True # 代表是否采用均匀网格
        has_label = json_data["hasLabel"] == 1

    with open(stme_file, 'rb') as f:  # open file with write-mode
        hulls = pickle.loads(f.read())
    df = pd.read_csv("./spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    lats = df['lat']
    lons = df['lon']
    # 研究区域和实验配置类
    region = SpatialRegion(cityname,
                           # 120.170, 30.232, # original 原来180
                           # 120.047, 30.228,  # original 原来180
                           120.098, 30.263, # moni
                           # 116.1, 39.5, # e2dtc
                           # 7.999, 39.999, # cross
                           # 8.001, 40.000,# lab
                           #    -8.735152, 40.953673, # porto
                           #    -8.156309, 41.307945,
                           # 120.233, 30.419,
                           120.197, 30.320, # moni
                           # 116.8, 40.3, # e2dtc
                           # 8.0031, 40.0019, # lab
                           # 8.007, 40.004, # cross
                           0, 86400,  # 时间范围,一天最大86400(以0点为相对值)
                           cellsize, cellsize,
                           timecellsize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           30,  # k
                           4,  # vocab_start 词汇表基准值
                           interesttime,  # 时间阈值
                           nointeresttime,
                           delta,
                           needTime,
                           2, 4000,
                           timefuzzysize, timestart,
                           hulls, use_grid, has_label)
                           #points, spatio_num, spatio_pos)
    for id, (lat, lon) in enumerate(zip(lats, lons)):
        x, y = lonlat2meters(lon, lat)
        belong_c = coord2cell(region, x, y)
        df.iloc[id, 4] = belong_c
    df.loc[:, 'lon'] -= 0.007
    df.loc[:, 'lat'] -= 0.0065
    lats = df['lat']
    lons = df['lon']
    # 点ok，更改面
    layer_point = data_source_point.CreateLayer("avg_split_moni_point_post", srs, ogr.wkbPoint)
    field_name = ogr.FieldDefn("label", ogr.OFTString)
    layer_point.CreateField(field_name)

    cluster_labels = df['cluster']
    hulls = []
    hull_cnt = []
    hull_label = []
    print(len(df[df['cluster'] == 0]))
    for k in list(cluster_labels.unique()):
        points = []
        for data in df[df['cluster'] == k].itertuples():
            points.append(Point(getattr(data, 'lat'), getattr(data,
                                                              'lon')))  # wkt = 'POINT(' + str(getattr(data, 'lon')) + ' ' + str(getattr(data, 'lat')) + ')'
            wkt = 'POINT(' + str(getattr(data, 'lon')) + ' ' + str(getattr(data, 'lat')) + ')'
            point = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(layer_point.GetLayerDefn())
            feature.SetField("label", str(k))
            feature.SetGeometry(point)
            layer_point.CreateFeature(feature)
        if len(points) <= 2:
            for data in df[df['cluster'] == k].itertuples():
                points.append(Point(getattr(data, 'lat') + 0.0000001, getattr(data,
                                                                  'lon') + 0.0000001))
                points.append(Point(getattr(data, 'lat') - 0.0000001, getattr(data,
                                                                  'lon') - 0.0000001))
                points.append(Point(getattr(data, 'lat') + 0.0000001, getattr(data,
                                                                  'lon') - 0.0000001))
                points.append(Point(getattr(data, 'lat') - 0.0000001, getattr(data,
                                                                  'lon') + 0.0000001))
            # continue
        # print(Point)
        # print(getattr(data, 'lat'), getattr(data, 'lon'))
        # print(getattr(data, 'lat') - 0.01, getattr(data, 'lon') - 0.01)
        multipoints = MultiPoint(points)
        multipoints.convex_hull.boundary.coords
        hulls.append(multipoints.convex_hull)
        hull_label.append(k)
        hull_cnt.append(len(df[df['cluster'] == k]))

    print(hull_cnt)

    boundary = []
    for hull in hulls:
        boundary.append(list(hull.boundary.coords))
    boundary = np.array(boundary)

    layer_polygon = data_source_mian.CreateLayer("avg_split_moni_polygon_post", srs, ogr.wkbPolygon)
    field_name = ogr.FieldDefn("label", ogr.OFTString)
    field_cnt = ogr.FieldDefn("count", ogr.OFTInteger)
    layer_polygon.CreateField(field_name)
    layer_polygon.CreateField(field_cnt)

    for (hull, cnt, k) in zip(hulls, hull_cnt, hull_label):
        print(cnt)
        print(k)
        print(hull)
        hull_coords = hull.boundary.coords
        hull_coords = [[x[1], x[0]] for x in hull_coords]
        wkt = 'POLYGON(('
        for id, x in enumerate(hull_coords):
            wkt += str(x[0]) + ' ' + str(x[1])
            wkt += ',' if id < len(hull_coords) - 1 else ''
        wkt += '))'
        polygon = ogr.CreateGeometryFromWkt(wkt)
        feature = ogr.Feature(layer_polygon.GetLayerDefn())
        feature.SetField("label", str(k))
        feature.SetField("count", cnt)
        feature.SetGeometry(polygon)
        layer_polygon.CreateFeature(feature)
