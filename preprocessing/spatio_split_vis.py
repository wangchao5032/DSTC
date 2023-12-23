import json, pickle, os
import plotly.graph_objects as go
import h5py
import numpy as np
import pandas as pd
import random
from shapely.geometry import Point, MultiPoint, GeometryCollection

from osgeo import gdal, ogr, osr

driver = ogr.GetDriverByName("ESRI Shapefile")
data_source_point = driver.CreateDataSource("./shp/spatio_split_moni_final_split_point.shp")
data_source_mian = driver.CreateDataSource("./shp/spatio_split_moni_final_split_mian.shp")
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
    with h5py.File(data_file,
                   "r") as f:  # 20200101_jianggan hzd2zjg_reorder_3_inte hangzhou_simu_400x10_v2_inte xihumake_2d
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
        stme_file = "./spatio_splits/stme_obj" + "_" + str(k) + "_" + str(kt) + "_" + str(
            min_pts) + ".pkl"  # 0.03749155812831645 867
        # stme_file = json_data["spatiosplitobj"] + "_21_10_14.pkl" # 0.02102267245537868 350
    with open(stme_file, 'rb') as f:  # open file with write-mode
        hulls = pickle.loads(f.read())
        
    centers = np.array([np.array(x.centroid.coords) for x in hulls]).squeeze()
    centers[:,0] -= 0.0065
    centers[:,1] -= 0.007
    df = pd.read_csv("./spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    df.loc[:,'lon']-= 0.007
    df.loc[:,'lat']-= 0.0065
    lats = df['lat']
    lons = df['lon']
    for id, (lat, lon) in enumerate(zip(lats, lons)):
        belong_c = np.argmin(np.sum((centers - [lat, lon]) ** 2, 1))
        df.iloc[id, 4] = belong_c
    # 点ok，更改面
    layer_point = data_source_point.CreateLayer("spatio_split_moni_point_post", srs, ogr.wkbPoint)
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
            points.append(Point(getattr(data, 'lat'), getattr(data, 'lon')))    #     wkt = 'POINT(' + str(getattr(data, 'lon')) + ' ' + str(getattr(data, 'lat')) + ')'
            wkt = 'POINT(' + str(getattr(data, 'lon')) + ' ' + str(getattr(data, 'lat')) + ')'
            point = ogr.CreateGeometryFromWkt(wkt)
            feature = ogr.Feature(layer_point.GetLayerDefn())
            feature.SetField("label", str(k))
            feature.SetGeometry(point)
            layer_point.CreateFeature(feature)
        multipoints = MultiPoint(points)
        hulls.append(multipoints.convex_hull)
        hull_label.append(k)
        hull_cnt.append(len(df[df['cluster'] == k]))
    
    print(hull_cnt)
            
    boundary = []
    for hull in hulls:
        boundary.append(list(hull.boundary.coords))
    boundary = np.array(boundary)
    
    layer_polygon = data_source_mian.CreateLayer("spatio_split_moni_polygon_post", srs, ogr.wkbPolygon)
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
