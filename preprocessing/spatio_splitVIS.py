import json, pickle
import plotly.graph_objects as go
import  h5py
import numpy as np
import pandas as pd
import random


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def read_data(data_file):
    ## 纬度在前，经度在后 [latitude, longitude]
    coords = []
    coords_set = set()
    with h5py.File(data_file, "r") as f:  # 20200101_jianggan hzd2zjg_reorder_3_inte hangzhou_simu_400x10_v2_inte xihumake_2d
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
        stme_file = "./spatio_splits/stme_obj" + "_" + str(k) + "_"+ str(kt) + "_" + str(min_pts) + ".pkl" # 0.03749155812831645 867
        # stme_file = json_data["spatiosplitobj"] + "_21_10_14.pkl" # 0.02102267245537868 350
    with open(stme_file, 'rb') as f:  # open file with write-mode
        hulls = pickle.loads(f.read())
    # df = pd.read_csv("./spatio_splits/lab_spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    df = pd.read_csv("./spatio_splits/spatio_point_split_" + str(k) + "_" + str(kt) + "_" + str(min_pts) + ".csv")
    cluster_labels = df['cluster']
    raito = len(cluster_labels[cluster_labels[:] == -1]) / len(cluster_labels)  # 计算噪声点个数占总数的比例
    print('rattio:' + str(raito))
    print('Clustered ' + str(len(df)) + ' points to ' + str(len(list(cluster_labels.unique())) - 1) + ' clusters')
    
    center_loc = df.iloc[len(df) // 2][['lat', 'lon']]
    lats = df['lat']
    lons = df['lon']
    texts = df['cluster']
    boundary = []
    for hull in hulls:
        boundary.append(list(hull.boundary.coords))
    boundary = np.array(boundary)
    fig = go.Figure()
    
    hull_boundary_lons = []
    hull_boundary_lats = []
    for j, hull in enumerate(hulls):
        hull_coords = hull.boundary.coords
        hull_coords = [[x[1], x[0]] for x in hull_coords]
        hull_boundary_lons.append(list(np.array(hull_coords)[:,0]))
        hull_boundary_lats.append(list(np.array(hull_coords)[:,1]))
        # wkt = 'POLYGON(('
        # for id, x in enumerate(hull_coords):
        #     wkt += str(x[0]) + ' ' + str(x[1])
        #     wkt += ',' if id < len(hull_coords) - 1 else ''
        # wkt +=  '))'
        # polygon = ogr.CreateGeometryFromWkt(wkt)
        # feature = ogr.Feature(layer.GetLayerDefn())
        # feature.SetField("label", str(j))
        # feature.SetGeometry(polygon)
        # layer.CreateFeature(feature)
            

    for idx in range(len(hulls)):
        fig.add_trace(go.Scattermapbox(
            name=str(idx),
            mode="lines", fill="toself",  # fillcolor = colors[idx],
            lon=hull_boundary_lons[idx],
            lat=hull_boundary_lats[idx]
        ))
        
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            style='stamen-terrain',
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=center_loc[0],
                lon=center_loc[1]
            ),
            pitch=0,
            zoom=10,
        ),
        margin = {'l':0, 'r':0, 'b':0, 't':0},
        showlegend = False,
    )
    # 
    # # fig.to_image(format="png", engine="orca")
    fig.show()
