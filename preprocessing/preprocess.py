import sys
sys.path.append('..')
import os
import json
import datetime, time
from SpatioSplit import divideByDensity
from GenerateToken import preprocess, PreprocParams

def reset_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        return
    files = os.listdir(dir_name)
    # 遍历删除指定目录下的文件
    for file in files:
        os.remove(os.path.join(dir_name, file))
        print(file, "删除成功")
    print(dir_name, "删除成功")

if __name__ == "__main__":
    # 时间计算
    print("START DATETIME ", datetime.datetime.now())

    ## 1, 初始化
    with open("conf/preprocess_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        minLon = json_data["minLon"]
        minLat = json_data["minLat"]
        maxLon = json_data["maxLon"]
        maxLat = json_data["maxLat"]
        assert minLon is not None, "minLon not set"
        assert minLat is not None, "minLat not set"
        assert maxLon is not None, "maxLon not set"
        assert maxLat is not None, "maxLat not set"
        if json_data["timeCellSize"] > 0:
            needTime = True
            minTime = json_data["minTime"]
            maxTime = json_data["maxTime"]
            assert minTime is not None, "minTime not set"
            assert maxTime is not None, "maxTime not set"
            timeCellSize = json_data["timeCellSize"]
        else:
            needTime = False
        gridCellSize = json_data["gridCellSize"]
        dataName = json_data["dataName"]
        k = json_data["stme_k"]
        kt = json_data["stme_kt"]
        min_pts = json_data["stme_min_pts"]
        if json_data["method"] == 'grid':
            use_grid = True
        elif json_data["method"] == 'density':
            use_grid = False
        else:
            print(f'method must be "density" or "grid", but {json_data["method"]} detected')
            exit(0)
        has_label = json_data["hasLabel"]
    # 删除上一次的结果
    reset_dir("../data/")
    hulls = None
    if not use_grid:
        h5_file = f'originData/{dataName}.h5'
        k, kt, min_pts = json_data['stme_k'], json_data['stme_kt'], json_data['stme_min_pts']
        hulls = divideByDensity(h5_file, k, kt, min_pts)

    preprocess(PreprocParams(dataName, minLon, minLat, maxLon, maxLat, minTime, maxTime,
                                    gridCellSize, timeCellSize, hulls, use_grid, has_label, needTime))

    print("END DATETIME ", datetime.datetime.now())