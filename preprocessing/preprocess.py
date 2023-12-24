import os
import pickle, json
from utils.SpatialRegionTools import SpatialRegion, createTrainVal, createTrainVal_OnlyOriginal, \
    saveKNearestVocabs, makeVocab
from utils.utils import downsamplingDistort
import datetime, time

from SpatioSplit import splitSpatioBasedDensity
from GenerateToken import runAndGetTrainData, PreTrainData
def dir_delete(dir_name):
    files = os.listdir(dir_name)
    # 遍历删除指定目录下的文件
    for file in files:
        os.remove(os.path.join(dir_name, file))
        print(file, "删除成功")
    print(dir_name, "删除成功")

if __name__ == "__main__":
    # 时间计算
    start_dt = datetime.datetime.now()
    start_t = time.time()
    print("START DATETIME")
    print(start_dt)

    ## 1, 初始化
    with open("conf/preprocess_conf.json") as conf:
        # 读取配置文件
        json_data = json.load(conf)
        needTime = json_data["needTime"]
        minLon = json_data["minLon"]
        minLat = json_data["minLat"]
        maxLon = json_data["maxLon"]
        maxLat = json_data["maxLat"]
        assert minLon is not None, "minLon not set"
        assert minLat is not None, "minLat not set"
        assert maxLon is not None, "maxLon not set"
        assert maxLat is not None, "maxLat not set"
        if needTime:
            minTime = json_data["minTime"]
            maxTime = json_data["maxTime"]
            assert minTime is not None, "minTime not set"
            assert maxTime is not None, "maxTime not set"
            timeCellSize = json_data["timeCellSize"]  # 时间上的size
        gridCellSize = json_data["gridCellSize"]
        dataName = json_data["dataName"]
        k = json_data["stme_k"]
        kt = json_data["stme_kt"]
        min_pts = json_data["stme_min_pts"]
        use_grid = json_data["isBasedGrid"] # 代表是否采用均匀网格
        has_label = json_data["hasLabel"]
    # 删除上一次的结果
    dir_delete("../data/")
    hulls = None
    if not use_grid:
        dir_delete('./spatio_splits')
        hulls = splitSpatioBasedDensity()

    runAndGetTrainData(PreTrainData(dataName, minLon, minLat, maxLon, maxLat, minTime, maxTime,
                                    gridCellSize, timeCellSize, hulls, use_grid, has_label, needTime))

    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))