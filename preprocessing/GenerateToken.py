import os
import pickle, json
from utils.SpatialRegionTools import SpatialRegion, createTrainVal, createTrainVal_OnlyOriginal, \
    saveKNearestVocabs, makeVocab
from utils.utils import downsamplingDistort
import datetime, time
def dir_delete(dir_name):
    files = os.listdir(dir_name)
    # 遍历删除指定目录下的文件
    for file in files:
        os.remove(os.path.join(dir_name, file))
        print(file, "删除成功")
    print(dir_name, "删除成功")

class PreTrainData:
    def __init__(self, dataName, minlon, minlat, maxlon, maxlat, mintime, maxtime,
                  gridCellSize, timeCellSize, hulls, use_grid, has_label, needTime):
        self.dataName = dataName
        self.minlon = minlon
        self.minlat = minlat
        self.maxlon = maxlon
        self.maxlat = maxlat
        self.mintime = mintime
        self.maxtime = maxtime
        self.gridCellSize = gridCellSize
        self.timeCellSize = timeCellSize
        self.hulls = hulls
        self.use_grid = use_grid
        self.has_label = has_label
        self.needTime = needTime

def runAndGetTrainData(preTrainData):
    # 读取该目录所有文件 也就是需要解析的所有文件
    h5_files = os.listdir("./originData")
    # 完善文件路径
    for i in range(len(h5_files)):
        h5_files[i] = os.path.join("./originData", h5_files[i])

    print("是否考虑时间: " + str(preTrainData.needTime))

    # 研究区域和实验配置类
    region = SpatialRegion(preTrainData.dataName,
                           preTrainData.minlon, preTrainData.minlat, # moni
                           preTrainData.maxlon, preTrainData.maxlat, # moni
                           preTrainData.mintime, preTrainData.maxtime,  # 时间范围,一天最大86400(以0点为相对值)
                           preTrainData.gridCellSize, preTrainData.gridCellSize,
                           preTrainData.timeCellSize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           15,  # k nearest neighbor
                           4,  # vocab_start 词汇表基准值
                           preTrainData.needTime,
                           2, 4000,
                           preTrainData.hulls, preTrainData.use_grid, preTrainData.has_label)

    ## 2, 处理数据
    print("Creating paramter file $paramfile")
    train_datas, train_labels, val_datas, val_labels, test_datas, test_labels = makeVocab(region, h5_files)
    
    # todo
    # makeKneighbor(region)
    createTrainVal(region, train_datas, False, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    createTrainVal(region, val_datas, True, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    # 生成原轨迹的训练集文件和验证集文件，不生成带噪声的训练集文件和验证集文件
    createTrainVal_OnlyOriginal(region, train_datas, train_labels, False, False)  # 生成train.ori，val.ori 
    createTrainVal_OnlyOriginal(region, val_datas, val_labels, True, False)  # 生成train.ori，val.ori 
    createTrainVal_OnlyOriginal(region, test_datas, test_labels, False, True)  # 生成train.ori，val.ori 
    print("createTrainVal() finished!")
    saveKNearestVocabs(region)
    
    # 存下区域信息
    output_hal = open("../data/region.pkl", 'wb')
    region_save = pickle.dumps(region)
    output_hal.write(region_save)
    output_hal.close()
    print("Vocabulary size %d with dist_cell size %d (meters) and time_cell size %d" % (region.vocab_size, preTrainData.gridCellSize, preTrainData.timeCellSize))