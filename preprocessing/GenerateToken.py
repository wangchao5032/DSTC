import pickle
from utils.SpatialRegionTools import SpatialRegion, createTrainVal, createTrainVal_OnlyOriginal, \
    saveKNearestVocabs, makeVocab
from utils.utils import downsamplingDistort

class PreprocParams:
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
        self.h5_file = f'originData/{dataName}.h5'

def preprocess(PreprocParams):
    # 研究区域和实验配置类
    region = SpatialRegion(PreprocParams.dataName,
                           PreprocParams.minlon, PreprocParams.minlat, # moni
                           PreprocParams.maxlon, PreprocParams.maxlat, # moni
                           PreprocParams.mintime, PreprocParams.maxtime,  # 时间范围,一天最大86400(以0点为相对值)
                           PreprocParams.gridCellSize, PreprocParams.gridCellSize,
                           PreprocParams.timeCellSize,  # 时间步
                           1,  # minfreq 最小击中次数
                           40_0000,  # maxvocab_size
                           15,  # k nearest neighbor
                           4,  # vocab_start 词汇表基准值
                           PreprocParams.needTime,
                           2, 4000,
                           PreprocParams.hulls, PreprocParams.use_grid, PreprocParams.has_label)

    ## 2, 处理数据
    print("Creating paramter file")
    train_datas, train_labels, val_datas, val_labels, test_datas, test_labels = makeVocab(region, PreprocParams.h5_file)
    
    createTrainVal(region, train_datas, False, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    createTrainVal(region, val_datas, True, downsamplingDistort)  # 生成train.src,train.trg,val.src,val.trg
    # 生成原轨迹的训练集文件和验证集文件，不生成带噪声的训练集文件和验证集文件
    createTrainVal_OnlyOriginal(region, train_datas, train_labels, False, False)
    createTrainVal_OnlyOriginal(region, val_datas, val_labels, True, False)
    createTrainVal_OnlyOriginal(region, test_datas, test_labels, False, True)
    print("createTrainVal() finished!")
    saveKNearestVocabs(region)
    
    # 存下区域信息
    output_hal = open("../data/region.pkl", 'wb')
    region_save = pickle.dumps(region)
    output_hal.write(region_save)
    output_hal.close()
    print("Vocabulary size %d with dist_cell size %d (meters) and time_cell size %d" % (region.vocab_size, PreprocParams.gridCellSize, PreprocParams.timeCellSize))