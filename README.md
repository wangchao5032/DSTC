# A Deep Spatiotemporal Trajectory Representation Learning Framework for Clustering

## Install
Run ``pip install -e .``

## Preprocess 
In the preprocess step, we divide the spatialtemporal space into tokens. Two division methods are reported: grid-based division and density-based division.

1. Put your data in `preprocessing/originData`. A sample data from [CVRR](http://cvrr-nas.ucsd.edu/bmorris/datasets/dataset_trajectory_clustering.html) is placed in this folder.

2. Modify the preprocess config file in `preprocessing/conf/preprocess_conf.json`.  
Parameters are list here:

```
"dataName": data file in h5 format,
"isBasedGrid": 是否基于⽹格划分， false为基于密度划分,
# parameters of grid-based division
"gridCellSize": grid size of space dimension (only used in grid-based method) 
"minLon": 最⼩经度,
"minLat": 最⼩纬度,
"maxLon": 最⼤经度,
"maxLat": 最⼤纬度,

"needTime": 是否需要考虑时间， false代表空间轨迹分析
"timeStart": 起始时间,设为0即可
"timeCellSize": grid size of time dimension (only used in grid-based method)
"minTime": 最⼩时间,
"maxTime": 最⼤时间,

"stme_k": stme模型参数k,
"stme_kt": stme模型参数kt,
"stme_min_pts": stme模型参数pts
"hasLabel": 数据集是否存在标签,
```

3. Run the script

```
cd preprrcessing
python preprocess.py
```
