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
"method": "density" or "grid"

# parameters for grid-based division
"gridCellSize": grid size in spatial dimension
"minLon": minimum longitude,
"minLat": minimum latitude,
"maxLon": maximum longitude,
"maxLat": maximum latitude,
"ignoreTemporal": if ignore the temporal dimension and only process spatial dimension
"timeStart": start time, usually set to zero
"timeCellSize": grid size in temporal dimension 
"minTime": minimum time,
"maxTime": maximum time,

# parameters for density-based division
"stme_k": parameter k of the density-based division algorithm STME,
"stme_kt": parameter kt of the density-based division algorithm STME,
"stme_min_pts": parameter min_pts of the density-based division algorithm STME
"hasLabel": if data has ground truth label,
```

3. Run the preprocess script

```
cd preprrcessing
python preprocess.py
```
After preprocess, token sequence are generated in `train/val/test.src` and `train/val/test.tar` files, which used for training, validation and testing. The region infomation and the vocabulary representing the division results are saved in `region.pkl` and `*-knearestvocabs.h5` respectively.

## Training 
The training includes two steps: pretraining and joint-training.

1. modify config file in `training/conf/train_conf.json`.
```
"method": "pretraining" or "joint-training"
"expId": experiment ID,
"vocabSize": vocabulary size,
"clusterNum": number of clusters,
"embeddingSize": length of embedding vector,
"hiddenSize": size of hidden layers,
"batch": batch size,
...
```
2. updating ...