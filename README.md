# A Deep Spatiotemporal Trajectory Representation Learning Framework for Clustering
## Install
Run `pip install -r requirements.txt`

## Preprocess 
In the preprocess step, we divide the spatialtemporal space into tokens. Two division methods are provided: grid-based division and density-based division.

1. Put your data in `preprocessing/originData`. A sample data `cross.h5` from [CVRR](http://cvrr-nas.ucsd.edu/bmorris/datasets/dataset_trajectory_clustering.html) is placed in this folder.

2. Modify the preprocess config file in `preprocessing/conf/preprocess_conf.json`.  
Parameters are list here:

```
"dataName": data file in h5 format
"method": "density" or "grid"

# parameters for grid-based division
"gridCellSize": grid size in spatial dimension
"timeCellSize": grid size in temporal dimension, set to -1 to ignore the time dimension
"minLon": minimum longitude
"minLat": minimum latitude
"maxLon": maximum longitude
"maxLat": maximum latitude
"minTime": minimum time
"maxTime": maximum time

# parameters for density-based division
"stme_k": parameter k of the density-based division algorithm STME
"stme_kt": parameter kt of the density-based division algorithm STME
"stme_min_pts": parameter min_pts of the density-based division algorithm STME

"hasLabel": if data has ground truth label
```

3. Run the preprocess script

```
cd preprocessing
python preprocess.py
```
After preprocess, token sequences are generated in `train/val/test.src` and `train/val/test.tar` files, which used for training, validation and testing. The region information and the vocabulary representing the division results are saved in `region.pkl` and `*-knearestvocabs.h5` respectively.

## Training 
The training includes two modes: pretraining and joint-training.

1. Modify config file in `training/conf/train_conf.json`.
```
"expId": experiment ID
"dataName": dataset name
"mode": "pretraining" or "joint-training"
"vocabSize": vocabulary size, output by the preprocess step
"clusterNum": number of clusters you want to get
"sourceData": dataset path
"epochs": epochs
"embeddingSize": length of embedding vector
"hiddenSize": size of hidden layers
"batch": batch size
"t2vecBatch": pre-train batch
"learningRate": learning rate
"dropout": dropout parameter
"m2LearningRate": joint-learning learning rate
"distDecaySpeed": penalty parameter for far distance cell
"alpha": reconstruction loss
"beta": soft cluster assignment loss
"gamma": inter-cluster distance loss
"delta": neighbor loss
"kmeans": k-means loss
"hasLabel": if data have ground truth label
"needSave": if save results
"saveFreq": save frequency
```
2. Run the training script
```
cd training
python dstc.py
```

3. Check results. Run 
```
cd showResult
python predict.py
```
The t-sne results are shown in `training/showResult/cluster_png/`.


# Acknowledgements
Thank to the authors of t2vec. We use code and are inspired by their work (https://github.com/boathit/t2vec).

# BibTex
```
@ARTICLE{10403544,
  author={Wang, Chao and Huang, Jiahui and Wang, Yongheng and Lin, Zhengxuan and Jin, Xiongnan and Jin, Xing and Weng, Di and Wu, Yingcai},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={A Deep Spatiotemporal Trajectory Representation Learning Framework for Clustering}, 
  year={2024},
  volume={25},
  number={7},
  pages={7687-7700},
  doi={10.1109/TITS.2024.3350339}}
```