# MGScoliosis
This repo is the official implementation of paper "MGScoliosis: Multi-Grained Scoliosis Detection with Joint Ordinal Regression from Natural Image" accepted by Alexandria Engineering Journal.


### Overview
<img src="figures/model(b).png"/>

### Datasets
Data preparation: The dataset should follow the ImageNet folder structure. Each image should be named like 'MT-18-35.jpg', where 18 represents the Cobb angle size.

```
│dataset/
├──train/
│  ├── 0
│  │   ├── MT-8-35.jpg
│  │   ├── L-9-36.jpg
│  │   ├── ......
│  ├── 1
│  │   ├── ......

├──val/
│  ├── 0
│  │   ├── L-6-40.jpg
│  │   ├── PT-9-41.jpg
│  │   ├── ......
│  ├── ......
```
### Requirement
```
1. Pytorch >= 1.7
2. timm == 0.4.12
```

### Train
Run 'train.sh' to train the model.
```
MODEL=MGScoliosis 
DROP_PATH=0.2 
CUDA_VISIBLE_DEVICES=0 bash distributed_train.sh 1 /path/to/dataset \
	  --model $MODEL -b 56 --lr 3e-4 --drop-path $DROP_PATH
```

### Validate
Run 'eval.sh' to validate.
```
MODEL=MGScoliosis 
python3 validate.py /path/to/dataset --model $MODEL \
  --checkpoint /path/to/model -b 64
```

### Citation
```
@article{ZHU2025329,
title = {MGScoliosis: Multi-grained scoliosis detection with joint ordinal regression from natural image},
journal = {Alexandria Engineering Journal},
volume = {111},
pages = {329-340},
year = {2025},
issn = {1110-0168},
doi = {https://doi.org/10.1016/j.aej.2024.10.066},
url = {https://www.sciencedirect.com/science/article/pii/S1110016824012225},
author = {Xiaojia Zhu and Rui Chen and Zhiwen Shao and Ming Zhang and Yuhu Dai and Wenzhi Zhang and Chuandong Lang}
}
```
