## Mean-Scale Hyperprior Image Compression Model (w/o Context model) 
original paper: http://arxiv.org/abs/1809.02736
<br>

## Introduction

**This repository is the implementation of training and evaluation code for the model above. It was written in advance to expand this repository through further research.**
<br>

I wrote this with reference to the following two codes.
- https://github.com/liujiaheng/compression 
- https://github.com/InterDigitalInc/CompressAI
<br>

## Dataset
(In root, create dataset directory structure below.)
```
├─data
│  ├─train
│  ├─test
│  └─val
└ ...
```

I used Flicker2W, DIV2K and CLIC2020's for training.
- Flicker2W dataset can be found on [liujiaheng's repository](https://github.com/liujiaheng/compression)
- 'Train data (HR images)' in [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- 'Training Dataset P' & 'Training Dataset M' in [CLIC2020](http://challenge.compression.cc/tasks/)
Data pre-processing for removing JPEG compression artifacts is performed in the training stage Automatically.
<br>

For evaluation, i used [Kodak24](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html) dataset.
<br>

For validation, you can use any dataset and it is not necessary. (It's not bad comment validation codes)
<br>


## Training
You can train the model with command `CUDA_VISIBLE_DEVICES={gpu num} python train.py` at the root directory, so that train.py creates `Solver` class and call the method `train`.
Before that, you have to modify the `config.py` to suit your purpose.

## Evaluation
