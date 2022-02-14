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

## Model
The model is Mean-scale hyperprior image compression model using a GMM(Gaussian Mixture Model) for entropy model instead of GSM(Gaussian Scale Mixture model) in J. Balle's paper.

The model has 8 quality hyperparameter lambda, controling the trade-off between distortion and bits.

I used `lambda = [64, 128, 256, 512, 1024, 2048, 4096, 8192]` for 8 different model.

4 low quality models use the convolution layers with the number of channnels N=192, M=192 and for 4 high quality models, N=192, M=320
<br>

## Dataset
(Create dataset directory structure below.)
```
├─data
│  ├─train
│  ├─test
│  └─val
└ ...
```

I used Flicker2W, DIV2K and CLIC2020 for training. (Flicker2W dataset is sufficient to train the model)
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
<br>

For training 8 different model, firstly train the highest quality(8) model and perform fine-tuning to other models.

Total training steps (batches): 1400K (until [1100K, 1300K, 1350K, 1400K], training with a learning rate [1e-4, 5e-5, 1e-5, 5e-6, 1e-6])

For fine-tuning, i used the highest quality model's pre-trained weigths until 900K. 


## Evaluation
You can test the model with command `CUDA_VISIBLE_DEVICES={gpu num} python test.py` at the root directory, so that test.py creates `Solver` class and call the method `test`.

Before that, you have to modify the `config.py` to suit your purpose.
<br>
