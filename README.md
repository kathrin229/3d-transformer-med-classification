# 3d-transformer-med-classification

This repository contains implementations for the thesis "Covid-19 Diagnosis In 3d Chest Ct
Scans With Attention-Based Models". The goal of this thesis is to investigate 3d image classification
in the medical context with attention based models, specifically, Covid-19 classification on 3d lung CT scans.
The models DenseNet and TimeSformer are used. Within this research, new attention schemes for TimeSformer are proposed.

## Table of Contents
* [Information about the project](#general-information)
* [Setup](#setup)
* [Models](#models)
* [Acknowledgements](#acknowledgements)


## Information about the project
<!-- abstract here -->

## Setup
Installation instructions for packages can be found in readme.txt.

<!-- `write-your-code-here` -->

## Models and Data
Finetuned models can be found here: https://drive.google.com/drive/folders/1Wf8sRcY-h5JbQsLkou1s--QE-uAMSr-g?usp=sharing

The pre-processed dataset is available here: https://drive.google.com/drive/folders/1xHaQyZCGjSYgd11fktD22Ue2WhFPPvwd?usp=sharing

<!-- more explanation -->
<!-- ![Example screenshot](./img/screenshot.png) -->

## Acknowledgements
- Dataset: https://github.com/wang-shihao/HKBU_HPML_COVID-19
- TimeSformer: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py
- DenseNet: https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8
- Attention Visualization: https://github.com/yiyixuxu/TimeSformer-rolled-attention