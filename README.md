Object-Guided-Relocalisation
========================================

## Introduction

This is the open source implementation of [1]. Demo video can be found [here](https://www.youtube.com/watch?v=H3i9Q4JvX2o) 

[1] Object-Augmented RGB-D SLAM for Wide-Disparity Relocalisation. Y. Ming, X. Yang & A. Calway. IROS, 2021

## Detection Network
[NOCS](https://arxiv.org/abs/1901.02970) network is used here for object detection and pose estimation.

[TF1](https://github.com/hughw19/NOCS_CVPR2019) implementation is the official implementation and tested with CUDA 10.0 & cuDNN 7.41, Python 3.5, Tensorflow 1.14.0 and Keras 2.3.0.

[TF2](https://github.com/YuhangMing/NOCS_CVPR2019) modification is tested with CUDA 11.1.1 & cuDNN 8.0.5, Python 3.7, Tensorflow nightly 2.5.0 and Keras 2.4.3.

## RGB-D SLAM 
Install the following dependencies:
- [OpenCV](https://opencv.org/), tested on 3.4.13
- [CUDA](https://developer.nvidia.com/cuda-zone), tested on 11.1.1
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), tested on 3.3.4
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
- [Sophus](https://github.com/strasdat/Sophus)
- (Optional) [OpenNI2](https://structure.io/openni)
- (Optional) [Azure Kinnet](https://docs.microsoft.com/en-us/azure/kinect-dk/)

## 2. REPO USAGE
In "main" branch: executable data_path folder_name sequence_number display_or_not

eg.
```shell
./bin/vil_const ~/SLAMs/datasets/ BOR 5 true
```

In "dev-submap" branch: executable sequence_number

eg.
```shell
./bin/vil_const 5
```
