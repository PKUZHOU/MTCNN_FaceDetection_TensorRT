# MTCNN_TensorRT

**MTCNN Face detection algorithm's C++ implementation with NVIDIA TensorRT Inference acceleration SDK.**

This repository is based on https://github.com/AlphaQi/MTCNN-light.git

## Notations

2018/10/2: Good news! Now you can run the whole MTCNN using TenorRT 3.0 or 4.0! 

I adopt the original models from offical project https://github.com/kpzhang93/MTCNN_face_detection_alignment and do the following modifications:
  Considering TensorRT don't support PRelu layer, which is widely used in MTCNN, one solution is to add Plugin Layer (costome layer) but experiments show that this method breaks the CBR process in TensorRT and is very slow. I use Relu layer, Scale layer and ElementWise addition Layer to replace Prelu (as illustrated below), which only adds a bit of computation and won't affect CBR process, the weights of scale layers derive from original Prelu layers. 
  
  ![modification](https://github.com/PKUZHOU/MTCNN_TensorRT/blob/master/pictures/modification.png)


## Required environments
1) CUDA 9.0
1) TensorRT 3.04 or TensorRT 4.16 (I only test these two versions)
1) Cmake >=3.5
1) A digital camera to run camera test.

## Build
1) Replace the tensorrt and cuda path in CMakeLists.txt
1) Configure the detection parameters in mtcnn.cpp (min face size, the nms thresholds , etc)
1) Choose the running modes (camera test or single image test)
1) cmake .
1) make -j
1) ./main

## Results
The results will be like this in single image test mode:

![single](https://github.com/PKUZHOU/MTCNN_TensorRT/blob/master/pictures/result.jpg)

## Speed
On my computer with nvidia-gt730 grapic card (it is very very poor) and intel i5 6500 cpu, when the min face-size is set to 60 pixels, the above image costs 20 to 30ms.

## TODO
Take other techniques (such as pipline and multithread) to speed up.
