# MTCNN_TensorRT
MTCNN C++ implementation with NVIDIA TensorRT Inference accelerator SDK

This repository is based on https://github.com/AlphaQi/MTCNN-light.git

Now it is under building, I will implement MTCNN face-detection algorithum with NVIDIA TensorRT C++ API to get high performance on platforms such as NVIDIA TX2 or any PC with NVIDIA-GPUs. 


6/14：
This project can run Pnet and Rnet, the speed is very fast. Just use cmake to build this project

9/5:
I fix some bugs, but still no inplemention of Onet.
If you want to run this project, you need

1.CUDA 9.0

2.TensorRT 3.04

3.cmake >=3.5

4.a digital camera to run camera test.

replace the tensorrt and cuda path in CMakeLists.txt, then run

cmake .

make -j

./main

the results will be

![camera](https://github.com/PKUZHOU/MTCNN_TensorRT/blob/master/pictures/camera.png)

on my computer with nvidia-gt7300 and intel i5 6500, when the min face-size is set to 20 pixels, the inference time is about 24ms perframe.
