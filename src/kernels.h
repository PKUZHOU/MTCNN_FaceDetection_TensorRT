#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#include<cuda.h>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
// #include<opencv2/gpu/gpu.hpp> 
#include<opencv2/core/cuda_devptrs.hpp> 
// #include<opencv2/gpu/stream_accessor.hpp> 
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "device_launch_parameters.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
using namespace cv;

void gpu_image2Matrix(int width, int height , cuda::GpuMat & image, float* matrix, cudaStream_t& stream);
void gpu_generatebox(void * score, void * location, float scale);
#endif //KERNEL_H_INCLUDED