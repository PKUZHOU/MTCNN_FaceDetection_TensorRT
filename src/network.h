//c++ network  author : liqi
//Nangjing University of Posts and Telecommunications
//date 2017.5.21,20:27
#ifndef NETWORK_H
#define NETWORK_H
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <cstring>
#include <string>
#include <math.h>
#include "pBox.h"
#include <assert.h>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
using namespace cv;
#define CPU
//void image2Matrix(const cuda::GpuMat &image, cuda::GpuMat &matrix);
bool cmpScore(struct orderScore lsh, struct orderScore rsh);
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width,bool square);
void continous(cuda::GpuMat & mat);

#endif