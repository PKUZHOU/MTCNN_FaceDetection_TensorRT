//
// Created by zhou on 18-4-30.
//

#ifndef MAIN_PNET_RT_H
#define MAIN_PNET_RT_H

#include "network.h"
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <string>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#endif //MAIN_PNET_RT_H
using namespace nvinfer1;
using namespace nvcaffeparser1;
class Pnet
{
public:
    Pnet();
    ~Pnet();

    void run(Mat &image, float scale);
    void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
                          const std::string& modelFile,				// name for model
                          const std::vector<std::string>& outputs,   // network outputs
                          unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
                          IHostMemory *&gieModelStream);    // output buffer for the GIE model
    float nms_threshold;
    mydataFmt Pthreshold;
    bool firstFlag;
    vector<struct Bbox> boundingBox_;
    vector<orderScore> bboxScore_;
private:

    IHostMemory *gieModelStream{nullptr};
    const string prototxt;
    const string model   ;
    const char *INPUT_BLOB_NAME;
    const char *OUTPUT_PROB_NAME;
    const char *OUTPUT_LOCATION_NAME;

    const int BatchSize ;
    const int INPUT_C ;

    //must be computed at runtime
    int INPUT_H ;
    int INPUT_W ;
    int OUT_PROB_SIZE;
    int OUT_LOCATION_SIZE;

    IRuntime* runtime;
    ICudaEngine *engine ;
    IExecutionContext *context;

    struct pBox *location_;
    struct pBox *score_;


    void generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale);
};