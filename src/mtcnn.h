#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"
#include "pnet_rt.h"
#include "rnet_rt.h"
#include "onet_rt.h"

class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    void findFace(cuda::GpuMat &image);

    //for second stage
    const static int pnet_max_pymid_depth = 256;
    const static int rnet_max_input_num = 2048;
    const static int rnet_streams_num = rnet_max_input_num;
    //for third stage
    const static int onet_max_input_num = 512;
    const static int onet_streams_num = onet_max_input_num;
    cuda::Stream cv_streams[rnet_streams_num];
    cudaStream_t cudastreams[rnet_streams_num];
    cuda::GpuMat firstImage_buffer[pnet_max_pymid_depth];
    cuda::GpuMat secImages_buffer[rnet_streams_num];
    cuda::GpuMat thirdImages_buffer[onet_streams_num];
private:

    float nms_threshold[3];
    vector<float> scales_;

    float *boxes_data;
    void* gpu_boxes_data;

    Pnet_engine *pnet_engine;
    Pnet **simpleFace_;
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;

    Rnet *refineNet;
    Rnet_engine *rnet_engine;
    vector<struct Bbox> secondBbox_;
    vector<struct orderScore> secondBboxScore_;

    Onet *outNet;
    Onet_engine *onet_engine;
    vector<struct Bbox> thirdBbox_;
    vector<struct orderScore> thirdBboxScore_;


};

#endif