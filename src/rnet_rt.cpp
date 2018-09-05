
//Created by zhou on 18-5-4.

#include "rnet_rt.h"


Rnet_engine::Rnet_engine() : baseEngine("22net.prototxt",
                                        "22net.caffemodel",
                                        "data",
                                        "output",
                                        "output") {
};

Rnet_engine::~Rnet_engine() {
    shutdownProtobufLibrary();
}

void Rnet_engine::init(int row, int col) {

    IHostMemory *gieModelStream{nullptr};
    int max_batch_size = 1;
    //generate Tensorrt model
    caffeToGIEModel(prototxt, model, std::vector<std::string>{OUTPUT_PROB_NAME, OUTPUT_LOCATION_NAME}, 1,
                    gieModelStream);

}


Rnet::Rnet(const Rnet_engine &rnet_engine) : BatchSize(1),
                                             INPUT_C(3),
                                             Engine(rnet_engine.context->getEngine()) {

    Rthreshold = 0.9;
    nms_threshold = 0.5;
    this->loc_score_ = new pBox;
    this->score_ = new pBox;
    this->location_ = new pBox;
    this->rgb = new pBox;
    INPUT_W = 22;
    INPUT_H = 22;
    //calculate output shape
    this->score_->width = 1;
    this->score_->height = 1;
    this->score_->channel = 2;

    this->location_->width = 1;
    this->location_->height = 1;
    this->location_->channel = 4;

    OUT_SIZE = this->score_->width * this->score_->height * (this->score_->channel+this->location_->channel);
    //allocate memory for outputs
    this->rgb->pdata = (float *) malloc(INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    this->score_->pdata = (float *) malloc(2 * sizeof(float));
    this->location_->pdata = (float *) malloc(4 * sizeof(float));
    this->loc_score_->pdata = (float *)malloc(6 * sizeof(float));

    assert(Engine.getNbBindings() == 2);
    inputIndex = Engine.getBindingIndex(rnet_engine.INPUT_BLOB_NAME);
    output = Engine.getBindingIndex(rnet_engine.OUTPUT_PROB_NAME);
    //creat GPU buffers and stream
    CHECK(cudaMalloc(&buffers[inputIndex], BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[output], BatchSize * OUT_SIZE * sizeof(float)));
    CHECK(cudaStreamCreate(&stream));
}

Rnet::~Rnet()  {

    delete (score_);
    delete (location_);
    delete (loc_score_);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[output]));
}

void Rnet::run(Mat &image,  const Rnet_engine &rnet_engine) {


    //DMA the input to the GPU ,execute the batch asynchronously and DMA it back;

    image2Matrix(image, this->rgb);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], this->rgb->pdata,
                          BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    rnet_engine.context->enqueue(BatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(this->loc_score_->pdata, buffers[output], BatchSize * OUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    memcpy(this->score_->pdata,this->loc_score_->pdata,2*sizeof(float));
    memcpy(this->location_->pdata,this->loc_score_->pdata+2,4*sizeof(float));

}
