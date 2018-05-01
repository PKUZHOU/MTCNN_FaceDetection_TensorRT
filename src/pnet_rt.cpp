//
// Created by zhou on 18-4-30.
//
#include "pnet_rt.h"

// stuff we know about the network and the caffe input/output blobs
static Logger gLogger;


Pnet_engine::Pnet_engine() : prototxt("/home/zhou/12net.prototxt"),
                             model("/home/zhou/12net.caffemodel"),
                             INPUT_BLOB_NAME("data"),
                             OUTPUT_LOCATION_NAME("conv4-2"),
                             OUTPUT_PROB_NAME("prob1") {
    caffeToGIEModel(prototxt, model, std::vector<std::string>{OUTPUT_PROB_NAME, OUTPUT_LOCATION_NAME}, 1,
                    gieModelStream);
};

void Pnet_engine::caffeToGIEModel(const std::string &deployFile,                // name for caffe prototxt
                                  const std::string &modelFile,                // name for model
                                  const std::vector<std::string> &outputs,   // network outputs
                                  unsigned int maxBatchSize,                    // batch size - NB must be at least as large as the batch we want to run with)
                                  IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    std::cout << "Beging parsing Pnet model..." << std::endl;
    const IBlobNameToTensor *blobNameToTensor = parser->parse(prototxt.c_str(),
                                                              model.c_str(),
                                                              *network,
                                                              nvinfer1::DataType::kFLOAT);

    std::cout << "End parsing Pnet model" << endl;

    // specify which tensors are outputs
    for (auto &s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(5 << 20);

    std::cout << "Begin building pnet engine..." << endl;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout << "End building pnet engine" << endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    runtime = createInferRuntime(gLogger);
    context = engine->createExecutionContext();
    builder->destroy();
    shutdownProtobufLibrary();

}


Pnet::Pnet() : BatchSize(1),
               INPUT_C(3) {

    Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;

    this->score_ = new pBox;
    this->location_ = new pBox;


}

Pnet::~Pnet() {

    delete (score_);
    delete (location_);

}


void Pnet::run(Mat &image, float scale, Pnet_engine &pnet_engine) {


//
//    if (firstFlag)

    INPUT_W = image.cols;
    INPUT_H = image.rows;

    //calculate output shape
    this->score_->width = int(ceil((INPUT_W - 2) / 2.) - 4);
    this->score_->height = int(ceil((INPUT_H - 2) / 2.) - 4);
    this->score_->channel = 2;

    this->location_->width = int(ceil((INPUT_W - 2) / 2.) - 4);
    this->location_->height = int(ceil((INPUT_H - 2) / 2.) - 4);
    this->location_->channel = 4;

    OUT_PROB_SIZE = this->score_->width * this->score_->height * this->score_->channel;
    OUT_LOCATION_SIZE = this->location_->width * this->location_->height * this->score_->channel;
    //allocate memory for outputs
    this->score_->pdata = (float *) malloc(OUT_PROB_SIZE * sizeof(float));
    this->location_->pdata = (float *) malloc(OUT_LOCATION_SIZE * sizeof(float));


    const ICudaEngine &Engine = pnet_engine.context->getEngine();
    assert(Engine.getNbBindings() == 3);
    void *buffers[3];

    int     inputIndex = Engine.getBindingIndex(pnet_engine.INPUT_BLOB_NAME),
            outputProb = Engine.getBindingIndex(pnet_engine.OUTPUT_PROB_NAME),
            outputLocation = Engine.getBindingIndex(pnet_engine.OUTPUT_LOCATION_NAME);

    //creat GPU buffers and stream
    CHECK(cudaMalloc(&buffers[inputIndex], BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));

    CHECK(cudaMalloc(&buffers[outputProb], BatchSize * OUT_PROB_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputLocation], BatchSize * OUT_LOCATION_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    //DMA the input to the GPU ,execute the batch asynchronously and DMA it back;

    CHECK(cudaMemcpyAsync(buffers[inputIndex], image.data, BatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    pnet_engine.context->enqueue(BatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(this->score_->pdata, buffers[outputProb], BatchSize * OUT_PROB_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(this->location_->pdata, buffers[outputLocation],
                          BatchSize * OUT_LOCATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));



    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputProb]));
    CHECK(cudaFree(buffers[outputLocation]));
    generateBbox(this->score_, this->location_, scale);

}

void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale) {
    //for pooling
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = score->pdata + score->width * score->height;
    mydataFmt *plocal = location->pdata;
    struct Bbox bbox;
    struct orderScore order;
    for (int row = 0; row < score->height; row++) {
        for (int col = 0; col < score->width; col++) {
            if (*p > Pthreshold) {
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride * row + 1) / scale);
                bbox.y1 = round((stride * col + 1) / scale);
                bbox.x2 = round((stride * row + 1 + cellsize) / scale);
                bbox.y2 = round((stride * col + 1 + cellsize) / scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                for (int channel = 0; channel < 4; channel++)
                    bbox.regreCoord[channel] = *(plocal + channel * location->width * location->height);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }

}