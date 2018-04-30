//
// Created by zhou on 18-4-30.
//
#include "pnet_rt.h"

// stuff we know about the network and the caffe input/output blobs
static Logger gLogger;

Pnet::Pnet() :prototxt(""),model(""),
              INPUT_BLOB_NAME("data"),
              OUTPUT_LOCATION_NAME("conv4-1"),
              OUTPUT_PROB_NAME(""),
              BatchSize(1),
              INPUT_C(3){
    Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;
    caffeToGIEModel(prototxt,model,std::vector<std::string>{OUTPUT_PROB_NAME,OUTPUT_LOCATION_NAME},1, gieModelStream);
}



void Pnet::caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
                     const std::string& modelFile,				// name for model
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
                     IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    std::cout<<"Beging parsing Pnet model..."<<std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(prototxt.c_str(),
        model.c_str(),
        *network,
        DataType::kFLOAT);

    std::cout<<"End parsing Pnet model"<<endl;

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(5 << 20);

    std::cout<<"Begin building pnet engine..."<<endl;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout<<"End building pnet engine"<<endl;

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    runtime = createInferRuntime(gLogger);
    context = engine->createExecutionContext();
    builder->destroy();
    shutdownProtobufLibrary();

}

void Pnet::run(Mat &image, float scale) {



    if (firstFlag)
    {
        INPUT_W = image.cols;
        INPUT_H = image.rows;
        OUT_PROB_SIZE =
        const ICudaEngine& Engine = context->getEngine();
        assert(Engine.getNbBindings()==3);
        void* buffers[3];
        int inputIndex = Engine.getBindingIndex(INPUT_BLOB_NAME),
                outputProb = Engine.getBindingIndex(OUTPUT_PROB_NAME),
                outputLocation = Engine.getBindingIndex(OUTPUT_LOCATION_NAME);

        CHECK(cudaMalloc(&buffers[inputIndex],BatchSize*INPUT_C*INPUT_H*INPUT_W* sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputProb],BatchSize*INPUT_H*INPUT_W* sizeof(float)));

    }










}






