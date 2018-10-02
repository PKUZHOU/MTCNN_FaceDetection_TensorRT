//
// Created by zhou on 18-5-4.
//

#include "baseEngine.h"
baseEngine::baseEngine(const char * prototxt,const char* model,const  char* input_name,const char*location_name,
                       const char* prob_name, const char *point_name) :
                             prototxt(prototxt),
                             model(model),
                             INPUT_BLOB_NAME(input_name),
                             OUTPUT_LOCATION_NAME(location_name),
                             OUTPUT_PROB_NAME(prob_name),
                             OUTPUT_POINT_NAME(point_name)
{
};
baseEngine::~baseEngine() {
    shutdownProtobufLibrary();
}

void baseEngine::init(int row,int col) {

}
void baseEngine::caffeToGIEModel(const std::string &deployFile,                // name for caffe prototxt
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

    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              nvinfer1::DataType::kFLOAT);
    // specify which tensors are outputs
    for (auto &s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 25);
    ICudaEngine*engine = builder->buildCudaEngine(*network);
    assert(engine);
    context = engine->createExecutionContext();

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    builder->destroy();

}