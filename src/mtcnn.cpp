#include "mtcnn.h"
#define LOG
Onet::Onet(){
    Othreshold = 0.8;
    this->rgb = new pBox;

    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;
    this->pooling1_out = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;
    this->pooling2_out = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;
    this->pooling3_out = new pBox;

    this->conv4_matrix = new pBox;
    this->conv4_out = new pBox;

    this->fc5_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;
    this->keyPoint_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
    this->conv4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;
    this->fc5_wb = new Weight;
    this->prelu_gmma5 = new pRelu;
    this->score_wb = new Weight;
    this->location_wb = new Weight;
    this->keyPoint_wb = new Weight;

    // //                             w        sc  lc ks s  p
    long conv1 = initConvAndFc(this->conv1_wb, 32, 3, 3, 1, 0);
    initpRelu(this->prelu_gmma1, 32);
    long conv2 = initConvAndFc(this->conv2_wb, 64, 32, 3, 1, 0);
    initpRelu(this->prelu_gmma2, 64);
    long conv3 = initConvAndFc(this->conv3_wb, 64, 64, 3, 1, 0);
    initpRelu(this->prelu_gmma3, 64);
    long conv4 = initConvAndFc(this->conv4_wb, 128, 64, 2, 1, 0);
    initpRelu(this->prelu_gmma4, 128);
    long fc5 = initConvAndFc(this->fc5_wb, 256, 1152, 1, 1, 0);
    initpRelu(this->prelu_gmma5, 256);
    long score = initConvAndFc(this->score_wb, 2, 256, 1, 1, 0);
    long location = initConvAndFc(this->location_wb, 4, 256, 1, 1, 0);
    long keyPoint = initConvAndFc(this->keyPoint_wb, 10, 256, 1, 1, 0);
    long dataNumber[21] = {conv1,32,32, conv2,64,64, conv3,64,64, conv4,128,128, fc5,256,256, score,2, location,4, keyPoint,10};
    mydataFmt *pointTeam[21] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->fc5_wb->pdata, this->fc5_wb->pbias, this->prelu_gmma5->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias, \
                                this->keyPoint_wb->pdata, this->keyPoint_wb->pbias \
                                };
    string filename = "Onet.txt";
    readData(filename, dataNumber, pointTeam);

    //Init the network
    OnetImage2MatrixInit(rgb);

    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    maxPoolingInit(this->conv1_out, this->pooling1_out, 3, 2);

    feature2MatrixInit(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    maxPoolingInit(this->conv2_out, this->pooling2_out, 3, 2);

    feature2MatrixInit(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    maxPoolingInit(this->conv3_out, this->pooling3_out, 2, 2);

    feature2MatrixInit(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolutionInit(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);

    fullconnectInit(this->fc5_wb, this->fc5_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
    fullconnectInit(this->keyPoint_wb, this->keyPoint_);
}
Onet::~Onet(){
    freepBox(this->rgb);
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
    freepBox(this->pooling1_out);
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
    freepBox(this->pooling2_out);
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
    freepBox(this->pooling3_out);
    freepBox(this->conv4_matrix);
    freepBox(this->conv4_out);
    freepBox(this->fc5_out);
    freepBox(this->score_);
    freepBox(this->location_);
    freepBox(this->keyPoint_);

    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);
    freeWeight(this->conv4_wb);
    freepRelu(this->prelu_gmma4);
    freeWeight(this->fc5_wb);
    freepRelu(this->prelu_gmma5);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
    freeWeight(this->keyPoint_wb);
}
void Onet::OnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 48;
    pbox->width = 48;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Onet::run(Mat &image){
    image2Matrix(image, this->rgb);

    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);

    //Pooling layer
    maxPooling(this->conv1_out, this->pooling1_out, 3, 2);

    feature2Matrix(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    maxPooling(this->conv2_out, this->pooling2_out, 3, 2);

    //conv3 
    feature2Matrix(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
    maxPooling(this->conv3_out, this->pooling3_out, 2, 2);

    //conv4
    feature2Matrix(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolution(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);
    prelu(this->conv4_out, this->conv4_wb->pbias, this->prelu_gmma4->pdata);

    fullconnect(this->fc5_wb, this->conv4_out, this->fc5_out);
    prelu(this->fc5_out, this->fc5_wb->pbias, this->prelu_gmma5->pdata);

    //conv6_1   score
    fullconnect(this->score_wb, this->fc5_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);

    //conv6_2   location
    fullconnect(this->location_wb, this->fc5_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);

    //conv6_2   location
    fullconnect(this->keyPoint_wb, this->fc5_out, this->keyPoint_);
    addbias(this->keyPoint_, this->keyPoint_wb->pbias);
    // pBoxShow(keyPoint_);
}


mtcnn::mtcnn(int row, int col){
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.2;
    nms_threshold[2] = 0.7;

    float minl = row<col?row:col;
    int MIN_DET_SIZE = 12;
    int minsize = 20;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;

    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    float minside = row<col ? row : col;
    int count = 0;
    for (vector<float>::iterator it = scales_.begin(); it != scales_.end(); it++){
        if (*it > 1){
            cout << "the minsize is too small" << endl;
            while (1);
        }
        if (*it < (MIN_DET_SIZE / minside)){
            scales_.resize(count);
            break;
        }
        count++;
    }

    cout<<"start generation TenosrRT runtime model"<<endl;

    pnet_engine = new Pnet_engine[scales_.size()];
    simpleFace_ = (Pnet**)malloc(sizeof(Pnet*)*scales_.size());
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(row*scales_.at(i));
        int changedW = (int)ceil(col*scales_.at(i));
        pnet_engine[i].init(changedH,changedW);
        simpleFace_[i] =  new Pnet(changedH,changedW,pnet_engine[i]);
    }
    rnet_engine = new Rnet_engine();
    rnet_engine->init(22,22);
    refineNet = new Rnet(*rnet_engine);

    cout<<"end generation TensorRT runtime model"<<endl;
}

mtcnn::~mtcnn(){
    //delete []simpleFace_;
}

void mtcnn::findFace(Mat &image){
    struct orderScore order;
    int count = 0;

    clock_t first_time = clock();
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        clock_t resize_time = clock();
        resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
#ifdef LOG
        cout<<"resize time "<<(clock()-resize_time)/1000.<<endl;
#endif
        (*simpleFace_[i]).run(reImage, scales_.at(i),pnet_engine[i]);

        clock_t nms_time = clock();
        nms((*simpleFace_[i]).boundingBox_, (*simpleFace_[i]).bboxScore_, (*simpleFace_[i]).nms_threshold);

        for(vector<struct Bbox>::iterator it=(*simpleFace_[i]).boundingBox_.begin(); it!= (*simpleFace_[i]).boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
#ifdef LOG
        cout<<"nms time "<<(clock()-nms_time)/1000.<<endl;
#endif
        (*simpleFace_[i]).bboxScore_.clear();
        (*simpleFace_[i]).boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols,true);

    first_time = clock() - first_time;
#ifdef LOG
    cout<<"first time is  "<<1000*(double)first_time/CLOCKS_PER_SEC<<endl;
#endif
    //second stage
    count = 0;
    clock_t second_time = clock();
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat secImage;
            resize(image(temp), secImage, Size(22, 22), 0, 0, cv::INTER_LINEAR);
            refineNet->run(secImage,*rnet_engine);
            if(*(refineNet->score_->pdata+1)>refineNet->Rthreshold){
                memcpy(it->regreCoord, refineNet->location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet->score_->pdata+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols,true);
    second_time = clock() - second_time;
#ifdef LOG
    cout<<"second time is  "<<1000*(double)second_time/CLOCKS_PER_SEC<<endl;
#endif
//    third stage
//    count = 0;
//    clock_t third_time = clock();
//    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
//        if((*it).exist){
//            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
//            Mat thirdImage;
//            resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
//            outNet.run(thirdImage);
//            mydataFmt *pp=NULL;
//            if(*(outNet.score_->pdata+1)>outNet.Othreshold){
//                memcpy(it->regreCoord, outNet.location_->pdata, 4*sizeof(mydataFmt));
//                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
//                it->score = *(outNet.score_->pdata+1);
//                pp = outNet.keyPoint_->pdata;
//                for(int num=0;num<5;num++){
//                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
//                }
//                for(int num=0;num<5;num++){
//                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
//                }
//                thirdBbox_.push_back(*it);
//                order.score = it->score;
//                order.oriOrder = count++;
//                thirdBboxScore_.push_back(order);
//            }
//            else{
//                it->exist=false;
//            }
//        }
//    }
//
//    if(count<1)return;
//    refineAndSquareBbox(thirdBbox_, image.rows, image.cols,false);
//    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
//
//    third_time = clock() - third_time;
//    cout<<"third time is  "<<1000*(double)third_time/CLOCKS_PER_SEC<<endl;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
}
