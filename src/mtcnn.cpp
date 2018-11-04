#include "mtcnn.h"
#include "kernels.h"
//#define LOG
mtcnn::mtcnn(int row, int col){
    //set NMS thresholds
    nms_threshold[0] = 0.6;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;
    //set minimal face size (weidth in pixels)
    int minsize = 25;
    /*config  the pyramids */
    float minl = row<col?row:col;
    int MIN_DET_SIZE = 12;
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

    cout<<"Start generating TenosrRT runtime models"<<endl;
    //generate pnet models
    pnet_engine = new Pnet_engine[scales_.size()];
    simpleFace_ = (Pnet**)malloc(sizeof(Pnet*)*scales_.size());
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(row*scales_.at(i));
        int changedW = (int)ceil(col*scales_.at(i));
        pnet_engine[i].init(changedH,changedW);
        simpleFace_[i] =  new Pnet(changedH,changedW,pnet_engine[i]);
    }

    //generate rnet model
    rnet_engine = new Rnet_engine();
    rnet_engine->init(24,24);
    refineNet = new Rnet(*rnet_engine);

    //generate onet model
    onet_engine = new Onet_engine();
    onet_engine->init(48,48);
    outNet = new Onet(*onet_engine);
    cout<<"End generating TensorRT runtime models"<<endl;
    for(int i = 0;i<rnet_streams_num;i++)
        cudastreams[i] = cuda::StreamAccessor::getStream(cv_streams[i]);
    boxes_data = (float*)malloc(sizeof(int)*4*rnet_max_input_num);
    CHECK(cudaMalloc(&gpu_boxes_data, sizeof(int)*4*rnet_max_input_num));

    cout<<"Input shape "<<row<<"*"<<col<<endl;
    cout<<"Min size "<<minsize<<endl;

}

mtcnn::~mtcnn(){
    //delete []simpleFace_;
}

void mtcnn::findFace(cuda::GpuMat &image){
    struct orderScore order;
    int count = 0;

    clock_t first_time = clock();
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        (*simpleFace_[i]).scale = scales_.at(i);
        cuda::resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR,simpleFace_[i]->cv_stream);
        (*simpleFace_[i]).run(reImage, scales_.at(i),pnet_engine[i]);
    }

    for(int i = int(scales_.size())-1; i >= 0 ; i--)
    {
        cudaStreamSynchronize(simpleFace_[i]->cuda_stream); //Synchronize
        simpleFace_[i]->cpu_generateBbox(simpleFace_[i]->score_, simpleFace_[i]->location_, simpleFace_[i]->scale);
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
        (*simpleFace_[i]).bboxScore_.clear();
        (*simpleFace_[i]).boundingBox_.clear();
    }

    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols,true);
    cout<<"Pnet time is "<<1000*double(clock()-first_time)/CLOCKS_PER_SEC<<endl;

    //second stage
    count = 0;
    clock_t second_time = clock();
    int inputed_num = 0;
    int step = 24*24*3*sizeof(float);

    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            cuda::resize(image(temp), secImages_buffer[inputed_num],
                    Size(24, 24), 0, 0, cv::INTER_LINEAR, cv_streams[inputed_num]);
            gpu_image2Matrix_with_transpose(24,24,secImages_buffer[inputed_num],
                    (float*)(refineNet->buffers[refineNet->inputIndex])+inputed_num*step,cudastreams[inputed_num]);
            inputed_num++;
        }
    }

    cout<<"Rnet input images number is "<<inputed_num<<endl;
    cudaDeviceSynchronize();
    refineNet->run(inputed_num, *rnet_engine, refineNet->stream);

    int ind = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++)
    {
        if(it->exist)
        {
            if(*(refineNet->score_->pdata+ ind*refineNet->OUT_PROB_SIZE+1)>refineNet->Rthreshold)
            {
                memcpy(it->regreCoord, refineNet->location_->pdata+ind*refineNet->OUT_LOCATION_SIZE, refineNet->OUT_LOCATION_SIZE*sizeof(float));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet->score_->pdata +ind*refineNet->OUT_PROB_SIZE+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
            ind++;
        }
    }

    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols,true);
    second_time = clock() - second_time;
    cout<<"Rnet time is  "<<1000*(double)second_time/CLOCKS_PER_SEC<<endl;

    //third stage
    count = 0;
    clock_t third_time = clock();
    inputed_num = 0;
    step = 48*48*3*sizeof(float);
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            cuda::resize(image(temp), thirdImages_buffer[inputed_num],
                         Size(48, 48), 0, 0, cv::INTER_LINEAR, cv_streams[inputed_num]);
            gpu_image2Matrix_with_transpose(48,48,thirdImages_buffer[inputed_num],
                                            (float*)(outNet->buffers[refineNet->inputIndex])+inputed_num*step,cudastreams[inputed_num]);
            inputed_num++;
        }
    }

    cudaDeviceSynchronize();
    outNet->run(inputed_num, *onet_engine, outNet->stream);
    cout<<"Onet input images number is "<<inputed_num<<endl;

    ind = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            mydataFmt *pp=NULL;
            if(*(outNet->score_->pdata + 2*ind +1)>outNet->Othreshold){
                memcpy(it->regreCoord, outNet->location_->pdata + 4*ind, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet->score_->pdata +2*ind +1);
                pp = outNet->points_->pdata + 10* ind;
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
        }
    }

    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols, true);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");

    cout<<"Onet time is  "<<1000*(double)(clock()-third_time)/CLOCKS_PER_SEC<<endl;
    cout<<"total run time "<<1000*(double)(clock()-first_time)/CLOCKS_PER_SEC<<endl;
    Mat cpuImage;
    image.download(cpuImage);
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            rectangle(cpuImage, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++)circle(cpuImage,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }

    imshow("result", cpuImage);
    waitKey(0);
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
}
