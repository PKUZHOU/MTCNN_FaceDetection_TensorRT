#include "kernels.h"
#include <thrust/device_vector.h>
using namespace cv::cuda;
__global__ void image2Matrix_kernel(int width, int height,  PtrStepSz<uchar3> image, float* matrix){

    const int w = blockIdx.x;
    const int h = blockIdx.y;
    
    if (w < width && h < height)
    {
        uchar3 v = image(h,w);
        *(matrix + 0*height*width + h*width + w) = (float(v.z)-127.5)*0.0078125;
        *(matrix + 1*height*width + h*width + w) = (float(v.y)-127.5)*0.0078125;
        *(matrix + 2*height*width + h*width + w) = (float(v.x)-127.5)*0.0078125;
    }

}
void gpu_image2Matrix(int width, int height,  GpuMat & image, float* matrix)
{
    /*
        image : input image in GpuMat format, WHC arrangement and BGR order
        matrix: gpu float array, CHW and RGB order
    */
    dim3 block(width, height); // width * height blocks, 1 thread each
    image2Matrix_kernel<<<block,1>>>(width,height,image,matrix);
}



//__global__ void generatebox_kernel(int width, int height, float * scores, float * location, float pthreshold )
//{
//    const int w = blockIdx.x;
//    const int h = blockIdx.y;
//    if(w<width && h< height)
//    {
//        float score  = *(scores + width*height + w*height+ h );
//        if(score > pthreshold)
//        {
//
//        }
//    }
//
//
//}
//void gpu_generatebox(int width , int height , void * score, void * location, float scale, float pthreshold)
//{
//
//    int stride = 2;
//    int cellsize = 12;
//    int count = 0;
//    //score p
//    void *p = (float*)score + width * height;
//    void *plocal = (float*)location;
//    struct Bbox bbox;
//    struct orderScore order;
//    for (int row = 0; row < score->height; row++) {
//        for (int col = 0; col < score->width; col++) {
//            if (*p > Pthreshold) {
//                bbox.score = *p;
//                order.score = *p;
//                order.oriOrder = count;
//                bbox.x1 = round((stride * row + 1) / scale);
//                bbox.y1 = round((stride * col + 1) / scale);
//                bbox.x2 = round((stride * row + 1 + cellsize) / scale);
//                bbox.y2 = round((stride * col + 1 + cellsize) / scale);
//                bbox.exist = true;
//                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
//                for (int channel = 0; channel < 4; channel++)
//                    bbox.regreCoord[channel] = *(plocal + channel * location->width * location->height);
//                boundingBox_.push_back(bbox);
//                bboxScore_.push_back(order);
//                count++;
//            }
//            p++;
//            plocal++;
//        }
//    }
//    dim3 block(width, height);
//    generatebox_kernel(width, height, score, location, pthreshold);

//}
