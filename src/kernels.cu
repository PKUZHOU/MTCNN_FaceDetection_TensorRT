#include "kernels.h"
#include <thrust/device_vector.h>
using namespace cv::cuda;

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

__global__ void image2Matrix_kernel(int width, int height,  PtrStepSz<uchar3> image, float* matrix){

    const int w = blockIdx.x;
    const int h = blockIdx.y;

    float alpha = -127.5;
    float beta = 0.0078125;
    if (w < width && h < height)
    {
        uchar3 v = image(h,w);
        *(matrix + 0*height*width + h*width + w) = (float(v.z)-alpha)*beta;
        *(matrix + 1*height*width + h*width + w) = (float(v.y)-alpha)*beta;
        *(matrix + 2*height*width + h*width + w) = (float(v.x)-alpha)*beta;
    }

}
__global__ void image2Matrix_with_transpose_kernel(int width, int height,  PtrStepSz<uchar3> image, float* matrix){

    const int w = blockIdx.x;
    const int h = blockIdx.y;
    float alpha = -127.5;
    float beta = 0.0078125;
    if (w < width && h < height)
    {
        uchar3 v = image(w,h); //swap w and h to transpose
        *(matrix + 0*height*width + h*width + w) = (float(v.z)-alpha)*beta;
        *(matrix + 1*height*width + h*width + w) = (float(v.y)-alpha)*beta;
        *(matrix + 2*height*width + h*width + w) = (float(v.x)-alpha)*beta;
    }

}
void gpu_image2Matrix(int width, int height,  cuda::GpuMat & image, float* matrix, cudaStream_t &stream)
{
     /*
        image : input image in GpuMat format, WHC arrangement and BGR order
        matrix: gpu float array, CHW and RGB order
    */
    dim3 block(width, height); // width * height blocks, 1 thread each
    image2Matrix_kernel<<<block,1,0,stream>>>(width,height,image,matrix);
}

void gpu_image2Matrix_with_transpose(int width, int height,  cuda::GpuMat & image, float* matrix, cudaStream_t &stream)
{
    dim3 block(width, height); // width * height blocks, 1 thread each
    image2Matrix_with_transpose_kernel<<<block,1,0,stream>>>(width,height,image,matrix);
}


//__global__ void crop_and_resize_kernel(int x1, int y1, int x2, int y2, int PtrStepSz<uchar3> image, int* temp_buffer)
//{
//    const int x = blockDim.x*blockIdx.x+threadIdx.x;
//    const int y = blockDim.y*blockIdx.y+threadIdx.y;
//    if(x>=(x2-x1)||y>=(y2-y1))
//        return;
//    uchar3 v = image(y,x);
//    temp_buffer[]
//}
//
//__global__ void generate_batch_kernel(int crop_size, int width, int height, int * boxes_data, PtrStepSz<uchar3> image, float * output_batch)
//{
//    const int box_idx = blockIdx.x*blockDim.x+threadIdx.x;
//    if(!boxes_data||!output_batch)
//        return cudaErrorInvalidDevicePointer;
//    if(crop_size==0||width==0||height==0||num==0)
//        return cudaErrorInvalidValue;
//
//    int offset = box_idx*4*sizeof(int);
//    //the bbox
//    int x1 = int(boxes_data+offset);
//    int y1 = int(boxes_data+offset+1);
//    int x2 = int(boxes_data+offset+2);
//    int y2 = int(boxes_data+offset+3);
//
//    // the width and height of area to crop
//    int w = x2-x1;
//    int h = y2-y1;
//
//    // total pixels of area to crop
//    int total_pixels = w*h;
//
//    // creat a temp buffer to store
//    float *temp_buffer = new float[total_pixels*3];
//    const dim3 blockDim(8,8);
//    const dim3 gridDim(iDivUp(w,blockDim.x),iDivUp(h,blockDim.y));
//    crop_kernel<<<gridDim,blockDim>>>(x1,y2,x2,y2,image, temp_buffer);
//}
//
//void boxes2bactch(int num, int crop_size, int width, int height, float * boxes_data, cuda::GpuMat image, float * output_batch, float * cudaStream_t& stream)
//{
//    generate_batch_kernel<<<num,1,0,stream>>>(crop_size,width, height, boxes_data, image, output_batch);
//}


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
