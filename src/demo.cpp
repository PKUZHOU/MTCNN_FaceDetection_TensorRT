#include "network.h"
#include "mtcnn.h"
#include "kernels.h"
#include <time.h>

//void camera_test(int frames)
//{
//    Mat image;
//    VideoCapture cap(0);
//    if(!cap.isOpened())
//        cout<<"fail to open!"<<endl;
//    cap>>image;
//    if(!image.data) {
//        cout << "unable to open camera" << endl;
//        return;
//    }
//    mtcnn find(image.rows, image.cols);
//    clock_t start;
//    int stop = frames;
//    while(stop--){
//        start = clock();
//        cap>>image;
//        find.findFace(image);
//        imshow("result", image);
//        waitKey(1);
//        start = clock() -start;
//        cout<<"time is  "<<(double)start/CLOCKS_PER_SEC<<endl;
//    }
//    cout<<"end"<<endl;
//}

void image_test(string image_path)
{
    //test using images
    Mat image = imread(image_path);
    cuda::GpuMat Gpuimage(image);
    cuda::resize(Gpuimage,Gpuimage,Size(1920,1080));
    mtcnn find(Gpuimage.rows, Gpuimage.cols);
    clock_t start;
    start = clock();
    find.findFace(Gpuimage);
    start = clock() -start;
    cout<<"time is  "<<1000*(double)start/CLOCKS_PER_SEC<<endl;
    imshow("result", image);
    imwrite("result.jpg",image);
    // waitKey(0);
    image.release();
}



int main()
{
//    camera_test(10000);
    image_test("4.jpg");
    return 0;
}
