#include "network.h"
#include "mtcnn.h"
#include <time.h>

int main()
{
    Mat image = imread("/home/zhou/renlian.jpg");
    resize(image,image,Size(640,480));
    mtcnn find(image.rows, image.cols);
    clock_t start;
    start = clock();
    find.findFace(image);
    start = clock() -start;
    imshow("result", image);
    imwrite("result.jpg",image);

    cout<<"time is  "<<1000*(double)start/CLOCKS_PER_SEC<<endl;
//     Mat image;
//     VideoCapture cap(0);
//     if(!cap.isOpened())
//         cout<<"fail to open!"<<endl;
//     cap>>image;
//     if(!image.data){
//         cout<<"读取视频失败"<<endl;
//         return -1;
//     }
//
//     mtcnn find(image.rows, image.cols);
//     clock_t start;
//     int stop = 12000;
//     while(stop--){
//         start = clock();
//         cap>>image;
//         find.findFace(image);
//         imshow("result", image);
//         if( waitKey(1)>=0 ) break;
//         start = clock() -start;
//         cout<<"time is  "<<start/1e3<<endl;
//     }
//
    waitKey(0);
    image.release();
    return 0;
}