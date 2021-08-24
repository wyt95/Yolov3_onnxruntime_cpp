#include "objectdetection.h"
#include <experimental/filesystem>
#include <boost/filesystem.hpp>

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv)
{
    std::chrono::_V2::system_clock::time_point t0, t1, t2, t3, t4, t5, t6, t7, t8;
    float diff1, diff2, diff3, diff4, diff5, diff6, diff7;
    //YoloV3
    Yolov3 *yolov3 = new Yolov3();

    t0 = std::chrono::high_resolution_clock::now();

    std::cout << "yolov3 init()" << std::endl;
    if ( !yolov3->init("../model/yolov3-ykx.onnx", "../model/coco_labels.txt", 608) )  //初始化的时间需要1.15-1.20s左右
    {
        std::cout << "yolov3 init() failed...." << std::endl;
        return 0;
    }

    t1 = std::chrono::high_resolution_clock::now();

    std::string AccurayFolderPath = "../pic/";
    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        std::cout << entry.path() << std::endl;

        string image_id = entry.path().string();

        boost::filesystem::path filePath(image_id);
        cout << "filePath.filename(): " << filePath.filename() << endl;

        t2 = std::chrono::high_resolution_clock::now();
        Mat img;
        img = imread(image_id.c_str());
        t3 = std::chrono::high_resolution_clock::now();

        if ( !yolov3->preProcessing(img) )
        {
            std::cerr << "Yolov3 preProcessing() failed...." << '\n';
            continue;
        }
        
        t4 = std::chrono::high_resolution_clock::now();

        if ( !yolov3->runmodel() )
        {
            std::cerr << "Yolov3 runmodel() failed...." << '\n';
            continue;
        }

        t5 = std::chrono::high_resolution_clock::now();

        if ( !yolov3->postProcessing() )
        {
            std::cerr << "Yolov3 postProcessing() failed...." << '\n';
            continue;
        }

        t6 = std::chrono::high_resolution_clock::now();

        if ( !yolov3->DrowBoxes(img, filePath.filename().string()) )
        {
            std::cerr << "Yolov3 DrowBoxes() failed...." << '\n';
            continue;
        }

        t7 = std::chrono::high_resolution_clock::now();

        yolov3->release();
    }

    t8 = std::chrono::high_resolution_clock::now();

    std::cout << "yolov3 finished...." << std::endl;

    diff1 = std::chrono::duration<double, std::milli>(t1 - t0).count();//计时init
    cout << "running time init: " << diff1 << "ms\n" << endl;

    diff2 = std::chrono::duration<double, std::milli>(t3 - t2).count();//计时imread
    cout << "running time imread: " << diff2 << "ms\n" << endl;

    diff3 = std::chrono::duration<double, std::milli>(t4 - t3).count();//计时preProcessing
    cout << "running time preProcessing: " << diff3 << "ms\n" << endl;

    diff4 = std::chrono::duration<double, std::milli>(t5 - t4).count();//计时runmodel CPU:2108.52ms GPU:656.729ms
    cout << "running time runmodel: " << diff4 << "ms\n" << endl;

    diff5 = std::chrono::duration<double, std::milli>(t6 - t5).count();//计时postProcessing
    cout << "running time postProcessing: " << diff5 << "ms\n" << endl;

    diff6 = std::chrono::duration<double, std::milli>(t7 - t6).count();//计时DrowBoxes
    cout << "running time DrowBoxes: " << diff6 << "ms\n" << endl;

    diff7 = std::chrono::duration<double, std::milli>(t7 - t1).count();//计时 -> GPU: 0.73-0.75(s/pic) CPU: 2.55(s/pic)
    cout << "running time without model-loading: " << diff7 << "ms\n" << endl;

    delete yolov3;

    return 0;
}

