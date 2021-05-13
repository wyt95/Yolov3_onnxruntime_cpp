#include "objectdetection.h"
#include <experimental/filesystem>
#include <boost/filesystem.hpp>

namespace fs = std::experimental::filesystem;

int main(int argc, char** argv)
{
    double timeStart = (double)getTickCount();
    //YoloV3
    Yolov3 *yolov3 = new Yolov3();

    std::cout << "yolov3 init()" << std::endl;
    if ( !yolov3->init("../model/yolov3-ykx.onnx", "../model/coco_labels.txt", 608) )
    {
        std::cout << "yolov3 init() failed...." << std::endl;
        return 0;
    }

    std::string AccurayFolderPath = "../pic/";
    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        std::cout << entry.path() << std::endl;

        string image_id = entry.path().string();

        boost::filesystem::path filePath(image_id);
        cout << "filePath.filename(): " << filePath.filename() << endl;

        Mat img;

        img = imread(image_id.c_str());

        if ( !yolov3->preProcessing(img) )
        {
            std::cerr << "Yolov3 preProcessing() failed...." << '\n';
            continue;
        }

        if ( !yolov3->runmodel() )
        {
            std::cerr << "Yolov3 runmodel() failed...." << '\n';
            continue;
        }
        
        if ( !yolov3->postProcessing() )
        {
            std::cerr << "Yolov3 postProcessing() failed...." << '\n';
            continue;
        }

        if ( !yolov3->DrowBoxes(img, filePath.filename().string()) )
        {
            std::cerr << "Yolov3 DrowBoxes() failed...." << '\n';
            continue;
        }

        yolov3->release();
    }

    delete yolov3;

    std::cout << "yolov3 finished...." << std::endl;

    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "running time ：" << nTime << "sec\n" << endl;

    return 0;
}

