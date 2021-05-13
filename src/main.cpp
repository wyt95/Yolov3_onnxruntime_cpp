#include "objectdetection.h"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    
    for (size_t i = 0; i != input.size(); ++i) {
        /*sum += */y[i] = std::exp(input[i] /*- rowmax*/);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input.size()/2; ++i) {
        sum = y[i] + y[i + input.size()/2];
        input[i] = y[i] / sum;
        input[i + input.size()/2] = y[ i + input.size()/2] / sum;
    }
}

int main(int argc, char** argv)
{
    double timeStart = (double)getTickCount();
    //YoloV3
    Yolov3 *yolov3 = new Yolov3();

    std::cout << "yolov3 init()" << std::endl;
    if ( !yolov3->init("/mnt/share/yolov3_onnx_pro-master/yolov3-ykx.onnx", "/mnt/share/yolov3_onnx_pro-master/coco_labels.txt", 608) )
    {
        std::cout << "yolov3 init() failed...." << std::endl;
        return 0;
    }

    std::cout << "yolov3 preProcessing...." << std::endl;
    std::string AccurayFolderPath = "/mnt/share/yolov3_onnx_pro-master/yolov3_cpp/pic/";
    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        std::cout << entry.path() << std::endl;

        string image_id = entry.path().string();

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

        if ( !yolov3->DrowBoxes(img) )
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

