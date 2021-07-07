#ifndef _OBJDETEC_H_
#define _OBJDETEC_H_

#include <iostream>
#include <fstream> 
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>
#include <vector>
#include <array>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <memory>

#include <onnxruntime_cxx_api.h>
#if defined(USE_CPU)
#include <cpu_provider_factory.h>
#elif defined(USE_CUDA)
#include <cuda_provider_factory.h>
#endif
//#include <core/providers/tensorrt/tensorrt_provider_factory.h>

#include "NumCpp.hpp"

using namespace cv;
using namespace std;

class Yolov3
{
private:
    // Res struct & function
    typedef struct DetectionRes {
        float x, y, w, h, prob;
        int classpos;
    } DetectionRes;

private:
    //set onnxruntime env value
    void setOnnxRuntimeEnv();
    bool setOnnxRuntimeModelInputOutput(int input_size);
    bool setSession();

    //get label's name
    bool GetLabelAndCategories(std::string labelFilePath);

    //reshape_output
    bool reshapeOutput(float* inputTensor, std::vector<int64_t> output_dims, float* outputTensor);

    //nmsBox
    void DoNms(vector<DetectionRes>& detections);

private:

    //ONNX RUNTIME
    Ort::SessionOptions m_OrtSessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;

    std::unique_ptr<Ort::Session> m_OrtSession;
    std::unique_ptr<Ort::Env> m_OrtEnv;

    //onnxModlPath
    std::string m_sModelOnnxPath;

    //OnnxRuntime Input Model
    size_t num_input_nodes;
    std::vector<const char *> input_node_names;
    std::vector< std::vector<int64_t> > input_node_dims;

    //OnnxRuntime Output Model
    size_t num_out_nodes;
    std::vector<const char *> output_node_names;
    std::vector< std::vector<int64_t> > output_node_dims;
    std::vector< std::vector<int64_t> > output_shapes;

    //Label
    std::vector<std::string> m_Categories;

    //
    std::vector<std::vector<int> > yolo_masks_;
    std::vector<std::vector<int> > yolo_anchors_;

    //neural network input dimension
    int m_iInput_h;
    int m_iInput_w;

    //data
    std::vector<float> input_image_;
    std::vector<Ort::Value> output_tensors;
    std::vector<std::vector<float> > outputs_reshaped; //H,W,3,85

    std::vector<DetectionRes> detections;

    //original image width and height
    int m_OriMwidth;
    int m_OriMheight;

    //bool m_Isyolov3tiny;
    float m_threshold;
    float m_Nmsthreshold;

    //handle initialization
    bool m_bInit;
    //used to call init only one time per instances
    bool m_bCheckInit;
    //used to verify if preprocess is called on the same run
    bool m_bCheckPre;
    //used to verify if run model is called on the same run
    //bool m_bCheckRun;
    //used to verify id post process is called
    //bool m_bCheckPost;

public:
    Yolov3();
    virtual ~Yolov3();

    void release();

    bool init(const std::string modelPathOnnx, const std::string labelFilePath, int input_size = 416, bool Isyolov3tiny = false, float threshold = 0.5);

    //pic load, resize and normalize
    bool preProcessing(const cv::Mat &inputImg);

    //run
    bool runmodel();

    //postprocessing
    bool postProcessing();

    //draw boxes
    bool DrowBoxes(cv::Mat &inputImg, const std::string image_id);
};

#endif