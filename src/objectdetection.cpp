#include "objectdetection.h"

float sigmoid(float in) {
	return 1.f / (1.f + exp(-in));
}
float exponential(float in) {
	return exp(in);
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

Yolov3::Yolov3()
{
    m_bInit = false;
    m_bCheckInit = false;
    m_bCheckPre = false;
    //m_bCheckRun = false;
    //m_bCheckPost = false;

    detections.clear();
    input_image_.clear();
    output_tensors.clear();
    outputs_reshaped.clear();
}

void Yolov3::release()
{
    if (m_bCheckInit)
    {
        detections.clear();
        input_image_.clear();
        output_tensors.clear();
        outputs_reshaped.clear();

        std::cout << "free all vec...." << std::endl;
    }
}

void Yolov3::setOnnxRuntimeEnv()
{
    //m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));
    m_OrtEnv = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test");

    Ort::SessionOptions m_OrtSessionOptions;

    m_OrtSessionOptions.SetIntraOpNumThreads(1);
    //ORT_ENABLE_ALL seems to have better perforamance
    m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    //m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    #if defined(USE_CPU)
    std::cout << "USE_CPU...." << std::endl;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(m_OrtSessionOptions, 0));
    #elif defined(USE_CUDA)
    std::cout << "USE_CUDA...." << std::endl;
    OrtCUDAProviderOptions cuda_options{
        0,
        OrtCudnnConvAlgoSearch::DEFAULT,
        std::numeric_limits<size_t>::max(),
        0,
        true,
        0,
        nullptr,
        nullptr};
    m_OrtSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_OrtSessionOptions, 0));
    #elif defined(USE_TRT)
    std::cout << "USE_TRT...." << std::endl;
    OrtTensorRTProviderOptions tensorrt_options{
            0,
            0,
            nullptr,
            1000,
            1,
            1 << 30,
            0,
            0,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            nullptr,
            0,
            nullptr,
            0};
    m_OrtSessionOptions.AppendExecutionProvider_TensorRT(tensorrt_options);
    OrtCUDAProviderOptions cuda_options{
        0,
        OrtCudnnConvAlgoSearch::DEFAULT,
        std::numeric_limits<size_t>::max(),
        0,
        true,
        0,
        nullptr,
        nullptr};
    m_OrtSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    #endif

    //m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_sModelOnnxPath.c_str(), m_OrtSessionOptions));
    m_OrtSession = Ort::Session(m_OrtEnv, m_sModelOnnxPath.c_str(), m_OrtSessionOptions);

}

bool Yolov3::setOnnxRuntimeModelInputOutput(int input_size)
{
    if (m_OrtSession == nullptr)
    {
        std::cerr << "m_OrtSession nullptr !!" << '\n';
        return false;
    }

    num_input_nodes = m_OrtSession.GetInputCount();
    input_node_names = std::vector<const char *>(num_input_nodes);

    // print model input layer (node names, types, shape etc.)
    std::cout << "Number of inputs :" << num_input_nodes << std::endl;
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) 
    {
        // print input node names
        char* input_name = m_OrtSession.GetInputName(i, allocator);
        std::cout << "Input " << i << " : " << "name = " << input_name << std::endl;
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = m_OrtSession.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();

        // print input shapes/dims
        std::vector<int64_t> inputDims = tensor_info.GetShape();
        std::cout << "Input Dimensions: " << inputDims << std::endl;
        input_node_dims.push_back(inputDims);
    }

    num_out_nodes = m_OrtSession.GetOutputCount();
    output_node_names = std::vector<const char *>(num_out_nodes);

    std::cout << "Number of outputs :" << num_out_nodes << std::endl;
    // iterate over all output nodes
    for (int i = 0; i < num_out_nodes; i++) 
    {
        // print output node names
        char* output_name = m_OrtSession.GetOutputName(i, allocator);
        std::cout << "output " << i << " : " << "name = " << output_name << std::endl;
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = m_OrtSession.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output shapes/dims
        std::vector<int64_t> outputDims = tensor_info.GetShape();
        std::cout << "Output Dimensions: " << outputDims << std::endl;
        output_node_dims.push_back(outputDims);

        int batch_size = 1;
        int output_size = (input_size/32)*(pow(2,i));
        
        //NCHW
        std::vector<int64_t> output_shape = { batch_size, 3*(5+(int)m_Categories.size()), output_size, output_size };
        output_shapes.push_back(output_shape);
    }

    return true;
}

bool Yolov3::GetLabelAndCategories(std::string labelFilePath)
{
    //读取分类标签信息
    ifstream f_val_label;
    cout << "labelFilePath :" << labelFilePath.c_str() << endl;
    f_val_label.open(labelFilePath.c_str(), ios::in);
    if (!f_val_label.is_open() || f_val_label.fail())
    {
        std::cerr << "coco_labels open failed!!" << '\n';
        return false;
    }

    while(!f_val_label.eof())
    {
        string tempValue;
        while(getline(f_val_label, tempValue))
        {
            m_Categories.push_back(tempValue);
        }
    }
    f_val_label.close();

    if(m_Categories.empty())
    {
        std::cerr << "m_Categories empty!!" << '\n';
        return false;
    }

    cout << "m_Categories num:" << m_Categories.size() << endl;

    return true;
}

//NCHW -> NHWC
bool Yolov3::reshapeOutput(float* inputTensor, std::vector<int64_t> output_dims, float* outputTensor)
{
    if ( (inputTensor == nullptr) || output_dims.empty() )
    {
        std::cerr << "inputTensor nullptr or output_dims empty..." << '\n';
        return false;
    }

    int64_t m_N = output_dims[0];
    int64_t m_C = output_dims[1];
    int64_t m_H = output_dims[2];
    int64_t m_W = output_dims[3];

    size_t stride = m_H * m_W;
    for (int c = 0; c < m_C; c++)
    {
        size_t t = c*stride;
        for (int i = 0; i < stride; i++)
        {
            outputTensor[i * m_C + c] = inputTensor[t + i];
        }
    }

    return true;
}

bool Yolov3::init(const std::string modelPathOnnx, const std::string labelFilePath, int input_size, bool Isyolov3tiny, float threshold)
{
    if (!m_bCheckInit)
    { 
        try
        {
            if ((access(modelPathOnnx.c_str(), F_OK) == -1) && (access(labelFilePath.c_str(), F_OK) == -1) )
            {
                std::cout << modelPathOnnx.c_str() << " or " << labelFilePath.c_str() << "param error!" << std::endl;
                return false;
            }
            else
            {
                m_sModelOnnxPath = modelPathOnnx;
                m_iInput_w = input_size;
                m_iInput_h = input_size;
                //m_Isyolov3tiny = Isyolov3tiny;
                m_threshold = threshold;
                m_Nmsthreshold = 0.5;

                //获取分类标签信息
                if (!GetLabelAndCategories(labelFilePath))
                {
                    std::cerr << "GetLabelAndCategories() failed!!" << '\n';
                    return false;
                }

                //OnnxRuntime set Env
                setOnnxRuntimeEnv();

                //model input output
                if (!setOnnxRuntimeModelInputOutput(input_size))
                {
                    std::cerr << "setOnnxRuntimeModelInputOutput() failed!!" << '\n';
                    return false;
                }

                if (Isyolov3tiny)
                {
                    std::vector<std::vector<int> > yolo_masks{ {3, 4, 5}, {0, 1, 2} };
                    std::vector<std::vector<int> > yolo_anchors{ {10,14}, {23,27}, {37,58}, {81,82}, {135,169}, {344,319} };
                    yolo_masks_.assign(yolo_masks.begin(), yolo_masks.end());
                    yolo_anchors_.assign(yolo_anchors.begin(), yolo_anchors.end());
                }
                else
                {
                    std::vector<std::vector<int> > yolo_masks{ {6, 7, 8}, {3, 4, 5}, {0, 1, 2} };
                    std::vector<std::vector<int> > yolo_anchors{ {10,13}, {16,30}, {33,23}, {30,61}, {62,45}, {59,119}, {116,90}, {156,198}, {373,326} };
                    yolo_masks_.assign(yolo_masks.begin(), yolo_masks.end());
                    yolo_anchors_.assign(yolo_anchors.begin(), yolo_anchors.end());
                }

                m_bInit = true;
                m_bCheckInit = true;

                return true;
            }

        } catch(const std::exception& ex) {
            std::cerr << ex.what() << '\n';
            return false;
        }
    }
    else
    {
        std::cout << "Yolov3 has been already init()!" << std::endl;
        return false;
    }
}

//convert it from HWC format to NCHW format
bool Yolov3::preProcessing(const cv::Mat &inputImg)
{
    try
    {
        detections.clear();
        input_image_.clear();
        output_tensors.clear();
        outputs_reshaped.clear();

        cv::Mat img = inputImg.clone();
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        m_OriMwidth = img.cols;
        m_OriMheight = img.rows;
        std::cout << "m_OriMwidth: " << m_OriMwidth << ", " << "m_OriMheight: " << m_OriMheight << std::endl;
        resize(img, img, cv::Size(m_iInput_w, m_iInput_h), cv::INTER_CUBIC);
        img.convertTo(img, CV_32F, 1.f / 255.0);

        cv::dnn::blobFromImage(img, img);

        size_t inputTensorSize = vectorProduct(input_node_dims[0]);
        input_image_.resize(inputTensorSize);
        input_image_.assign(img.begin<float>(),
                            img.end<float>());

        m_bCheckPre = true;
    } catch(const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return false;
    }
    return true;
}

bool Yolov3::runmodel()
{
    if (m_bCheckPre && m_bInit && m_bCheckInit)
    {
        std::cout << "Yolov3 runmodel() start....." << std::endl;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
        std::vector<int64_t> input_shape(input_node_dims[0]);
        auto input_tenser = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), \
                                                                    input_shape.data(), input_shape.size());

        output_tensors = m_OrtSession.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tenser, input_node_names.size(), \
                                                    output_node_names.data(), output_node_names.size());
        std::cout << "output_tensors size: " << output_tensors.size() << std::endl;

        //m_bCheckRun = true;
        std::cout << "Yolov3 runmodel() end....." << std::endl;
    }
    else
    {
        std::cerr << "Yolov3 runmodel() failed...." << '\n';
        return false;
    }

    return true;
}

void Yolov3::DoNms(vector<DetectionRes>& m_detections)
{
	auto iouCompute = [](float * lbox, float* rbox) {
		float interBox[] = {
			max(lbox[0], rbox[0]), //left
			min(lbox[0] + lbox[2], rbox[0] + rbox[2]), //right
			max(lbox[1], rbox[1]), //top
			min(lbox[1] + lbox[3], rbox[1] + rbox[3]), //bottom
		};

		if (interBox[2] >= interBox[3] || interBox[0] >= interBox[1])
			return 0.0f;

		float interBoxS = (interBox[1] - interBox[0] + 1) * (interBox[3] - interBox[2] + 1);
		return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
	};

	sort(m_detections.begin(), m_detections.end(), [=](const DetectionRes & left, const DetectionRes & right) {
		return left.prob > right.prob;
	});

	vector<DetectionRes> result;
	for (unsigned int m = 0; m < m_detections.size(); ++m) {
		result.push_back(m_detections[m]);
		for (unsigned int n = m + 1; n < m_detections.size(); ++n) {
			if (iouCompute((float *)(&m_detections[m]), (float *)(&m_detections[n])) > m_Nmsthreshold) {
				m_detections.erase(m_detections.begin() + n);
				--n;
			}
		}
	}
	m_detections = move(result);
}

bool Yolov3::postProcessing()
{
    if (m_bCheckPre && m_bInit && m_bCheckInit)
    {
        //todo ...
        if ( (output_tensors.size()!=output_shapes.size()?true:false) )
        {
            std::cerr << "output_tensors.size not equal to output_shapes.size.... " << '\n';
            return false;
        }

        std::vector<std::vector<int64_t> > shapes;
        for (auto& tensor : output_tensors)
        {
            auto type_info = tensor.GetTypeInfo();
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            size_t tensor_size = tensor_info.GetElementCount();
            std::vector<int64_t>  output_dims = tensor_info.GetShape();

            float* inputTensor = tensor.GetTensorMutableData<float>();
            float* outputTensor = (float*)malloc(tensor_size * sizeof(float));

            //NCHW -> NHWC
            if ( !reshapeOutput(inputTensor, output_dims, outputTensor) )
            {
                std::cerr << "reshapeOutput failed..." << '\n';
                continue;
            }

            std::vector<int64_t> m_shapes{output_dims[2], output_dims[3], 3, (4 + 1 + (int64_t)m_Categories.size())}; //h,w,3,85
            shapes.push_back(m_shapes);

            std::vector<float> outputs_{outputTensor, outputTensor+tensor_size};
            outputs_reshaped.push_back(outputs_);

            if (outputTensor != nullptr)
            {
                free(outputTensor);
                outputTensor = nullptr;
            }

        }

        for (int i = 0; i < outputs_reshaped.size(); i++)
        {
            auto masks = yolo_masks_[i];
            vector<vector<int> > anchors;
            for (auto mask : masks) 
                anchors.push_back(yolo_anchors_[mask]);
            auto shape = shapes[i];
            float *transposed_output = outputs_reshaped[i].data();
            for (int h = 0; h < shape[0]; h++)
            {
                int offset_h = h * shape[1] * shape[2] * shape[3];
                for (int w = 0; w < shape[1]; w++)
                {
                    int offset_w = offset_h + w * shape[2] * shape[3];
                    for (int c = 0; c < shape[2]; c++)
                    {
                        int offset_c = offset_w + c * shape[3];
                        float *ptr = transposed_output + offset_c;
                        ptr[4] = sigmoid(ptr[4]); //box_confidence

                        //5-84 box_class_probs
                        for (int cp = 5; cp < shape[3]; cp++)
                        {
                            ptr[cp] = sigmoid(ptr[cp]);
                        }

                        //float class_score = ptr[4] * ptr[5];
                        std::vector<float> clasScorVec{ptr+5, ptr+85};
                        auto maxPos = std::max_element(clasScorVec.begin(), clasScorVec.end());
                        float class_score = ptr[4] * ptr[maxPos-clasScorVec.begin()+5];

                        if (class_score < m_threshold)
                            continue;

                        int maxclassPos = maxPos-clasScorVec.begin();

                        ptr[0] = sigmoid(ptr[0]); //x
                        ptr[1] = sigmoid(ptr[1]); //y
                        ptr[2] = exponential(ptr[2]) * anchors[c][0]; //w
                        ptr[3] = exponential(ptr[3]) * anchors[c][1]; //h

                        ptr[0] += w;
                        ptr[1] += h;
                        ptr[0] /= shape[0];
                        ptr[1] /= shape[1];
                        ptr[2] /= m_iInput_w;
                        ptr[3] /= m_iInput_w;
                        ptr[0] -= (ptr[2] / 2.f);
                        ptr[1] -= (ptr[3] / 2.f);

                        DetectionRes det;;
                        det.x = ptr[0];
                        det.y = ptr[1];
                        det.w = ptr[2];
                        det.h = ptr[3];
                        det.prob = class_score;
                        det.classpos = maxclassPos;
                        detections.push_back(det);
                    }
                
                }
            }

        }

        //correct box
        std::cout << "detections size: " << detections.size() << std::endl;
        for (auto& bbox : detections) 
        {
            bbox.x *= m_OriMwidth;
            bbox.y *= m_OriMheight;
            bbox.w *= m_OriMwidth;
            bbox.h *= m_OriMheight;
        }

        //nms
        DoNms(detections);

        std::cout << "detections size: " << detections.size() << std::endl;

        //m_bCheckPost = true;
    }
    else
    {
        std::cerr << "Yolov3 postProcessing() failed.... , m_bCheckRun false..." << '\n';
        return false;
    }

    return true;
}

bool Yolov3::DrowBoxes(cv::Mat &inputImg, const std::string image_name)
{
    for (auto box : detections)
    {
        int x = box.x,
            y = box.y,
            w = box.w,
            h = box.h,
            clspos = box.classpos;
        cv::Rect rect = { x, y, w, h };
        cv::rectangle(inputImg, rect, cv::Scalar(0, 0, 255), 2);
        char boxProb[64];
        sprintf(boxProb, "%.2f", box.prob);
        std::string category= m_Categories[box.classpos] + ":" + boxProb;
        cv::putText(inputImg,
                    category.c_str(),
                    cv::Point(box.x-3, box.y-6),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.4,
                    CV_RGB(0, 0, 255),
                    1);
    }

    std::string PicName = "../out/prdiect_" + image_name;
    cv::imwrite(PicName.c_str(), inputImg);

    return true;
}