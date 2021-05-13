#include "objectdetection.h"

float sigmoid(float in) {
	return 1.f / (1.f + exp(-in));
}
float exponential(float in) {
	return exp(in);
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

Yolov3::~Yolov3()
{
    //release();
    if (m_bCheckInit)
    {
        m_OrtSession.reset();
        m_OrtEnv.reset();

        m_bInit = false;
        m_bCheckInit = false;
        m_bCheckPre = false;

        std::cout << "free all...." << std::endl;
    }
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
    m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    m_OrtSessionOptions.SetIntraOpNumThreads(1);
    //ORT_ENABLE_ALL seems to have better perforamance
    m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    //m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(m_OrtSessionOptions, 0));
}

bool Yolov3::setOnnxRuntimeModelInputOutput(int input_size)
{
    if (m_OrtSession == nullptr)
    {
        std::cerr << "m_OrtSession nullptr !!" << '\n';
        return false;
    }

    num_input_nodes = m_OrtSession->GetInputCount();
    input_node_names = std::vector<const char *>(num_input_nodes);

    // print model input layer (node names, types, shape etc.)
    std::cout << "Number of inputs :" << num_input_nodes << std::endl;
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) 
    {
        // print input node names
        char* input_name = m_OrtSession->GetInputName(i, allocator);
        std::cout << "Input " << i << " : " << "name = " << input_name << std::endl;
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = m_OrtSession->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();

        // print input shapes/dims
        std::vector<int64_t> inputNode_dims = tensor_info.GetShape();
        std::cout << "Input " << i << " : " << "num_dims = " << inputNode_dims.size() << std::endl;
        for (int j = 0; j < inputNode_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, inputNode_dims[j]);
        }
        input_node_dims.push_back(inputNode_dims);
    }

    num_out_nodes = m_OrtSession->GetOutputCount();
    output_node_names = std::vector<const char *>(num_out_nodes);

    std::cout << "Number of outputs :" << num_out_nodes << std::endl;
    // iterate over all output nodes
    for (int i = 0; i < num_out_nodes; i++) 
    {
        // print output node names
        char* output_name = m_OrtSession->GetOutputName(i, allocator);
        std::cout << "output " << i << " : " << "name = " << output_name << std::endl;
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = m_OrtSession->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output shapes/dims
        std::vector<int64_t> outputNode_dims = tensor_info.GetShape();
        std::cout << "output " << i << " : " << "num_dims = " << outputNode_dims.size() << std::endl;
        for (int j = 0; j < outputNode_dims.size(); j++)
        {
            printf("output %d : dim %d=%jd\n", i, j, outputNode_dims[j]);
        }
        output_node_dims.push_back(outputNode_dims);

        int batch_size = 1;
        int output_size = (input_size/32)*(pow(2,i));
        if (m_Categories.size() > 0)
        {
            //NCHW
            std::vector<int64_t> output_shape = { batch_size, 3*(5+(int)m_Categories.size()), output_size, output_size };
            output_shapes.push_back(output_shape);
        }
        else
        {
            std::cerr << "m_Categories empty !!!" << '\n';
        }
    }

    return true;
}

bool Yolov3::setSession()
{
    if ((m_sModelOnnxPath.length() == 0) || (m_OrtEnv == nullptr))
    {
        std::cerr << "m_sModelOnnxPath is null or m_OrtEnv nullptr !!" << '\n';
        return false;
    }
    m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_sModelOnnxPath.c_str(), m_OrtSessionOptions));
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
                m_Isyolov3tiny = Isyolov3tiny;
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

                if (!setSession())
                {
                    std::cerr << "setSession() failed!!" << '\n';
                    return false;
                }

                //model input output
                if (!setOnnxRuntimeModelInputOutput(input_size))
                {
                    std::cerr << "setOnnxRuntimeModelInputOutput() failed!!" << '\n';
                    return false;
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
        img.convertTo(img, CV_32FC3, 1.f / 255.0);

        int input_tensor_size = 1;
        for (auto it : input_node_dims)
        {
            for (auto dims : it)
            {
                input_tensor_size *= dims;
                //std::cout << "dims: " << dims << std::endl;
            }
        }
        input_image_.resize(input_tensor_size);

        float *data = input_image_.data();
        fill(input_image_.begin(), input_image_.end(), 0.f);

        for (int c = 0; c < 3; c++) {
            for (int w = 0; w < m_iInput_w; w++) {
                for (int h = 0; h < m_iInput_h; h++) {
                    data[c*m_iInput_w*m_iInput_h + w*m_iInput_h + h] = (img.ptr<float>(w)[h*3+c]);
                }
            }
        }

        // //HWC TO CHW
        // vector<Mat> input_channels(c);
        // cv::split(img_float, input_channels);

        // vector<float> result(h * w * c);
        // auto data = result.data();
        // int channelLength = h * w;
        // for (int i = 0; i < c; ++i) {
        //     memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        //     data += channelLength;
        // }

        // std::cout << "outputPicData: " << std::endl;
        // for (int i = 0; i < 608; i++)
        // {
        //     std::cout << data[i + 608] << " ";
        // }
        // std::cout << '\n';

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

        output_tensors = m_OrtSession->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tenser, input_node_names.size(), \
                                                    output_node_names.data(), output_node_names.size());
        std::cout << "output_tensors size: " << output_tensors.size() << std::endl;

        //todo .....

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

void Yolov3::DoNms(vector<DetectionRes>& detections)
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

	sort(detections.begin(), detections.end(), [=](const DetectionRes & left, const DetectionRes & right) {
		return left.prob > right.prob;
	});

	vector<DetectionRes> result;
	for (unsigned int m = 0; m < detections.size(); ++m) {
		result.push_back(detections[m]);
		for (unsigned int n = m + 1; n < detections.size(); ++n) {
			if (iouCompute((float *)(&detections[m]), (float *)(&detections[n])) > m_Nmsthreshold) {
				detections.erase(detections.begin() + n);
				--n;
			}
		}
	}
	detections = move(result);
}

bool Yolov3::postProcessing()
{
    if (m_bCheckPre && m_bInit && m_bCheckInit)
    {
        yolo_masks_.clear();
        yolo_anchors_.clear();
        if (m_Isyolov3tiny)
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
            printf("output_dims shape: ");
            for(auto it : output_dims)
            {
                printf("  %ld  ", it);
            }
            printf("\n");

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

                        for (int cp = 5; cp < shape[3]; cp++)
                        {
                            ptr[cp] = sigmoid(ptr[cp]);
                        }

                        //5-84 box_class_probs
                        // float class_score = max(ptr[4] * ptr[5], ptr[4] * ptr[6]);
                        // for (int cp = 7; cp < shape[3]; cp++)
                        // {
                        //     class_score = max(class_score, ptr[4] * ptr[cp]);
                        // }
                        int maxclassPos = 0;
                        float class_score = ptr[4] * ptr[5];
                        for (int cp = 5; cp < shape[3]; cp++)
                        {
                            if (class_score < ptr[4] * ptr[cp])
                            {
                                class_score = ptr[4] * ptr[cp];
                                maxclassPos = cp - 5;
                            }
                        }

                        // std::vector<float> classporb{ptr+4, ptr+shape[3]};
                        // auto maxValue = std::max_element(classporb.begin(), classporb.end());
                        // auto maxclassPos = std::distance(std::begin(classporb), maxValue);

                        if (class_score < m_threshold)
                            continue;

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

bool Yolov3::DrowBoxes(cv::Mat &inputImg)
{
    for (auto box : detections)
    {
        int x = box.x,
            y = box.y,
            w = box.w,
            h = box.h,
            clspos = box.classpos;
        std::cout << "x, y, w, h, clspos: " << box.x << " " << box.y << " " << box.w << " " << box.h << " " << box.classpos << std::endl;
        std::cout << "prob: " << box.prob << std::endl;
        cv::Rect rect = { x, y, w, h };
        cv::rectangle(inputImg, rect, cv::Scalar(0, 0, 255), 2);
        char boxProb[64] = "";
        sprintf(boxProb, "%.2f", box.prob);
        std::string category= m_Categories[box.classpos] + ":" + boxProb;
        std::cout << "category: " << category << std::endl;
        cv::putText(inputImg,
                    category.c_str(),
                    cv::Point(box.x-3, box.y-6),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.4,
                    CV_RGB(0, 0, 255),
                    1);
    }

    cv::imwrite("./prdiect.jpg", inputImg);

    return true;
}