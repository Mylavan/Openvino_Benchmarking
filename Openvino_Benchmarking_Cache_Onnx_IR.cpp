// Openvino_Benchmarking_Cache_Onnx_IR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <Windows.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>

#include <openvino/openvino.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

#define IM_SEGMENTATION_WIDTH            896    // default width of segmented frame
#define IM_SEGMENTATION_HEIGHT            640 // default height of segmented frame
#define IM_SEGMENTATION_CHANNEL            50  // max number of channel with segmented frame
using namespace cv;
using namespace std;
using namespace std::chrono;
int  PaddingTop = 0;
int PaddingBottom = 0;
int PaddingLeft = 0;
int PaddingRight = 0;
int Original_Input_Height = 0;
int Original_Input_Width = 0;
int outCroppedWidth = 0;
int outCroppedHeight = 0;
int outCroppedOriginY = 0;
int outCroppedOriginX = 0;
enum EFetalHeartViewClasss
{
    INVALID_VIEW_ID = -1,
    FOUR_CHAMBER = 0,
    LVOT = 1,
    RVOT = 2,
    THREE_VT = 3,
    THREE_VV = 4,
    OTHER_HEART = 5,  // use this as m_MAX_VIEW_ID similar to MAX_CLASS_ID as used by Fetal Biometry 
    NON_HEART = 6
};
vector <string> ImageNames;

Mat loadImageandPreProcess(const std::string& filename, int sizeX = IM_SEGMENTATION_WIDTH, int sizeY = IM_SEGMENTATION_HEIGHT)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "No image found.";
    }
    int OriginalInputImageWidth = image.size().width;
    int OriginalInputImageHight = image.size().height;

    //closeup crop calculation
    cv::Rect rect = cv::boundingRect(image);

    outCroppedOriginX = rect.x;
    outCroppedOriginY = rect.y;
    outCroppedWidth = rect.width;
    outCroppedHeight = rect.height;

    cv::Mat croppedImage = image(rect);
    cv::Mat fCroppedImage;
    croppedImage.convertTo(fCroppedImage, CV_32FC1);


    //mean and standard deviation calculation
    cv::Scalar mean, stddev;
    cv::meanStdDev(fCroppedImage, mean, stddev);

    double dMean = mean[0];
    double dStdDev = stddev[0];



    //normalize image pxiel values using image mean & standard deviation
    // using formula : (img � img.mean() / img.std())
    fCroppedImage = (fCroppedImage - dMean) / (dStdDev + 1e-8);



    //old code 
    //cv::Mat fCroppedImageResized;
    //cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), cv::INTER_NEAREST);



    //New Code 
    int hinput = fCroppedImage.size().height;
    int winput = fCroppedImage.size().width;
    float  aspectRatio = 0;
    int Target_Height = IM_SEGMENTATION_HEIGHT;
    int Target_Width = IM_SEGMENTATION_WIDTH;
    int Resized_Height = 0;
    int Resized_Width = 0;
    //Equal 
    if (winput < hinput)
    {
        aspectRatio = (float)winput / hinput;
        std::cout << aspectRatio << std::endl;
        Resized_Height = Target_Height;
        Resized_Width = (float)aspectRatio * Resized_Height;
        if (Resized_Width > Target_Width)
        {
            Resized_Height = Resized_Height - ((Resized_Width - Target_Width) / aspectRatio);
            Resized_Width = aspectRatio * Resized_Height;
        }
    }
    else
    {
        aspectRatio = (float)hinput / winput;
        Resized_Width = Target_Width;
        Resized_Height = (float)(aspectRatio * Resized_Width);
        if (Resized_Height > Target_Height)
        {
            Resized_Width = Resized_Width - ((Resized_Height - Target_Height) / aspectRatio);
            Resized_Height = aspectRatio * Resized_Width;
        }
    }
    cv::Mat fCroppedImageResized;
    Original_Input_Height = OriginalInputImageHight;
    Original_Input_Width = OriginalInputImageWidth;
    cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(Resized_Width, Resized_Height), cv::INTER_NEAREST);

    int DiffWidth = Target_Width - Resized_Width;
    int DiffHeight = Target_Height - Resized_Height;
    PaddingTop = DiffHeight / 2;
    PaddingBottom = DiffHeight / 2 + DiffHeight % 2;
    PaddingLeft = DiffWidth / 2;
    PaddingRight = DiffWidth / 2 + DiffWidth % 2;


    Mat PaddedImage;
    copyMakeBorder(fCroppedImageResized, PaddedImage, PaddingTop, PaddingBottom, PaddingLeft, PaddingRight, BORDER_CONSTANT, 0);

    //std::vector<float> vec;
    //int cn = 1;//RGBA , 4 channel
    //int iCount = 0;

    //const int inputNumChannel = 1;
    //const int inputH = IM_SEGMENTATION_HEIGHT;
    //const int inputW = IM_SEGMENTATION_WIDTH;

    //std::vector<float> vecR;

    //vecR.resize(inputH * inputW);


    //for (int i = 0; i < inputH; i++)
    //{
    //    for (int j = 0; j < inputW; j++)
    //    {
    //        float pixelValue = PaddedImage.at<float>(i, j);
    //        vecR[iCount] = pixelValue;
    //        iCount++;
    //    }
    //}
    //vector<float> input_tensor_values;
    //for (auto i = vecR.begin(); i != vecR.end(); ++i)
    //{
    //    input_tensor_values.push_back(*i);
    //}
    ////return input_tensor_values;
   // imwrite("NewpaddedImage.png", PaddedImage);
    return PaddedImage;

}
int main()
{
    ov::Core core;

    std::chrono::time_point<std::chrono::high_resolution_clock> ModelLoadstart, ModelLoadend, InferenceTimestart, InferenceTimeend, TotalTimeforAllFramesStart, TotalTimeforAllFramesEnd, TimeforOneFrameStart, TimeforOneFrameEnd, PreProcessstart, PreProcessend;
    fstream ModelFile;


    ModelFile.open("./Extra/Dependency/Models/ModelCurrentlyUsed.txt", ios::in); //open a file to perform read operation using file object
    string modelname;
    if (ModelFile.is_open()) { //checking whether the file is open

        while (getline(ModelFile, modelname)) { //read data from file object and put it into string.
            cout << "Loaded Model ( from .txt file ) Name is ::  " << modelname << "\n"; //print the data of the string
            break;
        }
        ModelFile.close(); //close the file object.
    }
    std::cout << " modelname :: " << modelname << std::endl;
  
    string ModelPrefix = "./Extra/Dependency/Models/";
    string ModelPostfixbin = ".bin";
    string ModelPostfixXml = ".xml";
    string ModelPostfixOnnx = ".onnx";
    std::string model_path = ModelPrefix + modelname + ModelPostfixXml;
    std::string weights_path = ModelPrefix + modelname + ModelPostfixbin;
    std::string onnx_path = ModelPrefix + modelname + ModelPostfixOnnx;
    const char* filexml = model_path.data();
    const char* filebin = weights_path.data();
    const char* fileonnx = onnx_path.data();


    struct stat sb;
    //if ((stat(filexml, &sb) == 0 && !(sb.st_mode & S_IFDIR))&& (stat(filebin, &sb) == 0 && !(sb.st_mode & S_IFDIR)))
    //{
    //    cout << "The path is valid!" << std::endl;
    //    network = ie.ReadNetwork(model_path, weights_path);
    //    //network = ie.ReadNetwork(onnx_path);  
    //}
    std::shared_ptr<ov::Model> model;
    if ((stat(fileonnx, &sb) == 0 && !(sb.st_mode & S_IFDIR)) && (stat(fileonnx, &sb) == 0 && !(sb.st_mode & S_IFDIR)))
    {
        cout << "The path is valid!" << std::endl;
        //  network = ie.ReadNetwork(model_path, weights_path);  
        model = core.read_model(onnx_path);
    }
    else
    {
        cout << "The Path is invalid!" << std::endl;
        return 0;
    }
    // Print input node details
    std::cout << "Input Node(s) Information:" << std::endl;
    for (const auto& input : model->inputs()) {
        auto input_shape = input.get_shape();
        std::cout << "Name: " << input.get_any_name() << ", Shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "], Type: " << input.get_element_type() << std::endl;
    }

    ModelLoadstart = std::chrono::high_resolution_clock::now();
    // Load the network into the Inference Engine
    // Compile the model for a specific device
    std::string cacheDirPath = "./Extra/Dependency/Models/";
    if (!std::filesystem::exists(cacheDirPath)) {
        std::filesystem::create_directories(cacheDirPath);
    }
    // Prepare the configuration for model compilation with caching enabled
    std::map<std::string, std::string> config;
    config[ov::cache_dir.name()] = cacheDirPath;

    // Compile the model for a specific device, e.g., "CPU", with caching enabled
    core.set_property(ov::cache_dir("./Extra/Dependency/Models"));
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    

    //Model Inference without Caching 
   // ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ModelLoadend = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = ModelLoadend - ModelLoadstart;
    auto num_requests = compiled_model.get_property(ov::optimal_number_of_infer_requests);
    std::cout << "num_requests ::" << num_requests << std::endl;
    std::cout << "Model Loading Time  " << elapsed_seconds.count() << std::endl;
    std::cout << " This Happens Only once , So relax and take a seat " << std::endl;
    TotalTimeforAllFramesStart = std::chrono::high_resolution_clock::now();
    // Create an infer request
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    std::vector< ov::InferRequest> inferRequests;
    inferRequests.push_back(compiled_model.create_infer_request());
    vector<String> fn;
    string Image_Name;
    int counter = 0;
    EFetalHeartViewClasss inFetalHeartViewClasss;
    int a = 0;
    inFetalHeartViewClasss = EFetalHeartViewClasss(a);
    glob("./Extra/input/*.*", fn);
    for (auto f : fn)
    {
        inferRequests.push_back(compiled_model.create_infer_request());
        counter++;
        std::cout << "-------------------------------------------NEW FRAME PROCESSING-------------------------------------------" << std::endl;
        TimeforOneFrameStart = std::chrono::high_resolution_clock::now();
        // std::cout << f << std::endl;



        string str1 = "./Extra/input";



        // Find first occurrence of "geeks"
        size_t found = f.find(str1);
        /* std::cout << str1.size() << std::endl;*/
        string r = f.substr(str1.size() + 1, f.size());
        r.erase(r.length() - 4);
        // prints the result
        cout << "String is: " << r << std::endl;
        ImageNames.push_back(r);
        /* cout << "-------------------------------------" << std::endl;*/



        const std::string imageFile = f;




        PreProcessstart = std::chrono::high_resolution_clock::now();
        cv::Mat Originalimage = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
        if (Originalimage.empty()) {
            std::cout << "No image found.";
        }
        int OriginalInputImageWidth = Originalimage.size().width;
        int OriginalInputImageHight = Originalimage.size().height;
        Mat PaddedImage = loadImageandPreProcess(imageFile);

        InferenceTimestart = std::chrono::high_resolution_clock::now();
        auto input_shape = model->input().get_shape();
        ov::Tensor input_tensor = ov::Tensor(model->input().get_element_type(), input_shape);
        // Copy image data to the tensor
        std::memcpy(input_tensor.data(), PaddedImage.data, input_shape[2] * input_shape[3] * sizeof(float));
        infer_request.set_input_tensor(input_tensor);

      
        // Run inference
        infer_request.infer();
        const int inputH = IM_SEGMENTATION_HEIGHT;
        const int inputW = IM_SEGMENTATION_WIDTH;

        InferenceTimeend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = InferenceTimeend - InferenceTimestart;
        std::cout << " Inference Time Taken :: " << elapsed_seconds.count() << std::endl;

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        float* output_data = output_tensor.data<float>();

         size_t output_channels;
         size_t output_height;
         size_t output_width ;
        // Print output node details
        std::cout << "Output Node(s) Information:" << std::endl;
        for (const auto& output : model->outputs()) {
            auto output_shape = output.get_shape();
            std::cout << "Name: " << output.get_any_name() << ", Shape: [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cout << output_shape[i];
                output_channels= output_shape[1];
                output_height = output_shape[2];
                output_width = output_shape[3];
                if (i < output_shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "], Type: " << output.get_element_type() << std::endl;
        }

        //Incase you want to check the Output Dimensions
        std::cout << " output_channels :: " << output_channels << std::endl;
        std::cout << " output_height :: " << output_height << std::endl;
        std::cout << " output_width :: " << output_width << std::endl;

        const size_t output_size = output_channels * output_height * output_width;
        std::vector<float> output_vec(output_data, output_data + output_size);


        /////////////////////////////////////////////////////////////////

        int imgSize = IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT;
        unsigned char frameWithMaxPixelValueIndex[IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT];
        memset(frameWithMaxPixelValueIndex, 0, imgSize * sizeof(unsigned char));
        //#pragma omp parallel for
        for (int iPixelIndex = 0; iPixelIndex < imgSize; iPixelIndex++)
        {
            float pixelValue = 0;
            float pixelMaxValue = -INFINITY; //Initialzie max pixel value holder to negative INFINITY
            int   channelIndexWithMaxPixelValue = 0;
            for (int iChannelIndex = 0; iChannelIndex < output_channels; iChannelIndex++)
            {
                pixelValue = *(output_data + (iChannelIndex * imgSize + iPixelIndex));
                if (pixelMaxValue < pixelValue)
                {
                    pixelMaxValue = pixelValue;
                    channelIndexWithMaxPixelValue = iChannelIndex;
                }
            }



            frameWithMaxPixelValueIndex[iPixelIndex] = channelIndexWithMaxPixelValue;
        }


        cv::Mat cvframeWithMaxPixelValueIndex = cv::Mat(cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), CV_8UC1, frameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
        cv::imwrite("./Extra/Output_Mask/" + r + ".png", cvframeWithMaxPixelValueIndex);


        //Remove applied padding  back to preprocessed cropped dimension
        cv::Mat preprocess_cvframeWithMaxPixelValueIndex;
        preprocess_cvframeWithMaxPixelValueIndex = cvframeWithMaxPixelValueIndex(Range(PaddingTop, IM_SEGMENTATION_HEIGHT - PaddingBottom), Range(PaddingLeft, IM_SEGMENTATION_WIDTH - PaddingRight));
        // cv::imwrite("frameWithMaxPixelValueIndex_after_preprocess_cropped_resize.png", preprocess_cvframeWithMaxPixelValueIndex);




         //Resize back to Cropped Size 
        cv::Mat FinalResized;
        cv::resize(preprocess_cvframeWithMaxPixelValueIndex, FinalResized, cv::Size(outCroppedWidth, outCroppedHeight), 0, 0, cv::INTER_NEAREST);
        cv::Rect preprocess_cropp_rect;
        preprocess_cropp_rect.x = outCroppedOriginX;
        preprocess_cropp_rect.y = outCroppedOriginY;
        preprocess_cropp_rect.width = outCroppedWidth;
        preprocess_cropp_rect.height = outCroppedHeight;



        //Resize Back to original Input Dimensions 
        cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex = cv::Mat::zeros(cv::Size(Original_Input_Width, Original_Input_Height), CV_8UC1);
        FinalResized.copyTo(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex(preprocess_cropp_rect));
        //  cv::imwrite("frameWithMaxPixelValueIndex_after_maskapplied_orginputimgsize_resize.png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);
        cv::imwrite("./Extra/Final_Resized_mask/" + r + ".png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);

       
        cvframeWithMaxPixelValueIndex.release();
        TimeforOneFrameEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> outerelapsed_seconds = TimeforOneFrameEnd - TimeforOneFrameStart;
        std::cout << "********************* TIME TAKEN for 1 FRAME (seconds) :: " << outerelapsed_seconds.count() << "  ***************************" << std::endl;

    }
    // system("pause"); // <----------------------------------
    return 0;
}