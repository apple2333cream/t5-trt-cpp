#include "logging.h"
#include "common.h"
#include "iostream"
#include "encoderdecoder_infer.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cmath>
#include <iomanip>

std::shared_ptr<T5Inference> encoder_handle = nullptr;
int main()
{
    // const std::string enginePath="/root/autodl-tmp/t5-engine/t5-encoder-12.engine"; 
    // const std::string enginePath="/home/wzp/t5-onnx/t5-encoder-12.engine"; 
    // const std::string enginePath="/home/wzp/t5-onnx/t5-engine/t5-base-encoder-12.onnx.engine"; 
    const std::string enginePath="/home/wzp/t5-onnx/t5-base-encoder-12.engine"; 
    const int maxBatchSize=1;
    const int seqLength=100;
    const bool enableGraph = true;
    encoder_handle = std::make_shared<T5Inference>();
    int  ret=  encoder_handle->Init(enginePath, maxBatchSize,seqLength,enableGraph);
    gLogInfo << "encoder_model done!"<<enginePath <<"\n";
    std::vector<int> input_ids;
    input_ids.push_back(32);
    input_ids.push_back(2);
    input_ids.push_back(128);
    input_ids.push_back(86);
    // 获取vector的原始数据指针，转换为const void*
    const void* rawPtr = static_cast<const void*>(input_ids.data());
    encoder_handle->InferT5(input_ids);
 
    int x=1;
    std::cout << "x="<<x<<"\n";
    gLogInfo << "ret="<<ret<<"\n";   

}

// Get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    case DataType::kUINT8:
        return std::string("UINT8");
    case DataType::kFP8:
        return std::string("FP8  ");
    case DataType::kINT64:
        return std::string("INT64");
    default:
        return std::string("Unknown");
    }
}

// Get the string of a TensorRT shape
std::string shapeToString(Dims64 dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

// Print data in the array
template<typename T>
void printArrayRecursion(const T *pArray, Dims64 dim, int iDim, int iStart)
{
    if (iDim == dim.nbDims - 1)
    {
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << double(pArray[iStart + i]) << " ";
        }
    }
    else
    {
        int nElement = 1;
        for (int i = iDim + 1; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }
        for (int i = 0; i < dim.d[iDim]; ++i)
        {
            printArrayRecursion<T>(pArray, dim, iDim + 1, iStart + i * nElement);
        }
    }
    std::cout << std::endl;
    return;
}
// Get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    case DataType::kINT64:
        return 8;
    default:
        return 4;
    }
}

template<typename T>
void printArrayInformation(
    T const *const     pArray,
    std::string const &name,
    Dims64 const      &dim,
    bool const         bPrintInformation,
    bool const         bPrintArray,
    int const          n)
{
    // Print shape information
    //int nElement = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>());
    std::cout << std::endl;
    std::cout << name << ": " << typeid(T).name() << ", " << shapeToString(dim) << std::endl;

    // Print statistic information of the array
    if (bPrintInformation)
    {
        int nElement = 1; // number of elements with batch dimension
        for (int i = 0; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }

        double sum      = double(pArray[0]);
        double absSum   = double(fabs(double(pArray[0])));
        double sum2     = double(pArray[0]) * double(pArray[0]);
        double diff     = 0.0;
        double maxValue = double(pArray[0]);
        double minValue = double(pArray[0]);
        for (int i = 1; i < nElement; ++i)
        {
            sum += double(pArray[i]);
            absSum += double(fabs(double(pArray[i])));
            sum2 += double(pArray[i]) * double(pArray[i]);
            maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
            minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
            diff += abs(double(pArray[i]) - double(pArray[i - 1]));
        }
        double mean = sum / nElement;
        double var  = sum2 / nElement - mean * mean;

        std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
        std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
        std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
        std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
        std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
        std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
        std::cout << std::endl;

        // print first n element and last n element
        for (int i = 0; i < n; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
        for (int i = nElement - n; i < nElement; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
    }

    // print the data of the array
    if (bPrintArray)
    {
        printArrayRecursion<T>(pArray, dim, 0, 0);
    }

    return;
}
template void printArrayInformation(float const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
// template void printArrayInformation(half const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(char const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(bool const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int8_t const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);
template void printArrayInformation(int64_t const *const, std::string const &, Dims64 const &, bool sondt, bool const, int const);

// int  main()
// {
//     cudaSetDevice(0);
//     // const std::string enginePath {"/home/wzp/t5-onnx/t5-encoder-12.engine"};
//     const std::string enginePath {"/home/wzp/t5-onnx/t5-engine/t5-base-encoder-12.onnx.engine"};
//     const char       *inputTensorName {"input_ids"};
//     Dims64            shape {3, {3, 4, 5}};
//     static Logger     gLogger(ILogger::Severity::kINFO);

//     IRuntime    *runtime {createInferRuntime(gLogger)};
//     ICudaEngine *engine {nullptr};

//     std::ifstream input(enginePath, std::ios::binary);
//     if (!input)
//     {
//         gLogError << "Error opening engine file: " << enginePath << "\n";
//         exit(-1);
//         return -1;
//     }

//     input.seekg(0, input.end);
//     const size_t fsize = input.tellg();
//     input.seekg(0, input.beg);

//     std::vector<char> bytes(fsize);
//     input.read(bytes.data(), fsize);

//     engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size());

//     if (engine == nullptr)
//     {
//         std::cout << "Fail getting engine for inference" << std::endl;
//         return -1;
//     }
//     std::cout << "Succeed getting engine for inference" << std::endl;

//     int const                 nIO = engine->getNbIOTensors();
//     std::cout <<"nIO="<<nIO<<std::endl;
//     std::vector<const char *> tensorNameList(nIO);
//     for (int i = 0; i < nIO; ++i)
//     {
//         tensorNameList[i] = engine->getIOTensorName(i);
//     }

//     IExecutionContext *context = engine->createExecutionContext();
//     context->setInputShape(inputTensorName, Dims2(1,18));

//     for (auto const name : tensorNameList)
//     {
    
//         TensorIOMode mode = engine->getTensorIOMode(name);
//         std::cout << (mode == TensorIOMode::kINPUT ? "Input " : "Output");
//         std::cout << "-> ";
//         std::cout << dataTypeToString(engine->getTensorDataType(name)) << ", ";
//         std::cout << shapeToString(engine->getTensorShape(name)) << ", ";
//         std::cout << shapeToString(context->getTensorShape(name)) << ", ";
//         std::cout << name << std::endl;
//     }

//     std::map<std::string, std::tuple<void *, void *, int>> bufferMap;
//     for (auto const name : tensorNameList)
//     {
//         Dims64 dim {context->getTensorShape(name)};
//         int    nByte        = std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<>()) * dataTypeToSize(engine->getTensorDataType(name));
//         void  *hostBuffer   = (void *)new char[nByte];
//         void  *deviceBuffer = nullptr;
//         cudaMalloc(&deviceBuffer, nByte);
//         bufferMap[name] = std::make_tuple(hostBuffer, deviceBuffer, nByte);
//     }

//     // float *pInputData = static_cast<float *>(std::get<0>(bufferMap[inputTensorName])); // We certainly know the data type of input tensors
//     // for (int i = 0; i < std::get<2>(bufferMap[inputTensorName]) / sizeof(float); ++i)
//     // {
//     //     pInputData[i] = float(i);
//     // }
//     std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
//        // 将 vector 转换为指向其数据的指针
//     void *dataPtr = reinterpret_cast<void*>(input_ids.data());

//         // 获取 vector 的大小，用于元组的第三个元素
//     int dataSize = static_cast<int>(input_ids.size());
//   // 获取元组的引用
//     auto& inputIdsTuple = bufferMap["input_ids"];

//     // 解构元组以访问其元素
//     void *&ptr = std::get<0>(inputIdsTuple);
//     void *dummyPtr = std::get<1>(inputIdsTuple); // 第二个元素，保持不变
//     int &size = std::get<2>(inputIdsTuple); // 第三个元素，更新为新 vector 的大小

//     // 更新第一个元素和第三个元素
//     ptr = dataPtr;
//     size = static_cast<int>(input_ids.size());

//     // 打印确认
//     std::cout << "Updated data pointer: " << ptr << ", Size: " << size << std::endl;
//     //  // 将 void * 指针转换为 int32_t *
//     // int32_t *int32Ptr = reinterpret_cast<int32_t *>(ptr);

//     // // 打印数据
//     // std::cout << "Data values: ";
//     // for (int i = 0; i < size; ++i) {
//     //     std::cout << int32Ptr[i] << ' ';
//     // }
//     // std::cout << std::endl;


//     for (auto const name : tensorNameList)
//     {
//         context->setTensorAddress(name, std::get<1>(bufferMap[name]));
//     }

//     for (auto const name : tensorNameList)
//     {
//         if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
//         {
//             void *hostBuffer   = std::get<0>(bufferMap[name]);
//             void *deviceBuffer = std::get<1>(bufferMap[name]);
//             int   nByte        = std::get<2>(bufferMap[name]);
//             cudaMemcpy(deviceBuffer, hostBuffer, nByte, cudaMemcpyHostToDevice);
//         }
//     }

//     bool status=context->enqueueV3(0);
//     std::cout<<"status:"<<status<<std::endl;

//     for (auto const name : tensorNameList)
//     {
//         if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT)
//         {
//             void *hostBuffer   = std::get<0>(bufferMap[name]);
//             void *deviceBuffer = std::get<1>(bufferMap[name]);
//             int   nByte        = std::get<2>(bufferMap[name]);
//             cudaMemcpy(hostBuffer, deviceBuffer, nByte, cudaMemcpyDeviceToHost);
//         }
//     }
//       // 获取元组的引用
//     auto& OutPutTuple = bufferMap["hidden_states"];
//         // 解构元组以访问其元素
//     void *&outptr = std::get<0>(OutPutTuple);
//     void *outdummyPtr = std::get<1>(OutPutTuple); // 第二个元素，保持不变
//     int &outsize = std::get<2>(OutPutTuple); // 第三个元素，更新为新 vector 的大小

//      //  // 将 void * 指针转换为 int32_t *
//     float *floatPtr = reinterpret_cast<float *>(outptr);

//     // 打印数据
//     std::cout << "output values: ";
//     for (int i = 0; i < 10; ++i) {
//         std::cout << floatPtr[i] << ',';
//     }
//     std::cout << std::endl;



//     for (auto const name : tensorNameList)
//     {
//         void *hostBuffer = std::get<0>(bufferMap[name]);
//         // printArrayInformation(static_cast<float *>(hostBuffer), name, context->getTensorShape(name), false, true);
//     }

//     for (auto const name : tensorNameList)
//     {
//         void *hostBuffer   = std::get<0>(bufferMap[name]);
//         void *deviceBuffer = std::get<1>(bufferMap[name]);
//         delete[] static_cast<char *>(hostBuffer);
//         cudaFree(deviceBuffer);
//     }
//     return -1;


// }
