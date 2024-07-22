#include "encoderdecoder_infer.h"

// Helper function to allocate device memory
void *allocateDeviceMemory(size_t size)
{
    void *devPtr;
    cudaMalloc(&devPtr, size);
    return devPtr;
}

// Helper function to free device memory
void freeDeviceMemory(void *devPtr)
{
    cudaFree(devPtr);
}

// Helper function to copy data from host to device
void copyToDevice(void *dst, const void *src, size_t size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

// Helper function to copy data from device to host
void copyToHost(void *dst, const void *src, size_t size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

int T5Inference::Init(
    const std::string &enginePath, const int maxBatchSize, const int seqLength, const bool enableGraph)
// : mSeqLength(seqLength), mEnableGraph(enableGraph)
{
    gLogInfo << "--------------------\n";
    gLogInfo << "Using BERT inference C++\n";
    if (enableGraph)
    {
        gLogInfo << "CUDA Graph is enabled\n";
    }
    else
    {
        gLogInfo << "CUDA Graph is disabled\n";
    }
    gLogInfo << "--------------------\n";
    int ret = initLibNvInferPlugins(&gLogger, ""); //// 初始化插件 ,"" 表示初始化所有插件通常在构建引擎
    if (ret)
    {
        gLogInfo << "初始化插件成功\n";
    }

    gLogInfo << "Loading BERT Inference Engine ... \n";
    std::ifstream input(enginePath, std::ios::binary);
    if (!input)
    {
        gLogError << "Error opening engine file: " << enginePath << "\n";
        exit(-1);
        return -1;
    }

    input.seekg(0, input.end);
    const size_t fsize = input.tellg();
    input.seekg(0, input.beg);

    std::vector<char> bytes(fsize);
    input.read(bytes.data(), fsize);

    mRuntime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
    if (mRuntime == nullptr)
    {
        gLogError << "Error creating TRT mRuntime\n";
        exit(-1);
        return -1;
    }

    mEngine = TrtUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (mEngine == nullptr)
    {
        gLogError << "Error deserializing CUDA engine\n";
        exit(-1);
        return -1;
    }
    gLogInfo << "Done\n";
    // getNbIOTensors() 函数返回引擎中 IO 张量的总数
    mEnableVariableLen = mEngine->getNbIOTensors() == kBERT_INPUT_NUM + 1 ? false : true;
     gLogInfo << "IO 张量的总数:"<<mEngine->getNbIOTensors()<<"\n";
    mEnableVariableLen = true; // wzp-add
    if (mEnableVariableLen)
    {
        gLogInfo << "Variable length is enabled\n";
    }
    else
    {
        gLogInfo << "Variable length is disabled\n";
    }

    mContext = TrtUniquePtr<IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext)
    {
        gLogError << "Error creating execution context\n";
        exit(-1);
        return -1;
    }

    // gpuErrChk(cudaStreamCreate(&mStream));

    allocateBindings(maxBatchSize); //?
    return 0;
}

void T5Inference::allocateBindings(const int maxBatchSize)
{
    // const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int32_t);
    // const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int64_t);
    const size_t numInputItems =  mSeqLength * maxBatchSize  ;
    size_t mInputSize = numInputItems * sizeof(int64_t);

    // Static sizes with implicit batch size: allocation sizes known to engine
    if (mEnableVariableLen)//输入
    { 
        // const size_t allocationSizes[] = {allocationSize};
      
            void *devBuf;
            gpuErrChk(cudaMalloc(&devBuf, mInputSize));
            gpuErrChk(cudaMemset(devBuf, 0, mInputSize));
            mDeviceBuffers.emplace_back(devBuf);
            mInputSizes.emplace_back(mInputSize);
        
    }
    const size_t numOutputItems = maxBatchSize * mSeqLength * 768;
    mOutputSize = numOutputItems * sizeof(float);
    gLogInfo << "numOutputItems.size():"<<numOutputItems<<",mOutputSize:"<<mOutputSize<<"\n";
    if (mEnableVariableLen)
    {
        mOutputDims = {maxBatchSize,mSeqLength,768};
    }   
    void *devBuf;
    gpuErrChk(cudaMalloc(&devBuf, mOutputSize));
    gpuErrChk(cudaMemset(devBuf, 1, mOutputSize));
    mDeviceBuffers.emplace_back(devBuf);
    mHostOutput.resize(numOutputItems);   

}

void T5Inference::InferT5(std::vector<int> input_ids)
{
    std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    int input_size=input_ids.size()*sizeof(int64_t);
    gLogInfo << "input_ids.size():"<<input_ids.size()<<",input_size:"<<input_size<<"\n";
    int batchSize=1; 

    if (mEnableVariableLen)
    {
        const int allocationSizes[] = {mSeqLength * batchSize}; // input_ids
        for (int i = 0; i < 1; i++)
        {
            auto const tensorName = mEngine->getIOTensorName(i);
            gLogInfo << "i:" << i << ",inputtensorName:" << tensorName << "\n";
            Dims inputdims = mEngine->getTensorShape(tensorName);
            int32_t nbDims = inputdims.nbDims;     
            gLogInfo << "in nbDims:" << nbDims << "\n";
            for (int m = 0; m < nbDims; m++)
            {
                gLogInfo << "in dim:" << m << ", size:" << inputdims.d[m] << "\n";
            }  
            // inputdims.d[1]=mSeqLength;
            // mContext->setInputShape(tensorName, inputdims);
            // Dims2
             mContext->setInputShape(tensorName, Dims2(batchSize,input_ids.size()));
        }
    }
    //输出张量形状也要指定！！！
           Dims outputdims = mEngine->getTensorShape("hidden_states");
           int32_t nbDims = outputdims.nbDims;
            //! The extent of each dimension.
            gLogInfo << "out nbDims:" << nbDims << "\n";
            for (int m = 0; m < nbDims; m++)
            {
                gLogInfo << "out dim:" << m << ", size:" << outputdims.d[m] << "\n";
            }  

    cudaMemcpy(mDeviceBuffers[0], input_ids.data(), input_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    //查看输入值
    std::vector<int64_t> input_vec(input_ids.size());
    gLogInfo << "mInputSizes[0]: "<<input_size<< "...\n";
    cudaMemcpy(input_vec.data(), mDeviceBuffers[0], input_size, cudaMemcpyDeviceToHost);
     gLogInfo << "input_vec size: "<<input_vec.size()<< "...\n";
    for (int i=0;i<input_vec.size();i++)
    {
        std::cout<<"input i="<<i<<","<<input_vec[i]<<std::endl;
    } 
    std::vector<float> times;
    // 执行模型推理
     // 准备绑定
    void* bindings[2];
    bindings[0] = mDeviceBuffers[0];
    bindings[1] = mDeviceBuffers[1];
    gLogInfo << "mDeviceBuffers.size()=" << mDeviceBuffers.size() << "...\n";
    bool status = mContext->executeV2(bindings); //executeV2同步，executeV3 异步
    gLogInfo <<"infer status:"<<status<<"\n";
    assert(status && "Inference failed");
    std::vector<float> output_vec(input_ids.size()*768);
    int output_size=output_vec.size()*sizeof(float);
    gLogInfo <<"output_vec.size="<<output_vec.size()<< ",output_size: "<<output_size<< "...\n";
    cudaMemcpy(output_vec.data(), bindings[1], output_size, cudaMemcpyDeviceToHost);
    for (int i=0;i<10;i++)
    {
        // std::cout<<"output i="<<i<<","<<mHostOutput[i]<<std::endl;
        std::cout<<"output i="<<i<<","<<output_vec[i]<<std::endl;
    }

    mTimes.push_back(times);

}



void T5Inference::InferEncoder(std::vector<int> input_ids)
{
    std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    int input_size=input_ids.size()*sizeof(int64_t);
    gLogInfo << "input_ids.size():"<<input_ids.size()<<",input_size:"<<input_size<<"\n";
    int batchSize=1; 

    if (mEnableVariableLen)
    {
        const int allocationSizes[] = {mSeqLength * batchSize}; // input_ids
        for (int i = 0; i < 1; i++)
        {
            auto const tensorName = mEngine->getIOTensorName(i);
            gLogInfo << "i:" << i << ",inputtensorName:" << tensorName << "\n";
            Dims inputdims = mEngine->getTensorShape(tensorName);
            int32_t nbDims = inputdims.nbDims;     
            gLogInfo << "in nbDims:" << nbDims << "\n";
            for (int m = 0; m < nbDims; m++)
            {
                gLogInfo << "in dim:" << m << ", size:" << inputdims.d[m] << "\n";
            }  
            // inputdims.d[1]=mSeqLength;
            // mContext->setInputShape(tensorName, inputdims);
            // Dims2
             mContext->setInputShape(tensorName, Dims2(batchSize,input_ids.size()));
        }
    }
    //输出张量形状也要指定！！！
           Dims outputdims = mEngine->getTensorShape("hidden_states");
           int32_t nbDims = outputdims.nbDims;
            //! The extent of each dimension.
            gLogInfo << "out nbDims:" << nbDims << "\n";
            for (int m = 0; m < nbDims; m++)
            {
                gLogInfo << "out dim:" << m << ", size:" << outputdims.d[m] << "\n";
            }  

    cudaMemcpy(mDeviceBuffers[0], input_ids.data(), input_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    //查看输入值
    std::vector<int64_t> input_vec(input_ids.size());
    gLogInfo << "mInputSizes[0]: "<<input_size<< "...\n";
    cudaMemcpy(input_vec.data(), mDeviceBuffers[0], input_size, cudaMemcpyDeviceToHost);
     gLogInfo << "input_vec size: "<<input_vec.size()<< "...\n";
    for (int i=0;i<input_vec.size();i++)
    {
        std::cout<<"input i="<<i<<","<<input_vec[i]<<std::endl;
    } 
    std::vector<float> times;
    // 执行模型推理
     // 准备绑定
    void* bindings[2];
    bindings[0] = mDeviceBuffers[0];
    bindings[1] = mDeviceBuffers[1];
    gLogInfo << "mDeviceBuffers.size()=" << mDeviceBuffers.size() << "...\n";
    bool status = mContext->executeV2(bindings); //executeV2同步，executeV3 异步
    gLogInfo <<"infer status:"<<status<<"\n";
    assert(status && "Inference failed");
    std::vector<float> output_vec(input_ids.size()*768);
    int output_size=output_vec.size()*sizeof(float);
    gLogInfo <<"output_vec.size="<<output_vec.size()<< ",output_size: "<<output_size<< "...\n";
    cudaMemcpy(output_vec.data(), bindings[1], output_size, cudaMemcpyDeviceToHost);
    for (int i=0;i<10;i++)
    {
        // std::cout<<"output i="<<i<<","<<mHostOutput[i]<<std::endl;
        std::cout<<"output i="<<i<<","<<output_vec[i]<<std::endl;
    }

    mTimes.push_back(times);

}


void T5Inference::reportTiming(int batchIndex, int batchSize)
{

    std::vector<float> &times = mTimes[batchIndex];
    const float totalTime = std::accumulate(times.begin(), times.end(), 0.0);
    const float avgTime = totalTime / times.size();

    sort(times.begin(), times.end());
    const float percentile95 = times[(int)((float)times.size() * 0.95)];
    const float percentile99 = times[(int)((float)times.size() * 0.99)];
    const int throughput = (int)((float)batchSize * (1000.0 / avgTime));
    gLogInfo << "Running " << times.size() << " iterations with Batch Size: " << batchSize << "\n";
    gLogInfo << "\tTotal Time: " << totalTime << " ms \n";
    gLogInfo << "\tAverage Time: " << avgTime << " ms\n";
    gLogInfo << "\t95th Percentile Time: " << percentile95 << " ms\n";
    gLogInfo << "\t99th Percentile Time: " << percentile99 << " ms\n";
    gLogInfo << "\tThroughput: " << throughput << " sentences/s\n";
}

