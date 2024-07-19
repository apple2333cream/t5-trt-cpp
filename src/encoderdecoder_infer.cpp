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

    gpuErrChk(cudaStreamCreate(&mStream));

    allocateBindings(maxBatchSize); //?
    return 0;
}

void T5Inference::allocateBindings(const int maxBatchSize)
{
    const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int32_t);
    // const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int64_t);

    // Static sizes with implicit batch size: allocation sizes known to engine
    if (mEnableVariableLen)
    {
 
        const size_t allocationSizes[] = {allocationSize};
        for (int i = 0; i < sizeof(allocationSizes) / sizeof(allocationSizes[0]); i++)
        {
            void *devBuf;
            gpuErrChk(cudaMalloc(&devBuf, allocationSizes[i]));
            gpuErrChk(cudaMemset(devBuf, 0, allocationSizes[i]));
            mDeviceBuffers.emplace_back(devBuf);
            mInputSizes.emplace_back(allocationSizes[i]);
        }
    }
   

    const size_t numOutputItems = maxBatchSize * mSeqLength * 768;
    mOutputSize = numOutputItems * sizeof(float);
    if (mEnableVariableLen)
    {
        mOutputDims = {maxBatchSize,mSeqLength,768};
    }   
    void *devBuf;
    gpuErrChk(cudaMalloc(&devBuf, mOutputSize));
    gpuErrChk(cudaMemset(devBuf, 0, mOutputSize));
    mDeviceBuffers.emplace_back(devBuf);
    mHostOutput.resize(numOutputItems);


    void * tt;
    std::vector<float> a {1, 2};
    std::cout << "len a: " << a.size() <<std::endl;
    for(auto i : a)
        std::cout << i << std::endl;
    std::vector<float> b;
    const size_t len = a.size() * sizeof(float);

    cudaMalloc(&tt, len);
    cudaMemset(tt, 0, len);
    cudaMemcpy(tt, a.data(), len, cudaMemcpyHostToDevice);
    cudaMemcpy(b.data(), tt, len, cudaMemcpyDeviceToHost);

    std::cout << "len b: " << b.size() <<std::endl;
    for(auto i : b)
        std::cout << i << std::endl;

}

// void T5Inference::InferT5(const void *const *inputBuffers)
void T5Inference::InferT5(std::vector<int> inputs)
{
     std::vector<int32_t> vec = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    int input_size=vec.size()*sizeof(int32_t);
    // prepare(1, 1);
     int numProfiles = mEngine->getNbOptimizationProfiles();
    gLogInfo << "numProfiles:" << numProfiles << "\n";
    int profIdx =0;
    int batchSize=1;
    mContext->setOptimizationProfileAsync(profIdx, mStream);
    // nvinfer1::Dims engineDims = mEngine->getInput(0).desc.dims; // 获取输入张量的原始维度

    if (mEnableVariableLen)
    {
        const int allocationSizes[] = {mSeqLength * batchSize}; // input_ids
        for (int i = 0; i < kBERT_INPUT_NUM; i++)
        {
            auto const tensorName = mEngine->getIOTensorName(i);
            gLogInfo << "i:" << i << ",inputtensorName:" << tensorName << "\n";
            Dims inputdims = mEngine->getTensorShape(tensorName);
            int32_t nbDims = inputdims.nbDims;
            //! The extent of each dimension.
            int64_t d[8];
            gLogInfo << "nbDims:" << nbDims << "\n";
            for (int m = 0; m < nbDims; m++)
            {
                gLogInfo << "dim:" << m << ", size:" << inputdims.d[m] << "\n";
            }  
            // inputdims.d[1]=mSeqLength;
            // mContext->setInputShape(tensorName, inputdims);
            // Dims2
             mContext->setInputShape(tensorName, Dims2(batchSize,vec.size()));
        }
    }
   
    // gpuErrChk(
    //         cudaMemcpyAsync(mDeviceBuffers[0], vec.data(), mInputSizes[0], cudaMemcpyHostToDevice, mStream));
  
    cudaMemcpy(mDeviceBuffers[0], vec.data(), input_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    //查看输入值
    std::vector<int32_t> input_vec;
    gLogInfo << "mInputSizes[0]: "<<input_size<< "...\n";
    // cudaMemcpyAsync(input_vec.data(), mDeviceBuffers[0], mInputSizes[0], cudaMemcpyDeviceToHost, mStream);
    cudaMemcpy(input_vec.data(), mDeviceBuffers[0], input_size, cudaMemcpyDeviceToHost);
    // copyToHost(input_vec.data(),mDeviceBuffers[0],mInputSizes[0]);
     gLogInfo << "input_vec size: "<<input_vec.size()<< "...\n";
    for (int i=0;i<input_vec.size();i++)
    {
        std::cout<<"input i="<<i<<","<<input_vec[i]<<std::endl;
    }
 
    std::vector<float> times;
    int iterations = 1;
    gLogInfo << "Running " << iterations << " iterations ...\n";
      // 获取输入和输出绑定的数量

    // 执行模型推理
    gLogInfo << "mDeviceBuffers.size() " << mDeviceBuffers.size() << "...\n";
    bool status = mContext->executeV2(mDeviceBuffers.data()); //executeV2同步，executeV3 异步
    gLogInfo <<"infer status:"<<status<<"\n";
    assert(status && "Inference failed");

    // cudaMemcpy(  mHostOutput.data(), mDeviceBuffers[1], mOutputSize, cudaMemcpyDeviceToHost);
    // gpuErrChk(cudaMemcpyAsync(
    //     mHostOutput.data(), mDeviceBuffers[1], mOutputSize, cudaMemcpyDeviceToHost, mStream));

    // gpuErrChk(cudaStreamSynchronize(mStream));
    for (int i=0;i<10;i++)
    {
        std::cout<<"i="<<i<<","<<mHostOutput[i]<<std::endl;
    }

    mTimes.push_back(times);

 
    // for (int it = 0; it < iterations; it++)
    // {
    //     gpuErrChk(cudaEventRecord(start, mStream));
    //     if (mEnableGraph)
    //     {
    //         gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
    //     }
    //     else
    //     {
    //         bool status = mContext->enqueueV3(mStream);
    //         if (!status)
    //         {
    //             gLogError << "Enqueue failed\n";
    //             exit(-1);
    //         }
    //     }
    //     gpuErrChk(cudaEventRecord(stop, mStream));
    //     gpuErrChk(cudaStreamSynchronize(mStream));
    //     float time;
    //     gpuErrChk(cudaEventElapsedTime(&time, start, stop));
    //     times.push_back(time);
    // }

    // gpuErrChk(cudaMemcpyAsync(
    //     mHostOutput.data(), mDeviceBuffers[1], mOutputSize, cudaMemcpyDeviceToHost, mStream));

    // gpuErrChk(cudaStreamSynchronize(mStream));

    // mTimes.push_back(times);
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

