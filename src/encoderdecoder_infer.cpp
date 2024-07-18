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

    allocateBindings(maxBatchSize);
    return 0;
}

void T5Inference::allocateBindings(const int maxBatchSize)
{
    const size_t allocationSize = mSeqLength * maxBatchSize * sizeof(int32_t);

    // Static sizes with implicit batch size: allocation sizes known to engine
    if (mEnableVariableLen)
    {
        const size_t allocationSizes[] = {allocationSize, allocationSize,
                                          sizeof(int32_t) * (maxBatchSize + 1),
                                          sizeof(int32_t) * (mSeqLength)};
        for (int i = 0; i < sizeof(allocationSizes) / sizeof(allocationSizes[0]); i++)
        {
            void *devBuf;
            gpuErrChk(cudaMalloc(&devBuf, allocationSizes[i]));
            gpuErrChk(cudaMemset(devBuf, 0, allocationSizes[i]));
            mDeviceBuffers.emplace_back(devBuf);
            mInputSizes.emplace_back(allocationSizes[i]);
        }
    }
    else
    {
        for (int i = 0; i < kBERT_INPUT_NUM; i++)
        {
            void *devBuf;
            gpuErrChk(cudaMalloc(&devBuf, allocationSize));
            gpuErrChk(cudaMemset(devBuf, 0, allocationSize));
            mDeviceBuffers.emplace_back(devBuf);
            mInputSizes.emplace_back(allocationSize);
        }
    }

    const size_t numOutputItems = maxBatchSize * mSeqLength * 2;
    mOutputSize = numOutputItems * sizeof(float);
    if (mEnableVariableLen)
    {
        mOutputDims = {maxBatchSize * mSeqLength * 2};
    }
    else
    {
        mOutputDims = {maxBatchSize, mSeqLength, 2, 1, 1};
    }
    void *devBuf;
    gpuErrChk(cudaMalloc(&devBuf, mOutputSize));
    gpuErrChk(cudaMemset(devBuf, 0, mOutputSize));
    mDeviceBuffers.emplace_back(devBuf);
    mHostOutput.resize(numOutputItems);
}

void T5Inference::prepare(int profIdx, int batchSize)
{
    int numProfiles = mEngine->getNbOptimizationProfiles();
    gLogInfo << "numProfiles:" << numProfiles << "\n";
    profIdx = std::min(numProfiles - 1, profIdx);

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

            // int bindingIndex = mEngine->getBindingIndex(tensorName);
            // if (bindingIndex != -1) {
            //     nvinfer1::Dims engineDims = mEngine->getBindingDimensions(bindingIndex);
            //     // 现在可以使用 engineDims
            // } else {
            //     // 处理错误，输入张量名未找到
            // }
            mContext->setInputShape(tensorName, inputdims);
            // mContext->setInputShape(tensorName, Dims2(batchSize, mSeqLength));
        }
    }
    else
    {
        for (int i = 0; i < kBERT_INPUT_NUM; i++)
        {
            auto const tensorName = mEngine->getIOTensorName(i);
            mContext->setInputShape(tensorName, Dims2(batchSize, mSeqLength));
        }
    }

    if (!mContext->allInputDimensionsSpecified())
    {
        gLogError << "Not all input dimensions are specified for the exeuction context\n";
        exit(-1);
    }

    if (mEnableGraph)
    {
        for (int32_t i = 0; i < mEngine->getNbIOTensors(); i++)
        {
         
            auto const &name = mEngine->getIOTensorName(i);
            gLogInfo << "i:" << i << ",outtensorName:" << name << "\n";
            mContext->setTensorAddress(name, mDeviceBuffers[i]);
        }

        cudaGraph_t graph;
        cudaGraphExec_t exec;
        // warm up and let mContext do cublas initialization
        bool status = mContext->enqueueV3(mStream);
        if (!status)
        {
            gLogError << "Enqueue failed\n";
            exit(-1);
        }
        gLogVerbose << "Capturing graph\n";

        gpuErrChk(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
        status = mContext->enqueueV3(mStream);
        if (!status)
        {
            gLogError << "Enqueue failed\n";
            exit(-1);
        }

        gpuErrChk(cudaStreamEndCapture(mStream, &graph));
        gpuErrChk(cudaStreamSynchronize(mStream));

        gpuErrChk(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
        mExecGraph = exec;
    }
    mCuSeqlens.resize(batchSize + 1);
    std::generate(mCuSeqlens.begin(), mCuSeqlens.end(), [pos = -mSeqLength, this]() mutable
                  { pos += mSeqLength; return pos; });
}

void T5Inference::run(const void *const *inputBuffers, int warmUps, int iterations)
{
    for (int i = 0; i < kBERT_INPUT_NUM; i++)
    {
        gpuErrChk(
            cudaMemcpyAsync(mDeviceBuffers[i], inputBuffers[i], mInputSizes[i], cudaMemcpyHostToDevice, mStream));
    }

    gLogInfo << "Warming up " << warmUps << " iterations ...\n";
    for (int it = 0; it < warmUps; it++)
    {
        if (mEnableGraph)
        {
            gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
        }
        else
        {
            bool status = mContext->enqueueV3(mStream);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
        }
    }
    gpuErrChk(cudaStreamSynchronize(mStream));

    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    std::vector<float> times;
    gLogInfo << "Running " << iterations << " iterations ...\n";
    for (int it = 0; it < iterations; it++)
    {
        gpuErrChk(cudaEventRecord(start, mStream));
        if (mEnableGraph)
        {
            gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
        }
        else
        {
            bool status = mContext->enqueueV3(mStream);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
        }
        gpuErrChk(cudaEventRecord(stop, mStream));
        gpuErrChk(cudaStreamSynchronize(mStream));
        float time;
        gpuErrChk(cudaEventElapsedTime(&time, start, stop));
        times.push_back(time);
    }

    gpuErrChk(cudaMemcpyAsync(
        mHostOutput.data(), mDeviceBuffers[mEnableVariableLen ? kBERT_INPUT_NUM + 1 : kBERT_INPUT_NUM], mOutputSize, cudaMemcpyDeviceToHost, mStream));

    gpuErrChk(cudaStreamSynchronize(mStream));

    mTimes.push_back(times);
}

// void T5Inference::InferT5(const void *const *inputBuffers)
void T5Inference::InferT5(std::vector<int> inputs)
{
    prepare(1, 1);
    std::vector<int> vec = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    const void *inputIds = static_cast<const void *>(vec.data());
    const std::vector<const void *> inputBuffers = {inputIds, mCuSeqlens.data()};
    for (int i = 0; i < kBERT_INPUT_NUM; i++)
    {
        gpuErrChk(
            cudaMemcpyAsync(mDeviceBuffers[i], inputBuffers[i], mInputSizes[i], cudaMemcpyHostToDevice, mStream));
    }

    bool status = mContext->enqueueV3(mStream);
    if (!status)
    {
        gLogError << "Enqueue failed\n";
        exit(-1);
    }

    gpuErrChk(cudaStreamSynchronize(mStream));

    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    std::vector<float> times;
    int iterations = 1;
    gLogInfo << "Running " << iterations << " iterations ...\n";
    for (int it = 0; it < iterations; it++)
    {
        gpuErrChk(cudaEventRecord(start, mStream));
        if (mEnableGraph)
        {
            gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
        }
        else
        {
            bool status = mContext->enqueueV3(mStream);
            if (!status)
            {
                gLogError << "Enqueue failed\n";
                exit(-1);
            }
        }
        gpuErrChk(cudaEventRecord(stop, mStream));
        gpuErrChk(cudaStreamSynchronize(mStream));
        float time;
        gpuErrChk(cudaEventElapsedTime(&time, start, stop));
        times.push_back(time);
    }

    gpuErrChk(cudaMemcpyAsync(
        mHostOutput.data(), mDeviceBuffers[1], mOutputSize, cudaMemcpyDeviceToHost, mStream));

    gpuErrChk(cudaStreamSynchronize(mStream));

    mTimes.push_back(times);
}

// void T5Inference::RunT5(const void *inputIds )
// {
//     if (mEnableVariableLen)
//     {
//         const std::vector<const void *> inputBuffers = {inputIds ,mCuSeqlens.data()};
//         InferT5(inputBuffers.data());
//     }
//     else
//     {
//         const std::vector<const void *> inputBuffers = {inputIds};
//         InferT5(inputBuffers.data());
//     }
// }

void T5Inference::run(const void *inputIds, const void *segmentIds, const void *inputMask, int warmUps, int iterations)
{
    if (mEnableVariableLen)
    {
        const std::vector<const void *> inputBuffers = {inputIds, segmentIds, mCuSeqlens.data()};
        run(inputBuffers.data(), warmUps, iterations);
    }
    else
    {
        const std::vector<const void *> inputBuffers = {inputIds, segmentIds, inputMask};
        run(inputBuffers.data(), warmUps, iterations);
    }
}

void T5Inference::run(int profIdx, int batchSize, const void *inputIds, const void *segmentIds, const void *inputMask,
                      int warmUps, int iterations)
{

    prepare(profIdx, batchSize);
    run(inputIds, segmentIds, inputMask, warmUps, iterations);
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

