#include "common.h"
#include "logging.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <string.h>
#include <vector>
#include <iostream>
// #include "t5_logger.h"

using namespace nvinfer1;

class T5Inference
{
public:
    T5Inference(){};
    ~T5Inference()
    {
        gLogInfo << "~T5Inference\n";
        gpuErrChk(cudaStreamDestroy(mStream));
        for (auto &buf : mDeviceBuffers)
        {
            gpuErrChk(cudaFree(buf));
        }
       
    };
    int Init(const std::string &enginePath, const int maxBatchSize, const int seqLength, const bool enableGraph);
    void allocateBindings(const int maxBatchSize);
    void prepare(int profIdx, int batchSize);
    void run(const void *const *inputBuffers, int warmUps, int iterations);
    void run(const void *inputIds, const void *segmentIds, const void *inputMask, int warmUps, int iterations);
    void run(int profIdx, int batchSize, const void *inputIds, const void *segmentIds, const void *inputMask,
             int warmUps, int iterations);
    void reportTiming(int batchIndex, int batchSize);
    // void RunT5(const void *inputIds );
    void InferT5(std::vector<int> inputs);

private:
    static const int kBERT_INPUT_NUM = 1; // input_ids
    const int mSeqLength = 20;
    const bool mEnableGraph = true;
    TrtUniquePtr<IRuntime> mRuntime{nullptr};
    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<IExecutionContext> mContext{nullptr};
    bool mEnableVariableLen = true; //是否变长
    std::vector<int> mCuSeqlens;
    cudaStream_t mStream{NULL};
    std::vector<void *> mDeviceBuffers; //输入输出的GPU缓存
    std::vector<float> mHostOutput; //CPU输出存放
    std::vector<size_t> mInputSizes;
    size_t mOutputSize = 1;
    std::vector<int> mOutputDims;
    std::vector<std::vector<float>> mTimes;
    cudaGraphExec_t mExecGraph;
};