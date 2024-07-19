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

#define MAX_SEQ_LENGTH 100

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
    void reportTiming(int batchIndex, int batchSize);
    // void RunT5(const void *inputIds );
    void InferT5(std::vector<int> inputs);

private:
    static const int kBERT_INPUT_NUM = 1; // input_ids
    const int mSeqLength = MAX_SEQ_LENGTH;
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
    size_t mOutputSize = 1*MAX_SEQ_LENGTH*768;
    std::vector<int> mOutputDims;
    std::vector<std::vector<float>> mTimes;
    cudaGraphExec_t mExecGraph;
};