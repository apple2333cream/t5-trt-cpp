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
#include <cmath>
#include "sentencepiece_tokenizer.h"


#define MAX_SEQ_LENGTH 18

using namespace nvinfer1;

class T5Inference
{
public:
    T5Inference(){};
    ~T5Inference()
    {
        gLogInfo << "~T5Inference\n";
        // gpuErrChk(cudaStreamDestroy(mStream));
        for (auto &buf : mDeviceBuffers)
        {
            gpuErrChk(cudaFree(buf));
        }
        for (auto &buf : mDeviceBuffersDec)
        {
            gpuErrChk(cudaFree(buf));
        }
       
    };
    int Init(const std::string &enginePath,const std::string &enginePathDec, const std::string &spiece_model,const int maxBatchSize, const int seqLength);

    void allocateBindings(const int maxBatchSize);
    void allocateBindingsDec(const int maxBatchSize);
    void reportTiming(int batchIndex, int batchSize);
    void InferT5(std::vector<int64_t> inputs);
    std::string InferEncoderDecoder(std::vector<int64_t> input_ids);

    void* InferEncoder(std::vector<int64_t> inputs);
    std::vector<std::vector<std::vector<float>>> InferDecoder(std::vector<int64_t> input_ids,std::vector<float> encoder_hidden_states);
    std::vector<int64_t> PreProcessing(const std::string &text);
    std::string PostProcessing(const std::vector<int> result);
private:
   std::shared_ptr<SentencePieceTokenizer> tokenizer_ = std::make_shared<SentencePieceTokenizer>();
    static const int ENC_INPUT_NUM = 1; // input_ids
    static const int DEC_INPUT_NUM = 2; // input_ids,encoder_hidden_states
    static const int VOCAB_SIZE = 32128; 
    static const int HIDDEN_NUM = 768; 
    const int mSeqLength = MAX_SEQ_LENGTH;
    const bool mEnableGraph = true;
    TrtUniquePtr<IRuntime> mRuntime{nullptr};
    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<ICudaEngine> mEngineDec{nullptr};
    TrtUniquePtr<IExecutionContext> mContext{nullptr};
    TrtUniquePtr<IExecutionContext> mContextDec{nullptr};
    bool mEnableVariableLen = true; //是否变长
    std::vector<int> mCuSeqlens;
    // cudaStream_t mStream{NULL};
    std::vector<void *> mDeviceBuffers; //输入输出的EncoderGPU缓存
    std::vector<float> mHostOutput; //CPU输出存放
    std::vector<void *> mDeviceBuffersDec; //输入输出的EncoderGPU缓存
    std::vector<float> mHostOutputDec; //CPU输出存放
    std::vector<size_t> mInputSizes;
    size_t mOutputSize = 1*MAX_SEQ_LENGTH*768;
    std::vector<int> mOutputDims;
    std::vector<std::vector<float>> mTimes;
    cudaGraphExec_t mExecGraph;
};