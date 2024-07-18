#include "logging.h"
#include "common.h"
#include "iostream"
#include "encoderdecoder_infer.h"

std::shared_ptr<T5Inference> encoder_handle = nullptr;
int main()
{
    // const std::string enginePath="/root/autodl-tmp/t5-engine/t5-encoder-12.engine"; 
    const std::string enginePath="/home/wzp/t5-onnx/t5-encoder-12.engine"; 
    const int maxBatchSize=1;
    const int seqLength=100;
    const bool enableGraph = true;
    encoder_handle = std::make_shared<T5Inference>();
    int  ret=  encoder_handle->Init(enginePath, maxBatchSize,seqLength,enableGraph);
    gLogInfo << "encoder_model done!\n";
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