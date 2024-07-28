#include "encoderdecoder_infer.h"

int FindMax(float* din ,int len)
	{
		int i;
        // int len=din.size();
		float max_val = -INFINITY;
		int max_idx = -1;
		for (i = 0; i < len; i++) {
			if (din[i] > max_val) {
				max_val = din[i];
				max_idx = i;
			}
		}
        return max_idx;
	}


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

// 读取模型文件
std::vector<char> readModelFile(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::vector<char> model((size_t)file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(model.data(), model.size());
    file.close();
    return model;
}

int T5Inference::Init(
    const std::string &enginePath,const std::string &enginePathDec, const std::string &spiece_model,const int maxBatchSize, const int seqLength)
{
    gLogInfo << "--------------------\n";
    gLogInfo << "Using T5 inference C++\n";

    gLogInfo << "--------------------\n";
    int ret = initLibNvInferPlugins(&gLogger, ""); //// 初始化插件 ,"" 表示初始化所有插件通常在构建引擎
    if (ret)
    {
        gLogInfo << "初始化插件成功\n";
    }

    gLogInfo << "Loading T5 Inference Engine ... \n";
    std::vector<char> enc_bytes=readModelFile(enginePath.c_str());
    mRuntime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
    if (mRuntime == nullptr)
    {
        gLogError << "Error creating TRT mRuntime\n";
        exit(-1);
        return -1;
    }

    mEngine = TrtUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(enc_bytes.data(), enc_bytes.size()));
    if (mEngine == nullptr)
    {
        gLogError << "Error deserializing CUDA engine\n";
        exit(-1);
        return -1;
    }

    std::vector<char> dec_bytes=readModelFile(enginePathDec.c_str());
    mEngineDec = TrtUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(dec_bytes.data(), dec_bytes.size()));
    if (mEngineDec == nullptr)
    {
        gLogError << "Error deserializing CUDA engine\n";
        exit(-1);
        return -1;
    }
    gLogInfo << "Done\n";
    // getNbIOTensors() 函数返回引擎中 IO 张量的总数
    gLogInfo << "Enc 张量的总数:"<<mEngine->getNbIOTensors()<<"\n";
    gLogInfo << "Dec 张量的总数:"<<mEngineDec->getNbIOTensors()<<"\n";
    mContext = TrtUniquePtr<IExecutionContext>(mEngine->createExecutionContext());
    mContextDec = TrtUniquePtr<IExecutionContext>(mEngineDec->createExecutionContext());
    if (!mContext)
    {
        gLogError << "Error creating execution context\n";
        exit(-1);
        return -1;
    }
    if (!mContextDec)
    {
        gLogError << "Error creating execution mContextDec\n";
        exit(-1);
        return -1;
    }
    //预分配显存
    allocateBindings(maxBatchSize); //
    allocateBindingsDec(maxBatchSize); //
    tokenizer_->InitTokenizer(spiece_model);
    return 0;
}

void T5Inference::allocateBindings(const int maxBatchSize)
{
    const size_t input_ids_item=mSeqLength * maxBatchSize;
    const size_t input_ids_size =  input_ids_item * sizeof(int64_t) ;
    void *devBuf;
    gpuErrChk(cudaMalloc(&devBuf, input_ids_size));
    gpuErrChk(cudaMemset(devBuf, 0, input_ids_size));
    mDeviceBuffers.emplace_back(devBuf);
    gLogInfo << "enc nuInputItems.size():"<<input_ids_item<<",mOutputSize:"<<input_ids_size<<"\n";
 

    const size_t hidden_states_item = maxBatchSize * mSeqLength * HIDDEN_NUM;
    const size_t hidden_states_size = hidden_states_item * sizeof(float);
    gLogInfo << "enc numOutputItems.size():"<<hidden_states_item<<",mOutputSize:"<<hidden_states_size<<"\n";
 
    void *devBuf2;
    gpuErrChk(cudaMalloc(&devBuf2, hidden_states_size));
    gpuErrChk(cudaMemset(devBuf2, 0, hidden_states_size));
    mDeviceBuffers.emplace_back(devBuf2);
    mHostOutput.resize(hidden_states_item);   
}

void T5Inference::allocateBindingsDec(const int maxBatchSize)
{
    const size_t input_ids_item=mSeqLength * maxBatchSize;
    const size_t input_ids_size =  input_ids_item * sizeof(int64_t) ;
    const size_t encoder_hidden_states_item=mSeqLength * maxBatchSize*HIDDEN_NUM;
    const size_t encoder_hidden_states_size =  encoder_hidden_states_item * sizeof(float) ;

    void *devBuf;
    gpuErrChk(cudaMalloc(&devBuf, input_ids_size));
    gpuErrChk(cudaMemset(devBuf, 0, input_ids_size));
    mDeviceBuffersDec.emplace_back(devBuf);  

    void *devBuf1;
    gpuErrChk(cudaMalloc(&devBuf1, encoder_hidden_states_size));
    gpuErrChk(cudaMemset(devBuf1, 0, encoder_hidden_states_size));
    mDeviceBuffersDec.emplace_back(devBuf1);  

    const size_t hidden_states_item = maxBatchSize * mSeqLength * VOCAB_SIZE;
    const size_t hidden_states_size = hidden_states_item * sizeof(float); 
    void *devBuf2;
    gpuErrChk(cudaMalloc(&devBuf2, hidden_states_size));
    gpuErrChk(cudaMemset(devBuf2, 0, hidden_states_size));
    mDeviceBuffersDec.emplace_back(devBuf2);    

}


void T5Inference::InferT5(std::vector<int64_t> input_ids)
{
    // std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
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

std::vector<int64_t> T5Inference::PreProcessing(const std::string &text)
{
    std::vector<int32_t> tokens_32 = tokenizer_->Encode(text);
    std::vector<int64_t> tokens_64; 
    // 将vec32的内容转换为int64_t并插入到vec64中
    for (const auto& elem : tokens_32) {
        tokens_64.push_back(static_cast<int64_t>(elem));
    }
    int token_size=tokens_64.size();
    tokens_64.push_back(1);//结束符
    std::cout<<"tokens_64:"<<token_size<<std::endl;
    sequence_size_=token_size;
    for (auto token : tokens_64)
    {
        std::cout << token << "|";
    }
    std::cout << std::endl;
    return tokens_64;
}
std::string T5Inference::PostProcessing(const std::vector<int> result)
{
    std::string output = tokenizer_->Decode(result);

    return output;
}

std::string T5Inference::InferEncoderDecoder(std::vector<int64_t> input_ids)
{

    std::vector<float>encoder_outputs_prompt;
    void * b= InferEncoder(input_ids);
    cudaMemcpy(mDeviceBuffersDec[1], b, 1*18*768*sizeof(float), cudaMemcpyDeviceToDevice);
    std::vector<int64_t>generated_id={0};
    std::vector<int> result_logits;
    printf("<input_ids.size()=%d\n",input_ids.size());
    for(int i=0;i<input_ids.size();i++)
    {    
        std::vector<std::vector<std::vector<float>>> dec_reult=InferDecoder(generated_id,encoder_outputs_prompt);
        //  for(int j=0;j<5;j++)
        //  {
        //     std::cout<<"t5 out i="<<dec_reult[0][i][j]<<",";
        //  }
         std::cout<<std::endl;
         int next_token=FindMax(dec_reult[0][i].data(),VOCAB_SIZE);
        //  printf("FindMax,%d,",next_token);
         generated_id.push_back(next_token);
         result_logits.push_back(next_token);

    }
    printf("\n");
    printf("result==\n");
    for (int i=0;i<result_logits.size();i++)
    {
        printf("%d,",result_logits[i]);
    }
     printf("\n");
    std::string result_text= PostProcessing(result);
        printf("result_text=%s\n",result_text.c_str());
    return result_text;



}

// std::vector<float> T5Inference::InferEncoder(std::vector<int64_t> input_ids)
void* T5Inference::InferEncoder(std::vector<int64_t> input_ids)
{
    // std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    int input_size=input_ids.size()*sizeof(int64_t);
    gLogInfo << "input_ids.size():"<<input_ids.size()<<"\n";
    int batchSize=1; 
    mContext->setInputShape("input_ids", Dims2(batchSize,input_ids.size()));
    cudaMemcpy(mDeviceBuffers[0], input_ids.data(), input_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

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
    std::vector<float> output_vec(input_ids.size()*HIDDEN_NUM);
    int output_size=output_vec.size()*sizeof(float);
    gLogInfo <<"output_vec.size="<<output_vec.size()<< ",output_size: "<<output_size<< "...\n";
    cudaMemcpy(output_vec.data(), bindings[1], output_size, cudaMemcpyDeviceToHost);
    // for (int i=0;i<10;i++)
    // {
    //     // std::cout<<"output i="<<i<<","<<mHostOutput[i]<<std::endl;
    //     std::cout<<"enc output i="<<i<<","<<output_vec[i]<<std::endl;
    // }

    mTimes.push_back(times);
    // return output_vec;
    return bindings[1];

}


std::vector<std::vector<std::vector<float>>> T5Inference::InferDecoder(std::vector<int64_t> input_ids,std::vector<float> encoder_hidden_states)
{

    int const nIO = mEngineDec->getNbIOTensors();
    std::vector<const char *> tensorNameList(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        tensorNameList[i] = mEngineDec->getIOTensorName(i);
        TensorIOMode mode = mEngineDec->getTensorIOMode(tensorNameList[i]);
        gLogInfo<<" dec getIOTensorName="<<tensorNameList[i]<<"," << (mode == TensorIOMode::kINPUT ? "Input " : "Output")<<"\n";
    }


    Dims input_idsdims = mEngineDec->getTensorShape("input_ids");
    int32_t nbDims = input_idsdims.nbDims;
    //! The extent of each dimension.
    gLogInfo << "out inputids nbDims:" << nbDims << "\n";
    for (int m = 0; m < nbDims; m++)
    {
        gLogInfo << "out dim:" << m << ", size:" << input_idsdims.d[m] << "\n";
    }

     input_idsdims = mEngineDec->getTensorShape("encoder_hidden_states");
     nbDims = input_idsdims.nbDims;
    //! The extent of each dimension.
    gLogInfo << "out encoder_hidden_states nbDims:" << nbDims << "\n";
    for (int m = 0; m < nbDims; m++)
    {
        gLogInfo << "out dim:" << m << ", size:" << input_idsdims.d[m] << "\n";
    }

    // std::vector<int64_t> input_ids = {13959, 1566, 12, 2379, 10, 27, 47, 3, 9, 7584, 13, 3, 9, 939, 13, 10649, 5, 1};
    int input_size=input_ids.size()*sizeof(int64_t);
    int encoder_hidden_states_size=encoder_hidden_states.size()*sizeof(float);
    gLogInfo << "input_ids.size():"<<input_ids.size()<<",input_size:"<<input_size<<"\n";
    int batchSize=1; 
    mContextDec->setInputShape("input_ids", Dims2(batchSize,input_ids.size()));
    // int ndim2=encoder_hidden_states.size()/(HIDDEN_NUM*batchSize);
    int ndim2=18;
    gLogInfo << "dec .ndim2:"<<ndim2<<"\n";
    // mContextDec->setInputShape("encoder_hidden_states", Dims3(batchSize,ndim2,HIDDEN_NUM));
    mContextDec->setInputShape("encoder_hidden_states", Dims3(batchSize,18,HIDDEN_NUM));

    cudaMemcpy(mDeviceBuffersDec[0], input_ids.data(), input_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(mDeviceBuffersDec[1], encoder_hidden_states.data(), encoder_hidden_states_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));

    //查看hidden输入值
    std::vector<float> hidden_tmp(1*18*768);
    gLogInfo << "mInputSizes[0]: "<<1*18*768<< "...\n";
    cudaMemcpy(hidden_tmp.data(), mDeviceBuffersDec[1], 1*18*768*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0;i<10;i++)
    // {
    //     std::cout<<"hidden_tmp input i="<<i<<","<<hidden_tmp[i]<<std::endl;
    // } 
    std::vector<float> times;
    // 执行模型推理
     // 准备绑定
    void* bindings[3];
    bindings[0] = mDeviceBuffersDec[0];
    bindings[1] = mDeviceBuffersDec[1];
    bindings[2] = mDeviceBuffersDec[2];
    gLogInfo << "dec mDeviceBuffers.size()=" << mDeviceBuffersDec.size() << "...\n";
    bool status = mContextDec->executeV2(bindings); //executeV2同步，executeV3 异步
    gLogInfo <<"infer status:"<<status<<"\n";
    assert(status && "Inference failed");
    std::vector<float> output_vec(ndim2*VOCAB_SIZE);
    cudaMemcpy(output_vec.data(), bindings[2], ndim2*VOCAB_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0;i<5;i++)
    // {
    //     std::cout<<"dec out:"<<output_vec[i]<<std::endl;
    // }

    std::vector<int> dec_shape = {batchSize,(int)input_ids.size(),VOCAB_SIZE}; //{1,18}
    std::vector<std::vector<std::vector<float>>> dec_reult(batchSize, 
                                                             std::vector<std::vector<float>>((int)input_ids.size(),  
                                                                      std::vector<float>(VOCAB_SIZE, 0.0f)));
    
    for (int i = 0; i < dec_shape[1]; i++)
    {
        for (int j = 0; j < dec_shape[2]; j++)
        {
            dec_reult[0][i][j] = output_vec[i * dec_shape[2] + j];
        }
    }
    return dec_reult;
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

