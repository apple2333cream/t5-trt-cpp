
## t5-trt-cpp
本项目是在英伟达的GPU显卡环境对谷歌的t5模型用tensorrt C++的api进行推理加速，本项目包含了推理demo和http的API服务，支持并发、支持linux/win 
若需要CPU环境的加速，请移步至另外一个子项目https://github.com/apple2333cream/t5-ort-cpp.git  
原始模型仓库：https://huggingface.co/google-t5/t5-base   
### 步骤   
#### 1.环境准备
      tensorrt 10.0.1 (理论上8.9.6以上版本即可)
      cudnn 8.9.4
      cuda 12.4
      python3.10
      pip install  -r HuggingFace/requirements.txt
#### 2.模型转换   
        2.1 huggingface->onnx   
        2.2 onnx->tensrrt engine   
        cd ./HuggingFace
        bash gen_t5_bs1_beam2.sh   
        说明，encoder和decoder分开转换，若合成一个模型导出在转trt时需要手写BeamSearch（下个版本中会导出一个模型进行推理）
  
#### 3.代码编译  
  mkdir build && cd build 
  make  -j8  

#### 4.运行示例  
    - demo ./t5_engine --use_mode=0
    - test ./t5_engine --use_mode=1
    - api ./t5_engine --use_mode=2
     服务请求示例： 
    curl -X POST -d "{ "RequestID": "65423221", "InputText": "translate English to French: I was a victim of a series of accidents." }" http://127.0.0.1:17653/T5/register  

#### 代码目录结构
    ├── CMakeLists.txt
    ├── HuggingFace  转换模型代码，参考 (https://github.com/kshitizgupta21/triton-trt-oss.git)
    ├── main.cpp
    ├── onnx2tensorrt.sh
    ├── readme.md
    ├── src 
    ├── third_party

#### benchmark
    以t5-base为例
    GPU显存占用 1.3Gb  
    平均推理耗时184ms 
    测试环境 V100 ,tensorrt 10.0.1,cudnn8.9.4, cuda12.4

#### TODO List 


#### 📣更新日志

- 20240728 v1.0.0 update:
  - 提交t5 tensorrt C++ api推理代码

#### Contact Author
qq:807876904 

### 参考
[triton-trt-oss]https://github.com/kshitizgupta21/triton-trt-oss.git  
[t5-ort-cpp]https://github.com/apple2333cream/t5-ort-cpp.git  
https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/t5  