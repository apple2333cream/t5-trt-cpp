1.将onnx格式模型转换成tensorrt
bash  onnx2tensorrt.sh
root@autodl-container-c7cb4299a4-5cc322d8:~/autodl-tmp/t5-onnx# tree -h
[  83]  .
├── [620M]  t5-decoder-with-lm-head-12.onnx
└── [418M]  t5-encoder-12.onnx

root@autodl-container-c7cb4299a4-5cc322d8:~/autodl-tmp/t5-engine# tree -h
[  87]  .
├── [264M]  t5-decoder-with-lm-head-12.engine
└── [210M]  t5-encoder-12.engine