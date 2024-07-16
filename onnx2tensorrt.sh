echo "convert encoder model……"
encoder_model=/root/autodl-tmp/t5-onnx/t5-encoder-12.onnx
encoder_engine=/root/autodl-tmp/t5-engine/t5-encoder-12.engine

decoder_model=/root/autodl-tmp/t5-onnx/t5-decoder-with-lm-head-12.onnx
decoder_engine=/root/autodl-tmp/t5-engine/t5-decoder-with-lm-head-12.engine
trtexec --onnx=${encoder_model} --saveEngine=${encoder_engine} --fp16 
echo "convert encoder done!"
echo "convert decoder model……"
trtexec --onnx=${decoder_model} --saveEngine=${decoder_engine} --fp16 
echo "convert decoder done!"