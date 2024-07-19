echo "convert encoder model……"
# encoder_model=/root/autodl-tmp/t5-onnx/t5-encoder-12.onnx
# encoder_engine=/root/autodl-tmp/t5-engine/t5-encoder-12.engine

# decoder_model=/root/autodl-tmp/t5-onnx/t5-decoder-with-lm-head-12.onnx
# decoder_engine=/root/autodl-tmp/t5-engine/t5-decoder-with-lm-head-12.engine

encoder_model=/home/wzp/t5-onnx/t5-encoder-12.onnx
encoder_engine=/home/wzp/t5-onnx/t5-encoder-12.engine

decoder_model=/home/wzp/t5-onnx/t5-decoder-with-lm-head-12.onnx
decoder_engine=/home/wzp/t5-onnx/t5-decoder-with-lm-head-12.engine


trtexec --onnx=${encoder_model} --saveEngine=${encoder_engine} --fp16 
echo "convert encoder done!"
echo "convert decoder model……"
trtexec --onnx=${decoder_model} --saveEngine=${decoder_engine} --fp16 
echo "convert decoder done!"


# trtexec --onnx=/home/wzp/t5-onnx/t5_base_beam_search.onnx --saveEngine=/home/wzp/t5-onnx/t5_base_beam_search.onnx --fp16  --verbose
# trtexec --onnx=/home/wzp/t5-onnx/t5-encoder-12.onnx --saveEngine=/home/wzp/t5-onnx/t5-encoder-12.onnx --fp16  --verbose

trtexec \
    --onnx=/home/wzp/t5-onnx/t5-encoder-12.onnx \
    --saveEngine=/home/wzp/t5-onnx/t5-encoder-12.engine \
    --minShapes=input_ids:1x1 \
    --optShapes=input_ids:1x20 \
    --maxShapes=input_ids:1x100 \
    --fp16 \
    --verbose >convert_encoder_log.txt