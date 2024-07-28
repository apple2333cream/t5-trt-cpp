import time
import torch
import onnx 
import numpy  as  np
from onnxruntime import InferenceSession
from transformers import T5Tokenizer,T5ForConditionalGeneration

pretrained_model = '/home/wzp/t5-onnx/t5-base' # This can be a pretrained version, or the path to a huggingface model
text = "translate English to French: I was a victim of a series of accidents."
tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
input_ids = tokenizer.encode(text, return_tensors='pt') 
# generative_t5 = GenerativeT5(simplified_encoder, decoder_with_lm_head, tokenizer ,onnx=False, cuda=True)
test_num=20
tic=time.time()
for i  in range(test_num):   
    # Output: "Je suis victime d'une série d'accidents."
    summary_ids = model.generate(input_ids)
    # 解码生成的结果
    decoded_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
toc=time.time()
print("result:",decoded_output) # J'ai été victime d'une série d'accidents.  "Je suis victime d'une série d'accidents."
print("avg time={:.4f}s".format((toc-tic)/test_num)) #1.046s torch  cpu
