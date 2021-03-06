import torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelWithLMHead



## translate Chinese to English
src_text = [
    '爱屋及乌。'
]

# model_zh2en = 'Helsinki-NLP/opus-mt-zh-en'
model_name = 'opus-mt-zh-en'  # pre-download from https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/Helsinki-NLP/
tokenizer_zh2en = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
inputs = tokenizer_zh2en(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
trans = [tokenizer_zh2en.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])


## translate English to Chinese
src_text = [
    'love me, love my dog.'
]
# model_en2zh = 'Helsinki-NLP/opus-mt-en-zh'
model_name = 'opus-mt-en-zh' # pre-download from https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/Helsinki-NLP/
tokenizer_en2zh = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


inputs = tokenizer_en2zh(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
trans = [tokenizer_en2zh.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])
