import torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelWithLMHead
src_text = [
    '爱屋及乌。'
]

# model_zh2en = 'Helsinki-NLP/opus-mt-zh-en'
model_zh2en = 'opus-mt-zh-en'
tokenizer_zh2en = MarianTokenizer.from_pretrained(model_zh2en)
model = MarianMTModel.from_pretrained(model_zh2en)
inputs = tokenizer_zh2en(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
trans = [tokenizer_zh2en.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])

src_text = [
    'love me, love my dog.'
]
# model_en2zh = 'Helsinki-NLP/opus-mt-en-zh'
model_en2zh = 'opus-mt-en-zh'
tokenizer_en2zh = MarianTokenizer.from_pretrained(model_en2zh)
model = MarianMTModel.from_pretrained(model_en2zh)


inputs = tokenizer_en2zh(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
trans = [tokenizer_en2zh.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])
