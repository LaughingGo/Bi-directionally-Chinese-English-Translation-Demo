import torch
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelWithLMHead
src_text = [
    '爱屋及乌。'
#     '>>en<< 爱屋及乌。',
#     '>>zh<< love me, love my dog.'
]

# model_zh2en = 'Helsinki-NLP/opus-mt-zh-en'
model_zh2en = 'opus-mt-zh-en'
tokenizer_zh2en = MarianTokenizer.from_pretrained(model_zh2en)
model = MarianMTModel.from_pretrained(model_zh2en)
inputs = tokenizer_zh2en(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
# translate = model.generate(**tokenizer_zh2en.prepare_translation_batch(src_text))
# examples = ["Truth, good and beauty have always been considered as the three top pursuits of human beings",
#             "Warm wind caresses my face",
#             "Sang Lan is one of the best athletes in our country."]

# trans = tokenizer_zh2en.decode(translate, skip_special_tokens=True)
trans = [tokenizer_zh2en.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])

src_text = [
    '>>fr<< love me, love my dog.'
#     '>>en<< 爱屋及乌.',
#     '>>zh<< love me, love my dog.'
]
# model_en2zh = 'Helsinki-NLP/opus-mt-en-zh'
model_en2zh = 'opus-mt-en-zh'
tokenizer_en2zh = MarianTokenizer.from_pretrained(model_en2zh)
model = MarianMTModel.from_pretrained(model_en2zh)

# tokenizer_en2zh = AutoTokenizer.from_pretrained(model_en2zh)
# model = AutoModelWithLMHead.from_pretrained(model_en2zh)
inputs = tokenizer_en2zh(src_text, padding=True, return_tensors="pt")
translate = model.generate(**inputs, max_length=128)
# translate = model.generate(**tokenizer_en2zh.prepare_translation_batch(src_text))
trans = tokenizer_en2zh.decode(translate, skip_special_tokens=True)
trans = [tokenizer_en2zh.decode(ids, skip_special_tokens=True) for ids in translate]
print(trans[0])
