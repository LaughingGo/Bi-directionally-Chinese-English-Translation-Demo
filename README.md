# Bi-directionally Chinese-English Translation Demo
This repo concisely demonstrates how to integrate a translation tool based on the [transformers](https://github.com/huggingface/transformershttps://github.com/huggingface/transformers "https://github.com/huggingface/transformers"). Here I'll take Chinese and English for examples, including translate Chinese to English and translate English to Chinese.

## Two steps you just need to do
* To modify the *model_name* according to your need. In my demo, the *model_name* are 'Helsinki-NLP/opus-mt-zh-en' and 'Helsinki-NLP/opus-mt-en-zh' respectively. While, alternatively, you could download the corresponding pretrained model from [Here](https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/Helsinki-NLP/ "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/Helsinki-NLP/") by offline, then directly load it like the demo.
* Just run the [demo](./demo.py "demo.py")!
