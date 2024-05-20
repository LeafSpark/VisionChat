# VisionChat
Simple Python GUI for llama.cpp multimodal models 

## Setup Guide

Download main.py and create a new folder named models. Download your vision model with the CLIP encoder, then set config values below:
```
{
    "model_paths": {
        "clip_model_path": "models/mmproj-model-f16.gguf",
        "llama_model_path": "models/ggml-model-f16.gguf",
        "text_model_path": "",
		    "intern_model_path": "models/internlm-xcomposer2-4khd-7b",
        "intern_tokenizer_path": "models/internlm-xcomposer2-4khd-7b"
    },
    "llama_settings": {
        "temperature": 0.7,
        "n_ctx": 4096
    },
    "proxy": false,
	"video_frame_count": 2,
	"use_transformers": false
}
```
**Explanation:**
- proxy (true/false, send to image model to analyze then send description to text_model_path, performs better on non vision operations but degraded performance on vision tasks)
- video_frame_count (int, maximum based on model context size, splits videos into frames evenly to pass to image models, also provides transcription)
- use_transformers (true/false, use transformers only models (set intern_model_path and intern_tokenizer_path))
