from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import tkinter as tk
from tkinter import scrolledtext, filedialog
from PIL import Image, ImageTk
import moviepy.editor as mp
import base64
import threading
import pyperclip
import io
import json
import speech_recognition as sr
import random

import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Check if we need to use transformers
use_transformers = config.get('use_transformers', False)

# Initialize models based on the config
if use_transformers:
    model = AutoModel.from_pretrained(config['model_paths']['intern_model_path'], torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(config['model_paths']['intern_tokenizer_path'], trust_remote_code=True)
    llama_model = None
    chat_handler = None
    print("Transformer model loaded.")
else:
    chat_handler = Llava15ChatHandler(clip_model_path=config['model_paths']['clip_model_path'])
    print("CLIP model loaded.")
    
    llama_model_path = config['model_paths']['llama_model_path']
    llama_model = Llama(
        model_path=llama_model_path,
        chat_handler=chat_handler,
        n_ctx=config['llama_settings']['n_ctx'],
        temperature=config['llama_settings']['temperature'],
    )

    text_model_path = config['model_paths']['text_model_path']
    text_model = None
    if text_model_path:
        text_model = Llama(
            model_path=text_model_path,
            n_ctx=config['llama_settings']['n_ctx'],
            temperature=config['llama_settings']['temperature'],
        )
        print("Text model loaded.")

# Proxy mode flag
proxy_mode = config.get('proxy', False)
print(f"Proxy mode {'enabled' if proxy_mode else 'disabled'}.")

# Video frame count from config
video_frame_count = config.get('video_frame_count', 10)

# Initialize messages list
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant that can analyze images and videos. Provide a detailed analysis of images and videos when provided and asked.'}
]

# Function to append messages to the chat window
def append_message(role, message_dict):
    chat_history.config(state=tk.NORMAL)
    if role == "user":
        if isinstance(message_dict["content"], list):
            chat_history.insert(tk.END, f'User: {message_dict["content"][0]["text"]}\n', 'user_message')
            for index, item in enumerate(message_dict["content"]):
                if item["type"] == "image_url":
                    img_data = base64.b64decode(item["image_url"]["url"].split(",")[1])
                    img = Image.open(io.BytesIO(img_data))
                    img.thumbnail((100, 100))
                    img_tk = ImageTk.PhotoImage(img)
                    
                    img_frame = tk.Frame(chat_history, bd=2, relief=tk.RIDGE)
                    img_label = tk.Label(img_frame, image=img_tk)
                    img_label.image = img_tk
                    img_label.pack()

                    del_button = tk.Button(img_frame, text='X', command=lambda m=message_dict, idx=index, il=img_frame: remove_image(m, idx, il),
                                           bg='#808080', fg='white', font=('Arial', 8), bd=0, relief=tk.FLAT, padx=2, pady=2)
                    del_button.place(relx=1.0, rely=0.0, anchor='ne')
                    del_button.config(highlightbackground="#808080", borderwidth=0, relief=tk.FLAT)

                    chat_history.window_create(tk.END, window=img_frame)    
        else:
            chat_history.insert(tk.END, f'User: {message_dict["content"]}\n', 'user_message')

    else:
        chat_history.insert(tk.END, f'Assistant: {message_dict["content"]}\n', 'assistant_message')

    chat_history.insert(tk.END, "\n")
    chat_history.config(state=tk.DISABLED)
    chat_history.yview(tk.END)

def remove_image(message_dict, img_index, img_frame):
    img_frame.destroy()
    del message_dict["content"][img_index]
    clear_uploaded_media()

def send_message(event=None):
    user_message = user_input.get("1.0", tk.END).strip()
    copied_image = pyperclip.paste().strip()

    if user_message or copied_image.startswith("data:") or uploaded_media_datas:
        message_content = [{"type": "text", "text": user_message}]

        if copied_image.startswith("data:"):
            message_content.append({"type": "image_url", "image_url": {"url": copied_image}})
        for media_data in uploaded_media_datas:
            message_content.append(media_data)
        
        message = {"role": "user", "content": message_content}

        append_message("user", message)
        messages.append(message)
        user_input.delete("1.0", tk.END)

        clear_uploaded_media()

        threading.Thread(target=get_response, args=(message,)).start()

uploaded_media_datas = []

def get_response(user_message):
    if proxy_mode:
        for item in user_message["content"]:
            if item["type"] == "image_url":
                description = generate_image_description(item["image_url"]["url"])
                item["type"] = "text"
                item["text"] = description

    response_message = transformers_response(user_message) if use_transformers else llama_response(user_message)
    
    append_message("assistant", {"role": "assistant", "content": response_message})
    messages.append({'role': 'assistant', 'content': response_message})

def generate_image_description(image_url):
    description_prompt = {
        'role': 'system', 
        'content': 'You are an image analysis model; describe the following image in great detail.'
    }
    image_content = {
        'role': 'user', 
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }

    description_messages = [description_prompt, image_content]
    description_response = llama_model.create_chat_completion(description_messages)
    return description_response["choices"][0]["message"]["content"]

def transformers_response(user_message):
    for item in user_message["content"]:
        if item["type"] == "text":
            query = item["text"]
            break

    # Remove any image-related prompts since it is a text-only transformers model usage
    query = query.replace('<ImageHere>', '')
    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=query, image=None, history=[], do_sample=False, num_beams=3)
    return response

def llama_response(user_message):
    model_to_use = text_model if text_model else llama_model
    response = model_to_use.create_chat_completion(messages)
    return response["choices"][0]["message"]["content"]

def clear_uploaded_media():
    global uploaded_media_datas
    for widget in uploaded_media_frame.winfo_children():
        widget.destroy()
    uploaded_media_datas = []

def select_media():
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image Files", "*.png;*.jpg;*.jpeg"),
        ("Video Files", "*.mp4;*.avi;*.mov")
    ])
    if file_path:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(file_path, "rb") as f:
                img_data = f.read()
            
            encoded_img = base64.b64encode(img_data).decode()
            mime_type = "image/png" if file_path.lower().endswith("png") else "image/jpeg"
            uploaded_media_data = {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_img}"}}
            
            uploaded_media_datas.append(uploaded_media_data)

            img = Image.open(io.BytesIO(base64.b64decode(encoded_img)))
            img.thumbnail((50, 50))
            img_tk = ImageTk.PhotoImage(img)

            thumbnail_frame = tk.Frame(uploaded_media_frame, bd=1, relief=tk.RIDGE)
            thumbnail_label = tk.Label(thumbnail_frame, image=img_tk)
            thumbnail_label.image = img_tk
            thumbnail_label.pack()
            thumbnail_frame.pack(side=tk.LEFT, padx=5, pady=5)

        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            video = mp.VideoFileClip(file_path)
            frame_count = int(video.fps * video.duration)
            frame_interval = max(1, frame_count // video_frame_count)
            frames = []

            for i in range(video_frame_count):
                frame_time = i * frame_interval / video.fps
                frame = video.get_frame(frame_time)
                
                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                encoded_frame = base64.b64encode(buffer.getvalue()).decode()
                
                mime_type = 'image/jpeg'
                video_frame_data = {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_frame}"}}
                frames.append(video_frame_data)

            # Select a random frame as the preview thumbnail
            preview_frame = random.choice(frames)
            uploaded_media_datas.extend(frames)
            uploaded_media_datas.append({"type": "text", "text": "Above are frames from a video the user uploaded."})

            img_data = base64.b64decode(preview_frame["image_url"]["url"].split(",")[1])
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail((50, 50))
            img_tk = ImageTk.PhotoImage(img)

            thumbnail_frame = tk.Frame(uploaded_media_frame, bd=1, relief=tk.RIDGE)
            thumbnail_label = tk.Label(thumbnail_frame, image=img_tk)
            thumbnail_label.image = img_tk
            thumbnail_label.pack()
            thumbnail_frame.pack(side=tk.LEFT, padx=5, pady=5)

            del_button = tk.Button(thumbnail_frame, text='X', command=lambda: remove_uploaded_media(thumbnail_frame),
                                   bg='#808080', fg='white', font=('Arial', 8), bd=0, relief=tk.FLAT, padx=2, pady=2)
            del_button.place(relx=1.0, rely=0.0, anchor='ne')
            del_button.config(highlightbackground="#808080", borderwidth=0, relief=tk.FLAT)

            # Extract audio and perform STT if audio track exists
            if video.audio:
                audio_file_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_file_path)
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = recognizer.record(source)
                    transcription = recognizer.recognize_google(audio_data)
                    print(f"Transcription:\n{transcription}")
                    uploaded_media_datas.append({"type": "text", "text": f"Transcription of the audio: {transcription}"})

def remove_uploaded_media(thumbnail_frame):
    thumbnail_frame.destroy()
    clear_uploaded_media()

root = tk.Tk()
root.title("MultiChat")
root.configure(bg='#1e1e1e')
root.geometry("1000x600")

chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, bg='#2e2e2e', fg='white', font=('Arial', 10), bd=0, relief=tk.FLAT)
chat_history.tag_config('user_message', foreground='cyan')
chat_history.tag_config('assistant_message', foreground='light green')
chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

user_input_frame = tk.Frame(root, bg='#1e1e1e')
user_input_frame.pack(fill=tk.X, padx=10, pady=10)

uploaded_media_frame = tk.Frame(user_input_frame, bg='#1e1e1e')
uploaded_media_frame.pack(side=tk.TOP, fill=tk.X, anchor='w')

user_input = tk.Text(user_input_frame, height=3, bg='#2e2e2e', fg='white', font=('Arial', 10), bd=0, relief=tk.FLAT, insertbackground='white')
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
user_input.bind("<Return>", send_message)

send_button = tk.Button(user_input_frame, text="Send", command=send_message, bg='#4CAF50', fg='white', bd=0, font=('Arial', 10), padx=10, pady=5, relief=tk.FLAT)
send_button.pack(side=tk.RIGHT, padx=(5, 10))
send_button.config(highlightbackground="#4CAF50", borderwidth=0, relief=tk.FLAT)

upload_button = tk.Button(user_input_frame, text="Add Media", command=select_media, bg='#4CAF50', fg='white', bd=0, font=('Arial', 10), padx=10, pady=5, relief=tk.FLAT)
upload_button.pack(side=tk.RIGHT, padx=(5, 0))
upload_button.config(highlightbackground="#4CAF50", borderwidth=0, relief=tk.FLAT)

root.tk.call('tk', 'scaling', 2.0)
root.option_add('*Font', 'Helvetica 10')

root.mainloop()
