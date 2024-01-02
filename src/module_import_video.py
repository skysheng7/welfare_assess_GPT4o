from dotenv import load_dotenv
import os
from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import numpy as np

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Research/PhD Project/Amazon project phase 2/Kay Yang/sorted_cow_videos_all'
good_videos, bad_videos, bad_videos_by_category = get_video_paths(root_folder_path)

# read in videos and seperate into frames
video_path = bad_videos_by_category["approach"][0]
extracted_frames = extract_frames(video_path, 1)

# display the frames extracted
show_extracted_frames(extracted_frames)

# prompt GPT-4V
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
            "type": "text",
            #"text": "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video."
            "text": "decribe what's in the image"
            },
            {
            "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, extracted_frames[0]",
                    "detail": "low"
                }
            }
            #*map(lambda x: {"image": x, "resize": 512, "detail": "low"}, extracted_frames[0::3]),
        ],
    },
]
params = {
    #"model": "gpt-4-1106-vision-preview",
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 300,
}




# Modify the prompt to use only the first frame
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "This is the first frame from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            {"image": extracted_frames[0], "detail": "low"}
        ],
    },
]

# Setup the parameters for the API call
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 300,
}





result = client.chat.completions.create(**params)
print(result.choices[0].message.content)


# convert the response Text To Speech
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    },
    json={
        "model": "tts-1-1106",
        "input": result.choices[0].message.content,
        "voice": "onyx",
    },
)

audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk
Audio(audio)






