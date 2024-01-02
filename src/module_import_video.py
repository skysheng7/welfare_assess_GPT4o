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

# generate descrition of 1 frame/image ended with base64
text_prompt = "Describe what's in the image"
describe_img(client, text_prompt, base64_image=extracted_frames[0], max_tokens=300, detail_level="low")

# generate descrition of the video (list of frames)
text_prompt = "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video."
describe_video(client, text_prompt, base64_frames=extracted_frames, max_tokens=500, detail_level="low")



            