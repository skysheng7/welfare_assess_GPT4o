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
#text_prompt = "Describe what's in the image"
#describe_img(client, text_prompt, base64_image=extracted_frames[0], max_tokens=300, detail_level="low")

# generate descrition of the video (list of frames)
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for lameness assessment. \n Criteria for good video: Shows a single dairy cow walking smoothly in a straight line, entering from the leftmost side of the screen and exiting on the rightmost side, at a normal speed. \n Criteria for bad video, in 8 categories: [1] `direction` - cow moves from right to left, or in any direction contrary to that described for a good video [2] `stop` - cow pauses or sniffs the ground while walking, [3] `run` - cow runs or jogs, [4] `approach` - cow comes towards the camera, [5] `human` - excessive human interference or obstruction, [6] `slip` - cow slips while walking, [7] `multiple` - more than 1 cows in the video, [8] `other` - any other issue making the video hard for lameness assessment."
user_prompt = "Your job is to review cow videos (a series of frames), and classify them as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` \n Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: JSON \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"\n}"
result = describe_video_0shot(client, system_prompt, text_prompt=user_prompt, base64_frames=extracted_frames, max_tokens=500, detail_level="low", s=7, temp=0.5)



            