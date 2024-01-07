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
import pandas as pd
from datetime import datetime
import json
import random

# define result directory
results_folder = '../results'
#results_file = 'video_quality_assess_GPT4V_results.csv' # store the results in a csv file

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Research/PhD Project/Amazon project phase 2/Kay Yang/sorted_cow_videos_all'
good_videos, bad_videos, bad_videos_by_category = get_video_paths(root_folder_path)

# choose which type of video to load
choosen_quality = "good" # "good", "bad"
choosen_category = "NA." # 'approach', 'direction', 'human', 'run', 'slip', 'stop', 'two', 'NA.'
frames_per_second=4 # how many frames to extract each second

# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for lameness assessment. \n "
user_prompt = f"Your job is to review cow videos (a series of frames) frame by frame, and classify them as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` Although there are more descriptions and creteria related to `bad` videos, please avoid any predisposition towards labeling videos as `bad`. While evaluating the cow's speed, bear in mind that you are reviewing a sequence of frames extracted at {frames_per_second} frames per second, and not the original video. Make sure to view the series of frames in ascending numerical order, starting from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame.\n "
user_prompt = user_prompt + "Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: ```json \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"\n}```"
max_tokens=500
detail_level="low"
seed = 7
temperature = 0.5

# randomly shuffle the video sequence
random.seed(seed)
random.shuffle(good_videos)
random.shuffle(bad_videos)
for category_videos in bad_videos_by_category.values(): # Shuffle each sublist in bad_videos_by_category
    random.shuffle(category_videos)


########################## GPT-4V generate description for good & bad video #######################
# select a good video
video_path1 = select_video_path(1, choosen_quality = "good", choosen_category = "NA.", bad_videos_by_category, bad_videos, good_videos)
extracted_frames1 = extract_frames(video_path1, frames_per_second)
show_extracted_frames(extracted_frames1)
print(video_path1)

# select a bad video containing running
video_path2 = select_video_path(1, choosen_quality, choosen_category = "bad", bad_videos_by_category = "run", bad_videos, good_videos)
extracted_frames2 = extract_frames(video_path2, frames_per_second)
show_extracted_frames(extracted_frames2)
print(video_path2)

result = describe_video_0shot(client, system_prompt, user_prompt, base64_frames=extracted_frames, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
    