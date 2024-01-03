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
results_file = 'video_quality_assess_GPT4V_results.csv' # store the results in a csv file

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
frames_per_second=7 # how many frames to extract each second

# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for lameness assessment. \n Criteria for good video: Shows a single dairy cow walking smoothly in a straight line, entering from the leftmost side of the screen and exiting on the rightmost side, at a normal speed. The presence of a person walking closely behind the cow is acceptable and does not disqualify the video from being considered good. \n Criteria for bad video, in 8 categories: [1] `direction` - cow moves from right to left, or in any direction contrary to that described for a good video [2] `stop` - cow pauses or sniffs the ground while walking, [3] `run` - cow runs through the scene, [4] `approach` - cow comes towards the camera, [5] `human` - excessive human interference or obstruction, [6] `slip` - cow slips while walking, [7] `multiple` - more than 1 cows in the video, [8] `other` - any other issue making the video hard for lameness assessment."
user_prompt = f"Your job is to review cow videos (a series of frames), and classify them as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` Although there are more descriptions and creteria related to `bad` videos, please avoid any predisposition towards labeling videos as `bad`. While evaluating the cow's speed, bear in mind that you are reviewing a sequence of frames extracted at {frames_per_second} frames per second, and not the original video. \n "
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

# read in videos and prompt to GPT-4V
start_index = 0
end_index = 1
process_videos_in_range(start_index, end_index, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos, frames_per_second, client, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature, results_folder, results_file)



# generate descrition of 1 frame/image ended with base64
#text_prompt = "Describe what's in the image"
#describe_img(client, text_prompt, base64_image=extracted_frames[0], max_tokens=300, detail_level="low")

