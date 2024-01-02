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

# define result directory
results_folder = '../results'

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
choosen_quality = "bad" # "good", "bad"
choosen_category = "approach" # 'approach', 'direction', 'human', 'run', 'slip', 'stop', 'two', 'NA.'
frames_per_second=1

# read in videos and seperate into frames
if ((choosen_quality == "bad") and (choosen_category != "NA.")):
    video_path = bad_videos_by_category[choosen_category][0]
elif (choosen_quality == "bad"):
    video_path = bad_videos[0]
else:
    video_path = good_videos[0]
extracted_frames = extract_frames(video_path, frames_per_second)

# display the frames extracted
show_extracted_frames(extracted_frames)

# generate descrition of 1 frame/image ended with base64
#text_prompt = "Describe what's in the image"
#describe_img(client, text_prompt, base64_image=extracted_frames[0], max_tokens=300, detail_level="low")

# generate descrition of the video (list of frames)
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for lameness assessment. \n Criteria for good video: Shows a single dairy cow walking smoothly in a straight line, entering from the leftmost side of the screen and exiting on the rightmost side, at a normal speed. \n Criteria for bad video, in 8 categories: [1] `direction` - cow moves from right to left, or in any direction contrary to that described for a good video [2] `stop` - cow pauses or sniffs the ground while walking, [3] `run` - cow runs or jogs, [4] `approach` - cow comes towards the camera, [5] `human` - excessive human interference or obstruction, [6] `slip` - cow slips while walking, [7] `multiple` - more than 1 cows in the video, [8] `other` - any other issue making the video hard for lameness assessment."
user_prompt = "Your job is to review cow videos (a series of frames), and classify them as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` \n Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: ```JSON \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"\n}```"
max_tokens=500
detail_level="low"
seed = 7
temperature = 0.5
result = describe_video_0shot(client, system_prompt, text_prompt=user_prompt, base64_frames=extracted_frames, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
result_content = result.choices[0].message.content
result_json = result_content.strip('```json\n').strip('```')
json_data = json.loads(result_json) # Convert the string to a Python dictionary

# store the results in a
results_file = 'video_quality_assess_GPT4V_results.csv'
full_path = os.path.join(results_folder, results_file)

# Check if the 'results' folder exists, if not, create it
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Prepare the data for the DataFrame
data = {
    "video_path": video_path,
    "true_quality": choosen_quality,
    "true_category": choosen_category,
    "predict_quality": json_data.get('quality', 'NA'),
    "predict_category": json_data.get('category', 'NA'),
    "predict_confidence": json_data.get('confidence', 'NA'),
    "predict_reason": json_data.get('reason', 'NA'),
    "predict_result": result_content,
    "model": "gpt-4-vision-preview",
    "date": datetime.now().date(),
    "frames_per_second": frames_per_second, 
    "system_prompt": system_prompt,
    "user_prompt": user_prompt,
    "max_tokens": max_tokens,
    "detail_level": detail_level,
    "seed": seed,
    "temperature": temperature
}

# Create a DataFrame from the data
df = pd.DataFrame([data])

# Check if the file exists and append data, else write a new file
if os.path.isfile(full_path):
    df.to_csv(full_path, mode='a', header=False, index=False)
    print(f"Data appended to {full_path}")
else:
    df.to_csv(full_path, mode='w', header=True, index=False)




            