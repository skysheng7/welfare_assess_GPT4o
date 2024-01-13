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

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *


###################################################################################################
#################################### welfare assessment: BCS ######################################
###################################################################################################
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/BCS'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_BCS.csv' # store the results in a csv file

# Get all PNG files in the train folder and sort them
png_files = [f for f in os.listdir(train) if f.endswith('.png')]
png_files.sort()
# Convert PNG images to JPEG and then to Base64
train_images = [convert_to_jpeg_base64(os.path.join(train, f)) for f in png_files]


# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal welfare science focusing on dairy cow behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
task = "Your job is to \n "
performance_emotion_boost ="\n Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\",\n  \"caution\": \"...\"}```"
user_prompt1 = task + performance_emotion_boost + answer_format

seed = 7
max_tokens=1000
detail_level="high"
temperature = 0.5

# select a good video
video_path1 = select_video_path(i=1, choosen_quality = "good", choosen_category = "NA.", bad_videos_by_category=bad_videos_by_category, bad_videos=bad_videos, good_videos=good_videos)
extracted_frames1 = extract_frames(video_path1, frames_per_second)
show_extracted_frames(extracted_frames1)
print(video_path1)

# select a bad video containing running
video_path2 = select_video_path(3, choosen_quality= "bad", choosen_category = "run", bad_videos_by_category=bad_videos_by_category , bad_videos=bad_videos, good_videos=good_videos)
extracted_frames2 = extract_frames(video_path2, frames_per_second)
show_extracted_frames(extracted_frames2)
print(video_path2)

# extract result answer
result = compare_2video(client, system_prompt, user_prompt1, user_prompt2, base64_frames1=extracted_frames1, base64_frames2=extracted_frames2, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
result_content = result.choices[0].message.content

# calculate usage
output_token = result.usage.completion_tokens
prompt_tokens = result.usage.prompt_tokens
output_token_p = output_token_cost(output_token)
prompt_tokens_p = input_token_cost(prompt_tokens)
total_cost = round((output_token_p+prompt_tokens_p), 3)
print(total_cost)


###################################################################################################
############################### welfare assessment: integument alterations ########################
###################################################################################################
# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for accurate lameness assessment. \n Criteria for good video: Shows a single dairy cow walking smoothly in a fairly straight path with steady pace, entering from the leftmost side of the screen and exiting into the area shaded green on the right, allowing for a detailed observation of its gait. The presence of a person walking closely behind the cow is acceptable and does not disqualify the video from being considered good. \n Criteria for bad video, in 8 categories: [1] `direction` - the cow is overlapped with the green-shaded area in the first few frames [2] `stop` - cow pauses momentarily in the same spot [3] `approach` - the cow comes so close to the camera that sometimes the cow is facing straight to the camera or its hooves are not visible at the bottom of the screen [4] `human` - excessive human interference or obstruction [5] `multiple` - more than 1 cows in the video, [6] `slip` - cow slips or stumbles while walking, [7] `run` - the cow is running or moving too fast for a proper assessment of gait due to quick pace, the legs appear blurred consistantly across frames, and an inability to observe consistent hoof placement, stride length or back posture."
user_prompt1 = f"You will be provided with 2 videos, first is an example `good` video to be used as a reference for good quality videos, second is the test video. Your job is to review cow videos (a series of frames) frame by frame, and classify test video as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` Although there are more descriptions and creteria related to `bad` videos, please avoid any predisposition towards labeling test videos as `bad`. While evaluating the cow's speed, bear in mind that you are reviewing a sequence of frames extracted at {frames_per_second} frames per second, and not the original video. Make sure to view the series of frames in ascending numerical order, starting from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame.\n "
user_prompt1 = user_prompt1 + "Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: ```json \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"\n}``` First, here is an example `good` video: "
user_prompt2 = "Second, here is a test video needs your assessment for video quality:"

max_tokens=1000
detail_level="low"
temperature = 0.5

# select a good video as example
video_path1 = select_video_path(i=644, choosen_quality = "good", choosen_category = "NA.", bad_videos_by_category=bad_videos_by_category, bad_videos=bad_videos, good_videos=good_videos)
extracted_frames1 = extract_frames(video_path1, frames_per_second)
show_extracted_frames(extracted_frames1)
print(video_path1)

# select a good video as test
video_path2 = select_video_path(i=9, choosen_quality= "good", choosen_category = "NA.", bad_videos_by_category=bad_videos_by_category , bad_videos=bad_videos, good_videos=good_videos)
extracted_frames2 = extract_frames(video_path2, frames_per_second)
show_extracted_frames(extracted_frames2)
print(video_path2)

# extract result answer
result = compare_2video(client, system_prompt, user_prompt1, user_prompt2, base64_frames1=extracted_frames1, base64_frames2=extracted_frames2, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
result_content = result.choices[0].message.content

# calculate usage
output_token = result.usage.completion_tokens
prompt_tokens = result.usage.prompt_tokens
output_token_p = output_token_cost(output_token)
prompt_tokens_p = input_token_cost(prompt_tokens)
total_cost = round((output_token_p+prompt_tokens_p), 3)
print(total_cost)

###################################################################################################
############################## welfare assessment: nasal discharge ################################
###################################################################################################


###################################################################################################
####################### welfare assessment: udder, hindquarter, hindleg ###########################
###################################################################################################



###################################################################################################
################################### welfare assessment: water #####################################
###################################################################################################




