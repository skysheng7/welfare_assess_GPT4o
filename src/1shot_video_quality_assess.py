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
results_file = '1shot_good_description_quality_assess.csv' # store the results in a csv file

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Research/PhD Project/Amazon project phase 2/Kay Yang/sorted_cow_videos_all'
good_videos, bad_videos, bad_videos_by_category = get_video_paths(root_folder_path)

# randomly shuffle the video sequence
frames_per_second=2 # how many frames to extract each second
seed = 7
random.seed(seed)
random.shuffle(good_videos)
random.shuffle(bad_videos)
for category_videos in bad_videos_by_category.values(): # Shuffle each sublist in bad_videos_by_category
    random.shuffle(category_videos)


###################################################################################################
########################## GPT-4V generate description for good & bad video #######################
###################################################################################################
# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for lameness assessment. \n "
user_prompt1 = "Your job is to review 2 cow videos, the first video is a good-quality video, and the second video is a bad-quality video. Create a detailed description for each video focusing only on the movement of the cow (not environment or lighting conditions), focusing on characteristics that help differentiate good-quality videos from bad-quality ones that are independent from the frame number. Create a prompt to ask the GPT-4V model in identify video quality in future tasks to identify bad-quality videos due to cow is running. Make sure to view the series of frames in ascending numerical order, starting to review the frames from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame. The first video that you will be presented below is a good video: "
user_prompt2 = "The second video is a bad video because the cow runs through the scene, as shown below:"

max_tokens=1000
detail_level="low"
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
########################### GPT-4V 1-shot learning video quality assessment #######################
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
########################### GPT-4V 1-shot learning describe good video only #######################
###################################################################################################
# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for accurate lameness assessment. Criteria for good video: Shows a single dairy cow walking smoothly in a fairly straight path with steady pace. The cow should start walking from the side of the screen shaded in green and exit to the opposite side. The presence of a person walking closely behind the cow is acceptable and does not disqualify the video from being considered good. \n "
user_prompt1 = f"You will be provided with 2 videos, first is an example `good` video to be used as a reference for good quality videos, second is the test video. Your job is to review cow videos (a series of frames) frame by frame, and classify test video as `good` or `bad` based on these criteria. While evaluating the cow's speed, bear in mind that you are reviewing a sequence of frames extracted at {frames_per_second} frames per second, and not the original video. Make sure to view the series of frames in ascending numerical order, starting from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame.\n "
user_prompt1 = user_prompt1 + "Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: ```json \n {\n  \"quality\": \"...\",\n \"confidence\": \"...\",\n  \"reason\": \"...\"\n}``` First, here is an example `good` video: "
user_prompt2 = "Second, here is a test video needs your assessment for video quality:"

max_tokens=1000
detail_level="low"
temperature = 0.5
choosen_quality1 = "good"
choosen_quality2 = "good"
full_path = crete_result_path(results_folder, results_file)

# select a good video as example
video_path1 = select_video_path(i=644, choosen_quality = choosen_quality1, choosen_category = "NA", bad_videos_by_category=bad_videos_by_category, bad_videos=bad_videos, good_videos=good_videos)
extracted_frames1 = extract_frames(video_path1, frames_per_second)
show_extracted_frames(extracted_frames1)
print(video_path1)

for i in range(5, 10):
    # select a good video as test
    video_path2 = select_video_path(i, choosen_quality= choosen_quality2, choosen_category = "NA", bad_videos_by_category=bad_videos_by_category , bad_videos=bad_videos, good_videos=good_videos)
    extracted_frames2 = extract_frames(video_path2, frames_per_second)
    show_extracted_frames(extracted_frames2)
    print(video_path2)

    # extract result answer
    result = compare_2video(client, system_prompt, user_prompt1, user_prompt2, base64_frames1=extracted_frames1, base64_frames2=extracted_frames2, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
    save_1shot_quality_assess_results_to_csv(full_path, video_path1, video_path2, result, system_prompt, user_prompt1, user_prompt2, choosen_quality1, choosen_quality2, frames_per_second, max_tokens, detail_level, seed, temperature)


###################################################################################################
####################################### pairwise lameness comparison ##############################
###################################################################################################
