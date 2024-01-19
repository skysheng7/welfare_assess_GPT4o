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
results_file = 'pairwise_lameness_compare.csv' # store the results in a csv file

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
from pairwise_lameness_compare_helpers import *
mp4_folder_path = 'C:\\Users\\skysheng\\OneDrive - UBC\\University of British Columbia\\Research\\PhD Project\\Amazon project phase 2\\Sora Jeong\\results\\30cow_artificial_group_compressed'
all_mp4 = list_all_mp4(mp4_folder_path)

# get a list of pairwise lameness comparison results
pairwise_result_path = 'C:\\Users\\skysheng\\OneDrive - UBC\\R package project and Git\\lameness_rank\\05-Amazon_MTurk_expert_response_30cow_pairwise\\results\\all_experts'
pairwise_result = pd.read_csv(os.path.join(pairwise_result_path, "winner_loser_avg_DW_NV_SB_KI.csv"))
sorted_pairwise_result = pairwise_result.sort_values(by='degree', ascending=False).reset_index(drop=True)
pairwise_result['winner'] = pairwise_result['winner'].astype(int)
pairwise_result['loser'] = pairwise_result['loser'].astype(int)

# randomly shuffle the video sequence
frames_per_second=3 # how many frames to extract each second
seed = 7


###################################################################################################
####################################### pairwise lameness comparison ##############################
###################################################################################################
# choose model parameters and write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow behavior and health, with 50 years of experience in observing dairy cow gait and behavior through video. You are expert in assessing the quality of videos to select the ones suitable for accurate lameness assessment. \n Criteria for good video: Shows a single dairy cow walking smoothly in a fairly straight path with steady pace, entering from the leftmost side of the screen and exiting into the area shaded green on the right, allowing for a detailed observation of its gait. The presence of a person walking closely behind the cow is acceptable and does not disqualify the video from being considered good. \n Criteria for bad video, in 8 categories: [1] `direction` - the cow is overlapped with the green-shaded area in the first few frames [2] `stop` - cow pauses momentarily in the same spot [3] `approach` - the cow comes so close to the camera that sometimes the cow is facing straight to the camera or its hooves are not visible at the bottom of the screen [4] `human` - excessive human interference or obstruction [5] `multiple` - more than 1 cows in the video, [6] `slip` - cow slips or stumbles while walking, [7] `run` - the cow is running or moving too fast for a proper assessment of gait due to quick pace, the legs appear blurred consistantly across frames, and an inability to observe consistent hoof placement, stride length or back posture."
user_prompt1 = f"You will be provided with 2 videos, first is an example `good` video to be used as a reference for good quality videos, second is the test video. Your job is to review cow videos (a series of frames) frame by frame, and classify test video as `good` or `bad` based on these criteria. If `bad`, specify which category or categories apply; if good, mark the category as `NA.` Although there are more descriptions and creteria related to `bad` videos, please avoid any predisposition towards labeling test videos as `bad`. While evaluating the cow's speed, bear in mind that you are reviewing a sequence of frames extracted at {frames_per_second} frames per second, and not the original video. Make sure to view the series of frames in ascending numerical order, starting from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame.\n "
user_prompt1 = user_prompt1 + "Essential: Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n Answer format: ```json \n {\n  \"quality\": \"...\",\n  \"category\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"\n}``` First, here is an example `good` video: "
user_prompt2 = "Second, here is a test video needs your assessment for video quality:"

max_tokens=1000
detail_level="low"
temperature = 0.5

for i in range(0, 10):
    

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



