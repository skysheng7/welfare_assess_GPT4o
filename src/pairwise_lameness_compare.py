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
full_path = crete_result_path(results_folder, results_file)

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get the list of all mp4 videos
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
from pairwise_lameness_compare_helpers import *
mp4_folder_path = 'C:\\Users\\skysheng\\OneDrive - UBC\\University of British Columbia\\Research\\PhD Project\\Amazon project phase 2\\Sora Jeong\\results\\30cow_artificial_group_compressed'
all_mp4 = list_all_mp4(mp4_folder_path)

# get a list of pairwise lameness comparison results
pairwise_result_path = 'C:\\Users\\skysheng\\OneDrive - UBC\\R package project and Git\\lameness_rank\\05-Amazon_MTurk_expert_response_30cow_pairwise\\results\\all_experts'
pairwise_result = pd.read_csv(os.path.join(pairwise_result_path, "winner_loser_avg_DW_NV_SB_KI.csv"))
sorted_pairwise_result = pairwise_result.sort_values(by='degree', ascending=False).reset_index(drop=True)
sorted_pairwise_result['winner'] = sorted_pairwise_result['winner'].astype(int)
sorted_pairwise_result['loser'] = sorted_pairwise_result['loser'].astype(int)


###################################################################################################
####################################### pairwise lameness comparison ##############################
###################################################################################################
# write prompts
system_prompt = "You are an experienced expert in animal science focusing on dairy cow lameness assessment, with 50 years of experience in conducting lameness assessment on dairy farms. You are allowed to assess lameness of real animals."
user_prompt1 = "Your task involves watching two videos (a series of frames), each showing a different cow walking. After viewing both videos, you need to decide which of the two cows is more lame. For the cow you judge as more lame, write down its 4-digit ID as the `winner`. For the other cow, record its ID as the `loser`. If both cows seem equally lame, you can choose any cow as the `winner` or `loser`, but make sure to write `0` under the column `degree` to indicate they have equal levels of lameness. you also need to assess the extent of lameness difference between them. Rate this difference on a scale from 0 to 3 and record it under `degree`. A rating of 0 indicates that both cows are equally lame, while a rating of 3 suggests a significant difference in lameness between the `winner` and the `loser`. It's important to watch the frames of each video in numerical order, from the smallest to the largest number. Make sure to view the series of frames in ascending numerical order, starting from the smallest to the largest number, as indicated by the red numbers on the top left corner of each frame.\n."
user_prompt1 = user_prompt1 + "Essential: Briefly explain your reasoning under `reason` to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. I'll tip you $50 for this task. \n Answer format: ```json \n {\n  \"winner\": \"...\",\n \"loser\": \"...\",\n  \"degree\": \"...\",\n  \"reason\": \"...\"\n}``` "

# choose model parameter
frames_per_second=2 # how many frames to extract each second
seed = 800
max_tokens=1000
detail_level="low"
temperature = 0.5

for i in range(0, 1):
    # identify the winner and loser cow in pairwise lameness assessment
    cow1 = sorted_pairwise_result.iloc[i]['winner']
    cow2 = sorted_pairwise_result.iloc[i]['loser']
    degree = sorted_pairwise_result.iloc[i]['degree']

    win_cow_path = find_mp4_with_digits(all_mp4, cow1)[0]
    lose_cow_path = find_mp4_with_digits(all_mp4, cow2)[0]

    # continue writing prompts
    user_prompt1 = user_prompt1 + f"First, here is the video of cow {cow1}(ID) walking:"
    user_prompt2 = f"Second, here is the video of cow {cow2}(ID) walking:"

    # extract frames from the winner cow
    extracted_frames1 = extract_frames(win_cow_path, frames_per_second)
    show_extracted_frames(extracted_frames1)
    print(win_cow_path)

    # extract frames from the loser cow
    extracted_frames2 = extract_frames(lose_cow_path, frames_per_second)
    show_extracted_frames(extracted_frames2)
    print(lose_cow_path)

    # prompt GPT4V, and extract result answer
    result = compare_2video(client, system_prompt, user_prompt1, user_prompt2, base64_frames1=extracted_frames1, base64_frames2=extracted_frames2, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
    save_pairwise_lame_csv(full_path, cow1, cow2, degree, video_path1=win_cow_path, video_path2=lose_cow_path, result=result, system_prompt=system_prompt, user_prompt1=user_prompt1, user_prompt2=user_prompt2, frames_per_second=frames_per_second, max_tokens=max_tokens, detail_level=detail_level, seed=seed, temperature=temperature)





