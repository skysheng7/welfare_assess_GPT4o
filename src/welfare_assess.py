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

# choose model parameters  
seed = 7
max_tokens=1000
detail_level="high"
temperature = 0.5


###################################################################################################
#################################### welfare assessment: BCS ######################################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/BCS'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_BCS.csv' # store the results in a csv file
results_path = crete_result_path(results_folder, results_file)

# train image examples: Get all PNG files in the train folder and sort them
train_files = [f for f in os.listdir(train) if f.lower().endswith('.png')]
train_files.sort()
train_images = [convert_to_jpeg_base64(os.path.join(train, f)) for f in train_files] # Convert PNG images to JPEG and then to Base64

# test image examples: 
test_files = [f for f in os.listdir(test) if f.lower().endswith('.jpg')]
test_images = [convert_jpg_to_base64(os.path.join(test, f)) for f in test_files]

# generate prompts
system_prompt = "You are an experienced expert in animal welfare science focusing on dairy cow behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the body condition of dairy cows. Please read these textual descriptions and examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the body condition score (body_condition_score) of the dairy cow shown in the subsequent image, based on the previously provided criteria and examples. In cases where the image quality impedes a clear assessment, please mark the `body_condition_score` as `uncertain`. "
performance_emotion_boost ="\n Give your assessment with a confidence score from 0-1 and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"body_condition_score\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}```"
test_image_lead = "\n The following image requires your assessment of the cow's body condition: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = 1
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature)












###################################################################################################
############################### welfare assessment: integument alterations ########################
###################################################################################################


###################################################################################################
############################## welfare assessment: nasal discharge ################################
###################################################################################################


###################################################################################################
####################### welfare assessment: udder, hindquarter, hindleg ###########################
###################################################################################################



###################################################################################################
################################### welfare assessment: water #####################################
###################################################################################################




