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
results_folder = '../results_welfare_assess/cleanliness'

# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
from welfare_assess_helpers import *

# choose model parameters  
seed = 7
max_tokens=1000
detail_level="high"
temperature = 0.5


###################################################################################################
############################## welfare assessment: Hindleg_cleanliness ############################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Hindleg_cleanliness'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
treatment = "original" # specify the image processing treatment: "original", "segment", or "segment_bodyPart"

description_path = os.path.join(train, 'description.txt')
with open(description_path, 'r', encoding='utf-8') as file:
    description = file.read()
results_file = 'welfare_assess_cleanliness_result.csv' # store the results in a csv file
results_path = crete_result_path(results_folder, results_file)

# Get all PNG and JPG files in the train folder and sort them
train_files = [f for f in os.listdir(train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
train_files.sort()
train_images = convert_images_to_base64(train, train_files) # convert to base64 format

# test image examples: 
test_files = [f for f in os.listdir(test) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
test_images = convert_images_to_base64(test, test_files)

# generate prompts
system_prompt = "You are an experienced expert in animal welfare science focusing on dairy cow behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the cleanliness of dairy cows. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the cleanliness of the hindleg of dairy cow shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria and examples. 0: clean; 2: dirty.\n "
performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature)





###################################################################################################
########################## welfare assessment: Hindquarter_cleanliness ############################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Hindquarter_cleanliness'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_Hindquarter_cleanliness_test.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the cleanliness of dairy cows. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the cleanliness of the hindquarter of dairy cow shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria and examples. 0: clean; 2: dirty.\n "
performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature)


###################################################################################################
############################ welfare assessment: Udder_cleanliness ################################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Udder_cleanliness'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_Udder_cleanliness_test.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the cleanliness of dairy cows. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the cleanliness of the udder of dairy cow shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria and examples. 0: clean; 2: dirty.\n "
performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature)

