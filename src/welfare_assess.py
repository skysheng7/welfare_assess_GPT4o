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

os.chdir('src')
from utils import *
from welfare_assess_helpers import *

# define result directory
results_folder = '../results_welfare_assess/cleanliness'
results_file = 'welfare_assess_cleanliness_gpt4o.csv' # store the results in a csv file
results_path = crete_result_path(results_folder, results_file)

# connect to OpenAI API
# Path to the file containing the API key
file_path = 'C:\\Users\\skysheng\\OneDrive - UBC\\R package project and Git\\API_keys\\openAI_key.txt'
# Open the file and read the API key
with open(file_path, 'r') as file:
    api_key = file.read().strip()  # Using strip() to remove any leading/trailing whitespace
openai_api_key = api_key
client = OpenAI(api_key=openai_api_key)

# choose model parameters  
mother_seed = 70
max_tokens=1000
detail_level="high"
temperature = 0.2
total_random_rounds = 9 # run multiple rounds of assessments

###################################################################################################
############################## welfare assessment: Hindleg_cleanliness ############################
###################################################################################################
# set input and output dir

root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Hindleg_cleanliness'
body_type = "hindleg cleanliness"
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
treatment_list = [item for item in os.listdir(train) if os.path.isdir(os.path.join(train, item))]

for rd in range(0, total_random_rounds):
    seed = mother_seed + (10*rd)

    for treatment in treatment_list: # specify the image processing treatment: "original", "segment", or "segment_bodyPart"
        description_path = os.path.join(train, 'description.txt')
        with open(description_path, 'r', encoding='utf-8') as file:
            description = file.read()
        
        if (treatment == 'original_boxed'):
            cur_test = os.path.join(test, 'original')
            description = description + "\n\n Please be aware that the body area of interest for evaluation is highlighted with red box(es)."
        else:
            cur_test = os.path.join(test, treatment)
        
        # Get all PNG and JPG files in the train folder and sort them
        cur_train = os.path.join(train, treatment)
        train_files = [f for f in os.listdir(cur_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_files.sort()
        train_images = convert_images_to_base64(cur_train, train_files) # convert to base64 format

        # test image examples: 
        test_files = [f for f in os.listdir(cur_test) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        test_images = convert_images_to_base64(cur_test, test_files)

        # generate prompts
        system_prompt = "You are an experienced expert in animal welfare science focusing on cattle behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
        user_prompt1 = "Below are text instructions for assessing **" + body_type + "** of cattle as part of the routine welfare assessment on farm: " + description + "\n Below are some example images you can learn from: \n "
        task = "\nYour task involves evaluating the **" + body_type + "** of cattle shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria descriptions and examples. \n "
        performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
        answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
        test_image_lead = "\n The following image requires your assessment: \n"
        user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

        # prompt GPT-4V
        start_index = 0
        end_index = len(test_files)
        test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, train_files, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, body_type=body_type, treatment=treatment)



###################################################################################################
########################## welfare assessment: Hindquarter_cleanliness ############################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Hindquarter_cleanliness'
body_type = "hindquarter cleanliness"
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
treatment_list = [item for item in os.listdir(train) if os.path.isdir(os.path.join(train, item))]

for rd in range(0, total_random_rounds):
    seed = mother_seed + (10*rd)

    for treatment in treatment_list: # specify the image processing treatment: "original", "segment", or "segment_bodyPart"
        description_path = os.path.join(train, 'description.txt')
        with open(description_path, 'r', encoding='utf-8') as file:
            description = file.read()

        if (treatment == 'original_boxed'):
            cur_test = os.path.join(test, 'original')
            description = description + "\n\n Please be aware that the body area of interest for evaluation is highlighted with red box(es)."
        else:
            cur_test = os.path.join(test, treatment)

        # Get all PNG and JPG files in the train folder and sort them
        cur_train = os.path.join(train, treatment)
        train_files = [f for f in os.listdir(cur_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_files.sort()
        train_images = convert_images_to_base64(cur_train, train_files) # convert to base64 format

        # test image examples: 
        test_files = [f for f in os.listdir(cur_test) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        test_images = convert_images_to_base64(cur_test, test_files)

        # generate prompts
        system_prompt = "You are an experienced expert in animal welfare science focusing on cattle behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
        user_prompt1 = "Below are text instructions for assessing **" + body_type + "** of cattle as part of the routine welfare assessment on farm: " + description + "\n Below are some example images you can learn from: \n "
        task = "\nYour task involves evaluating the **" + body_type + "** of cattle shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria descriptions and examples. \n "
        performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
        answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
        test_image_lead = "\n The following image requires your assessment: \n"
        user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

        # prompt GPT-4V
        start_index = 0
        end_index = len(test_files)
        test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, train_files, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, body_type=body_type, treatment=treatment)


###################################################################################################
############################ welfare assessment: Udder_cleanliness ################################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data_official/Udder_cleanliness'
body_type = "udder cleanliness"
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
treatment_list = [item for item in os.listdir(train) if os.path.isdir(os.path.join(train, item))]

for rd in range(0, total_random_rounds):
    seed = mother_seed + (10*rd)

    for treatment in treatment_list: # specify the image processing treatment: "original", "segment", or "segment_bodyPart"
        description_path = os.path.join(train, 'description.txt')
        with open(description_path, 'r', encoding='utf-8') as file:
            description = file.read()

        if (treatment == 'original_boxed'):
            cur_test = os.path.join(test, 'original')
            description = description + "\n\n Please be aware that the body area of interest for evaluation is highlighted with red box(es)."
        else:
            cur_test = os.path.join(test, treatment)

        # Get all PNG and JPG files in the train folder and sort them
        cur_train = os.path.join(train, treatment)
        train_files = [f for f in os.listdir(cur_train) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_files.sort()
        train_images = convert_images_to_base64(cur_train, train_files) # convert to base64 format

        # test image examples: 
        test_files = [f for f in os.listdir(cur_test) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        test_images = convert_images_to_base64(cur_test, test_files)

        # generate prompts
        system_prompt = "You are an experienced expert in animal welfare science focusing on cattle behavior and health, with 20 years of experience in conducting farm audit for welfare assessment. \n "
        user_prompt1 = "Below are text instructions for assessing **" + body_type + "** of cattle as part of the routine welfare assessment on farm: " + description + "\n Below are some example images you can learn from: \n "
        task = "\nYour task involves evaluating the **" + body_type + "** of cattle shown in the subsequent image, and record your score under \"assessment_result\", based on the previously provided criteria descriptions and examples. \n "
        performance_emotion_boost ="\n Give your assessment with a confidence score (low, medium or high) and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
        answer_format = "\n Answer format: ```json \n {\n  \"assessment_result\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
        test_image_lead = "\n The following image requires your assessment: \n"
        user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

        # prompt GPT-4V
        start_index = 0
        end_index = len(test_files)
        test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, train_files, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, body_type=body_type, treatment=treatment)
