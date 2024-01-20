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
from welfare_assess_helpers import *

# choose model parameters  
seed = 7
max_tokens=1000
detail_level="high"
temperature = 0.5


###################################################################################################
#################################### welfare assessment: BCS ######################################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data/BCS'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_BCS_dairy_dual_purpose_test.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the body condition of dairy cows. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the body condition score (body_condition_score) of the dairy cow shown in the subsequent image, based on the previously provided criteria and examples. 0: regular body condition; 1: very lean; 2: very fat.\n "
performance_emotion_boost ="\n Give your assessment with a confidence score and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"body_condition_score\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment of the cow's body condition: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = 1
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, assessment_type = "BCS")



###################################################################################################
############################### welfare assessment: integument alterations ########################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/data/Integument_alterations'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_integument_alterations_test.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the integument alterations (3 types: [1] hairless patches, [2] lesions, [3] swellings) on the legs of dairy cows. Please read these text descriptions and examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the integument alterations of the dairy cow shown in the subsequent image, based on the previously provided criteria and examples. Specifically, check for any abnormalities such as hairless patches, lesions, or swellings on the cow. If you identify any such integument alterations, please record the type(s) of these alterations under `integument_alterations`. If no alterations are present, simply write `NA` \n "
performance_emotion_boost ="\n Give your assessment with a confidence score and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"integument_alterations\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment of the cow's integument alterations: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_images)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, assessment_type = "integument_alterations")

###################################################################################################
############################## welfare assessment: nasal discharge ################################
###################################################################################################

# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/Nasal_discharge'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_nasal_discharge2.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing nasal discharge of dairy cows. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the nasal discharge of the dairy cow shown in the subsequent image, based on the previously provided criteria and examples. Enter `0` under `nasal_discharge` if there is no evidence of nasal discharge. Enter `2` under `nasal_discharge` if there is evidence of nasal discharge.  Enter `NA` if the image is not clear enough for assessment. Focus specifically on identifying any discharge (liquid) present. It's common for cows to have remnants of feed or straw on their noses from eating, appearing as brownish debris. Please note that these remnants are not considered discharge\n "
performance_emotion_boost ="\n Give your assessment with a confidence score and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"nasal_discharge\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment of the cow's nasal discharge conditions: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, assessment_type = "nasal")


###################################################################################################
####################### welfare assessment: udder, hindquarter, hindleg ###########################
###################################################################################################

# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/udder_hindquarter_hindleg'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_udder_hindquarter_hindleg.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing cleanliness of dairy cows' [1] udder, [2]hindquarter, and [3]hindleg. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the cleanliness of dairy cows' [1]udder, [2]hindquarter, and [3]hindleg shown in the subsequent image, based on the previously provided criteria and examples.  `udder`: Enter `0` if no dirt or minor splashing; Enter `2` if there are distinct plaques of dirt on udder or any dirt on and around the teats. `hindquarter`: Enter `0` if no dirt or minor splashing; Enter `2` if there are separate or continuous plaques of dirt. `hindleg`: Enter `0` if no dirt or minor splashing; Enter `2` if there are separate or continuous plaques of dirt above the coronary band. If you can not assess due to image angle, enter `NA`\n "
performance_emotion_boost ="\n Give your assessment with a confidence score and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"udder\": \"...\",\n \"hindquarter\": \"...\",\n \"hindleg\": \"...\",\n \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment of the cow's cleanliness of udder, hindquarter, and hindleg: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, assessment_type = "cleanliness")



###################################################################################################
################################### welfare assessment: water #####################################
###################################################################################################
# set input and output dir
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Other projects/welfare_assessment_GPT4V/Water'
train = os.path.join(root_folder_path, "train")
test = os.path.join(root_folder_path, "test")
results_file = 'welfare_assess_water.csv' # store the results in a csv file
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
user_prompt1 = "Below are images containing text descriptions and criteria for assessing the cleanliness of water points. Please read these examples in the images. Creteria and example images: \n "
task = "Your task involves evaluating the cleanliness of water points shown in the subsequent image, based on the previously provided criteria and examples. Enter `clean` under `cleanliness` if drinkers and water are clean at the moment of inspection. Enter `partly dirty` under `cleanliness` if drinkers are dirty, but water is fresh and clean at moment of inspection or only part of several drinkers clean and containing clean water. Enter `dirty` under `cleanliness` if drinkers and water are both dirty at moment of inspection. Enter `NA` if the image is not clear enough for inspection. \n "
performance_emotion_boost ="\n Give your assessment with a confidence score and briefly explain your reasoning to clarify your thought process step by step. Take a deep breath before you answer. This task is vital to my career, and I greatly value your thorough analysis. \n"
answer_format = "\n Answer format: ```json \n {\n  \"cleanliness\": \"...\",\n  \"confidence\": \"...\",\n  \"reason\": \"...\"}``` \n"
test_image_lead = "\n The following image requires your assessment of the cleanliness of water points: \n"
user_prompt2 = task + performance_emotion_boost + answer_format + test_image_lead

# prompt GPT-4V
start_index = 0
end_index = len(test_files)
test_images_in_range(results_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s=seed, temp=temperature, assessment_type = "water")





