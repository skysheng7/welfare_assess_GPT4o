import os
import glob
import re
import numpy as np
import cv2
import base64
import pandas as pd
from datetime import datetime
import json
import time
import requests
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from PIL import Image
from io import BytesIO
import re


def convert_to_jpeg_base64(png_image_path):
    # Open the PNG image
    with Image.open(png_image_path) as img:
        # Convert to JPEG
        buffer = BytesIO()
        img.convert('RGB').save(buffer, format="JPEG")
        # Encode to Base64
        jpeg_base64 = base64.b64encode(buffer.getvalue()).decode()
    return jpeg_base64

def convert_jpg_to_base64(jpg_image_path):
    # Open the JPG image
    with Image.open(jpg_image_path) as img:
        # Create a BytesIO object to hold the byte stream
        buffer = BytesIO()
        
        # Save the image to the buffer
        img.save(buffer, format="JPEG")
        
        # Get the byte stream and encode it to Base64
        base64_string = base64.b64encode(buffer.getvalue()).decode()

    return base64_string


def prompt_welfare_assess_test_image(client, system_prompt, user_prompt1, user_prompt2, train_images, cur_test, detail_level, max_tokens, s, temp):
    # Generate the content for the train images
    train_content = generate_image_content(base64_frames=train_images, detail_level=detail_level)

    # Constructing prompt messages
    prompt_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt1},
                *train_content,  # Using the list of dictionaries generated by the function
                {"type": "text", "text": user_prompt2},
                {"type": "image_url",
                 "image_url": {
                    "url": f"data:image/jpg;base64, {cur_test}",
                    "detail": detail_level}
                }
            ],
        },
    ]

    # Parameters for the API call
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "seed": s,
        "temperature": temp
    }

    # Assuming client.chat.completions.create is a function call to an external API
    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)  # print out the response
    print(result.usage)  # print out how many tokens were used

    return result



def save_bcs_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    

    # extract the true label of the image
    parts = re.split(r'[_.]', cur_file_name)
    true_bcs = parts[0]  
    true_note = parts[1]

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_bcs = json_data.get('body_condition_score', 'NA')
    conf = json_data.get('confidence', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {
        "test_image": cur_file_name,
        "true_bcs": true_bcs,
        "true_note": true_note,
        "predict_bcs": predict_bcs,
        "predict_confidence": conf,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
        "detail_level": detail_level,
        "seed": seed,
        "temperature": temperature,
        "completion_tokens": output_token,
        "prompt_tokens": prompt_tokens,
        "total_cost": total_cost

    }

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")

def save_integument_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    

    # extract the true label of the image
    parts = re.split(r'[_.]', cur_file_name)
    true_integument_alterations = parts[0]  

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_integument_alterations = json_data.get('integument_alterations', 'NA')
    conf = json_data.get('confidence', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {
        "test_image": cur_file_name,
        "true_integument_alterations": true_integument_alterations,
        "predict_integument_alterations": predict_integument_alterations,
        "predict_confidence": conf,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
        "detail_level": detail_level,
        "seed": seed,
        "temperature": temperature,
        "completion_tokens": output_token,
        "prompt_tokens": prompt_tokens,
        "total_cost": total_cost

    }

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")


def save_nasal_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    

    # extract the true label of the image
    parts = re.split(r'[_.]', cur_file_name)
    true_nasal_discharge = parts[0]  

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_nasal_discharge = json_data.get('nasal_discharge', 'NA')
    conf = json_data.get('confidence', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {
        "test_image": cur_file_name,
        "true_nasal_discharge": true_nasal_discharge,
        "predict_nasal_discharge": predict_nasal_discharge,
        "predict_confidence": conf,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
        "detail_level": detail_level,
        "seed": seed,
        "temperature": temperature,
        "completion_tokens": output_token,
        "prompt_tokens": prompt_tokens,
        "total_cost": total_cost

    }

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")


def save_cleanliness_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    

    # extract the true label of the image
    parts = re.split(r'[_.]', cur_file_name)
    true_cleanliness = parts[0]  
    # gather true label
    true_udder = 0
    if 'udder' in cur_file_name:
        true_udder = 2
    true_hindquarter = 0
    if 'hindquarter' in cur_file_name:
        true_hindquarter = 2
    true_hindleg = 0
    if 'hindleg' in cur_file_name:
        true_hindleg = 2

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_udder = json_data.get('udder', 'NA')
    predict_hindquarter = json_data.get('hindquarter', 'NA')
    predict_hindleg = json_data.get('hindleg', 'NA')
    conf = json_data.get('confidence', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {
        "test_image": cur_file_name,
        "true_cleanliness": true_cleanliness,
        "true_udder": true_udder,
        "predict_udder": predict_udder,
        "true_hindquarter": true_hindquarter,
        "predict_hindquarter": predict_hindquarter,
        "true_hindleg": true_hindleg,
        "predict_hindleg": predict_hindleg,
        "predict_confidence": conf,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
        "detail_level": detail_level,
        "seed": seed,
        "temperature": temperature,
        "completion_tokens": output_token,
        "prompt_tokens": prompt_tokens,
        "total_cost": total_cost

    }

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")

def save_water_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    

    # extract the true label of the image
    parts = re.split(r'[_.]', cur_file_name)
    true_water = parts[0]  

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_water = json_data.get('cleanliness', 'NA')
    conf = json_data.get('confidence', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {
        "test_image": cur_file_name,
        "true_water": true_water,
        "predict_water": predict_water,
        "predict_confidence": conf,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_tokens": max_tokens,
        "detail_level": detail_level,
        "seed": seed,
        "temperature": temperature,
        "completion_tokens": output_token,
        "prompt_tokens": prompt_tokens,
        "total_cost": total_cost

    }

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")


def test_images_in_range(full_path, start_index, end_index, client, system_prompt, user_prompt1, user_prompt2, train_images, test_images, test_files, detail_level, max_tokens, s, temp, assessment_type):
    user_prompt = user_prompt1 + "\n**example images**\n" + user_prompt2 + "\n**test images**\n"
    for i in range(start_index, end_index):
        cur_file_name = test_files[i]
        result = prompt_welfare_assess_test_image(client, system_prompt, user_prompt1, user_prompt2, train_images, test_images[i], detail_level, max_tokens, s, temp)
        if (assessment_type == "BCS"):
            save_bcs_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, s, temp)
        elif (assessment_type == "integument_alterations"):
            save_integument_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, s, temp)
        elif (assessment_type == "nasal"):
            save_nasal_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, s, temp)
        elif(assessment_type == "cleanliness"):
            save_cleanliness_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, s, temp)
        elif(assessment_type == "water"):
            save_water_results_to_csv(full_path, cur_file_name, result, system_prompt, user_prompt, max_tokens, detail_level, s, temp)

