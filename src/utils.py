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

## Video reading and processing
def get_video_paths(root_folder):
    good_videos = []
    bad_videos = []
    bad_videos_by_category = {'approach': [], 'direction': [], 'human': [], 'run': [], 'slip': [], 'stop': [], 'two': []}

    # Regular expression to match a folder name with exactly four digits
    cow_id_pattern = re.compile(r'^\d{4}$')

    # Iterate through each folder in the root directory
    for folder in glob.glob(os.path.join(root_folder, '*/')):
        cow_id_folder = os.path.basename(os.path.dirname(folder))
        
        # Check if folder name is exactly four digits
        if cow_id_pattern.match(cow_id_folder):
            # Paths for good and bad videos inside each cow's folder
            good_folder = os.path.join(folder, 'good')
            bad_folder = os.path.join(folder, 'bad')

            # Add good video paths from both good and duplicate folders
            good_videos.extend(glob.glob(os.path.join(good_folder, '**/*'), recursive=True))

            # Filter for mp4 files, case-insensitive
            good_videos = [video for video in good_videos if video.lower().endswith('.mp4')]

            # Add bad video paths and categorize them
            for category in bad_videos_by_category.keys():
                category_folder = os.path.join(bad_folder, category)
                category_videos = glob.glob(os.path.join(category_folder, '*'))
                # Filter for mp4 files, case-insensitive
                category_videos = [video for video in category_videos if video.lower().endswith('.mp4')]
                bad_videos.extend(category_videos)
                bad_videos_by_category[category].extend(category_videos)

    return good_videos, bad_videos, bad_videos_by_category


def select_video_path(i, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos):
    if ((choosen_quality == "bad") and (choosen_category != "NA"))():
        video_path = bad_videos_by_category[choosen_category][i]
    elif choosen_quality == "bad":
        video_path = bad_videos[i]
    else:
        video_path = good_videos[i]

    return video_path

def extract_frames(video_path, frames_per_second=2):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    # Calculate the interval at which to capture frames
    frame_interval = int(frame_rate / frames_per_second)

    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        # Capture the frame if it's the right interval
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            # Print the dimensions of the frame
            height, width, channels = frame.shape
            print(f"Frame {frame_count}: dimensions = {width}x{height} pixels")

        frame_count += 1

    video.release()
    print(len(base64Frames), "frames extracted.")
    return base64Frames


def show_extracted_frames(base64Frames):
    # Assuming base64Frames is a list of base64 encoded images
    for img_base64 in base64Frames:
        # Decode the base64 string
        img_bytes = base64.b64decode(img_base64.encode("utf-8"))
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Display the image
        cv2.imshow('Frame', img)

        # Wait for 25 ms and check if the 'Esc' key is pressed
        if cv2.waitKey(25) & 0xFF == 27:  # 27 is the ASCII code for the 'Esc' key
            break

    # Close the window
    cv2.destroyAllWindows()

def describe_img(client, text_prompt, base64_image, max_tokens=200, detail_level="low"):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64, {base64_image}", "detail": detail_level}}
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": max_tokens,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)  # print out the response
    print(result.usage)  # print out how many tokens were used


def describe_video(client, text_prompt, base64_frames, max_tokens=200, detail_level="low"):
    # Incorporating lambda function to iterate through base64_frames
    content = list(map(
        lambda frame: {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpg;base64, {frame}", 
                "detail": detail_level
            }
        }, 
        base64_frames
    ))

    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                *content  # Using the list of dictionaries generated by the lambda function
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": max_tokens,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)  # print out the response
    print(result.usage)  # print out how many tokens were used

    return result


def describe_video_0shot(client, system_prompt, user_prompt, base64_frames, max_tokens=200, detail_level="low", s=7, temp=0.7):
    # Incorporating lambda function to iterate through base64_frames
    content = list(map(
        lambda frame: {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpg;base64, {frame}", 
                "detail": detail_level
            }
        }, 
        base64_frames
    ))

    prompt_messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *content  # Using the list of dictionaries generated by the lambda function
            ],
        },
    ]
    
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "seed": s,
        "temperature": temp
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)  # print out the response
    print(result.usage)  # print out how many tokens were used

    return result



def save_results_to_csv(results_folder, results_file, video_path, choosen_quality, choosen_category, result, frames_per_second, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature):
    # create result file path
    full_path = os.path.join(results_folder, results_file)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary


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

    df = pd.DataFrame([data])

    if os.path.isfile(full_path):
        df.to_csv(full_path, mode='a', header=False, index=False)
        print(f"Data appended to {full_path}")
    else:
        df.to_csv(full_path, mode='w', header=True, index=False)
        print(f"Data written to {full_path}")

def process_and_describe_video(i, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos, frames_per_second, client, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature, results_folder, results_file):
    video_path = select_video_path(i, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos)
    extracted_frames = extract_frames(video_path, frames_per_second)
    show_extracted_frames(extracted_frames)

    result = describe_video_0shot(client, system_prompt, user_prompt, base64_frames=extracted_frames, max_tokens=max_tokens, detail_level=detail_level, s=seed, temp=temperature)
    save_results_to_csv(results_folder, results_file, video_path, choosen_quality, choosen_category, result, frames_per_second, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature)

def process_videos_in_range(start_index, end_index, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos, frames_per_second, client, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature, results_folder, results_file):
    for i in range(start_index, end_index):
        process_and_describe_video(i, choosen_quality, choosen_category, bad_videos_by_category, bad_videos, good_videos, frames_per_second, client, system_prompt, user_prompt, max_tokens, detail_level, seed, temperature, results_folder, results_file)
