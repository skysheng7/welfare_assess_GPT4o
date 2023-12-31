from dotenv import load_dotenv
import os
from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests


# connect to OpenAI API
load_dotenv()  # This loads the variables (API key) from .env
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# get a list of good videos, bad videos, bad videos by categories
root_folder_path = 'C:\Users\skysheng\OneDrive - UBC\University of British Columbia\Research\PhD Project\Amazon project phase 2\Kay Yang\sorted_cow_videos_all'
good_videos, bad_videos, bad_videos_by_category = get_video_paths(root_folder_path)



# read in videos and seperate into frames
video = cv2.VideoCapture("data/bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")