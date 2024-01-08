import os
from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# define result directory
results_folder = '../results'
results_file = 'video_dur_compare_good_VS_run.csv' # store the results in a csv file

# get a list of good videos, bad videos, bad videos by categories
os.chdir('C:/Users/skysheng/OneDrive - UBC/R package project and Git/lameness_GPT4V/src')
from utils import *
root_folder_path = 'C:/Users/skysheng/OneDrive - UBC/University of British Columbia/Research/PhD Project/Amazon project phase 2/Kay Yang/sorted_cow_videos_all'
good_videos, bad_videos, bad_videos_by_category = get_video_paths(root_folder_path)

# record duration of each video that were classified as good and bad (due to the cow runs)
df = create_duration_dataframe(good_videos, bad_videos_by_category["run"])

# Save to CSV
results_path = os.path.join(results_folder, results_file)
df.to_csv(results_path, index=False)
