import os
import glob
import re
import numpy as np
import cv2
import base64

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
