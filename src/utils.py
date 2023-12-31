import os
import glob
import re

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
