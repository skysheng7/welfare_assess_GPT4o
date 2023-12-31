import os
import glob

def get_video_paths(root_folder):
    good_videos = []
    bad_videos = []
    bad_videos_by_category = {'approach': [], 'direction': [], 'human': [], 'run': [], 'slip': [], 'stop': [], 'two': []}

    # Iterate through each cow's folder
    for cow_id_folder in glob.glob(os.path.join(root_folder, '*/')):
        # Paths for good and bad videos inside each cow's folder
        good_folder = os.path.join(cow_id_folder, 'good')
        bad_folder = os.path.join(cow_id_folder, 'bad')

        # Add good video paths
        good_videos.extend(glob.glob(os.path.join(good_folder, '**/*.mp4'), recursive=True))
        good_videos.extend(glob.glob(os.path.join(good_folder, '**/*.MP4'), recursive=True))

        # Add bad video paths and categorize them
        for category in bad_videos_by_category.keys():
            category_folder = os.path.join(bad_folder, category)
            category_videos = glob.glob(os.path.join(category_folder, '*.mp4')) + glob.glob(os.path.join(category_folder, '*.MP4'))
            bad_videos.extend(category_videos)
            bad_videos_by_category[category].extend(category_videos)

    return good_videos, bad_videos, bad_videos_by_category

