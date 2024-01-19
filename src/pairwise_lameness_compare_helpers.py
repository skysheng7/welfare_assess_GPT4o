import os
import glob

def list_all_mp4(folder_path):
    all_mp4 = []

    # Add good video paths from both good and duplicate folders
    all_mp4.extend(glob.glob(os.path.join(folder_path, '**/*'), recursive=True))

    # Filter for mp4 files, case-insensitive
    all_mp4 = [video for video in all_mp4 if video.lower().endswith('.mp4')]

    return all_mp4

def find_mp4_with_digits(all_mp4, digits):
    # Ensure the 'digits' parameter is a string
    digits = str(digits)

    # Check if the length of the digits is exactly 4
    if len(digits) != 4:
        raise ValueError("The 'digits' parameter must be exactly four digits long.")

    # Filter the list for paths containing the specified digits
    matching_paths = [path for path in all_mp4 if digits in path]

    return matching_paths