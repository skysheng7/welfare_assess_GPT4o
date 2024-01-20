import os
import glob
import pandas as pd
from datetime import datetime
import json



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

def save_pairwise_lame_csv(full_path, cow1, cow2, degree, video_path1, video_path2, result, system_prompt, user_prompt1, user_prompt2, frames_per_second, max_tokens, detail_level, seed, temperature):

    # extract content from result
    result_content = result.choices[0].message.content
    result_json = result_content.strip('```json\n').strip('```')
    json_data = json.loads(result_json) # Convert the string to a Python dictionary
    predict_winner = json_data.get('winner', 'NA')
    predict_loser = json_data.get('loser', 'NA')
    predict_degree = json_data.get('degree', 'NA')
    reason = json_data.get('reason', 'NA')

    # calculate usage
    output_token = result.usage.completion_tokens
    prompt_tokens = result.usage.prompt_tokens
    output_token_p = output_token_cost(output_token)
    prompt_tokens_p = input_token_cost(prompt_tokens)
    total_cost = round((output_token_p+prompt_tokens_p), 3)

    data = {

        "win_cow_path": video_path1,
        "lose_cow_path": video_path2,
        "true_win_cow": cow1,
        "true_lose_cow": cow2,
        "true_lame_dif": degree,
        "predict_win_cow": predict_winner,
        "predict_lose_cow": predict_loser,
        "predict_degree": predict_degree,
        "predict_reason": reason,
        "predict_result": result,
        "model": "gpt-4-vision-preview",
        "date": datetime.now().date(),
        "frames_per_second": frames_per_second, 
        "system_prompt": system_prompt,
        "user_prompt": user_prompt1 + "\n ** video 1 ** \n" + user_prompt2 + "\n ** video 2 ** \n",
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
