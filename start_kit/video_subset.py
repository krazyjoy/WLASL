import json
import os
import shutil


json_file = "WLASL100.json"
video_source_dir = "/home/chuan194/work/roboticVision/WLASL/start_kit/raw_videos_mp4"
output_dir = "/home/chuan194/work/roboticVision/WLASL/start_kit/WLASL2000_videos"
os.makedirs(output_dir, exist_ok=True)

with open(json_file, "r") as f:
    wlasl_subset = json.load(f)

video_files = []
for entry in wlasl_subset:
    for instance in entry["instances"]:
        video_filename = instance["video_id"] + ".mp4"
        print("video_filename: ", video_filename)
        video_files.append(video_filename)


for video_file in video_files:
    src_path = os.path.join(video_source_dir, video_file)
    dest_path = os.path.join(output_dir, video_file)

    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
        print(f"Copied: {video_file}")
    else:
        print(f"Missing: {video_file}")

print(f"Finished copying {len(video_file)} videos to {output_dir}")
