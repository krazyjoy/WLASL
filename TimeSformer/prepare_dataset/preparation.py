import os
import json
import csv
import subprocess
import random

def format_annotation(json_file: str, csv_file: str, timesformer_annotation: str):
    print("input file: ", os.path.basename(json_file)[:5])
    # if os.path.basename(json_file)[:5] == "WLASL":
    with open(json_file, "r") as jsonf:
        WLASL_json = json.load(jsonf)
    
    # print("wlasl json: ", WLASL_json)
    
    with open(csv_file, "w", newline="") as csvf:
        writer = csv.writer(csvf, delimiter=' ')
        for entry in WLASL_json:
            gloss = entry["gloss"]
            for inst in entry["instance"]:
                path = os.path.join("./WLASL2000", f"{inst['video_id']}.mp4")
                if os.path.exists(path):
                    print("path: ", path)
                    writer.writerow([f"{inst['video_id']}.mp4", gloss])
                else:
                    print(f"{inst['video_id']}.mp4 does not exist")
        csvf.close()
    label_to_index = {}
    next_index = 1
    rows = []
    with open(csv_file, "r") as infile:
        for line in infile:
            # print("line: ", line)
            line = line.strip()
            if not line:
                continue
            video, label = line.split( maxsplit=1)
            # print(f"video: {video}, label: {label}")
            label = label.strip('"')
            # video, label = line.split(',')
            if label not in label_to_index:
                label_to_index[label] = next_index
                next_index += 1
            index = label_to_index[label]
            rows.append([video, index])
    with open(timesformer_annotation, "w", newline="") as timesfile:
        writer = csv.writer(timesfile, delimiter=" ")
        writer.writerows(rows)
    print("total labels: ", next_index)
    


def resize_videos(input_folder: str, output_folder: str, size: int, fps: int):
    input_folder = input_folder
    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)
    video_extension = '.mp4'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(video_extension):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            command = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"scale='if(gt(iw,ih),-2,{size})':'if(gt(iw,ih),{size},-2)',fps={fps}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-y",
                output_path
            ]
            print(f"Processing: {filename}")
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("All videos resized")



import av
import os

def check_crop_compatibility(video_path, min_scale=256, crop_size=224):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        frame = next(container.decode(video=0))  # decode one frame
        height, width = frame.height, frame.width
        container.close()

        short_side = min(height, width)
        if short_side < min_scale:
            print(f"❌ {video_path} - Too small for TRAIN_JITTER_SCALES (short side = {short_side})")
            return False
        if height < crop_size or width < crop_size:
            print(f"❌ {video_path} - Too small to crop {crop_size}x{crop_size} (actual: {width}x{height})")
            return False

        print(f"✅ {video_path} - OK ({width}x{height})")
        return True

    except Exception as e:
        print(f"⚠️  Error loading {video_path}: {e}")
        return False


def split_csv(input_csv: str, train_csv: str, val_csv: str, test_csv: str,
    val_ratio: int, test_ratio: int, seed:int
):
    with open(input_csv, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        # print("data: ", data[:2])
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data)*(1-test_ratio) * (1-val_ratio))
    print("train_len ", split_idx)
    train_data = data[:split_idx]
    
    test_idx = int(len(data)*(1-test_ratio))
    val_data = data[split_idx:test_idx]
    
    print("val len: ",len(val_data))
    test_data = data[test_idx:]
    print("test len: ", len(test_data))

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_data)

    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(val_data)

    with open(test_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

    print(f"Total samples: {len(data)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

def rewrite_clean(input_json: str, output_json: str):
    with open(input_json, "r") as f:
        input_data = json.load(f)
    
    json_list = []
    for entry in input_data:
        new_instances = []
        for instance in entry['instance']:
            path = f"./WLASL2000/{instance['video_id']}.mp4"
            if os.path.exists(path):
                new_instances.append(instance)
                new_json = {
                    "gloss": entry['gloss'],
                    "instance": new_instances
                }
                # print("new_json: ", new_json)
                json_list.append(new_json)
    
    with open(output_json, "w") as f:
        json.dump(json_list, f)

if __name__ == "__main__":
    # format_annotation(json_file="./cleaner_video2000.json", csv_file="./WLASL2000/WLASL2000.csv", timesformer_annotation="./WLASL2000/timesformer_data2000.csv")
    # resize_videos(input_folder="../../data/WLASL2000", output_folder="./WLASL2000", size=256, fps=30)
    # Example usage
    # video_folder = "./WLASL2000"
    # for fname in os.listdir(video_folder):
    #     if fname.endswith(".mp4"):
    #         check_crop_compatibility(os.path.join(video_folder, fname))
    split_csv(input_csv="./WLASL2000/timesformer_data2000.csv", train_csv="./WLASL2000/train.csv", val_csv="./WLASL2000/val.csv", test_csv="./WLASL2000/test.csv", val_ratio=0.2, test_ratio=0.3, seed=42)
    # rewrite_clean(input_json='clean_video2000.json', output_json='cleaner_video2000.json')