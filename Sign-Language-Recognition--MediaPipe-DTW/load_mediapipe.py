import os
import numpy as np
from SignVideoDataset import load_mediapipe_annotated_frames, get_num_class, make_dataset, video_to_tensor, load_flow_frames, load_rgb_frames
import mediapipe
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
import random 

from torchvision import transforms
import videotransforms
import torch
import json
import shutil
import glob

def pad(imgs, label, total_frames):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    label = label[:, 0]
    label = np.tile(label, (total_frames, 1)).transpose((1, 0))

    return padded_imgs, label


def pad_wrap(imgs, label, total_frames):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            pad = imgs[:min(num_padding, imgs.shape[0])]
            k = num_padding // imgs.shape[0]
            tail = num_padding % imgs.shape[0]

            pad2 = imgs[:tail]
            if k > 0:
                pad1 = np.array(k * [pad])[0]

                padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
            else:
                padded_imgs = np.concatenate([imgs, pad2], axis=0)
    else:
        padded_imgs = imgs

    label = label[:, 0]
    label = np.tile(label, (total_frames, 1)).transpose((1, 0))

    return padded_imgs, label
def get_transforms(split: str):
    if split == "train" or split == "val":
        transform = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip() ])
        print("transform", transform)
    elif split == "test":
        transform = transforms.Compose([videotransforms.CenterCrop(224)])

    return transform

def mediapipe_dataset(data, split, holistic, webcam_manager, sign_recorder, output_dir):
    for index in range(len(data)):
        vid, label, src, start_frame, nf = data[index]
        if os.path.exists(f"./data/npy/{split}/{vid}.pt"):
            print(f"skipping create {vid}.pt")
            continue
        total_frames = 16
        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        try:
            if split == 'val':
                shutil.copy(f"../start_kit/val_videos/{vid}.mp4", f"{root['word']}/{split}/{vid}.mp4")
            imgs = load_mediapipe_annotated_frames(root['word'], vid, start_f, 
            total_frames, holistic, webcam_manager, sign_recorder, split)

            if imgs is None or len(imgs) == 0:
                raise ValueError(f"Failed to load RGB frames for video: {vid}")
                continue

            imgs, label = pad(imgs, label, total_frames)
            
            imgs = imgs.transpose(0, 2, 3, 1)
            transforms = get_transforms(split)
            imgs = transforms(imgs)
           
            ret_lab = torch.from_numpy(label)
            ret_img = video_to_tensor(imgs)
            del imgs
            if ret_img.shape != (3, 16, 224, 224):
                print("ret_img.shape: ", ret_img.shape)
                return None
            
            torch.cuda.empty_cache()
            torch.save(ret_img, os.path.join(output_dir, f"{vid}.pt"))
        except Exception as e:
            print(f"Skipping {vid} due to error: {e}")
            with open("failed_videos.txt", "a") as log_file:
                log_file.write(f"{vid}\n")
            return None
def prepare_tools():
    videos = load_dataset(mode='train')
    reference_signs = load_reference_signs(videos)
    holistic = mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    webcam_manager = WebcamManager()
    sign_recorder = SignRecorder(reference_signs)

    return holistic, webcam_manager, sign_recorder

def prepare_mediapipe_json(split_file):
    
    with open(split_file, "r") as f:
        nslt_json = json.load(f)
    mediapipe_set = {}
    train_count = {}
    val_count = 0
    test_count = 0
    for entry in nslt_json:
        
        if nslt_json[entry]['subset'] == 'train':
            class_name = nslt_json[entry]['action'][0]
            if class_name not in train_count.keys():
                train_count[class_name] = 1
                mediapipe_set[entry] = nslt_json[entry]
            elif train_count[class_name] < 2:
                train_count[class_name] += 1
                mediapipe_set[entry] = nslt_json[entry]
        elif nslt_json[entry]['subset'] == 'val' and val_count < 500:
            if os.path.join("../start_kit/val_videos", f"{class_name}.mp4"):
                mediapipe_set[entry] = nslt_json[entry]
                val_count += 1
        elif nslt_json[entry]['subset'] == 'test' and test_count < 500:
            mediapipe_set[entry] = nslt_json[entry]
            test_count += 1
    
    all_class_ids = sorted(list({entry['action'][0] for entry in mediapipe_set.values()}))
    class_id_to_index = {orig_id: new_idx for new_idx, orig_id in enumerate(all_class_ids)}

    for entry in mediapipe_set.values():
        entry['original_action'] = entry['action'][0]
        entry['action'][0] = class_id_to_index[entry['action'][0]]


    with open("mediapipe_dataset.json", "w") as f:
        json.dump(mediapipe_set, f)

    print(f"Classes reduced to {len(class_id_to_index)} and remapped.")
    return len(class_id_to_index)

if __name__ == "__main__":
    split = "test"
    output_dir = f'./data/npy/{split}'

    os.makedirs(output_dir, exist_ok=True)
    root = {'word': './data/videos'}
    # for video_path in glob.glob(f"{root['word']}/*.mp4"):
    #     video_name = os.path.basename(video_path)
    #     dst_path = f"./data/videos/val/{video_name}"
    #     shutil.copy2(video_path, dst_path)

    source_file = '/home/chuan194/work/roboticVision/WLASL/code/I3D/preprocess/nslt_2000.json'
    with open(source_file, "r") as f:
        nslt_json = json.load(f)
    
    num_classes = prepare_mediapipe_json(source_file)
    split_file = "/home/chuan194/work/roboticVision/WLASL/Sign-Language-Recognition--MediaPipe-DTW/mediapipe_dataset.json"
    
    mode = 'rgb'
    num_classes = get_num_class(split_file)
    data = make_dataset(split_file, split, root, mode, num_classes)
    holistic, webcam_manager, sign_recorder = prepare_tools()
    mediapipe_dataset(data, split, holistic, webcam_manager, sign_recorder, output_dir)