import cv2
import torch
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
import json
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from pytorch_i3d import InceptionI3d
import contextlib
import numpy as np
import torch.nn.functional as F
import shutil
import csv


FRAME_SIZE = 224
NUM_FRAMES = 32

def count_classes():
    with open("./clean_video2000.json", "r") as f:
        data = json.load(f)
        class_names = []
        for entry in data:
            class_names.append(entry['gloss'])

    return class_names, len(class_names)
    
class_names, NUM_CLASSES  = count_classes()


@contextlib.contextmanager
def suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)


def load_i3d_model(weight_path: str):
    model = InceptionI3d(2000, in_channels=3)
    model.replace_logits(2000)
    model.load_state_dict(torch.load(weight_path))
    model.eval().cuda()
    return model

def preprocess_frame(frame):
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  #[3, H, W]
    return tensor


def predict_gloss(model, frame_stack):
    video_tensor = torch.stack(frame_stack, dim=1).unsqueeze(0).cuda()
    with torch.no_grad():
        logits = model(video_tensor) #[1, 2000, T] , T: time frame
        reduced_logits = torch.max(logits, dim=2)[0]
        
        probs = F.softmax(reduced_logits[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, k=5)
        top_label = class_names[top_idxs[0]]
        top_prob = top_probs[0].item()
        return top_label, top_prob


def mediapipe_processed_eval(input_path: str, videos: list, reference_signs: list, class_name: str):
   
    # === specificy output video file ===
    gloss = input_path.split("/")[-2]
    filename = input_path.split("/")[-1]
    output_path = f"./data/processed/{gloss}/{filename}"
    matches = []
    # === Initialize SignRecorder and WebcamManager ===
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()

    # === Video I/O Setup ===
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video:", input_path)
        return 2

    model = load_i3d_model('/home/chuan194/work/roboticVision/WLASL/code/I3D/checkpoints/nslt_2000_041003_0.397727.pt')
    frame_buffer = []


    # === Mediapipe Setup ===
    with suppress_stderr():
        import mediapipe
        with mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            print("📽️ Processing video...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("✅ Done processing.")
                    break

                # Run Mediapipe detection
                image, results = mediapipe_detection(frame, holistic)

                # Process results using SignRecorder
                sign_detected, is_recording = sign_recorder.process_results(results)

                # Update frame using WebcamManager (draw + optional info)
                webcam_manager.update(frame, results, sign_detected, is_recording)


                # Preprocess for I3D
                tensor_frame = preprocess_frame(image)
                frame_buffer.append(tensor_frame)

                if len(frame_buffer) > NUM_FRAMES:
                    frame_buffer = frame_buffer[-NUM_FRAMES:]
                
                if len(frame_buffer) == NUM_FRAMES:
                    gloss, prob = predict_gloss(model, frame_buffer)
                    # print(f"prediction: {gloss} ({prob:.2f})")
                    # print(f"correct label: {class_name}")
                    if gloss == class_name:
                        matches.append(True)
                    else:
                        matches.append(False)

                    

            cap.release()
    if not matches or sum(matches)/len(matches) > 0.5:
        return 1
    return 0


def mediapipe_processed_train(input_path: str, videos: list, reference_signs: list, class_name: str):
   
    # === specificy output video file ===
    gloss = input_path.split("/")[-2]
    filename = input_path.split("/")[-1]
    output_path = f"./data/processed/{gloss}/{filename}"
    matches = []
    # === Initialize SignRecorder and WebcamManager ===
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()

    # === Video I/O Setup ===
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video:", input_path)
        return 2

    model = load_i3d_model('/home/chuan194/work/roboticVision/WLASL/code/I3D/checkpoints/nslt_2000_041003_0.397727.pt')
    frame_buffer = []


    # === Mediapipe Setup ===
    with suppress_stderr():
        import mediapipe
        with mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            print("📽️ Processing video...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("✅ Done processing.")
                    break

                # Run Mediapipe detection
                image, results = mediapipe_detection(frame, holistic)

                # Process results using SignRecorder
                sign_detected, is_recording = sign_recorder.process_results(results)

                # Update frame using WebcamManager (draw + optional info)
                webcam_manager.update(frame, results, sign_detected, is_recording)


                # Preprocess for I3D
                tensor_frame = preprocess_frame(image)
                frame_buffer.append(tensor_frame)

                if len(frame_buffer) > NUM_FRAMES:
                    frame_buffer = frame_buffer[-NUM_FRAMES:]
                
                if len(frame_buffer) == NUM_FRAMES:
                    gloss, prob = predict_gloss(model, frame_buffer)
                    # print(f"prediction: {gloss} ({prob:.2f})")
                    # print(f"correct label: {class_name}")
                    if gloss == class_name:
                        matches.append(True)
                    else:
                        matches.append(False)

                    

            cap.release()

    return 0

def read_json(mode: str, input_json: str, videos: list, reference_signs: list):
    
    with open("./dataset.json", "r") as f:
        split = json.load(f)

    train_set = set(split['train'])
    test_set = set(split['test'])
    with open(input_json, "r") as f:
        data = json.load(f)
    
    with open("mediapipe_i3d_prediction.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'class', 'prediction'])
        writer.writeheader()
    
    num_sample = 0
    correct_prediction = 0
    no_prediction = 0
    results = []
    stop = False

    for entry in data:
        video_dir = "./data/videos/" + mode
        class_name = entry['gloss']
        if stop:
            break
        for instance in entry["instance"]:
            result = {}
            result['video_id'] = instance['video_id']
            if (mode == 'train' and result['video_id'] in train_set) or (mode=='test' and result['video_id'] in test_set):

                result['class'] = class_name
                num_sample += 1
                
                video = instance['video_id'] + '.mp4'
                print("video: ", video)
                
                input_path = os.path.join(video_dir, video)
                match = mediapipe_processed(input_path, videos, reference_signs, class_name)
                if match==1:
                    correct_prediction += 1
                if match==2:
                    no_prediction += 1
                print(f"num_sample: {num_sample}, correct_prediction: {correct_prediction}")
                result['prediction'] = match
                results.append(result)

                if num_sample % 20 == 0:
                    with open("mediapipe_i3d_prediction.csv", "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=['video_id', 'class', 'prediction'])
                        writer.writerows(results)
                    acc = correct_prediction / (num_sample-no_prediction)
                    print("acc: ", acc)
                    results.clear()
            else:
                print(f"Video {result['video_id']} is not in {mode} set")
    


def split_train_test_ds(split_json: str):
    with open(split_json, "r") as f:
        ref = json.load(f)

    train_ids = set()
    test_ids = set()

    for instance in ref:
        split = ref[instance]['subset']
        if split == "train":
            train_ids.add(instance)
        elif split == "test":
            test_ids.add(instance)
    print("train ids: ", train_ids)
    print(f"train length: {len(train_ids)}, test length: {len(test_ids)}")


    root_dir = "./data/src"
    data_dir = "./data/videos"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    train_count = 0 
    test_count = 0
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        print("class_path: ", class_path)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            file_id, ext = os.path.splitext(filename)
            src_path = os.path.join(class_path, filename)
            print("src_path: ", src_path)
            if file_id in train_ids:
                dst_path = os.path.join(train_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Moved {filename} to train")
                train_count += 1
            elif file_id in test_ids:
                dst_path = os.path.join(test_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Moved {filename} to test")
                test_count += 1
    print(f"moved total train: {train_count}") # 6563
    print(f"moved total test: {test_count}") # 1255

def create_train_test_json():
    data_dir = "./data/videos"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_videos = set()
    test_videos = set()
    for video in os.listdir(train_dir):
        video_id, ext = os.path.splitext(video)
        train_videos.add(video_id)
    for video in os.listdir(test_dir):
        video_id, ext = os.path.splitext(video)
        test_videos.add(video_id)

    print("len(train_videos): ", len(train_videos))
    print("len(test_videos): ", len(test_videos))
    dataset = {'train': list(train_videos), 'test': list(test_videos)}

    with open('dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    split_json = "../code/I3D/preprocess/nslt_2000.json"
    # execute this line if your data/videos does not contain train test folders
    # split_train_test_ds(split_json)
    create_train_test_json()
    videos = load_dataset(mode='test') # load data from data/videos to data/dataset with pickle format
    print("len(videos): ", len(videos))
    reference_signs = load_reference_signs(videos)
    read_json(mode='test', input_json="./clean_video2000.json", videos=videos, reference_signs=reference_signs)
    