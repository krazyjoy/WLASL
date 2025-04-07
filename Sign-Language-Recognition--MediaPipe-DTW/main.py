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


def mediapipe_processed(input_path: str, videos: list, reference_signs: list):
   
    # === specificy output video file ===
    gloss = input_path.split("/")[-2]
    filename = input_path.split("/")[-1]
    output_path = f"./data/processed/{gloss}/{filename}"

    # === Initialize SignRecorder and WebcamManager ===
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()

    # === Video I/O Setup ===
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Failed to open video:", input_path)
        exit()

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
                    print(f"prediction: {gloss} ({prob:.2f})")

                    

            cap.release()

        

def read_json(input_json: str, videos: list, reference_signs: list):
    
    with open(input_json, "r") as f:
        data = json.load(f)
        
    for entry in data:
        video_dir = "./data/videos/" + entry['gloss']
        for instance in entry["instance"]:
            video = instance['video_id'] + '.mp4'
            print("video: ", video)
            input_path = os.path.join(video_dir, video)
            mediapipe_processed(input_path, videos, reference_signs)
            break




if __name__ == "__main__":
    
    videos = load_dataset()
    reference_signs = load_reference_signs(videos)
    read_json(input_json="./clean_video2000.json", videos=videos, reference_signs=reference_signs)
