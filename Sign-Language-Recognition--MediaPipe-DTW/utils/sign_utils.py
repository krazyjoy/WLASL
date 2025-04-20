import cv2
import mediapipe as mp
import os
import numpy as np
import pickle
import json
import time

holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def save_landmark_tensors_from_video(video_name, sign_name, video_dir, landmark_dir, split, NUM_FRAMES):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}

    cap = cv2.VideoCapture(os.path.join(video_dir, video_name+'.mp4'))
    
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
        return
    os.makedirs(os.path.join("data", landmark_dir, split), exist_ok=True)
    save_path = os.path.join("data", landmark_dir, split, f"{video_name}.pickle")
    if os.path.exists(save_path):
        print(f"pickle file {save_path} is already generated")
        with open(save_path, "rb") as f:
            data = pickle.load(f)

        sequence = np.array(data["keypoints"]) #[T, N]
        print("sequence.shape: ", sequence.shape)
        return
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        pose = results.pose_landmarks.landmark if results.pose_landmarks else None
        lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
        rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
        pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose]) if pose else np.zeros((33, 3))
        lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lh]) if lh else np.zeros((21, 3))
        rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in rh]) if rh else np.zeros((21, 3))

        frame_vec = np.concatenate([pose, lh, rh]).flatten() # (225)
        # normalization
    
        landmark_list["pose"].append(frame_vec)
    cap.release()

    sequence = np.array(landmark_list["pose"])
    
    # print("frame_count: ", frame_count)
    # if sequence.shape[0] < NUM_FRAMES:
    #     padding = np.zeros((NUM_FRAMES - sequence.shape[0], sequence.shape[1]))
    #     sequence = np.concatenate((sequence, padding), axis=0)

    # else:
    #     sequence = sequence[:NUM_FRAMES]
    
    data_dict = {"keypoints": sequence, "label": sign_name}
    os.makedirs(os.path.join("data", landmark_dir, split), exist_ok=True)
    save_path = os.path.join("data", landmark_dir, split, f"{video_name}.pickle")
    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"saved: {save_path}")

    

if __name__ == "__main__":
    clean_videos_json = "../start_kit/WLASL2000.json"
    with open(clean_videos_json, "r") as f:
        clean_videos = json.load(f)
        
    glosses = set()
 
    for entry in clean_videos:
        for inst in entry['instances']:
        
            video_id = inst['video_id']
            gloss = entry['gloss']
            split = inst['split']
            if split != "test":
                continue
            """
            video_path = os.path.join(f"./data/videos/{split}", f"{video_id}.mp4")
            video_path = os.path.join(f"../start_kit/reencoded_videos", f"{video_id}.mp4")
            """
            video_path = os.path.join(f"./data/videos/{split}", f"{video_id}.mp4")
            if os.path.exists(video_path):
                glosses.add(gloss)
                save_landmark_tensors_from_video(video_name=video_id, sign_name=gloss, video_dir=f"./data/videos/{split}", landmark_dir="landmarks", split=split, NUM_FRAMES=64)
    print("len (gloss): ", len(glosses))
