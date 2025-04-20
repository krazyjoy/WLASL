import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os


def drop_z_axis(sequence, num_keypoints=75):
    # sequence shape: [T, D] where D = 3 * num_keypoints
    # Drop every 3rd element (z) in the flattened keypoints
    T, D = sequence.shape
    if D < 3 * num_keypoints:
        return sequence.reshape(T, num_keypoints, 2)
    reshaped = sequence[:, :3*num_keypoints].reshape(T, num_keypoints, 3)
    xy = reshaped[:, :, :2]
    return xy.reshape(T, num_keypoints, 2)

def normalize_keypoints(sequence):
    # sequence: (T, 75, 2)
    if isinstance(sequence, np.ndarray):
        sequence = torch.tensor(sequence, dtype=torch.float32)
    reference = sequence[:, 0:1, :]  # e.g., nose or middle of body
    centered = sequence - reference  # shift
    scale = torch.norm(centered[:, 11, :] - centered[:, 12, :], dim=-1).mean()  # shoulder width
    return centered / (scale + 1e-6)



class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, data_json, seq_len, num_files, label_map):
       
        self.files = self.create_files(data_dir, data_json, num_files)
        print("len(self.files): ", len(self.files))
        self.label_map = label_map
        self.seq_len = seq_len


    def create_files(self, data_dirs, data_json, num_files):
        with open(data_json, "r") as f:
            video_data = json.load(f)
        video_list = []
        for video_id in video_data:
            video_list.append(video_id)

        files = []
        for data_dir in data_dirs:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.pickle')]
            
            for f in all_files:
                if len(files) == num_files:
                    break
                base = os.path.splitext(f)[0]
                for video_id in video_list:
                    if base == video_id or base.startswith(f"{video_id}_aug"):
                        files.append(os.path.join(data_dir, f))

        # files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pickle') and os.path.splitext(f)[0] in video_list]
        return files
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print(f"idx: {idx}, files[idx]: {self.files[idx]}")
        with open(self.files[idx], "rb") as f:
            data = pickle.load(f)

        sequence = np.array(data["keypoints"]) #[T, N]
        # print("sequence.shape: ", sequence.shape)
        label = self.label_map[data["label"]]
        
        sequence = drop_z_axis(sequence)
        
        sequence = normalize_keypoints(sequence)
        # mean = sequence.mean()
        # std = sequence.std() + 1e-6
        # sequence = (sequence - mean) / std

        if sequence.shape[0] < self.seq_len:
            pad = torch.zeros((self.seq_len - sequence.shape[0], sequence.shape[1], sequence.shape[2]))
            sequence = torch.cat((sequence, pad), dim=0)
        else:
            sequence = sequence[:self.seq_len]
        # print("padded sequence.shape: ", sequence.shape)
        return sequence, torch.tensor(label, dtype=torch.long)
