import os
import torch
from torch.utils.data import Dataset
import cv2
import json
from torchvision import transforms
import videotransforms

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
import matplotlib.pyplot as plt
import mediapipe
import numpy as np
import random
from utils.dataset_utils import load_dataset, load_reference_signs

FRAME_SIZE = 256

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        except:
            print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  #[3, H, W]
    return tensor


def load_mediapipe_annotated_frames(vid_root, vid, start, num, holistic, webcam_manager, sign_recorder, mode, resize=(256, 256)):
    video_path = os.path.join(os.path.join(vid_root, mode), vid + '.mp4')
    
    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, image = vidcap.read()
        if not success:
            break

        image, results = mediapipe_detection(image, holistic)
        sign_detected, is_recording = sign_recorder.process_results(results)
        image = webcam_manager.update(image, results, sign_detected, is_recording)
        
        if image is None:
            continue

        tensor_frame = preprocess_frame(image)
        frames.append(tensor_frame)

    vidcap.release()

    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    count_skipping = 0
    count_test = 0
    for vid in data.keys():
        if vid == "26569":
            print("vid: 26569 split: ", split)
        if count_test== 400:
            break
        # if split == 'train':
        #     if data[vid]['subset'] not in ['train', 'val']:
        #         continue
        # else:
        #     if data[vid]['subset'] != 'test':
        #         continue
        #     if data[vid]['subset'] == "test":
        #         count_test += 1
        if data[vid]['subset'] != split:
            continue

        if split == 'val' or split == 'train':
            source_dir = 'train'
        else:
            source_dir = 'test'
        vid_root = os.path.join(root['word'], split)

        src = 0

        video_path = os.path.join(vid_root, vid + '.mp4')
        # print("video_path: ", video_path)
        if not os.path.exists(video_path):
            print("filepath does not exist: ", video_path)
            continue

        num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames - 0 < 9:
            print("Skip video ", vid)
            count_skipping += 1
            continue

        label = np.zeros((num_classes, num_frames), np.float32)

        for l in range(num_frames):
            c_ = data[vid]['action'][0]
            label[c_][l] = 1

        if len(vid) == 5:
            dataset.append((vid, label, src, 0, data[vid]['action'][2] - data[vid]['action'][1]))
        elif len(vid) == 6:  ## sign kws instances
            dataset.append((vid, label, src, data[vid]['action'][1], data[vid]['action'][2] - data[vid]['action'][1]))

        i += 1
    print("Skipped videos: ", count_skipping)
    print("make dataset: ", len(dataset))
    return dataset


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)

class SignVideoDataset(Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        self.num_classes = get_num_class(split_file)
        self.split = split
        print("init split: ", split)
        with open(split_file, "r") as f:
            videos_json = json.load(f)

        mediapipe_json = {}
        for video_pt in os.listdir(f"./data/npy/{split}"):
            video_id, ext = os.path.splitext(video_pt)
            if videos_json[video_id]['subset'] != split:
                continue
            # print(f"video_id {video_id} is in test subset")
            mediapipe_json[video_id] = videos_json[video_id]
        
        with open(f"./mediapipe_tensors_{split}.json", "w") as f:
            json.dump(mediapipe_json, f)
        split_file = f"./mediapipe_tensors_{split}.json"
        self.split_file = split_file
        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.error_log = "failed_videos.txt"

        if not os.path.exists(self.error_log):
            open(self.error_log, "w").close()

        self.webcam_manager = WebcamManager()
        videos = load_dataset(mode=split)
        reference_signs = load_reference_signs(videos)
        self.sign_recorder = SignRecorder(reference_signs)
        self.holistic = mediapipe.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, src, start_frame, nf = self.data[index]
        

        total_frames = 16

        
        try:
            start_f = random.randint(0, nf - total_frames - 1) + start_frame
        except ValueError:
            start_f = start_frame

        try:
            # imgs = load_mediapipe_annotated_frames(self.root['word'], vid, start_f, total_frames, self.holistic, self.webcam_manager, self.sign_recorder, self.split)
            imgs = torch.load(os.path.join(f"./data/npy/{self.split}", f"{vid}.pt"))
            imgs = imgs.permute(1, 0, 2, 3)
            # print("imgs.shape: ", imgs.shape)
            # add exception
            if imgs is None or len(imgs) == 0:
                raise ValueError(f"Failed to load RGB frames for video: {vid}")

            ret_img, label = self.pad(imgs, label, total_frames)
            # print(f"{vid}'s label: {label.shape}")
            # imgs = imgs.transpose(0, 2, 3, 1)
            # imgs = self.transforms(imgs)
           
        
            ret_lab = torch.from_numpy(label)

            
            ret_img = ret_img.permute(1, 0, 2, 3) # 16, 3, 224, 224
 
            torch.cuda.empty_cache()
            return ret_img, ret_lab, vid

        except Exception as e:
            print(f"Skipping {vid} due to error: {e}")
            with open(self.error_log, "a") as log_file:
                log_file.write(f"{vid}\n")
            return None

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
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

    @staticmethod
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


