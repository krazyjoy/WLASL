import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from scipy.interpolate import interp1d


# ---------------------- Augmentation Functions ----------------------

def add_noise(sequence, noise_std=0.01):
    noise = np.random.normal(0, noise_std, sequence.shape)
    return sequence + noise

def rotate_2d(sequence, angle_deg=10):
    angle_rad = np.deg2rad(np.random.uniform(-angle_deg, angle_deg))
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad),  np.cos(angle_rad)]])
    T, D = sequence.shape
    rotated = []
    for t in range(T):
        coords = sequence[t].reshape(-1, 2)
        rotated_coords = coords @ rot_matrix.T
        rotated.append(rotated_coords.flatten())
    return np.stack(rotated)

def temporal_warp(sequence, warp_factor_range=(0.9, 1.1)):
    warp_factor = np.random.uniform(*warp_factor_range)
    T, D = sequence.shape
    new_T = max(4, int(T * warp_factor))  # prevent collapse
    new_times = np.linspace(0, T-1, new_T)
    f = interp1d(np.arange(T), sequence, axis=0, kind='linear', fill_value="extrapolate")
    warped = f(new_times)
    if warped.shape[0] > T:
        warped = warped[:T]
    elif warped.shape[0] < T:
        pad = np.zeros((T - warped.shape[0], D))
        warped = np.concatenate((warped, pad), axis=0)
    return warped


# ---------------------- Utility Functions ----------------------

def drop_z_axis(sequence):
    # sequence shape: [T, D] where D = 3 * num_keypoints
    # Drop every 3rd element (z) in the flattened keypoints
    T, D = sequence.shape
    coords = sequence.reshape(T, -1, 3)
    coords_2d = coords[:, :, :2]  # drop z-axis
    return coords_2d.reshape(T, -1)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


# ---------------------- Main Augmentation Pipeline ----------------------

def augment_dataset(data_dir, output_dir, min_samples=20, augment_factor=5):
    os.makedirs(output_dir, exist_ok=True)

    label_counts = defaultdict(list)
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pickle')]

    # Step 1: Load all files and count class distribution
    for file in data_files:
        data = load_pickle(os.path.join(data_dir, file))
        label = data['label']
        label_counts[label].append(file)

    # Step 2: Identify underrepresented classes
    underrepresented = {label: files for label, files in label_counts.items() if len(files) < min_samples}

    print(f"Found {len(underrepresented)} underrepresented classes")

    # Step 3: For each underrepresented class, augment samples
    for label, files in tqdm(underrepresented.items(), desc="Augmenting data"):
        for file in files:
            original_data = load_pickle(os.path.join(data_dir, file))
            sequence = np.array(original_data["keypoints"])
            sequence = drop_z_axis(sequence)

            for i in range(augment_factor):
                aug_sequence = sequence.copy()

                if np.random.rand() < 0.5:
                    aug_sequence = add_noise(aug_sequence)
                if np.random.rand() < 0.5:
                    aug_sequence = rotate_2d(aug_sequence)
                if np.random.rand() < 0.5:
                    aug_sequence = temporal_warp(aug_sequence)

                new_data = {
                    "keypoints": aug_sequence.tolist(),
                    "label": original_data["label"]
                }

                base_name = os.path.splitext(file)[0]
                new_file = f"{base_name}_aug{i}.pickle"
                save_pickle(new_data, os.path.join(output_dir, new_file))

    print("Augmentation complete. Augmented files saved to:", output_dir)


# ---------------------- Run Example ----------------------

if __name__ == "__main__":
    input_dir = "../data/landmarks/train"       # change as needed
    output_dir = "../data/landmarks/augmented"  # change as needed
    augment_dataset(input_dir, output_dir, min_samples=15, augment_factor=5)
