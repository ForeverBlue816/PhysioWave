import glob
import json
import os
import h5py
import numpy as np
from tqdm.auto import tqdm

# Dataset parameters
fs, n_ch = 200.0, 8  # Sampling frequency and number of EMG channels

# Gesture mapping dictionary
gesture_map = {
    "noGesture": 0,
    "waveIn": 1,
    "waveOut": 2,
    "pinch": 3,
    "open": 4,
    "fist": 5,
    "notProvided": 6,  # Invalid gesture
}

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize the signal by its maximum absolute value per channel.
    
    Args:
        signal: Input signal array of shape (channels, samples)
    
    Returns:
        Normalized signal with same shape
    """
    max_abs_value = np.max(np.abs(signal), axis=1, keepdims=True)
    max_abs_value[max_abs_value == 0] = 1  # Avoid division by zero
    return signal / max_abs_value

def adjust_length(x: np.ndarray, max_len: int) -> np.ndarray:
    """
    Adjust signal to fixed length by truncation or zero-padding.
    
    Args:
        x: Input signal array of shape (channels, samples)
        max_len: Target sequence length
    
    Returns:
        Adjusted signal of shape (channels, max_len)
    """
    n_ch, seq_len = x.shape
    if seq_len >= max_len:
        return x[:, :max_len]
    padding = np.zeros((n_ch, max_len - seq_len), dtype=x.dtype)
    return np.concatenate((x, padding), axis=1)

def extract_emg_signal(data_struct: dict, seq_len: int) -> tuple[np.ndarray, int]:
    """
    Extract EMG signal and label from the data structure.
    
    Args:
        data_struct: Dictionary containing EMG data and gesture information
        seq_len: Target sequence length
    
    Returns:
        Tuple of (processed EMG signal, gesture label)
    """
    # Stack EMG channels and scale values
    emg = np.stack([emg_i for emg_i in data_struct["emg"].values()], dtype=np.float32) / 128
    # Adjust to target length
    emg = adjust_length(emg, seq_len)
    # Apply normalization
    emg = normalize_signal(emg)
    # Get gesture label
    label = gesture_map.get(data_struct.get("gestureName", "notProvided"), 6)
    return emg, label

def save_h5_file(file_path, emg_data, labels):
    """
    Save data to HDF5 file.
    
    Args:
        file_path: Output file path
        emg_data: List of EMG arrays
        labels: List of corresponding labels
    """
    with h5py.File(file_path, "w") as h5f:
        h5f.create_dataset("data", data=np.array(emg_data, dtype=np.float32))
        h5f.create_dataset("label", data=np.array(labels, dtype=np.int64))

def process_training_json(source_folder, seq_len, train_data, train_labels, val_data, val_labels):
    """
    Process training JSON files.
    
    Args:
        source_folder: Path to training JSON folder
        seq_len: Target sequence length
        train_data: List to store training samples
        train_labels: List to store training labels
        val_data: List to store validation samples
        val_labels: List to store validation labels
    """
    user_file_paths = glob.glob(os.path.join(source_folder, "user*", "user*.json"))
    
    for file_path in tqdm(user_file_paths, desc="Processing Training JSON Data", leave=False):
        with open(file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        # Process training samples
        for sample in user_data.get("trainingSamples", {}).values():
            emg, label = extract_emg_signal(sample, seq_len)
            if label == 6:  # Skip invalid gestures
                continue
            train_data.append(emg)
            train_labels.append(label)

        # Process testing samples as validation data
        testing_samples = list(user_data.get("testingSamples", {}).values())
        for sample in testing_samples:
            emg, label = extract_emg_signal(sample, seq_len)
            if label == 6:  # Skip invalid gestures
                continue
            val_data.append(emg)
            val_labels.append(label)

def process_testing_json(source_folder, seq_len, train_data, train_labels, test_data, test_labels):
    """
    Process testing JSON files.
    
    Args:
        source_folder: Path to testing JSON folder
        seq_len: Target sequence length
        train_data: List to store additional training samples
        train_labels: List to store additional training labels
        test_data: List to store test samples
        test_labels: List to store test labels
    """
    user_file_paths = glob.glob(os.path.join(source_folder, "user*", "user*.json"))
    
    for file_path in tqdm(user_file_paths, desc="Processing Testing JSON Data", leave=False):
        with open(file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        # Organize samples by gesture type
        gesture_map_data = {gesture: [] for gesture in gesture_map.keys()}

        for sample in user_data.get("trainingSamples", {}).values():
            gesture_name = sample.get("gestureName", "notProvided")
            if gesture_name in gesture_map_data:
                gesture_map_data[gesture_name].append(sample)

        # Split samples: first 10 for training, rest for testing
        for gesture, samples in gesture_map_data.items():
            for i, sample in enumerate(samples):
                emg, label = extract_emg_signal(sample, seq_len)
                if label == 6:  # Skip invalid gestures
                    continue
                if i < 10:
                    train_data.append(emg)
                    train_labels.append(label)
                else:
                    test_data.append(emg)
                    test_labels.append(label)

def main():
    # Define paths
    source_training = "path/to/EPN612/trainingJSON"  # Path to training JSON folder
    source_testing = "path/to/EPN612/testingJSON"    # Path to testing JSON folder
    dest_folder = "./EPN612_processed"               # Output folder
    seq_len = 1024                                   # Sequence length in samples

    # Create output directory
    os.makedirs(dest_folder, exist_ok=True)

    # Initialize data containers
    train_data, train_labels = [], []
    val_data, val_labels = [], []
    test_data, test_labels = [], []

    # Print processing information
    print("Processing EPN612 dataset with max-abs normalization")
    print(f"Sampling rate: {fs} Hz")
    print(f"Sequence length: {seq_len} samples")
    print(f"Preprocessing: Max absolute value normalization")
    print(f"Channels: {n_ch}")
    
    # Process data
    process_training_json(source_training, seq_len, train_data, train_labels, val_data, val_labels)
    process_testing_json(source_testing, seq_len, train_data, train_labels, test_data, test_labels)

    # Print dataset statistics
    print(f"\nSaving max-abs normalized datasets:")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save to HDF5 files
    save_h5_file(os.path.join(dest_folder, "epn612_train_set.h5"), train_data, train_labels)
    save_h5_file(os.path.join(dest_folder, "epn612_val_set.h5"), val_data, val_labels)
    save_h5_file(os.path.join(dest_folder, "epn612_test_set.h5"), test_data, test_labels)

    print("\nProcessing complete!")
    print("Data preprocessed with max absolute value normalization")

if __name__ == "__main__":
    main()