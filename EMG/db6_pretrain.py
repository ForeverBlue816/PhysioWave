import os
import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

###############################################################################
# 1) Z-score normalization function
###############################################################################

def zscore_normalize(emg_data):
    """
    Apply z-score normalization to emg_data (N, channels).
    
    Each channel is normalized to have zero mean and unit variance.
    A small epsilon (1e-12) is added to prevent division by zero.
    
    Args:
        emg_data: numpy array of shape (N, channels)
    
    Returns:
        Normalized data with same shape
    """
    mu = np.mean(emg_data, axis=0)
    sigma = np.std(emg_data, axis=0, ddof=1) + 1e-12
    return (emg_data - mu) / sigma

###############################################################################
# 2) Sliding window segmentation
###############################################################################

def sliding_window_emg(emg_array, wsize=1024, step=512):
    """
    Segment emg_array (N, channels) using sliding windows.
    
    Args:
        emg_array: Input EMG data of shape (N, channels)
        wsize: Window size (default 1024 samples)
        step: Step size between windows (default 512 samples)
    
    Returns:
        Segmented data of shape (B, channels, wsize) where B is number of windows
    """
    segments = []
    num_samples = emg_array.shape[0]
    num_channels = emg_array.shape[1]

    for start_idx in range(0, num_samples - wsize + 1, step):
        end_idx = start_idx + wsize
        window_data = emg_array[start_idx:end_idx, :]  # (wsize, channels)
        window_data = window_data.T                    # (channels, wsize)
        segments.append(window_data)

    if len(segments) == 0:
        return None
    return np.stack(segments, axis=0)  # (B, channels, wsize)

###############################################################################
# 3) Save to HDF5
###############################################################################

def save_to_h5(data, filepath):
    """
    Save data to HDF5 file with compression.
    
    Args:
        data: numpy array to save
        filepath: Output file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", dtype=np.float32)
    print(f"Saved {data.shape[0]} samples to {filepath}")

###############################################################################
# 4) Main processing pipeline
###############################################################################

if __name__ == "__main__":
    # DB6 sampling rate is 2000 Hz
    FS = 2000
    
    # Select which 8 channels to keep from the original 14 channels
    # We'll keep the first 8 electrodes at the radio humeral joint level
    # (these are the most relevant for hand gesture recognition)
    CHANNELS_TO_KEEP = [0, 1, 2, 3, 4, 5, 6, 7]  # First 8 channels after removing bad ones

    # Input and output directories
    data_root = "path/to/DB6/data"  # Path to DB6 dataset folder
    output_dir = "./DB6_processed_8ch"  # Output directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary HDF5 file for memory-efficient processing
    temp_h5 = os.path.join(output_dir, "temp_all_data.h5")
    
    # Initialize temporary HDF5 file
    with h5py.File(temp_h5, "w") as f:
        f.create_dataset("data", shape=(0, 8, 1024),
                         maxshape=(None, 8, 1024),
                         compression="gzip",
                         dtype=np.float32)

    total_samples = 0

    # Process each subject folder
    for subj_folder in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj_folder)
        if not os.path.isdir(subj_path):
            continue
        print(f"\nProcessing folder: {subj_folder}")

        # Process each .mat file in the subject folder
        for mat_file in sorted(os.listdir(subj_path)):
            if not mat_file.endswith(".mat"):
                continue
            mat_path = os.path.join(subj_path, mat_file)
            print(f"  Loading {mat_file}")

            # Load .mat file
            mat_data = loadmat(mat_path)
            if "emg" not in mat_data:
                print(f"    => No 'emg' field in file, skipping.")
                continue

            emg_raw = mat_data["emg"]  # (N, 16) - DB6 has 16 channels total
            print(f"    Original shape: {emg_raw.shape}")

            # ========= 1) Remove bad channels (index 8, 9) ========= #
            # According to DB6 documentation, channels 8 and 9 are problematic
            print(f"    => Removing bad channels (index 8, 9)")
            emg_clean = np.delete(emg_raw, [8, 9], axis=1)  # Now (N, 14)
            print(f"    After removing bad channels: {emg_clean.shape}")

            # ========= 2) Select only 8 channels ========= #
            # Keep the first 8 channels
            print(f"    => Selecting 8 channels")
            emg_8ch = emg_clean[:, CHANNELS_TO_KEEP]  # (N, 8)
            print(f"    After channel selection: {emg_8ch.shape}")

            # ========= 3) Z-score normalization ========= #
            print(f"    => Applying z-score normalization")
            emg_normalized = zscore_normalize(emg_8ch)

            # ========= 4) Sliding window segmentation ========= #
            print(f"    => Applying sliding window segmentation")
            emg_segmented = sliding_window_emg(emg_normalized, wsize=1024, step=512)
            
            if emg_segmented is None:
                print("    => Not enough samples for one window, skipping.")
                continue

            # Convert to float32
            emg_segmented = emg_segmented.astype(np.float32)
            print(f"    => Generated {emg_segmented.shape[0]} windows")
            
            # Append to temporary HDF5 file
            with h5py.File(temp_h5, "a") as f:
                old_shape = f["data"].shape
                new_shape = (old_shape[0] + emg_segmented.shape[0], 8, 1024)
                f["data"].resize(new_shape)
                f["data"][-emg_segmented.shape[0]:] = emg_segmented
            
            total_samples += emg_segmented.shape[0]

    # Split and save train/validation sets
    if total_samples > 0:
        print(f"\n" + "="*50)
        print(f"Total samples collected: {total_samples}")
        print("Creating train/validation split...")
        
        # Load all data from temporary file
        with h5py.File(temp_h5, "r") as f:
            all_data = f["data"][:]
        
        # Create 80:20 train/validation split
        indices = np.arange(len(all_data))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Split data
        train_data = all_data[train_idx]
        val_data = all_data[val_idx]
        
        # Save train and validation sets
        train_path = os.path.join(output_dir, "train.h5")
        val_path = os.path.join(output_dir, "val.h5")
        
        save_to_h5(train_data, train_path)
        save_to_h5(val_data, val_path)
        
        # Clean up temporary file
        os.remove(temp_h5)
        print(f"Removed temporary file: {temp_h5}")
        
        # Print summary
        print(f"\n" + "="*50)
        print(f"Dataset processing complete!")
        print(f"Train set: {train_data.shape[0]} samples, shape: {train_data.shape}")
        print(f"Val set: {val_data.shape[0]} samples, shape: {val_data.shape}")
        print(f"Output directory: {output_dir}")
        print(f"\nData format:")
        print(f"  - 8 channels")
        print(f"  - 1024 samples per window at 2000 Hz")
        print(f"  - Z-score normalized")
        print(f"  - No additional filtering applied")
        
    else:
        print("\nError: No valid data found!")
        # Clean up empty temporary file
        if os.path.exists(temp_h5):
            os.remove(temp_h5)