#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIMIC-IV-ECG Pretraining Data Preprocessing
---------------------------------------------------------
• Uses Z-score normalization (mean=0, std=1)
• Window length: 2048, sliding step: 1024
• Generates only train and val files for pretraining (data key only)
• Output HDF5 format: (N, 12, 2048)
• Sampling rate: 500Hz
"""

import os
import json
import h5py
import numpy as np
import wfdb
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ──────────────────── Global Parameters ────────────────────
FS = 500               # Sampling rate
WINDOW_SIZE = 2048     # Window size (~4 seconds)
STEP_SIZE = 1024       # Sliding step (50% overlap)
CHUNK_SIZE = 500       # HDF5 write batch size

TRAIN_RATIO = 0.85     # 85% train
VAL_RATIO = 0.15       # 15% val

# ──────────────────── Z-score Normalization ────────────────────
def zscore_normalize(windows, eps=1e-8):
    """
    Z-score normalization (mean=0, std=1)
    
    Args:
        windows: (num_win, 12, win_size) - Input windows
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized windows with same shape
    """
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + eps
    return ((windows - mean) / std).astype(np.float32)


# ──────────────────── Sliding Window ────────────────────
def sliding_window(ecg_2d, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Sliding window segmentation for ECG signals
    
    Args:
        ecg_2d: (12, L) - ECG signal
        window_size: Size of each window
        step_size: Step between windows
    
    Returns:
        list of (12, window_size) arrays
        Pads with zeros if last window is incomplete
    """
    segs = []
    L = ecg_2d.shape[1]
    
    starts = range(0, L, step_size)
    
    for start in starts:
        if start + window_size <= L:
            # Complete window
            seg = ecg_2d[:, start:start+window_size]
        else:
            # Incomplete window: pad with zeros
            seg = np.zeros((12, window_size), dtype=ecg_2d.dtype)
            remaining = L - start
            seg[:, :remaining] = ecg_2d[:, start:]
        
        segs.append(seg)
    
    return segs


# ──────────────────── HDF5 Operations ────────────────────
def create_pretrain_h5(path, data_shape, chunk_size=CHUNK_SIZE):
    """Create pretraining HDF5 file (data key only)"""
    f = h5py.File(path, "w")
    f.create_dataset(
        "data",
        shape=(0,) + data_shape,
        maxshape=(None,) + data_shape,
        chunks=(chunk_size,) + data_shape,
        dtype=np.float32,
        compression="gzip",
        compression_opts=4
    )
    return f


def append_pretrain_h5(f, data_batch):
    """Append data to pretraining HDF5 file"""
    ds = f["data"]
    old_size = ds.shape[0]
    new_size = old_size + data_batch.shape[0]
    ds.resize((new_size,) + ds.shape[1:])
    ds[old_size:new_size] = data_batch


# ──────────────────── Process Single Split ────────────────────
def process_mimic_split(record_paths, output_h5, split_name, input_dir):
    """
    Process a single data split (for pretraining: no labels needed)
    Uses optimized progress bar display to reduce flickering
    """
    hf = create_pretrain_h5(output_h5, data_shape=(12, WINDOW_SIZE))
    
    data_buffer = []
    valid_records = 0
    total_windows = 0
    skipped = 0
    
    # Use smooth progress bar with controlled update frequency
    pbar = tqdm(total=len(record_paths), 
                desc=f"{split_name.capitalize()}", 
                unit='rec',
                ncols=100,
                miniters=1,  # Minimum update interval
                mininterval=0.5)  # Minimum time interval (seconds)
    
    for idx, rec_path in enumerate(record_paths):
        # Build full path
        full_path = os.path.join(input_dir, rec_path)
        rec_name = os.path.splitext(os.path.basename(rec_path))[0]
        rec_path_no_ext = os.path.join(os.path.dirname(full_path), rec_name)
        
        try:
            # Read ECG signal
            signals, _ = wfdb.rdsamp(rec_path_no_ext)
            ecg = signals.T  # (N, 12) -> (12, N)
        except Exception as e:
            skipped += 1
            pbar.update(1)
            continue
        
        # Handle NaN values
        np.nan_to_num(ecg, copy=False)
        
        # Check signal length
        if ecg.shape[1] < WINDOW_SIZE:
            skipped += 1
            pbar.update(1)
            continue
        
        # Sliding window segmentation
        segments = sliding_window(ecg)
        
        # Z-score normalization
        if segments:
            segs_array = np.array(segments)  # (num_win, 12, 2048)
            normalized = zscore_normalize(segs_array)
            data_buffer.extend(normalized)
            total_windows += len(normalized)
        
        valid_records += 1
        
        # Batch write
        if len(data_buffer) >= CHUNK_SIZE:
            data_batch = np.stack(data_buffer[:CHUNK_SIZE])
            append_pretrain_h5(hf, data_batch)
            data_buffer = data_buffer[CHUNK_SIZE:]
        
        # Update progress bar with key statistics
        pbar.set_postfix({
            'valid': valid_records,
            'skip': skipped,
            'windows': total_windows
        })
        pbar.update(1)
    
    pbar.close()
    
    # Write remaining data
    if data_buffer:
        data_batch = np.stack(data_buffer)
        append_pretrain_h5(hf, data_batch)
    
    # Print statistics
    final_shape = hf["data"].shape
    print(f"\n[Statistics] {split_name}")
    print(f"  Valid records: {valid_records}/{len(record_paths)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total windows: {total_windows}")
    print(f"  HDF5 shape: {final_shape}")
    
    hf.close()
    return valid_records, total_windows


# ──────────────────── Main Function ────────────────────
def main():
    # Root directory (containing record_list.csv)
    root_dir = "path/to/mimic"  # Update to your MIMIC-IV-ECG directory
    record_list_csv = os.path.join(root_dir, "record_list.csv")
    output_dir = os.path.join(root_dir, "processed_pretrain")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== MIMIC-IV-ECG Pretrain Preprocessing ===")
    
    # Read record list
    if os.path.exists(record_list_csv):
        df = pd.read_csv(record_list_csv)
        # Assume CSV has 'path' column, adjust if column name differs
        if 'path' in df.columns:
            all_paths = df['path'].tolist()
        else:
            # If no path column, try building from other columns
            # Adjust based on actual CSV structure
            all_paths = df.iloc[:, 0].tolist()  # Use first column
        
        # CSV paths already include 'files/' prefix, input_dir should be root
        input_dir = root_dir
    else:
        # If no CSV, scan directory
        print("Warning: record_list.csv not found, scanning directory...")
        input_dir = os.path.join(root_dir, "files")
        all_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.dat'):
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    all_paths.append(rel_path)
    
    print(f"Total records: {len(all_paths)}")
    print(f"Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}")
    print(f"Sampling rate: {FS} Hz")
    print("Normalization: Z-score (mean=0, std=1)")
    print("Task: Pretrain (data only, no labels)")
    
    # Data split (only need train and val)
    train_paths, val_paths = train_test_split(
        all_paths, 
        test_size=VAL_RATIO, 
        random_state=42, 
        shuffle=True
    )
    
    print(f"\nData splits:")
    print(f"  train: {len(train_paths)} records")
    print(f"  val:   {len(val_paths)} records")
    
    # Save split information
    split_info = {
        'train': train_paths[:100],  # Save first 100 as example
        'val': val_paths[:100],
        'train_count': len(train_paths),
        'val_count': len(val_paths)
    }
    with open(os.path.join(output_dir, 'record_splits.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Process train set
    print(f'\n{"="*50}')
    print('Processing train split...')
    train_h5 = os.path.join(output_dir, 'mimic_pretrain_train.h5')
    train_valid, train_windows = process_mimic_split(
        train_paths, train_h5, 'train', input_dir
    )
    
    # Process val set
    print(f'\n{"="*50}')
    print('Processing val split...')
    val_h5 = os.path.join(output_dir, 'mimic_pretrain_val.h5')
    val_valid, val_windows = process_mimic_split(
        val_paths, val_h5, 'val', input_dir
    )
    
    # Save metadata
    meta = {
        "num_records": {
            "train": train_valid,
            "val": val_valid
        },
        "num_windows": {
            "train": train_windows,
            "val": val_windows
        },
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_channels": 12,
        "normalization": "z-score",
        "sampling_rate": FS,
        "task": "pretrain",
        "data_only": True,
    }
    
    meta_path = os.path.join(output_dir, 'mimic_pretrain_info.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f'\n{"="*50}')
    print("Preprocessing completed!")
    print(f"\nStatistics:")
    print(f"  Train: {train_valid} records, {train_windows} windows")
    print(f"  Val:   {val_valid} records, {val_windows} windows")
    print(f"\nOutput files:")
    print(f"  • mimic_pretrain_train.h5 - Training data")
    print(f"  • mimic_pretrain_val.h5   - Validation data")
    print(f"  • mimic_pretrain_info.json - Metadata")
    print(f"  • record_splits.json - Record split details")
    print(f"\nUsage example:")
    print(f"   train_loader, val_loader, _ = create_dataloaders(")
    print(f"       train_files='mimic_pretrain_train.h5',")
    print(f"       val_files='mimic_pretrain_val.h5',")
    print(f"       batch_size=32,")
    print(f"       task='pretrain',")
    print(f"       max_length=2048")
    print(f"   )")


if __name__ == "__main__":
    main()