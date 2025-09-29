import os
import ast
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------
# Global Parameters (adjustable as needed)
# ------------------------------------------------------
FS = 500            # Sampling rate
WINDOW_SIZE = 2048  # Sliding window size
STEP_SIZE = 1024    # Sliding step
CHUNK_SIZE = 500    # Batch size for HDF5 writes

# 5 Superclasses for single-label classification
SUPERCLASS_MAP = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "CD":   3,
    "HYP":  4,
}
NUM_CLASSES = len(SUPERCLASS_MAP)

# ------------------------------------------------------
# 1. Build SCP code to diagnostic_class mapping
# ------------------------------------------------------
def load_diag_map(scp_csv_path):
    """Load SCP code to diagnostic class mapping"""
    df_scp = pd.read_csv(scp_csv_path, index_col=0)
    df_scp = df_scp[df_scp["diagnostic"] == 1]
    return {code: row["diagnostic_class"] for code, row in df_scp.iterrows()}

# ------------------------------------------------------
# 2. Parse scp_codes to single label (Priority: NORM < MI < STTC < CD < HYP)
# ------------------------------------------------------
def parse_scp_codes_single_label(scp_str, diag_map, threshold=80.0):
    """
    Parse SCP codes string to single label with priority selection.
    
    - Deserialize scp_codes string -> dict{code:likelihood}
    - Find all classes with likelihood >= threshold that are in SUPERCLASS_MAP
    - Select single label by priority: NORM < MI < STTC < CD < HYP
    - Returns single class index, or -1 if none found
    """
    try:
        scp_dict = ast.literal_eval(scp_str)
    except Exception:
        return -1

    # Collect all qualifying classes
    found_classes = []
    for code, lk in scp_dict.items():
        if lk < threshold:
            continue
        cls_name = diag_map.get(code)
        if cls_name in SUPERCLASS_MAP:
            found_classes.append(cls_name)
    
    if not found_classes:
        return -1
    
    # Select by priority (higher number = higher priority)
    # Priority: NORM=0 < MI=1 < STTC=2 < CD=3 < HYP=4
    priority_map = SUPERCLASS_MAP
    selected_class = max(found_classes, key=lambda x: priority_map[x])
    
    return SUPERCLASS_MAP[selected_class]

# ------------------------------------------------------
# 3. Read ECG signal (.dat/.hea) - Using 500Hz data
# ------------------------------------------------------
def read_ecg_data(record_path):
    """Read 500Hz ECG data"""
    sig, _ = wfdb.rdsamp(record_path)
    return sig.T  # Convert to (12, n_samples)

# ------------------------------------------------------
# 4. MinMax normalization (without bandpass filtering)
# ------------------------------------------------------
def normalize_ecg_minmax(ecg):
    """
    Apply MinMax normalization to [-1,1] range for each lead.
    
    Args:
        ecg: shape (12, n_samples)
    
    Returns:
        Normalized ECG signal
    """
    normalized = np.zeros_like(ecg)
    for i in range(ecg.shape[0]):
        channel_data = ecg[i]
        min_val = channel_data.min()
        max_val = channel_data.max()
        
        # Avoid division by zero
        if max_val - min_val > 1e-8:
            # Normalize to [-1, 1] range
            normalized[i] = 2 * (channel_data - min_val) / (max_val - min_val) - 1
        else:
            normalized[i] = 0.0  # If signal is constant, set to 0 (middle value)
            
    return normalized

# ------------------------------------------------------
# 5. Sliding window segmentation
# ------------------------------------------------------
def sliding_window_ecg(ecg, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Sliding window segmentation of ECG signal.
    Each window is independently MinMax normalized.
    """
    segs = []
    L = ecg.shape[1]
    
    for start in range(0, L - window_size + 1, step_size):
        seg = ecg[:, start:start+window_size]
        # MinMax normalize each window
        normalized_seg = normalize_ecg_minmax(seg)
        segs.append(normalized_seg)
    
    # Process last segment (if padding needed)
    if L % step_size != 0:
        pad = np.zeros((ecg.shape[0], window_size))
        rem = ecg[:, (L//step_size)*step_size:]
        pad[:, :rem.shape[1]] = rem
        # Normalize padded segment
        normalized_pad = normalize_ecg_minmax(pad)
        segs.append(normalized_pad)
        
    return segs

# ------------------------------------------------------
# 6. HDF5 creation/writing (compatible format: 'data' & 'label')
# ------------------------------------------------------
def create_h5(path, data_shape, chunk_size=CHUNK_SIZE):
    """Create HDF5 file with compatible format"""
    f = h5py.File(path, 'w')
    
    # Create 'data' dataset (N, C, T)
    f.create_dataset(
        'data',
        shape=(0,) + data_shape,
        maxshape=(None,) + data_shape,
        chunks=(chunk_size,) + data_shape,
        dtype=np.float32
    )
    
    # Create 'label' dataset (N,) - single label
    f.create_dataset(
        'label',
        shape=(0,),
        maxshape=(None,),
        chunks=(chunk_size,),
        dtype=np.int64  # Use int64 to match torch.long
    )
    
    return f

def append_h5(f, data_batch, label_batch):
    """Append data to HDF5 file"""
    x_ds = f['data']
    y_ds = f['label']
    
    old_size = x_ds.shape[0]
    new_size = old_size + data_batch.shape[0]
    
    # Resize datasets
    x_ds.resize((new_size,) + x_ds.shape[1:])
    y_ds.resize((new_size,))
    
    # Write new data
    x_ds[old_size:new_size] = data_batch
    y_ds[old_size:new_size] = label_batch

# ------------------------------------------------------
# 7. Process one split
# ------------------------------------------------------
def process_split(df, root, out_file, diag_map, threshold=50.0):
    """Process a single data split (train/val/test)"""
    
    # Check sample shape
    example_path = os.path.join(root, df.iloc[0]['filename_hr'])
    example = read_ecg_data(example_path)
    
    # Ensure at least 12 leads
    assert example.shape[0] >= 12, f"Expected at least 12 leads, got {example.shape[0]}"
    
    data_shape = (12, WINDOW_SIZE)  # (channels, time)
    
    # Create HDF5 file
    h5f = create_h5(out_file, data_shape)
    
    # Data and label buffers
    data_buffer = []
    label_buffer = []
    
    valid_samples = 0
    total_segments = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {os.path.basename(out_file)}'):
        # Parse single label
        single_label = parse_scp_codes_single_label(row['scp_codes'], diag_map, threshold)
        
        # Skip invalid labels
        if single_label == -1:
            continue
        
        # Read ECG signal (500Hz)
        record_path = os.path.join(root, row['filename_hr'])
        try:
            sig = read_ecg_data(record_path)
        except Exception as e:
            print(f"Error reading {record_path}: {e}")
            continue
        
        # Check signal length
        if sig.shape[1] < WINDOW_SIZE:
            continue
        
        # Use only first 12 leads
        sig = sig[:12]
        
        # Sliding window segmentation (includes normalization)
        segments = sliding_window_ecg(sig)
        
        # Add to buffers
        for seg in segments:
            data_buffer.append(seg.astype(np.float32))
            label_buffer.append(single_label)
            total_segments += 1
        
        valid_samples += 1
        
        # Write to HDF5 when buffer is full
        if len(data_buffer) >= CHUNK_SIZE:
            data_batch = np.stack(data_buffer)
            label_batch = np.array(label_buffer, dtype=np.int64)
            
            append_h5(h5f, data_batch, label_batch)
            
            data_buffer.clear()
            label_buffer.clear()
    
    # Write remaining data
    if data_buffer:
        data_batch = np.stack(data_buffer)
        label_batch = np.array(label_buffer, dtype=np.int64)
        append_h5(h5f, data_batch, label_batch)
    
    # Print statistics
    print(f'\n[Statistics] {out_file}')
    print(f'  Valid samples: {valid_samples}/{len(df)} ({valid_samples/len(df)*100:.1f}%)')
    print(f'  Total segments: {total_segments}')
    print(f'  HDF5 keys: {list(h5f.keys())}')
    print(f'  Data shape: {h5f["data"].shape}')
    print(f'  Label shape: {h5f["label"].shape}')
    
    # Print label distribution
    labels = h5f['label'][:]
    print(f'  Label distribution:')
    for class_name, class_idx in SUPERCLASS_MAP.items():
        count = np.sum(labels == class_idx)
        print(f'    {class_name} ({class_idx}): {count} ({count/len(labels)*100:.1f}%)')
    
    h5f.close()

# ------------------------------------------------------
# 8. Main function
# ------------------------------------------------------
def main():
    # PTB-XL dataset root directory
    root = 'path/to/PTBXL'  # Update to your PTB-XL directory
    
    # Load database and diagnostic mapping
    df = pd.read_csv(os.path.join(root, 'ptbxl_database.csv'))
    diag_map = load_diag_map(os.path.join(root, 'scp_statements.csv'))
    
    print("=== PTB-XL Single Label Preprocessing ===")
    print(f"Total records: {len(df)}")
    print(f"Superclasses: {list(SUPERCLASS_MAP.keys())}")
    print(f"Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}")
    print(f"Sampling rate: {FS} Hz")
    print("Normalization: MinMax [-1,1]")
    print("Filtering: None (raw signals)")
    print(f"Data source: records500/ (500Hz)")
    
    # Define data splits
    splits = {
        'train': df[df['strat_fold'] <= 8].reset_index(drop=True),
        'val':   df[df['strat_fold'] == 9].reset_index(drop=True),
        'test':  df[df['strat_fold'] == 10].reset_index(drop=True),
    }
    
    print(f"\nData splits:")
    for name, subdf in splits.items():
        print(f"  {name}: {len(subdf)} records")
    
    # Process each split
    for name, subdf in splits.items():
        out_file = os.path.join(root, f'{name}.h5')
        print(f'\n{"="*50}')
        print(f'Processing {name} split...')
        process_split(subdf, root, out_file, diag_map, threshold=80.0)
        print(f'Saved to: {out_file}')

    print(f'\n{"="*50}')
    print("Preprocessing completed!")
    print("\nGenerated files are compatible with dataset loader:")
    print("- Keys: 'data' (N, C, T), 'label' (N,)")
    print("- Data type: float32, Label type: int64")
    print("- Single label classification with 5 classes")

if __name__ == '__main__':
    main()