import os
import glob
import numpy as np
import h5py
from scipy.io import loadmat
from tqdm import tqdm
import json

# ================== Global Parameters ===================
SAMPLING_RATE = 500
WINDOW_SIZE   = 2048
STEP_SIZE     = 1024

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1

ROOT_DIR   = r"a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords"
OUTPUT_DIR = r"a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"

# ================== 4-Class Mapping (Option 1: ST merged into SR) ===================
MERGE_MAP = {
    'SB':   ['SB'],                                     # Sinus Bradycardia
    'AFIB': ['AFIB', 'AF', 'AFL'],                    # Atrial Fibrillation/Flutter
    'GSVT': ['SVT', 'AT', 'AVNRT', 'AVRT', 'SAAWR'],  # Supraventricular Tachycardia (excluding ST)
    'SR':   ['SR', 'SI', 'ST']                         # Sinus Rhythm (including Tachycardia)
}
LABEL_TO_ID = {cls: idx for idx, cls in enumerate(MERGE_MAP)}
NUM_CLASSES = len(LABEL_TO_ID)

# SNOMED CT Code Mapping
CODE_TO_CLASS = {
    '426177001': 'SB',      # Sinus Bradycardia
    '164889003': 'AFIB',    # Atrial Fibrillation
    '164890007': 'AFIB',    # Atrial Flutter
    '426761007': 'GSVT',    # Supraventricular tachycardia
    '713422000': 'GSVT',    # Paroxysmal SVT
    '233896004': 'GSVT',    # Other GSVT
    '233897008': 'GSVT',    # Other GSVT
    '195101003': 'GSVT',    # Other GSVT
    '427084000': 'SR',      # Sinus Tachycardia → merged into SR (corrected)
    '426783006': 'SR',      # Sinus Rhythm
    '427393009': 'SR'       # Sinus Arrhythmia
}

# ================== Parse Multi-hot Labels ===================
def parse_multihot_label(hea_path):
    """Parse multi-hot labels from .hea file"""
    labels = np.zeros(NUM_CLASSES, dtype=np.int8)
    if not os.path.exists(hea_path):
        return labels
    
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#Dx:'):
                    # Extract diagnosis codes
                    items = [x.strip() for x in line.split(':', 1)[1].split(',')]
                    
                    # Match SNOMED codes first
                    for it in items:
                        if it.isdigit() and it in CODE_TO_CLASS:
                            cls = CODE_TO_CLASS[it]
                            labels[LABEL_TO_ID[cls]] = 1
                    
                    # Match abbreviations
                    for it in items:
                        for cls, acrs in MERGE_MAP.items():
                            if it in acrs:
                                labels[LABEL_TO_ID[cls]] = 1
                    break
    except Exception as e:
        pass  # Ignore read errors
    
    return labels

# ================== Normalization ===================
def normalize_ecg(sig):
    """
    Z-score normalization per channel (lead)
    sig: (channels, time_steps)
    Returns: normalized signal
    """
    m = np.mean(sig, axis=1, keepdims=True)
    s = np.std(sig, axis=1, keepdims=True) + 1e-8
    return (sig - m) / s

# ================== Sliding Window Segmentation ===================
def slide_and_cut(sig, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Segment signal using sliding window
    sig: (channels, time_steps)
    Returns: (num_segments, channels, window_size)
    """
    ch, L = sig.shape
    segs = []
    start = 0
    
    # Sliding window segmentation
    while start + window_size <= L:
        segs.append(sig[:, start:start+window_size])
        start += step_size
    
    # Handle remaining part (zero-padding)
    if start < L:
        pad = np.zeros((ch, window_size), dtype=sig.dtype)
        rem = sig[:, start:]
        pad[:, :rem.shape[1]] = rem
        segs.append(pad)
    
    if len(segs) == 0:
        return None
    
    return np.stack(segs, axis=0)

# ================== Process Single Record ===================
def process_one_record(mat_path, label):
    """
    Process a single record (read .mat, segment, normalize)
    Returns: (segments, labels) or (None, None)
    """
    try:
        mat = loadmat(mat_path)
        
        # Try multiple possible signal field names
        sig = None
        for key in ['val', 'ECG', 'signal', 'p_signals', 'data']:
            if key in mat:
                sig = mat[key]
                break
        
        if sig is None:
            return None, None
        
        # Ensure 2D array
        if sig.ndim != 2:
            return None, None
        
        # Transpose to (12, N) if needed
        if sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T
        
        # Check for 12-lead
        if sig.shape[0] != 12:
            return None, None
        
        # Check signal length
        if sig.shape[1] < WINDOW_SIZE:
            return None, None
        
        # Convert to float32
        sig = sig.astype(np.float32)
        
        # Normalize + segment
        sig = normalize_ecg(sig)
        segs = slide_and_cut(sig)
        
        if segs is None:
            return None, None
        
        # Replicate label for each window
        labels = np.tile(label, (segs.shape[0], 1))
        
        return segs, labels
    
    except Exception as e:
        return None, None

# ================== HDF5 Helper Functions ===================
def create_h5(path):
    """Create HDF5 file"""
    f = h5py.File(path, 'w')
    f.create_dataset('input',
                     shape=(0, 12, WINDOW_SIZE),
                     maxshape=(None, 12, WINDOW_SIZE),
                     chunks=(100, 12, WINDOW_SIZE),
                     compression='gzip',
                     compression_opts=4,
                     dtype='f4')
    f.create_dataset('label',
                     shape=(0, NUM_CLASSES),
                     maxshape=(None, NUM_CLASSES),
                     chunks=(100, NUM_CLASSES),
                     compression='gzip',
                     compression_opts=4,
                     dtype='i1')
    return f

def append_h5(f, data, labels):
    """Append data to HDF5"""
    x_ds, y_ds = f['input'], f['label']
    old = x_ds.shape[0]
    new = old + data.shape[0]
    x_ds.resize((new, 12, WINDOW_SIZE))
    y_ds.resize((new, NUM_CLASSES))
    x_ds[old:new] = data
    y_ds[old:new] = labels

# ================== Main Program ===================
if __name__ == '__main__':
    print("="*80)
    print("Chapman-Shaoxing ECG 4-Class Multi-label Preprocessing")
    print("Record-Level Split - No Data Leakage")
    print("="*80)
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Step size: {STEP_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class definitions:")
    for i, (cls, abbrs) in enumerate(MERGE_MAP.items()):
        print(f"  {i}. {cls:5s}: {', '.join(abbrs)}")
    print("\nSplitting strategy:")
    print("  • Record-level split")
    print("  • Train : Val : Test = 7 : 2 : 1")
    print("  • All windows from the same record stay in the same split")
    print("  • Ensures no data leakage and reliable model generalization")
    print("="*80)
    
    # 1) Scan all .mat files
    print("\n[Step 1] Scanning .mat files...")
    mats = glob.glob(os.path.join(ROOT_DIR, '**', '*.mat'), recursive=True)
    print(f"✓ Found {len(mats)} .mat files")
    
    if len(mats) == 0:
        print("❌ Error: No .mat files found!")
        print(f"Please check path: {ROOT_DIR}")
        exit(1)

    # 2) Filter records with valid labels (record-level)
    print("\n[Step 2] Parsing labels and filtering records...")
    records = []  # Store record info: (record_id, mat_path, label)
    
    for m in tqdm(mats, desc="Parsing labels", ncols=100):
        hea_path = m.replace('.mat', '.hea')
        ml = parse_multihot_label(hea_path)
        
        if ml.sum() > 0:
            # Extract record ID (to identify the same record)
            record_id = os.path.splitext(os.path.basename(m))[0]
            records.append({
                'record_id': record_id,
                'mat_path': m,
                'label': ml
            })
    
    print(f"✓ Filtered {len(records)} records with 4-class labels")
    
    if len(records) == 0:
        print("❌ Error: No valid data found!")
        exit(1)

    # 3) Record-level split: train/val/test
    print("\n[Step 3] Record-level data splitting...")
    np.random.seed(42)
    np.random.shuffle(records)
    
    N = len(records)
    t = int(TRAIN_RATIO * N)
    v = int(VAL_RATIO * N)
    
    record_splits = {
        'train': records[:t],
        'val':   records[t:t+v],
        'test':  records[t+v:]
    }
    
    print(f"  Train: {len(record_splits['train'])} records ({len(record_splits['train'])/N*100:.1f}%)")
    print(f"  Val:   {len(record_splits['val'])} records ({len(record_splits['val'])/N*100:.1f}%)")
    print(f"  Test:  {len(record_splits['test'])} records ({len(record_splits['test'])/N*100:.1f}%)")

    # Save record split information
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_info = {
        'train': [r['record_id'] for r in record_splits['train']],
        'val':   [r['record_id'] for r in record_splits['val']],
        'test':  [r['record_id'] for r in record_splits['test']]
    }
    
    split_path = os.path.join(OUTPUT_DIR, 'record_splits.json')
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Record split info saved: {split_path}")
    print("  ℹ️  All windows from the same record stay in the same split")

    # 4) Process each split and write to HDF5
    print("\n[Step 4] Processing data and writing to HDF5...")
    
    # Create HDF5 files
    h5_files = {}
    for s in record_splits:
        h5_path = os.path.join(OUTPUT_DIR, f"{s}.h5")
        h5_files[s] = create_h5(h5_path)
        print(f"  Created: {h5_path}")

    # Process records for each split
    split_stats = {}
    
    for split, record_list in record_splits.items():
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*80}")
        
        processed_records = 0
        error_records = 0
        total_windows = 0
        
        # Label distribution statistics (record-level)
        label_counts_records = np.zeros(NUM_CLASSES, dtype=int)
        
        # Progress bar
        pbar = tqdm(record_list, desc=f"Processing {split}", ncols=100, unit="record")
        
        for rec in pbar:
            mat_path = rec['mat_path']
            label = rec['label']
            
            # Process single record (includes all windows)
            segs, labels = process_one_record(mat_path, label)
            
            if segs is not None and labels is not None:
                # Append to HDF5
                append_h5(h5_files[split], segs, labels)
                
                processed_records += 1
                total_windows += segs.shape[0]
                label_counts_records += label.astype(int)
            else:
                error_records += 1
            
            # Update progress bar
            pbar.set_postfix({
                'success': processed_records,
                'failed': error_records,
                'windows': total_windows
            })
        
        pbar.close()
        
        print(f"✓ Completed: {processed_records}/{len(record_list)} records processed")
        print(f"  Failed: {error_records} records")
        print(f"  Total windows: {total_windows}")
        
        # Save statistics
        split_stats[split] = {
            'num_records': len(record_list),
            'processed_records': processed_records,
            'error_records': error_records,
            'total_windows': total_windows,
            'label_counts_records': label_counts_records.tolist()
        }

    # 5) Output detailed statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    for split, f in h5_files.items():
        inp, lab = f['input'], f['label']
        
        print(f"\n【{split.upper()}】")
        print(f"  HDF5 shapes:")
        print(f"    input: {inp.shape}")
        print(f"    label: {lab.shape}")
        
        # Sample data statistics
        sample_size = min(1000, inp.shape[0])
        if sample_size > 0:
            sample_data = inp[:sample_size]
            print(f"\n  Data statistics (based on {sample_size} samples):")
            print(f"    Range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
            print(f"    Mean: {sample_data.mean():.3f}")
            print(f"    Std: {sample_data.std():.3f}")
        
        # Label statistics (window-level)
        labels = lab[:]
        label_counts_windows = labels.sum(axis=0)
        avg_labels = labels.sum(axis=1).mean()
        
        print(f"\n  Label statistics (window-level):")
        print(f"    Total windows: {len(labels):,}")
        print(f"    Avg labels per window: {avg_labels:.2f}")
        
        print(f"\n  Label statistics (record-level):")
        print(f"    Total records: {split_stats[split]['processed_records']:,}")
        
        print(f"\n  Class distribution:")
        class_names = list(MERGE_MAP.keys())
        
        # Record-level statistics
        label_counts_records = np.array(split_stats[split]['label_counts_records'])
        num_records = split_stats[split]['processed_records']
        
        print(f"    Record-level (each record counted once):")
        for i, cls in enumerate(class_names):
            count_rec = int(label_counts_records[i])
            pct_rec = count_rec / num_records * 100 if num_records > 0 else 0
            print(f"      {i}. {cls:5s}: {count_rec:7,} ({pct_rec:5.1f}%)")
        
        print(f"\n    Window-level (each window counted once):")
        for i, cls in enumerate(class_names):
            count_win = int(label_counts_windows[i])
            pct_win = count_win / len(labels) * 100 if len(labels) > 0 else 0
            print(f"      {i}. {cls:5s}: {count_win:7,} ({pct_win:5.1f}%)")
        
        # Check anomalies
        no_label = (labels.sum(axis=1) == 0).sum()
        if no_label > 0:
            print(f"\n    ⚠️  Warning: {no_label} windows have no labels")
        
        multi_label = (labels.sum(axis=1) > 1).sum()
        if multi_label > 0:
            print(f"    ℹ️  Multi-label windows: {multi_label} ({multi_label/len(labels)*100:.1f}%)")
        
        f.close()

    # 6) Save complete metadata
    meta = {
        'task': 'Chapman-Shaoxing 4-class Multi-label Classification',
        'num_classes': NUM_CLASSES,
        'class_names': list(MERGE_MAP.keys()),
        'class_mapping': MERGE_MAP,
        'snomed_mapping': CODE_TO_CLASS,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'sampling_rate': SAMPLING_RATE,
        'num_channels': 12,
        'normalization': 'z-score (per-channel)',
        'split_method': 'record-level split (no data leakage)',
        'split_ratio': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
        'seed': 42,
        'statistics': {
            'train': split_stats['train'],
            'val': split_stats['val'],
            'test': split_stats['test']
        }
    }
    
    meta_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Metadata saved: {meta_path}")

    # 7) Output final file information
    print("\n" + "="*80)
    print("✓ Preprocessing completed!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for split in record_splits:
        output_path = os.path.join(OUTPUT_DIR, f"{split}.h5")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**3)
            print(f"  • {split}.h5")
            print(f"    Path: {output_path}")
            print(f"    Size: {file_size:.2f} GB")
    print(f"  • dataset_info.json")
    print(f"  • record_splits.json")
    
    print(f"\nSplit description:")
    print(f"  ✓ Record-level split adopted")
    print(f"  ✓ All windows from the same record stay in the same split")
    print(f"  ✓ No record overlap between train/val/test sets")
    print(f"  ✓ No data leakage, reliable model generalization")
    
    print(f"\nNext steps:")
    print(f"  1. Validate data:")
    print(f"     python validate_data.py --data_file {os.path.join(OUTPUT_DIR, 'train.h5')}")
    print(f"\n  2. Start training:")
    print(f"     torchrun --nproc_per_node=4 finetune_multilabel.py \\")
    print(f"         --train_file {os.path.join(OUTPUT_DIR, 'train.h5')} \\")
    print(f"         --val_file {os.path.join(OUTPUT_DIR, 'val.h5')} \\")
    print(f"         --test_file {os.path.join(OUTPUT_DIR, 'test.h5')} \\")
    print(f"         --task_type multilabel --in_channels 12 \\")
    print(f"         --batch_size 16 --epochs 50 --lr 1e-4")