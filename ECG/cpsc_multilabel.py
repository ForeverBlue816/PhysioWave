#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPSC 2018 Official 9-Class Multi-label Preprocessing
Based on CPSC 2018 Challenge Official Standards
--------------------------------------------------------------------------------
- Data sources: cpsc_2018, cpsc_2018_extra
- Record-level split: train:val:test = 7:2:1 (record-level to avoid data leakage)
- Sliding window: length 2048, step 1024
- Normalization: Z-score per window
- Only 12-lead records processed
- 9 classes (multi-label) - CPSC 2018 official standard
"""

import os
import re
import json
import random
import numpy as np
import h5py
import scipy.io
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ========================= Parameters =========================
FS = 500
WINDOW_SIZE = 2048
STEP_SIZE = 1024
CHUNK_SIZE = 500
SEED = 42
NUM_LEADS = 12

ROOT_DIRS = [r"CPSC\cpsc_2018", r"CPSC\cpsc_2018_extra"]
OUT_DIR = r"CPSC"

# ========================= CPSC 2018 Official 9-Class Definition =========================
# Based on CPSC 2018 Challenge official scoring classes
CLASS_SPECS = [
    ("SNR",    {"426783006"}),                              # Sinus rhythm (Normal)
    ("AF",     {"164889003"}),                              # Atrial fibrillation
    ("IAVB",   {"270492004"}),                              # First degree AV block
    ("LBBB",   {"733534002", "164909002"}),                 # Left bundle branch block
    ("RBBB",   {"713427006", "59118001"}),                  # Right bundle branch block
    ("PAC",    {"284470004"}),                              # Premature atrial contraction
    ("PVC",    {"427172004", "164884008", "17338001"}),     # Premature ventricular contraction
    ("STD",    {"429622005"}),                              # ST-segment depression
    ("STE",    {"164931005"}),                              # ST-segment elevation
]

# Build mappings
LABELS_9 = [name for name, _ in CLASS_SPECS]
LABEL2IDX = {name: i for i, name in enumerate(LABELS_9)}
CODE2IDX = {}
for i, (name, codes) in enumerate(CLASS_SPECS):
    for c in codes:
        CODE2IDX[str(c)] = i

print("="*80)
print("CPSC 2018 Official 9-Class Multi-label Classification")
print("="*80)
for i, (name, codes) in enumerate(CLASS_SPECS):
    snomed_str = ", ".join(codes)
    print(f"  {i}. {name:<10} - SNOMED: {snomed_str}")
print("="*80)
print("\nSplitting strategy:")
print("  • Record-level split")
print("  • Train : Val : Test = 7 : 2 : 1")
print("  • All windows from the same record stay in the same split")
print("  • Ensures no data leakage and reliable model generalization")
print("="*80)

# ========================= Utility Functions =========================
def read_ecg_mat(mat_path: str) -> np.ndarray:
    """Read .mat ECG file, return (12, L) float32 array"""
    md = scipy.io.loadmat(mat_path)
    sig = None
    for key in ("val", "data", "ecg"):
        if key in md and isinstance(md[key], np.ndarray):
            sig = md[key]
            break
    if sig is None:
        for v in md.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                sig = v
                break
    if sig is None:
        raise ValueError(f"No 2D ECG array in {mat_path}")
    
    sig = np.asarray(sig, dtype=np.float32)
    if sig.shape[0] != NUM_LEADS and sig.shape[1] == NUM_LEADS:
        sig = sig.T
    np.nan_to_num(sig, copy=False)
    return sig

def parse_dx_codes_from_header(hea_path: str):
    """Parse SNOMED codes from '# Dx:' or '#Dx' line in .hea file"""
    codes = []
    encodings = ['utf-8', 'latin-1', 'gbk', 'cp1252']
    
    for enc in encodings:
        try:
            with open(hea_path, 'r', encoding=enc, errors='strict') as f:
                for line in f:
                    line = line.strip()
                    # Support multiple formats
                    if line.startswith('# Dx:') or line.startswith('#Dx:'):
                        dx_part = line.split(':', 1)[1].strip()
                    elif line.startswith('# Dx') or line.startswith('#Dx'):
                        dx_part = line[4:].strip() if line.startswith('# Dx') else line[3:].strip()
                    else:
                        continue
                    
                    # Split and extract numeric codes
                    tokens = re.split(r'[,\s;]+', dx_part)
                    for tok in tokens:
                        tok = tok.strip()
                        if tok and tok.isdigit():
                            codes.append(tok)
                    return codes  # Return immediately after finding
            break  # Break encoding loop after successful read
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        except Exception as e:
            print(f"Warning: Error reading {hea_path}: {e}")
            continue
    
    return codes

def zscore_normalize(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization (per-window independent normalization)"""
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + eps
    return ((windows - mean) / std).astype(np.float32)

def sliding_window_keep_tail(ecg: np.ndarray,
                             window_size: int = WINDOW_SIZE,
                             step_size: int = STEP_SIZE):
    """
    Segment ECG signal using sliding window, keep tail
    Returns: list of (C, window_size) arrays
    """
    C, L = ecg.shape
    segs = []
    
    # Generate sliding window start positions
    starts = list(range(0, max(L - window_size + 1, 0), step_size))
    if not starts:
        starts = [0]
    
    # Add tail window if the last window doesn't cover signal end
    if starts[-1] + window_size < L:
        tail_start = starts[-1] + step_size
        if tail_start < L:
            starts.append(tail_start)
    
    # Segment windows
    for st in starts:
        ed = st + window_size
        if ed <= L:
            seg = ecg[:, st:ed]
        else:
            # Tail insufficient, zero-pad
            seg = np.zeros((C, window_size), dtype=ecg.dtype)
            remain = L - st
            if remain > 0:
                seg[:, :remain] = ecg[:, st:]
        segs.append(seg)
    
    return segs

def create_h5(path: str, data_shape, num_classes: int, chunk_size: int = CHUNK_SIZE):
    """Create HDF5 file with extensible datasets"""
    f = h5py.File(path, 'w')
    f.create_dataset(
        'input',
        shape=(0,) + data_shape,
        maxshape=(None,) + data_shape,
        chunks=(chunk_size,) + data_shape,
        dtype=np.float32,
        compression='gzip',
        compression_opts=4
    )
    f.create_dataset(
        'label',
        shape=(0, num_classes),
        maxshape=(None, num_classes),
        chunks=(chunk_size, num_classes),
        dtype=np.float32,
        compression='gzip',
        compression_opts=4
    )
    return f

def append_h5(f: h5py.File, x_batch: np.ndarray, y_batch: np.ndarray):
    """Append data to HDF5 file"""
    dx, dy = f['input'], f['label']
    n0 = dx.shape[0]
    n1 = n0 + x_batch.shape[0]
    dx.resize((n1,) + dx.shape[1:])
    dy.resize((n1, dy.shape[1]))
    dx[n0:n1] = x_batch
    dy[n0:n1] = y_batch

def collect_all_records(root_dirs):
    """Recursively collect all .hea/.mat record pairs"""
    items = []
    for root in root_dirs:
        if not os.path.isdir(root):
            print(f"⚠️  Directory not found: {root}")
            continue
        
        print(f"\nScanning directory: {root}")
        for dirpath, _, filenames in os.walk(root):
            hea_files = [fn for fn in filenames if fn.lower().endswith('.hea')]
            if not hea_files:
                continue
            
            for hea_name in hea_files:
                rec_id = os.path.splitext(hea_name)[0]
                hea_path = os.path.join(dirpath, hea_name)
                
                # Try multiple .mat file naming conventions
                mat_path = None
                for ext in ['.mat', '.MAT']:
                    candidate = os.path.join(dirpath, rec_id + ext)
                    if os.path.exists(candidate):
                        mat_path = candidate
                        break
                
                if mat_path is None:
                    continue
                
                codes = parse_dx_codes_from_header(hea_path)
                items.append({
                    'rec_id': rec_id,
                    'hea_path': hea_path,
                    'mat_path': mat_path,
                    'codes': codes,
                })
    
    print(f"✓ Found {len(items)} records")
    return items

def codes_to_multihot_9(codes):
    """
    Map SNOMED codes to 9-dim multi-hot label vector
    Returns: (found, label_vector)
        found: bool, whether at least one official 9-class is matched
        label_vector: (9,) float32 array
    """
    y = np.zeros((len(LABELS_9),), dtype=np.float32)
    found = False
    for c in codes:
        idx = CODE2IDX.get(str(c))
        if idx is not None:
            y[idx] = 1.0
            found = True
    return found, y

# ========================= Core Processing =========================
def process_split(items, split_name: str):
    """Process a single dataset split (train/val/test)"""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} set")
    print(f"{'='*80}")
    
    out_path = os.path.join(OUT_DIR, f"cpsc_9class_{split_name}.h5")
    K = len(LABELS_9)
    hf = create_h5(out_path, data_shape=(NUM_LEADS, WINDOW_SIZE), num_classes=K)
    
    buf_x, buf_y = [], []
    valid_records = 0
    no_target_labels = 0
    non12_skipped = 0
    read_errors = 0
    total_windows = 0
    
    # Count samples per class (record-level)
    label_counts = np.zeros(K, dtype=int)
    
    pbar = tqdm(items, desc=f"{split_name} records", ncols=120)
    for it in pbar:
        rec_id, mat_path, codes = it['rec_id'], it['mat_path'], it['codes']
        
        # Read ECG
        try:
            ecg = read_ecg_mat(mat_path)
        except Exception as e:
            read_errors += 1
            pbar.set_postfix_str(f"read_err={read_errors}")
            continue
        
        # Check number of leads
        if ecg.shape[0] != NUM_LEADS:
            non12_skipped += 1
            pbar.set_postfix_str(f"non12={non12_skipped}")
            continue
        
        # Map labels to official 9 classes
        has_any, y = codes_to_multihot_9(codes)
        if not has_any:
            no_target_labels += 1
            pbar.set_postfix_str(f"no_target={no_target_labels}")
            continue  # Skip records without official 9-class labels
        
        # Count labels (record-level)
        label_counts += y.astype(int)
        
        # Sliding window segmentation
        segs = sliding_window_keep_tail(ecg, WINDOW_SIZE, STEP_SIZE)
        xs = np.array(segs, dtype=np.float32)
        xs = zscore_normalize(xs)
        
        # Replicate label for each window
        ys = np.repeat(y[None, :], xs.shape[0], axis=0)
        
        buf_x.extend(xs)
        buf_y.extend(ys)
        total_windows += xs.shape[0]
        valid_records += 1
        
        # Batch write to HDF5
        while len(buf_x) >= CHUNK_SIZE:
            x_b = np.stack(buf_x[:CHUNK_SIZE])
            y_b = np.stack(buf_y[:CHUNK_SIZE])
            append_h5(hf, x_b, y_b)
            buf_x = buf_x[CHUNK_SIZE:]
            buf_y = buf_y[CHUNK_SIZE:]
        
        pbar.set_postfix_str(f"valid={valid_records}, windows={total_windows}")
    
    # Write remaining data
    if buf_x:
        x_b = np.stack(buf_x)
        y_b = np.stack(buf_y)
        append_h5(hf, x_b, y_b)
    
    shape_x, shape_y = hf['input'].shape, hf['label'].shape
    hf.close()
    
    # Print statistics
    print(f"\n✓ Completed {split_name}")
    print(f"  Valid records: {valid_records}/{len(items)}")
    print(f"  Skipped (no official 9-class labels): {no_target_labels}")
    print(f"  Skipped (non-12-lead): {non12_skipped}")
    print(f"  Read errors: {read_errors}")
    print(f"  Total windows: {total_windows}")
    print(f"  HDF5 shapes: input={shape_x}, label={shape_y}")
    
    # Print label distribution (record-level)
    print(f"\n  Label distribution (record-level):")
    avg_labels = label_counts.sum() / valid_records if valid_records > 0 else 0
    print(f"    Avg labels per record: {avg_labels:.2f}")
    
    # Display sorted by frequency
    sorted_indices = np.argsort(-label_counts)  # Descending
    for idx in sorted_indices:
        name = LABELS_9[idx]
        count = label_counts[idx]
        pct = 100.0 * count / valid_records if valid_records > 0 else 0
        if count > 0:  # Only show classes with samples
            print(f"    {idx}. {name:<10}: {count:>5} ({pct:>5.1f}%)")
    
    return {
        "valid_records": valid_records,
        "no_target_labels": no_target_labels,
        "skipped_non12": non12_skipped,
        "read_errors": read_errors,
        "total_windows": total_windows,
        "label_counts": label_counts.tolist(),
        "h5_path": out_path,
        "final_shape_input": list(shape_x),
        "final_shape_label": list(shape_y),
    }

# ========================= Main Process =========================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    print("\n" + "="*80)
    print("CPSC 2018 Official 9-Class Multi-label Preprocessing")
    print("="*80)
    print(f"Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}")
    print(f"Sampling rate: {FS} Hz, Number of leads: {NUM_LEADS}")
    print(f"Number of classes: {len(LABELS_9)} (CPSC 2018 official standard)")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Step 1: Collect records
    print("\n[Step 1] Collecting records...")
    items = collect_all_records(ROOT_DIRS)
    
    if len(items) == 0:
        print("❌ Error: No records found!")
        return
    
    # Save class mappings
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({name: LABEL2IDX[name] for name in LABELS_9}, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUT_DIR, "label_vocab.txt"), "w", encoding="utf-8") as f:
        f.write("Index\tAbbreviation\tSNOMED_Codes\n")
        for i, (name, codes) in enumerate(CLASS_SPECS):
            snomed_str = ",".join(codes)
            f.write(f"{i}\t{name}\t{snomed_str}\n")
    
    with open(os.path.join(OUT_DIR, "snomed_mapping.json"), "w", encoding="utf-8") as f:
        snomed_map = {}
        for name, codes in CLASS_SPECS:
            for code in codes:
                snomed_map[code] = name
        json.dump(snomed_map, f, indent=2, ensure_ascii=False)
    
    # Step 2: Record-level split (avoid data leakage)
    print("\n[Step 2] Record-level data splitting (no data leakage)...")
    print("  Split ratio: Train 70% | Val 20% | Test 10%")
    
    # First split: 70% train, 30% temp
    train_items, temp_items = train_test_split(
        items, 
        test_size=0.3, 
        random_state=SEED, 
        shuffle=True
    )
    
    # Second split: 30% temp -> 20% val + 10% test
    val_items, test_items = train_test_split(
        temp_items, 
        test_size=1/3,  # 1/3 of 30% = 10%
        random_state=SEED, 
        shuffle=True
    )
    
    print(f"  Train: {len(train_items)} records ({len(train_items)/len(items)*100:.1f}%)")
    print(f"  Val:   {len(val_items)} records ({len(val_items)/len(items)*100:.1f}%)")
    print(f"  Test:  {len(test_items)} records ({len(test_items)/len(items)*100:.1f}%)")
    
    # Save split information
    with open(os.path.join(OUT_DIR, "record_splits.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train": [it["rec_id"] for it in train_items],
            "val": [it["rec_id"] for it in val_items],
            "test": [it["rec_id"] for it in test_items],
        }, f, indent=2, ensure_ascii=False)
    
    print("\n  ✓ Record IDs saved to record_splits.json")
    print("  ℹ️  All windows from the same record stay in the same split, ensuring no data leakage")
    
    # Step 3: Process each split
    print("\n[Step 3] Processing data (sliding window + normalization)...")
    stat_train = process_split(train_items, "train")
    stat_val = process_split(val_items, "val")
    stat_test = process_split(test_items, "test")
    
    # Save metadata
    meta = {
        "task": "CPSC 2018 Official 9-class Multi-label Classification",
        "num_classes": len(LABELS_9),
        "class_names": LABELS_9,
        "snomed_codes": {name: list(codes) for name, codes in CLASS_SPECS},
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_channels": NUM_LEADS,
        "sampling_rate": FS,
        "normalization": "z-score (per-window)",
        "split_method": "record-level split (no data leakage)",
        "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
        "sources": ROOT_DIRS,
        "seed": SEED,
        "num_records": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items)
        },
        "num_windows": {
            "train": stat_train["total_windows"],
            "val": stat_val["total_windows"],
            "test": stat_test["total_windows"]
        },
        "label_distribution": {
            "train": {LABELS_9[i]: stat_train["label_counts"][i] for i in range(len(LABELS_9))},
            "val": {LABELS_9[i]: stat_val["label_counts"][i] for i in range(len(LABELS_9))},
            "test": {LABELS_9[i]: stat_test["label_counts"][i] for i in range(len(LABELS_9))}
        },
        "final_shapes": {
            "train": {"input": stat_train["final_shape_input"], "label": stat_train["final_shape_label"]},
            "val": {"input": stat_val["final_shape_input"], "label": stat_val["final_shape_label"]},
            "test": {"input": stat_test["final_shape_input"], "label": stat_test["final_shape_label"]}
        }
    }
    
    with open(os.path.join(OUT_DIR, "cpsc_9class_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✓ Preprocessing completed!")
    print("="*80)
    print(f"\nOutput directory: {OUT_DIR}")
    print("Generated files:")
    print("  • cpsc_9class_train.h5 / val.h5 / test.h5")
    print("  • label_map.json (class name → index)")
    print("  • label_vocab.txt (index → class name → SNOMED codes)")
    print("  • snomed_mapping.json (SNOMED code → class name)")
    print("  • record_splits.json (record-level split information)")
    print("  • cpsc_9class_info.json (complete metadata)")
    
    print(f"\nDataset statistics:")
    print(f"  Train: {stat_train['total_windows']:,} windows (from {len(train_items)} records)")
    print(f"  Val:   {stat_val['total_windows']:,} windows (from {len(val_items)} records)")
    print(f"  Test:  {stat_test['total_windows']:,} windows (from {len(test_items)} records)")
    
    print(f"\nNext steps:")
    print(f"  1. Validate data:")
    print(f"     python validate_9class_data.py")
    print(f"\n  2. Start training (CPSC 2018 official 9-class):")
    print(f"     torchrun --nproc_per_node=4 finetune_multilabel.py \\")
    print(f"         --train_file {os.path.join(OUT_DIR, 'cpsc_9class_train.h5')} \\")
    print(f"         --val_file {os.path.join(OUT_DIR, 'cpsc_9class_val.h5')} \\")
    print(f"         --test_file {os.path.join(OUT_DIR, 'cpsc_9class_test.h5')} \\")
    print(f"         --in_channels 12 --task_type multilabel \\")
    print(f"         --batch_size 16 --epochs 50 --lr 1e-4 --threshold 0.3")

if __name__ == "__main__":
    main()