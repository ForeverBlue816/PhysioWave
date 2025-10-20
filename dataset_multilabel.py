"""
Multi-Label Classification Data Loader Module
Supports loading and preprocessing of time series data for multi-label classification
Supports loading multiple HDF5 files
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Union, Optional


class MultiLabelTimeSeriesDataset(Dataset):
    """
    Multi-label time series dataset class
    Supports HDF5 format with multi-hot encoded labels
    """
    def __init__(self, 
                 file_paths: Union[str, List[str]],
                 transform=None,
                 max_length=None,
                 data_key='data',
                 label_key='label'):
        """
        Args:
            file_paths: str or List[str] - HDF5 file path or list of paths
            transform: Data transformation function
            max_length: Maximum sequence length, truncate if exceeded
            data_key: Data key name (default: 'data' or 'input')
            label_key: Label key name (default: 'label')
        """
        super().__init__()
        
        # Unified handling of file paths
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = list(file_paths)
        
        # Check if files exist
        for file_path in self.file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} not found.")
        
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.max_length = max_length
        
        # Load all data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load data from all H5 files"""
        all_data = []
        all_labels = []
        
        print(f"Loading {len(self.file_paths)} file(s)...")
        
        for i, file_path in enumerate(self.file_paths):
            print(f"  Loading file {i+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, "r") as h5f:
                # Try different key names for data
                data = None
                for key in [self.data_key, 'input', 'data', 'X']:
                    if key in h5f:
                        data = h5f[key][:]
                        if i == 0:
                            print(f"    Using data key: '{key}'")
                        break
                
                if data is None:
                    raise KeyError(f"Data key not found in {file_path}. "
                                 f"Available keys: {list(h5f.keys())}")
                
                # Load labels
                labels = None
                for key in [self.label_key, 'label', 'labels', 'y', 'Y']:
                    if key in h5f:
                        labels = h5f[key][:]
                        if i == 0:
                            print(f"    Using label key: '{key}'")
                        break
                
                if labels is None:
                    raise KeyError(f"Label key not found in {file_path}. "
                                 f"Available keys: {list(h5f.keys())}")
                
                all_data.append(data)
                all_labels.append(labels)
                
                print(f"    Data shape: {data.shape}")
                print(f"    Labels shape: {labels.shape}")
        
        # Combine all data
        self._data = np.concatenate(all_data, axis=0)
        self._labels = np.concatenate(all_labels, axis=0)
        self._num_samples = len(self._data)
        
        print(f"\nCombined dataset:")
        print(f"  Data shape: {self._data.shape}")
        print(f"  Labels shape: {self._labels.shape}")
        print(f"  Total samples: {self._num_samples}")
        
        # Check if labels are multi-hot encoded
        if len(self._labels.shape) == 1:
            raise ValueError(f"Labels should be 2D (N, num_classes) for multi-label classification, "
                           f"got shape {self._labels.shape}")
        
        self.num_classes = self._labels.shape[1]
        
        # Print label statistics
        label_counts = self._labels.sum(axis=0)
        avg_labels_per_sample = self._labels.sum(axis=1).mean()
        
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Average labels per sample: {avg_labels_per_sample:.2f}")
        print(f"  Label distribution (count per class):")
        for class_idx, count in enumerate(label_counts):
            percentage = count / self._num_samples * 100
            print(f"    Class {class_idx}: {int(count)} ({percentage:.1f}%)")
    
    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        # Get data
        x = self._data[idx].copy()  # => shape (C, T)
        
        # Length processing
        if self.max_length is not None:
            current_length = x.shape[-1]
            
            if current_length > self.max_length:
                # Random crop for training, center crop for validation
                if self.transform and hasattr(self.transform, 'random_crop') and self.transform.random_crop:
                    start = np.random.randint(0, current_length - self.max_length + 1)
                    x = x[:, start:start + self.max_length]
                else:
                    # Center crop
                    start = (current_length - self.max_length) // 2
                    x = x[:, start:start + self.max_length]
            
            elif current_length < self.max_length:
                # Pad to specified length
                pad_length = self.max_length - current_length
                x = np.pad(x, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        
        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self._labels[idx], dtype=torch.float32)  # Multi-hot labels as float
        
        # Apply transformation
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @property
    def data_shape(self):
        """Returns the shape of a single sample"""
        return self._data[0].shape


class SingleLabelTimeSeriesDataset(Dataset):
    """
    Single-label (multi-class) time series dataset class
    For backward compatibility
    """
    def __init__(self, 
                 file_paths: Union[str, List[str]],
                 transform=None,
                 max_length=None,
                 data_key='data',
                 label_key='label'):
        super().__init__()
        
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = list(file_paths)
        
        for file_path in self.file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} not found.")
        
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.max_length = max_length
        
        self._load_all_data()
    
    def _load_all_data(self):
        """Load data from all H5 files"""
        all_data = []
        all_labels = []
        
        print(f"Loading {len(self.file_paths)} file(s)...")
        
        for i, file_path in enumerate(self.file_paths):
            print(f"  Loading file {i+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, "r") as h5f:
                # Try different key names
                data = None
                for key in [self.data_key, 'input', 'data', 'X']:
                    if key in h5f:
                        data = h5f[key][:]
                        break
                
                if data is None:
                    raise KeyError(f"Data key not found in {file_path}")
                
                labels = None
                for key in [self.label_key, 'label', 'labels', 'y', 'Y']:
                    if key in h5f:
                        labels = h5f[key][:]
                        break
                
                if labels is None:
                    raise KeyError(f"Label key not found in {file_path}")
                
                all_data.append(data)
                all_labels.append(labels)
                
                print(f"    Data shape: {data.shape}, Labels shape: {labels.shape}")
        
        self._data = np.concatenate(all_data, axis=0)
        self._labels = np.concatenate(all_labels, axis=0)
        self._num_samples = len(self._data)
        
        print(f"\nCombined dataset: {self._data.shape} data, {self._labels.shape} labels")
        print(f"Total samples: {self._num_samples}")
        
        unique_labels, counts = np.unique(self._labels, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({count/self._num_samples*100:.1f}%)")
        
        self.num_classes = len(unique_labels)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        x = self._data[idx].copy()
        
        if self.max_length is not None:
            current_length = x.shape[-1]
            if current_length > self.max_length:
                if self.transform and hasattr(self.transform, 'random_crop') and self.transform.random_crop:
                    start = np.random.randint(0, current_length - self.max_length + 1)
                    x = x[:, start:start + self.max_length]
                else:
                    start = (current_length - self.max_length) // 2
                    x = x[:, start:start + self.max_length]
            elif current_length < self.max_length:
                pad_length = self.max_length - current_length
                x = np.pad(x, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self._labels[idx], dtype=torch.long)  # Class indices as long
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @property
    def data_shape(self):
        return self._data[0].shape


class DataAugmentation:
    """Time series data augmentation"""
    def __init__(self, 
                 noise_std=0.01,
                 time_stretch_range=(0.8, 1.2),
                 amplitude_scale_range=(0.8, 1.2),
                 random_crop=False,
                 prob=0.5):
        self.noise_std = noise_std
        self.time_stretch_range = time_stretch_range
        self.amplitude_scale_range = amplitude_scale_range
        self.random_crop = random_crop
        self.prob = prob
    
    def __call__(self, x):
        """
        Apply augmentation to input data
        x: torch.Tensor [C, T]
        """
        if torch.rand(1) > self.prob:
            return x
        
        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Amplitude scaling
        if self.amplitude_scale_range != (1.0, 1.0):
            scale = torch.FloatTensor(1).uniform_(*self.amplitude_scale_range).item()
            x = x * scale
        
        return x


def collate_multilabel_fn(batch):
    """Collate function for multi-label classification"""
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)  # (B, C, T)
    ys = torch.stack(ys, dim=0)  # (B, num_classes)
    return xs, ys


def collate_singlelabel_fn(batch):
    """Collate function for single-label classification"""
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)  # (B, C, T)
    ys = torch.tensor(ys, dtype=torch.long)  # (B,)
    return xs, ys


def parse_file_paths(file_paths_str: str) -> List[str]:
    """Parse file path string"""
    if not file_paths_str:
        return []
    if ',' in file_paths_str:
        paths = [path.strip() for path in file_paths_str.split(',')]
    else:
        paths = file_paths_str.strip().split()
    return [path for path in paths if path]


def get_dataset_stats(file_paths: Union[str, List[str]], 
                     data_key='data', 
                     label_key='label',
                     task='multilabel'):
    """Get dataset statistics"""
    if isinstance(file_paths, str):
        file_paths = parse_file_paths(file_paths)
    
    all_data = []
    all_labels = []
    
    print(f"Analyzing {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths):
        print(f"  File {i+1}: {os.path.basename(file_path)}")
        
        with h5py.File(file_path, "r") as h5f:
            print(f"    Available keys: {list(h5f.keys())}")
            
            # Find data key
            data = None
            for key in [data_key, 'input', 'data', 'X']:
                if key in h5f:
                    data = h5f[key][:]
                    print(f"    Using data key: '{key}'")
                    break
            
            if data is not None:
                print(f"    Data shape: {data.shape}")
                print(f"    Data type: {data.dtype}")
                print(f"    Data range: [{np.min(data):.4f}, {np.max(data):.4f}]")
                print(f"    Data mean: {np.mean(data):.4f}")
                print(f"    Data std: {np.std(data):.4f}")
                all_data.append(data)
            
            # Find label key
            labels = None
            for key in [label_key, 'label', 'labels', 'y', 'Y']:
                if key in h5f:
                    labels = h5f[key][:]
                    print(f"    Using label key: '{key}'")
                    break
            
            if labels is not None:
                print(f"    Labels shape: {labels.shape}")
                if task == 'multilabel':
                    if len(labels.shape) == 2:
                        label_counts = labels.sum(axis=0)
                        print(f"    Labels per class: {label_counts}")
                        print(f"    Avg labels per sample: {labels.sum(axis=1).mean():.2f}")
                    else:
                        print(f"    Warning: Expected 2D labels for multi-label, got {labels.shape}")
                else:
                    print(f"    Unique labels: {np.unique(labels)}")
                    print(f"    Label distribution: {np.bincount(labels.flatten().astype(int))}")
                all_labels.append(labels)
    
    if all_data:
        combined_data = np.concatenate(all_data, axis=0)
        print(f"\nCombined statistics:")
        print(f"  Total shape: {combined_data.shape}")
        print(f"  Total range: [{np.min(combined_data):.4f}, {np.max(combined_data):.4f}]")
        print(f"  Total mean: {np.mean(combined_data):.4f}")
        print(f"  Total std: {np.std(combined_data):.4f}")
    
    if all_labels:
        combined_labels = np.concatenate(all_labels, axis=0)
        print(f"  Total labels shape: {combined_labels.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', type=str, required=True)
    parser.add_argument('--task', type=str, default='multilabel', 
                       choices=['multilabel', 'singlelabel'])
    args = parser.parse_args()
    
    print("Dataset statistics:")
    get_dataset_stats(args.data_files, task=args.task)
    
    print(f"\nTesting {args.task} data loader:")
    
    if args.task == 'multilabel':
        dataset = MultiLabelTimeSeriesDataset(args.data_files)
    else:
        dataset = SingleLabelTimeSeriesDataset(args.data_files)
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_multilabel_fn if args.task == 'multilabel' else collate_singlelabel_fn
    )
    
    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: x={x.shape}, y={y.shape}")
        if i >= 2:
            break
    
    print("Data loader test completed!")