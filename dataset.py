"""
Data Loader Module
Supports loading and preprocessing of various time series data formats
Supports loading multiple HDF5 files
No data normalization is performed
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Union


class TimeSeriesDataset(Dataset):
    """
    General time series dataset class
    Supports HDF5 format time series data, can load multiple files
    No normalization is performed
    """
    def __init__(self, 
                 file_paths,  # Supports single file path or list of file paths
                 transform=None,
                 max_length=None,
                 labels_key='label',
                 data_key='data'):
        """
        Args:
            file_paths: str or List[str] - HDF5 file path or list of paths
            transform: Data transformation function
            max_length: Maximum sequence length, truncate if exceeded
            labels_key: Label key name
            data_key: Data key name
        """
        super().__init__()
        
        # Unified handling of file paths
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
        
        # Check if files exist
        for file_path in self.file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} not found.")
        
        # Load all data
        self._load_all_data(data_key, labels_key)
        
        self._num_samples = len(self._data)
        self.transform = transform
        self.max_length = max_length
    
    def _load_all_data(self, data_key, labels_key):
        """Load data from all H5 files"""
        all_data = []
        all_labels = []
        self.has_labels = True
        
        print(f"Loading data from {len(self.file_paths)} files...")
        
        for i, file_path in enumerate(self.file_paths):
            print(f"  Loading file {i+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, "r") as h5f:
                # Load data
                data = h5f[data_key][:]  # => (N, C, T)
                all_data.append(data)
                
                # Load labels (if exists) - Fix: check labels_key is not None first
                if labels_key is not None and labels_key in h5f:
                    labels = h5f[labels_key][:]  # => (N,)
                    all_labels.append(labels)
                else:
                    if i == 0:  # First file determines whether labels exist
                        self.has_labels = False
                    if self.has_labels:
                        print(f"Warning: {file_path} missing labels while previous files had labels")
        
        # Combine all data
        self._data = np.concatenate(all_data, axis=0)
        print(f"  Combined data shape: {self._data.shape}")
        
        # Combine all labels
        if self.has_labels and all_labels:
            self._labels = np.concatenate(all_labels, axis=0)
            print(f"  Combined labels shape: {self._labels.shape}")
            print(f"  Unique labels: {np.unique(self._labels)}")
        else:
            self._labels = None
            self.has_labels = False
    
    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        # Get data (without normalization)
        x = self._data[idx].copy()  # => shape (C, T)
        
        # Length processing
        if self.max_length is not None and x.shape[-1] > self.max_length:
            # Random crop or center crop
            if self.transform and hasattr(self.transform, 'random_crop') and self.transform.random_crop:
                start = np.random.randint(0, x.shape[-1] - self.max_length + 1)
                x = x[:, start:start + self.max_length]
            else:
                # Center crop
                start = (x.shape[-1] - self.max_length) // 2
                x = x[:, start:start + self.max_length]
        elif self.max_length is not None and x.shape[-1] < self.max_length:
            # Pad to specified length
            pad_length = self.max_length - x.shape[-1]
            x = np.pad(x, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        
        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        
        # Apply transformation
        if self.transform:
            x = self.transform(x)
        
        if self.has_labels:
            y = torch.tensor(self._labels[idx], dtype=torch.long)
            return x, y
        else:
            return x
    
    @property
    def data_shape(self):
        """Returns the shape of a single sample"""
        return self._data[0].shape
    
    @property
    def num_classes(self):
        """Returns the number of classes"""
        if self.has_labels:
            return len(np.unique(self._labels))
        return 0


class PretrainTimeSeriesDataset(TimeSeriesDataset):
    """
    Dataset specifically for pretraining, no labels required
    """
    def __init__(self, 
                 file_paths,  # Supports multiple files
                 transform=None,
                 max_length=None,
                 data_key='data'):
        # Pretraining doesn't need labels
        super().__init__(
            file_paths=file_paths,
            transform=transform,
            max_length=max_length,
            labels_key=None,
            data_key=data_key
        )
        self.has_labels = False
    
    def __getitem__(self, idx):
        # Only returns data, no labels
        x = super().__getitem__(idx)
        if isinstance(x, tuple):
            return x[0]
        return x


class DataAugmentation:
    """
    Time series data augmentation class
    """
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
        
        # Time stretching (simple interpolation implementation)
        if self.time_stretch_range != (1.0, 1.0):
            stretch_factor = torch.FloatTensor(1).uniform_(*self.time_stretch_range).item()
            if stretch_factor != 1.0:
                C, T = x.shape
                new_T = int(T * stretch_factor)
                # Use interpolation for time stretching
                x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, C, T]
                x = F.interpolate(x, size=(C, new_T), mode='bilinear', align_corners=False)
                x = x.squeeze(0).squeeze(0)  # [C, new_T]
                
                # If length changes after stretching, crop or pad back to original length
                if new_T > T:
                    # Crop
                    start = (new_T - T) // 2
                    x = x[:, start:start + T]
                elif new_T < T:
                    # Pad
                    pad_length = T - new_T
                    x = F.pad(x, (0, pad_length), mode='constant', value=0)
        
        return x


def parse_file_paths(file_paths_str: str) -> List[str]:
    """
    Parse file path string, supports multiple formats:
    1. Single file: 'train.h5'
    2. Multiple files (comma-separated): 'train1.h5,train2.h5,train3.h5'
    3. Multiple files (space-separated): 'train1.h5 train2.h5 train3.h5'
    """
    if ',' in file_paths_str:
        # Comma-separated
        paths = [path.strip() for path in file_paths_str.split(',')]
    else:
        # Space-separated or single file
        paths = file_paths_str.strip().split()
    
    return [path for path in paths if path]  # Filter empty strings


def collate_pretrain_fn(batch):
    """
    Collate function for pretraining data
    """
    # batch is a list, each element is a tensor [C, T]
    xs = torch.stack(batch, dim=0)  # => (B, C, T)
    return xs


def collate_classify_fn(batch):
    """
    Collate function for classification data
    """
    xs, ys = zip(*batch)
    xs_tensor = torch.stack(xs, dim=0)  # => (B, C, T)
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return xs_tensor, ys_tensor


def create_dataloaders(train_files,
                      val_files=None,
                      test_files=None,
                      batch_size=32,
                      num_workers=4,
                      max_length=None,
                      use_augmentation=False,
                      task='classify',
                      distributed=False,
                      **kwargs):
    """
    Create data loaders, supports multiple H5 files
    
    Args:
        train_files: str or List[str] - Training data file path(s)
        val_files: str or List[str] - Validation data file path(s)
        test_files: str or List[str] - Test data file path(s)
        batch_size: Batch size
        num_workers: Number of worker processes
        max_length: Maximum sequence length
        use_augmentation: Whether to use data augmentation
        task: 'pretrain' or 'classify'
        distributed: Whether to use distributed training
        **kwargs: Other parameters
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Parse file paths
    if isinstance(train_files, str):
        train_files = parse_file_paths(train_files)
    
    if val_files:
        if isinstance(val_files, str):
            val_files = parse_file_paths(val_files)
    
    if test_files:
        if isinstance(test_files, str):
            test_files = parse_file_paths(test_files)
    
    # Data augmentation
    transform = None
    if use_augmentation and task == 'classify':
        transform = DataAugmentation(
            noise_std=kwargs.get('noise_std', 0.01),
            time_stretch_range=kwargs.get('time_stretch_range', (0.9, 1.1)),
            amplitude_scale_range=kwargs.get('amplitude_scale_range', (0.9, 1.1)),
            random_crop=True,
            prob=0.5
        )
    
    # Select dataset class
    if task == 'pretrain':
        dataset_class = PretrainTimeSeriesDataset
        collate_fn = collate_pretrain_fn
    else:
        dataset_class = TimeSeriesDataset
        collate_fn = collate_classify_fn
    
    # Create datasets (without normalization)
    train_dataset = dataset_class(
        file_paths=train_files,
        transform=transform,
        max_length=max_length
    )
    
    val_dataset = None
    if val_files:
        val_dataset = dataset_class(
            file_paths=val_files,
            transform=None,  # No augmentation for validation
            max_length=max_length
        )
    
    test_dataset = None
    if test_files:
        test_dataset = dataset_class(
            file_paths=test_files,
            transform=None,  # No augmentation for testing
            max_length=max_length
        )
    
    # Create samplers (distributed training)
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if val_dataset:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        if test_dataset:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader


# Utility functions
def get_dataset_stats(file_paths, data_key='data'):
    """Get dataset statistics, supports multiple files"""
    if isinstance(file_paths, str):
        file_paths = parse_file_paths(file_paths)
    
    all_data = []
    all_labels = []
    
    print(f"Analyzing {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths):
        print(f"  File {i+1}: {os.path.basename(file_path)}")
        
        with h5py.File(file_path, "r") as h5f:
            data = h5f[data_key]
            print(f"    Data shape: {data.shape}")
            print(f"    Data type: {data.dtype}")
            print(f"    Data range: [{np.min(data):.4f}, {np.max(data):.4f}]")
            print(f"    Data mean: {np.mean(data):.4f}")
            print(f"    Data std: {np.std(data):.4f}")
            
            all_data.append(data[:])
            
            if 'label' in h5f:
                labels = h5f['label'][:]
                all_labels.append(labels)
                print(f"    Labels shape: {labels.shape}")
                print(f"    Unique labels: {np.unique(labels)}")
                print(f"    Label distribution: {np.bincount(labels)}")
    
    # Combined statistics
    combined_data = np.concatenate(all_data, axis=0)
    print(f"\nCombined statistics:")
    print(f"  Total shape: {combined_data.shape}")
    print(f"  Total range: [{np.min(combined_data):.4f}, {np.max(combined_data):.4f}]")
    print(f"  Total mean: {np.mean(combined_data):.4f}")
    print(f"  Total std: {np.std(combined_data):.4f}")
    
    if all_labels:
        combined_labels = np.concatenate(all_labels, axis=0)
        print(f"  Total labels shape: {combined_labels.shape}")
        print(f"  Total unique labels: {np.unique(combined_labels)}")
        print(f"  Total label distribution: {np.bincount(combined_labels)}")


if __name__ == "__main__":
    # Test data loader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', type=str, required=True, 
                       help='Data file path(s), supports multiple files (comma or space separated)')
    parser.add_argument('--task', type=str, default='classify', choices=['pretrain', 'classify'])
    args = parser.parse_args()
    
    print("Dataset statistics:")
    get_dataset_stats(args.data_files)
    
    print(f"\nTesting {args.task} data loader:")
    train_loader, _, _ = create_dataloaders(
        train_files=args.data_files,
        batch_size=4,
        num_workers=0,
        task=args.task,
        max_length=1000
    )
    
    for i, batch in enumerate(train_loader):
        if args.task == 'pretrain':
            print(f"Batch {i}: {batch.shape}")
        else:
            x, y = batch
            print(f"Batch {i}: x={x.shape}, y={y.shape}")
        
        if i >= 2:  # Only test first few batches
            break
    
    print("Data loader test completed!")