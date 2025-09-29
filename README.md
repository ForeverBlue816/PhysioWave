# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10351-b31b1b.svg)](https://arxiv.org/abs/2506.10351)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **PhysioWave**, accepted at NeurIPS 2025. PhysioWave is a novel wavelet-based architecture for physiological signal processing that leverages adaptive multi-scale decomposition and frequency-guided masking to advance self-supervised learning.



## 🌟 Key Features

- **Learnable Wavelet Decomposition**: Adaptive multi-resolution analysis with soft gating mechanism
- **Frequency-Guided Masking**: Novel masking strategy that prioritizes high-energy frequency components
- **Cross-Scale Feature Fusion**: Attention-based fusion across different wavelet decomposition levels
- **Multi-Modal Support**: Unified framework for ECG and EMG signals
- **Large-Scale Pretraining**: Models trained on 182GB of ECG and 823GB of EMG data

## 🏗️ Model Architecture

![PhysioWave Architecture](fig/model.png)

**Model Overview**: The PhysioWave pretraining pipeline begins by initializing a set of standard wavelet functions (e.g., 'db6', 'sym4'), from which learnable low-pass and high-pass filters are generated. These filters are then used for wavelet decomposition to obtain multi-scale frequency-band representations. The decomposed features are processed into spatio-temporal patches, with importance scores computed using FFT-based spectral energy. High-scoring patches are masked and passed through Transformer layers, followed by a lightweight decoder for patch reconstruction.

Key components:
1. **Learnable Wavelet Decomposition** - Adaptively selects optimal wavelet bases for input signals
2. **Multi-Scale Feature Reconstruction** - Hierarchical decomposition with soft gating between scales
3. **Frequency-Guided Masking** - Identifies and masks high-energy patches for self-supervised learning
4. **Transformer Encoder/Decoder** - Processes masked patches with rotary position embeddings

## 📊 Performance

### State-of-the-Art Results (Example)
- **PTB-XL (ECG Arrhythmia)**: 73.1% Accuracy
- **EPN-612 (EMG Gesture)**: 94.5% Accuracy
  
We will release the code for multi-label and multi-modal tasks soon.

## 💾 Pretrained Models

Download our pretrained models from Google Drive:
[**Download Pretrained Models**](https://drive.google.com/drive/folders/1CobMgFT1WIOAHfz1j7Yij3BL6kkjm59k?dmr=1&ec=wgc-drive-globalnav-goto)

Available models:
- `ecg.pth` - ECG model (14M params)
- `emg.pth` - EMG model (5M params)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ForeverBlue816/PhysioWave.git
cd PhysioWave

# Create conda environment
conda create -n physiowave python=3.11
conda activate physiowave

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### Data Preprocessing

#### Dataset Download Links

Download the datasets from the following sources:

**ECG Datasets**:
- **PTB-XL**: [PhysioNet PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/)
- **MIMIC-IV-ECG**: [PhysioNet MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **Other ECG datasets**: [PhysioNet Challenge 2021](https://physionet.org/content/challenge-2021/1.0.3/)

**EMG Datasets**:
- **EPN-612**: [Zenodo EPN-612 Dataset](https://zenodo.org/records/4421500)
- **NinaPro**: [NinaPro Database](https://ninapro.hevs.ch/instructions/DB6.html)

#### Data Format Requirements

All preprocessed data should be saved in HDF5 format with the following structure:
- **Keys**: 
  - `data`: Signal data array (required)
  - `label`: Classification labels (required for supervised tasks)
- **Data Shape**: `(N, C, T)`
  - `N`: Number of samples
  - `C`: Number of channels
  - `T`: Time points (sequence length)
- **Data Type**: `float32` for signals, `int64` for labels

#### ECG Data (PTB-XL Example)

```bash
# Download PTB-XL dataset
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# Preprocess PTB-XL dataset for fine-tuning
python ECG/ptbxl_finetune.py
```

**Input Requirements**:
- Raw PTB-XL dataset with 12-lead ECG recordings
- Sampling rate: 500 Hz
- Signal length: 10 seconds (5000 samples)

**Output Format**:
- Files: `train.h5`, `val.h5`, `test.h5`
- Shape: `(N, 12, 2048)` - 12 ECG leads, 2048 time points per window
- Preprocessing: MinMax normalization to [-1, 1]
- Sliding window: 2048 samples window, 1024 stride
- Labels: 5 superclasses (NORM, MI, STTC, CD, HYP)

```bash
# Preprocess MIMIC-IV for pretraining
python ECG/mimic_pretrain.py
```

**Input Requirements**:
- MIMIC-IV-ECG dataset (.dat/.hea files)
- Sampling rate: 500 Hz
- Variable length recordings

**Output Format**:
- Files: `mimic_pretrain_train.h5`, `mimic_pretrain_val.h5`
- Shape: `(N, 12, 2048)` - 12 ECG leads, 2048 time points
- Preprocessing: Z-score normalization (mean=0, std=1)
- No labels (unsupervised pretraining)

#### EMG Data (EPN-612 Example)

```bash
# Download EPN-612 dataset from Zenodo
# https://zenodo.org/records/4421500

# Preprocess EPN-612 dataset
python EMG/epn_finetune.py
```

**Input Requirements**:
- EPN-612 JSON files with EMG recordings
- Sampling rate: 200 Hz
- 8 EMG channels from Myo armband

**Output Format**:
- Files: `epn612_train_set.h5`, `epn612_val_set.h5`, `epn612_test_set.h5`
- Shape: `(N, 8, 1024)` - 8 EMG channels, 1024 time points
- Preprocessing: Max absolute value normalization
- Labels: 6 gesture classes (0=noGesture, 1=waveIn, 2=waveOut, 3=pinch, 4=open, 5=fist)

```bash
# Download NinaPro DB6 from official website
# https://ninapro.hevs.ch/instructions/DB6.html

# Preprocess NinaPro DB6 for pretraining
python EMG/db6_pretrain.py
```

**Input Requirements**:
- NinaPro DB6 .mat files
- Sampling rate: 2000 Hz
- 14 EMG channels (2 bad channels removed, 8 selected)

**Output Format**:
- Files: `train.h5`, `val.h5`
- Shape: `(N, 8, 1024)` - 8 EMG channels, 1024 time points
- Preprocessing: Z-score normalization
- Sliding window: 1024 samples window, 512 stride
- Train/val split: 80/20

#### Custom Dataset Preparation

To prepare your own dataset, ensure it follows this format:

```python
import h5py
import numpy as np

# Create HDF5 file
with h5py.File('your_dataset.h5', 'w') as f:
    # Signal data: (num_samples, num_channels, sequence_length)
    data = np.random.randn(1000, 8, 1024).astype(np.float32)
    f.create_dataset('data', data=data)
    
    # Labels: (num_samples,)
    labels = np.random.randint(0, 6, size=(1000,)).astype(np.int64)
    f.create_dataset('label', data=labels)
```

**Key Parameters by Signal Type**:

| Signal | Channels | Typical Length | Sampling Rate | Normalization |
|--------|----------|----------------|---------------|---------------|
| ECG    | 12       | 2048           | 500 Hz        | MinMax [-1,1] or Z-score |
| EMG    | 8        | 1024           | 200-2000 Hz   | Max-abs or Z-score |
```

### Pretraining

#### ECG Pretraining
```bash
bash ECG/pretrain_ecg.sh
```

Or run directly:
```bash
torchrun --nproc_per_node=4 pretrain.py \
  --train_files path/to/ecg_train.h5 \
  --val_files path/to/ecg_val.h5 \
  --in_channels 12 \
  --max_level 3 \
  --embed_dim 384 \
  --depth 8 \
  --epochs 50 \
  --mask_ratio 0.7 \
  --output_dir ./pretrain_ecg
```

#### EMG Pretraining
```bash
bash EMG/pretrain_emg.sh
```

### Fine-tuning

#### Standard Fine-tuning
```bash
bash ECG/finetune_ecg.sh  # For ECG
bash EMG/finetune_emg.sh  # For EMG
```

Or run directly:
```bash
torchrun --nproc_per_node=4 finetune.py \
  --train_file path/to/train.h5 \
  --val_file path/to/val.h5 \
  --test_file path/to/test.h5 \
  --pretrained_path ./pretrained/physiowave_ecg_large.pth \
  --num_classes 5 \
  --epochs 20 \
  --lr 2e-4 \
  --output_dir ./finetune_output
```

#### Zero-Shot Evaluation (Linear Probing)
For zero-shot evaluation, simply freeze the encoder and only train the classification head:

```bash
torchrun --nproc_per_node=4 finetune.py \
  --train_file path/to/train.h5 \
  --val_file path/to/val.h5 \
  --test_file path/to/test.h5 \
  --pretrained_path ./pretrained/physiowave_ecg_large.pth \
  --freeze_encoder \  # This freezes the encoder for zero-shot
  --num_classes 5 \
  --epochs 10 \
  --lr 1e-3 \
  --output_dir ./zeroshot_output
```

## 📁 Project Structure

```
PhysioWave/
├── model.py                 # Main PhysioWave model
├── wavelet_modules.py       # Wavelet decomposition components
├── transformer_modules.py   # Transformer encoder with RoPE
├── head_modules.py         # Task-specific output heads
├── dataset.py              # Data loading utilities
├── pretrain.py             # Pretraining script
├── finetune.py             # Fine-tuning script
├── ECG/                    # ECG-specific preprocessing
│   ├── ptbxl_finetune.py  # PTB-XL preprocessing
│   ├── mimic_pretrain.py  # MIMIC-IV preprocessing
│   ├── pretrain_ecg.sh    # ECG pretraining script
│   └── finetune_ecg.sh    # ECG fine-tuning script
├── EMG/                    # EMG-specific preprocessing
│   ├── epn_finetune.py    # EPN-612 preprocessing
│   ├── db6_pretrain.py    # NinaPro DB6 preprocessing
│   ├── pretrain_emg.sh    # EMG pretraining script
│   └── finetune_emg.sh    # EMG fine-tuning script
└── requirements.txt        # Dependencies
```

## 🔧 Key Parameters

### Pretraining Parameters
- `--mask_ratio`: Masking ratio for pretraining (default: 0.7 for ECG, 0.15 for EMG)
- `--masking_strategy`: Choose between 'frequency_guided' or 'random'
- `--importance_ratio`: Weight for frequency importance scoring (default: 0.6)

### Fine-tuning Parameters
- `--freeze_encoder`: Freeze encoder for zero-shot evaluation (linear probing)
- `--label_smoothing`: Label smoothing factor (default: 0.1)
- `--pooling`: Pooling strategy ('mean', 'max', 'first', 'last')

### Model Parameters

- `--in_channels`: Input channels (12 for ECG, 8 for EMG)
- `--max_level`: Wavelet decomposition levels (default: 3)
- `--wavelet_names`: Wavelet families to use (e.g., 'db4', 'db6', 'sym4')
  - **Tip**: You can experiment with different wavelet combinations for your specific signal characteristics. Try:
    - ECG: `db4 db6 sym4 coif2` - Good for capturing QRS complexes
    - EMG: `sym4 sym5 db6 coif3 bior4.4` - Effective for muscle activity patterns
    - Custom combinations: `db1 db2 db3 db4 db5 db6` - Explore Daubechies family
    - The model will learn to adaptively select the best wavelets for your data
- `--embed_dim`: Embedding dimension (256/384/512)
- `--depth`: Number of transformer layers (6/8/12)

## 📈 Training Tips

1. **Data Format**: All data should be in HDF5 format with keys:
   - `data`: Shape (N, C, T) - N samples, C channels, T time points
   - `label`: Shape (N,) - For classification tasks

2. **Multiple Files**: You can train with multiple HDF5 files:
   ```bash
   --train_files file1.h5,file2.h5,file3.h5
   ```

3. **Distributed Training**: Use multiple GPUs for faster training:
   ```bash
   torchrun --nproc_per_node=8 pretrain.py ...
   ```

4. **Memory Optimization**: Use gradient accumulation for larger effective batch sizes:
   ```bash
   --grad_accumulation_steps 4 --batch_size 8
   ```
## 📖 Citation

If you use this code or our pretrained models, please cite our paper:

```bibtex
@article{chen2025physiowave,
  title={PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation},
  author={Chen, Yanlong and Orlandi, Mattia and Rapa, Pierangelo Maria and Benatti, Simone and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2506.10351},
  year={2025}
}
```

## 🙏 Acknowledgments

We thank the authors of the datasets used in this work and the PyTorch team for their excellent framework.

---

**Note**: For the latest updates and discussions, please check our [GitHub repository](https://github.com/ForeverBlue816/PhysioWave) and [paper](https://arxiv.org/abs/2506.10351).


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests. 
