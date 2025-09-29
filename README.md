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
- **Multi-Modal Support**: Unified framework for ECG, EMG, and EEG signals
- **Large-Scale Pretraining**: Models trained on 182GB of ECG and 823GB of EMG data

## 📊 Performance

### State-of-the-Art Results(Example)
- **PTB-XL (ECG Arrhythmia)**: 73.1% Accuracy
- **EPN-612 (EMG Gesture)**: 94.5% Accuracy

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

#### ECG Data (PTB-XL Example)
```bash
# Preprocess PTB-XL dataset for fine-tuning
python ECG/ptbxl_finetune.py

# Preprocess MIMIC-IV for pretraining
python ECG/mimic_pretrain.py
```

#### EMG Data (EPN-612 Example)
```bash
# Preprocess EPN-612 dataset
python EMG/epn_finetune.py

# Preprocess NinaPro DB6 for pretraining
python EMG/db6_pretrain.py
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

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.


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
