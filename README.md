# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation

<div align="center">

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10351-b31b1b.svg)](https://arxiv.org/abs/2506.10351)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

<div align="center">

**Official PyTorch implementation of PhysioWave, accepted at NeurIPS 2025**

*A novel wavelet-based architecture for physiological signal processing that leverages adaptive multi-scale decomposition and frequency-guided masking to advance self-supervised learning*

</div>

---

## 🌟 Key Features

<table>
<tr>
<td width="50%">

✨ **Learnable Wavelet Decomposition**
- Adaptive multi-resolution analysis
- Soft gating mechanism for optimal wavelet selection

📊 **Frequency-Guided Masking**
- Novel masking strategy prioritizing high-energy components
- Superior to random masking for signal representation

</td>
<td width="50%">

🔗 **Cross-Scale Feature Fusion**
- Attention-based fusion across decomposition levels
- Hierarchical feature integration

🧠 **Multi-Modal Support**
- Unified framework for ECG and EMG signals
- Extensible to other physiological signals

</td>
</tr>
</table>

<div align="center">

📈 **Large-Scale Pretraining**: Models trained on **182GB of ECG** and **823GB of EMG** data

</div>

---

## 🏗️ Model Architecture

<div align="center">

<img src="fig/model.png" alt="PhysioWave Architecture" width="90%">

</div>

### Pipeline Overview

The PhysioWave pretraining pipeline consists of five key stages:

1. **Wavelet Initialization**: Standard wavelet functions (e.g., 'db6', 'sym4') generate learnable low-pass and high-pass filters
2. **Multi-Scale Decomposition**: Adaptive wavelet decomposition produces multi-scale frequency-band representations
3. **Patch Embedding**: Decomposed features are processed into spatio-temporal patches with FFT-based importance scoring
4. **Masked Encoding**: High-scoring patches are masked and processed through Transformer layers with rotary position embeddings
5. **Reconstruction**: Lightweight decoder reconstructs masked patches for self-supervised learning

### Core Components

| Component | Description |
|-----------|-------------|
| 🌊 **Learnable Wavelet Decomposition** | Adaptively selects optimal wavelet bases for input signals |
| 📐 **Multi-Scale Feature Reconstruction** | Hierarchical decomposition with soft gating between scales |
| 🎯 **Frequency-Guided Masking** | Identifies and masks high-energy patches for self-supervised learning |
| 🔄 **Transformer Encoder/Decoder** | Processes masked patches with rotary position embeddings |

---

## 📊 Performance Highlights

### Benchmark Results

<div align="center">

| Task | Dataset | Metric | Performance |
|------|---------|--------|-------------|
| **ECG Arrhythmia** | PTB-XL | Accuracy | **73.1%** |
| **ECG Multi-Label** | CPSC 2018 | F1-Micro | **77.1%** |
| **ECG Multi-Label** | Shaoxing | F1-Micro | **94.6%** |
| **EMG Gesture** | EPN-612 | Accuracy | **94.5%** |

</div>

### Multi-Label Classification Detailed Metrics

<details>
<summary><b>CPSC 2018 Dataset (9-Class Multi-Label)</b></summary>

<div align="center">

| Metric | Micro-Average | Macro-Average |
|--------|---------------|---------------|
| **Precision** | 0.7389 | 0.6173 |
| **Recall** | 0.8059 | 0.6883 |
| **F1-Score** | 0.7709 | 0.6500 |
| **AUROC** | 0.9584 | 0.9280 |

</div>

**Dataset Details:**
- 9 official diagnostic classes (SNR, AF, IAVB, LBBB, RBBB, PAC, PVC, STD, STE)
- 12-lead ECG signals at 500 Hz
- Record-level split to prevent data leakage

</details>

<details>
<summary><b>Chapman-Shaoxing Dataset (4-Class Multi-Label)</b></summary>

<div align="center">

| Metric | Micro-Average | Macro-Average |
|--------|---------------|---------------|
| **Precision** | 0.9389 | 0.9361 |
| **Recall** | 0.9536 | 0.9470 |
| **F1-Score** | 0.9462 | 0.9413 |
| **AUROC** | 0.9949 | 0.9930 |

</div>

**Dataset Details:**
- 4 merged diagnostic classes (SB, AFIB, GSVT, SR)
- 12-lead ECG signals at 500 Hz
- Balanced multi-label distribution

</details>

---

## 💾 Pretrained Models

<div align="center">

### [📥 Download Pretrained Models](https://drive.google.com/drive/folders/1CobMgFT1WIOAHfz1j7Yij3BL6kkjm59k?dmr=1&ec=wgc-drive-globalnav-goto)

</div>

| Model | Parameters | Training Data | Description |
|-------|------------|---------------|-------------|
| `ecg.pth` | 14M | 182GB ECG | ECG pretrained model |
| `emg.pth` | 5M | 823GB EMG | EMG pretrained model |

**Usage:**
```python
# Load pretrained model
checkpoint = torch.load('ecg.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🚀 Quick Start

### Prerequisites

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

### 📦 Data Preparation

<details>
<summary><b>Dataset Download Links</b></summary>

#### ECG Datasets

- [PTB-XL Database](https://physionet.org/content/ptb-xl/1.0.3/) - 21,837 clinical ECG records
- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) - 800K+ ECG recordings
- [PhysioNet Challenge 2021](https://physionet.org/content/challenge-2021/1.0.3/) - Multi-database ECG
- [CPSC 2018](https://www.kaggle.com/competitions/cpsc-2018) - Arrhythmia classification challenge
- [Chapman-Shaoxing](https://www.kaggle.com/datasets/yuty2022/chapmanshaoxing-ecg) - Large-scale 12-lead ECG

#### EMG Datasets

- [EPN-612 Dataset](https://zenodo.org/records/4421500) - 612 hand gestures
- [NinaPro Database DB6](https://ninapro.hevs.ch/instructions/DB6.html) - HD-sEMG recordings

</details>

<details>
<summary><b>Data Format Specifications</b></summary>

#### HDF5 Structure

```python
# Single-label classification
{
    'data': (N, C, T),   # Signal data: float32
    'label': (N,)        # Labels: int64
}

# Multi-label classification
{
    'data': (N, C, T),   # Signal data: float32
    'label': (N, K)      # Multi-hot labels: float32
}
```

**Dimensions:**
- `N` = Number of samples
- `C` = Number of channels
- `T` = Time points
- `K` = Number of classes (multi-label only)

#### Signal Specifications

| Signal | Channels | Length | Sampling Rate | Normalization |
|--------|----------|--------|---------------|---------------|
| **ECG** | 12 | 2048 | 500 Hz | MinMax [-1,1] or Z-score |
| **EMG** | 8 | 1024 | 200-2000 Hz | Max-abs or Z-score |

</details>

### 🔄 Preprocessing Examples

<details>
<summary><b>ECG Preprocessing (PTB-XL - Single-Label)</b></summary>

```bash
# Download PTB-XL dataset
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# Preprocess for single-label classification
python ECG/ptbxl_finetune.py
```

**Output files:**
- `train.h5` - Training data with shape `(N, 12, 2048)`
- `val.h5` - Validation data
- `test.h5` - Test data

**Label format:** `(N,)` with 5 superclasses (NORM, MI, STTC, CD, HYP)

</details>

<details>
<summary><b>ECG Preprocessing (CPSC 2018 - Multi-Label)</b></summary>

```bash
# Preprocess CPSC 2018 dataset
python ECG/cpsc_multilabel.py
```

**Output files:**
- `cpsc_9class_train.h5` - Training data
- `cpsc_9class_val.h5` - Validation data
- `cpsc_9class_test.h5` - Test data
- `cpsc_9class_info.json` - Dataset metadata
- `label_map.json` - Class mappings
- `record_splits.json` - Record-level split info

**Label format:** `(N, 9)` with 9 official CPSC classes

</details>

<details>
<summary><b>ECG Preprocessing (Chapman-Shaoxing - Multi-Label)</b></summary>

```bash
# Preprocess Chapman-Shaoxing dataset
python ECG/shaoxing_multilabel.py
```

**Output files:**
- `train.h5` - Training data
- `val.h5` - Validation data
- `test.h5` - Test data
- `dataset_info.json` - Metadata
- `record_splits.json` - Split information

**Label format:** `(N, 4)` with 4 merged classes (SB, AFIB, GSVT, SR)

</details>

<details>
<summary><b>EMG Preprocessing (EPN-612)</b></summary>

```bash
# Download from Zenodo and preprocess
python EMG/epn_finetune.py
```

**Output files:**
- `epn612_train_set.h5` - Training set `(N, 8, 1024)`
- `epn612_val_set.h5` - Validation set
- `epn612_test_set.h5` - Test set

**Label format:** `(N,)` with 6 gesture classes

</details>

---

## 🎯 Training

### Pretraining

<details>
<summary><b>ECG Pretraining</b></summary>

```bash
# Edit ECG/pretrain_ecg.sh to set data paths
bash ECG/pretrain_ecg.sh
```

**Key parameters:**
```bash
--mask_ratio 0.7                    # Mask 70% of patches
--masking_strategy frequency_guided # Use frequency-guided masking
--importance_ratio 0.7              # Balance importance vs randomness
--epochs 100                        # Pretraining epochs
```

</details>

<details>
<summary><b>EMG Pretraining</b></summary>

```bash
# Edit EMG/pretrain_emg.sh to set data paths
bash EMG/pretrain_emg.sh
```

**Key parameters:**
```bash
--mask_ratio 0.6                    # Mask 60% of patches
--in_channels 8                     # 8-channel EMG
--wave_kernel_size 16               # Smaller kernel for EMG
```

</details>

---

### Fine-tuning

#### Single-Label Classification

<details>
<summary><b>Standard Fine-tuning (ECG/EMG)</b></summary>

```bash
# ECG fine-tuning (PTB-XL)
bash ECG/finetune_ecg.sh

# EMG fine-tuning (EPN-612)
bash EMG/finetune_emg.sh
```

**Example command:**
```bash
torchrun --nproc_per_node=4 finetune.py \
  --train_file path/to/train.h5 \
  --val_file path/to/val.h5 \
  --test_file path/to/test.h5 \
  --pretrained_path pretrained/ecg.pth \
  --task_type classification \
  --num_classes 5 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4
```

</details>

#### Multi-Label Classification

<details>
<summary><b>Multi-Label Fine-tuning (CPSC/Shaoxing)</b></summary>

This repository uses `finetune_multilabel.py` for multi-label classification tasks. First, prepare your data using the corresponding preprocessing scripts.

**CPSC 2018 Example:**

```bash
# Edit paths in ECG/cpsc_multilabel.sh
bash ECG/cpsc_multilabel.sh
```

**Shaoxing Example:**

```bash
# Edit paths in ECG/shaoxing_multilabel.sh
bash ECG/shaoxing_multilabel.sh
```

**Manual command:**

```bash
NUM_GPUS=4
torchrun --nproc_per_node=${NUM_GPUS} finetune_multilabel.py \
  --train_file "path/to/train.h5" \
  --val_file "path/to/val.h5" \
  --test_file "path/to/test.h5" \
  --pretrained_path "path/to/pretrained_ecg/best_model.pth" \
  \
  `# Task Configuration` \
  --task_type multilabel \
  --threshold 0.3 \
  \
  `# Model Architecture` \
  --in_channels 12 \
  --max_level 3 \
  --wave_kernel_size 24 \
  --wavelet_names db4 db6 sym4 coif2 \
  --use_separate_channel \
  --patch_size 64 \
  --embed_dim 384 \
  --depth 8 \
  --num_heads 12 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  \
  `# Training Parameters` \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_epochs 5 \
  --grad_clip 1.0 \
  --use_amp \
  \
  `# Classification Head` \
  --pooling mean \
  --head_hidden_dim 512 \
  --head_dropout 0.2 \
  --label_smoothing 0.1 \
  \
  `# Output` \
  --seed 42 \
  --output_dir "./checkpoints_multilabel"
```

**Key Parameters for Multi-Label:**
- `--task_type multilabel` - Enable multi-label classification
- `--threshold 0.3` - Decision threshold (adjust based on validation)
- `--label_smoothing 0.1` - Regularization for better generalization

</details>

#### Zero-Shot Evaluation

<details>
<summary><b>Linear Probing</b></summary>

Evaluate pretrained representations by freezing the encoder and training only the classification head:

```bash
torchrun --nproc_per_node=4 finetune.py \
  --train_file path/to/train.h5 \
  --val_file path/to/val.h5 \
  --test_file path/to/test.h5 \
  --pretrained_path pretrained/ecg.pth \
  --freeze_encoder \
  --num_classes 5 \
  --epochs 10 \
  --lr 1e-3
```

</details>

---

## 🔧 Configuration Guide

### Model Configuration

<details>
<summary><b>Architecture Parameters</b></summary>

| Parameter | Description | Options | Recommendation |
|-----------|-------------|---------|----------------|
| `--in_channels` | Input channels | 12 (ECG), 8 (EMG) | Match your data |
| `--max_level` | Wavelet decomposition levels | 2-4 | 3 (default) |
| `--wave_kernel_size` | Wavelet kernel size | 16-32 | 24 (ECG), 16 (EMG) |
| `--wavelet_names` | Wavelet families | db, sym, coif, bior | See tips below |
| `--embed_dim` | Embedding dimension | 128-768 | 256/384/512 |
| `--depth` | Transformer layers | 4-12 | 6/8/12 |
| `--num_heads` | Attention heads | 4-16 | 8/12 |
| `--patch_size` | Temporal patch size | 20-128 | 64 (ECG), 32 (EMG) |

**💡 Wavelet Selection Tips:**

| Signal Type | Recommended Wavelets | Rationale |
|-------------|---------------------|-----------|
| **ECG** | `db4 db6 sym4 coif2` | Optimal for QRS complex detection |
| **EMG** | `sym4 sym5 db6 coif3 bior4.4` | Best for muscle activation patterns |
| **Custom** | Experiment with combinations | Domain-specific optimization |

</details>

<details>
<summary><b>Training Configuration</b></summary>

#### Pretraining Parameters

| Parameter | Description | ECG | EMG |
|-----------|-------------|-----|-----|
| `--mask_ratio` | Masking ratio | 0.7 | 0.6 |
| `--masking_strategy` | Masking type | `frequency_guided` | `frequency_guided` |
| `--importance_ratio` | Importance weight | 0.7 | 0.6 |
| `--epochs` | Training epochs | 100 | 100 |
| `--lr` | Learning rate | 2e-5 | 5e-5 |

#### Fine-tuning Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--batch_size` | Batch size per GPU | 16 | 8-64 |
| `--epochs` | Training epochs | 50 | 20-100 |
| `--lr` | Learning rate | 1e-4 | 1e-5 to 1e-3 |
| `--weight_decay` | L2 regularization | 1e-4 | 1e-5 to 1e-3 |
| `--scheduler` | LR scheduler | `cosine` | cosine/step/plateau |
| `--warmup_epochs` | Warmup epochs | 5 | 0-10 |
| `--grad_clip` | Gradient clipping | 1.0 | 0.5-2.0 |

#### Multi-Label Specific

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--threshold` | Decision threshold | 0.3-0.5 | Tune on validation set |
| `--label_smoothing` | Label smoothing | 0.1 | 0.0-0.2 for regularization |
| `--use_class_weights` | Class balancing | False | Enable for imbalanced data |

</details>

<details>
<summary><b>Hardware and Performance</b></summary>

#### Performance Tips

```bash
# Enable mixed precision for 2x speedup
--use_amp

# Increase batch size with gradient accumulation
--batch_size 8 --grad_accumulation_steps 4  # Effective batch size: 32

# Multi-GPU training
torchrun --nproc_per_node=4 [script.py]
```

</details>


## 📖 Citation

If you find our work helpful, please cite:

```bibtex
@article{chen2025physiowave,
  title={PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation},
  author={Chen, Yanlong and Orlandi, Mattia and Rapa, Pierangelo Maria and Benatti, Simone and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2506.10351},
  year={2025}
}
```

---

## 🤝 Contact & Contributions

<div align="center">

**Lead Author:** Yanlong Chen  
**Email:** [yanlchen@student.ethz.ch](mailto:yanlchen@student.ethz.ch)

</div>

We welcome contributions! Feel free to:

- 🐛 [Report bugs](https://github.com/ForeverBlue816/PhysioWave/issues)
- 💡 Suggest enhancements
- 🔧 Submit Pull Requests
- ⭐ Star this repository if you find it useful!

---

## 🙏 Acknowledgments

We thank:
- The authors of PTB-XL, MIMIC-IV-ECG, CPSC 2018, Chapman-Shaoxing, and EPN-612 datasets
- The PyTorch team for their excellent framework
- The open-source community for inspiration and tools

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

<sub>Built with ❤️ for the physiological signal processing community</sub>

[![GitHub stars](https://img.shields.io/github/stars/ForeverBlue816/PhysioWave?style=social)](https://github.com/ForeverBlue816/PhysioWave/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ForeverBlue816/PhysioWave?style=social)](https://github.com/ForeverBlue816/PhysioWave/network/members)

</div>
