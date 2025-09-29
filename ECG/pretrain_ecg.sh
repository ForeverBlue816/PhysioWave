#!/bin/bash

# ============================================================================
# ECG Pretraining Script for PhysioWave
# ============================================================================
# This script demonstrates how to launch distributed pretraining for 
# ECG signals using the BERT Wavelet Transformer architecture.
#
# Usage:
#   1. Modify the paths to point to your ECG data files
#   2. Adjust hyperparameters based on your hardware and dataset
#   3. Run: bash pretrain_ecg_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=4

# Data paths (modify these to point to your preprocessed ECG HDF5 files)
# Multiple files can be specified using comma separation
TRAIN_FILES="path/to/ecg_train1.h5,path/to/ecg_train2.h5,path/to/ecg_train3.h5"
VAL_FILES="path/to/ecg_val.h5"

# Output directory for checkpoints and logs
OUTPUT_DIR="./pretrain_ecg_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed training for ECG
torchrun --nproc_per_node=${NUM_GPUS} pretrain.py \
  --train_files "${TRAIN_FILES}" \
  --val_files "${VAL_FILES}" \
  \
  `# ECG Model Architecture Parameters` \
  --in_channels 12 \                    # 12-lead ECG
  --max_level 3 \                        # Number of wavelet decomposition levels
  --wave_kernel_size 24 \                # Size of wavelet kernels for ECG
  --wavelet_names db4 db6 sym4 coif2 \   # Wavelet families suitable for ECG
  --use_separate_channel \               # Channel-wise processing for each lead
  --patch_size 64 \                      # Temporal patch size
  --embed_dim 384 \                      # Transformer embedding dimension
  --depth 8 \                            # Number of Transformer layers
  --num_heads 12 \                       # Number of attention heads
  --mlp_ratio 4.0 \                      # MLP expansion ratio
  --dropout 0.1 \                        # Dropout rate
  \
  `# Position Embedding` \
  --use_pos_embed \                      # Enable position embeddings
  --pos_embed_type 2d \                  # 2D position encoding for time-frequency
  \
  `# ECG Data Processing` \
  --max_length 2048 \                    # Maximum ECG sequence length
  --batch_size 16 \                      # Batch size per GPU
  --num_workers 8 \                      # Data loading workers
  \
  `# Training Parameters` \
  --epochs 20 \                          # Total training epochs
  --lr 2e-5 \                           # Learning rate
  --weight_decay 1e-3 \                  # Weight decay for regularization
  --grad_accumulation_steps 2 \          # Gradient accumulation steps
  --grad_clip 1.0 \                      # Gradient clipping threshold
  --use_amp \                            # Use automatic mixed precision
  \
  `# Learning Rate Scheduler` \
  --scheduler cosine \                   # Cosine annealing scheduler
  --warmup_epochs 5 \                    # Linear warmup epochs
  \
  `# Frequency-Guided Masking for ECG` \
  --mask_ratio 0.7 \                     # Mask 70% of patches
  --masking_strategy frequency_guided \  # Frequency-based masking for ECG features
  --importance_ratio 0.7 \                # Weight for frequency importance scoring
  \
  `# Checkpointing` \
  --save_freq 10 \                       # Save checkpoint every 10 epochs
  --seed 42 \                            # Random seed for reproducibility
  --output_dir "${OUTPUT_DIR}"

echo "ECG pretraining completed. Results saved to ${OUTPUT_DIR}"