#!/bin/bash

# ============================================================================
# EMG Pretraining Script for PhysioWave
# ============================================================================
# This script demonstrates how to launch distributed pretraining for 
# EMG signals using the BERT Wavelet Transformer architecture.
#
# Usage:
#   1. Modify the paths to point to your EMG data files
#   2. Adjust hyperparameters based on your hardware and dataset
#   3. Run: bash pretrain_emg_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=4

# Data paths (modify these to point to your preprocessed EMG HDF5 files)
# Multiple files can be specified using comma separation
TRAIN_FILES="path/to/emg_train1.h5,path/to/emg_train2.h5,path/to/emg_train3.h5"
VAL_FILES="path/to/emg_val.h5"

# Output directory for checkpoints and logs
OUTPUT_DIR="./pretrain_emg_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed training for EMG
torchrun --nproc_per_node=${NUM_GPUS} pretrain.py \
  --train_files "${TRAIN_FILES}" \
  --val_files "${VAL_FILES}" \
  \
  `# EMG Model Architecture Parameters` \
  --in_channels 8 \                        # 8 channels for EMG
  --max_level 3 \                          # Number of wavelet decomposition levels
  --wave_kernel_size 16 \                  # Size of wavelet kernels for EMG
  --wavelet_names sym4 sym5 db6 coif3 bior4.4 \  # Wavelet families suitable for EMG
  --use_separate_channel \                 # Channel-wise processing for each electrode
  --patch_size 64 \                        # Temporal patch size
  --embed_dim 256 \                        # Transformer embedding dimension
  --depth 6 \                              # Number of Transformer layers
  --num_heads 8 \                          # Number of attention heads
  --mlp_ratio 4.0 \                        # MLP expansion ratio
  --dropout 0.1 \                          # Dropout rate
  \
  `# Position Embedding` \
  --use_pos_embed \                        # Enable position embeddings
  --pos_embed_type 2d \                    # 2D position encoding for time-frequency
  \
  `# EMG Data Processing` \
  --max_length 2048 \                      # Maximum EMG sequence length
  --batch_size 32 \                        # Batch size per GPU
  --num_workers 8 \                        # Data loading workers
  \
  `# Training Parameters` \
  --epochs 30 \                            # Total pretraining epochs
  --lr 1e-4 \                             # Learning rate for pretraining
  --weight_decay 0.01 \                    # Weight decay for regularization
  --grad_accumulation_steps 2 \            # Gradient accumulation steps
  --grad_clip 1.0 \                        # Gradient clipping threshold
  --use_amp \                              # Use automatic mixed precision
  \
  `# Learning Rate Scheduler` \
  --scheduler cosine \                     # Cosine annealing scheduler
  --warmup_epochs 10 \                     # Linear warmup epochs
  \
  `# Frequency-Guided Masking for EMG` \
  --mask_ratio 0.6 \                      # Mask 60% of patches
  --masking_strategy frequency_guided \    # Frequency-based masking for EMG features
  --importance_ratio 0.6 \                 # Weight for frequency importance scoring
  \
  `# Checkpointing` \
  --save_freq 10 \                         # Save checkpoint every 10 epochs
  --seed 42 \                              # Random seed for reproducibility
  --output_dir "${OUTPUT_DIR}"

echo "EMG pretraining completed. Results saved to ${OUTPUT_DIR}"
echo "Use the best_model.pth from ${OUTPUT_DIR} for downstream fine-tuning tasks"
