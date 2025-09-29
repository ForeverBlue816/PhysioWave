#!/bin/bash

# ============================================================================
# Fine-tuning Script for BERT Wavelet Transformer
# ============================================================================
# This script demonstrates how to fine-tune a pretrained BERT Wavelet 
# Transformer model for downstream classification tasks.
#
# Usage:
#   1. Modify the paths to point to your dataset files
#   2. Update the pretrained model path
#   3. Adjust num_classes based on your classification task
#   4. Run: bash finetune_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=4

# Data paths (modify these to point to your dataset)
TRAIN_FILE="path/to/train_set.h5"
VAL_FILE="path/to/val_set.h5"
TEST_FILE="path/to/test_set.h5"

# Pretrained model checkpoint
PRETRAINED_MODEL="path/to/pretrained/best_model.pth"

# Output directory for fine-tuning results
OUTPUT_DIR="./finetune_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed fine-tuning
torchrun --nproc_per_node=${NUM_GPUS} finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  --pretrained_path "${PRETRAINED_MODEL}" \
  \
  `# Model Architecture (must match pretrained model)` \
  --in_channels 8 \                        # Number of input channels
  --max_level 3 \                          # Wavelet decomposition levels
  --wave_kernel_size 16 \                  # Wavelet kernel size
  --wavelet_names sym4 sym5 db6 coif3 bior4.4 \  # Wavelet families
  --use_separate_channel \                 # Channel-wise processing
  --patch_size 64 \                        # Temporal patch size
  --embed_dim 256 \                        # Embedding dimension
  --depth 6 \                              # Number of Transformer layers
  --num_heads 8 \                          # Number of attention heads
  --mlp_ratio 4.0 \                        # MLP expansion ratio
  --dropout 0.1 \                          # Dropout rate
  \
  `# Position Embedding` \
  --use_pos_embed \                        # Enable position embeddings
  --pos_embed_type 2d \                    # 2D position encoding
  \
  `# Fine-tuning Parameters` \
  --batch_size 32 \                        # Batch size per GPU
  --epochs 5 \                             # Fine-tuning epochs (fewer than pretraining)
  --lr 2e-4 \                              # Learning rate (higher than pretraining)
  --weight_decay 1e-3 \                    # Weight decay
  --grad_clip 1.0 \                        # Gradient clipping
  --use_amp \                              # Use automatic mixed precision
  --num_workers 8 \                        # Data loading workers
  --world_size ${NUM_GPUS} \               # Number of distributed processes
  \
  `# Learning Rate Scheduler` \
  --scheduler cosine \                     # Cosine annealing scheduler
  --warmup_epochs 2 \                      # Short warmup for fine-tuning
  \
  `# Classification Head Configuration` \
  --num_classes 6 \                        # Number of classes for your task
  --pooling mean \                         # Pooling strategy (mean, max, first, last)
  --head_dropout 0.1 \                     # Dropout in classification head
  --head_hidden_dim 512 \                  # Hidden dimension in classification head
  --label_smoothing 0.1 \                  # Label smoothing for regularization
  \
  `# Other Settings` \
  --seed 42 \                              # Random seed for reproducibility
  --output_dir "${OUTPUT_DIR}"

echo "Fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"