#!/bin/bash

# ============================================================================
# ECG Fine-tuning Script for PhysioWave
# ============================================================================
# This script demonstrates how to fine-tune a pretrained BERT Wavelet 
# Transformer model for ECG classification tasks (e.g., arrhythmia detection).
#
# Usage:
#   1. Modify the paths to point to your ECG dataset files
#   2. Update the pretrained model path
#   3. Adjust num_classes based on your ECG classification task
#   4. Run: bash finetune_ecg_example.sh
# ============================================================================

# Number of GPUs to use for distributed training
NUM_GPUS=4

# Data paths (modify these to point to your ECG dataset)
TRAIN_FILE="path/to/ecg_train.h5"
VAL_FILE="path/to/ecg_val.h5"
TEST_FILE="path/to/ecg_test.h5"

# Pretrained ECG model checkpoint
PRETRAINED_MODEL="path/to/pretrained_ecg/best_model.pth"

# Output directory for fine-tuning results
OUTPUT_DIR="./finetune_ecg_output"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Launch distributed fine-tuning for ECG
torchrun --nproc_per_node=${NUM_GPUS} finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  --pretrained_path "${PRETRAINED_MODEL}" \
  \
  `# ECG Model Architecture (must match pretrained model)` \
  --in_channels 12 \                       # 12-lead ECG
  --max_level 3 \                          # Wavelet decomposition levels
  --wave_kernel_size 24 \                  # Wavelet kernel size for ECG
  --wavelet_names db4 db6 sym4 coif2 \     # Wavelet families for ECG
  --use_separate_channel \                 # Process each ECG lead separately
  --patch_size 64 \                        # Temporal patch size
  --embed_dim 384 \                        # Embedding dimension
  --depth 8 \                              # Number of Transformer layers
  --num_heads 12 \                         # Number of attention heads
  --mlp_ratio 4.0 \                        # MLP expansion ratio
  --dropout 0.1 \                          # Dropout rate
  \
  `# Position Embedding` \
  --use_pos_embed \                        # Enable position embeddings
  --pos_embed_type 2d \                    # 2D position encoding for time-frequency
  \
  `# Fine-tuning Parameters` \
  --batch_size 16 \                        # Batch size per GPU
  --epochs 20 \                            # Fine-tuning epochs
  --lr 2e-4 \                              # Initial learning rate
  --min_lr 1e-6 \                          # Minimum learning rate
  --weight_decay 1e-3 \                    # Weight decay for regularization
  --grad_clip 1.0 \                        # Gradient clipping threshold
  --use_amp \                              # Use automatic mixed precision
  --num_workers 8 \                        # Data loading workers
  --world_size ${NUM_GPUS} \               # Number of distributed processes
  \
  `# Learning Rate Scheduler` \
  --scheduler cosine \                     # Cosine annealing scheduler
  --warmup_epochs 5 \                      # Warmup epochs for stable training
  \
  `# ECG Classification Head Configuration` \
  --num_classes 5 \                        # Number of ECG classes (e.g., rhythm types)
  --pooling mean \                         # Mean pooling over sequence
  --head_hidden_dim 1024 \                 # Hidden dimension in classification head
  --head_dropout 0.1 \                     # Dropout in classification head
  \
  `# Other Settings` \
  --seed 42 \                              # Random seed for reproducibility
  --output_dir "${OUTPUT_DIR}"

echo "ECG fine-tuning completed. Results saved to ${OUTPUT_DIR}"
echo "Best model checkpoint: ${OUTPUT_DIR}/best_model.pth"
echo "Test results: ${OUTPUT_DIR}/test_results.json"
echo "Training metrics: ${OUTPUT_DIR}/training_metrics.json"