"""
BERT-style Wavelet Transformer Multi-Label Classification Fine-tuning Script
Supports distributed training, AMP, and various learning rate schedulers
"""

import os
import math
import argparse
import random
import numpy as np
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score, 
    average_precision_score,
    hamming_loss,
    jaccard_score
)

from model import BERTWaveletTransformer
from dataset_multilabel import (
    MultiLabelTimeSeriesDataset,
    SingleLabelTimeSeriesDataset,
    collate_multilabel_fn,
    collate_singlelabel_fn,
    parse_file_paths,
    DataAugmentation
)


############################################################
# Learning Rate Schedulers
############################################################
class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay"""
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch)
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


############################################################
# Pretrained Model Loading
############################################################
def load_pretrained_feature_extractor(model, pretrained_path, rank=0):
    """Load pretrained feature extractor weights"""
    if not pretrained_path or not os.path.isfile(pretrained_path):
        if rank == 0:
            print(f"No pretrained model found at {pretrained_path}")
        return
    
    if rank == 0:
        print(f"Loading pretrained feature extractor from {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint
    
    model_dict = model.state_dict()
    
    filtered_dict = {}
    skipped_keys = []
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                if rank == 0:
                    print(f"Skipping {k}: shape mismatch {model_dict[k].shape} vs {v.shape}")
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
    
    if rank == 0:
        print(f"Loaded {len(filtered_dict)} pretrained parameters")
        print(f"Skipped {len(skipped_keys)} parameters")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")


############################################################
# Training and Evaluation Functions for Multi-Label
############################################################
def train_one_epoch_multilabel(epoch, rank, model, optimizer, train_loader, device, 
                               criterion, scaler=None, grad_clip=0.0, scheduler=None, 
                               scheduler_per_batch=False):
    """Training function for multi-label classification"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    loader = train_loader
    if rank == 0:
        loader = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, task='downstream', task_name='multilabel', return_logits=True)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, task='downstream', task_name='multilabel', return_logits=True)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            loader.set_postfix({
                "loss": f"{total_loss/total_samples:.4f}",
                "lr": f"{current_lr:.6f}"
            })

    avg_loss = total_loss / total_samples
    avg_lr = optimizer.param_groups[0]['lr']
    if rank == 0:
        print(f"[Train] Epoch {epoch}: Loss={avg_loss:.4f}, LR={avg_lr:.6f}")
    return avg_loss, avg_lr


@torch.no_grad()
def eval_one_epoch_multilabel(epoch, rank, model, loader, device, criterion, 
                              threshold=0.5, desc_prefix="Val"):
    """Evaluation function for multi-label classification"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_probs, all_labels = [], [], []

    display_loader = loader
    if rank == 0:
        display_loader = tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}", ncols=120)

    for x, y in display_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, task='downstream', task_name='multilabel', return_logits=True)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

        if rank == 0:
            display_loader.set_postfix({"loss": f"{total_loss/total_samples:.4f}"})

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    avg_loss = total_loss / total_samples
    
    # Multi-label metrics
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        auroc_micro = roc_auc_score(y_true, y_prob, average='micro')
        auroc_macro = roc_auc_score(y_true, y_prob, average='macro')
    except Exception:
        auroc_micro = float('nan')
        auroc_macro = float('nan')
    
    try:
        ap_micro = average_precision_score(y_true, y_prob, average='micro')
        ap_macro = average_precision_score(y_true, y_prob, average='macro')
    except Exception:
        ap_micro = float('nan')
        ap_macro = float('nan')
    
    hamming = hamming_loss(y_true, y_pred)
    jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)

    if rank == 0:
        print(f"[{desc_prefix}] Epoch {epoch}: Loss={avg_loss:.4f}")
        print(f"  Precision(micro/macro): {precision_micro:.4f}/{precision_macro:.4f}")
        print(f"  Recall(micro/macro):    {recall_micro:.4f}/{recall_macro:.4f}")
        print(f"  F1(micro/macro):        {f1_micro:.4f}/{f1_macro:.4f}")
        print(f"  AUROC(micro/macro):     {auroc_micro:.4f}/{auroc_macro:.4f}")
        print(f"  AP(micro/macro):        {ap_micro:.4f}/{ap_macro:.4f}")
        print(f"  Hamming Loss:           {hamming:.4f}")
        print(f"  Jaccard(micro/macro):   {jaccard_micro:.4f}/{jaccard_macro:.4f}")

    metrics = {
        'loss': avg_loss,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'auroc_micro': auroc_micro,
        'auroc_macro': auroc_macro,
        'ap_micro': ap_micro,
        'ap_macro': ap_macro,
        'hamming_loss': hamming,
        'jaccard_micro': jaccard_micro,
        'jaccard_macro': jaccard_macro
    }
    
    return metrics


############################################################
# Main Training Function
############################################################
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main_worker(rank, world_size, args):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Parse file paths
    train_files = parse_file_paths(args.train_file)
    val_files = parse_file_paths(args.val_file)
    test_files = parse_file_paths(args.test_file) if args.test_file else []

    if rank == 0:
        print("=" * 60)
        print("MULTI-LABEL CLASSIFICATION FINE-TUNING")
        print("=" * 60)
        print(f"\nTask Type: {args.task_type}")
        print(f"Train files: {len(train_files)}")
        for i, f in enumerate(train_files):
            print(f"  {i+1}: {os.path.basename(f)}")
        print(f"Val files: {len(val_files)}")
        for i, f in enumerate(val_files):
            print(f"  {i+1}: {os.path.basename(f)}")
        if test_files:
            print(f"Test files: {len(test_files)}")
            for i, f in enumerate(test_files):
                print(f"  {i+1}: {os.path.basename(f)}")

    # Create datasets
    if args.task_type == 'multilabel':
        train_ds = MultiLabelTimeSeriesDataset(
            train_files, 
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        )
        val_ds = MultiLabelTimeSeriesDataset(
            val_files,
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        )
        test_ds = MultiLabelTimeSeriesDataset(
            test_files,
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        ) if test_files else None
        collate_fn = collate_multilabel_fn
        num_classes_or_labels = train_ds.num_classes
    else:
        train_ds = SingleLabelTimeSeriesDataset(
            train_files,
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        )
        val_ds = SingleLabelTimeSeriesDataset(
            val_files,
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        )
        test_ds = SingleLabelTimeSeriesDataset(
            test_files,
            max_length=args.max_length,
            data_key=args.data_key,
            label_key=args.label_key
        ) if test_files else None
        collate_fn = collate_singlelabel_fn
        num_classes_or_labels = train_ds.num_classes

    if rank == 0:
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val:   {len(val_ds)} samples")
        if test_ds:
            print(f"  Test:  {len(test_ds)} samples")
        print(f"  Number of {'labels' if args.task_type == 'multilabel' else 'classes'}: {num_classes_or_labels}")

    # Create data loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if test_ds else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, 
                             collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                           collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True) if test_ds else None

    # Build head config
    head_config = {
        'hidden_dims': [args.head_hidden_dim] if args.head_hidden_dim else None,
        'dropout': args.head_dropout,
        'pooling': args.pooling,
        'hidden_factor': args.hidden_factor
    }
    
    if args.task_type == 'multilabel':
        head_config['label_smoothing'] = args.label_smoothing
        head_config['use_class_weights'] = args.use_class_weights

    # Create model
    model = BERTWaveletTransformer(
        in_channels=args.in_channels,
        max_level=args.max_level,
        wave_kernel_size=args.wave_kernel_size,
        wavelet_names=args.wavelet_names,
        use_separate_channel=args.use_separate_channel,
        patch_size=(1, args.patch_size),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_pos_embed=args.use_pos_embed,
        pos_embed_type=args.pos_embed_type,
        task_type=args.task_type,
        num_labels=num_classes_or_labels if args.task_type == 'multilabel' else None,
        num_classes=num_classes_or_labels if args.task_type == 'classification' else None,
        head_config=head_config,
        pooling=args.pooling
    ).to(device)
    
    # Initialize weights
    if hasattr(model, 'initialize_weights'):
        model.initialize_weights()
        if rank == 0:
            print("Initialized model weights")
    
    # Load pretrained weights
    if args.pretrained_path:
        load_pretrained_feature_extractor(model, args.pretrained_path, rank)
    
    # Freeze encoder
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if 'task_heads' not in name:
                param.requires_grad = False
        if rank == 0:
            print("Frozen encoder parameters (excluding task heads)")
    
    if rank == 0:
        print("\n===== Model Info =====")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        if frozen_params > 0:
            print(f"Frozen params: {frozen_params:,} ({frozen_params/1e6:.2f}M)")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Loss function
    if args.task_type == 'multilabel':
        criterion = nn.BCEWithLogitsLoss()
        if rank == 0:
            print("Using BCEWithLogitsLoss for multi-label classification")
    else:
        criterion = nn.CrossEntropyLoss()
        if rank == 0:
            print("Using CrossEntropyLoss for multi-class classification")
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Learning rate scheduler
    scheduler = None
    scheduler_per_batch = False
    
    if args.scheduler == 'cosine':
        steps_per_epoch = len(train_loader)
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch if args.warmup_epochs > 0 else int(0.1 * total_steps)
        
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
        scheduler_per_batch = True
        if rank == 0:
            print(f"Using Warmup Cosine Scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}")

    # Training loop
    best_val_metric = float('inf') if args.task_type == 'multilabel' else 0.0
    best_epoch = 0
    history = {'train_loss': [], 'learning_rates': []}
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        if args.task_type == 'multilabel':
            train_loss, current_lr = train_one_epoch_multilabel(
                epoch, rank, model, optimizer, train_loader, device,
                criterion, scaler, args.grad_clip, scheduler, scheduler_per_batch
            )
            
            val_metrics = eval_one_epoch_multilabel(
                epoch, rank, model, val_loader, device, criterion,
                threshold=args.threshold, desc_prefix="Val"
            )
            val_loss = val_metrics['loss']
            val_f1_macro = val_metrics['f1_macro']
        
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['learning_rates'].append(current_lr)
            
            # Save best model
            if args.task_type == 'multilabel':
                current_metric = val_loss
                is_better = current_metric < best_val_metric
            
            if is_better:
                best_val_metric = current_metric
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_metrics': val_metrics if args.task_type == 'multilabel' else {},
                    'args': vars(args),
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"Saved best model at epoch {epoch}")

    if rank == 0:
        print(f"\nBest model at Epoch {best_epoch}, Metric: {best_val_metric:.4f}")
        
        with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Metric: {best_val_metric:.4f}\n")

    # Testing
    if test_loader and rank == 0:
        print("\n===> Testing with best model <===")
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        
        if args.task_type == 'multilabel':
            test_metrics = eval_one_epoch_multilabel(
                "Test", rank, model, test_loader, device, criterion,
                threshold=args.threshold, desc_prefix="Test"
            )
            with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
                json.dump(test_metrics, f, indent=4)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Multi-Label Wavelet Transformer Fine-tuning')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--data_key', type=str, default='data')
    parser.add_argument('--label_key', type=str, default='label')
    parser.add_argument('--max_length', type=int, default=None)
    
    # Task arguments
    parser.add_argument('--task_type', type=str, default='multilabel',
                       choices=['multilabel', 'classification'])
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for multi-label prediction')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default="./multilabel_output")
    
    # Pretrained model
    parser.add_argument('--pretrained_path', type=str, default="")
    parser.add_argument('--freeze_encoder', action='store_true')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'none'])
    
    # Label smoothing (for multi-label)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_class_weights', action='store_true')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--wave_kernel_size', type=int, default=16)
    parser.add_argument('--wavelet_names', nargs='+', default=['db6'])
    parser.add_argument('--use_separate_channel', action='store_true', default=True)
    parser.add_argument('--patch_size', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Head parameters
    parser.add_argument('--head_hidden_dim', type=int, default=None)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'first', 'cls'])
    parser.add_argument('--hidden_factor', type=int, default=2)
    
    # Position embedding
    parser.add_argument('--use_pos_embed', action='store_true', default=True)
    parser.add_argument('--pos_embed_type', type=str, default='2d',
                       choices=['1d', '2d'])

    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    env_world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    main_worker(local_rank, env_world_size, args)


if __name__ == "__main__":
    main()