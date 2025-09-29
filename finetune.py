"""
BERT-style Wavelet Transformer Downstream Task Fine-tuning Script
Uses feature extractor + classification head architecture
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
import h5py
from tqdm import tqdm

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from model import BERTWaveletTransformer  # Use original BERT model with built-in classification head




############################################################
# Label Smoothing Cross Entropy Loss
############################################################
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_prob).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


############################################################
# Learning Rate Schedulers
############################################################
class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay."""
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
# Dataset Definition (No normalization version)
############################################################
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Time series dataset without normalization"""
    def __init__(self, file_paths, data_key="data", label_key="label"):
        super().__init__()
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        elif isinstance(file_paths, (list, tuple)):
            file_paths = list(file_paths)
        else:
            raise ValueError("file_paths must be a string or list of strings")
        
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File {file_path} not found.")
        
        self.file_paths = file_paths
        self.data_key = data_key
        self.label_key = label_key
        
        self._load_data()
    
    def _load_data(self):
        all_data = []
        all_labels = []
        
        print(f"Loading {len(self.file_paths)} file(s)...")
        
        for i, file_path in enumerate(self.file_paths):
            print(f"  Loading file {i+1}/{len(self.file_paths)}: {os.path.basename(file_path)}")
            
            with h5py.File(file_path, "r") as h5f:
                if self.data_key not in h5f:
                    raise KeyError(f"Key '{self.data_key}' not found in {file_path}")
                if self.label_key not in h5f:
                    raise KeyError(f"Key '{self.label_key}' not found in {file_path}")
                
                data = h5f[self.data_key][:]
                labels = h5f[self.label_key][:]
                
                all_data.append(data)
                all_labels.append(labels)
                
                print(f"    Data shape: {data.shape}, Labels shape: {labels.shape}")
        
        self._data = np.concatenate(all_data, axis=0)
        self._labels = np.concatenate(all_labels, axis=0)
        self._num_samples = len(self._data)
        
        print(f"Combined dataset: {self._data.shape} data, {self._labels.shape} labels")
        print(f"Total samples: {self._num_samples}")
        
        unique_labels, counts = np.unique(self._labels, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({count/self._num_samples*100:.1f}%)")

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int):
        x = torch.tensor(self._data[idx], dtype=torch.float32)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y
    
    @property
    def data_shape(self):
        return self._data[0].shape
    
    @property
    def num_classes(self):
        return len(np.unique(self._labels))


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_tensor = torch.stack(xs, dim=0)
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return xs_tensor, ys_tensor


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
            if rank == 0:
                print(f"Skipping {k}: not found in current model")
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
# Training and Evaluation Functions
############################################################
def train_one_epoch(epoch, rank, model, optimizer, train_loader, device, criterion, 
                    scaler=None, grad_clip=0.0, scheduler=None, scheduler_per_batch=False):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    loader = train_loader
    if rank == 0:
        loader = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, task='classify')  # BERT model uses task='classify'
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, task='classify')  # BERT model uses task='classify'
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
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            loader.set_postfix({
                "loss": f"{total_loss/total_samples:.4f}", 
                "acc": f"{total_correct/total_samples:.4f}",
                "lr": f"{current_lr:.6f}"
            })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    if rank == 0:
        print(f"[Train] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, LR={current_lr:.6f}")
    return avg_loss, avg_acc, current_lr


@torch.no_grad()
def eval_one_epoch(epoch, rank, model, loader, device, criterion, desc_prefix="Eval"):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    display_loader = loader
    if rank == 0:
        display_loader = tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}", ncols=120)

    for x, y in display_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, task='classify')  # BERT model uses task='classify'
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (preds == y).sum().item()

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

        if rank == 0:
            display_loader.set_postfix({"loss": f"{total_loss/total_samples:.4f}", "acc": f"{total_correct/total_samples:.4f}"})

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
    except Exception:
        auroc = float('nan')

    if rank == 0:
        print(f"[{desc_prefix}] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
              f"BalAcc={balanced_acc:.4f}, Kappa={kappa:.4f}, WF1={weighted_f1:.4f}, AUROC={auroc:.4f}")

    return avg_loss, avg_acc, balanced_acc, kappa, weighted_f1, auroc


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

    def parse_file_paths(file_arg):
        if not file_arg:
            return []
        return [path.strip() for path in file_arg.split(',') if path.strip()]
    
    train_files = parse_file_paths(args.train_file)
    val_files = parse_file_paths(args.val_file)
    test_files = parse_file_paths(args.test_file) if args.test_file else []

    # Create datasets
    if val_files:
        train_ds = TimeSeriesDataset(train_files)
        val_ds = TimeSeriesDataset(val_files)
        
        if rank == 0:
            print("===== Dataset Info =====")
            print("Using separate validation files")
    else:
        if rank == 0:
            print("===== Dataset Info =====")
            print(f"No validation files provided. Splitting from training data (ratio: {args.val_split_ratio})")
        
        full_train_ds = TimeSeriesDataset(train_files)
        
        total_samples = len(full_train_ds)
        val_samples = int(total_samples * args.val_split_ratio)
        train_samples = total_samples - val_samples
        
        if rank == 0:
            print(f"Total samples: {total_samples}")
            print(f"Train samples: {train_samples}")
            print(f"Val samples: {val_samples}")
        
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = torch.utils.data.random_split(
            full_train_ds, [train_samples, val_samples], generator=generator
        )
    
    test_ds = TimeSeriesDataset(test_files) if test_files else None

    if rank == 0:
        print(f"Train files: {len(train_files)}")
        for i, f in enumerate(train_files):
            print(f"  {i+1}: {os.path.basename(f)}")
        if val_files:
            print(f"Val files: {len(val_files)}")
            for i, f in enumerate(val_files):
                print(f"  {i+1}: {os.path.basename(f)}")
        if test_files:
            print(f"Test files: {len(test_files)}")
            for i, f in enumerate(test_files):
                print(f"  {i+1}: {os.path.basename(f)}")
        print()
        print(f"Final dataset sizes:")
        print(f"  Train samples = {len(train_ds)}")
        print(f"  Val   samples = {len(val_ds)}")
        if test_ds:
            print(f"  Test  samples = {len(test_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if test_ds else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True) if test_ds else None

    # Get data shape
    if val_files:
        sample_x, _ = train_ds[0]
    else:
        sample_x, _ = train_ds.dataset[0]
    C, T = sample_x.shape
    freq_bands = args.in_channels * (args.max_level + 1)
    time_patches = T // args.patch_size
    
    head_config = {
        'hidden_dims': [args.head_hidden_dim] if args.head_hidden_dim else None,
        'dropout': args.head_dropout,
        'pooling': args.pooling
    }
    
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
        task_type='classification',
        num_classes=args.num_classes,
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
    
    # Freeze encoder (excluding task heads)
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
        
        print(f"Total params: {total_params:,}  ({total_params/1e6:.2f} M)")
        print(f"Trainable params: {trainable_params:,}  ({trainable_params/1e6:.2f} M)")
        if frozen_params > 0:
            print(f"Frozen params: {frozen_params:,}  ({frozen_params/1e6:.2f} M)")
        print(f"Pooling strategy: {args.pooling}")
        print(f"Head hidden dim: {args.head_hidden_dim}")
        if args.pretrained_path:
            print(f"Pretrained model: {args.pretrained_path}")
            print(f"Freeze encoder: {args.freeze_encoder}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        if rank == 0:
            print(f"Using label smoothing: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
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
    
    elif args.scheduler == 'cosine_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.min_lr
        )
        scheduler_per_batch = True
        if rank == 0:
            print(f"Using Cosine Annealing with Warm Restarts")
    
    elif args.scheduler == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1
        )
        scheduler_per_batch = True
        if rank == 0:
            print(f"Using OneCycle Scheduler")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_balanced_acc = 0.0
    best_kappa = 0.0
    best_weighted_f1 = 0.0
    best_auroc = 0.0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': [],
        'val_kappa': [],
        'val_weighted_f1': [],
        'val_auroc': [],
        'learning_rates': []
    }
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        train_loss, train_acc, current_lr = train_one_epoch(
            epoch, rank, model, optimizer, train_loader, device, 
            criterion, scaler, args.grad_clip, scheduler, scheduler_per_batch
        )
        
        val_metrics = eval_one_epoch(
            epoch, rank, model, val_loader, device, criterion, desc_prefix="Val"
        )
        val_loss, val_acc, val_balanced_acc, val_kappa, val_weighted_f1, val_auroc = val_metrics
        
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_balanced_acc'].append(val_balanced_acc)
            history['val_kappa'].append(val_kappa)
            history['val_weighted_f1'].append(val_weighted_f1)
            history['val_auroc'].append(val_auroc)
            history['learning_rates'].append(current_lr)
        
        if rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_balanced_acc = val_balanced_acc
                best_kappa = val_kappa
                best_weighted_f1 = val_weighted_f1
                best_auroc = val_auroc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'val_kappa': val_kappa,
                    'val_weighted_f1': val_weighted_f1,
                    'val_auroc': val_auroc,
                    'args': vars(args),
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"Saved best model at epoch {epoch}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args),
            }, os.path.join(args.output_dir, "latest_model.pth"))

    if rank == 0:
        print(f"\nBest Validation: Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}, "
              f"BalAcc={best_balanced_acc:.4f}, Kappa={best_kappa:.4f}, "
              f"WF1={best_weighted_f1:.4f}, AUROC={best_auroc:.4f} at Epoch {best_epoch}")
        
        with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
            f.write("="*50 + "\n")
            f.write("BERT WAVELET TRANSFORMER FINETUNING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Best Validation Metrics:\n")
            f.write("-"*30 + "\n")
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"best_val_loss: {best_val_loss:.4f}\n")
            f.write(f"best_val_acc: {best_val_acc:.4f}\n")
            f.write(f"best_balanced_acc: {best_balanced_acc:.4f}\n")
            f.write(f"best_kappa: {best_kappa:.4f}\n")
            f.write(f"best_weighted_f1: {best_weighted_f1:.4f}\n")
            f.write(f"best_auroc: {best_auroc:.4f}\n")
            if args.pretrained_path:
                f.write(f"pretrained_model: {args.pretrained_path}\n")
                f.write(f"freeze_encoder: {args.freeze_encoder}\n")

    # Testing
    if test_loader:
        if rank == 0:
            print("\n===> Testing with best model <===")
            checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"))
            model.module.load_state_dict(checkpoint['model_state_dict'])
        dist.barrier()
        for param in model.parameters(): 
            dist.broadcast(param.data, src=0)
        
        test_metrics = eval_one_epoch(
            "Test", rank, model, test_loader, device, criterion, desc_prefix="Test"
        )
        
        if rank == 0:
            test_loss, test_acc, test_balanced_acc, test_kappa, test_weighted_f1, test_auroc = test_metrics
            test_results = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_balanced_acc': test_balanced_acc,
                'test_kappa': test_kappa,
                'test_weighted_f1': test_weighted_f1,
                'test_auroc': test_auroc
            }
            
            with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='BERT Wavelet Transformer Finetuning')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True, help='Training data file(s), comma-separated')
    parser.add_argument('--val_file', type=str, default="", help='Validation file(s). If not provided, will split from training data')
    parser.add_argument('--test_file', type=str, default="", help='Test data file(s), comma-separated')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Validation split ratio when val_file is not provided')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient clipping (0 to disable)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs for distributed training')
    parser.add_argument('--output_dir', type=str, default="./bert_finetune_output", help='Output directory')
    
    # Pretrained model arguments
    parser.add_argument('--pretrained_path', type=str, default="", help='Path to pretrained feature extractor checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze feature extractor, only train classification head')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'cosine_restarts', 'onecycle', 'linear', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--T_0', type=int, default=10, help='CosineAnnealingWarmRestarts T_0')
    parser.add_argument('--T_mult', type=int, default=2, help='CosineAnnealingWarmRestarts T_mult')
    
    # Label smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    
    # Model parameters - Feature Extractor
    parser.add_argument('--in_channels', type=int, default=8, help='Input channels')
    parser.add_argument('--max_level', type=int, default=3, help='Wavelet decomposition levels')
    parser.add_argument('--wave_kernel_size', type=int, default=16, help='Wavelet kernel size')
    parser.add_argument('--wavelet_names', nargs='+', default=['db6'], help='Wavelet names')
    parser.add_argument('--use_separate_channel', action='store_true', default=True, help='Separate channel wavelet processing')
    parser.add_argument('--patch_size', type=int, default=20, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Model parameters - Classification Head
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--head_hidden_dim', type=int, default=None, help='Classification head hidden dimension')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='Classification head dropout')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'first', 'last'],
                        help='Pooling strategy for classification')
    
    # Position embedding parameters
    parser.add_argument('--use_pos_embed', action='store_true', default=True, help='Use position embedding')
    parser.add_argument('--pos_embed_type', type=str, default='2d', choices=['1d', '2d'], help='Position embedding type')

    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    env_world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    if env_world_size != args.world_size:
        print(f"[Warning] WORLD_SIZE {env_world_size} != --world_size {args.world_size}")
    
    # Start training
    main_worker(local_rank, env_world_size, args)


if __name__ == "__main__":
    main()