"""
BERT-style Wavelet Transformer Pretraining Script
Supports multi-GPU training, AMP, gradient accumulation, checkpoint save/resume
Supports loading multiple HDF5 files
"""

import os
import math
import argparse
import random
import numpy as np
from datetime import datetime
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from model import BERTWaveletTransformer
from dataset import create_dataloaders, parse_file_paths


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class BERTPretrainer:
    """BERT-style pretrainer"""
    
    def __init__(self, model, device, rank=0, world_size=1):
        self.model = model
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
    def compute_masked_reconstruction_loss(self, pred_patches, target_patches, mask):
        """
        Compute masked reconstruction loss (MSE loss)
        Fixed version: use pred_patches.new_tensor to avoid gradient issues
        """
        if mask.sum() == 0:
            return pred_patches.new_tensor(0.0)
        
        masked_pred = pred_patches[mask]
        masked_target = target_patches[mask]
        
        loss = F.mse_loss(masked_pred, masked_target)
        return loss
    
    def compute_metrics(self, pred_patches, target_patches, mask, threshold=0.1):
        """Compute evaluation metrics"""
        if mask.sum() == 0:
            return {'accuracy': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
        with torch.no_grad():
            masked_pred = pred_patches[mask]
            masked_target = target_patches[mask]
            
            # RMSE
            mse = F.mse_loss(masked_pred, masked_target)
            rmse = torch.sqrt(mse)
            
            # MAE
            mae = F.l1_loss(masked_pred, masked_target)
            
            # Relative error accuracy
            relative_error = torch.abs(masked_pred - masked_target) / (torch.abs(masked_target) + 1e-8)
            accuracy = (relative_error < threshold).float().mean()
            
            return {
                'accuracy': accuracy.item(),
                'rmse': rmse.item(),
                'mae': mae.item()
            }
    
    def train_step(self, batch, optimizer, scaler, grad_accumulation_steps, 
                   mask_ratio=0.15, grad_clip=1.0):
        """Single training step"""
        x = batch.to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            pred_patches, mask, target_patches = self.model(x, mask_ratio=mask_ratio, task='pretrain')
            loss = self.compute_masked_reconstruction_loss(pred_patches, target_patches, mask)
            
            # Gradient accumulation
            loss = loss / grad_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Compute metrics
        metrics = self.compute_metrics(pred_patches, target_patches, mask)
        metrics['loss'] = loss.item() * grad_accumulation_steps
        metrics['mask_ratio'] = mask.float().mean().item()
        
        return metrics
    
    def validation_step(self, batch, mask_ratio=0.15):
        """Validation step"""
        x = batch.to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_patches, mask, target_patches = self.model(x, mask_ratio=mask_ratio, task='pretrain')
                loss = self.compute_masked_reconstruction_loss(pred_patches, target_patches, mask)
        
        metrics = self.compute_metrics(pred_patches, target_patches, mask)
        metrics['loss'] = loss.item()
        metrics['mask_ratio'] = mask.float().mean().item()
        
        return metrics


def train_one_epoch(epoch, trainer, train_loader, optimizer, scaler, scheduler,
                   grad_accumulation_steps, grad_clip, mask_ratio, rank):
    """Train one epoch"""
    trainer.model.train()
    
    total_metrics = {'loss': 0, 'accuracy': 0, 'rmse': 0, 'mae': 0, 'mask_ratio': 0}
    num_steps = 0
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        # Training step
        metrics = trainer.train_step(
            batch, optimizer, scaler, grad_accumulation_steps, mask_ratio, grad_clip
        )
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_steps += 1
        
        # Gradient accumulation and update
        if (step + 1) % grad_accumulation_steps == 0:
            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
        
        # Update progress bar
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'rmse': f"{metrics['rmse']:.4f}",
                'lr': f"{current_lr:.6f}"
            })
    
    # Calculate average metrics
    avg_metrics = {key: value / num_steps for key, value in total_metrics.items()}
    avg_metrics['lr'] = optimizer.param_groups[0]['lr']
    
    return avg_metrics


@torch.no_grad()
def validate_one_epoch(epoch, trainer, val_loader, mask_ratio, rank):
    """Validate one epoch"""
    trainer.model.eval()
    
    total_metrics = {'loss': 0, 'accuracy': 0, 'rmse': 0, 'mae': 0, 'mask_ratio': 0}
    num_steps = 0
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
    else:
        pbar = val_loader
    
    for batch in pbar:
        metrics = trainer.validation_step(batch, mask_ratio)
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_steps += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'rmse': f"{metrics['rmse']:.4f}"
            })
    
    # Calculate average metrics
    avg_metrics = {key: value / num_steps for key, value in total_metrics.items()}
    
    return avg_metrics


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, metrics, args, 
                   is_best=False, filename='checkpoint.pth'):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'args': args,
    }
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = os.path.join(args.output_dir, 'best_model.pth')
        torch.save(checkpoint, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main_worker(rank, world_size, args):
    """Main training function"""
    # Initialize distributed training
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", 
            init_method="env://", 
            rank=rank, 
            world_size=world_size
        )
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # Print used files
        print("Training files:")
        train_files = parse_file_paths(args.train_files)
        for i, f in enumerate(train_files):
            print(f"  {i+1}. {f}")
        
        if args.val_files:
            print("Validation files:")
            val_files = parse_file_paths(args.val_files)
            for i, f in enumerate(val_files):
                print(f"  {i+1}. {f}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        train_files=args.train_files,
        val_files=args.val_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        normalize=args.normalize,
        use_augmentation=args.use_augmentation,
        task='pretrain',
        distributed=(world_size > 1)
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Val samples: {len(val_loader.dataset)}")
    
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
        masking_strategy=args.masking_strategy,
        importance_ratio=args.importance_ratio,
        mask_ratio=args.mask_ratio,
        task_type='pretrain'
    ).to(device)

    # Initialize weights
    model.initialize_weights()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Distributed model - fixed version
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # Changed to True
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        steps_per_epoch = len(train_loader) // args.grad_accumulation_steps
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)
        
        if rank == 0:
            print(f"Using Cosine scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}")
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # Create trainer
    trainer = BERTPretrainer(model, device, rank, world_size)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print(f"Loading checkpoint '{args.resume}'")
            start_epoch, metrics = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler
            )
            best_val_loss = metrics.get('val_loss', float('inf'))
            if rank == 0:
                print(f"Resumed from epoch {start_epoch}")
        else:
            if rank == 0:
                print(f"No checkpoint found at '{args.resume}'")
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_rmse': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_rmse': [],
        'learning_rates': []
    }
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch (for distributed sampling)
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            if val_loader:
                val_loader.sampler.set_epoch(epoch)
        
        # Training
        start_time = time.time()
        train_metrics = train_one_epoch(
            epoch, trainer, train_loader, optimizer, scaler, scheduler,
            args.grad_accumulation_steps, args.grad_clip, args.mask_ratio, rank
        )
        train_time = time.time() - start_time
        
        # Validation
        val_metrics = {}
        if val_loader:
            start_time = time.time()
            val_metrics = validate_one_epoch(
                epoch, trainer, val_loader, args.mask_ratio, rank
            )
            val_time = time.time() - start_time
        
        # Print results
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, "
                  f"Acc={train_metrics['accuracy']:.3f}, "
                  f"RMSE={train_metrics['rmse']:.4f}, "
                  f"LR={train_metrics['lr']:.6f}, "
                  f"Time={train_time:.1f}s")
            
            if val_metrics:
                print(f"  Val:   Loss={val_metrics['loss']:.4f}, "
                      f"Acc={val_metrics['accuracy']:.3f}, "
                      f"RMSE={val_metrics['rmse']:.4f}, "
                      f"Time={val_time:.1f}s")
        
        # Record history
        if rank == 0:
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['train_rmse'].append(train_metrics['rmse'])
            history['learning_rates'].append(train_metrics['lr'])
            
            if val_metrics:
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_rmse'].append(val_metrics['rmse'])
        
        # Save checkpoints
        if rank == 0:
            # Current checkpoint
            save_checkpoint(
                epoch, model, optimizer, scheduler, scaler,
                {**train_metrics, **val_metrics}, args,
                filename='latest_checkpoint.pth'
            )
            
            # Best checkpoint
            current_val_loss = val_metrics.get('loss', train_metrics['loss'])
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(
                    epoch, model, optimizer, scheduler, scaler,
                    {**train_metrics, **val_metrics}, args,
                    is_best=True, filename='best_checkpoint.pth'
                )
                print(f"  New best model saved with val_loss={current_val_loss:.4f}")
            
            # Periodic save
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(
                    epoch, model, optimizer, scheduler, scaler,
                    {**train_metrics, **val_metrics}, args,
                    filename=f'checkpoint_epoch_{epoch+1}.pth'
                )
            
            # Save training history
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=4)
    
    # Training completed
    if rank == 0:
        print(f"\nTraining completed! Best val_loss: {best_val_loss:.4f}")
        
        # Generate training report
        with open(os.path.join(args.output_dir, 'training_summary.txt'), 'w') as f:
            f.write("BERT Wavelet Transformer Pretraining Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed at: {datetime.now()}\n")
            f.write(f"Total epochs: {args.epochs}\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            f.write(f"Final learning rate: {train_metrics['lr']:.6f}\n")
            f.write(f"Model parameters: {total_params:,}\n")
            f.write(f"Training files: {len(parse_file_paths(args.train_files))}\n")
            if args.val_files:
                f.write(f"Validation files: {len(parse_file_paths(args.val_files))}\n")
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='BERT Wavelet Transformer Pretraining')
    
    # Data arguments - modified to support multiple files
    parser.add_argument('--train_files', type=str, required=True, 
                       help='Training data files (comma or space separated for multiple files)')
    parser.add_argument('--val_files', type=str, 
                       help='Validation data files (comma or space separated for multiple files)')
    parser.add_argument('--max_length', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize data')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'])
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    # Model arguments
    parser.add_argument('--in_channels', type=int, default=8, help='Input channels')
    parser.add_argument('--max_level', type=int, default=3, help='Wavelet decomposition levels')
    parser.add_argument('--wave_kernel_size', type=int, default=16, help='Wavelet kernel size')
    parser.add_argument('--wavelet_names', nargs='+', default=['db4', 'db6', 'sym4'], help='Wavelet names')
    parser.add_argument('--use_separate_channel', action='store_true', default=True, help='Separate channel processing')
    parser.add_argument('--patch_size', type=int, default=20, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Position embedding
    parser.add_argument('--use_pos_embed', action='store_true', default=True, help='Use position embedding')
    parser.add_argument('--pos_embed_type', type=str, default='2d', choices=['1d', '2d'], help='Position embedding type')
    
    # Masking strategy
    parser.add_argument('--masking_strategy', type=str, default='frequency_guided', 
                        choices=['random', 'frequency_guided'], help='Masking strategy')
    parser.add_argument('--importance_ratio', type=float, default=0.6, help='Frequency importance ratio')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Mask ratio')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data workers')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--output_dir', type=str, default='./pretrain_output', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Get distributed information
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    
    # Start training
    main_worker(local_rank, world_size, args)


if __name__ == "__main__":
    main()