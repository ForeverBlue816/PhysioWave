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
    classification_report,
    confusion_matrix
)

from model import BERTWaveletTransformer  # Use original BERT model with built-in classification head




############################################################
# Label Smoothing Cross Entropy Loss
############################################################
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss, optionally weighted per class.

    ``weight`` follows nn.CrossEntropyLoss: a per-class tensor, and the 'mean'
    reduction divides by the summed weight of the targets rather than by the
    batch size. Without that normalisation the loss scale would move with the
    class mix of each batch, which changes the effective learning rate from
    step to step on an imbalanced task.
    """
    def __init__(self, smoothing=0.1, reduction='mean', weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer('weight', weight)

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_prob).sum(dim=1)

        if self.weight is not None:
            w = self.weight.to(dtype=loss.dtype, device=loss.device)[target]
            loss = loss * w
            if self.reduction == 'mean':
                return loss.sum() / w.sum().clamp_min(torch.finfo(loss.dtype).eps)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def _labels_of(dataset):
    """Every label of a dataset, or None when it exposes none.

    get_labels() is what dataset.py provides; ``_labels`` is the attribute
    behind it and is read as a fallback so a stale dataset.py degrades into a
    message rather than an AttributeError raised after DDP, the model and the
    optimizer have all been built.
    """
    getter = getattr(dataset, 'get_labels', None)
    if callable(getter):
        return getter()
    return getattr(dataset, '_labels', None)


def class_weights_from(dataset, num_classes, mode):
    """Inverse-frequency class weights from a labelled dataset, normalised to mean 1.

    'balanced' is sklearn's definition, n / (k * count_c). Mean-1 normalisation
    keeps the loss on the same scale as the unweighted run, so a learning rate
    tuned without weighting does not have to be retuned with it.

    A class absent from the training split gets weight 0 rather than infinity;
    it contributes no gradient either way, and 1/0 would poison the mean.
    """
    if mode == 'none':
        return None
    # train_ds is a Subset when the validation split is carved out of the
    # training file, and a Subset has no labels of its own. Weighting the whole
    # file there would count the validation windows too, so index through.
    if isinstance(dataset, torch.utils.data.Subset):
        labels = _labels_of(dataset.dataset)
        if labels is not None:
            labels = np.asarray(labels)[np.asarray(dataset.indices)]
    else:
        labels = _labels_of(dataset)
    if labels is None:
        raise SystemExit(
            "--class_weight needs labels, and this dataset exposes none.\n\n"
            "  Expected TimeSeriesDataset.get_labels(), added to dataset.py in\n"
            "  the same commit as this function. If dataset.py has it and you\n"
            "  still see this, a stale copy is being imported -- check with\n\n"
            "      python -c \'import dataset; print(dataset.__file__)\'\n\n"
            "  and clear the bytecode cache:\n\n"
            "      find . -name __pycache__ -type d -exec rm -rf {} +\n\n"
            "  --class_weight none skips this entirely."
        )
    labels = np.asarray(labels)
    if labels.dtype == object or not np.issubdtype(labels.dtype, np.integer):
        raise SystemExit(
            f"--class_weight needs integer labels, got dtype {labels.dtype}"
        )
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    with np.errstate(divide='ignore'):
        w = np.where(counts > 0, len(labels) / (num_classes * np.maximum(counts, 1)), 0.0)
    present = counts > 0
    if present.any():
        w = w / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


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
#: Every metadata dataset a file may carry. Read as a block so a file with some
#: of them is rejected rather than half-read.
#: See test_results['result_schema_version'] for what each value means.
RESULT_SCHEMA_VERSION = 2

CHANNEL_META_KEYS = (
    "channel_ids", "electrode_xyz", "positive_electrode_index",
    "negative_electrode_index", "valid_channel_mask", "electrode_names",
    "derivation_matrix", "channel_center_xyz",
)
#: Present only for montages where the field means something. A monopolar
#: recording -- 58 electrodes against a common reference, as in erpbci -- has no
#: electrode pairs, and writing a placeholder pair for each channel would be
#: inventing geometry rather than recording it. Absence is the honest value, so
#: it is not required; what IS required is that the files in one run agree about
#: whether it is there.
CHANNEL_META_OPTIONAL = ("bipolar_endpoints",)
#: The subset that becomes tensors on the device. The rest is provenance and
#: stays on the host -- the forward path takes numeric tensors only.
CHANNEL_META_TENSORS = (
    "channel_ids", "electrode_xyz", "positive_electrode_index",
    "negative_electrode_index", "valid_channel_mask",
)


def read_channel_metadata(path):
    """Global channel metadata from one HDF5, or ``None`` if it carries none.

    One copy per file rather than per window: the montage is a property of the
    recording set-up, identical for every 30 s epoch in the corpus.
    """
    with h5py.File(path, "r") as f:
        if "channel_ids" not in f:
            return None
        missing = [k for k in CHANNEL_META_KEYS if k not in f]
        if missing:
            raise KeyError(
                f"{os.path.basename(path)} has channel metadata but is missing "
                f"{missing}.\n  It was written by an older schema. Rebuild it "
                f"with the --stage split of whichever\n  preparation script "
                f"made it (EEG/sleep_edf_finetune.py or "
                f"EEG/physio_p300_finetune.py).")
        meta = {k: f[k][:] for k in CHANNEL_META_KEYS}
        meta.update({k: f[k][:] for k in CHANNEL_META_OPTIONAL if k in f})
        meta["_attrs"] = {k: f.attrs[k] for k in f.attrs}
        meta["_channel_names"] = [c.decode() for c in f["channel_names"][:]]
    return meta


def _meta_signature(meta):
    """What two files must agree on, as comparable plain values."""
    if meta is None:
        return None
    sig = {k: np.asarray(meta[k]).tobytes() for k in CHANNEL_META_KEYS}
    # Compared as present-or-absent too: one file carrying bipolar endpoints and
    # another not means two different montages, whatever else matches.
    sig.update({k: (np.asarray(meta[k]).tobytes() if k in meta else None)
                for k in CHANNEL_META_OPTIONAL})
    sig["_hash"] = meta["_attrs"].get("metadata_hash")
    sig["_schema"] = meta["_attrs"].get("metadata_schema_version")
    sig["_names"] = tuple(meta["_channel_names"])
    return sig


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
        
        # Channel metadata is per file and must be identical across them.
        # Concatenating two corpora with different montages would produce a
        # dataset with no single channel semantics, and nothing downstream
        # inspects it closely enough to notice.
        metas = [read_channel_metadata(f) for f in self.file_paths]
        first = _meta_signature(metas[0])
        for path, m in zip(self.file_paths[1:], metas[1:]):
            if _meta_signature(m) != first:
                raise ValueError(
                    f"channel metadata differs between "
                    f"{os.path.basename(self.file_paths[0])} and "
                    f"{os.path.basename(path)}.\n  These files describe "
                    f"different montages and must not be concatenated.")
        self.channel_metadata = metas[0]

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

    def get_labels(self):
        """Every label, in dataset order.

        This class is the one the trainer instantiates -- dataset.py defines a
        second TimeSeriesDataset that finetune.py does not import, and adding
        the method only there is what produced
        "'TimeSeriesDataset' object has no attribute 'get_labels'" on a cluster
        run. Both have it now; this is the one that matters here.
        """
        return self._labels


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

    # An eeg_c1 export names itself. Say so here rather than letting it reach
    # the key-by-key comparison below, which is correct but reports the
    # mismatch as a hundred missing tensor names -- the reader's question is
    # "which trainer does this checkpoint belong to", and the answer is in the
    # file. The two architectures share three tensors out of a hundred and six.
    if isinstance(checkpoint, dict) and 'route_id' in checkpoint \
            and 'channel_vocab_sha256' in checkpoint:
        raise SystemExit(
            f"{os.path.basename(pretrained_path)} is an EEG C1 export (route "
            f"{checkpoint['route_id']}), and this is the legacy\n"
            f"  BERTWaveletTransformer. They are different architectures: the "
            f"transformer is\n  `shared_transformer.*` in one and `encoder.*` "
            f"in the other, the frontend\n  `wavelet_frontend.*` against "
            f"`wavelet_decomp.*`. Three tensors of a hundred\n  and six would "
            f"load.\n\n"
            f"  Fine-tune it with the entry point that builds it:\n\n"
            f"      python -m physiowave.train.finetune_main \\\n"
            f"          --config finetune/eeg_c1_p300 --data-dir <split> "
            f"--num-classes 2 \\\n"
            f"          --output-dir <out> \\\n"
            f"          --set model.eeg_c1.pretrained={pretrained_path}\n\n"
            f"  This script trains the legacy architecture, whose pretraining "
            f"is pretrain.py.")

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

    # strict=False is necessary -- a feature extractor legitimately has no task
    # head -- but it will also swallow a genuinely wrong checkpoint without a
    # word. The channel modules are the keys a pre-channel-embedding checkpoint
    # is *expected* to lack; anything else missing means the architectures
    # disagree, and that has to be said rather than absorbed.
    core = _unwrap(model)
    allowed = set(core.channel_parameter_names()) if hasattr(
        core, 'channel_parameter_names') else set()
    allowed |= {f'module.{n}' for n in allowed}
    head_missing = [k for k in missing_keys
                    if k.split('module.', 1)[-1].startswith('task_heads.')]
    unexplained = [k for k in missing_keys
                   if k not in allowed and k not in head_missing]

    if rank == 0:
        print(f"Loaded {len(filtered_dict)} pretrained parameters")
        print(f"Skipped {len(skipped_keys)} parameters")
        n_channel = len([k for k in missing_keys if k in allowed])
        if n_channel:
            print(f"Missing {n_channel} channel-embedding parameter(s) -- expected "
                  f"for a checkpoint from before that feature; they keep their "
                  f"fresh initialisation.")
        if head_missing:
            print(f"Missing {len(head_missing)} task-head parameter(s) -- expected "
                  f"for a feature-extractor checkpoint.")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexplained:
        raise SystemExit(
            f"{len(unexplained)} parameter(s) are missing from "
            f"{os.path.basename(pretrained_path)} and are not channel-embedding "
            f"or task-head keys:\n  "
            + "\n  ".join(unexplained[:10])
            + (f"\n  ... and {len(unexplained) - 10} more" if len(unexplained) > 10 else "")
            + "\n\n  This checkpoint's architecture does not match the model being "
              "built.\n  Loading it would leave those tensors at their random "
              "initialisation and\n  the run would look like a fine-tune of "
              "something it is not.")


############################################################
# Training and Evaluation Functions
############################################################
def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def train_one_epoch(epoch, rank, model, optimizer, train_loader, device, criterion, 
                    scaler=None, grad_clip=0.0, scheduler=None, scheduler_per_batch=False,
                    fold_kl=0.0, channel_meta=None):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    core = _unwrap(model)

    def total_loss_of(logits, y):
        """Task loss plus the fold's KL-to-uniform, when one is asked for.

        The task loss alone has no objection to a fold that reads a single
        band: fewer effective inputs fit the training set just as well. This is
        the only term that prefers a fold still using its scales.
        """
        loss = criterion(logits, y)
        if fold_kl > 0.0:
            reg = core.scale_fold_reg() if hasattr(core, 'scale_fold_reg') else None
            if reg is not None:
                loss = loss + fold_kl * reg
        return loss

    loader = train_loader
    if rank == 0:
        loader = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x, task='classify', channel_meta=channel_meta)
                loss = total_loss_of(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, task='classify', channel_meta=channel_meta)
            loss = total_loss_of(logits, y)
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
        line = f"[Train] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, LR={current_lr:.6f}"
        # A dynamic fold that has collapsed onto one band still trains, still
        # reports a falling loss, and is no longer multi-scale. The weights are
        # the only place that shows, so they are printed every epoch.
        alpha = core.scale_fold_alpha() if hasattr(core, 'scale_fold_alpha') else None
        if alpha is not None:
            line += ", alpha=[" + " ".join(f"{v:.3f}" for v in alpha.tolist()) + "]"
            # alpha alone is the marginal over batch, channel and time block: a
            # fold that swings per block and one frozen at 1/S print the same
            # vector. sd_t is the spread across time blocks and is what the
            # "weights decided per block" claim actually rests on -- if it stays
            # at zero the fold is static and the mean will not say so.
            spread = (core.scale_fold_spread()
                      if hasattr(core, 'scale_fold_spread') else (None, None))
            if spread[0] is not None:
                line += (", sd_t=[" + " ".join(f"{v:.3f}" for v in spread[0].tolist()) + "]"
                         + ", sd_c=[" + " ".join(f"{v:.3f}" for v in spread[1].tolist()) + "]")
            # Per (channel, scale), not just per scale: with a channel prior in
            # the logits the interesting question is whether the two
            # derivations end up wanting different bands, and a marginal over
            # channels cannot answer it.
            per_c = (core.scale_fold_per_channel()
                     if hasattr(core, 'scale_fold_per_channel') else None)
            if per_c is not None and per_c.shape[0] > 1:
                line += " | alpha/chan=" + " ".join(
                    "[" + " ".join(f"{v:.3f}" for v in row.tolist()) + "]"
                    for row in per_c)
        # The two gates and the size of what the token branch is actually
        # adding. A gate that never leaves zero and a branch whose output is
        # zero are different failures, and only the second shows in the norm.
        gates = core.channel_gate_values() if hasattr(core, 'channel_gate_values') else (None, None)
        if gates[0] is not None or gates[1] is not None:
            line += (f" | g_f={'-' if gates[0] is None else f'{gates[0]:+.4f}'}"
                     f" g_t={'-' if gates[1] is None else f'{gates[1]:+.4f}'}")
            tok = getattr(core, 'channel_to_token', None)
            if tok is not None:
                line += f" tok_w={float(tok.weight.detach().norm()):.3f}"
        print(line)
    return avg_loss, avg_acc, current_lr


@torch.no_grad()
def eval_one_epoch(epoch, rank, model, loader, device, criterion, desc_prefix="Eval",
                   channel_meta=None, return_preds=False):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    display_loader = loader
    if rank == 0:
        display_loader = tqdm(loader, desc=f"{desc_prefix} Epoch {epoch}", ncols=120)

    for x, y in display_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, task='classify', channel_meta=channel_meta)
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
    
    # sklearn routes on the shape of y_true, not on y_prob: with two classes
    # present it calls the binary path, which rejects a 2-column y_score with
    # "y should be a 1d array". Passing multi_class='ovo' does not change that.
    # Left to the except below it becomes a silent NaN, and --select_by auroc
    # then never improves on its initial value, so no checkpoint is ever saved.
    try:
        n_seen = len(np.unique(y_true))
        if n_seen < 2:
            # AUROC is undefined when the split holds a single class. This is
            # real on a per-subject test fold, so it is not an error.
            auroc = float('nan')
        elif y_prob.shape[1] == 2:
            auroc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
    except Exception:
        auroc = float('nan')

    if rank == 0:
        print(f"[{desc_prefix}] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
              f"BalAcc={balanced_acc:.4f}, Kappa={kappa:.4f}, WF1={weighted_f1:.4f}, AUROC={auroc:.4f}")

    if return_preds:
        # Only the test call asks for these. Returning them unconditionally
        # would change the tuple every caller unpacks, for the benefit of one.
        return (avg_loss, avg_acc, balanced_acc, kappa, weighted_f1, auroc,
                y_true, y_pred)
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

#: Which converter wrote a split, so a refusal can name the one to re-run.
#: This file is shared by every downstream task, and both messages below used
#: to hardcode EEG/sleep_edf_finetune.py -- so a P300 run that hit them was
#: sent to rebuild a sleep dataset it does not have. Named from evidence, and
#: where there is none it says so rather than guessing confidently.
_CONVERTERS = (
    ("p300", "EEG/physio_p300_finetune.py --stage split --fold <k> \\\n"
             "          --edf-dir <erpbci> --out-dir <new dir>"),
    ("erpbci", "EEG/physio_p300_finetune.py --stage split --fold <k> \\\n"
               "          --edf-dir <erpbci> --out-dir <new dir>"),
    ("sleep", "EEG/sleep_edf_finetune.py --stage split \\\n"
              "          --cache-dir <existing cache> --out-dir <new dir>"),
    ("tuab", "EEG/tuab_finetune.py --stage split --out-dir <new dir>"),
    ("db5", "EMG/db5_finetune.py --out-dir <new dir>"),
    ("db6", "EMG/db6_finetune.py --out-dir <new dir>"),
)


def rebuild_command(paths) -> str:
    """The re-run line for whichever converter wrote these files."""
    hay = " ".join(str(p).lower() for p in paths if p)
    for key, cmd in _CONVERTERS:
        if key in hay:
            return "      python " + cmd
    return ("      the converter under EEG/ or EMG/ that wrote this dataset,\n"
            "      with --stage split")


    # ------------------------------------------------------------------ #
    # Channel metadata: resolved once here, not per sample.
    # ------------------------------------------------------------------ #
    def _base(ds):
        return ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds

    split_meta = {'train': _base(train_ds).channel_metadata,
                  'val': _base(val_ds).channel_metadata}
    if test_ds is not None:
        split_meta['test'] = test_ds.channel_metadata
    sigs = {k: _meta_signature(v) for k, v in split_meta.items()}
    disagree = [k for k, v in sigs.items() if v != sigs['train']]
    if disagree:
        raise SystemExit(
            f"channel metadata differs between train and {', '.join(disagree)}.\n"
            f"  The splits describe different montages. Rebuild all three from "
            f"ONE run of:\n\n"
            f"{rebuild_command([args.train_file, args.val_file, args.test_file])}\n")
    channel_metadata = split_meta['train']

    if args.channel_encoding != 'none':
        if channel_metadata is None:
            raise SystemExit(
                f"--channel_encoding {args.channel_encoding} needs channel "
                f"metadata and these HDF5 files carry none.\n"
                f"  ({args.train_file})\n\n"
                f"  A split written with --no-channel-metadata carries the "
                f"array and the channel\n  names but not the electrode "
                f"coordinates and ids this encoding reads. Rebuild\n  it -- the "
                f"per-subject .npz cache is reused, so this is a split and not "
                f"a decode:\n\n"
                f"{rebuild_command([args.train_file, args.val_file, args.test_file])}\n\n"
                f"  Or run with --channel_encoding none, which trains the same "
                f"architecture\n  without the channel-identity path.")
        n_meta = len(channel_metadata['channel_ids'])
        if n_meta != args.in_channels:
            raise SystemExit(
                f"--in_channels {args.in_channels} but the metadata describes "
                f"{n_meta} channels ({channel_metadata['_channel_names']}).")

    # Numeric tensors only, one copy on this rank's device. The strings and the
    # provenance stay on the host: the forward path never parses a name.
    channel_meta = None
    if args.channel_encoding != 'none':
        channel_meta = {
            k: torch.as_tensor(channel_metadata[k]).to(device)
            for k in CHANNEL_META_TENSORS
        }
        channel_meta['channel_ids'] = channel_meta['channel_ids'].long()
        channel_meta['positive_electrode_index'] = channel_meta['positive_electrode_index'].long()
        channel_meta['negative_electrode_index'] = channel_meta['negative_electrode_index'].long()
        channel_meta['electrode_xyz'] = channel_meta['electrode_xyz'].float()
        if rank == 0:
            a = channel_metadata['_attrs']
            print(f"Channel metadata: {channel_metadata['_channel_names']} "
                  f"schema v{a.get('metadata_schema_version')} "
                  f"hash {a.get('metadata_hash')} ({a.get('coordinate_source')})")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # ----------------------------------------------------------------------- #
    # Validation and test are deliberately NOT sharded.
    #
    # They used to be, with a DistributedSampler each -- and nothing in this
    # file gathers the shards back. eval_one_epoch computes balanced accuracy,
    # kappa, weighted F1 and AUROC from whatever that rank happened to see, and
    # only rank 0 prints or saves. So every val and test number this script has
    # ever reported was computed on 1/world_size of the set, and so was the
    # value model selection compared against. On four GPUs and a 12417-window
    # test set that is 3105 windows; the confusion matrix summing to 3105
    # instead of 12417 is the same fact seen from the other side.
    #
    # The fix is replication rather than an all_gather. Every rank evaluates the
    # whole set and arrives at the same number, so there is nothing to collect
    # and nothing to correct for: DistributedSampler pads the final shard by
    # repeating windows from the front, and a naive gather would score those
    # twice. The cost is that ranks 1..N-1 redo the validation pass, which on
    # this task is seconds per epoch against a number that is otherwise wrong.
    # ----------------------------------------------------------------------- #
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True) if test_ds else None

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
        wave_init_mode=args.wave_init_mode,
        patch_size=(1, args.patch_size),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        norm=args.norm,
        ffn=args.ffn,
        qk_norm=args.qk_norm,
        channel_encoding=args.channel_encoding,
        channel_injection=args.channel_injection,
        channel_embed_dim=args.channel_embed_dim,
        channel_fold_gate_init=args.channel_fold_gate_init,
        channel_token_gate_init=args.channel_token_gate_init,
        channel_vocab_size=(None if channel_metadata is None
                            else int(channel_metadata['_attrs'].get(
                                'channel_vocab_size', 0)) or None),
        scale_fold=args.scale_fold,
        fold_patch_len=args.fold_patch_len,
        fold_synthesis=args.fold_synthesis,
        fold_synthesis_norm=args.fold_synthesis_norm,
        fold_share_channels=args.fold_share_channels,
        fold_shrinkage=args.fold_shrinkage,
        fold_scale_dropout=args.fold_scale_dropout,
        fold_gamma=args.fold_gamma,
        use_pos_embed=args.use_pos_embed,
        pos_embed_type=args.pos_embed_type,
        task_type='classification',
        num_classes=args.num_classes,
        head_config=head_config,
        pooling=args.pooling
    ).to(device)
    
    def _git_commit():
        """The commit the run was launched from, or None. Never fails the run."""
        try:
            import subprocess
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:                                      # noqa: BLE001
            return None

    # Everything needed to say what this run *was*, written into the checkpoint
    # and the result file. A directory name is not provenance: an ablation is a
    # set of runs that differ in a few flags, and reading those flags back off
    # the artefact is the only way to be sure which row a number belongs to.
    provenance = {
        'channel_encoding': args.channel_encoding,
        'channel_injection': args.channel_injection,
        'channel_embed_dim': args.channel_embed_dim,
        'channel_fold_gate_init': args.channel_fold_gate_init,
        'channel_token_gate_init': args.channel_token_gate_init,
        'channel_vocab': (None if channel_metadata is None
                          else int(channel_metadata['_attrs'].get('channel_vocab_size', 0))),
        'metadata_hash': (None if channel_metadata is None
                          else str(channel_metadata['_attrs'].get('metadata_hash'))),
        'metadata_schema_version': (None if channel_metadata is None
                                    else int(channel_metadata['_attrs'].get(
                                        'metadata_schema_version', 0))),
        'channel_names': (None if channel_metadata is None
                          else list(channel_metadata['_channel_names'])),
        'git_commit': _git_commit(),
        'seed': args.seed,
        'resolved_model_config': {
            k: getattr(args, k) for k in (
                'in_channels', 'max_level', 'wave_kernel_size', 'wave_init_mode',
                'patch_size', 'embed_dim', 'depth', 'num_heads', 'mlp_ratio',
                'dropout', 'norm', 'ffn', 'qk_norm', 'scale_fold',
                'fold_synthesis', 'fold_gamma', 'fold_kl', 'pos_embed_type',
                'pooling', 'head_hidden_dim', 'head_dropout', 'num_classes',
                'label_smoothing', 'class_weight', 'lr', 'weight_decay',
                'batch_size', 'epochs', 'warmup_epochs', 'scheduler', 'select_by')
        },
    }

    if rank == 0:
        print("Run provenance: "
              + " ".join(f"{k}={provenance[k]}" for k in
                         ('channel_encoding', 'channel_injection',
                          'channel_embed_dim', 'metadata_hash', 'seed'))
              + f" git={str(provenance['git_commit'])[:8]}", flush=True)
        # State the block that is actually running: these are CLI defaults rather
        # than anything the output directory records, and an ablation that
        # silently kept the defaults would look identical to one that took effect.
        print(f"Transformer block: norm={args.norm} ffn={args.ffn} "
              f"qk_norm={args.qk_norm} rope=True | scale_fold={args.scale_fold} "
              f"| wave_init={args.wave_init_mode}", flush=True)
        if args.scale_fold == 'dynamic':
            blk = args.patch_size if args.fold_patch_len is None else args.fold_patch_len
            print(f"Dynamic fold: block={blk or 'window'} synthesis={args.fold_synthesis} "
                  f"shrinkage={args.fold_shrinkage} scale_dropout={args.fold_scale_dropout} "
                  f"gamma={args.fold_gamma} kl={args.fold_kl}", flush=True)

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
    
    # Optimizer. The fold's mixing parameters are held out of weight decay.
    # They are a convex combination and a delta-initialised synthesis filter,
    # not capacity: decaying them towards zero shrinks the folded signal
    # itself rather than penalising overfitting, which is the mechanism that
    # would make a fold "quietly get smaller" over a long run. Its MLP is an
    # ordinary MLP and keeps the decay.
    channel_gate_names = {'channel_fold_gate', 'channel_token_gate'}
    fold_no_decay, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Names are DDP-prefixed here ("module.fold.x"), but the rule must not
        # depend on that -- an unwrapped model would silently skip the group.
        stem = name.split('module.', 1)[-1]
        if stem in channel_gate_names:
            # The two channel gates are scalars whose whole job is to sit at
            # zero until the task asks otherwise. Decaying them pulls them back
            # towards zero every step, which is a prior against the branch
            # rather than a penalty on its capacity -- and the branch is
            # precisely what the ablation is measuring. The encoder and the
            # projections beside them keep the normal decay: those are capacity.
            fold_no_decay.append(p)
        elif stem.startswith('fold.') and not stem.startswith('fold.mlp.'):
            fold_no_decay.append(p)
        else:
            rest.append(p)
    optimizer = optim.AdamW(
        [{'params': rest, 'weight_decay': args.weight_decay},
         {'params': fold_no_decay, 'weight_decay': 0.0}],
        lr=args.lr,
    )
    if rank == 0 and fold_no_decay:
        print(f"Optimizer: {sum(p.numel() for p in fold_no_decay)} fold/gate parameters "
              f"held out of weight decay", flush=True)
    
    # Loss function
    cls_weight = class_weights_from(train_ds, args.num_classes, args.class_weight)
    if cls_weight is not None:
        cls_weight = cls_weight.to(device)
        if rank == 0:
            print(f"Class weights ({args.class_weight}): "
                  + " ".join(f"{w:.3f}" for w in cls_weight.tolist()))
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing,
                                               weight=cls_weight)
        if rank == 0:
            print(f"Using label smoothing: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_weight)
    criterion = criterion.to(device)
    
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

    best_selection = float('-inf')     # see --select-by below
    epochs_without_improvement = 0
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
        
        train_loss, train_acc, current_lr = train_one_epoch(
            epoch, rank, model, optimizer, train_loader, device, 
            criterion, scaler, args.grad_clip, scheduler, scheduler_per_batch,
            fold_kl=args.fold_kl, channel_meta=channel_meta
        )
        
        val_metrics = eval_one_epoch(
            epoch, rank, model, val_loader, device, criterion, desc_prefix="Val",
            channel_meta=channel_meta
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
        
        improved = False
        if rank == 0:
            # Which validation number decides the checkpoint. Loss is the
            # historical default but disagrees with accuracy once the model
            # starts overfitting: it keeps rising from the epoch the model
            # turns overconfident, while accuracy carries on improving for
            # tens of epochs. On an imbalanced label set balanced accuracy is
            # the honest target, since plain accuracy rewards leaning on the
            # frequent classes.
            _selection = {
                'loss': -val_loss,
                'acc': val_acc,
                'balanced_acc': val_balanced_acc,
                'kappa': val_kappa,
                'weighted_f1': val_weighted_f1,
                'auroc': val_auroc,
            }[args.select_by]
            # NaN compares False against everything, so a metric that is
            # undefined for one epoch would silently stop the run from ever
            # checkpointing again rather than just skipping that epoch. Say so
            # instead: on a per-subject fold AUROC really can be undefined.
            if _selection != _selection:                       # NaN
                print(f"Epoch {epoch}: val {args.select_by} is undefined (nan); "
                      f"not a selection candidate.")
                improved = False
            else:
                improved = _selection > best_selection + args.min_delta
            if improved:
                best_selection = _selection
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
                    'provenance': provenance,
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
                'provenance': provenance,
            }, os.path.join(args.output_dir, "latest_model.pth"))

        # Early stopping.
        #
        # Rank 0 decides and the answer is broadcast, because the ranks must
        # leave the loop together: one rank breaking while the others continue
        # leaves them blocked forever in the next gradient all-reduce, and the
        # job sits burning its allocation until the wall clock kills it.
        #
        # Note the interaction with the cosine schedule -- stopping early means
        # the learning rate never finishes decaying, so the final weights are
        # from a point the schedule considered mid-flight. best_model.pth is
        # the epoch that actually scored best, so the reported numbers are
        # unaffected; it is `latest_model.pth` that is left mid-anneal.
        if args.patience > 0:
            stop_signal = torch.zeros(1, device=device)
            if rank == 0:
                epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stop at epoch {epoch}: val {args.select_by} has not "
                          f"improved by more than {args.min_delta} in {args.patience} "
                          f"epochs (best was epoch {best_epoch}).")
                    stop_signal[0] = 1.0
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() > 0:
                break

    if rank == 0:
        print(f"\n(model selected on val {args.select_by})")
        print(f"Best Validation: Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}, "
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
            "Test", rank, model, test_loader, device, criterion, desc_prefix="Test",
            channel_meta=channel_meta, return_preds=True
        )
        
        if rank == 0:
            (test_loss, test_acc, test_balanced_acc, test_kappa,
             test_weighted_f1, test_auroc, y_true_test, y_pred_test) = test_metrics
            test_results = {
                # Bumped when a result stops being comparable to the ones
                # before it for a reason that is not visible in `provenance`.
                #   1  the original
                #   2  val and test are evaluated on the WHOLE set. Before this,
                #      a DistributedSampler gave each rank a shard and nothing
                #      gathered them, so every metric -- including the one model
                #      selection used -- came from rank 0's 1/world_size.
                # A sweep runner refuses to skip a run below its minimum, which
                # is what a config comparison alone cannot catch: the stale runs
                # had identical hyper-parameters and a different code path.
                'result_schema_version': RESULT_SCHEMA_VERSION,
                'test_samples': int(len(y_true_test)),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_balanced_acc': test_balanced_acc,
                'test_kappa': test_kappa,
                'test_weighted_f1': test_weighted_f1,
                'test_auroc': test_auroc,
                'provenance': provenance,
                'best_epoch': best_epoch,
                'best_val': {
                    'loss': best_val_loss, 'acc': best_val_acc,
                    'balanced_acc': best_balanced_acc, 'kappa': best_kappa,
                    'weighted_f1': best_weighted_f1, 'auroc': best_auroc,
                },
            }
            # Per-class F1 and the confusion matrix, so the collector does not
            # have to reload a checkpoint to report them. Balanced accuracy over
            # five sleep stages hides which stage moved, and N1 is both the
            # rarest and the one every method loses on.
            try:
                cm = confusion_matrix(y_true_test, y_pred_test,
                                      labels=list(range(args.num_classes)))
                test_results['confusion_matrix'] = cm.tolist()
                test_results['per_class_f1'] = f1_score(
                    y_true_test, y_pred_test, average=None,
                    labels=list(range(args.num_classes)), zero_division=0).tolist()
                test_results['per_class_support'] = cm.sum(axis=1).tolist()
            except Exception as exc:                           # noqa: BLE001
                test_results['per_class_error'] = repr(exc)

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
    parser.add_argument('--channel_encoding', type=str, default='none',
                        choices=['none', 'id', 'signed', 'hybrid'],
                        help="How a channel's identity is encoded. 'id' is the "
                             "EEGPT-style learned name embedding; 'signed' is the "
                             "derivation's geometry, keeping midpoint and "
                             "direction apart so A-B and B-A differ; 'hybrid' is "
                             "both. Default 'none' -- the model is then the one "
                             "that existed before this feature.")
    parser.add_argument('--channel_injection', type=str, default='none',
                        choices=['none', 'token', 'fold', 'dual'],
                        help="Where the code enters. 'token' adds it to each "
                             "patch token of its own channel; 'fold' biases the "
                             "dynamic fold's scale logits and needs "
                             "--scale_fold dynamic; 'dual' does both. Never "
                             "added to the waveform, which would put a DC "
                             "offset on the signal.")
    parser.add_argument('--channel_embed_dim', type=int, default=64,
                        help='width of the channel code before it is projected')
    parser.add_argument('--channel_fold_gate_init', type=float, default=0.0,
                        help='initial value of the fold gate; 0 leaves the '
                             'backbone bit-identical at step 0')
    parser.add_argument('--channel_token_gate_init', type=float, default=0.0,
                        help='initial value of the token gate; 0 leaves the '
                             'backbone bit-identical at step 0')
    parser.add_argument('--class_weight', type=str, default='none',
                        choices=['none', 'balanced'],
                        help="Weight the loss by inverse class frequency, measured on "
                             "the training split. 'balanced' is needed on a task whose "
                             "minority class is small enough that argmax at 0.5 would "
                             "otherwise collapse onto the majority -- P300 is 1:5 -- "
                             "and is a no-op change for a balanced task. Default 'none' "
                             "so existing runs are unaffected.")
    
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
    # Transformer block variants. The defaults reproduce the original block
    # exactly, so turning one on is an ablation row rather than a new model.
    parser.add_argument('--norm', type=str, default='layernorm',
                        choices=['layernorm', 'rmsnorm'],
                        help='block normaliser; rmsnorm drops the mean subtraction and the bias')
    parser.add_argument('--ffn', type=str, default='mlp', choices=['mlp', 'swiglu'],
                        help="feed-forward; swiglu's width is scaled by 2/3 so the "
                             'parameter count stays level with the GELU MLP')
    parser.add_argument('--qk_norm', action='store_true',
                        help='RMS-normalise q and k per head, bounding the attention logits')
    parser.add_argument('--wave_init_mode', type=str, default='interp',
                        choices=['interp', 'pad'],
                        help="how a wavelet shorter than --wave_kernel_size is fitted to "
                             "it. 'interp' stretches the taps, which is what the original "
                             "did and which does not give a wavelet filter bank: sym4 at "
                             "16 taps cuts at 0.203*pi instead of 0.5*pi and fails the "
                             "power-complementary condition by 1.999 out of 2. 'pad' "
                             "centres the native taps and zero-pads, which preserves the "
                             "response exactly but needs every wavelet to fit.")
    parser.add_argument('--scale_fold', type=str, default='none',
                        choices=['none', 'mean', 'learned', 'softmax', 'dynamic'],
                        help="fold Spec(X)'s scale axis before patching, so the backbone "
                             "sees C*S tokens instead of (J+1)*C*S. The modes form a "
                             "ladder -- mean (plain reconstruction), learned (free per "
                             "channel weights), softmax (the same, made convex), dynamic "
                             "(weights predicted per channel and time block from the "
                             "bands' own statistics). Nothing else about the model "
                             "changes. See wavelet_modules.ScaleFold.")
    parser.add_argument('--fold_patch_len', type=int, default=None,
                        help='dynamic fold only: samples per weighting block. Defaults to '
                             '--patch_size so one weight backs one token; 0 decides a '
                             'single weight for the whole window.')
    parser.add_argument('--fold_synthesis', type=int, default=0,
                        help='any folding mode: odd kernel size for a per-scale synthesis '
                             'filter, shared across channels and initialised to a delta. '
                             '0 disables it. The bands reach the fold after soft gating '
                             'and nearest upsampling, so they are not phase-aligned; a '
                             'scalar weight cannot correct that and a 3-tap filter can.')
    parser.add_argument('--fold_synthesis_norm', action='store_true',
                        help='constrain each synthesis kernel to unit DC gain, so it can '
                             'reshape a band but not rescale it. Trained unconstrained, '
                             'the kernels move mostly in gain (1.40x on the finest detail '
                             'band against 1.13x on the approximation), so this is what '
                             'separates the two effects.')
    parser.add_argument('--fold_share_channels', action='store_true',
                        help="learned/softmax folds only: one weight per scale instead of "
                             "per (scale, channel). 4 parameters rather than 4*C, and "
                             "channel-count independent like the dynamic fold.")
    parser.add_argument('--fold_shrinkage', action='store_true',
                        help='any folding mode: soft-threshold each band before folding, '
                             'with a learned threshold against a MAD noise estimate. '
                             'Starts at ~0.0025 sigma, i.e. off.')
    parser.add_argument('--fold_scale_dropout', type=float, default=0.0,
                        help='dynamic fold only: probability of dropping a scale from the '
                             'mixture during training, the survivors renormalised.')
    parser.add_argument('--fold_gamma', type=float, default=0.1,
                        help='dynamic fold only: initial gate on the deviation from the '
                             'plain mean. At 0 the fold is exactly --scale_fold mean.')
    parser.add_argument('--fold_kl', type=float, default=0.0,
                        help='dynamic fold only: weight on KL(alpha || uniform), added to '
                             'the task loss to keep the mixture from collapsing onto a '
                             'single band. 1e-4 to 1e-3 is the useful range.')
    
    # Model parameters - Classification Head
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--patience', type=int, default=0,
                        help='Stop when the --select_by metric has not improved for this '
                             'many epochs. 0 disables early stopping (the default, so '
                             'existing commands are unaffected).')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Improvement below this does not reset the patience counter '
                             'and does not update best_model.pth.')
    parser.add_argument('--select_by', type=str, default='loss',
                        choices=['loss', 'acc', 'balanced_acc', 'kappa',
                                 'weighted_f1', 'auroc'],
                        help='Validation metric that decides which epoch is kept as '
                             'best_model.pth. Default loss, for backwards '
                             'compatibility; balanced_acc is the right choice on an '
                             'imbalanced label set.')
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