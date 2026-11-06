"""
The ECG C1 trainer: the EEG C1 training loop, on the ECG registry and model.

Nothing about training differs between the modalities -- the route-pure
DDP-safe schedule, the fixed validation masks, the per-route and per-dataset
metrics, the four checkpoint-selection bars, exact resume and ``--init-from``
are all ``physiowave.eeg_c1.train.EEGC1Trainer``. What this class names is the
part that is ECG: the routes and datasets the schedule draws from, the model
it builds, and the lead vocabulary its checkpoints record.
"""

from __future__ import annotations

from typing import Dict

from ..eeg_c1.train import EEGC1Trainer
from .leads import ecg_vocab_payload, save_ecg_vocab
from .model import MultiRouteECGPretrainer
from .routes import DEFAULT_BATCH_BY_ROUTE, PRETRAIN_DATASETS, ROUTES


class ECGC1Trainer(EEGC1Trainer):
    routes_table = ROUTES
    datasets_table = PRETRAIN_DATASETS
    default_batch_by_route = DEFAULT_BATCH_BY_ROUTE
    banner_title = "ECG C1 multi-route pretraining"

    def build_model(self, mcfg: Dict):
        return MultiRouteECGPretrainer(
            embed_dim=int(mcfg.get("embed_dim", 512)),
            depth=int(mcfg.get("depth", 9)),
            num_heads=int(mcfg.get("num_heads", 8)),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            dropout=float(mcfg.get("dropout", 0.0)),
            norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
            qk_norm=bool(mcfg.get("qk_norm", True)),
            max_level=int(mcfg.get("max_level", 3)),
            wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
            # None -> the ECG families (physiowave.ecg_c1.model.ECG_WAVELETS),
            # not the EEG ones the parent would fall back to.
            wavelet_names=mcfg.get("wavelet_names"),
            wave_init_mode=mcfg.get("wave_init_mode", "pad"),
            use_separate_channel=bool(mcfg.get("use_separate_channel", True)),
            mask_before_frontend=self.mask_before_frontend,
            normalize_spec_target=self.normalize_spec_target,
            fold_synthesis=int(mcfg.get("fold_synthesis", 3)),
            fold_gamma=float(mcfg.get("fold_gamma", 0.1)),
            masking_strategy=mcfg.get("masking_strategy", "frequency_guided"),
            importance_ratio=float(mcfg.get("importance_ratio", 0.6)),
            mask_ratio=self.mask_ratio,
            lead_group_masking=bool(mcfg.get("lead_group_masking", True)),
            channel_encoding=mcfg.get("channel_encoding", "id"),
            channel_injection=mcfg.get("channel_injection", "token"),
            channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
            channel_token_gate_init=float(mcfg.get("channel_token_gate_init", 0.0)),
        )

    def vocab_payload(self) -> Dict:
        return ecg_vocab_payload()

    def save_vocab(self, path: str):
        save_ecg_vocab(path)
