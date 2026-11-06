"""
The sEMG C1 trainer: the EEG C1 training loop on the sEMG registry and model.
"""

from __future__ import annotations

from typing import Dict

from ..eeg_c1.train import EEGC1Trainer
from .electrodes import emg_vocab_payload, save_emg_vocab
from .model import MultiRouteEMGPretrainer
from .routes import DEFAULT_BATCH_BY_ROUTE, PRETRAIN_DATASETS, ROUTES


class EMGC1Trainer(EEGC1Trainer):
    routes_table = ROUTES
    datasets_table = PRETRAIN_DATASETS
    default_batch_by_route = DEFAULT_BATCH_BY_ROUTE
    banner_title = "sEMG C1 multi-route pretraining"

    def build_model(self, mcfg: Dict):
        return MultiRouteEMGPretrainer(
            embed_dim=int(mcfg.get("embed_dim", 384)),
            depth=int(mcfg.get("depth", 6)),
            num_heads=int(mcfg.get("num_heads", 6)),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            dropout=float(mcfg.get("dropout", 0.0)),
            norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
            qk_norm=bool(mcfg.get("qk_norm", True)),
            max_level=int(mcfg.get("max_level", 3)),
            wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
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
            channel_encoding=mcfg.get("channel_encoding", "id"),
            channel_injection=mcfg.get("channel_injection", "token"),
            channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
            channel_token_gate_init=float(mcfg.get("channel_token_gate_init", 0.0)),
        )

    def vocab_payload(self) -> Dict:
        return emg_vocab_payload()

    def save_vocab(self, path: str):
        save_emg_vocab(path)
