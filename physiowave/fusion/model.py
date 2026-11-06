"""Summary tokens, RALF, the two baselines, and the container around the encoders.

The three fusion cores share one interface -- ``core(features, task, mask)``
-> logits and friends -- so the training loop and the test matrix are the
same code for all of them:

    ralf            8 fusion tokens cross-attend, in two layers, over the
                    modalities' summary tokens; a per-item reliability gates
                    each modality and weights its unimodal logit, which is
                    added as a residual. Absent modalities are masked out of
                    the attention.
    concat          mean-pool each modality, concatenate, classify. Absence
                    is a zero vector -- the failure RALF exists to avoid.
    weighted_logit  the earlier scheme: a logit per modality and ONE learned
                    weight per modality, renormalised over what is present.
                    No reliability, nothing per item, no cross-modal term.

Every core carries one head per TASK, so MPDB (5 classes) and SeizeIT2 (2)
train the same encoders and the same fusion with their own outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prep_common import MODALITIES


class SummaryPool(nn.Module):
    """``[B, L, D] -> [B, n, D]``: n learned queries attend over an encoder's tokens."""

    def __init__(self, dim: int, n_queries: int = 4, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(tokens)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm_out(q + a)


class _Core(nn.Module):
    def __init__(self, modalities: Sequence[str], encoder_dims: Dict[str, int],
                 tasks: Dict[str, int], embed_dim: int, dropout: float):
        super().__init__()
        self.modalities = list(modalities)
        self.tasks = dict(tasks)
        self.embed_dim = int(embed_dim)
        self.dropout = float(dropout)
        self.proj = nn.ModuleDict({m: nn.Linear(encoder_dims[m], embed_dim)
                                   for m in self.modalities})
        self.modality_token = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) for m in self.modalities})

    def _present(self, features, modality_mask, B, device) -> Dict[str, torch.Tensor]:
        out = {}
        for m in self.modalities:
            if m not in features or features[m] is None:
                out[m] = torch.zeros(B, device=device)
            elif modality_mask is not None and modality_mask.get(m) is not None:
                out[m] = modality_mask[m].to(device=device, dtype=torch.float32).view(B)
            else:
                out[m] = torch.ones(B, device=device)
        total = torch.stack([out[m] for m in self.modalities]).sum(0)
        if bool((total <= 0).any()):
            idx = torch.nonzero(total <= 0).flatten().tolist()
            raise ValueError(f"no available modality for batch items {idx}; at least one "
                             "modality must be present per sample")
        return out

    def _task_heads(self, out_dim_fn):
        return nn.ModuleDict({t: out_dim_fn(k) for t, k in self.tasks.items()})

    def reliability_loss(self, reliability, target, present) -> torch.Tensor:
        return next(self.parameters()).new_zeros(())

    @staticmethod
    def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        """Symmetric KL between two predictive distributions (batch mean)."""
        pa, pb = F.log_softmax(logits_a.float(), -1), F.log_softmax(logits_b.float(), -1)
        return 0.5 * (F.kl_div(pb, pa, log_target=True, reduction="batchmean")
                      + F.kl_div(pa, pb, log_target=True, reduction="batchmean"))


class RALFCore(_Core):
    """``gate_floor`` keeps a modality's tokens at least that fraction of their
    size however low its reliability (0 = the tokens vanish with it);
    ``gate_detach`` stops the task loss from reaching the reliability head
    through the gate, so the head learns reliability from its own target only.
    Both default to the original behaviour."""

    def __init__(self, modalities, encoder_dims, tasks, embed_dim=256, num_fusion_tokens=8,
                 num_heads=8, depth=2, dropout=0.1, residual_weight=0.5,
                 gate_floor=0.0, gate_detach=False):
        super().__init__(modalities, encoder_dims, tasks, embed_dim, dropout)
        D = self.embed_dim
        self.residual_weight = float(residual_weight)
        self.gate_floor = float(gate_floor)
        self.gate_detach = bool(gate_detach)
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError(f"gate_floor must be in [0, 1), got {gate_floor}")
        self.reliability_head = nn.ModuleDict({
            m: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D // 2), nn.GELU(),
                             nn.Linear(D // 2, 1)) for m in self.modalities})
        self.unimodal_head = nn.ModuleDict({
            m: self._task_heads(lambda k: nn.Sequential(nn.LayerNorm(D), nn.Linear(D, k)))
            for m in self.modalities})
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, D) * 0.02)
        self.cross = nn.ModuleList(
            nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
            for _ in range(depth))
        self.cross_norm = nn.ModuleList(nn.LayerNorm(D) for _ in range(depth))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(4 * D, D)) for _ in range(depth))
        self.out_norm = nn.LayerNorm(D)
        self.fusion_head = self._task_heads(lambda k: nn.Linear(D, k))

    def forward(self, features: Dict[str, torch.Tensor], task: str,
                modality_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
        any_feat = next(v for v in features.values() if v is not None)
        B, device = any_feat.shape[0], any_feat.device
        present = self._present(features, modality_mask, B, device)
        seq, key_mask, reliability, uni = [], [], {}, {}
        for m in self.modalities:
            f = features.get(m)
            if f is None:
                continue
            h = self.proj[m](f) + self.modality_token[m]            # [B, n, D]
            pooled = h.mean(dim=1)
            r = torch.sigmoid(self.reliability_head[m](pooled).float()).squeeze(-1) * present[m]
            reliability[m] = r
            uni[m] = self.unimodal_head[m][task](pooled)
            g = self.gate_floor + (1.0 - self.gate_floor) * r
            if self.gate_detach:
                g = g.detach()
            seq.append(h * g.view(B, 1, 1).to(h.dtype))
            key_mask.append(present[m].view(B, 1).expand(B, h.shape[1]) > 0.5)
        mem = torch.cat(seq, dim=1)
        keep = torch.cat(key_mask, dim=1)
        x = self.fusion_tokens.unsqueeze(0).expand(B, -1, -1).to(mem.dtype)
        for attn, norm, ffn in zip(self.cross, self.cross_norm, self.ffn):
            a, _ = attn(norm(x), mem, mem, key_padding_mask=~keep, need_weights=False)
            x = x + a
            x = x + ffn(x)
        fused = self.out_norm(x).mean(dim=1)
        fusion_logits = self.fusion_head[task](fused)
        residual = torch.zeros_like(fusion_logits)
        denom = torch.zeros(B, device=device)
        for m, lg in uni.items():
            w = reliability[m] * present[m]
            residual = residual + w.view(B, 1).to(lg.dtype) * lg
            denom = denom + w
        logits = fusion_logits + self.residual_weight * residual / denom.clamp_min(1e-6).view(B, 1).to(residual.dtype)
        return {"logits": logits, "fusion_logits": fusion_logits, "unimodal_logits": uni,
                "reliability": reliability, "present": present, "fused": fused}

    def reliability_loss(self, reliability, target, present) -> torch.Tensor:
        """MSE between the predicted reliability and ``1 - corruption level``."""
        losses = []
        for m, r in reliability.items():
            if m not in target or target[m] is None:
                continue
            w = present[m]
            if w.sum() <= 0:
                continue
            losses.append((((r - target[m].to(r.dtype)) ** 2) * w).sum() / w.sum())
        if not losses:
            return next(iter(reliability.values())).new_zeros(())
        return torch.stack(losses).mean()


class ConcatCore(_Core):
    def __init__(self, modalities, encoder_dims, tasks, embed_dim=256, dropout=0.1, **_):
        super().__init__(modalities, encoder_dims, tasks, embed_dim, dropout)
        D = self.embed_dim
        n = len(self.modalities)
        self.head = self._task_heads(lambda k: nn.Sequential(
            nn.LayerNorm(D * n), nn.Linear(D * n, D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D, k)))

    def forward(self, features, task, modality_mask=None):
        any_feat = next(v for v in features.values() if v is not None)
        B, device = any_feat.shape[0], any_feat.device
        present = self._present(features, modality_mask, B, device)
        parts = []
        for m in self.modalities:
            f = features.get(m)
            if f is None:
                parts.append(torch.zeros(B, self.embed_dim, device=device, dtype=any_feat.dtype))
                continue
            parts.append(self.proj[m](f).mean(dim=1) * present[m].view(B, 1).to(f.dtype))
        logits = self.head[task](torch.cat(parts, dim=-1))
        return {"logits": logits, "fusion_logits": logits, "unimodal_logits": {},
                "reliability": {m: present[m] for m in present}, "present": present,
                "fused": torch.cat(parts, dim=-1)}


class WeightedLogitCore(_Core):
    """The earlier fusion: frozen-encoder logits, one global weight per modality."""

    def __init__(self, modalities, encoder_dims, tasks, embed_dim=256, dropout=0.1, **_):
        super().__init__(modalities, encoder_dims, tasks, embed_dim, dropout)
        D = self.embed_dim
        self.unimodal_head = nn.ModuleDict({
            m: self._task_heads(lambda k: nn.Sequential(nn.LayerNorm(D), nn.Dropout(dropout),
                                                        nn.Linear(D, k)))
            for m in self.modalities})
        self.logit_weight = nn.Parameter(torch.zeros(len(self.modalities)))

    def forward(self, features, task, modality_mask=None):
        any_feat = next(v for v in features.values() if v is not None)
        B, device = any_feat.shape[0], any_feat.device
        present = self._present(features, modality_mask, B, device)
        w_all = torch.softmax(self.logit_weight.float(), 0)
        uni, weights = {}, {}
        for i, m in enumerate(self.modalities):
            f = features.get(m)
            if f is None:
                continue
            uni[m] = self.unimodal_head[m][task](self.proj[m](f).mean(dim=1))
            weights[m] = w_all[i] * present[m]
        denom = torch.stack(list(weights.values())).sum(0).clamp_min(1e-6)
        logits = None
        for m, lg in uni.items():
            term = (weights[m] / denom).view(B, 1).to(lg.dtype) * lg
            logits = term if logits is None else logits + term
        return {"logits": logits, "fusion_logits": logits, "unimodal_logits": uni,
                "reliability": {m: weights[m] / denom for m in weights}, "present": present,
                "fused": None}


CORES = {"ralf": RALFCore, "concat": ConcatCore, "weighted_logit": WeightedLogitCore}


class C1Multimodal(nn.Module):
    """Per-task, per-modality C1 encoders -> summary tokens -> one fusion core."""

    def __init__(self, encoders: Dict[str, Dict[str, nn.Module]],
                 metas: Dict[str, Dict[str, object]], tasks: Dict[str, int],
                 fusion: str = "ralf", summary_tokens: int = 4, summary_heads: int = 4,
                 keep_index: Optional[Dict[str, Dict[str, Optional[List[int]]]]] = None,
                 **core_kwargs):
        super().__init__()
        self.tasks = dict(tasks)
        self.encoders = nn.ModuleDict({t: nn.ModuleDict(dict(ms)) for t, ms in encoders.items()})
        self.metas = metas
        # Rows of the stored window an encoder takes, when its route left a
        # few electrodes out (encoders.build_encoder); None = all of them.
        self.keep_index: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for t, per in (keep_index or {}).items():
            self.keep_index[t] = {}
            for m, idx in per.items():
                if idx is not None:
                    buf = torch.tensor(idx, dtype=torch.long)
                    self.register_buffer(f"keep_{t}_{m}", buf, persistent=False)
                    self.keep_index[t][m] = buf
                else:
                    self.keep_index[t][m] = None
        first = next(iter(encoders.values()))
        self.modalities = [m for m in MODALITIES if m in first]
        dims = {m: int(first[m].embed_dim) for m in self.modalities}
        self.pools = nn.ModuleDict({m: SummaryPool(dims[m], summary_tokens, summary_heads)
                                    for m in self.modalities})
        if fusion not in CORES:
            raise ValueError(f"fusion must be one of {sorted(CORES)}, got {fusion!r}")
        self.fusion_name = fusion
        self.core = CORES[fusion](self.modalities, dims, self.tasks, **core_kwargs)

    def encode(self, x: Dict[str, torch.Tensor], task: str) -> Dict[str, torch.Tensor]:
        feats = {}
        for m in self.modalities:
            if x.get(m) is None:
                continue
            xm = x[m]
            keep = self.keep_index.get(task, {}).get(m)
            if keep is not None:
                xm = xm.index_select(1, keep.to(xm.device))
            tokens = self.encoders[task][m].encode(xm, self.metas[task][m])
            feats[m] = self.pools[m](tokens)
        return feats

    def forward(self, x: Dict[str, torch.Tensor], task: str,
                modality_mask: Optional[Dict[str, torch.Tensor]] = None,
                features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
        feats = features if features is not None else self.encode(x, task)
        return self.core(feats, task, modality_mask)

    def encoder_depth(self) -> int:
        enc = next(iter(next(iter(self.encoders.values())).values()))
        if hasattr(enc, "depth"):
            return int(enc.depth)
        return len(enc.shared_transformer.blocks)


__all__ = ["SummaryPool", "RALFCore", "ConcatCore", "WeightedLogitCore", "CORES",
           "C1Multimodal"]
