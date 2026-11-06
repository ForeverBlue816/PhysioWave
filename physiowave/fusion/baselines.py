"""EEGPT, LaBraM and LUNA as EEG encoders of the fusion trainer, and the channel adapter.

The point is a like-for-like single-modality comparison on SeizeIT2: the
same windows, split, head, schedule, model selection and event-level scorer
for the C1 EEG encoder and for two published EEG foundation models. The
fusion trainer only needs an encoder to offer ``embed_dim`` and
``encode(x, meta) -> [B, L, D]`` tokens; the two classes here wrap the
published backbones to that interface.

Backbones -- the published encoders, re-implemented with the checkpoints'
parameter names so the released weights load strictly:

    eegpt   EEGPT (Wang et al., NeurIPS 2024), the target encoder of
            ``eegpt_mcae_58chs_4s_large4E``: 64-sample patches at 256 Hz,
            each patch's channels plus 4 summary tokens through 8 blocks
            (512 wide); the encoder's output is the summary tokens,
            [B, patches, 4, 512] -> [B, 4*patches, 512]. Input in mV.
            Source: github.com/BINE022/EEGPT (Apache-2.0),
            downstream/Modules/models/EEGPT_mcae.py, ``EEGTransformer``.
    labram  LaBraM-Base (Jiang et al., ICLR 2024), ``labram-base.pth``'s
            student: 200-sample patches at 200 Hz through a temporal conv,
            electrode and time embeddings, 12 blocks (200 wide) with q/k
            LayerNorm and layer scale. Input in units of 0.1 mV.
            Source: github.com/935963004/LaBraM (MIT), modeling_finetune.py,
            ``NeuralTransformer``.
    luna_base / luna_large / luna_huge
            LUNA (Doner et al., NeurIPS 2025), ``LUNA_<size>.safetensors``:
            40-sample patches at 256 Hz embedded by a temporal conv plus an
            MLP on each patch's rfft magnitude and phase, a NeRF encoding of
            the electrodes' standard_1005 coordinates (min-max normalised over
            the channels present), Q learned queries cross-attending over the
            channels of each patch (channel unification), then a RoPE
            Transformer over the patch sequence, Q*D wide: base 4x64 / 8
            blocks, large 6x96 / 10, huge 8x128 / 24. Input z-scored per
            channel. The pretraining decoder and channel table are dropped.
            Source: github.com/pulp-bio/BioFoundation (code Apache-2.0),
            models/LUNA.py; weights huggingface.co/PulpBio/LUNA (CC BY-ND 4.0:
            fine-tuned weights may not be redistributed).

Montage. SeizeIT2 has two behind-the-ear channels that are no electrode of
either model's vocabulary (nor of C1's). EEGPT's own recipe for a two-channel
dataset (Sleep-EDF, downstream/finetune_EEGPT_SleepEDF.py) is a learned 1x1
convolution with a max-norm constraint from the recorded channels onto named
10-20 electrodes; :class:`ChannelAdapter` is that layer, and the same adapter
onto the same electrodes (default the temporal chains F7 T7 P7 / F8 T8 P8,
which every model knows and which put C1 on its E19_256 route) is given to
all three models, so none gets a montage advantage. The adapter always
trains, also when the backbone is frozen.

Amplitude. The stored windows are z-scored per channel (prep_common); the
adapter's output is z-scored per channel again and then scaled to the input
convention of each model's pretraining: C1 and LUNA z-scored (scale 1), EEGPT
in mV and LaBraM in 0.1 mV for a channel of ``input_uv`` microvolts RMS (30,
typical behind-the-ear EEG; SeizeIT2's reference code accepts 13-150 uV).

Window. LUNA's patches are 40 samples; a 4 s window at 256 Hz is 1024, so the
centred 1000 samples (25 patches, 12 dropped at each end) are what it sees.

Frozen means frozen: a frozen backbone is kept in eval mode (no dropout or
drop-path), so its features are the same every time it sees a window.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: EEGPT's channel vocabulary (EEGPT_mcae.py CHANNEL_DICT), in index order.
EEGPT_CHANNELS: Tuple[str, ...] = (
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2")

#: LaBraM's electrode list (utils.py standard_1020); position i+1 of pos_embed
#: (0 is the CLS token).
LABRAM_CHANNELS: Tuple[str, ...] = tuple(
    "FP1 FPZ FP2 AF9 AF7 AF5 AF3 AF1 AFZ AF2 AF4 AF6 AF8 AF10 "
    "F9 F7 F5 F3 F1 FZ F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 FT10 "
    "T9 T7 C5 C3 C1 CZ C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 TP10 "
    "P9 P7 P5 P3 P1 PZ P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POZ PO2 PO4 PO6 PO8 PO10 "
    "O1 OZ O2 O9 CB1 CB2 IZ O10 T3 T5 T4 T6 M1 M2 A1 A2 "
    "CFC1 CFC2 CFC3 CFC4 CFC5 CFC6 CFC7 CFC8 CCP1 CCP2 CCP3 CCP4 CCP5 CCP6 CCP7 CCP8 "
    "T1 T2 FTT9h TTP7h TPP9h FTT10h TPP8h TPP10h".split()) + (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2")

DEFAULT_TARGETS: Tuple[str, ...] = ("F7", "T7", "P7", "F8", "T8", "P8")
LUNA_SIZES: Dict[str, Dict[str, int]] = {          # config/model/LUNA_<size>.yaml
    "luna_base": {"embed_dim": 64, "num_queries": 4, "depth": 8, "num_heads": 2},
    "luna_large": {"embed_dim": 96, "num_queries": 6, "depth": 10, "num_heads": 2},
    "luna_huge": {"embed_dim": 128, "num_queries": 8, "depth": 24, "num_heads": 2},
}
ARCHS = ("c1", "eegpt", "labram") + tuple(LUNA_SIZES)

#: standard_1005 electrode positions in MNE's head frame (metres), exactly what
#: LUNA's get_channel_locations returns: mne.create_info + set_montage(
#: make_standard_montage("standard_1005")) + get_montage().get_positions().
#: The head transform comes from the montage fiducials, so it does not depend on
#: which channels are present. Other names are looked up the same way in MNE.
#: Full precision matters: LUNA's NeRF encoding multiplies the (normalised)
#: coordinates by up to 2^15 before the sine, so six decimals already move its
#: highest bands.
STANDARD_1005: Dict[str, Tuple[float, float, float]] = {
    "FP1": (-0.030902595782959074, 0.1145851808674079, 0.027866574329093563),
    "FP2": (0.028409487887091203, 0.11534630809630636, 0.02772125772253031),
    "F7": (-0.07187663250970616, 0.07310353217275958, 0.02579045587891774),
    "F3": (-0.05180904586237076, 0.0866879023191724, 0.0787140930288383),
    "FZ": (-0.0012292786120998045, 0.09327445212894307, 0.10263929347682083),
    "F4": (0.05027427028109521, 0.087438385860065, 0.07727065434234637),
    "F8": (0.07143526180097565, 0.07450512184827827, 0.025101027393181548),
    "T7": (-0.08598208540859081, 0.014871643830998479, 0.03117337373904913),
    "C3": (-0.06714873038365092, 0.023358234962598565, 0.10451068106025274),
    "CZ": (-0.0013741408533525957, 0.027617093717945417, 0.1401994941744),
    "C4": (0.06532887326785189, 0.02357310071942504, 0.10369243420048097),
    "T8": (0.08326136240706139, 0.015258183861365685, 0.030972968524064372),
    "P7": (-0.07445797197401204, -0.042123160070390187, 0.041273631387385656),
    "P3": (-0.05503823925315043, -0.04421030104524479, 0.09990897692201318),
    "PZ": (-0.0017094524972804195, -0.04521299319629392, 0.12667292418210066),
    "P4": (0.053636008012841616, -0.044334525348021875, 0.10051602773215543),
    "P8": (0.07103246343848865, -0.04225998064272267, 0.04119886278195534),
    "O1": (-0.0315735653251896, -0.08056835005318111, 0.054789645624052044),
    "O2": (0.027683093109090593, -0.08048884161275736, 0.05473408147779854),
    "FT7": (-0.082489318340554, 0.04484874350085445, 0.027680584654195248),
    "FT8": (0.08010431965005083, 0.045553577040141964, 0.027412390613802098),
    "TP7": (-0.08675728536194152, -0.014950971454822274, 0.03515865615910235),
    "TP8": (0.08362231532894225, -0.01508567532515739, 0.03505769683479039),
    "TP9": (-0.08755584081204944, -0.017629016355695474, -0.0034024089114834885),
    "TP10": (0.08422219202241313, -0.018769085061997915, -0.003534771527952618),
    "T9": (-0.08772226675034184, 0.01286266211360428, -0.007711907574576064),
    "T10": (0.08372876068402091, 0.01172162898342885, -0.007669869674859137),
    "FT9": (-0.0857965490757633, 0.04308171037172904, -0.01157567583844054),
    "FT10": (0.0823906466577189, 0.04227490959920134, -0.011673128497876314),
    "F9": (-0.07172633848477208, 0.07010037597452946, -0.012633170672804984),
    "F10": (0.07049013385387566, 0.06997984301139885, -0.013155918880650479),
    "P9": (-0.07504186115216518, -0.04461432864236574, 0.002842148832066481),
    "P10": (0.07185894733971461, -0.04577312410626676, 0.002655786411093812),
}


def electrode_positions(names: Sequence[str]) -> np.ndarray:
    """``[C, 3]`` standard_1005 coordinates; a bipolar name ``A-B`` sits at the
    midpoint of A and B, as in LUNA's get_channel_locations."""
    def one(n: str):
        k = n.upper().strip()
        if "-" in k:
            a, b = k.split("-", 1)
            return (np.asarray(one(a)) + np.asarray(one(b))) / 2
        if k in STANDARD_1005:
            return STANDARD_1005[k]
        try:
            import mne
            montage = mne.channels.make_standard_montage("standard_1005")
            match = [c for c in montage.ch_names if c.upper() == k]
            if match:
                info = mne.create_info(ch_names=match, sfreq=256, ch_types=["eeg"])
                info.set_montage(montage, match_case=False)
                return tuple(float(v) for v in info.get_montage().get_positions()["ch_pos"][match[0]])
        except ImportError:
            pass
        raise ValueError(f"no standard_1005 position for electrode {n!r}")
    return np.asarray([one(n) for n in names], dtype=np.float64).astype(np.float32)


# --------------------------------------------------------------------------- #
# Checkpoint reading
# --------------------------------------------------------------------------- #

_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "F64": torch.float64, "I64": torch.int64, "I32": torch.int32}


def read_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """A ``.safetensors`` file as a state dict, without the safetensors package."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()
    header.pop("__metadata__", None)
    out = {}
    for k, spec in header.items():
        a, b = spec["data_offsets"]
        dt = _ST_DTYPES[spec["dtype"]]
        t = torch.frombuffer(bytearray(blob[a:b]), dtype=dt) if b > a else torch.empty(0, dtype=dt)
        out[k] = t.reshape(spec["shape"]).clone()
    return out


def read_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """The flat state dict of a released checkpoint (.ckpt / .pth / .safetensors)."""
    if path.endswith(".safetensors"):
        return read_safetensors(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "module"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            return ck[key]
    return ck


def sub_state(sd: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    """The entries under the first prefix that has any, prefix removed."""
    for p in prefixes:
        got = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        if got:
            return got
    return {}


# --------------------------------------------------------------------------- #
# The channel adapter (EEGPT's Conv1dWithConstraint, max_norm=1)
# --------------------------------------------------------------------------- #

class ChannelAdapter(nn.Conv1d):
    """``[B, C_in, T] -> [B, C_out, T]``: a 1x1 convolution whose output
    filters are renormalised to L2 norm <= ``max_norm`` before every use."""

    def __init__(self, in_channels: int, out_channels: int, max_norm: float = 1.0):
        super().__init__(in_channels, out_channels, kernel_size=1)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.std(dim=-1, keepdim=True).clamp_min(eps)


# --------------------------------------------------------------------------- #
# EEGPT target encoder
# --------------------------------------------------------------------------- #

class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _EEGPTAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class _EEGPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _EEGPTAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _EEGPTPatch(nn.Module):
    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=(1, patch), stride=(1, patch))

    def forward(self, x):                                    # [B, C, T] -> [B, N, C, D]
        return self.proj(x.unsqueeze(1)).transpose(1, 3)


class EEGPTBackbone(nn.Module):
    """EEGPT's ``EEGTransformer`` (the target encoder), large: 512 x 8, 4 summary tokens."""

    def __init__(self, patch: int = 64, dim: int = 512, depth: int = 8, heads: int = 8,
                 embed_num: int = 4):
        super().__init__()
        self.patch, self.embed_dim, self.embed_num = patch, dim, embed_num
        self.patch_embed = _EEGPTPatch(patch, dim)
        self.chan_embed = nn.Embedding(len(EEGPT_CHANNELS), dim)
        self.blocks = nn.ModuleList(_EEGPTBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for i, b in enumerate(self.blocks):                   # EEGPT's fix_init_weight
            b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            b.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    @staticmethod
    def channel_ids(names: Sequence[str]) -> List[int]:
        idx = {c: i for i, c in enumerate(EEGPT_CHANNELS)}
        out = []
        for n in names:
            k = n.upper().strip(".")
            if k not in idx:
                raise ValueError(f"EEGPT has no electrode {n!r}")
            out.append(idx[k])
        return out

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` (mV) -> summary tokens ``[B, N * embed_num, D]``."""
        x = self.patch_embed(x)                              # B, N, C, D
        B, N, C, D = x.shape
        x = x + self.chan_embed(chan_ids.long()).view(1, 1, C, D)
        x = x.flatten(0, 1)                                  # B*N, C, D
        x = torch.cat([x, self.summary_token.expand(x.shape[0], -1, -1).to(x.dtype)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -self.embed_num:, :])             # B*N, E, D
        return x.reshape(B, N * self.embed_num, D)


# --------------------------------------------------------------------------- #
# LaBraM-Base
# --------------------------------------------------------------------------- #

class _LaBraMAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(hd, eps=1e-6)
        self.k_norm = nn.LayerNorm(hd, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).type_as(v), self.k_norm(k).type_as(v)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, -1))


class _LaBraMBlock(nn.Module):
    def __init__(self, dim: int, heads: int, init_values: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _LaBraMAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * 4)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class _TemporalConv(nn.Module):
    def __init__(self, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)

    def forward(self, x):                                    # [B, C, A, 200] -> [B, C*A, 200]
        B, C, A, T = x.shape
        x = x.reshape(B, 1, C * A, T)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))                # B, 8, C*A, 25
        return x.permute(0, 2, 3, 1).reshape(B, C * A, -1)   # (T C) order, as rearrange does


class LaBraMBackbone(nn.Module):
    """LaBraM-Base's ``NeuralTransformer`` (200 x 12, 10 heads), the pretrained final norm kept."""

    def __init__(self, dim: int = 200, depth: int = 12, heads: int = 10, patch: int = 200):
        super().__init__()
        self.embed_dim, self.patch = dim, patch
        self.patch_embed = _TemporalConv()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 16, dim))
        self.blocks = nn.ModuleList(_LaBraMBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        for p in (self.cls_token, self.pos_embed, self.time_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for i, b in enumerate(self.blocks):
            b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            b.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    @staticmethod
    def channel_ids(names: Sequence[str]) -> List[int]:
        """Rows of ``pos_embed`` (1-based; 0 is CLS), as LaBraM's get_input_chans."""
        idx = {c.upper(): i for i, c in enumerate(LABRAM_CHANNELS)}
        out = []
        for n in names:
            k = n.upper()
            if k not in idx:
                raise ValueError(f"LaBraM has no electrode {n!r}")
            out.append(idx[k] + 1)
        return out

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """``[B, C, A, 200]`` (0.1 mV) -> patch tokens ``[B, C*A, D]``."""
        B, C, A, _ = x.shape
        x = self.patch_embed(x)
        pos = self.pos_embed[:, chan_ids.long()]             # 1, C, D
        pos = pos.unsqueeze(2).expand(B, C, A, -1).reshape(B, C * A, -1)
        time = self.time_embed[:, :A].unsqueeze(1).expand(B, C, A, -1).reshape(B, C * A, -1)
        x = x + pos + time
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        x = torch.cat([cls.to(x.dtype), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 1:]


# --------------------------------------------------------------------------- #
# LUNA
# --------------------------------------------------------------------------- #

class _TimmMlp(nn.Module):
    """timm.layers.Mlp: fc1 -> GELU -> drop -> norm -> fc2 -> drop."""

    def __init__(self, in_f: int, hidden: int, out_f: int, norm: bool = False, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden) if norm else nn.Identity()
        self.fc2 = nn.Linear(hidden, out_f)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class _LUNAPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        oc, k = embed_dim // 4, patch // 2
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, oc, kernel_size=(1, k - 1), stride=(1, k // 2), padding=(0, k // 2 - 1)),
            nn.GroupNorm(4, oc), nn.GELU(),
            nn.Conv2d(oc, oc, kernel_size=(1, 3), padding=(0, 1)), nn.GroupNorm(4, oc), nn.GELU(),
            nn.Conv2d(oc, oc, kernel_size=(1, 3), padding=(0, 1)), nn.GroupNorm(4, oc), nn.GELU())

    def forward(self, x):                                    # [B, C, T] -> [B, C*S, D]
        B, C, T = x.shape
        x = x.reshape(B, C * (T // self.patch), self.patch).unsqueeze(1)
        x = self.proj_in(x)                                  # B, E, C*S, L
        return x.permute(0, 2, 3, 1).reshape(B, x.shape[2], -1)   # (L E) order, as rearrange


class _LUNAFreqEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch: int):
        super().__init__()
        self.patch = patch
        n = 2 * (patch // 2 + 1)
        self.frequency_to_embed = _TimmMlp(n, 4 * n, embed_dim)

    def forward(self, x):                                    # [B, C, T] -> [B, C*S, D]
        B, C, T = x.shape
        f = torch.fft.rfft(x.float().reshape(B, C, T // self.patch, self.patch), dim=-1)
        feats = torch.cat([f.abs(), torch.angle(f)], dim=-1)
        return self.frequency_to_embed(feats).reshape(B, -1, self.frequency_to_embed.fc2.out_features)


def _nerf(coords: torch.Tensor, size: int) -> torch.Tensor:
    N, C, dim = coords.shape
    freqs = size // (2 * dim)
    left = size - freqs * 2 * dim
    bands = 2.0 ** torch.arange(freqs, device=coords.device).float()
    sc = coords.unsqueeze(-1) * bands.view(1, 1, 1, -1)
    enc = torch.stack([torch.sin(sc), torch.cos(sc)], dim=-1).permute(0, 1, 3, 2, 4)
    enc = enc.reshape(N, C, freqs * dim * 2)
    if left > 0:
        enc = torch.cat([enc, enc.new_zeros(N, C, left)], dim=-1)
    return enc


class _LUNACrossAttn(nn.Module):
    """The channel-unification module: Q learned queries attend over a patch's channels."""

    def __init__(self, num_queries: int, dim: int, heads: int, ff_dim: int, drop: float = 0.1):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, dim))
        nn.init.orthogonal_(self.query_embed)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ffn = _TimmMlp(dim, ff_dim, dim, norm=True, drop=drop)
        self.keys_norm = nn.LayerNorm(dim)
        self.values_norm = nn.LayerNorm(dim)
        self.queries_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=heads, activation="gelu", dim_feedforward=ff_dim,
                                           batch_first=True, norm_first=True)
        self.query_self_attn = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)

    def forward(self, x):                                    # [N, C, D] -> [N, Q, D]
        q = self.queries_norm(self.query_embed.expand(x.shape[0], -1, -1))
        a, _ = self.cross_attention(q, self.keys_norm(x), self.values_norm(x), need_weights=False)
        a = self.ffn(a) + a
        return self.query_self_attn(a)


class _DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class _LUNARotary(nn.Module):
    """rotary_embedding_torch.RotaryEmbedding (lang frequencies, interleaved pairs)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)),
                                  requires_grad=False)

    def rotate(self, t):                                     # [B, H, N, d]
        n = t.shape[-2]
        ang = torch.arange(n, device=t.device, dtype=torch.float32)[:, None] * self.freqs.float()[None]
        ang = ang.repeat_interleave(2, dim=-1)
        tf = t.float()
        x1, x2 = tf.reshape(*tf.shape[:-1], -1, 2).unbind(-1)
        rot = torch.stack((-x2, x1), dim=-1).reshape(tf.shape)
        return (tf * ang.cos() + rot * ang.sin()).type(t.dtype)


class _LUNAAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.rotary_emb = _LUNARotary(dim // heads)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv_proj(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rotary_emb.rotate(q), self.rotary_emb.rotate(k)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, C))


class _LUNAFeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.norm(F.gelu(self.fc1(x))))


class _LUNABlock(nn.Module):
    def __init__(self, dim: int, heads: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _LUNAAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _LUNAFeedForward(dim, 4 * dim)
        self.drop_path = _DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class LUNABackbone(nn.Module):
    """LUNA's encoder (models/LUNA.py up to ``x_latent``): patch + frequency
    embedding, electrode-position encoding, channel unification, RoPE blocks."""

    def __init__(self, embed_dim: int = 64, num_queries: int = 4, depth: int = 8,
                 num_heads: int = 2, patch: int = 40, drop_path: float = 0.1):
        super().__init__()
        self.patch, self.unit, self.num_queries = patch, embed_dim, num_queries
        self.embed_dim = embed_dim * num_queries
        self.patch_embed = _LUNAPatchEmbed(embed_dim, patch)
        self.freq_embed = _LUNAFreqEmbed(embed_dim, patch)
        self.channel_location_embedder = nn.Sequential(
            _TimmMlp(embed_dim, 2 * embed_dim, embed_dim, norm=True))
        self.cross_attn = _LUNACrossAttn(num_queries, embed_dim, num_heads, 4 * embed_dim)
        self.blocks = nn.ModuleList(_LUNABlock(self.embed_dim, num_heads * num_queries, drop_path)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """``x`` [B, C, T] z-scored, T a multiple of the patch; ``locations`` [C, 3]
        -> latent [B, T / patch, Q*D]."""
        B, C, T = x.shape
        S = T // self.patch
        tok = self.patch_embed(x) + self.freq_embed(x)       # B, C*S, D
        loc = locations[None].float()
        lo, hi = loc.min(dim=1, keepdim=True)[0], loc.max(dim=1, keepdim=True)[0]
        loc = (loc - lo) / (hi - lo + 1e-8)
        loc = self.channel_location_embedder(_nerf(loc, self.unit))   # 1, C, D
        tok = tok.reshape(B, C, S, -1).transpose(1, 2).reshape(B * S, C, -1) + loc.to(tok.dtype)
        z = self.cross_attn(tok)                             # B*S, Q, D
        z = z.reshape(B, S, -1)
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


# --------------------------------------------------------------------------- #
# The encoders the fusion trainer builds
# --------------------------------------------------------------------------- #

def _decay_for(key: str, prefix: str, depth: int, d: float) -> float:
    """Layer-wise decay for a backbone parameter named ``<prefix>blocks.<i>....``:
    block i at d**(depth - i), the final norm at 1, everything feeding block 0
    (patch embedding, electrode/time embeddings, tokens) at d**(depth + 1)."""
    m = re.match(re.escape(prefix) + r"blocks\.(\d+)\.", key)
    if m:
        return d ** (depth - int(m.group(1)))
    if key.startswith(prefix + "norm."):
        return 1.0
    return d ** (depth + 1)


class BaselineEEGEncoder(nn.Module):
    """Adapter -> per-channel z-score -> amplitude scale -> EEGPT, LaBraM or LUNA."""

    def __init__(self, arch: str, in_channels: int, fs: float, window_samples: int,
                 targets: Sequence[str] = DEFAULT_TARGETS, input_uv: float = 30.0):
        super().__init__()
        if arch not in ("eegpt", "labram") + tuple(LUNA_SIZES):
            raise ValueError(arch)
        self.frozen = False
        self.arch, self.fs, self.window_samples = arch, float(fs), int(window_samples)
        self.targets = [str(t) for t in targets]
        self.adapter = ChannelAdapter(in_channels, len(self.targets))
        if arch == "eegpt":
            if int(round(fs)) != 256:
                raise ValueError("EEGPT was pretrained at 256 Hz; the window is at %s Hz" % fs)
            self.backbone = EEGPTBackbone()
            if self.window_samples % self.backbone.patch:
                raise ValueError(f"{window_samples} samples is not a whole number of "
                                 f"{self.backbone.patch}-sample EEGPT patches")
            self.scale = float(input_uv) * 1e-3                 # mV
            ids = EEGPTBackbone.channel_ids(self.targets)
        elif arch in LUNA_SIZES:
            if int(round(fs)) != 256:
                raise ValueError("LUNA was pretrained at 256 Hz; the window is at %s Hz" % fs)
            self.backbone = LUNABackbone(**LUNA_SIZES[arch])
            self.crop = (self.window_samples // self.backbone.patch) * self.backbone.patch
            self.crop_start = (self.window_samples - self.crop) // 2
            self.scale = 1.0                                    # z-scored, as pretrained
            ids = []
            self.register_buffer("locations", torch.from_numpy(electrode_positions(self.targets)),
                                 persistent=False)
        else:
            self.backbone = LaBraMBackbone()
            self.resampled = int(round(self.window_samples * 200.0 / self.fs))
            if self.resampled % self.backbone.patch or self.resampled // self.backbone.patch > 16:
                raise ValueError(f"{self.resampled} samples at 200 Hz do not make 1-16 "
                                 "LaBraM patches")
            self.scale = float(input_uv) * 1e-2                 # 0.1 mV
            ids = LaBraMBackbone.channel_ids(self.targets)
        self.register_buffer("chan_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.embed_dim = int(self.backbone.embed_dim)
        self.depth = len(self.backbone.blocks)

    def encode(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        h = zscore(self.adapter(x.float())) * self.scale
        if self.arch in LUNA_SIZES:
            h = h[..., self.crop_start:self.crop_start + self.crop]
            return self.backbone(h, self.locations)
        if self.arch == "labram":
            h = F.interpolate(h, size=self.resampled, mode="linear", align_corners=False)
            h = h.reshape(h.shape[0], h.shape[1], -1, self.backbone.patch)
        return self.backbone(h, self.chan_ids)

    # ------------------------------------------------------------------ #
    def load_pretrained(self, path: str) -> Dict[str, List[str]]:
        sd = read_checkpoint(path)
        prefixes = (("target_encoder.",) if self.arch == "eegpt"
                    else ("model.", "") if self.arch in LUNA_SIZES else ("student.", ""))
        st = sub_state(sd, prefixes)
        if self.arch in LUNA_SIZES:
            # pretraining-only parts: the decoder, the channel table, the mask token,
            # and an unused temperature
            st = {k: v for k, v in st.items()
                  if not k.startswith(("decoder_head.", "channel_emb.", "classifier."))
                  and k not in ("mask_token", "cross_attn.temparature")}
        if self.arch == "labram":
            # the pretraining-only parts: the mask token and the codebook head
            st = {k: v for k, v in st.items()
                  if k != "mask_token" and not k.startswith("lm_head.")}
        if not st:
            raise SystemExit(f"{path}: no {self.arch} encoder weights under {prefixes}")
        try:
            missing, unexpected = self.backbone.load_state_dict(st, strict=False)
        except RuntimeError as e:                            # shapes: another size or model
            raise SystemExit(f"{path} does not fit the {self.arch} backbone: "
                             f"{str(e).splitlines()[1].strip() if len(str(e).splitlines()) > 1 else e}")
        if missing or unexpected:
            raise SystemExit(f"{path} does not fit the {self.arch} backbone: missing "
                             f"{list(missing)[:8]}, unexpected {list(unexpected)[:8]}")
        return {"taken": sorted({k.split(".")[0] for k in st}), "skipped": []}

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.frozen = True
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()                             # no dropout in a frozen encoder
        return self

    def layer_lr_scale(self, key: str, encoder_lr_scale: float, layer_decay: float) -> float:
        if not key.startswith("backbone."):
            return 1.0                                       # the adapter learns at lr
        f = float(encoder_lr_scale)
        if layer_decay != 1.0:
            f *= _decay_for(key, "backbone.", self.depth, float(layer_decay))
        return f


class AdaptedC1Encoder(nn.Module):
    """The same adapter in front of a C1 EEG encoder built for the target electrodes."""

    def __init__(self, inner: nn.Module, in_channels: int, targets: Sequence[str]):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels, len(targets))
        self.inner = inner
        self.targets = [str(t) for t in targets]
        from .encoders import channel_meta
        self.inner_meta = channel_meta(self.targets)
        self.embed_dim = int(inner.embed_dim)
        self.depth = len(inner.shared_transformer.blocks)

    def encode(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        return self.inner.encode(zscore(self.adapter(x.float())).to(x.dtype), self.inner_meta)

    def layer_lr_scale(self, key: str, encoder_lr_scale: float, layer_decay: float) -> float:
        if not key.startswith("inner."):
            return 1.0
        from physiowave.train.finetune_main import c1_lr_scale
        return c1_lr_scale(self.depth, encoder_lr_scale, layer_decay)(key[len("inner."):])


def build_baseline(arch: str, names: Sequence[str], fs: float, window_samples: int,
                   checkpoint: Optional[str], freeze: bool,
                   targets: Sequence[str] = DEFAULT_TARGETS, input_uv: float = 30.0
                   ) -> Tuple[nn.Module, dict]:
    enc = BaselineEEGEncoder(arch, len(names), fs, window_samples, targets, input_uv)
    report = {"modality": "eeg", "arch": arch, "route": None, "exported": checkpoint,
              "embed_dim": enc.embed_dim, "adapter": {"from": list(names), "to": enc.targets},
              "input_uv": float(input_uv), "taken": [], "skipped": [],
              "dropped_channels": [], "keep_index": None}
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise SystemExit(f"no {arch} checkpoint at {checkpoint}")
        report.update(enc.load_pretrained(checkpoint))
        report["summary"] = (f"{arch} loaded from {os.path.basename(checkpoint)}; adapter "
                             f"{len(names)} -> {'/'.join(enc.targets)} trains")
    else:
        report["summary"] = f"{arch} from scratch"
    if freeze:
        enc.freeze_backbone()
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    report["trainable_params"] = int(n_tr)
    return enc, report


__all__ = ["ARCHS", "DEFAULT_TARGETS", "EEGPT_CHANNELS", "LABRAM_CHANNELS", "LUNA_SIZES",
           "STANDARD_1005", "electrode_positions", "ChannelAdapter",
           "EEGPTBackbone", "LaBraMBackbone", "LUNABackbone", "BaselineEEGEncoder", "AdaptedC1Encoder",
           "build_baseline", "read_checkpoint", "read_safetensors"]
