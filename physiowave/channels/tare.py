"""TARE -- Topology-and-Reference-Aware channel Encoder.

TARE turns per-channel *metadata* into a per-channel embedding that the
tokenizer's output is conditioned on.  The design follows three rules that come
from the physics of scalp recording (constraint C in ``docs/terminology.md``):

1. **Geometry is primary, names are auxiliary.**  A 3-D scalp coordinate is the
   thing that actually determines what an electrode measures; the label is a
   convention.  Under a standard montage the two are nearly redundant, so adding
   their embeddings would make the contributions unidentifiable.  TARE therefore
   *concatenates* the components and fuses them with an MLP (``fusion_mode
   ='concat_mlp'``, default), or drives a coordinate trunk with FiLM modulation
   from the metadata (``fusion_mode='film'``, ablation).  A recording with
   coordinates but no usable names must still work -- see
   ``tests/test_channels.py::test_coordinates_only``.

2. **Channel order carries no meaning.**  Nothing in TARE depends on the index of
   a channel: permuting the signal and its metadata together permutes the
   per-channel embeddings and leaves the pooled representation unchanged.

3. **A bipolar derivation is a directed pair, not a point.**  ``F7-T7`` and
   ``T7-F7`` are the same electrodes with opposite sign, so encoding the midpoint
   would make them identical.  TARE encodes both endpoints *and* their signed
   difference.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from ..data.montages import TEMPLATE_POSITIONS, canonical_name

logger = logging.getLogger(__name__)

#: Reference schemes TARE can distinguish.  ``unknown`` is always index 0.
REFERENCE_TYPES = (
    "unknown", "original", "common_average", "linked_mastoids", "linked_ears",
    "left_mastoid", "right_mastoid", "left_ear", "right_ear",
    "single_channel", "cz", "bipolar",
)

#: Derivation schemes.  ``bipolar`` routes through the two-endpoint encoder.
DERIVATION_TYPES = ("unknown", "monopolar", "bipolar", "laplacian", "csd")

#: Montage families, used only as a coarse categorical hint.
MONTAGE_TYPES = ("unknown", "standard_1020", "standard_1010", "hd", "custom", "clinical_bipolar")

FUSION_MODES = ("concat_mlp", "film")


@dataclass
class TAREConfig:
    """Configuration for :class:`ChannelEncoder`."""

    embed_dim: int = 256
    fusion_mode: str = "concat_mlp"
    coord_fourier_bands: int = 8
    coord_fourier_scale: float = 2.0
    coord_dim: int = 128
    name_dim: int = 32
    meta_dim: int = 32
    name_vocab_size: int = 512
    hidden_dim: int = 256
    dropout: float = 0.0
    warn_on_unknown_coord: bool = True
    # Component switches for the tier-1 ablation ladder.  Each rung of
    # configs/experiments/channel_ablation.yaml turns exactly one of these on, so
    # the ladder measures a mechanism rather than a hyperparameter change.
    use_coordinates: bool = True        # the primary spatial evidence
    use_name_embedding: bool = True     # auxiliary identity
    use_reference_metadata: bool = True # reference / derivation / montage type

    def __post_init__(self) -> None:
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")


class FourierPositionEncoding(nn.Module):
    """NeRF-style Fourier features of a 3-D coordinate.

    ``x -> [x, sin(2^k pi s x), cos(2^k pi s x)]`` for ``k < n_bands``.  A plain
    linear layer on raw ``xyz`` cannot represent the high spatial frequencies
    needed to separate neighbouring electrodes on a dense montage; the Fourier
    lift makes that separation available while keeping the encoding smooth in
    space, which is what lets an unseen montage interpolate.
    """

    def __init__(self, n_bands: int = 8, scale: float = 2.0) -> None:
        super().__init__()
        self.n_bands = n_bands
        freqs = scale * (2.0 ** torch.arange(n_bands, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 3 * (1 + 2 * self.n_bands)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """``[..., 3] -> [..., out_dim]``."""
        proj = xyz.unsqueeze(-1) * self.freqs.view(*([1] * xyz.dim()), -1)
        return torch.cat([xyz, proj.sin().flatten(-2), proj.cos().flatten(-2)], dim=-1)


def _name_index(name: str, vocab: Dict[str, int], table_size: int) -> int:
    """Map a channel label to a vocabulary index; 0 is the unnamed slot.

    A label the template montage does not know -- ``ch00`` on an sEMG ring, a
    lead label, anything outside the 10-20/10-10 families -- is hashed into the
    slots above the template vocabulary rather than sharing slot 0 with every
    other unrecognised label.  Collapsing them gave every channel of a non-EEG
    montage the *same* embedding, and since such a montage also has no template
    coordinates, the encoder was left with nothing that distinguished one
    channel from another: the whole model became invariant to permuting the
    channel axis.  See ``tests/test_channels.py::test_unknown_names_stay_distinct``.

    Identity still comes from the label, never from the position, so rule 2 of
    the module docstring holds: permuting the signal and its metadata together
    permutes the embeddings and changes nothing else.

    crc32 rather than the built-in ``hash``: the latter is salted per process,
    and two DDP ranks that disagreed about which row a channel owns would
    all-reduce gradients for different embeddings.
    """
    canon = canonical_name(name)
    if canon in vocab:
        return vocab[canon]
    if not canon:
        return 0                                    # genuinely unnamed
    reserved = len(vocab) + 1
    span = table_size - reserved
    if span <= 0:                                   # table too small to hash into
        return 0
    return reserved + zlib.crc32(canon.encode("utf-8")) % span


def build_name_vocab() -> Dict[str, int]:
    """Vocabulary over the template montage labels; index 0 is reserved.

    Labels outside it are not in this mapping at all -- :func:`_name_index`
    hashes them into the slots above ``len(vocab)``.
    """
    return {n: i + 1 for i, n in enumerate(sorted(TEMPLATE_POSITIONS.keys()))}


@dataclass
class ChannelMeta:
    """Per-recording channel metadata consumed by :class:`ChannelEncoder`."""

    channel_names: Sequence[str]
    channel_xyz: torch.Tensor                       # [C, 3]; all-zero row = unknown
    channel_mask: Optional[torch.Tensor] = None     # [C] bool, False = missing/bad
    channel_quality: Optional[torch.Tensor] = None  # [C] float in [0, 1]
    montage_type: str = "unknown"
    reference_type: str = "unknown"
    reference_channel: Optional[str] = None
    derivation_type: str = "monopolar"
    bipolar_endpoints: Optional[Sequence[Sequence[str]]] = None  # [C][2] labels

    def num_channels(self) -> int:
        return len(self.channel_names)


class ChannelEncoder(nn.Module):
    """Static per-channel encoder producing ``[C, D]`` channel embeddings."""

    def __init__(self, cfg: TAREConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab = build_name_vocab()

        self.coord_pe = FourierPositionEncoding(cfg.coord_fourier_bands, cfg.coord_fourier_scale)
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.coord_pe.out_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.coord_dim),
        )
        # Learnable fallback used when a channel has no usable coordinate.
        self.unknown_coord = nn.Parameter(torch.zeros(cfg.coord_dim))

        self.name_embed = nn.Embedding(max(cfg.name_vocab_size, len(self.vocab) + 1), cfg.name_dim)
        self.ref_embed = nn.Embedding(len(REFERENCE_TYPES), cfg.meta_dim)
        self.deriv_embed = nn.Embedding(len(DERIVATION_TYPES), cfg.meta_dim)
        self.montage_embed = nn.Embedding(len(MONTAGE_TYPES), cfg.meta_dim)
        self.quality_mlp = nn.Sequential(nn.Linear(2, cfg.meta_dim), nn.GELU(),
                                         nn.Linear(cfg.meta_dim, cfg.meta_dim))
        # Reference *position* matters for lateralised references (a single ear or
        # mastoid biases one hemisphere), so the reference channel's coordinate is
        # encoded too rather than only its categorical type.
        self.ref_coord_mlp = nn.Sequential(nn.Linear(self.coord_pe.out_dim, cfg.hidden_dim),
                                           nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.meta_dim))
        self.unknown_ref_coord = nn.Parameter(torch.zeros(cfg.meta_dim))

        self.bipolar_flag = nn.Embedding(2, cfg.meta_dim)
        self.bipolar_mlp = nn.Sequential(
            nn.Linear(3 * self.coord_pe.out_dim + cfg.meta_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.coord_dim),
        )

        meta_total = 4 * cfg.meta_dim
        if cfg.fusion_mode == "concat_mlp":
            in_dim = cfg.coord_dim + cfg.name_dim + meta_total
            self.fuse = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim), nn.GELU(),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(cfg.hidden_dim, cfg.embed_dim),
            )
        else:  # FiLM: coordinates are the trunk, metadata modulates it
            self.trunk = nn.Sequential(nn.Linear(cfg.coord_dim + cfg.name_dim, cfg.hidden_dim),
                                       nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.embed_dim))
            self.film = nn.Linear(meta_total, 2 * cfg.embed_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.out_norm = nn.LayerNorm(cfg.embed_dim)
        self._warned_unknown_coord = False

    # -- metadata -> tensors ---------------------------------------------------
    def _name_ids(self, names: Sequence[str], device) -> torch.Tensor:
        table = self.name_embed.num_embeddings
        return torch.tensor([_name_index(n, self.vocab, table) for n in names],
                            dtype=torch.long, device=device)

    @staticmethod
    def _cat_index(value: Optional[str], table: Sequence[str]) -> int:
        if not value:
            return 0
        v = value.strip().lower()
        return table.index(v) if v in table else 0

    def _coord_embedding(self, xyz: torch.Tensor) -> torch.Tensor:
        """``[C, 3] -> [C, coord_dim]`` with a learnable unknown-coordinate fallback."""
        known = xyz.norm(dim=-1) > 1e-8
        emb = self.coord_mlp(self.coord_pe(xyz))
        if not bool(known.all()):
            if self.cfg.warn_on_unknown_coord and not self._warned_unknown_coord:
                logger.warning(
                    "%d/%d channels have no 3-D coordinate; using the learnable "
                    "unknown-coordinate fallback. Spatial reasoning (and the SSL "
                    "branch) will be degraded for those channels.",
                    int((~known).sum()), known.numel(),
                )
                self._warned_unknown_coord = True
            emb = torch.where(known.unsqueeze(-1), emb,
                              self.unknown_coord.to(emb.dtype).expand_as(emb))
        return emb

    def _bipolar_embedding(self, meta: ChannelMeta, device) -> Optional[torch.Tensor]:
        """Directed two-endpoint encoding, or ``None`` if not a bipolar montage."""
        if not meta.bipolar_endpoints:
            return None
        pos_a, pos_b = [], []
        for pair in meta.bipolar_endpoints:
            a, b = (list(pair) + ["", ""])[:2]
            pos_a.append(TEMPLATE_POSITIONS.get(canonical_name(a), (0.0, 0.0, 0.0)))
            pos_b.append(TEMPLATE_POSITIONS.get(canonical_name(b), (0.0, 0.0, 0.0)))
        ra = self.coord_pe(torch.tensor(pos_a, dtype=torch.float32, device=device))
        rb = self.coord_pe(torch.tensor(pos_b, dtype=torch.float32, device=device))
        flag = self.bipolar_flag(torch.ones(len(pos_a), dtype=torch.long, device=device))
        # The signed difference is what makes (a-b) distinguishable from (b-a).
        return self.bipolar_mlp(torch.cat([ra, rb, ra - rb, flag], dim=-1))

    # -- forward ---------------------------------------------------------------
    def forward(self, meta: ChannelMeta, device=None) -> torch.Tensor:
        """``ChannelMeta -> [C, embed_dim]`` channel embeddings."""
        device = device if device is not None else self.unknown_coord.device
        C = meta.num_channels()
        xyz = meta.channel_xyz.to(device=device, dtype=torch.float32)
        assert xyz.shape == (C, 3), f"channel_xyz must be [C, 3], got {tuple(xyz.shape)}"

        if self.cfg.use_coordinates:
            coord = self._coord_embedding(xyz)
            bip = self._bipolar_embedding(meta, device)
            if bip is not None:
                coord = bip                               # endpoints replace the point encoding
        else:
            # Coordinate-free ablation: every channel shares the unknown-coordinate
            # vector, so only the remaining components can distinguish channels.
            coord = self.unknown_coord.expand(C, -1)

        name = self.name_embed(self._name_ids(meta.channel_names, device)) \
            if self.cfg.use_name_embedding \
            else self.name_embed.weight[:1].expand(C, -1) * 0.0

        ref_id = self._cat_index(meta.reference_type, REFERENCE_TYPES)
        deriv_id = self._cat_index(meta.derivation_type, DERIVATION_TYPES)
        mont_id = self._cat_index(meta.montage_type, MONTAGE_TYPES)
        ref = self.ref_embed(torch.full((C,), ref_id, dtype=torch.long, device=device))
        deriv = self.deriv_embed(torch.full((C,), deriv_id, dtype=torch.long, device=device))
        mont = self.montage_embed(torch.full((C,), mont_id, dtype=torch.long, device=device))

        if meta.reference_channel:
            p = TEMPLATE_POSITIONS.get(canonical_name(meta.reference_channel))
            if p is not None:
                rc = self.ref_coord_mlp(self.coord_pe(
                    torch.tensor(p, dtype=torch.float32, device=device))).expand(C, -1)
            else:
                rc = self.unknown_ref_coord.expand(C, -1)
        else:
            rc = self.unknown_ref_coord.expand(C, -1)
        ref = ref + rc

        mask = torch.ones(C, device=device) if meta.channel_mask is None \
            else meta.channel_mask.to(device=device, dtype=torch.float32)
        qual = torch.ones(C, device=device) if meta.channel_quality is None \
            else meta.channel_quality.to(device=device, dtype=torch.float32)
        quality = self.quality_mlp(torch.stack([mask, qual], dim=-1))

        if not self.cfg.use_reference_metadata:
            ref = torch.zeros_like(ref)
            deriv = torch.zeros_like(deriv)
            mont = torch.zeros_like(mont)
        meta_vec = torch.cat([ref, deriv, mont, quality], dim=-1)
        if self.cfg.fusion_mode == "concat_mlp":
            out = self.fuse(torch.cat([coord, name, meta_vec], dim=-1))
        else:
            h = self.trunk(torch.cat([coord, name], dim=-1))
            gamma, beta = self.film(meta_vec).chunk(2, dim=-1)
            out = h * (1.0 + gamma) + beta
        return self.out_norm(out)
