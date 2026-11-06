"""Shared pytest fixtures and a repo-root import path."""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from physiowave.channels.tare import ChannelMeta  # noqa: E402
from physiowave.data.montages import montage  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def montage_19():
    return montage("standard_1020_19")


@pytest.fixture
def montage_64():
    return montage("standard_1010_64")


@pytest.fixture
def meta_64(montage_64):
    names, xyz = montage_64
    return ChannelMeta(channel_names=names, channel_xyz=xyz,
                       montage_type="standard_1010", reference_type="original")


@pytest.fixture
def small_encoder():
    """A small full-featured encoder, fast enough for unit tests."""
    from physiowave.models.encoder import EncoderConfig, PhysioWaveEncoder
    from physiowave.wavelet.wast import WASTConfig

    cfg = EncoderConfig(modality="eeg", embed_dim=32,
                        wast=WASTConfig(patch_size=32, embed_dim=32, level=2))
    cfg.backbone.depth = 1
    cfg.backbone.num_heads = 4
    cfg.backbone.slot_heads = 2
    cfg.compression.num_queries = 6
    cfg.compression.num_heads = 2
    return PhysioWaveEncoder(cfg).eval()
