"""The model must accept the channel counts the datasets actually have.

Two attention blocks in the wavelet front-end split a width taken from the
electrode count rather than from a power-of-two embedding dimension, against a
hardcoded four heads. That silently restricts the model to montages whose
channel count is a multiple of four -- which the 16-channel Myo and 12-lead ECG
happen to satisfy, so the limitation stayed invisible until a 14-channel sEMG
array reached it. 19-channel 10-20 EEG and 22-channel BCI IV-2a would have hit
it too.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pywt")

from model import BERTWaveletTransformer          # noqa: E402
from wavelet_modules import heads_dividing        # noqa: E402


# (name, channels, window, classes) taken from the converters and launch scripts
REAL_LAYOUTS = [
    ("db6_semg", 14, 512, 8),
    ("db5_semg", 16, 256, 53),
    ("epn612_semg", 8, 1024, 6),
    ("ecg_12lead", 12, 2048, 5),
    ("eeg_1020", 19, 512, 2),
    ("bci_iv_2a", 22, 512, 4),
]


@pytest.mark.parametrize("width,expected", [
    (8, 4), (12, 4), (16, 4), (64, 4),      # already divisible
    (14, 2), (22, 2),                        # fall back to two
    (13, 1), (19, 1),                        # prime: fall back to one
    (3, 3), (1, 1),                          # narrower than the request
])
def test_heads_divide_the_width(width, expected):
    h = heads_dividing(width, 4)
    assert h == expected
    assert width % h == 0


@pytest.mark.parametrize("name,channels,window,classes", REAL_LAYOUTS,
                         ids=[r[0] for r in REAL_LAYOUTS])
def test_forward_accepts_real_channel_counts(name, channels, window, classes):
    model = BERTWaveletTransformer(
        in_channels=channels, max_level=3, wave_kernel_size=16,
        wavelet_names=["sym4", "db6"], use_separate_channel=True,
        patch_size=(1, 64), embed_dim=128, depth=2, num_heads=8, mlp_ratio=4.0,
        dropout=0.1, use_pos_embed=True, pos_embed_type="2d",
        task_type="classification", num_classes=classes,
        head_config={"hidden_dims": [64], "dropout": 0.1, "pooling": "mean"},
        pooling="mean",
    ).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, channels, window), task="classify")
    assert logits.shape == (2, classes)
