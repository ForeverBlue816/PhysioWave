# The learned wavelet front ends

Selector weights and effective analysis filters of the three C1 encoders, from up to 2000 val windows per corpus (levels 2-3 on the first 256). Figure: `figures/fig_wavelet.pdf` (`scripts/visualize_wavelet_frontend.py`).

## Checkpoints and tensor matching

| modality | checkpoint | epoch | route | front-end tensors matched | selector |
|---|---|---|---|---|---|
| EEG | `pretrain_eeg_c1_moe/best.pth` | 58 | E19_256 | 74/74 | 14/14 |
| EEG | `pretrain_eeg_c1_moe/best.pth` | 58 | E32_512 | 74/74 | 14/14 |
| EEG | `pretrain_eeg_c1_moe/best.pth` | 58 | E64_256 | 74/74 | 14/14 |
| EEG | `pretrain_eeg_c1_moe/best.pth` | 58 | E128_512 | 74/74 | 14/14 |
| ECG | `pretrain_ecg_c1_moe/best.pth` | 33 | L12_500 | 74/74 | 14/14 |
| ECG | `pretrain_ecg_c1_moe/best.pth` | 33 | L1_250 | 74/74 | 14/14 |
| sEMG | `pretrain_emg_c1_moe/epoch38.pth` | 39 | W16_2000 | 74/74 | 14/14 |
| sEMG | `pretrain_emg_c1_moe/epoch38.pth` | 39 | A24_2000 | 74/74 | 14/14 |
| sEMG | `pretrain_emg_c1_moe/epoch38.pth` | 39 | G64_2000 | 74/74 | 14/14 |

## Windows used per corpus

| modality | corpus | route | windows (level 1) | windows (levels 2-3) |
|---|---|---|---|---|
| EEG | FACED | E32_512 | 2000 | 256 |
| EEG | HBN | E128_512 | 2000 | 256 |
| EEG | HGD | E128_512 | 2000 | 256 |
| EEG | M3CV | E64_256 | 2000 | 256 |
| EEG | PhysioNet-MI | E64_256 | 2000 | 256 |
| EEG | TDBRAIN | E32_512 | 2000 | 256 |
| EEG | TUEG | E19_256 | 2000 | 256 |
| ECG | CODE-15% | L12_500 | 2000 | 256 |
| ECG | Georgia | L12_500 | 525 | 256 |
| ECG | Icentia11k | L1_250 | 2000 | 256 |
| ECG | MedalCare-XL | L12_500 | 1380 | 256 |
| ECG | MIMIC-IV-ECG | L12_500 | 2000 | 256 |
| ECG | PulseDB | L1_250 | 2000 | 256 |
| ECG | SPH | L12_500 | 1301 | 256 |
| sEMG | CEMHSEY 5×13 | G64_2000 | 2000 | 256 |
| sEMG | CEMHSEY 8×8 | G64_2000 | 2000 | 256 |
| sEMG | emg2pose | W16_2000 | 2000 | 256 |
| sEMG | emg2qwerty | W16_2000 | 2000 | 256 |
| sEMG | Hyser | G64_2000 | 2000 | 256 |
| sEMG | putEMG | A24_2000 | 2000 | 256 |

## Mean selector weights, level 1 (corpus × basis)

**EEG** (bank: sym4, sym5, db6, sym8, db8)

| corpus | sym4 | sym5 | db6 | sym8 | db8 | max |w − 1/M| |
|---|---|---|---|---|---|---|
| FACED | 0.175 | 0.371 | 0.126 | 0.227 | 0.101 | 0.171 |
| HBN | 0.269 | 0.351 | 0.097 | 0.232 | 0.052 | 0.151 |
| HGD | 0.269 | 0.351 | 0.097 | 0.232 | 0.052 | 0.151 |
| M3CV | 0.344 | 0.113 | 0.118 | 0.343 | 0.082 | 0.144 |
| PhysioNet-MI | 0.344 | 0.113 | 0.118 | 0.343 | 0.082 | 0.144 |
| TDBRAIN | 0.175 | 0.371 | 0.126 | 0.227 | 0.101 | 0.171 |
| TUEG | 0.935 | 0.001 | 0.003 | 0.059 | 0.002 | 0.735 |

Levels 2 and 3, per route (mean over the route's corpora):

| route | level | sym4 | sym5 | db6 | sym8 | db8 |
|---|---|---|---|---|---|---|
| E19_256 | 2 | 0.073 | 0.924 | 0.000 | 0.002 | 0.000 |
| E19_256 | 3 | 0.053 | 0.946 | 0.000 | 0.001 | 0.000 |
| E32_512 | 2 | 0.129 | 0.531 | 0.087 | 0.198 | 0.056 |
| E32_512 | 3 | 0.100 | 0.634 | 0.060 | 0.174 | 0.033 |
| E64_256 | 2 | 0.001 | 0.991 | 0.000 | 0.008 | 0.000 |
| E64_256 | 3 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| E128_512 | 2 | 0.008 | 0.308 | 0.632 | 0.052 | 0.000 |
| E128_512 | 3 | 0.001 | 0.852 | 0.143 | 0.003 | 0.000 |

**ECG** (bank: db4, db6, sym4, sym8, coif2)

| corpus | db4 | db6 | sym4 | sym8 | coif2 | max |w − 1/M| |
|---|---|---|---|---|---|---|
| CODE-15% | 0.114 | 0.082 | 0.149 | 0.160 | 0.495 | 0.295 |
| Georgia | 0.114 | 0.082 | 0.149 | 0.160 | 0.495 | 0.295 |
| Icentia11k | 0.182 | 0.203 | 0.217 | 0.213 | 0.184 | 0.018 |
| MedalCare-XL | 0.114 | 0.082 | 0.149 | 0.160 | 0.495 | 0.295 |
| MIMIC-IV-ECG | 0.114 | 0.082 | 0.149 | 0.160 | 0.495 | 0.295 |
| PulseDB | 0.182 | 0.203 | 0.217 | 0.213 | 0.184 | 0.018 |
| SPH | 0.114 | 0.082 | 0.149 | 0.160 | 0.495 | 0.295 |

Levels 2 and 3, per route (mean over the route's corpora):

| route | level | db4 | db6 | sym4 | sym8 | coif2 |
|---|---|---|---|---|---|---|
| L12_500 | 2 | 0.083 | 0.052 | 0.118 | 0.131 | 0.616 |
| L12_500 | 3 | 0.082 | 0.052 | 0.118 | 0.131 | 0.617 |
| L1_250 | 2 | 0.182 | 0.204 | 0.217 | 0.213 | 0.184 |
| L1_250 | 3 | 0.182 | 0.204 | 0.217 | 0.213 | 0.185 |

**sEMG** (bank: db2, db4, db7, sym4, sym5)

| corpus | db2 | db4 | db7 | sym4 | sym5 | max |w − 1/M| |
|---|---|---|---|---|---|---|
| CEMHSEY 5×13 | 0.346 | 0.000 | 0.000 | 0.206 | 0.448 | 0.248 |
| CEMHSEY 8×8 | 0.346 | 0.000 | 0.000 | 0.206 | 0.448 | 0.248 |
| emg2pose | 0.270 | 0.057 | 0.054 | 0.310 | 0.309 | 0.146 |
| emg2qwerty | 0.269 | 0.057 | 0.054 | 0.310 | 0.309 | 0.146 |
| Hyser | 0.346 | 0.000 | 0.000 | 0.206 | 0.448 | 0.248 |
| putEMG | 0.245 | 0.082 | 0.100 | 0.279 | 0.294 | 0.118 |

Levels 2 and 3, per route (mean over the route's corpora):

| route | level | db2 | db4 | db7 | sym4 | sym5 |
|---|---|---|---|---|---|---|
| W16_2000 | 2 | 0.130 | 0.006 | 0.005 | 0.504 | 0.355 |
| W16_2000 | 3 | 0.003 | 0.000 | 0.000 | 0.000 | 0.997 |
| A24_2000 | 2 | 0.184 | 0.001 | 0.001 | 0.132 | 0.682 |
| A24_2000 | 3 | 0.080 | 0.000 | 0.000 | 0.037 | 0.883 |
| G64_2000 | 2 | 0.000 | 0.000 | 0.000 | 0.929 | 0.071 |
| G64_2000 | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Is the selector input-dependent?

The selector's input is the per-channel mean of the window (global average pooling over time). Windows are normalised per channel, so that mean is near zero and the softmax can only vary as far as the residual means do. Below, the across-window standard deviation of the level-1 weights and the mean |window mean| the selector actually saw.

| modality | corpus | std of weights across windows (per basis) | mean |window mean| |
|---|---|---|---|
| EEG | FACED | 0.0000, 0.0001, 0.0001, 0.0000, 0.0000 | 0.0000 |
| EEG | HBN | 0.0002, 0.0000, 0.0001, 0.0001, 0.0000 | 0.0000 |
| EEG | HGD | 0.0003, 0.0000, 0.0001, 0.0002, 0.0000 | 0.0000 |
| EEG | M3CV | 0.0003, 0.0000, 0.0001, 0.0003, 0.0000 | 0.0000 |
| EEG | PhysioNet-MI | 0.0003, 0.0000, 0.0001, 0.0003, 0.0000 | 0.0000 |
| EEG | TDBRAIN | 0.0000, 0.0002, 0.0001, 0.0000, 0.0000 | 0.0000 |
| EEG | TUEG | 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 | 0.0000 |
| ECG | CODE-15% | 0.0002, 0.0002, 0.0001, 0.0002, 0.0004 | 0.0001 |
| ECG | Georgia | 0.0000, 0.0000, 0.0000, 0.0002, 0.0000 | 0.0000 |
| ECG | Icentia11k | 0.0000, 0.0001, 0.0000, 0.0000, 0.0001 | 0.0000 |
| ECG | MedalCare-XL | 0.0000, 0.0000, 0.0000, 0.0002, 0.0000 | 0.0000 |
| ECG | MIMIC-IV-ECG | 0.0001, 0.0000, 0.0000, 0.0002, 0.0000 | 0.0000 |
| ECG | PulseDB | 0.0000, 0.0001, 0.0000, 0.0000, 0.0001 | 0.0000 |
| ECG | SPH | 0.0000, 0.0000, 0.0000, 0.0002, 0.0000 | 0.0000 |
| sEMG | CEMHSEY 5×13 | 0.0035, 0.0000, 0.0000, 0.0045, 0.0077 | 0.0001 |
| sEMG | CEMHSEY 8×8 | 0.0028, 0.0000, 0.0000, 0.0034, 0.0058 | 0.0002 |
| sEMG | emg2pose | 0.0005, 0.0002, 0.0003, 0.0008, 0.0009 | 0.0001 |
| sEMG | emg2qwerty | 0.0002, 0.0000, 0.0001, 0.0001, 0.0001 | 0.0000 |
| sEMG | Hyser | 0.0010, 0.0000, 0.0000, 0.0008, 0.0018 | 0.0000 |
| sEMG | putEMG | 0.0000, 0.0000, 0.0001, 0.0000, 0.0000 | 0.0000 |

## −3 dB cut-offs (Hz)

Low-pass: the learned effective filter (bank mixed with the level's mean weights), cut-off of the channel-mean response, with the 10-90th percentile across channels; initialisation is the bank mixed uniformly. NOTE: each orthogonal wavelet's own low-pass is half-band (−3 dB at fs/4), but the uniform mix of five different wavelets is not -- their phases do not line up and the sum cancels at mid frequencies -- so the initialisation's cut-off sits well below fs/4. The per-basis table that follows separates a change of mixture from a change of the filters themselves. High-pass: the rising −3 dB point.

| route | fs | init LP | learned LP L1 (p10–p90) | L2 | L3 | init HP | learned HP L1 | tap RMS change / init RMS (per basis) |
|---|---|---|---|---|---|---|---|---|
| E19_256 | 256 | 12.0 | 56.1 (14.2–63.9) | 64.1 | 64.3 | 115.3 | 63.0 | 0.31, 0.22, 0.30, 0.30, 0.31 |
| E32_512 | 512 | 24.0 | 35.3 (32.1–43.6) | 120.7 | 123.1 | 230.6 | 129.2 | 0.08, 0.07, 0.07, 0.08, 0.08 |
| E64_256 | 256 | 12.0 | 13.0 (10.0–62.5) | 59.5 | 59.7 | 115.3 | 111.6 | 0.26, 0.18, 0.21, 0.27, 0.21 |
| E128_512 | 512 | 24.0 | 55.4 (35.7–121.9) | 36.2 | 56.6 | 230.6 | 137.8 | 0.32, 0.30, 0.32, 0.32, 0.36 |
| L12_500 | 500 | 36.3 | 57.5 (56.1–58.3) | 60.6 | 60.6 | 213.7 | 198.9 | 0.31, 0.33, 0.29, 0.29, 0.29 |
| L1_250 | 250 | 18.1 | 16.0 (16.0–16.0) | 15.9 | 15.9 | 106.9 | 107.0 | 0.12, 0.12, 0.11, 0.11, 0.12 |
| W16_2000 | 2000 | 124.8 | 355.9 (107.7–396.0) | 387.5 | 445.7 | 551.6 | 265.8 | 0.28, 0.38, 0.42, 0.28, 0.25 |
| A24_2000 | 2000 | 124.8 | 112.0 (78.1–239.8) | 357.2 | 386.8 | 551.6 | 719.4 | 0.20, 0.18, 0.18, 0.20, 0.19 |
| G64_2000 | 2000 | 124.8 | 418.3 (361.0–462.2) | 433.4 | 431.4 | 551.6 | nan | 0.51, 0.66, 0.59, 0.63, 0.47 |

Per basis, the low-pass cut-off of each wavelet's own filter (channel mean), initial → learned:

**EEG**

| route | sym4 | sym5 | db6 | sym8 | db8 |
|---|---|---|---|---|---|
| E19_256 | 64.0 → 47.4 | 64.0 → 62.6 | 64.0 → 51.4 | 64.0 → 47.3 | 64.0 → 44.6 |
| E32_512 | 128.0 → 119.9 | 128.0 → 120.5 | 128.0 → 119.3 | 128.0 → 120.2 | 128.0 → 119.8 |
| E64_256 | 64.0 → 48.9 | 64.0 → 52.1 | 64.0 → 43.4 | 64.0 → 48.4 | 64.0 → 39.0 |
| E128_512 | 128.0 → 77.1 | 128.0 → 91.6 | 128.0 → 99.8 | 128.0 → 80.5 | 128.0 → 88.0 |

**ECG**

| route | db4 | db6 | sym4 | sym8 | coif2 |
|---|---|---|---|---|---|
| L12_500 | 125.0 → 105.9 | 125.0 → 122.4 | 125.0 → 100.6 | 125.0 → 104.9 | 125.0 → 100.5 |
| L1_250 | 62.5 → 62.6 | 62.5 → 62.3 | 62.5 → 62.0 | 62.5 → 61.9 | 62.5 → 63.0 |

**sEMG**

| route | db2 | db4 | db7 | sym4 | sym5 |
|---|---|---|---|---|---|
| W16_2000 | 500.0 → 381.6 | 500.0 → 288.5 | 500.0 → 194.8 | 500.0 → 380.0 | 500.0 → 412.0 |
| A24_2000 | 500.0 → 297.1 | 500.0 → 319.6 | 500.0 → 311.6 | 500.0 → 307.8 | 500.0 → 309.1 |
| G64_2000 | 500.0 → 377.8 | 500.0 → 252.5 | 500.0 → 299.9 | 500.0 → 417.2 | 500.0 → 412.9 |

## What does not show the expected pattern

- ECG / Icentia11k: level-1 selector weights within 0.03 of uniform (max deviation 0.018).
- ECG / PulseDB: level-1 selector weights within 0.03 of uniform (max deviation 0.018).
- EEG / FACED: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / HBN: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / HGD: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / M3CV: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / PhysioNet-MI: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / TDBRAIN: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / TUEG: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG / TUEG: the selector has collapsed onto one basis (sym4, 0.935).
- ECG / CODE-15%: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / Georgia: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / Icentia11k: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / MedalCare-XL: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / MIMIC-IV-ECG: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / PulseDB: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- ECG / SPH: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- sEMG / emg2pose: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- sEMG / emg2qwerty: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- sEMG / putEMG: the level-1 selector weights do not vary across windows (std < 1e-3): on normalised windows the selector's input is ~0 and its output is a learned per-route constant, not an input-adaptive choice.
- EEG: fs ratio 2.00, learned cut-off ratio 1.31 -- the cut-offs are closer to a shared physical frequency than to fs/4.
- ECG: fs ratio 2.00, learned cut-off ratio 3.61 -- the cut-offs scale with the sampling rate (not a shared physical frequency).

The automatic checks above use fixed thresholds (uniform within 0.03; cut-off within 5% and taps within 5% RMS of initialisation; cut-off ratio against sampling-rate ratio). They flag, they do not interpret.
