# The C1 pretrained encoders: EEG, ECG and sEMG

One architecture, three independently pretrained models. This page is the
reference for the methods section: what each stage computes, what the three
models share, where they differ, and what each was trained on. Numbers come
from the runs' own `dataset_manifest.json` and `config_resolved.yaml`
(`docs/runs/<modality>_c1_moe/`).

## 1. Overview

```
window  x  [B, C, T]      one route: C electrodes at rate f, T = window seconds x f
   │
   ├─ WaveletFrontend (per route)                                  [B, C, T]
   │     SoftGateWaveletDecomp:  J = 3 levels, 5 learnable wavelets, soft gates,
   │                             cross-scale channel-mixing FFN   -> Spec(x) [B, 4C, T]
   │     ScaleFold (dynamic):    4 bands -> 1 row per channel      -> [B, C, T]
   │
   ├─ PatchEmbed (per sampling rate)   Conv2d over [B, 1, C, T], kernel (1, patch)
   │                                                               [B, C·P, D]
   ├─ + channel identity   tanh(g) · W_c · Embed(channel name)     one vector per channel
   ├─ + 2-D sinusoidal position (channel index, time patch)
   │
   ├─ shared Transformer   pre-RMSNorm, QK-norm, RoPE, SwiGLU      [B, C·P, D]
   │
   └─ pretraining only: masked tokens -> two decoders per rate
         spec head  -> the folded wavelet patch   (MSE, per-patch normalised target)
         raw  head  -> the preprocessed signal    (SmoothL1)
```

| | EEG | ECG | sEMG |
|---|---|---|---|
| Transformer width / depth / heads | 512 / 8 / 8 | 512 / 9 / 8 | 384 / 6 / 6 |
| model parameters (total; of which the Transformer) | 30.3 M; 25.2 M | 30.1 M; 28.4 M | 12.1 M; 10.6 M |
| routes | E19_256, E32_512, E64_256, E128_512 | L12_500, L1_250 | W16_2000, A24_2000, G64_2000 |
| window / patch | 4 s / 0.5 s (8 patches per channel) | 10 s / 0.5 s (20) | 1 s / 0.125 s (8) |
| wavelet bank (initialisation) | sym4, sym5, db6, sym8, db8 | db4, db6, sym4, sym8, coif2 | db2, db4, db7, sym4, sym5 |
| channel vocabulary | 334 names: 10-20 / 10-10 electrodes and bipolar derivations | 30: the 12 leads, V3R–V9, `patch1` | 234: wristband, forearm rings, HD-grid positions |
| masking | frequency-guided, 70% | the same, limb leads masked as one group | the same |
| pretraining corpus size (windows, not parameters) | 15.4 M windows (4 s: 17,100 h) | 7.6 M windows (10 s: 21,000 h) | 5.1 M windows (1 s: 1,420 h) |
| optimizer updates | 60 × 768 = 46,080 | 60 × 1000 = 60,000 | 39 × 1000 = 39,000 (epoch 38 checkpoint, see §7) |

Everything not in the table is identical across the three: 16 taps per
wavelet filter, 3 decomposition levels, dynamic ScaleFold with a 3-tap
synthesis filter and γ initialised at 0.1, a 64-dimensional channel
embedding with a gate initialised at 0, no dropout in pretraining, AdamW
(β = 0.9, 0.95) at 3e-4 with 2 warm-up epochs and cosine decay to 1%,
weight decay 0.05, gradient clipping at 1, bf16, 16 GPUs (4 nodes).

## 2. Routes: the shape a window arrives in

A **route** is a (channel count, sampling rate) pair fixed by the recording
and carried in the data; it is never predicted. Each route has its own
wavelet frontend (the filters are per channel); each sampling rate has its
own patch embedding and decoders; the channel embedding and the Transformer
are shared by every route of a modality. Because the patch is a fixed
duration, a token means the same thing on every route and the sequence
length is simply channels × patches.

| modality | route | channels | rate | window | tokens | corpora |
|---|---|---|---|---|---|---|
| EEG | E19_256 | 19 (10-20) | 256 Hz | 4 s | 152 | TUEG |
| | E32_512 | 32 | 512 | 4 s | 256 | FACED, TDBRAIN |
| | E64_256 | 64 (10-10) | 256 | 4 s | 512 | PhysioNet MI, M3CV |
| | E128_512 | 128 | 512 | 4 s | 1024 | HBN, HGD |
| ECG | L12_500 | 12 leads | 500 | 10 s | 240 | MIMIC-IV-ECG, CODE-15%, SPH, Georgia, MedalCare-XL |
| | L1_250 | 1 | 250 | 10 s | 20 | Icentia11k (chest patch), PulseDB (lead II) |
| sEMG | W16_2000 | 16 (wristband) | 2000 | 1 s | 128 | emg2pose, emg2qwerty |
| | A24_2000 | 24 (three forearm rings) | 2000 | 1 s | 192 | putEMG |
| | G64_2000 | 64 (one 8×8 or 5×13 grid) | 2000 | 1 s | 512 | Hyser, CEMHSEY |

A route's slot list is its channel axis: row *i* of every window on the
route is the electrode named at slot *i*, on every corpus. A corpus that
lacks a slot leaves the row zero with `valid_channel_mask` False; padded
slots are never masked and never scored.

## 3. The wavelet frontend

**Soft-gated wavelet decomposition** (`SoftGateWaveletDecomp`, J = 3). At
each level the current approximation goes through an **adaptive wavelet
selector**: a bank of five learnable analysis filter pairs (16 taps,
initialised from the named PyWavelets filters, per channel), mixed by
weights a small network predicts from the window (global average pool →
MLP → softmax over the five). The decomposed approximation and detail are
upsampled back to T and **soft-gated** against the undecomposed signal,
`new_a = g·a + (1−g)·up(a₂)`, with the gate predicted per channel and time
by a multi-head gate; a **cross-scale channel-mixing FFN** then updates the
(approximation, detail) pair with a learned residual scale that starts at
zero. The detail band of every level and the final approximation are
stacked: `Spec(x) ∈ [B, (J+1)·C, T]`.

**ScaleFold** (`dynamic`) collapses the four bands back to one row per
channel before patching: `fold = base + γ·(Σ_s α_s · g_s * b_s − base)`,
where `base` is the plain mean of the bands, `g_s` a 3-tap depthwise
synthesis filter per scale (delta-initialised), and the mixing weights α
are predicted per (channel, 0.5 s block) by an MLP shared across channels
from three statistics of each band -- log-RMS, temporal entropy and kurtosis,
normalised across scales. γ starts at 0.1. A KL(α ‖ uniform) term weighted
1e-3 keeps the fold from collapsing onto one band. This is differentiable
best-basis selection: the cost is learned instead of being Shannon entropy,
and the choice is soft. Because the MLP is shared across channels, the fold
carries no channel-shaped parameter and transfers to any montage.

The frontend is **convolutional and length-agnostic**: a downstream window
can be shorter or longer than the pretraining window as long as it is a
whole number of patches.

## 4. Tokens: patch, channel identity, position

`PatchEmbed` is a Conv2d over `[B, 1, C, T]` with kernel and stride
(1, patch samples), so a token is one channel's 0.5 s (sEMG: 0.125 s) of the
folded representation, and the sequence is channel-major: token `c·P + p`
is channel *c*, patch *p*.

**Channel identity (C1).** Every channel name resolves to a row of a
learned embedding table (the modality's vocabulary; unknown names map to
`<unk>`). The row is normalised, projected to the model width and added to
all of that channel's tokens, scaled by `tanh(g)` with `g` a scalar that
starts at 0 -- the path opens as training finds it useful (final values
of tanh(g): EEG 0.39, ECG 0.35, sEMG 0.36). The model is therefore told *which*
electrode a row is, by name, independently of where that row sits, and a
downstream montage can place its electrodes in any order. Downstream, a name
outside the vocabulary gets a new row of its own, seeded from `<unk>` and
trained as a montage adapter (`physiowave/eeg_c1/downstream.py`).

**Position.** A 2-D sinusoidal encoding over (channel index, time patch) is
added to the tokens, and the attention applies a rotary position embedding
(RoPE) to queries and keys on top, so relative offsets along the sequence
are available to every layer.

## 5. The Transformer

Pre-norm blocks with RMSNorm, multi-head self-attention with QK-norm and
RoPE, and a SwiGLU feed-forward network (ratio 4); a final RMSNorm. Depth,
width and heads per modality are in §1. It is shared by every route of the
modality and is what downstream fine-tuning transfers, together with the
channel embedding and its gate.

## 6. The pretraining objective

Two views of one window. The **clean** view gives the targets and chooses
the mask; the **online** view has the masked patches zeroed **in the
signal, before the frontend**, so nothing inside a masked patch reaches a
visible token through the frontend's temporal convolutions or cross-scale
attention. Masked tokens are then replaced by a learned mask token and the
Transformer encodes the sequence; two decoders per sampling rate read it:

```
L = 0.5 · MSE( spec_head, per-patch-normalised Spec patch )     masked valid tokens only
  + 0.5 · SmoothL1( raw_head, preprocessed signal patch; β = 0.5 )
  + 1e-3 · KL( fold α ‖ uniform )
```

The spectral target is the frontend's own output, detached and normalised
per patch, so the frontend is trained by the reconstruction and cannot move
the target toward the prediction. The raw head predicts the preprocessed
signal itself, which keeps the frontend from drifting to a representation
that is easy to predict but says nothing about the signal.

**Masking.** 70% of the valid tokens. Frequency-guided: a token's score is
0.6 × its spectral importance (the summed magnitude spectrum along the
sequence, interpolated to the token grid) + 0.4 × uniform noise, and the
highest scores are masked, so informative stretches are hidden more often
than quiet ones. On ECG the six limb leads of a time patch are masked as one
unit (I, II, III, aVR, aVL, aVF are six projections of one vector; with two
of them visible the rest is arithmetic) and the precordial leads per token.

## 7. The three pretraining runs

| | EEG | ECG | sEMG |
|---|---|---|---|
| corpora | TUEG 13.7 M windows, HBN 1.35 M, FACED 129 k, TDBRAIN 87 k, M3CV 82 k, PhysioNet MI 37 k, HGD 23 k | PulseDB 4.99 M, Icentia11k 1.65 M, MIMIC-IV-ECG 749 k, CODE-15% 123 k, SPH 25 k, MedalCare-XL 15 k, Georgia 10 k | emg2qwerty 2.36 M, emg2pose 1.20 M, CEMHSEY 8×8 663 k, CEMHSEY 5×13 453 k, Hyser 274 k, putEMG 153 k |
| sampling of corpora | temperature 0.5: P(d) ∝ n_d^0.5, so a large corpus is read less than once per epoch and a small one several times | the same | the same |
| per-GPU batch by route | E19 128, E32 96, E64 48, E128 24 | L12 96, L1 1024 | W16 192, A24 128, G64 48 |
| epochs × steps | 60 × 768 | 60 × 1000 | 60 × 1000 scheduled; stopped at epoch 38 |
| validation | fixed windows, fixed mask seed (a curve, not a final number) | the same | the same |
| checkpoint used downstream | lowest validation loss | lowest validation loss | end of epoch 38 |

**sEMG, epoch 38.** The run was healthy for 38 epochs (validation loss
falling, gradient norm 0.1–0.5). At epoch 39 the gradient on the G64 route
rose to 10⁴ and then overflowed; with bf16 there is no loss scaler to refuse
the step, and `clip_grad_norm_` with an infinite norm scales every gradient
by NaN. The trainer now skips non-finite or oversized steps and refuses to
resume from non-finite weights (`docs/emg_c1_pretraining.md`). The epoch-38
weights are the last written before the divergence and are what every sEMG
downstream number uses; the downstream models rebuild their frontends
anyway, so the G64 frontend, where the instability sat, is not transferred.

## 8. Preprocessing, per modality

Each corpus is converted to one physical unit, filtered, resampled to its
route's rate, placed on the route's slots by name, cut into windows and
normalised; nothing is band-pass filtered, so the encoder sees the whole
band up to Nyquist.

| | EEG | ECG | sEMG |
|---|---|---|---|
| unit | µV | mV | mV |
| filters | linear detrend; mains notch with 2 harmonics; 0.5 Hz high-pass | mains notch (2 harmonics); 0.5 Hz high-pass | mains notch (3 harmonics); 20 Hz high-pass |
| resampling | polyphase (anti-aliased) to the route rate | the same; missing limb leads derived from I and II | the same |
| normalisation | per channel: z-score, clip at 20 σ | per window: de-mean each lead, divide all by one RMS (keeps lead ratios), clip | as ECG (keeps relative electrode amplitude) |
| quality | windows with non-finite or flat channels dropped | flat lead < 5 µV, any sample > 25 mV dropped | flat < 0.5 µV, > 25 mV dropped |
| split | by subject (5–10% validation) | by subject | by subject |

Downstream data goes through the same pipeline at the same rates
(`physiowave/fusion/prep_common.py`, `ECG/ecg_downstream_prep.py`,
`EMG/emg_downstream_prep.py`), so a pretrained frontend meets the
distribution it was trained on.

## 9. What a downstream model takes

`physiowave/<modality>_c1/downstream.py` builds the same encoder for a
task's montage and loads from an exported encoder
(`scripts/export_eeg_pretrained_encoder.py`):

- always: the channel embedding, its projection and gate, and the
  Transformer (`TRANSFERABLE`);
- the wavelet frontend and patch embedding too, when the montage sits on a
  pretraining route at that route's rate (PTB-XL on L12_500, a chest ECG on
  L1_250, 62 P300 electrodes in E64_256's slots); otherwise a fresh frontend
  is built for the montage and trained;
- names outside the vocabulary get their own embedding rows (§4).

Modes: `ft` (fine-tune, with layer-wise learning-rate decay over the
Transformer on sEMG), `probe` (what came from pretraining frozen; the
montage adapters, pools and head train), `scratch` (same architecture,
nothing loaded). The multimodal fusion (`docs/fusion.md`) uses exactly these
encoders and hands each one's tokens to a 4-query summary pool.

## 10. Where things are

| | code | config | results and figures |
|---|---|---|---|
| EEG | `physiowave/eeg_c1/` (`model.py`, `routes.py`, `train.py`), `wavelet_modules.py`, `transformer_modules.py`, `channel_embedding.py` | `configs/pretrain/eeg_c1_moe.yaml`, `eeg_c1_datasets.yaml` | `docs/runs/eeg_c1_moe/` |
| ECG | `physiowave/ecg_c1/` (subclasses the EEG pretrainer: lead vocabulary, routes, lead-group masking) | `configs/pretrain/ecg_c1_moe.yaml` | `docs/runs/ecg_c1_moe/` |
| sEMG | `physiowave/emg_c1/` (electrode vocabulary, routes) | `configs/pretrain/emg_c1_moe.yaml` | `docs/runs/emg_c1_moe/` |

`docs/ecg_c1_pretraining.md` and `docs/emg_c1_pretraining.md` give the
corpora, licences and the differences from EEG in detail;
`docs/figure_brief_architecture.md` describes the architecture figure.
