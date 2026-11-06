# Earlier extension design (superseded)

> **Superseded design notes.** This section of the old README documented an
> earlier extension design (the WAST/TARE path: SSL/GL spatial branches,
> `A_dyn`, `physiowave/models`). That code is kept and still tested, but it is
> **not** what the PhysioWave-T paper reports; the paper's C1 encoders are in
> `physiowave/{eeg,ecg,emg}_c1` and described in
> [c1_pretraining_architecture.md](c1_pretraining_architecture.md).

> The sections above describe the original NeurIPS 2025 PhysioWave and are
> unchanged. Everything below documents the TPAMI extension: a token-efficient,
> topology- and reference-aware framework covering **EEG**, **ECG** and
> **limb surface EMG**. The legacy code path is untouched and still runs; every
> new capability is behind a config switch.

## Table of contents

- [What changed and why](#what-changed-and-why)
- [Architecture and tensor shapes](#architecture-and-tensor-shapes)
- [Installation](#installation-tpami)
- [One-command entry point](#one-command-entry-point)
- [Configuration](#configuration)
- [EEG conventions: reference and coordinates](#eeg-conventions-reference-and-coordinates)
- [SSL vs GL: two spatial branches](#ssl-vs-gl-two-spatial-branches)
- [`A_dyn`: what it is and what it is not](#a_dyn-what-it-is-and-what-it-is-not)
- [Limb sEMG vs facial EMG](#limb-semg-vs-facial-emg)
- [Data preparation](#data-preparation-tpami)
- [Single GPU, multi-GPU and Slurm](#single-gpu-multi-gpu-and-slurm)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Reproducing the experiment matrix](#reproducing-the-experiment-matrix)
- [Testing](#testing)
- [Common errors](#common-errors)

---

## What changed and why

The original model's wavelet stage upsamples every decomposition band back to the
full time axis and concatenates them on the channel axis, producing a token
sequence of length

```
N_old = (J + 1) · C · S
```

so the sequence grows linearly with the number of decomposition levels *and* with
the electrode count. On a 64-channel montage with `J = 3` and `S = 16` that is
4096 tokens — and on this machine the legacy path runs out of memory before it
can produce them (a measured `20.32 GiB` allocation at `C=64, T=1024`, recorded
in `results/benchmark_tokens.json` rather than hidden).

The extension replaces that with:

| Module | What it does | Token effect |
|---|---|---|
| **WAST** (`physiowave/wavelet/`) | Wavelet **A**nalysis–**S**ynthesis **T**okenizer: critically-sampled DWT → per-subband depthwise conv + gate → inverse DWT → projection | `C · S`, independent of `J` |
| **TARE** (`physiowave/channels/tare.py`) | Topology-and-**R**eference-**A**ware channel encoder: 3-D coordinates (primary) fused with names, reference and derivation metadata | — |
| **Channel compression** (`physiowave/channels/compression.py`) | `K` learnable queries, each with a scalp anchor, cross-attend over channels | `K · S`, independent of `C` |
| **SSL / GL branches** (`physiowave/spatial/`) | strict spline surface Laplacian + CSD-inspired learnable graph Laplacian, both gated onto the raw branch | — |
| **RALF** (`physiowave/models/fusion.py`) | Reliability-**A**ware **L**atent **F**usion of EEG + ECG + sEMG summary tokens | few tokens per modality |

```
N_new = K · S
```

Measured on this machine (`results/tables/token_efficiency.md`), at `C=64`,
`T=2048`, `J=3`:

| variant | tokens | vs. legacy |
|---|---|---|
| legacy | 8192 | 1× (OOM before it ran) |
| WAST (no compression) | 2048 | 4× |
| WAST + TARE, `K=32` | 1024 | 8× |
| WAST + TARE, `K=16` (EEG default) | 512 | **16×** |
| WAST + TARE, `K=8` | 256 | 32× |
| WAST + TARE, `K=4` | 128 | 64× |

**Why `K ≪ C` is defensible, not just cheap.** Volume conduction low-pass filters
the potential field between cortex and scalp, so neighbouring electrodes measure
strongly overlapping mixtures. The *effective spatial degrees of freedom* of a
scalp recording are therefore far fewer than the electrode count — the same fact
that lets classical pipelines work with a handful of CSP or ICA components. A
small set of topology-aware queries, each anchored at a learnable scalp location,
retains that information while making the token count independent of the montage:
a 19-channel clinical recording and a 64-channel HD recording both produce `K · S`
tokens, which is what makes montage transfer possible at all.

---

## Architecture and tensor shapes

```
input                      [B, C, T]     raw signal, channel-first
  │
  ├─ SpatialFrontend                     EEG only; ECG/sEMG skip the SSL branch
  │    X       = X_raw  +  g_gl·(L_geo X)  +  g_ssl·(L_ssl X)      [B, C, T]
  │    A       = λ_g·A_geo  +  λ_d·A_dyn   (detached)              [B, C, C]
  │
  ├─ WAST                                                          tokenizer
  │    patchify                                     [B, C, S, P]   S = T / P
  │    flatten + critically-sampled DWT             [B·C·S, P]  →  Σ|coeffs| = P
  │       cA_J:[P/2^J]  cD_J:[P/2^J]  …  cD_1:[P/2]
  │    per-subband depthwise conv + norm + gate     (lengths preserved)
  │    inverse DWT                                  [B·C·S, P]
  │    project (+ band log-energy features)         [B, C, S, D]
  │
  ├─ TARE                                channel metadata → [C, D]
  │
  ├─ ChannelCompressor                   K learnable scalp-anchored queries
  │    query·channel attention + distance bias + graph bias from A
  │                                                  [B, K, S, D]   K ≪ C
  │
  ├─ FactorizedBackbone                  time attention over S, then slot mixing
  │                                                  [B, K, S, D]
  │       cost O(B·K·S²) + O(B·S·K²), never O((K·S)²)
  │
  └─ summary attention                               [B, n_summary, D]
       → pooled [B, D], logits [B, n_classes], quality [B]
```

Every stage asserts its shapes; the assertions carry the expected tuple in their
message so a mismatch names the offending axis.

**Critical sampling.** For a patch of length `P`, the `J`-level analysis returns
exactly `P` coefficients (`P/2^J` approximation plus `P/2^j` detail for
`j = J…1`). `physiowave/wavelet/dwt.py::dwt` asserts this on every call.

**Boundary handling.** Naively padding a patch, filtering and cropping back to
`P/2` coefficients destroys information — the cropped coefficients are exactly
the ones carrying the padded samples, and the resulting square operator is
numerically singular (condition number ~10¹⁷ measured for `db4`). The
implementation instead uses the classical symmetric-extension filter bank, the
same construction JPEG2000 uses: extend the patch `P → 2P` with the boundary
rule, run the exactly-critical periodic filter bank, keep the first half of each
subband. That step is invertible only for **symmetric (biorthogonal)** analysis
filters, so:

| wavelet | `reflect` | `periodization` |
|---|---|---|
| `bior4.4` (default) | cond 3.2 – 42 ✅ | cond 1.3 – 1.5 ✅ |
| `rbio4.4` | cond 2.2 – 12 ✅ | cond 1.3 – 1.5 ✅ |
| `db4`, `sym4`, `coif*` | **singular** — auto-falls back, with a warning | cond 1.0 ✅ |

Default is `wavelet: bior4.4` with `boundary_mode: reflect`. **`zero` is never
the default**: zero extension asserts the signal drops to 0 just outside the
patch, which for a stationary physiological signal with any baseline drift is a
step discontinuity — and because the DWT is applied *per patch*, that artefact
recurs `S` times per channel instead of twice per recording. Measured on a
drifting signal (`tests/test_wavelet.py::test_boundary_analysis_artifact`), the
patch-edge deviation from an infinite-context transform is **0.019 for `reflect`
vs 0.120 for `zero`**.

---

## Installation (TPAMI)

```bash
conda create -n physiowave python=3.11 && conda activate physiowave
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pytest                      # for the test suite
```

No new heavyweight dependency was introduced: the extension uses `torch`,
`numpy`, `scipy`, `PyWavelets`, `h5py` and `PyYAML`, all already in
`requirements.txt`. `omegaconf` is used if present but is not required.

`requirements.txt` also lists `transformers`, `accelerate`, `datasets`,
`evaluate`, `peft`, `matplotlib` and `seaborn`, which nothing in the repository
imports. On a cluster whose PyTorch comes from a module, skip them: `accelerate`
and `peft` depend on `torch`, and pip will pull its own wheel over the tuned
build. See `docs/cineca_setup.md`.

---

## One-command entry point

```bash
bash scripts/run_tpami.sh smoke                              # ~1 min, synthetic, CPU
bash scripts/run_tpami.sh pretrain --modality eeg
bash scripts/run_tpami.sh pretrain --modality ecg
bash scripts/run_tpami.sh pretrain --modality semg
bash scripts/run_tpami.sh fusion   --config ralf
bash scripts/run_tpami.sh eval      --suite eeg
bash scripts/run_tpami.sh benchmark --suite tokens
bash scripts/run_tpami.sh benchmark --suite multimodal
bash scripts/run_tpami.sh experiments --tier 1               # main-table ablations
bash scripts/run_tpami.sh report                             # CSV / Markdown / LaTeX
bash scripts/run_tpami.sh all --dry-run                      # validate everything
```

The script uses `set -euo pipefail` and reports the failing stage by name.
Nothing is hardcoded to a cluster; override through the environment:

```bash
PYTHON=/path/to/python NUM_GPUS=4 OUTPUT_ROOT=/scratch/$USER/physiowave \
EXTRA="--set train.epochs=100 data.datasets=[siena] data.roots.siena=/data/siena" \
bash scripts/run_tpami.sh pretrain --modality eeg
```

`NUM_GPUS` is taken from the environment, else `SLURM_GPUS_ON_NODE`, else
`nvidia-smi`, else CPU. With more than one GPU the script launches `torchrun`
automatically.

`smoke` runs on synthetic data and covers forward, backward, checkpointing,
resume, fusion, evaluation, **SSL cache build *and* hit**, and the token
benchmark.

---

## Configuration

Plain YAML under `configs/`, composed through a `defaults:` list, with dotted
CLI overrides. Unknown keys raise rather than being ignored — a silently dropped
config key is a reproducibility bug.

```
configs/
├── base.yaml                              training / data defaults
├── model/legacy.yaml                      the original model, unchanged
├── model/wast.yaml                        tokenizer only
├── model/wast_tare.yaml                   full model
├── pretrain/{eeg,ecg,semg}.yaml           per-modality pretraining
├── fusion/ralf.yaml                       multimodal fusion
└── experiments/
    ├── channel_ablation.yaml              tier 1 — main table
    ├── spatial_branch_ablation.yaml       tier 2 — SSL/GL, A_dyn, anchor, K
    ├── token_efficiency.yaml              tier 2 — tokens / FLOPs / memory
    ├── reference_robustness.yaml          tier 3 — reference and channel robustness
    └── multimodal_robustness.yaml         tier 3 — RALF under missing/corrupted input
```

```bash
python -m physiowave.train.pretrain_main --config pretrain/eeg \
  --set model.compression.num_queries=8 \
        model.spatial.dyn.dyn_graph_type=wpli \
        pretrain.ref_consistency.anchor=pairwise \
        train.epochs=50
```

The fully resolved config, the environment versions and the git commit are saved
next to every run's checkpoints.

---

## EEG conventions: reference and coordinates

Three physical facts drive every reference-related design decision
(full statement in [`docs/terminology.md`](terminology.md)):

1. **Re-referencing is a linear transformation of the channel axis**,
   `V' = (I − 1 wᵀ)V` for a weight vector `w` over *recorded* channels. Only views
   expressible that way may be constructed — anything else describes a recording
   that was never made. Every view returns its operator `M`, and the test suite
   checks `view == M @ X` exactly.
2. **Reference invariance of a representation is therefore well posed**: all legal
   views span the same measured field.
3. **The surface Laplacian is reference free.** It is a second spatial derivative
   of the potential, so it annihilates the all-ones channel direction — exactly
   what a re-reference adds. `L_ssl · 1 = 0` holds by construction, which makes
   the SSL view the natural **anchor** for reference consistency. Measured:
   CAR, linked-mastoid and single-mastoid re-references change `L_ssl X` by a
   relative **5.9 × 10⁻⁷**.

**View tiers.** Single-sided references (one ear, one mastoid, one arbitrary
channel) subtract a signal recorded over one hemisphere from every channel,
injecting a systematic lateralisation bias. So:

| tier | views | where used |
|---|---|---|
| `standard` | `original`, `common_average`, `linked_mastoids` | pretraining; **the default downstream eval input**; may be an anchor |
| `hard` | `left_ear`, `right_ear`, `left_mastoid`, `right_mastoid`, `random_channel` | pretraining at `hard_view_prob` (default 0.2); reference-robustness eval only. **Never an anchor.** |

`loss_ref_standard` and `loss_ref_hard` are logged separately — averaging them
hides how much harder the lateralised case is. A common average over fewer than
`car_min_channels` (default 32) electrodes is skipped with a logged reason.

**Coordinates.** Channels carry `channel_xyz` (unit-sphere scalp coordinates).
Template 10-20 / 10-10 positions are constructed from the *definition* of those
systems (`physiowave/data/montages.py`) rather than copied from a table; real
digitised positions are always preferable and can be supplied per recording. A
channel with no coordinate routes through a learnable unknown-coordinate fallback
and logs a warning; a dataset with no coordinates has the SSL branch disabled.

---

## SSL vs GL: two spatial branches

These are **parallel branches with different names and different status**.
Never used interchangeably (enforced by `tests/test_terminology.py`).

| | **SSL branch** | **GL branch** |
|---|---|---|
| Module | `physiowave/spatial/spline_laplacian.py` | `physiowave/spatial/graph_laplacian.py` |
| Method | Perrin et al. (1989) spherical spline surface Laplacian; `G`/`H` from electrode coordinates and spline parameters | normalised graph Laplacian of the geometric affinity, learnable edge weights |
| May be called | "surface Laplacian", "spline CSD", **"strict CSD"** | **"CSD-inspired"** only — *never* "CSD" |
| Data dependent? | No — `L_ssl` is a fixed `[C, C]` operator per montage, precomputed and cached | trainable |
| Reference invariant? | **Yes**, exactly (`L_ssl · 1 = 0`) | no guarantee |

Because `G` and `H` depend only on electrode coordinates and the spline
parameters `(m, λ, n_legendre)`, the whole surface Laplacian collapses to one
fixed linear operator, cached per `(montage_hash, m, λ, n_legendre)` and applied
in the forward pass as a single `[C,C] @ [C,T]` matmul. That is what promotes it
from an offline preprocessing step to a first-class branch.

**Both are gated additions, never replacements:**

```
H = H_raw + g_gl · (L_geo X) + g_ssl · (L_ssl X)        gates initialised at 0.1
```

The surface Laplacian is a spatial **band-pass**, not a high-pass: it sharpens
local, superficial generators and attenuates deep or widely distributed ones. A
Laplacian-only model would therefore discard real signal.

**SSL degrades gracefully, and always logs the reason:**

- fewer than `ssl.min_channels` (default 16) usable electrodes → skipped (spline
  CSD is unreliable at low spatial sampling), and the reference-consistency anchor
  automatically falls back from `ssl` to `pairwise`;
- a bipolar derivation → skipped (the surface Laplacian is defined on monopolar
  potentials);
- no coordinates → skipped;
- **bad/missing channels are spherical-spline interpolated *first*, then the
  Laplacian is built.** One bad electrode would otherwise contaminate every output
  channel, since each CSD value is a weighted sum over all electrodes. The bad
  channel's column in `L_ssl` is exactly zero, and the test suite verifies that
  corrupting it changes nothing.

Ablation axis: `raw / raw+GL / raw+SSL / raw+GL+SSL` in
`configs/experiments/spatial_branch_ablation.yaml`.

---

## `A_dyn`: what it is and what it is not

> `A_dyn` is a **spatial statistic of the recorded signals** — a
> **channel-relation graph**. It is contaminated by the reference montage and by
> volume conduction and must never be interpreted or described as functional or
> brain connectivity. `tests/test_terminology.py` greps the core package to
> enforce this.

Three things enter any scalp channel-relation matrix at once: genuine source
correlation; the **reference**, which adds a rank-one common term to every pair
so the matrix is a property of the montage as much as of the brain; and
**volume conduction**, which spreads each generator over many electrodes
essentially *instantaneously* at EEG frequencies, producing large spurious
correlation concentrated at **zero (and π) phase**.

**Defaults and options** (`model.spatial.dyn`):

| option | estimator | volume-conduction robust? |
|---|---|---|
| `cov` *(default)* | band-wise shrinkage covariance/correlation (Ledoit–Wolf) | no — cheap, stable, consistent with the CSP/Riemannian tradition |
| `wpli` | debiased weighted phase-lag index (Vinck et al. 2011) | **yes** — uses only the imaginary part of the cross-spectrum |
| `imcoh` | imaginary part of coherency | **yes** |

Ordinary magnitude coherence is **not offered as a default anywhere**: it is
maximally sensitive to exactly the zero-phase component volume conduction
produces, and it shifts with the reference. It is kept only as a *negative
control* used by the tests. Measured on two channels driven by one shared alpha
source with zero phase lag:

```
wPLI = 0.0000   imCoh = 0.0111   |coherence| = 0.9942   correlation = 0.8744
```

and on a genuinely phase-lagged pair, `wPLI = 1.0000`.

**Band-wise, not broadband.** A per-sample broadband covariance of scalp EEG is
dominated by whatever carries the most amplitude — usually alpha or a
low-frequency ocular drift — so it encodes "where the biggest slow signal is"
rather than the spatial structure of the other bands. Band matrices (δ/θ/α/β/γ by
default) are combined with learnable weights; broadband is an ablation
(`dyn.band_wise: false`).

`dyn_graph_input: {raw, ssl}` chooses whether the statistic is computed on the raw
signal or on the SSL-transformed signal, which is far less affected by reference
and volume conduction. Both are in the ablation matrix.

`A_dyn` is **detached**: it enters the model as an attention bias and as graph
structure, never as a differentiable feature.

---

## Limb sEMG vs facial EMG

This framework's sEMG modality means **limb / skeletal surface EMG**. Facial EMG
differs in generator anatomy (thin, overlapping mimetic muscles versus large
skeletal muscle bellies), bandwidth, amplitude range and artefact structure — and
it both contaminates and is contaminated by EEG. Mixing the two would make "sEMG
pretraining" mean two different things at once.

The dataset registry carries an `emg_region` field, and
`physiowave.data.registry.assert_limb_semg` is called before sEMG pretraining
starts. It rejects `facial`, `trunk` **and `unknown`** — an unlabelled region is
not evidence of a limb recording.

---

## Data preparation (TPAMI)

Every dataset yields the same `Sample` schema
(`physiowave/data/schema.py`): `signal`, `modality`, `sampling_rate`,
`subject_id`, `recording_id`, `dataset_id`, `channel_names`, `channel_xyz`,
`channel_mask`, `channel_quality`, `montage_type`, `reference_type`,
`reference_channel`, `derivation_type`, `bipolar_endpoints`, `label`,
`window_start`/`window_end`, `emg_region`.

Registered corpora (`physiowave/data/registry.py`) — **nothing is downloaded
automatically**; datasets behind a data use agreement must be obtained separately
and pointed at with `data.roots.<id>`:

| modality | datasets |
|---|---|
| EEG pretraining | `tueg`†, `siena` |
| EEG downstream | `tuab`†, `tuar`†, `tusl`†, `bci_iv_2a` (motor imagery) |
| Multimodal | `mpdb`, `seizeit2`† |
| ECG | `mimic_iv_ecg`, `ptbxl`, `cpsc2018`, `shaoxing` |
| limb sEMG | `ninapro_db6`, `epn612` |
| Smoke tests | `synthetic_{eeg,ecg,semg}` |

† requires a signed data use agreement.

```bash
python -m physiowave.train.pretrain_main --config pretrain/eeg --dry-run \
  --set data.datasets=[siena] data.roots.siena=/data/siena
```

The pipeline builds a manifest (with checksums and statistics), splits it
**subject-wise**, and runs a programmatic leakage check that **raises** —
not warns — if any subject or recording appears in two splits. Random
segment-level splits are never used for subject-level tasks.

Configurable preprocessing: resample, notch, band-pass, z-score/min-max/max-abs,
with a content-addressed on-disk cache; multi-corpus weighted sampling mixes
datasets at a configured ratio regardless of their relative sizes.

---

## Single GPU, multi-GPU and Slurm

```bash
# Single GPU / CPU / Apple MPS
python -m physiowave.train.pretrain_main --config pretrain/eeg

# Multi-GPU, one node
torchrun --standalone --nproc_per_node=4 \
  -m physiowave.train.pretrain_main --config pretrain/eeg

# Multi-node on CINECA Leonardo (or any Slurm cluster)
sbatch --export=ALL,MODALITY=eeg,EPOCHS=100 scripts/slurm/cineca_pretrain.sbatch
sbatch --export=ALL,EEG_CKPT=/path/best.pth scripts/slurm/cineca_fusion.sbatch
```

The Slurm templates follow the CINECA Leonardo conventions (`boost_usr_prod`,
`module load profile/deeplrn cineca-ai`, `srun … torchrun --rdzv_backend=c10d`).
Modules, virtualenv and every path come from `scripts/cineca_env.sh`, which the
launchers source and which falls back to repository-relative paths off-cluster,
so **local runs do not depend on Slurm at all**. Override any of it through
`--export`: `PW_CKPT_ROOT`, `PW_DATA_EEG`, `DATASETS`, `PW_VENV`, `PROJECT_DIR`,
`EPOCHS`, `BATCH_SIZE`, `PRECISION`, `EXTRA`. Full runbook: `docs/cineca_setup.md`.

Precision is requested as `bf16` and degrades loudly: bf16 → fp16 on pre-Ampere
GPUs → fp32 on CPU/MPS, with the substitution logged.

Logged every `train.log_every` steps: each loss term separately, learning rate,
`samples/sec`, `tokens/sec`, peak memory, and the token compression ratio.
FLOPs are measured in the benchmark suite. NaN/Inf are checked every step and
skip the update rather than poisoning the optimizer state.

---

## Checkpoints and resuming

```bash
python -m physiowave.train.pretrain_main --config pretrain/eeg --resume auto
python -m physiowave.train.pretrain_main --config pretrain/eeg --resume path/to/epoch_0007.pth
```

Each checkpoint holds the model, optimizer, scheduler, AMP scaler, **all RNG
states** (Python, NumPy, torch, CUDA), the resolved config, the environment
versions and the git commit. Writes are atomic (`.tmp` → `os.replace`), so a
crash mid-write cannot leave a truncated checkpoint.

**Key mismatches are never silently ignored.** `migrate_state_dict` applies the
declared renames and reports every remapped, missing and unexpected key; loading
raises unless the caller explicitly passes `strict=False` (used when loading a
pretrained encoder into a model that has an extra head).

Old PhysioWave checkpoints load into the legacy model unchanged.

---

## Reproducing the experiment matrix

Three priority tiers; tier 1 is the main paper table and is completed first when
compute is short.

**Tier 1 — channel/spatial encoding ladder** (`channel_ablation.yaml`):
legacy → legacy + channel ID → WAST → + 3-D coordinates → + reference metadata →
+ TARE geometry (`A_geo`) → + `A_dyn` (band-wise shrinkage cov) → full
(SSL + GL).

**Tier 2 — spatial branch and statistics ablations** (`spatial_branch_ablation.yaml`,
`token_efficiency.yaml`): `raw / raw+GL / raw+SSL / raw+GL+SSL`; `A_dyn` type
(band-wise cov / broadband cov / wPLI / imCoh); `A_dyn` input (raw / SSL);
consistency anchor (SSL / pairwise); `K ∈ {4, 8, 16, 32}`.

**Tier 3 — multimodal and robustness** (`multimodal_robustness.yaml`,
`reference_robustness.yaml`): original fusion vs RALF; complete / missing /
corrupted modalities; channel robustness (permutation, missing, bad channel,
unknown montage, 19↔64 transfer); reference robustness (original, CAR, mastoid,
**single-ear lateralised hard view**, bipolar, offline spline-CSD input).

```bash
bash scripts/run_tpami.sh experiments --tier 1
bash scripts/run_tpami.sh report
```

Outputs land in `results/`: raw JSON per run, then CSV, Markdown and LaTeX
tables with `mean ± std` across seeds, parameter counts, FLOPs, peak memory,
throughput, token counts and performance–efficiency Pareto data. **Rows produced
on the synthetic corpus keep a `synthetic=True` flag all the way into the
rendered table** so a smoke-test number can never be mistaken for a result.

---

## Testing

```bash
pytest tests/ -q                 # 151 tests, ~7 s on CPU
pytest tests/ -q -s -k boundary  # print the measured numbers
```

| file | covers |
|---|---|
| `test_wavelet.py` | DWT/IWT round trip, critical-sampling length conservation, boundary artefacts (`reflect` must pass, `zero` recorded), wavelet gradients, token compression |
| `test_spatial.py` | SSL reference invariance, cache hits, bad-channel interpolation, low-density and bipolar skips, wPLI on zero-phase synthetic data, shrinkage stability, band-wise `A_dyn` |
| `test_channels.py` | coordinates-only operation, permutation equivariance, reference distinguishability, bipolar endpoint order, masked-channel handling, query specialisation |
| `test_reference.py` | every view is a channel-linear map, CAR density guard, view tiering, anchor stop-gradient |
| `test_legacy.py` | the original forward shapes are unchanged |
| `test_checkpoint.py` | save/resume, RNG restore, explicit key-mismatch failure |
| `test_fusion.py` | every modality subset, per-item masks, all-missing raises, reliability learning |
| `test_data.py` | montage geometry, schema, subject-wise splits, leakage detection, preprocessing, registry |
| `test_terminology.py` | constraints A/B/D (the "connectivity" grep, SSL vs GL naming, limb sEMG) |
| `test_end_to_end.py` | three encoders, variable montages, the objective, smoke train + resume, dry run |

---

## Common errors

| Message | Cause and fix |
|---|---|
| `signal length T=… must be a multiple of patch_size=…` | crop or pad the window in the data layer, or change `model.wast.patch_size` |
| `patch_size=… must be divisible by 2**level=…` | reduce `model.wast.level` or raise `patch_size` |
| `Wavelet 'db4' has non-symmetric … falling back to periodization` | expected: orthogonal wavelets cannot use `reflect` with critical sampling. Use `bior4.4`/`rbio4.4` to keep `reflect` |
| `SSL branch skipped: N usable electrodes < min_channels=16` | montage too sparse for spline CSD; the anchor falls back to `pairwise` automatically |
| `SSL skipped: … input is a bipolar derivation` | expected: the surface Laplacian is defined on monopolar potentials |
| `common_average skipped: N good channels < car_min_channels` | expected on sparse montages; lower `car_min_channels` only if you can justify it |
| `Checkpoint key mismatch: … MISSING … UNEXPECTED …` | the config does not match the checkpoint's architecture. Fix the config, or pass `strict=False` for a deliberate partial load |
| `subject overlap between 'train' and 'test'` | the split leaked. Set `subject_from_path` in the registry so subject ids are extracted correctly |
| `limb sEMG pretraining received non-limb EMG datasets` | set `emg_region='limb'` only for limb/skeletal recordings |
| `RALF received no available modality for batch items […]` | at least one modality must be present per sample; check the modality mask |
| `Unknown config keys for …` | a typo in a YAML key or `--set` override; the valid keys are listed in the message |
| `the legacy CrossScaleCAFFN needs an even channel count` | an original limitation: the legacy path cannot run 19-channel montages at all |
| legacy `Invalid buffer size: 20.32 GiB` | the legacy `(J+1)·C·S` sequence at `C=64`; this is the behaviour WAST removes |

---

## Layout

```
physiowave/
├── wavelet/       dwt.py (critically-sampled DWT/IWT)  wast.py  fgm.py
├── spatial/       spline_laplacian.py (SSL)  graph_laplacian.py (GL)
│                  spatial_stats.py (A_dyn)  geometry.py  branches.py
├── channels/      tare.py  compression.py
├── models/        encoder.py  backbone.py  fusion.py (RALF)
│                  build.py  checkpoint.py  legacy.py
├── pretrain/      objectives.py  losses.py  reference.py  corruption.py
├── data/          schema.py  registry.py  manifest.py  splits.py
│                  preprocess.py  datasets.py  montages.py  synthetic.py
├── train/         pretrain_main.py  fusion_main.py  evaluate.py
│                  benchmark.py  data_builder.py  utils.py
├── experiments/   runner.py  report.py
└── config.py

configs/   scripts/run_tpami.sh   scripts/slurm/   tests/   docs/terminology.md
```

The original top-level modules (`model.py`, `wavelet_modules.py`,
`transformer_modules.py`, `head_modules.py`, `dataset.py`, `pretrain.py`,
`finetune.py`, `ECG/`, `EMG/`) are **unchanged** and remain the legacy entry
points.
