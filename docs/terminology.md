# PhysioWave terminology

This glossary is normative for the whole repository: code, comments, docstrings,
log messages, configs, README and paper text.  `tests/test_terminology.py`
enforces the parts that can be checked mechanically, and the check runs in CI.

Where a term is banned, an occurrence that *explains or forbids* it is allowed if
the line (or the two lines around it) carries the inline marker
`TERMINOLOGY-ALLOW`.  The marker exists so the prohibition can be documented in
the source without tripping its own guard.

---

## Constraint A — a statistic is not connectivity

> Any channel-by-channel matrix computed from recorded data is called
> **spatial statistics** (a **spatial statistic**, singular) or a
> **channel-relation graph**.  It is never called
> *connectivity*, *functional connectivity*, or *brain connectivity*.

| Use this | Not this |
|---|---|
| `A_dyn`, `spatial_stat_graph` | `A_conn`, `connectivity_matrix` |
| "channel-relation graph" | "functional connectivity graph" |
| "band-wise spatial covariance" | "band-wise connectivity" |
| "the statistic relates channels *i* and *j*" | "channels *i* and *j* are connected" |

**Why.**  A scalp channel-relation matrix is not a measurement of neural
interaction.  Three things enter it at once:

1. **Genuine source correlation** — the part one would like to talk about.
2. **The reference montage.**  Re-referencing subtracts one channel-linear
   combination from every channel, which adds a rank-one common term to every
   pair.  The same recording yields visibly different matrices under an original,
   average, or mastoid reference, so the matrix is a property of the *montage as
   much as of the brain*.
3. **Volume conduction.**  The skull and scalp smear each cortical generator over
   many electrodes essentially **instantaneously** at EEG frequencies.  Two
   electrodes picking up one source therefore show a large, spurious correlation
   concentrated at **zero (and π) phase lag**.  This is a direct consequence of
   the physics of the head as a volume conductor, not evidence of interaction.

Because of point 3, **ordinary magnitude coherence is never a default anywhere in
this codebase**: it is maximally sensitive to exactly the zero-phase component
volume conduction produces.  It is retained only as a *negative control* in
`physiowave/spatial/spatial_stats.py::magnitude_coherence`, where the test suite
uses it to demonstrate the failure mode.

**The robust options.**  `dyn_graph_type` accepts:

| value | what it uses | volume-conduction robust? |
|---|---|---|
| `cov` (default) | band-wise shrinkage covariance/correlation | no — cheap, stable, consistent with the CSP/Riemannian tradition |
| `wpli` | weighted phase-lag index (debiased, Vinck et al. 2011) | yes — uses only the *imaginary* part of the cross-spectrum |
| `imcoh` | imaginary part of coherency | yes — same rationale |

`wpli` is in the experiment matrix as a required control.  A measured
demonstration lives in
`tests/test_spatial.py::test_wpli_is_near_zero_for_a_zero_phase_shared_source`:
two channels driven by one shared alpha source with zero phase lag give
**wPLI = 0.0000** and **imCoh = 0.011**, while **|coherence| = 0.994** and
**correlation = 0.874**.

**Band-wise, not broadband.**  `A_dyn` is computed per frequency band by default
and the band matrices are combined with learnable weights.  A per-sample
broadband covariance of scalp EEG is dominated by whatever carries the most
amplitude — usually alpha or a low-frequency ocular drift — so it encodes "where
the biggest slow signal is" rather than the spatial structure of the other bands.
Broadband is retained as an ablation only (`dyn.band_wise: false`).

**`L_covariance` is exempt from the estimator discussion but not from the naming
rule.**  It is a *reconstruction-fidelity* term that asks the reconstruction to
preserve the input's band-wise spatial covariance.  It is unrelated to `A_dyn`
and carries no interaction claim, and it is still never described as preserving
connectivity.

---

## Constraint B — SSL and GL are different things with different names

> The spherical-spline surface Laplacian (G/H matrix method) is the
> **SSL branch (Spline Surface Laplacian)** and is the only thing in this
> repository called **strict CSD**.  The learnable graph-Laplacian branch is the
> **GL branch (Graph-Laplacian)** and may only ever be called **CSD-inspired**.

| Branch | Module | Definition | May be called |
|---|---|---|---|
| **SSL** | `physiowave/spatial/spline_laplacian.py` | Perrin et al. (1989) spherical spline surface Laplacian; `G`/`H` from electrode coordinates and spline parameters; `L_ssl = H · C · M_interp` | "surface Laplacian", "spline CSD", "strict CSD", "SSL branch" |
| **GL** | `physiowave/spatial/graph_laplacian.py` | normalised graph Laplacian of the geometric affinity, with learnable edge weights and a learnable gate | "CSD-inspired", "learnable graph Laplacian", "GL branch" — **never** "CSD" |

The two are **parallel branches**, not alternative names for one idea:
`raw only / raw+GL / raw+SSL / raw+GL+SSL` is an ablation axis in
`configs/experiments/spatial_branch_ablation.yaml`.

**Both are gated additions, never replacements.**  The surface Laplacian is a
spatial **band-pass**, not a high-pass: it sharpens local, superficial generators
and attenuates deep or widely distributed ones.  A Laplacian-only model therefore
discards real signal.  Both branches are fused as
`H = H_raw + g_gl · (L_geo X) + g_ssl · (L_ssl X)` with gates initialised at 0.1.

**When SSL is unavailable** (all conditions are logged, never silent):

* fewer than `ssl.min_channels` (default 16) usable electrodes — spline CSD is
  unreliable at low spatial sampling;
* a bipolar derivation — the surface Laplacian is defined on monopolar potentials;
* no electrode coordinates — the spline fit is undefined.

Bad or missing channels are **spherical-spline interpolated first**, then the
Laplacian is built; one bad electrode would otherwise contaminate every output
channel, since each CSD value is a weighted sum over all electrodes.

---

## Constraint C — three physical facts about the reference

1. **Re-referencing is a linear transformation of the channel axis.**
   `V' = (I − 1 wᵀ) V` for a weight vector `w` over the *recorded* channels.
   Therefore only views that can be written that way may be constructed; a
   "reference" that is not a linear combination of recorded channels describes a
   recording that was never made.  `physiowave/pretrain/reference.py` returns the
   operator `M` alongside every view, and `tests/test_reference.py` checks
   `view == M @ X` exactly.
2. **Reference invariance of a representation is therefore well posed.**  All
   legal views span the same measured field, so asking the encoder to map them to
   one representation is a physically meaningful objective, not a regulariser
   pulled out of thin air.
3. **The surface Laplacian is reference free.**  It is a second spatial
   derivative of the potential, so it annihilates the all-ones channel direction —
   exactly the direction a re-reference adds.  `L_ssl · 1 = 0` holds by
   construction, which makes the SSL view the natural **anchor** of the
   reference-consistency objective.  Measured on the 64-channel template montage,
   CAR, linked-mastoid and single-mastoid re-references change `L_ssl X` by a
   relative **5.9 × 10⁻⁷** (`tests/test_spatial.py::test_ssl_operator_is_reference_invariant`).

### View tiers

Single-sided references (one ear, one mastoid, one arbitrary channel) subtract a
signal recorded over one hemisphere from every channel, injecting a systematic
lateralisation bias.  Views are therefore tiered:

| tier | views | used for |
|---|---|---|
| `standard` | `original`, `common_average`, `linked_mastoids` | pretraining; **the default downstream evaluation input**; may be a consistency anchor |
| `hard` | `left_ear`, `right_ear`, `left_mastoid`, `right_mastoid`, `random_channel` | pretraining at `hard_view_prob` (default 0.2) and reference-robustness evaluation only; **never an anchor**, never the default eval view |

Consistency loss is logged separately per tier (`loss_ref_standard`,
`loss_ref_hard`) — averaging them hides how much harder the lateralised case is.
A common average over fewer than `car_min_channels` (default 32) electrodes is
skipped with a logged reason: an average over too few electrodes is not neutral.

---

## Constraint D — limb sEMG is not facial EMG

`emg_region` in the dataset registry must be `limb` for anything entering the
limb sEMG pretraining corpus.  `physiowave.data.registry.assert_limb_semg` raises
otherwise, and `unknown` is rejected too — an unlabelled region is not evidence of
a limb recording.

Facial EMG differs in generator anatomy (thin, overlapping mimetic muscles versus
large skeletal muscle bellies), bandwidth, amplitude range and artefact structure
(it is heavily contaminated by, and contaminates, EEG).  Mixing the two would make
"sEMG pretraining" mean two different things at once.

---

## Other naming conventions

| Term | Meaning |
|---|---|
| **WAST** | Wavelet Analysis–Synthesis Tokenizer — critically-sampled DWT → per-subband processing → IWT → projection |
| **TARE** | Topology-and-Reference-Aware channel Encoder |
| **SSL branch** | strict spline surface Laplacian (§ Constraint B) — *not* "self-supervised learning"; self-supervision is always written out in full |
| **GL branch** | learnable graph Laplacian, CSD-inspired |
| **RALF** | Reliability-Aware Latent Fusion |
| **FgM** | Frequency-guided masking |
| `A_geo` | static geometric affinity from electrode coordinates (data independent) |
| `A_dyn` | dynamic spatial-statistics graph (§ Constraint A) |
| `K` | number of latent spatial slots after channel compression (`K ≪ C`) |
| `S` | number of temporal patches; `P` patch length; `J` decomposition levels |
| `N_old` | legacy token count `(J+1)·C·S` |
| `N_new` | compressed token count `K·S` |
