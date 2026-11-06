# Prompt: PhysioWave EEG architecture figure (PNG + editable PPTX)

You are making the main architecture figure for a paper submitted to IEEE
TPAMI.

The code is public. Work from branch **`fix/channel-identity-and-block-variants`**
of https://github.com/ForeverBlue816/PhysioWave-T, NOT `main`.

**Inputs in the repository:**

- **Old figure**: [`fig/model.png`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/fig/model.png). It shows an earlier
  version of the model. Use it only for the general idea; it is wrong in the
  ways listed in section 6.
- **Data panels**: SVGs drawn from the trained checkpoint on one real EEG
  window (route E64_256, 64 channels at 256 Hz), without axes.
  - Use [`docs/runs/eeg_c1_moe/architecture_panels_bare/E64_256/`](https://github.com/ForeverBlue816/PhysioWave-T/tree/fix/channel-identity-and-block-variants/docs/runs/eeg_c1_moe/architecture_panels_bare/E64_256).
  - If that folder is not there yet, use
    [`architecture_panels/E64_256/`](https://github.com/ForeverBlue816/PhysioWave-T/tree/fix/channel-identity-and-block-variants/docs/runs/eeg_c1_moe/architecture_panels/E64_256)
    and drop the tick labels.
  - `panels.json` in the folder says which window and channels they show.
  - The checkpoint is not public, so you cannot redraw them: use these.
- **Reference figures for style**: [`docs/runs/ecg_c1_moe/figures/`](https://github.com/ForeverBlue816/PhysioWave-T/tree/fix/channel-identity-and-block-variants/docs/runs/ecg_c1_moe/figures).
  Match their palette and typography.

**Code to check the description below against.** If this prompt and the code
disagree, the code wins; list any discrepancy you find in your reply.

| what | file |
|---|---|
| model forward: two views, masking, heads, loss terms | [`physiowave/eeg_c1/model.py`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/physiowave/eeg_c1/model.py) (`MultiRouteEEGPretrainer.forward`, `WaveletFrontend`, `_mask_scores`) |
| wavelet selector, decomposition, ScaleFold | [`wavelet_modules.py`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/wavelet_modules.py) (`AdaptiveWaveletSelector`, `SoftGateWaveletDecomp`, `ScaleFold`) |
| C1 channel embedding | [`channel_embedding.py`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/channel_embedding.py) (`ChannelEncoder`) |
| routes and datasets | [`physiowave/eeg_c1/routes.py`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/physiowave/eeg_c1/routes.py) |
| hyper-parameters actually used | [`configs/pretrain/eeg_c1_moe.yaml`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/configs/pretrain/eeg_c1_moe.yaml) |
| reconstruction heads | [`head_modules.py`](https://github.com/ForeverBlue816/PhysioWave-T/blob/fix/channel-identity-and-block-variants/head_modules.py) (`ReconstructionHead`) |

Produce a clean, uncluttered and accurate figure. Use only what this prompt
and the code contain; do not invent components.

## 1. Deliverables

- **`physiowave_architecture.pptx`**: one slide, fully editable.
  - Every box, arrow, label and symbol is a native PowerPoint object: shape,
    connector or text box.
  - Group each module's objects together, so a module can be moved as one
    unit.
  - The data panels are pictures. Convert each SVG to a transparent PNG at
    600 dpi (e.g. `cairosvg in.svg -o out.png --dpi 600`) and insert it at
    its true aspect ratio.
  - Build the slide with python-pptx (or equivalent) so it can be
    regenerated, and include the script.
- **`physiowave_architecture.png`**: the same slide at 600 dpi, white
  background. Export via PDF (LibreOffice headless → PDF →
  `pdftoppm -r 600 -png`), not as a screenshot.
- Optionally, `physiowave_architecture.pdf`, vector, with fonts embedded.

## 2. Canvas and typography

- Slide size: 18.2 cm × 10.0 cm (TPAMI double-column width, 7.16 in). It
  must read clearly at exactly that size in print.
- Margins: 0.25 cm. Lay modules out on an even grid with equal gaps, aligned
  centres, and one left-to-right reading line.
- Font: **Times New Roman** everywhere.
  - Module titles: bold, 8–9 pt.
  - Annotations and tensor shapes: regular, 7 pt.
  - Minimum size: 6.5 pt.
- Math as Unicode italic: d⁽¹⁾ d⁽²⁾ d⁽³⁾ a⁽³⁾, α, γ, ℒ, C×T, (J+1)C×T,
  CP×D, ↓2, ↑2, ⊕. Do not use images of equations.

## 3. Visual style

- **Palette.** Purple, orange and blue on white; no other hues.

  | role | outline / text | fill |
  |---|---|---|
  | data path (signal, patches, tokens) | blue `#2166AC` | `#D1E5F0` |
  | wavelet frontend and the mask | purple `#5E3C99` | `#E7E1EF` |
  | learning and prediction (encoder, heads, predictions, loss) | orange `#E66101` | `#FDE0C5` |
  | arrows and text | `#4D4D4D` | — |
  | blank / masked cells | — | `#EDEDED` |

- **Modules.** Rounded rectangles, corner radius ≈ 0.08 cm, 0.75 pt outline,
  flat fill.
  - No gradients, shadows, glow, 3-D or clip-art.
  - Use icons only where noted: wavelet glyphs, ⊕, a lookup-table glyph, a
    padlock for stop-gradient.
- **Arrows.** 0.75 pt, small solid heads, orthogonal elbow connectors, no
  crossings.
  - Dashed: stop-gradient (detached target).
  - Dotted orange: loss (training only).
- **Tensor shapes.** Small grey italic labels beside the arrows.
- **Stages.** Four very light background panels (`#F7F7F7`, no outline),
  each with a bold label at its top left: **(a) Input & route**,
  **(b) Route-specific wavelet frontend**,
  **(c) Two views & frequency-guided masking**,
  **(d) Shared encoder & dual reconstruction**.

## 4. Where the data panels go (from the panel folder above)

| file | meaning | place |
|---|---|---|
| `01_raw_input.svg` | the EEG window x (3 of 64 channels shown) | (a), input |
| `H1_raw_target.svg` | the same window, all 64 channels × 4 s | (a), behind or beside the input as the "C×T" view |
| `02_mask_grid.svg` / `H0_mask_grid_all.svg` | which 0.5 s patches are masked (purple) | (c), mask |
| `03_masked_input.svg` | masked patches zeroed (shaded) | (c), entering the online frontend |
| `04_wavelet_band_1_d1.svg` … `04_wavelet_band_4_approx.svg` | the four decomposition outputs d⁽¹⁾ d⁽²⁾ d⁽³⁾ a⁽³⁾ | (b), decomposition outputs, stacked |
| `05_folded_spec.svg` | the ScaleFold output | (b), after the fold |
| `H5_spec_target.svg` | spec target (folded wavelet, per-patch normalised) | (c)/(d), the spec target |
| `H7_spec_composite.svg` | spec prediction (visible = target, masked = prediction) | (d), spec head output |
| `H3_raw_composite.svg` | raw prediction, the same way | (d), raw head output |
| `09_raw_overlay.svg` | raw target (blue) vs prediction (orange) on masked spans | optional inset at the loss |

- Keep each panel's aspect ratio and give panels of the same kind the same
  size.
- The panels carry no text. All labels belong to the slide.

## 5. The model

### (a) Input and route

- The input is one EEG window x ∈ ℝ^{C×T}, 4 s long. EEG comes in four
  acquisition shapes, called routes:
  - E19_256: 19 channels, 256 Hz (TUEG)
  - E32_512: 32 channels, 512 Hz (FACED, TDBrain)
  - E64_256: 64 channels, 256 Hz (PhysioNet-MI, M3CV)
  - E128_512: 128 channels, 512 Hz (HBN, HGD)
- The route is determined by the recording's montage and sampling rate. It
  is NOT a learned router or gate, so draw no gating network.
- Draw the input choosing one of four stacked, offset purple boxes labelled
  "route-specific frontend ×4", with E64_256 highlighted.

### (b) Route-specific wavelet frontend (purple)

1. **Adaptive Wavelet Selector.**
   - A bank of 5 learnable 16-tap wavelet filter pairs, initialised as sym4,
     sym5, db6, sym8 and db8.
   - Global average pooling → MLP → softmax gives mixing weights.
   - The output is one adaptive low-/high-pass pair per channel.
   - Draw five tiny wavelet glyphs feeding a softmax mixer.
2. **Learnable decomposition, J = 3 levels.** Each level:
   - low/high-pass filtering with ↓2;
   - ↑2 back to length T;
   - adaptive gating g·x + (1−g)·↑(·);
   - a Cross-Scale CAFFN that attends to earlier levels;
   - outputs the detail band d⁽ˡ⁾ and passes the approximation on.

   **Draw ONE level in detail with a "×J" recurrence bracket.** Do not repeat
   it three times; that repetition is what crowded the old figure.

   The output is Spec(x) = [d⁽¹⁾, d⁽²⁾, d⁽³⁾, a⁽³⁾], of shape (J+1)C×T. Place
   the four band panels here.
3. **Dynamic ScaleFold** (new; make it prominent).
   - An MLP shared across channels reads each (channel, 0.5 s patch) block's
     band statistics and predicts softmax weights α over the 4 scales.
   - Each scale gets a 3-tap depthwise synthesis filter.
   - Output: b̄ + γ·(Σₛ αₛ·bₛ − b̄), where b̄ is the scale mean and γ is a
     learned gate.
   - It folds (J+1)C×T back to C×T, one row per electrode, so the token
     count stays C·P rather than (J+1)·C·P.
   - Draw the four bands funnelling through a Σ labelled α into one C×T map,
     the `05_folded_spec` panel.
   - Add a small note: "KL(α ‖ uniform) → loss".

### (c) Two views and frequency-guided masking

The same frontend runs twice on the same window.

- **Clean view** (upper lane).
  - x → frontend → folded spec.
  - It gives two targets, both detached (dashed arrow + padlock,
    "stop-grad"):
    - the spec target: per-patch-normalised folded-wavelet patches;
    - the raw target: the waveform patches.
- **Mask selection**, on the clean tokens.
  - Importance comes from the FFT magnitude along the token sequence, mixed
    with random noise (ratio 0.6).
  - The top **70%** of tokens are masked.
  - Draw a thin importance heat strip feeding the mask grid.
- **Online view** (lower lane).
  - The mask is applied to the SIGNAL before the frontend: masked 0.5 s
    patches are zeroed (`03_masked_input`), then pass through the same
    frontend.
  - Label it "mask before frontend". It stops a masked patch from leaking
    into visible tokens through the frontend's convolutions.

### (d) Shared encoder and dual reconstruction (orange)

1. **Patch embedding**, one per sampling rate.
   - A 0.5 s patch is 128 samples at 256 Hz and 256 samples at 512 Hz.
   - It is embedded linearly to D = 512, giving C·P tokens (P = 8 per
     channel).
2. **C1 channel-identity embedding.**
   - A learned lookup of the electrode NAME (Fp1, Cz, …), 64-dimensional,
     projected to D.
   - It is added to every token of that channel, scaled by tanh(gate), with
     the gate initialised at 0.
   - Draw a lookup-table glyph, a ⊕ and a small gate symbol.
3. **[MASK] token.** Masked positions are replaced by one learnable token.
4. **Shared Transformer encoder**, used by all four routes.
   - 8 layers, width 512, 8 heads.
   - Draw one block with "×8" and four small tags: RMSNorm · QK-norm ·
     SwiGLU · RoPE.
   - Label it "shared across routes".
5. **Two lightweight MLP heads, one pair per sampling rate. NOT a
   Transformer decoder.**
   - The spec head outputs `H7_spec_composite`.
   - The raw head outputs `H3_raw_composite`.
6. **Loss on masked tokens only** (dotted orange arrows from each prediction
   to its detached target):

   ℒ = 0.5·MSE_spec + 0.5·SmoothL1_raw (β = 0.5) + 0.001·KL_fold

7. **Downstream.** A small bracket under (b) and (d): "kept for fine-tuning:
   frontend + embeddings + encoder; heads discarded".

## 6. Errors in the old model.png that must not reappear

- It shows a "Transformer Decoder". There is none: there are two per-rate MLP
  heads, spec and raw.
- It shows one reconstruction target. There are two (spec and raw), and the
  loss is the one above.
- It patchifies the bands directly as a (J+1)C grid. Now ScaleFold folds
  them back to C rows first.
- It masks tokens after the frontend. Now the mask is chosen on the clean
  view and applied to the signal before the online frontend, and the targets
  come detached from the clean view.
- It has no routes, no C1 channel embedding and no [MASK] token.
- It repeats the decomposition three times. Draw it once, with ×J.
- Its legend typo "Rotraty" should be "Rotary position embedding (RoPE)".
- Its raster screenshots are replaced by the repository's panels.

## 7. Legend

Bottom right, compact, 6.5–7 pt:

↓2 / ↑2: down- / up-sampling; d⁽ˡ⁾, a⁽ˡ⁾: detail and approximation bands at
level l; α: ScaleFold weights; ⊕: addition; dashed: stop-gradient; dotted:
training-only loss.

## 8. Before delivering, check

- [ ] Every component in section 5 is present, and nothing else is.
- [ ] Opened at 18.2 cm wide, all text is ≥ 6.5 pt, nothing overlaps, and
      every arrow's direction is clear.
- [ ] Only the palette in section 3 is used; no gradients or shadows.
- [ ] Every box and label in the PPTX is editable; the panels sit at 600 dpi.
- [ ] The PNG is 600 dpi (≈ 4300 × 2360 px) with a white background.
- [ ] The generating script is included.
- [ ] Any place where this prompt and the code disagree is listed.
