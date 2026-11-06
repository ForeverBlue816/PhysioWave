# Brief: redraw the PhysioWave EEG architecture figure (fig/model.png)

You are redrawing the main architecture figure of a TPAMI submission. The
current figure, `fig/model.png`, describes an earlier single-route version of
the model and is wrong in several places (listed under "What changed"). Redraw
it for the current EEG model, described below. The description is
authoritative: it is taken from the code (`physiowave/eeg_c1/model.py`,
`wavelet_modules.py`, `configs/pretrain/eeg_c1_moe.yaml`). Do not add any
component that is not described here.

## Deliverable

- One vector figure: `fig/model_v2.svg` and `fig/model_v2.pdf`, plus a 600 dpi
  `fig/model_v2.png` preview.
- Width: 7.16 in, the TPAMI double-column width. Height: 3.6 to 4.2 in. It must
  stay legible when printed at that size.
- Font: Times New Roman for all text. Module names are bold, 8 to 9 pt. Tensor
  shapes and annotations are regular, 7 pt. Math is in italic serif, e.g.
  $d^{(l)}$, $a^{(J)}$, $\alpha$, $\mathcal{L}$. Nothing smaller than 6.5 pt.
- Embed the fonts in the PDF (TrueType / Type 42, not Type 3).
- Build it with code or an editable vector tool (TikZ, matplotlib + SVG
  assembly, or draw.io exported to SVG), so it can be revised. A hand-painted
  raster is not acceptable.

## Visual language

- Palette: purple, orange and blue on white, matching the paper's data
  figures.
  - Blue `#2166AC` (light fill `#D1E5F0`): the signal and the data path
    (input, patches, tokens).
  - Purple `#5E3C99` (light fill `#E7E1EF`): the wavelet frontend (selector,
    decomposition, ScaleFold) and the mask.
  - Orange `#E66101` (light fill `#FDE0C5`): learning and prediction (encoder,
    heads, predictions, the loss).
  - Neutral grey `#4D4D4D` for arrows and text; `#EDEDED` for masked or blank
    cells.
  - Use no other hues. No gradient fills, no drop shadows, no 3-D effects.
- Modules: rounded rectangles (corner radius about 2 pt) with a 0.6 pt
  outline in the module's dark colour and its light colour as the fill.
- Arrows: 0.6 to 0.8 pt, with small solid heads. Run them orthogonally (no
  diagonals) and avoid crossings.
  - Dashed arrows for detached / stop-gradient paths (the targets).
  - Dotted arrows for the training-only loss.
- Show tensor shapes in small grey italics beside the arrows, e.g.
  $C\times T$, $(J{+}1)C\times T$, $CP\times D$.
- Group the stages with very light background panels, each with a bold
  panel label in its top-left corner: (a), (b), (c), (d).
- Lay it out on one horizontal reading line, left to right, with an even
  grid. Align module centres and leave equal gaps.

## Data panels (use the real ones)

Every signal, wavelet band, mask and reconstruction drawn inside the figure
is a real panel from a trained checkpoint, not a sketch. Produce them with:

    python scripts/draw_architecture_panels.py \
        --run-dir $PW_FAST_ROOT/runs/pretrain_eeg_c1_moe \
        --route E64_256 --out-dir fig/panels --bare

and place the SVGs from `fig/panels/E64_256/`:

| panel | where it goes |
|---|---|
| `01_raw_input.svg` | the input window, panel (a) |
| `02_mask_grid.svg` or `H0_mask_grid_all.svg` | the patch mask, panel (b) |
| `03_masked_input.svg` | the masked signal entering the online frontend |
| `04_wavelet_band_1_d1` … `04_wavelet_band_4_approx` | the four outputs of the decomposition: $d^{(1)}, d^{(2)}, d^{(3)}, a^{(3)}$ |
| `05_folded_spec.svg` | the ScaleFold output |
| `H5_spec_target.svg`, `H1_raw_target.svg` | the two targets |
| `H7_spec_composite.svg`, `H3_raw_composite.svg` | the two predictions, shown as composites |

Scale panels uniformly and keep their aspect ratios. Panels carry no text of
their own; every label belongs to the figure.

## The model, stage by stage

### (a) Input and route

- The input is one EEG window $x \in \mathbb{R}^{C\times T}$, 4 s long. EEG
  arrives in four acquisition shapes, called routes:

  | route | channels | rate | datasets |
  |---|---|---|---|
  | E19_256 | 19 | 256 Hz | TUEG |
  | E32_512 | 32 | 512 Hz | FACED, TDBrain |
  | E64_256 | 64 | 256 Hz | PhysioNet-MI, M3CV |
  | E128_512 | 128 | 512 Hz | HBN, HGD |

- The route is decided by the recording's montage and sampling rate. It is
  NOT a learned router or gate, so do not draw a gating network. Every batch
  belongs to a single route.
- Draw this as one input selecting one of four parallel route-specific
  frontends. Show the four as a short stack of identical, offset purple
  boxes labelled "route-specific frontend ×4", with E64_256 highlighted.

### (b) Route-specific wavelet frontend (one per route, purple)

1. **Adaptive Wavelet Selector.**
   - A bank of five learnable wavelet filter pairs (16 taps), initialised
     from sym4, sym5, db6, sym8 and db8.
   - Selection weights come from global average pooling of the input followed
     by an MLP and a softmax.
   - The output is one adaptive low-pass / high-pass pair per channel.
   - Draw five tiny wavelet glyphs feeding a softmax mixer that outputs the
     filter pair.
2. **Learnable multi-level decomposition, $J=3$ levels.** At each level $l$:
   - low/high-pass filtering with ↓2;
   - nearest ↑2 back to length $T$;
   - adaptive gating, $g\cdot x + (1-g)\cdot\uparrow(\cdot)$;
   - a Cross-Scale CAFFN that attends to the previous levels' features;
   - outputs: the detail band $d^{(l)}$ and the approximation passed on.

   The result is $\mathrm{Spec}(x) = [d^{(1)}, d^{(2)}, d^{(3)}, a^{(3)}]$,
   of shape $(J{+}1)C \times T$.

   Draw ONE level in detail and indicate "×J" (J = 3) with a recurrence
   bracket. Do not repeat the block three times as the old figure does. That
   repetition is what made the old figure crowded.
3. **Dynamic ScaleFold**, which is new and must be prominent.
   - An MLP shared across channels reads each (channel, patch) block's band
     statistics and predicts weights $\alpha \in \Delta^{J}$ over the four
     scales via a softmax.
   - A short depthwise synthesis filter (3 taps) is applied per scale.
   - The fold is $\bar b + \gamma\,(\sum_s \alpha_s b_s - \bar b)$, where
     $\bar b$ is the scale mean and $\gamma$ is a learned residual gate.
   - It folds $(J{+}1)C \times T$ back to $C \times T$, one row per electrode.
     This keeps the token count at $C\cdot P$ instead of $(J{+}1)\cdot C\cdot P$.
   - A KL regulariser $\mathrm{KL}(\alpha\,\|\,\mathcal{U})$ enters the loss.
   - Draw the four bands converging through a funnel or $\Sigma$ symbol
     labelled with $\alpha$ into one $C\times T$ map.

### (c) Two views and frequency-guided masking

The training step runs the frontend twice on the same window:

1. **Clean view** (top lane): $x \to$ frontend $\to$ folded spec. It provides
   both targets, each detached (dashed arrow, "stop-grad"):
   - the spec target: folded-wavelet patches, normalised per patch;
   - the raw target: the preprocessed waveform patches.
2. **Mask selection, on the clean tokens.**
   - Frequency-guided masking: an importance score from the FFT magnitude
     along the token sequence, mixed with random noise (importance ratio
     0.6).
   - The top 70% of tokens are masked.
   - Draw a small importance heat strip feeding the mask grid.
3. **Online view** (bottom lane).
   - The mask is applied to the SIGNAL before the frontend: masked 0.5 s
     patches are zeroed.
   - The masked signal then goes through the same frontend.
   - Make "mask before the frontend" explicit, because it is a design choice:
     it prevents information inside a masked patch from leaking to visible
     tokens through the frontend's convolutions.

### (d) Tokens, shared encoder, two heads (orange)

1. **Patch embedding, one per sampling rate.**
   - A patch is 0.5 s: 128 samples at 256 Hz, 256 samples at 512 Hz.
   - The embedding is linear to $D = 512$.
   - The result is $C\cdot P$ tokens; $P = 8$ patches per channel.
2. **C1 channel-identity embedding.**
   - A learned embedding of the electrode NAME (e.g. "Fp1", "Cz"; a shared
     vocabulary), dimension 64, projected to $D$.
   - It is added to every token of that channel, scaled by
     $\tanh(\text{gate})$, with the gate initialised at 0.
   - Draw it as a small lookup table joined by a ⊕ with a gate symbol.
3. **[MASK] token.** Masked positions are replaced by one learnable mask
   token.
4. **Shared Transformer encoder**, used by ALL routes:
   - 8 layers, width 512, 8 heads;
   - RMSNorm, QK-norm, SwiGLU feed-forward;
   - rotary position embedding (RoPE) inside attention.

   Draw one block with "×8" and list the four ingredients as small tags.
   Mark it "shared across routes".
5. **Two lightweight MLP reconstruction heads, one pair per sampling rate.**
   These are NOT a Transformer decoder.
   - the spec head predicts the normalised folded-wavelet patches;
   - the raw head predicts the raw waveform patches.
6. **Loss, on masked tokens only** (dotted orange):

   $$\mathcal{L} = 0.5\,\mathrm{MSE}_{\text{spec}} + 0.5\,\mathrm{SmoothL1}_{\text{raw}}(\beta{=}0.5) + 10^{-3}\,\mathrm{KL}_{\text{fold}}$$

   Connect each head's prediction to its detached target from the clean view.
7. **Downstream.**
   - A small inset or bracket: "fine-tuning keeps frontend + embeddings +
     encoder; heads are discarded".
   - This distinction matters: only the encoder is transferred.

## What changed from fig/model.png (must be fixed)

- "Transformer Decoder" is wrong. There is no decoder Transformer, only two
  per-rate MLP heads, a spec head and a raw head.
- There is one reconstruction target in the old figure; there are two now, a
  folded-wavelet target and a raw-waveform target, with the loss above.
- Old: the four bands were patchified directly as a $(J{+}1)C$ grid. Now
  Dynamic ScaleFold folds them back to $C$ rows before patching.
- Old: masking acted on tokens after the frontend. Now the mask is chosen on
  the clean view and applied to the signal before the online frontend, with
  detached targets from the clean view.
- Old: there was no notion of routes. Now four route-specific frontends share
  one patcher per sampling rate and one Transformer.
- The C1 channel embedding and the [MASK] token were missing.
- Drop the three-times repeated decomposition rows; one detailed level plus
  "×J" is enough.
- Fix the legend typo "Rotraty position embedding" → "Rotary position
  embedding (RoPE)".
- The old sliding-window and wavelet-basis plots were raster screenshots.
  Replace them with the real vector panels listed above.

## Legend (bottom right, compact, single column)

$\downarrow 2$ / $\uparrow 2$: down- / up-sampling by 2;
$d^{(l)}$, $a^{(l)}$: detail and approximation bands at level $l$;
$\alpha$: ScaleFold weights;
⊕: addition;
dashed: stop-gradient;
dotted: training-only loss.

## Checklist before you hand it back

- [ ] Every component above appears, and nothing else does.
- [ ] Read at 7.16 in wide, the smallest text is ≥ 6.5 pt and every arrow
      has an unambiguous direction.
- [ ] Only the purple, orange, blue and grey palette is used.
- [ ] The PDF embeds Times New Roman as TrueType (`pdffonts fig/model_v2.pdf`).
- [ ] The data panels are the real SVGs from `draw_architecture_panels.py`.
- [ ] The figure source file is committed next to the outputs.
