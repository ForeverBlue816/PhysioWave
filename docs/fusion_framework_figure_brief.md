# Brief: the RALF multimodal framework figure (v2, lean)

The architecture figure of the multimodal seizure-detection pipeline (three
pretrained C1 encoders + RALF), at the standard of a TPAMI Figure 1. The
first draft (v1) failed on three counts: a wall of text, MPDB in the figure,
and no visual hierarchy. This version fixes the brief, not just the drawing.

**The principle: the figure shows the structure; the caption carries the
numbers.** Nothing below asks for a sentence inside the figure. Every block
has a name of at most three words, and at most one short line under it.
Total text on the figure: under 120 words. Everything else goes into a
caption paragraph the drawer delivers alongside (template at the end).

## Deliverable

- Width **17.8 cm**, height **7.5 to 8.5 cm** (one row, not a page).
- Vector (SVG + PDF, text as text, fonts embedded) and a 600 dpi PNG;
  the editable source file.
- Times New Roman bold, 7 pt minimum, panel letters 9 pt. One font, two
  sizes (7 and 8 pt) plus the panel letters. No 6 pt text anywhere.
- Palette, used for nothing else: EEG `#0072B2`, ECG `#D55E00`, EMG
  `#009E73`, reliability `#CC79A7`, losses `#E69F00`, shared blocks:
  `#4D4D4D` outline, white fill. Lane tint 20 % of the lane colour. Flat:
  no gradients, shadows, 3-D, icons, clip art.
- Lines 0.6 pt, arrows 0.8 pt with one arrowhead style, orthogonal elbows,
  left-to-right flow. Blocks: rounded rectangles, one corner radius, equal
  heights within a region, aligned on a 12-column grid with equal gutters.
- Greyscale-safe: every modality block carries its text tag (EEG / ECG /
  EMG) as well as its colour.
- **Not in the figure:** MPDB (anywhere), training hyper-parameters,
  pretraining corpus sizes, optimizer, epochs, the list of corruption kinds,
  the padlock legend sentence, any tensor shape other than the four listed
  below.

## Layout: one row of five regions, a thin dashed strip below

```text
(a) inputs   (b) C1 encoders, frozen      (c) pool   (d) RALF                 (e) output
 10 %         30 %                         7 %        40 %                     13 %
──────────────────────────────────────────────────────────────────────────────────────
(f) training only: corruption, four losses              (one line of labels, no formulas)
```text

### (a) Inputs

Three real waveform thumbnails, supplied as `fig_window_eeg.svg`,
`fig_window_ecg.svg`, `fig_window_emg.svg` (use them as given, in their
colours, no redrawing). Under each, one line, 7 pt:

- `bte-EEG · 2 ch · 256 Hz`
- `ECG · 1 ch · 250 Hz`
- `EMG · 1 ch · 2 kHz`

Above the column: `4 s window`. Nothing else.

### (b) Three C1 encoders

Three lanes (tinted, outlined in the lane colour), each containing **one**
block labelled `C1 encoder`, with a small padlock glyph (outline) at its
corner and, inside the block, four words in a row separated by thin arrows:
`wavelet → fold → patch → Transformer`. Under the block one line:

- EEG lane: `512-d × 8 layers`
- ECG lane: `512-d × 9 layers`
- EMG lane: `384-d × 6 layers`

The word `frozen` appears once, as the padlock's legend in the region title:
`(b) C1 encoders (frozen)`. The lane output arrow carries the only shape in
this region: `[C·P, D]`.

### (c) Summary pool

Per lane one small block `pool` (a circle or short rounded box), output
arrow labelled `[4, 256]`. Region title: `(c) 4-query summary pool`.

### (d) RALF

A grey-outlined rounded box titled `(d) RALF: reliability-aware late fusion`.
Inside, four things only:

1. **Reliability.** On each lane's arrow a small purple circle `r` (the
   reliability head) and a ⊗ where it gates the lane's tokens. One purple
   label for all three: `reliability r_m ∈ [0, 1]`.
2. **Cross-attention.** One block `8 fusion tokens` feeding one block
   `2 × cross-attention` whose key/value input is the three gated arrows.
   A single 7 pt note on the K/V input: `absent modality: masked, r_m = 0`.
3. **Unimodal logits.** From each lane's token a thin arrow to a small
   block `z_m` in the lane colour.
4. **Residual sum.** A ⊕ joining the cross-attention output and the three
   `z_m` arrows, with the one equation set in math type beside it:

   `logits = z_fusion + ½ · Σ r_m z_m / Σ r_m`

No MLP internals, no LayerNorm, no FFN boxes, no shapes inside (d).

### (e) Output

One block `P(seizure)` and a small probability-trace glyph with a dashed
threshold line. One line under it: `per 4 s window`. Nothing about
post-processing (caption).

### (f) Training strip

A thin dashed band under (b)–(d), titled `(f) training only`, holding:

- a small block `corrupt` with a dashed arrow into the input side of (b)
  and the label `level ℓ_m`;
- four orange labels placed under what they train, each one token plus a
  short tail: `L_task (CE)` under ⊕; `L_uni` under the `z_m` row; `L_rel:
  r_m → 1 − ℓ_m` under the reliability circles; `L_cons: subset ≈ full`
  under the cross-attention block.

No formulas other than these four labels. No optimizer line. No "shared"
ghost copy.

## Checks before delivery

- Count the words on the figure: under 120. If over, cut text, not font size.
- MPDB does not appear. No hyper-parameter appears.
- The four shapes present are exactly `[C·P, D]`, `[4, 256]` and the two
  in the inputs' labels (channels, rate); nothing else carries a shape.
- EEG, ECG and EMG lanes are three different hues with the hex values above.
- Print at 17.8 cm: every label legible, no text under 7 pt.
- Delivered: `fig_framework.svg`, `.pdf`, `.png` (600 dpi), the source, and
  the caption paragraph below filled in.

## Caption template (delivered as text, not drawn)

> **Fig. 1.** Multimodal seizure detection with three frozen C1 encoders and
> reliability-aware late fusion (RALF). (a) One 4 s window of behind-the-ear
> EEG (2 channels, 256 Hz), ECG (250 Hz) and EMG (2 kHz), z-scored per
> channel. (b) Each modality has its own C1 encoder, pretrained by masked
> reconstruction and frozen: a learnable wavelet frontend (SoftGate
> decomposition, dynamic ScaleFold), patch embedding, channel-identity
> embedding, and a Transformer of 512-d × 8 (EEG), 512-d × 9 (ECG) or
> 384-d × 6 (EMG) layers; the montage adapter trains. (c) Four learned
> queries summarise each encoder's tokens; a linear map brings them to 256-d.
> (d) A reliability head scores each modality, r_m ∈ [0, 1], and gates its
> tokens; eight learned fusion tokens cross-attend over the gated tokens in
> two layers; the fused logit is combined with the reliability-weighted
> unimodal logits z_m as a residual. An absent modality is masked out and
> its r_m set to 0. (e) The window's seizure probability; events are formed
> by merging detections closer than 2 s and dropping those shorter than 8 s.
> (f) In training, inputs are corrupted with probability 0.5 per modality
> (noise, drift, saturation, bad channel, dropout) at level ℓ_m; the losses
> are the task cross-entropy, the unimodal cross-entropies (0.2), the
> reliability regression r_m → 1 − ℓ_m, and a consistency term that makes a
> random modality subset agree with the full prediction.

## v2.1: the visual language of (c) and (d)

The second draft had the right content and read as a block diagram: every
element a rectangle, every arrow an elbow, the fusion core a grid of boxes.
These rules replace the "orthogonal elbows" rule for regions (c)–(e) and add
visual devices; content and word budget are unchanged.

- **Tokens are drawn as tokens.** A summary token is a small rounded square
  in the lane colour. The pool is four such tokens fanning out of the
  encoder (four thin curved lines from the encoder's right edge converging
  on the four squares), not a circle with an icon. After the gate the four
  squares keep their colour but take their opacity from r (the EEG row at
  full, one other row visibly paler): the gating is seen, not only labelled.
- **Reliability is a dial.** Each r is a small semicircular gauge in purple
  with a needle, positioned above its lane; a thin purple line drops to the
  ⊗. The purple label `r_m ∈ [0, 1]` appears once.
- **Cross-attention is attention.** The eight fusion tokens are a column of
  eight small grey squares on the left of the core; the twelve gated tokens
  stand in a column on the right; a fan of thin curved lines connects a few
  fusion tokens to several gated tokens, with line weight suggesting
  attention weight. `2 ×` is written once beside the column. No box with a
  grid inside it.
- **Arrows curve.** From (c) onwards, lines are smooth cubic curves with a
  single bend (no S-shapes), 0.8 pt, in the lane colour where they carry one
  modality and grey where they carry the fused signal. Elbows remain in (b).
- **The residual is a braid.** The three z_m lines approach the ⊕ as three
  thin coloured curves that merge into one grey line; the equation sits
  under the ⊕ in math type, nothing else near it.
- **Whitespace is part of the design.** Regions (c) and (d) together take
  half the width; nothing in (d) touches the box edge; the core's box has a
  2 mm inner margin and a 1.5 mm corner radius like every other block.
- **(e) shows a result.** The probability trace is a real one (take it from
  `fig_seizure_example`'s bottom panel, a 60 s excerpt), with the dashed 0.5
  line and a short black bar over the detected event.
- **Icons, if any, are three.** The four words inside an encoder may carry
  one glyph each only if the glyphs share one stroke weight and one height;
  otherwise use the words alone. No other icon anywhere.

## v2.2: colour in (d)–(e), and the encoder glyphs

- **The fusion core is not monochrome.** Colour follows the signal: a line,
  token or attention edge keeps its modality colour until the ⊕. So inside
  RALF the gated tokens, the attention fan (each edge in the colour of the
  gated token it comes from), the three z_m blocks and the three curves
  into the ⊕ are blue / vermilion / green; the dials are purple; only the
  eight fusion tokens, the fused line after the ⊕ and the P(seizure) block
  are the neutral `#4D4D4D`. The RALF box gets a 4 % grey fill so the region
  reads as one unit. In (e) the probability trace is drawn in `#4D4D4D` with
  the detected-event bar in black.
- **Encoder glyphs: one set or none.** If glyphs are kept, they are these
  four, one stroke weight (0.8 pt), one height (2.5 mm), in the lane colour:
  wavelet = one Morlet-like wavelet; fold = four short horizontal lines
  converging to one (a ⟩ shape of lines); patch = one horizontal strip cut
  into 8 cells; Transformer = three stacked rounded rectangles (layers).
  No grids, no cubes. If any of the four cannot be drawn to that standard,
  use the four words alone.
- **The pool is the four-token fan** of v2.1, not a circle with an icon.

## v2.3: the training signals dissolve into the figure; math is typeset

- **No region (f).** The dashed band is removed. `corrupt` becomes a small
  dashed-outline tag `corrupt (train)` sitting on the three input arrows
  between (a) and (b). The four losses become small orange tags, each at
  the element it trains, joined to it by a short dashed orange leader:
  `L_task` at the ⊕, `L_uni` at the z_m column, `L_rel` at the dials,
  `L_cons` at the cross-attention. Tags carry the symbol only; what each
  loss does is in the caption.
- **Math is typeset, not typed.** Every formula is set as mathematics (LaTeX
  via the drawing tool, or MathJax/KaTeX to SVG, then placed): italic
  variables, upright text subscripts, no spaces before subscripts, a real
  fraction. The one equation in the figure is

  `y = z_fus + ½ · (Σ_m r_m z_m) / (Σ_m r_m)`   set as
  $y = z_{\mathrm{fus}} + \tfrac{1}{2}\,\dfrac{\sum_m r_m z_m}{\sum_m r_m}$

  and the loss tags are $\mathcal{L}_{\mathrm{task}}$,
  $\mathcal{L}_{\mathrm{uni}}$, $\mathcal{L}_{\mathrm{rel}}$,
  $\mathcal{L}_{\mathrm{cons}}$. Symbols in running labels ($r_m$, $z_m$,
  $\ell_m$) are set the same way.
- **Text pared to the names.** Remaining text on the figure, and nothing
  else: the panel letters (a)–(e); `4 s window`; the three input lines; per
  lane `EEG encoder` / `ECG encoder` / `EMG encoder` with the four words
  inside and the size line under it; `(b) C1 encoders (frozen)`; `pool`
  once, `[4, 256]` once; `(d) RALF`; `$r_m$` once by the dials;
  `fusion tokens`; `2 × cross-attention`; `absent → masked` once; the three
  `$z_m$` blocks (`EEG`, `ECG`, `EMG` + $z_m$); the equation; `P(seizure)`;
  `60 s` under the trace; the five tags of this section. Everything else
  (`[C·P, D]`, `4-query summary`, `r_m ∈ [0, 1]`, `per 4 s window`,
  `level ℓ_m`) goes.
- **Attention fan: fewer edges.** Each fusion token connects to two or
  three gated tokens, 20 edges at most, so the fan reads as attention and
  not as wiring.
