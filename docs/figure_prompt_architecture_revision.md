# Revision prompt: panels (c) and (d) of the PhysioWave architecture figure

This is a revision of the figure you produced from
`docs/figure_prompt_architecture_pptx.md` (same repository,
https://github.com/ForeverBlue816/PhysioWave-T, branch
`fix/channel-identity-and-block-variants`). Everything in that prompt still
holds: canvas 18.2 × 10.0 cm, Times New Roman, the purple / orange / blue
palette, editable PPTX + 600 dpi PNG + the generating script.

Panels (a) and (b) are accepted with the small fixes in section 4. Panels
**(c) and (d) must be re-laid out**: they are the right half of the figure
and at the moment they are the hard part to read.

## 1. What is wrong with (c) and (d) now

Fix every item; section 2 says how.

1. **The reading direction zig-zags.** (c) reads top → bottom; the data path
   then enters (d) at its bottom-left corner and (d) reads bottom → top, with
   the targets at the top. A reader goes down, across and back up.
2. **A dashed stop-gradient line runs along the top edge of the figure and
   strikes through the panel titles** "FFT-guided masking" and "dual
   reconstruction". No connector may cross text.
3. **The same operation is drawn twice in two styles.** The blue box
   "Patch + C1 + position" in (c) and "Patch embed ⊕ C1 identity + position"
   in (d) are the same tokeniser (`_build_token_view` in
   `physiowave/eeg_c1/model.py`).
4. **𝓕ᵣ is never defined.** "𝓕ᵣ" and "𝓕ᵣ (shared weights)" appear in (c)
   with no visual link to the frontend drawn in (b).
5. **The input x enters (c) from the left and drops down a long blue vertical
   line** to the masked signal. It is hard to follow and it fights the
   left-to-right flow.
6. **(d) stacks seven objects vertically in each of two columns** (patch
   embed, [MASK], Transformer, MLP, prediction, loss box, target). The loss
   boxes "MSE" and "SmoothL1" sit between prediction and target as if they
   were layers. A loss is an edge, not a layer.
7. **Legend and drawing disagree.** The legend says dotted orange = training
   loss and dashed + padlock = stop-gradient; the loss edges are drawn solid,
   and one padlock sits on a line, the other on a target box.
8. **Missing from the brief:** the four tags on the Transformer
   (RMSNorm · QK-norm · SwiGLU · RoPE); "per sampling rate" on the patch
   embedding and on the heads; the tanh gate on the C1 ⊕; tensor shapes
   along the path; the downstream bracket ("kept for fine-tuning …").
9. **One correction to the earlier brief, in your favour:** "+ position" is
   right. The tokeniser adds a parameter-free 2-D sinusoidal position
   encoding (channel × time) after the C1 code (`self.pos_embed` in
   `model.py`), and RoPE acts additionally inside the attention. Keep both,
   labelled precisely: "+ 2-D sin position" at the tokeniser, "RoPE" as a
   tag on the Transformer.

## 2. Required layout for (c) + (d): two lanes, both left → right

Treat the right half of the figure as **one region with two horizontal
lanes**. Panel (c) is the left part of that region, (d) the right part; the
lanes run across both. Every arrow inside a lane points right. Only three
kinds of connector are vertical: the mask dropping from the upper lane to
the lower lane, the loss arrows at the far right, and nothing else.

```
          (c) Two views & FFT-guided masking      │  (d) Shared encoder & dual reconstruction
 UPPER   x ─► 𝓕ᵣ ─► tokenise ─► FFT importance ─► mask grid     ····► spec target   raw target
 (clean)      (b)   (same)      strip             top 70%             (patch-norm)   (patchify x)
                 ╲ dashed, padlock ───────────────────────────────────────►┘            ▲ dashed, padlock, from x
                                                   │ mask                 ▲ dotted  ▲ dotted
                                                   ▼                      │ MSE     │ SmoothL1
 LOWER   x ─► zero masked ─► 𝓕ᵣ ─► tokenise ─► [MASK] ─► Transformer ×8 ─► Spec MLP ─► spec pred
 (online)     patches        (shared) ⊕C1 +pos  at masked           shared     Raw MLP  ─► raw pred
              "mask before frontend"             positions          4 tags
```

Concretely:

### Upper lane: the clean view (targets and the mask)

Left to right:

1. **x** (small label; the signal panel is already in (a), do not repeat it).
2. **𝓕ᵣ** — a compact purple box labelled "route frontend 𝓕ᵣ (b)". Define 𝓕ᵣ
   by adding it to (b)'s title: "(b) Route frontend 𝓕ᵣ".
3. **tokenise** — a small blue box labelled "tokenise (as below)". No
   internals here; they are drawn once, in the lower lane.
4. **FFT importance** — the thin heat strip you already have, labelled
   "|FFT| along tokens, 0.6 importance + 0.4 noise".
5. **mask grid** — the purple/grey grid (`H0_mask_grid_all.svg` or
   `02_mask_grid.svg`), labelled "top 70 % of valid tokens masked".
6. At the **far right of the upper lane, two heatmaps side by side**:
   **spec target** (`H5_spec_target.svg`) and **raw target**
   (`H1_raw_target.svg`).
   - The spec target is fed by a **dashed** arrow from the 𝓕ᵣ box (routed
     below the FFT/mask items, inside the lane), labelled "patch-normalised"
     with **one padlock on the arrow**.
   - The raw target is fed by a **short dashed** arrow from x labelled
     "patchify", also with a padlock on the arrow. It must stay inside the
     upper lane. **Delete the line along the top edge of the figure.**

### Lower lane: the online view (the path that is trained)

Left to right, boxes aligned to the same columns as the upper lane:

1. **x** → **"zero masked patches"** — a small purple box holding the masked
   signal panel (`03_masked_input.svg`), with the caption "mask before
   frontend". The **mask arrives here by a vertical purple arrow** from the
   mask grid above. This is the first of the two vertical connectors.
2. **𝓕ᵣ** — the same purple box as above, labelled "𝓕ᵣ, shared weights".
   Place it directly under the upper lane's 𝓕ᵣ.
3. **tokenise** — the one detailed tokeniser, blue:
   "patch embed (per rate: 128 / 256 samples) ⊕ tanh(g)·C1 identity
   + 2-D sin position". Draw the lookup-table glyph, the ⊕ and a tiny gate
   symbol here and nowhere else. Output label: *CP × D*, D = 512.
4. **[MASK]** — a small orange box "[MASK] token at masked positions". The
   **second vertical connector**: a thin purple arrow from the mask grid
   down to this box (route it so it does not cross the upper lane's
   dashed arrows; if it would, bring both mask arrows down as one trunk
   with a branch).
5. **Shared Transformer** — one orange box, wider than the others: title
   "Shared Transformer ×8 — shared across all routes", subtitle "D = 512,
   8 heads", and four small tags in one row inside the box: RMSNorm ·
   QK-norm · SwiGLU · RoPE.
6. **Two heads** — two small orange boxes stacked within the lane:
   "Spec MLP (per rate)" above "Raw MLP (per rate)". Output label *CP × p*.
7. At the **far right of the lower lane, two heatmaps side by side, directly
   under the two targets**: **spec prediction** (`H7_spec_composite.svg`)
   under the spec target, **raw prediction** (`H3_raw_composite.svg`) under
   the raw target. Caption under them, one line: "visible = target, masked
   = prediction".

### The loss: vertical dotted orange arrows at the far right

- From the spec prediction **up** to the spec target: dotted orange arrow
  labelled "MSE, masked tokens".
- From the raw prediction up to the raw target: dotted orange arrow labelled
  "SmoothL1 (β = 0.5), masked tokens".
- **No MSE / SmoothL1 boxes.** The loss is an edge. The full formula stays
  where it is, under the figure, and gains nothing else.
- The KL(α ‖ u) term already has its note in (b); leave it there.

### Downstream bracket

Under the lower lane, spanning 𝓕ᵣ → tokenise → Transformer, a thin grey
bracket with the text "kept for fine-tuning: frontend + tokeniser + encoder
(heads discarded)".

### Grid, spacing, size

- Two lanes of equal height; the lane gap holds only the two mask arrows and
  the two lane captions ("clean view — targets, stop-grad" / "online view —
  trained").
- Boxes in the two lanes share column positions; align their vertical centre
  lines. Equal horizontal gaps.
- The four heatmaps at the right are the same size (≈ 1.6 × 1.6 cm) and form
  a clean 2 × 2 block: targets above, predictions below.
- Give the (c)+(d) region the width it needs: (a) and (b) together may shrink
  to ≈ 45 % of the slide width; (c)+(d) take ≈ 55 %.
- Nothing crosses a panel title or a lane caption. No connector crosses
  another. If a route cannot be drawn without a crossing, move a box, not
  the arrow.

## 3. Style reminders for the redrawn half

- Only the palette: data path blue `#2166AC` / `#D1E5F0`; frontend and mask
  purple `#5E3C99` / `#E7E1EF`; encoder, heads, predictions and loss orange
  `#E66101` / `#FDE0C5`; arrows and text `#4D4D4D`.
- Solid arrow = data; dashed + padlock = stop-gradient (exactly two: spec
  target, raw target); dotted orange = loss (exactly two). Make the legend
  match the drawing, and put the padlock on the arrow in both cases.
- Text ≥ 6.5 pt at 18.2 cm width; module titles bold 8–9 pt; tensor shapes
  grey italic 7 pt beside arrows: *C × T* after 𝓕ᵣ, *CP × D* after the
  tokeniser and after the Transformer, *CP × p* after the heads.
- Heatmap and signal panels are the repository SVGs
  (`docs/runs/eeg_c1_moe/architecture_panels_bare/E64_256/`), inserted as
  600 dpi transparent PNGs at their true aspect ratio.

## 4. Small fixes outside (c) and (d)

- (b): rename the panel title to "(b) Route frontend 𝓕ᵣ" so (c) can refer
  to it. Replace the four-colour donut at the ScaleFold's Σ with a plain Σ
  in a circle; the α-per-patch bar chart beside it already carries the
  colour. Change the fourth band's colour from green to grey `#4D4D4D` if
  your panel set still has green (the repository's current panels are
  grey).
- (a): the C × T heatmap and the route stack are fine. Move the legend to
  the bottom-right of the whole figure if it fits better after the
  re-layout; otherwise leave it.

## 5. Deliverables and checks

Deliver the updated `physiowave_architecture.pptx`, `.png` (600 dpi, white
background) and the generating script, plus a short list of anything in
this prompt that disagrees with the code (the code wins).

Before delivering, check the right half with these questions:

- [ ] Can you trace the online path with a finger from x to the raw
      prediction moving only rightwards? And the clean path from x to the
      targets likewise?
- [ ] Are there exactly: 2 𝓕ᵣ boxes, 2 tokeniser boxes (one detailed), 1
      Transformer, 2 heads, 2 targets, 2 predictions, 2 dashed arrows with
      padlocks, 2 dotted loss arrows, 0 loss boxes?
- [ ] Does any line touch a panel title, lane caption or another line?
- [ ] Do the Transformer tags, "per rate" notes, tanh gate, tensor shapes
      and downstream bracket all appear?
- [ ] Does every symbol in the legend appear in the drawing in exactly that
      style, and vice versa?
- [ ] At 18.2 cm wide, is every label ≥ 6.5 pt and nothing overlapping?
