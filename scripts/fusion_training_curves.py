"""Validation curves of the released RALF run beside the paper's schedule sweep."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root, out = sys.argv[1], sys.argv[2]
runs = [("ralf_frozen_szo", "released: lr 3e-4, 60-epoch cosine (original)"),
        ("ralf_frozen_e15_szo", "lr 1e-4, 15 epochs"), ("ralf_frozen_e15rw_szo", "lr 1e-4, 15 ep., reliability weight 0.3"),
        ("ralf_frozen_e15gd_szo", "lr 1e-4, 15 ep., detached gate"), ("ralf_frozen_ep12_szo", "lr 3e-4, 12-epoch cosine"),
        ("ralf_frozen_e15siena_szo", "lr 1e-4, 15 ep., Siena-continued EEG encoder")]
H = {r: json.load(open(os.path.join(root, r, "history.json"))) for r, _ in runs}
auprc = lambda h: [e["val"]["seizeit2"]["auprc"] for e in h]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.titleweight": "bold", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
INDIGO, GOLD, MAUVE, BLUE, LAV, GREY = "#3F4280", "#DCA94F", "#C3609E", "#7F97C2", "#8A84BF", "#A9ADB8"
cols = [INDIGO, MAUVE, BLUE, LAV, GREY, "#DCA94F"]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]})
for (r, lab), c in zip(runs, cols):
    h = H[r]; y = auprc(h); b_i = max(range(len(y)), key=lambda i: y[i])
    main = r == "ralf_frozen_szo"
    a.plot(range(len(y)), y, "-", color=c, lw=2.2 if main else 1.2, alpha=1 if main else 0.9, label=lab, zorder=4 if main else 2)
    a.plot([b_i], [y[b_i]], "o", color=c, ms=7 if main else 4.5, mec="white", mew=0.8, zorder=6)
h0 = H["ralf_frozen_szo"]; y0 = auprc(h0); i0 = max(range(len(y0)), key=lambda i: y0[i])
a.annotate(f"released checkpoint\nepoch {i0}, AUPRC {y0[i0]:.3f}", (i0, y0[i0]), xytext=(10, 18), textcoords="offset points",
           fontsize=8.5, color=INDIGO, arrowprops={"arrowstyle": "-", "color": INDIGO, "lw": 0.8})
a.set_xlabel("epoch"); a.set_ylabel("validation AUPRC (windows, complete modalities)")
a.set_title("(a) Checkpoint selection, six frozen-encoder RALF runs")
a.legend(frameon=False, fontsize=7.6, loc="lower left"); a.set_ylim(0.12, 0.43)
tr = lambda k: [e["train"]["seizeit2"][k] for e in h0]
b.plot(range(len(h0)), tr("loss_task"), "-", color=INDIGO, lw=1.8, label="task CE")
b.plot(range(len(h0)), tr("loss_reliability"), "-", color=MAUVE, lw=1.6, label="reliability MSE")
b.plot(range(len(h0)), tr("loss_consistency"), "-", color=GOLD, lw=1.6, label="subset consistency")
b.axvline(i0, color=INDIGO, lw=1, ls="--"); b.set_xlabel("epoch"); b.set_ylabel("training loss")
b.set_title("(b) Released run, loss terms"); b.legend(frameon=False, fontsize=8.5)
fig.tight_layout(w_pad=2.0); fig.savefig(out, bbox_inches="tight"); print("wrote", out)
