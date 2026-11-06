#!/usr/bin/env python
"""
The SeizeIT2 event-level table: our runs beside the published numbers.

    python scripts/collect_seizeit2_events.py --root $PW_CKPT_ROOT/fusion [--gate cross] \\
        [--markdown out.md] [--condition complete]

Reads <root>/*/events_<gate>.json (Fusion/seizeit2_events.py). Columns:

  sens% FA/h score   threshold 0.5, the dataset paper's scoring (means over
                     recordings of the any-overlap sensitivity and FA/h; the
                     challenge score 100*sens - 0.4*epoch FA/h)
  s@11               the best mean any-overlap sensitivity at <= 11 FA/h on the
                     threshold curve: the SVM's operating point
  AUROC AUPR (paper) the paper's epoch-based areas (seizure_events.paper_areas);
                     the published rows' AUROC / AUPR go here
  pooled ...         the pooled event definition (seizures hit / seizures,
                     FA/24h, sensitivity at 1 and 6.22 FA/h) and the pooled
                     per-second AUROC / AUPRC before post-processing -- our own
                     numbers, NOT comparable with the paper's AUROC / AUPR

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

#: (label, sens %, FA/h, AUROC, AUPRC, note)
PUBLISHED = [
    ("SVM, bte-EEG (Bhagubai et al., Sci. Data 2025)", 71.1, 11.0, 0.6995, 0.0810,
     "same split and scoring"),
    ("ChronoNet, bte-EEG (Bhagubai et al., Sci. Data 2025)", 84.2, 100.5, 0.7921, 0.1042,
     "same split and scoring"),
    ("bte-EEG + ECG algorithm (Swinnen et al., Epilepsia Open 2026)", 78.0, 6.22, None, None,
     "per-patient mean, own cohort and protocol"),
]


def fmt(v, nd=3):
    return "  -  " if v is None or v != v else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", default="cross")
    ap.add_argument("--condition", default=None, help="one condition; default all")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physiowave.fusion.seizure_events import paper_areas
    files = sorted(glob.glob(os.path.join(args.root, "*", f"events_{args.gate}.json")))
    if not files:
        print(f"no events_{args.gate}.json under {args.root}")
        return 1
    head = (f"{'run / condition':42s} {'sens%':>6s} {'FA/h':>6s} {'score':>6s} {'s@11':>5s}"
            f" | {'AUROC':>6s} {'AUPR':>6s} (paper)"
            f" | pooled {'sens':>5s} {'FA/24h':>6s} {'s@1':>5s} {'s@6.2':>5s} {'AUROC':>6s} {'AUPRC':>6s}")
    md = ["# SeizeIT2, event-level, official split (test = sub-097..125)", "",
          f"RMS gate: {args.gate}. Threshold 0.5 unless stated. AUROC / AUPR (paper) are the "
          "dataset paper's epoch-based areas; the pooled per-second AUROC / AUPRC are ours and "
          "not comparable with them.", "",
          "| run / condition | sens % | FA/h | score | sens @<=11 FA/h | AUROC (paper) | "
          "AUPR (paper) | pooled sens | FA/24h | sens @1 FA/h | sens @6.22 FA/h | "
          "per-second AUROC | per-second AUPRC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    print(head)
    print("-" * len(head))
    for f in files:
        name = os.path.basename(os.path.dirname(f))
        ev = json.load(open(f))
        for cond, r in ev["conditions"].items():
            if args.condition and cond != args.condition:
                continue
            a = r["at_0.5"]
            o, e = a["official"], a["events"]
            au_p, ap_p = r.get("auroc_paper"), r.get("aupr_paper")
            if au_p is None:                              # an older file: from its curve
                au_p, ap_p = paper_areas(r["curve"])
            s11 = r["sens_at_fa_official"].get("11.0")
            row = [100 * o["sens_ovlp"], o["fa_ovlp_h"], o["score"],
                   None if s11 is None else 100 * s11, au_p, ap_p,
                   e["sens"], e["fa_24h"], r["sens_at_fa_events"].get("1.0"),
                   r["sens_at_fa_events"].get("6.22"), r["auroc_epoch"], r["auprc_epoch"]]
            label = f"{name} / {cond}"
            print(f"{label[:42]:42s} {fmt(row[0], 1):>6s} {fmt(row[1], 2):>6s} {fmt(row[2], 1):>6s}"
                  f" {fmt(row[3], 1):>5s} | {fmt(row[4]):>6s} {fmt(row[5], 4):>6s}        "
                  f" |        {fmt(row[6]):>5s} {fmt(row[7], 1):>6s} {fmt(row[8]):>5s}"
                  f" {fmt(row[9]):>5s} {fmt(row[10]):>6s} {fmt(row[11], 4):>6s}")
            md.append(f"| {label} | " + " | ".join(
                fmt(v, nd) for v, nd in zip(row, (1, 2, 1, 1, 3, 4, 3, 1, 3, 3, 3, 4))) + " |")
    print("-" * len(head))
    md.append("| **published** | | | | | | | | | | | | |")
    for label, sens, fa, auroc, auprc, note in PUBLISHED:
        s11 = f"{sens:.1f}" if fa <= 11.0 and "same split" in note else ""
        print(f"{label[:42]:42s} {sens:6.1f} {fa:6.2f} {'':>6s} {s11:>5s} | {fmt(auroc):>6s} "
              f"{fmt(auprc, 4):>6s}         |   ({note})")
        md.append(f"| {label} ({note}) | {sens:.1f} | {fa:.2f} | | {s11} | {fmt(auroc)} | "
                  f"{fmt(auprc, 4)} | | | | | | |")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
