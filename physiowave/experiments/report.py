"""Aggregate raw result JSON into CSV, Markdown and LaTeX tables.

Everything downstream of a run reads the raw JSON, so tables can be regenerated
without re-running anything.  Multi-seed results are reported as ``mean +/- std``;
a single-seed entry is printed without a spread rather than with a fake one.
Rows produced on the synthetic corpus keep their ``synthetic`` flag all the way
into the rendered table, so a smoke-test number can never be read as a result.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load_results(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load every JSON file (globs allowed) into a flat list of records."""
    out: List[Dict[str, Any]] = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []):
            with open(path) as f:
                payload = json.load(f)
            records = payload.get("rows", payload if isinstance(payload, list) else [payload])
            for r in records:
                r = dict(r)
                r.setdefault("source", os.path.basename(path))
                for k in ("synthetic", "seed", "suite", "benchmark"):
                    if k in payload and k not in r:
                        r[k] = payload[k]
                out.append(r)
    return out


def mean_std(values: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Mean and (sample) standard deviation; ``None`` spread for a single value."""
    vals = [float(v) for v in values if v is not None and not isinstance(v, bool)]
    if not vals:
        return float("nan"), None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def aggregate(
    records: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    """Group by ``group_keys`` and reduce each metric to mean/std/n."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in records:
        key = tuple(str(r.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(r)
    rows: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        row: Dict[str, Any] = dict(zip(group_keys, key, strict=True))
        row["n_seeds"] = len({str(i.get("seed", 0)) for i in items})
        row["synthetic"] = any(bool(i.get("synthetic")) for i in items)
        for m in metric_keys:
            vals = [i.get(m) for i in items if isinstance(i.get(m), (int, float))]
            mu, sd = mean_std(vals)
            row[m] = mu
            row[f"{m}_std"] = sd
        rows.append(row)
    return rows


def format_value(mu: Any, sd: Optional[float] = None, digits: int = 4) -> str:
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        return "-"
    if not isinstance(mu, (int, float)):
        return str(mu)
    if sd is None:
        return f"{mu:.{digits}g}"
    return f"{mu:.{digits}g} ± {sd:.{max(digits - 2, 1)}g}"


def to_csv(rows: Sequence[Dict[str, Any]], path: str) -> str:
    if not rows:
        raise ValueError("no rows to write")
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def to_markdown(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
                title: str = "", digits: int = 4) -> str:
    """Markdown table with ``mean ± std`` cells."""
    lines = [f"### {title}", ""] if title else []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        cells = [format_value(r.get(c), r.get(f"{c}_std"), digits) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    if any(r.get("synthetic") for r in rows):
        lines += ["", "> Rows marked `synthetic=True` were produced on the synthetic "
                      "smoke corpus and are **not** benchmark results."]
    return "\n".join(lines)


def to_latex(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
             caption: str = "", label: str = "", digits: int = 4) -> str:
    """LaTeX ``tabular`` ready to paste into the paper."""
    def esc(value) -> str:
        return str(value).replace("_", r"\_").replace("%", r"\%")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(esc(c) for c in columns) + r" \\", r"\midrule"]
    for r in rows:
        cells = [format_value(r.get(c), r.get(f"{c}_std"), digits).replace("±", r"$\pm$")
                 for c in columns]
        lines.append(" & ".join(esc(c) if not any(ch.isdigit() for ch in c) else c
                                for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{" + esc(caption) + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def pareto_front(rows: Sequence[Dict[str, Any]], performance: str, cost: str,
                 maximize_performance: bool = True) -> List[Dict[str, Any]]:
    """Rows not dominated on (performance, cost); cost is always minimised."""
    pts = [r for r in rows if isinstance(r.get(performance), (int, float))
           and isinstance(r.get(cost), (int, float))]
    front: List[Dict[str, Any]] = []
    for r in pts:
        dominated = False
        for o in pts:
            if o is r:
                continue
            better_perf = (o[performance] >= r[performance]) if maximize_performance \
                else (o[performance] <= r[performance])
            strictly = (o[performance] > r[performance]) if maximize_performance \
                else (o[performance] < r[performance])
            if better_perf and o[cost] <= r[cost] and (strictly or o[cost] < r[cost]):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[cost])


def write_report(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    out_dir: str,
    name: str,
    title: str = "",
    pareto: Optional[Tuple[str, str]] = None,
) -> Dict[str, str]:
    """Write ``<name>.{csv,md,tex}`` (and optional Pareto JSON) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "csv": to_csv(rows, os.path.join(out_dir, f"{name}.csv")),
        "md": os.path.join(out_dir, f"{name}.md"),
        "tex": os.path.join(out_dir, f"{name}.tex"),
    }
    with open(paths["md"], "w") as f:
        f.write(to_markdown(rows, columns, title or name))
    with open(paths["tex"], "w") as f:
        f.write(to_latex(rows, columns, caption=title or name, label=f"tab:{name}"))
    if pareto:
        perf, cost = pareto
        front = pareto_front(rows, perf, cost)
        paths["pareto"] = os.path.join(out_dir, f"{name}_pareto.json")
        with open(paths["pareto"], "w") as f:
            json.dump({"performance": perf, "cost": cost, "front": front,
                       "all_points": list(rows)}, f, indent=2, default=str)
    return paths


def main(argv=None) -> int:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Aggregate PhysioWave results into tables")
    p.add_argument("--inputs", nargs="+", default=["./results/*.json"])
    p.add_argument("--out-dir", default="./results/tables")
    p.add_argument("--name", default="summary")
    p.add_argument("--group-by", nargs="+", default=["variant"])
    p.add_argument("--metrics", nargs="+",
                   default=["tokens", "token_compression_ratio", "params",
                            "flops_forward", "peak_mem_mb", "samples_per_sec"])
    p.add_argument("--pareto", nargs=2, default=None,
                   metavar=("PERFORMANCE", "COST"))
    args = p.parse_args(argv)

    records = load_results(args.inputs)
    if not records:
        print(f"No results found under {args.inputs}")
        return 1
    rows = aggregate(records, args.group_by, args.metrics)
    cols = list(args.group_by) + ["n_seeds"] + list(args.metrics)
    paths = write_report(rows, cols, args.out_dir, args.name,
                         pareto=tuple(args.pareto) if args.pareto else None)
    print(json.dumps(paths, indent=2))
    print()
    print(to_markdown(rows, cols, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
