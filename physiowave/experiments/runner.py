"""Experiment-matrix runner.

Executes the variants declared in an ``configs/experiments/*.yaml`` file over the
configured seeds, writing one raw JSON per (variant, seed) so
:mod:`physiowave.experiments.report` can aggregate them.

Priority tiers (matching the project plan):

``tier 1``  the channel/spatial encoding ladder -- the main paper table
``tier 2``  spatial-branch and channel-relation-graph ablations, and K
``tier 3``  multimodal and robustness

``--tier`` restricts execution so the highest-priority table is completed first
when compute is short.  ``--smoke`` shrinks every run to a few steps on the
synthetic corpus, which is what CI and ``run_tpami.sh smoke`` use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ..config import deep_merge, load_config
from ..train.pretrain_main import main as pretrain_main

logger = logging.getLogger(__name__)


def variant_config(cfg: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a variant's overrides onto the experiment's base config."""
    out = json.loads(json.dumps(cfg))
    for key in ("model", "pretrain", "data", "train"):
        if key in variant:
            out[key] = deep_merge(out.get(key, {}), variant[key])
    out.pop("experiment", None)
    return out


def run_matrix(
    config: str,
    out_dir: str = "./results/experiments",
    tiers: Sequence[int] = (1,),
    seeds: Optional[Sequence[int]] = None,
    smoke: bool = False,
    max_steps: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run (or list) every variant of an experiment config."""
    cfg = load_config(config)
    exp = cfg.get("experiment", {}) or {}
    tier = int(exp.get("tier", 1))
    if tier not in tiers:
        logger.info("Skipping %s: tier %d not in requested tiers %s",
                    config, tier, list(tiers))
        return {"skipped": True, "config": config, "tier": tier}

    # Three kinds of experiment live in configs/experiments/, dispatched here:
    #   `variants`   -> pretraining runs (the ablation ladders)
    #   `benchmark`  -> the token-efficiency sweep (no training involved)
    #   `conditions` -> multimodal robustness (evaluation of a fusion model)
    if exp.get("benchmark") and not exp.get("variants"):
        from ..train.benchmark import run_token_benchmark

        out_path = os.path.join(out_dir, f"{exp.get('name', 'benchmark')}.json")
        if dry_run:
            for var in exp["benchmark"].get("variants", []):
                logger.info("[dry-run] would benchmark %s -> %s", var.get("id"), out_path)
            return {"config": config, "rows": exp["benchmark"].get("variants", [])}
        result = run_token_benchmark(config, output=out_path, quick=smoke)
        return {"config": config, "rows": result["rows"]}

    if exp.get("conditions"):
        from ..train.evaluate import run_multimodal_robustness

        out_path = os.path.join(out_dir, f"{exp.get('name', 'multimodal')}.json")
        if dry_run:
            for cond in exp["conditions"]:
                logger.info("[dry-run] would evaluate condition %s -> %s",
                            cond.get("id"), out_path)
            return {"config": config, "rows": exp["conditions"]}
        result = run_multimodal_robustness(config, output=out_path, quick=smoke)
        return {"config": config, "rows": result["rows"]}

    variants = exp.get("variants") or []
    if not variants:
        logger.warning("No variants declared in %s", config)
        return {"config": config, "runs": []}
    seed_list = list(seeds if seeds is not None else exp.get("seeds", [0]))
    os.makedirs(out_dir, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for var in variants:
        for seed in seed_list:
            vid = var.get("id", "variant")
            run_dir = os.path.join(out_dir, f"{exp.get('name', 'exp')}__{vid}__seed{seed}")
            entry = {"experiment": exp.get("name"), "variant": vid, "seed": seed,
                     "tier": tier, "output_dir": run_dir}
            if dry_run:
                logger.info("[dry-run] would run %s seed=%d -> %s", vid, seed, run_dir)
                runs.append({**entry, "dry_run": True})
                continue

            vcfg = variant_config(cfg, var)
            overrides = [f"seed={seed}"]
            if smoke:
                overrides += ["train.epochs=1", "train.batch_size=2", "train.num_workers=0",
                              "data.synthetic.num_samples=8",
                              "data.synthetic.window_samples=512",
                              "model.backbone.depth=2"]
            tmp_cfg = os.path.join(run_dir, "variant_config.yaml")
            os.makedirs(run_dir, exist_ok=True)
            from ..config import save_resolved
            save_resolved(vcfg, tmp_cfg)

            t0 = time.time()
            try:
                pretrain_main(["--config", tmp_cfg, "--output-dir", run_dir,
                               "--set", *overrides] +
                              (["--max-steps", str(max_steps)] if max_steps else []))
                status = "ok"
                history_path = os.path.join(run_dir, "history.json")
                metrics = {}
                if os.path.exists(history_path):
                    with open(history_path) as f:
                        hist = json.load(f)
                    metrics = hist[-1] if hist else {}
            except Exception as exc:                       # keep the matrix going
                logger.exception("variant %s seed %d failed", vid, seed)
                status, metrics = f"failed: {exc}", {}
            entry.update({"status": status, "seconds": time.time() - t0,
                          "synthetic": smoke or not vcfg.get("data", {}).get("datasets"),
                          **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
            runs.append(entry)
            with open(os.path.join(out_dir, f"{exp.get('name','exp')}_runs.json"), "w") as f:
                json.dump({"config": config, "rows": runs}, f, indent=2)
    return {"config": config, "rows": runs}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Run a PhysioWave experiment matrix")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="./results/experiments")
    p.add_argument("--tier", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    result = run_matrix(args.config, args.out_dir, args.tier, args.seeds,
                        args.smoke, args.max_steps, args.dry_run)
    print(json.dumps({"config": result.get("config"),
                      "runs": len(result.get("rows", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
