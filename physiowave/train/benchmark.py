"""Token-efficiency and multimodal benchmarks.

Reports, per configuration: parameter count, forward FLOPs, peak memory,
throughput, and -- the number this project exists for -- the token count and its
compression ratio against the legacy path.

``N_old = (J + 1) * C * S`` is the legacy sequence length: the legacy wavelet
stage upsamples every band back to ``T`` and stacks the bands on the channel
axis, so the token count grows linearly with the number of decomposition levels.
``N_new = K * S`` is what WAST + topology-aware compression produces, and it is
independent of both ``J`` and ``C``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch

from ..channels.tare import ChannelMeta
from ..config import load_config
from ..data.montages import montage
from ..models.build import build_model, count_parameters
from ..models.legacy import legacy_token_count
from .utils import estimate_flops, peak_memory_mb, pick_device, reset_peak_memory, set_seed

logger = logging.getLogger(__name__)


def _meta_for(C: int, device) -> ChannelMeta:
    """Channel metadata for a benchmark run, using a template montage when possible."""
    for name in ("standard_1020_19", "standard_1010_61", "standard_1010_64"):
        names, xyz = montage(name)
        if len(names) >= C:
            return ChannelMeta(channel_names=names[:C], channel_xyz=xyz[:C].to(device),
                               montage_type="standard_1010", reference_type="original")
    names, xyz = montage("standard_1010_64")
    return ChannelMeta(channel_names=names, channel_xyz=xyz.to(device))


@torch.no_grad()
def benchmark_variant(
    cfg: Dict[str, Any],
    C: int,
    T: int,
    batch_size: int = 4,
    warmup: int = 2,
    iters: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Measure one model configuration on a synthetic batch."""
    device = device or pick_device()
    model_cfg = dict(cfg.get("model", {}) or {})
    is_legacy = model_cfg.get("name") == "legacy"
    if is_legacy:
        model_cfg.setdefault("legacy", {})["in_channels"] = C
        cfg = {**cfg, "model": model_cfg}
    model = build_model(cfg).to(device).eval()

    x = torch.randn(batch_size, C, T, device=device)
    meta = _meta_for(C, device)
    args, kwargs = (x,), {}
    if is_legacy:
        kwargs = {"task": "features"}
    else:
        args = (x, meta)

    for _ in range(warmup):
        model(*args, **kwargs)
    reset_peak_memory(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        out = model(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    if is_legacy:
        J = model_cfg["legacy"].get("max_level", 3)
        patch = model_cfg["legacy"].get("patch_size", (1, 64))
        tok = legacy_token_count(C, J, T, tuple(patch) if isinstance(patch, list) else patch)
        n_tokens = tok["num_tokens"]
        stats = {"N_old_legacy": n_tokens, "N_new": n_tokens, "S": tok["patches_per_time"],
                 "J": J, "C": C}
    else:
        stats = dict(out["token_stats"])
        n_tokens = int(stats.get("N_new", stats.get("N_wast", 0)))

    flops = estimate_flops(model, args, kwargs)
    n_old = int(stats.get("N_old_legacy", n_tokens))
    return {
        "channels": C,
        "window_samples": T,
        "batch_size": batch_size,
        "params": count_parameters(model)["total"],
        "flops_forward": flops,
        "peak_mem_mb": peak_memory_mb(device),
        "latency_s": dt,
        "samples_per_sec": batch_size / max(dt, 1e-9),
        "tokens": n_tokens,
        "tokens_per_sec": n_tokens * batch_size / max(dt, 1e-9),
        "N_old_legacy": n_old,
        "N_new": n_tokens,
        "token_compression_ratio": n_old / max(n_tokens, 1),
        "device": str(device),
        "token_stats": stats,
    }


def run_token_benchmark(config: str = "experiments/token_efficiency",
                        overrides: List[str] = (),
                        output: Optional[str] = None,
                        quick: bool = False) -> Dict[str, Any]:
    """Run the whole token-efficiency sweep declared in the experiment config."""
    cfg = load_config(config, overrides)
    bench = cfg.get("experiment", {}).get("benchmark", {}) or {}
    channels = bench.get("channels", [19, 64])
    windows = bench.get("window_samples", [1024])
    batch = int(bench.get("batch_size", 4))
    variants = bench.get("variants", [{"id": "wast_tare", "model": {}}])
    if quick:
        channels, windows, batch = channels[:1], windows[:1], min(batch, 2)

    rows: List[Dict[str, Any]] = []
    for var in variants:
        for C in channels:
            for T in windows:
                vcfg = _merge_variant(cfg, var)
                if var.get("model", {}).get("name") == "legacy" and C % 2 != 0:
                    logger.warning("Skipping legacy at C=%d: the legacy CrossScaleCAFFN "
                                   "needs an even channel count.", C)
                    continue
                try:
                    row = benchmark_variant(vcfg, C, T, batch)
                    row["status"] = "ok"
                except Exception as exc:
                    # A failure is itself a measurement -- the legacy path runs out
                    # of memory at high channel counts precisely because its
                    # sequence length is (J+1)*C*S. Record it instead of hiding it.
                    row = _failed_row(vcfg, var, C, T, batch, exc)
                    logger.warning("variant %s C=%d T=%d failed (%s): %s",
                                   var["id"], C, T, row["status"], exc)
                row["variant"] = var["id"]
                rows.append(row)
                if row["status"] != "ok":
                    continue
                logger.info("%-16s C=%-3d T=%-5d tokens=%-6d compression=%.2fx "
                            "params=%.2fM samples/s=%.1f",
                            var["id"], C, T, row["tokens"], row["token_compression_ratio"],
                            row["params"] / 1e6, row["samples_per_sec"])
    result = {"benchmark": "token_efficiency", "rows": rows}
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", output)
    return result


def _failed_row(cfg: Dict[str, Any], var: Dict[str, Any], C: int, T: int,
                batch: int, exc: Exception) -> Dict[str, Any]:
    """Row describing a configuration that could not be run, with its token count.

    The token count is analytic, so it is still reported: the point of the table
    is that the legacy sequence length grows as ``(J+1)*C*S``, and "it no longer
    fits" is the strongest possible version of that point.
    """
    text = str(exc).lower()
    status = "oom" if ("memory" in text or "buffer size" in text or "alloc" in text) \
        else "error"
    row: Dict[str, Any] = {"channels": C, "window_samples": T, "batch_size": batch,
                           "status": status, "error": str(exc)[:200]}
    mcfg = cfg.get("model", {}) or {}
    if mcfg.get("name") == "legacy":
        lg = mcfg.get("legacy", {})
        patch = lg.get("patch_size", (1, 64))
        tok = legacy_token_count(C, lg.get("max_level", 3), T,
                                 tuple(patch) if isinstance(patch, list) else patch)
        row.update({"tokens": tok["num_tokens"], "N_old_legacy": tok["num_tokens"],
                    "N_new": tok["num_tokens"], "token_compression_ratio": 1.0})
    return row


def _merge_variant(cfg: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    from ..config import deep_merge

    out = json.loads(json.dumps(cfg))
    for key in ("model", "pretrain", "data", "train"):
        if key in variant:
            out[key] = deep_merge(out.get(key, {}), variant[key])
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="PhysioWave benchmarks")
    p.add_argument("--suite", default="tokens", choices=["tokens", "multimodal"])
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    p.add_argument("--output", default="./results/benchmark_tokens.json")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    set_seed(args.seed)

    if args.suite == "tokens":
        run_token_benchmark(args.config or "experiments/token_efficiency",
                            args.overrides, args.output, args.quick)
    else:
        from .evaluate import run_multimodal_robustness

        run_multimodal_robustness(args.config or "experiments/multimodal_robustness",
                                  args.overrides, args.output, args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
