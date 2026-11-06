"""Checkpoint I/O: atomic writes, full training state, and explicit migration.

Two rules the project spec calls out and this module enforces:

* a checkpoint stores **model, optimizer, scheduler, scaler and RNG state**, plus
  the resolved config, environment versions and git commit, so a resumed run is
  reproducible rather than merely restartable;
* loading a checkpoint whose keys do not match the current model **never silently
  ignores** the mismatch.  :func:`migrate_state_dict` reports every remapped,
  missing and unexpected key, and :func:`load_model_state` refuses to load unless
  the caller explicitly allows a partial load.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Renames applied when loading an old checkpoint into the new package layout.
LEGACY_KEY_MAP: Dict[str, str] = {
    "wavelet_decomp.": "legacy_wavelet_decomp.",
    "patch_embed.": "legacy_patch_embed.",
}


def git_commit(repo_dir: Optional[str] = None) -> str:
    """Current git commit, or ``'unknown'`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir or os.getcwd(),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, str]:
    """Versions that change numerics, recorded alongside every checkpoint."""
    import platform
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda or "cpu",
        "git_commit": git_commit(),
    }
    try:
        import pywt
        info["pywt"] = pywt.__version__
    except Exception:
        pass
    return info


def rng_state() -> Dict[str, Any]:
    """Snapshot of every RNG the training loop touches."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state saved by :func:`rng_state`."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as exc:                              # different device count
            logger.warning("Could not restore CUDA RNG state: %s", exc)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Atomically write a full training checkpoint.

    The payload is written to ``<path>.tmp`` and then ``os.replace``-d onto
    ``path``, so a crash mid-write can never leave a truncated checkpoint behind.
    """
    module = model.module if hasattr(model, "module") else model
    payload: Dict[str, Any] = {
        "model": module.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
        "env": environment_info(),
        "rng": rng_state(),
        "format_version": 2,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


@dataclass
class MigrationReport:
    """Result of :func:`migrate_state_dict`, always logged, never swallowed."""

    remapped: List[Tuple[str, str]]
    missing: List[str]
    unexpected: List[str]
    matched: int

    @property
    def clean(self) -> bool:
        return not self.missing and not self.unexpected

    def describe(self) -> str:
        lines = [f"matched {self.matched} tensors"]
        if self.remapped:
            lines.append(f"remapped {len(self.remapped)} keys, e.g. "
                         + ", ".join(f"{a} -> {b}" for a, b in self.remapped[:5]))
        if self.missing:
            lines.append(f"MISSING {len(self.missing)} keys the model needs: "
                         + ", ".join(self.missing[:10])
                         + (" ..." if len(self.missing) > 10 else ""))
        if self.unexpected:
            lines.append(f"UNEXPECTED {len(self.unexpected)} keys in the checkpoint: "
                         + ", ".join(self.unexpected[:10])
                         + (" ..." if len(self.unexpected) > 10 else ""))
        return "; ".join(lines)


def migrate_state_dict(
    state: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    key_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], MigrationReport]:
    """Remap an old checkpoint onto ``model`` and report exactly what happened.

    Shape mismatches are treated as *unexpected* rather than being force-fitted:
    a silently truncated or broadcast tensor is far harder to debug later than a
    loud refusal now.
    """
    key_map = key_map if key_map is not None else LEGACY_KEY_MAP
    target = model.module if hasattr(model, "module") else model
    tgt_state = target.state_dict()

    out: Dict[str, torch.Tensor] = {}
    remapped: List[Tuple[str, str]] = []
    unexpected: List[str] = []

    for k, v in state.items():
        nk = k
        for old, new in key_map.items():
            if not nk.startswith(old):
                continue
            candidate = new + nk[len(old):]
            # A MIGRATION, NOT A RENAME. The rename used to be unconditional,
            # so a model that legitimately owns `patch_embed.` -- the EEG C1
            # downstream encoder does -- had its patcher renamed out of
            # existence on every save/load round trip: reported as "remapped",
            # loaded as missing, and evaluated with a freshly initialised
            # patcher while the log said the checkpoint had loaded. It is only
            # a migration when the target wants the new name and does not know
            # the old one.
            if candidate in tgt_state and nk not in tgt_state:
                nk = candidate
                remapped.append((k, nk))
            break
        if nk in tgt_state and tgt_state[nk].shape == v.shape:
            out[nk] = v
        else:
            unexpected.append(k if nk == k else f"{k} (as {nk})")

    missing = [k for k in tgt_state if k not in out]
    return out, MigrationReport(remapped, missing, unexpected, len(out))


def load_model_state(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    strict: bool = True,
    key_map: Optional[Dict[str, str]] = None,
) -> MigrationReport:
    """Load ``state`` into ``model``, migrating keys and logging the outcome.

    Args:
        strict: if True (default) any missing or unexpected key raises.  Set it to
            False only when a partial load is intended (e.g. loading a pretrained
            encoder into a model that has an extra task head), in which case the
            report is logged at WARNING level.
    """
    migrated, report = migrate_state_dict(state, model, key_map)
    if not report.clean:
        message = f"Checkpoint key mismatch: {report.describe()}"
        if strict:
            raise RuntimeError(
                message + "\nPass strict=False to accept a partial load, or fix the "
                          "config so the architecture matches the checkpoint."
            )
        logger.warning(message)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(migrated, strict=False)
    logger.info("Loaded checkpoint: %s", report.describe())
    return report


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    strict: bool = True,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore the full training state."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        load_model_state(model, payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng"):
        set_rng_state(payload["rng"])
    return payload
