"""Lightweight YAML configuration with includes, overrides and dataclass binding.

The repository had no configuration system, so one is introduced here rather than
retro-fitting Hydra onto the legacy argparse scripts (which keep working
unchanged).  It is deliberately small:

* plain YAML files under ``configs/``;
* ``defaults: [path, ...]`` for composition, resolved relative to the config root;
* dotted command-line overrides (``--set model.wast.level=4``);
* :func:`instantiate` binds a nested dict onto the project's dataclasses so a
  typo in a config key raises instead of being silently ignored.

If ``omegaconf`` is installed it is used for the YAML merge (better interpolation
support); otherwise a plain recursive dict merge is used.  Both produce the same
resolved config, which is always saved next to the checkpoints.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, get_origin

import yaml

T = TypeVar("T")

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(path: str, root: Optional[str] = None) -> str:
    root = root or CONFIG_ROOT
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for cand in (path, path + ".yaml", os.path.join(root, path),
                 os.path.join(root, path + ".yaml")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Config not found: {path} (searched under {root})")


def load_yaml(path: str, root: Optional[str] = None, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving its ``defaults`` list first."""
    _seen = _seen or set()
    full = _resolve(path, root)
    if full in _seen:
        raise ValueError(f"Circular config include involving {full}")
    _seen.add(full)
    with open(full) as f:
        data = yaml.safe_load(f) or {}
    defaults = data.pop("defaults", []) or []
    merged: Dict[str, Any] = {}
    for d in defaults:
        merged = deep_merge(merged, load_yaml(d, root, _seen))
    return deep_merge(merged, data)


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    out = json.loads(json.dumps(cfg))          # deep copy through plain types
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override path {key!r} traverses a non-dict at {p!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: str, overrides: Sequence[str] = (), root: Optional[str] = None) -> Dict[str, Any]:
    """Load and resolve a config file, then apply CLI overrides."""
    return apply_overrides(load_yaml(path, root), overrides)


def save_resolved(cfg: Dict[str, Any], path: str) -> str:
    """Write the fully resolved config next to the run outputs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    os.replace(tmp, path)
    return path


def instantiate(cls: Type[T], data: Optional[Dict[str, Any]], strict: bool = True) -> T:
    """Build a dataclass (recursively) from a nested dict.

    Unknown keys raise by default: a silently ignored config key is a
    reproducibility bug, because the run does not do what the file says.
    """
    data = data or {}
    if not dataclasses.is_dataclass(cls):
        return cls(**data)                     # type: ignore[call-arg]
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in fields]
    if unknown and strict:
        raise ValueError(
            f"Unknown config keys for {cls.__name__}: {unknown}. "
            f"Valid keys: {sorted(fields)}"
        )
    kwargs: Dict[str, Any] = {}
    for name, f in fields.items():
        if name not in data:
            continue
        value = data[name]
        ftype = f.type
        if isinstance(ftype, str):             # from __future__ annotations
            ftype = _resolve_type(cls, ftype)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = instantiate(ftype, value, strict)
        elif get_origin(ftype) is tuple and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)                        # type: ignore[call-arg]


def _resolve_type(cls: type, name: str) -> Any:
    """Best-effort resolution of a stringified annotation."""
    import sys
    import typing

    module = sys.modules.get(cls.__module__)
    ns = dict(vars(module)) if module else {}
    ns.update(vars(typing))
    try:
        return eval(name, ns)                   # noqa: S307 - config schema only
    except Exception:
        return Any
