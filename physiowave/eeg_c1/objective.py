"""
One place that says what the EEG C1 objective is.

The trainer computed its weights from ``objective:``, the visualizer read
``train.spec_recon_weight`` -- a key the config had stopped writing -- and fell
back to a literal 1.0/0.25; the progress script quoted neither. Those numbers
agreed only for as long as nobody changed them. Moving the config to 0.5/0.5
would have left the figures captioned 0.25 with nothing in the pipeline able to
notice: both numbers were defaults, both were plausible, and only one was the
one the optimizer followed.

So the resolution lives here and every consumer calls it -- the trainer, the
startup banner, the figures, the progress report. A default that is wrong is
then wrong in one place and shows up everywhere at once, which is the only
version of that bug anybody finds.

    L_total = spec_weight * L_spec
            + raw_weight  * L_raw
            + fold_kl     * L_foldKL

    L_spec  masked MSE      ( pred_spec, stopgrad(normalised folded wavelet) )
    L_raw   masked SmoothL1 ( pred_raw,  stopgrad(preprocessed EEG), beta )

The canonical configuration is 0.5 / 0.5: two reconstruction terms of equal
standing, not one objective and one hint. See configs/pretrain/eeg_c1_moe.yaml.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: The canonical objective. Every default in this file and in
#: ``masked_reconstruction_loss`` comes from here, so there is one number to
#: change and no second copy to forget.
DEFAULT_OBJECTIVE: Dict[str, object] = {
    "spec_weight": 0.5,
    "raw_weight": 0.5,
    "raw_beta": 0.5,
    "fold_kl": 1e-3,
    "mask_before_frontend": True,
    "normalize_spec_target": True,
}

#: Where an older config kept the same value. Runs before 2026-08 wrote these
#: under ``train:``; they are read, but nothing writes them any more.
LEGACY_TRAIN_KEYS: Dict[str, Tuple[str, ...]] = {
    "spec_weight": ("spec_recon_weight", "spec_weight"),
    "raw_weight": ("raw_recon_weight", "raw_weight"),
    "raw_beta": ("raw_smooth_l1_beta", "raw_beta"),
    "fold_kl": ("fold_kl",),
    "mask_before_frontend": ("mask_before_frontend",),
    "normalize_spec_target": ("normalize_spec_target",),
}

_BOOL_KEYS = ("mask_before_frontend", "normalize_spec_target")


def resolve_eeg_c1_objective(cfg: Optional[Dict]) -> Dict[str, object]:
    """The objective this config actually asks for, as plain Python scalars.

    ``objective:`` wins. A ``train:`` key is read only where ``objective:`` is
    silent, which is what keeps a 2026-07 run's config resolvable without
    keeping its spelling alive anywhere else.
    """
    cfg = cfg or {}
    ocfg = cfg.get("objective") or {}
    tcfg = cfg.get("train") or {}
    out: Dict[str, object] = {}
    for name, default in DEFAULT_OBJECTIVE.items():
        if name in ocfg and ocfg[name] is not None:
            value = ocfg[name]
        else:
            value = None
            for legacy in LEGACY_TRAIN_KEYS[name]:
                if legacy in tcfg and tcfg[legacy] is not None:
                    value = tcfg[legacy]
                    break
            if value is None:
                value = default
        out[name] = bool(value) if name in _BOOL_KEYS else float(value)
    return out


def objective_equation(obj: Optional[Dict] = None) -> str:
    """``0.5*L_spec + 0.5*L_raw + 0.001*L_foldKL`` -- one line, for a caption."""
    o = obj or DEFAULT_OBJECTIVE
    return (f"{float(o['spec_weight']):g}*L_spec"
            f" + {float(o['raw_weight']):g}*L_raw"
            f" + {float(o['fold_kl']):g}*L_foldKL")


def objective_banner(obj: Optional[Dict] = None) -> List[str]:
    """The banner block, unindented. The caller adds its own left margin."""
    o = obj or DEFAULT_OBJECTIVE
    return [
        "objective:",
        f"  {float(o['spec_weight']):g} x spec MSE",
        f"  {float(o['raw_weight']):g} x raw SmoothL1(beta={float(o['raw_beta']):g})",
        f"  {float(o['fold_kl']):g} x ScaleFold KL",
    ]


# --------------------------------------------------------------------------- #
# What may not change across an exact resume
# --------------------------------------------------------------------------- #

#: Model fields whose value changes what the weights MEAN, not merely how they
#: are optimised. ``dropout`` is here because a checkpoint trained without it
#: and one trained with it are two experiments; ``mask_ratio`` because the
#: reconstruction task itself differs.
ARCH_KEYS: Tuple[str, ...] = (
    "embed_dim", "depth", "num_heads", "mlp_ratio", "norm", "ffn", "qk_norm",
    "max_level", "wave_kernel_size", "wavelet_names", "wave_init_mode",
    "use_separate_channel", "scale_fold", "fold_synthesis", "fold_gamma",
    "masking_strategy", "importance_ratio", "mask_ratio", "dropout",
    "channel_encoding", "channel_injection", "channel_embed_dim",
    # ECG only (physiowave.ecg_c1): masking the limb leads as one unit is a
    # different reconstruction task. Absent from every EEG config, so it is
    # skipped there.
    "lead_group_masking",
)

#: Schedule fields. A resume restores the optimizer and the cosine's position,
#: both of which count in units of one epoch's steps.
SCHEDULE_KEYS: Tuple[str, ...] = ("steps_per_epoch", "grad_accumulation_steps")


def _norm(v):
    """Compare 1.0e-3 to 0.001 and [sym4] to ('sym4',) as equal."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 12)
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return v


def resume_incompatibilities(old_cfg: Optional[Dict],
                             new_cfg: Optional[Dict],
                             sampler_state: Optional[Dict] = None
                             ) -> List[Tuple[str, object, object]]:
    """``[(dotted key, checkpoint value, this run's value)]``, empty if it fits.

    An exact resume claims that this process is a continuation of the one that
    wrote the checkpoint: same loss, same task, same schedule. Changing
    spec/raw from 1.0/0.25 to 0.5/0.5 is a different loss, and the optimizer
    moments, the cosine's position and the recorded best scores all belong to
    the old one. The result is not wrong so much as unlabelled -- one curve in
    metrics_epoch.jsonl covering two objectives, with nothing in it saying
    where the change was.

    ``--init-from`` is the operation for that, and the refusal says so.
    """
    old_cfg = old_cfg or {}
    new_cfg = new_cfg or {}
    diffs: List[Tuple[str, object, object]] = []

    old_obj = resolve_eeg_c1_objective(old_cfg)
    new_obj = resolve_eeg_c1_objective(new_cfg)
    for key in DEFAULT_OBJECTIVE:
        if _norm(old_obj[key]) != _norm(new_obj[key]):
            diffs.append((f"objective.{key}", old_obj[key], new_obj[key]))

    old_m = old_cfg.get("model") or {}
    new_m = new_cfg.get("model") or {}
    for key in ARCH_KEYS:
        if key not in old_m and key not in new_m:
            continue
        a, b = old_m.get(key), new_m.get(key)
        if _norm(a) != _norm(b):
            diffs.append((f"model.{key}", a, b))

    old_t = old_cfg.get("train") or {}
    new_t = new_cfg.get("train") or {}
    for key in SCHEDULE_KEYS:
        a, b = old_t.get(key), new_t.get(key)
        if key == "steps_per_epoch":
            # null means "derived from the mixture", and what it derived TO is
            # in the sampler state -- which is the number the restored
            # scheduler is counting in. A config that says null on both sides
            # but resolved to two different lengths is still a mismatch.
            if a is None and sampler_state:
                a = sampler_state.get("steps_per_epoch")
            if a is None or b is None:
                continue
        if _norm(a) != _norm(b):
            diffs.append((f"train.{key}", a, b))
    return diffs


def resume_refusal_message(path: str,
                           diffs: List[Tuple[str, object, object]]) -> str:
    lines = [
        f"refusing to resume {path}",
        "",
        "  This checkpoint belongs to a different objective or schedule.",
        "  Use --init-from for weights-only initialization.",
        "",
        "  what differs (checkpoint -> this run):",
    ]
    for key, old, new in diffs:
        lines.append(f"    {key:<38s} {old!r} -> {new!r}")
    lines += [
        "",
        "  --resume continues ONE run: it restores the optimizer moments, the",
        "  cosine's position, the sampler and the best-so-far scores, all of",
        "  which were accumulated under the values on the left. --init-from",
        "  takes the weights and starts a new run:",
        f"    --init-from {path}",
    ]
    return "\n".join(lines)
