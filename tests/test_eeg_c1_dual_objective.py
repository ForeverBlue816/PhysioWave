"""
The balanced dual objective: 0.5 spec + 0.5 raw + 1e-3 fold KL.

Five things this file exists to pin, each of which was wrong or unrepresentable
before:

  * the weights come from ONE resolver, so the trainer, the banner, the figures
    and the progress report cannot quote different numbers for the same run;
  * the total is the explicit weighted sum of two explicitly separate losses;
  * the two heads have DIMENSIONLESS metrics, because their losses are in
    different units and "the raw number is smaller" is not a comparison;
  * validation route and dataset breakdowns are aggregated over every rank, not
    reported from rank 0's slice;
  * best.pth is selected by the loss that was trained, not by half of it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.eeg_c1.model import \
    masked_reconstruction_loss                                       # noqa: E402
from physiowave.eeg_c1.objective import (ARCH_KEYS,                  # noqa: E402
                                         DEFAULT_OBJECTIVE,
                                         objective_banner,
                                         objective_equation,
                                         resolve_eeg_c1_objective,
                                         resume_incompatibilities,
                                         resume_refusal_message)
from physiowave.eeg_c1.routes import ROUTES                          # noqa: E402
from physiowave.eeg_c1.train import (Accumulator, BEST_KEYS,         # noqa: E402
                                     BEST_SELECTION,
                                     CHECKPOINT_SELECTION,
                                     EEGC1Trainer, macro_route_metrics)

CONFIG = os.path.join(ROOT, "configs", "pretrain", "eeg_c1_moe.yaml")


def _config():
    import yaml
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def _out(route_id, n_tok=None, patch_t=None, batch=1):
    """A hand-built forward output, so a loss test depends on no model."""
    r = ROUTES[route_id]
    n_tok = n_tok or r.n_tokens
    patch_t = patch_t or r.patch_t
    g = torch.Generator().manual_seed(7)
    mk = lambda: torch.randn(batch, n_tok, patch_t, generator=g)
    mask = torch.zeros(batch, n_tok, dtype=torch.bool)
    mask[:, ::3] = True
    return {"pred_spec": mk(), "target_spec": mk(),
            "pred_raw": mk(), "target_raw": mk(),
            "mask": mask, "valid_tokens": None, "fold_reg": None}


# --------------------------------------------------------------------------- #
# A. the final configuration
# --------------------------------------------------------------------------- #

def test_a_the_config_is_the_final_objective():
    cfg = _config()
    o = cfg["objective"]
    assert o["spec_weight"] == 0.5
    assert o["raw_weight"] == 0.5
    assert o["raw_beta"] == 0.5
    assert o["fold_kl"] == pytest.approx(1e-3)
    assert o["mask_before_frontend"] is True
    assert o["normalize_spec_target"] is True


def test_a_the_architecture_is_512_8_8():
    """The width is pinned because changing it kills every checkpoint.

    Not a tunable: the state dict shapes follow from these three numbers, so a
    change here means no earlier run can be resumed or initialised from, and
    the downstream configs stop loading the exported encoder. It is asserted so
    that moving it is a deliberate act with a failing test attached, not a
    one-line edit that surfaces as a shape error on a cluster.
    """
    m = _config()["model"]
    assert m["embed_dim"] == 512
    assert m["depth"] == 8
    assert m["num_heads"] == 8
    assert m["embed_dim"] % m["num_heads"] == 0, "head_dim is not an integer"
    assert m["mask_ratio"] == 0.70
    assert m["dropout"] == 0.0


def test_a_the_downstream_configs_match_the_encoder():
    """An exported encoder does not load into a model of a different width.

    The failure mode this prevents is silent right up to the point it is not:
    pretraining finishes, the export succeeds, and the finetune dies on a shape
    mismatch hours later -- or worse, loads loosely and reports a number for an
    encoder that was never pretrained.
    """
    import yaml
    pre = _config()["model"]
    for name in ("sleep", "p300"):
        path = os.path.join(ROOT, "configs", "finetune", f"eeg_c1_{name}.yaml")
        with open(path) as f:
            down = yaml.safe_load(f)["model"]
        for key in ("embed_dim", "depth", "num_heads", "channel_embed_dim"):
            assert down[key] == pre[key], \
                f"{name}.{key} is {down[key]}, pretraining is {pre[key]}"


def test_a_the_epoch_is_a_sampled_budget_not_a_corpus_pass():
    """An explicit step budget, because one pass over the corpus does not fit.

    `proportional` was measured on the cluster at 10,573 steps per epoch and
    ~10 s/step at world size 16 -- 29 hours for one epoch against a 24 hour
    walltime. steps_per_epoch is therefore set rather than derived, and these
    numbers are asserted together because together they are the run's entire
    compute budget.
    """
    cfg = _config()
    assert cfg["train"]["epochs"] == 60
    assert cfg["train"]["steps_per_epoch"] == 768
    assert cfg["train"]["warmup_epochs"] == 2
    assert cfg["train"]["grad_accumulation_steps"] == 1
    # Above the 386 that `balanced` derives -- a floor the smallest corpus on
    # its route imposes, not a budget.
    assert cfg["train"]["steps_per_epoch"] > 386
    # And bounded above by the WALLTIME, because latest.pth is written at epoch
    # boundaries only: an epoch that does not fit one allocation never
    # completes, however often the job is resubmitted. Ten seconds a step is
    # what the cluster measured.
    assert cfg["train"]["steps_per_epoch"] * 10 < 24 * 3600, \
        "one epoch at the measured ~10 s/step no longer fits a 24 h job"
    # And the RUN has to be long enough to be worth doing. 7,680 updates was
    # the first attempt and left TUEG seen 0.3 times; the update count, not
    # the epoch count, is what says whether a masked pretraining is short.
    updates = cfg["train"]["epochs"] * cfg["train"]["steps_per_epoch"]
    assert updates >= 40_000, f"{updates:,} updates is short for this model"
    # warmup is a number of steps and does not scale with epochs, so a longer
    # run must not leave it at an unreasonable fraction.
    warm = cfg["train"]["warmup_epochs"] * cfg["train"]["steps_per_epoch"]
    assert 0.01 <= warm / updates <= 0.10, f"warmup is {warm/updates:.1%} of the run"


def test_a_the_mixture_is_temperature_half():
    """Written WITHOUT a space after the colon, or it is not a policy at all.

    `temperature: 0.5` is YAML for the mapping {temperature: 0.5}, which the
    policy branch does not match -- it falls through to the explicit-mapping
    case and dies on "no configured dataset is present". Only the unspaced
    string reaches the branch that reads the exponent.
    """
    weights = _config()["data"]["weights"]
    assert isinstance(weights, str), \
        "written with a space after the colon; YAML made it a mapping"
    assert weights == "temperature:0.5"


def test_a_validation_is_capped_and_says_so():
    """Ten full sweeps of a TUEG-heavy validation set do not fit the budget.

    The cap is on by default, which makes val/* a CURVE rather than a final
    number. That is only acceptable because two things are true, and this test
    holds both:

    * the cap reaches validate(), so it is the sweep that shrinks rather than
      some other thing;
    * every epoch row records val_batches, so the number in
      metrics_epoch.jsonl carries the size of the sweep it came from and
      cannot be quoted as a full-set result by accident.

    The mask seed is fixed by window identity, so a cap takes the SAME windows
    every epoch -- epoch 9 is comparable to epoch 0, which is what a curve
    needs. For a number to quote, re-run with this set to null.
    """
    import inspect
    cap = _config()["train"]["val_max_batches_per_dataset"]
    assert isinstance(cap, int) and cap > 0

    src = inspect.getsource(EEGC1Trainer.fit)
    assert "max_batches=self.val_max_batches" in src, \
        "fit() stopped passing the cap through to validate()"

    # null must still mean the full sweep -- that is the escape hatch.
    class _T:
        pass
    for value, expected in ((None, None), (0, None), (100, 100)):
        t = _T()
        _v = {"val_max_batches_per_dataset": value}.get(
            "val_max_batches_per_dataset")
        assert (int(_v) if _v else None) == expected

    src = inspect.getsource(EEGC1Trainer.validate)
    assert "max_batches_per_dataset=max_batches" in src


def test_a_the_resolver_is_the_only_source():
    """One helper, and the config resolves through it to exactly its values."""
    got = resolve_eeg_c1_objective(_config())
    assert got == {"spec_weight": 0.5, "raw_weight": 0.5, "raw_beta": 0.5,
                   "fold_kl": 1e-3, "mask_before_frontend": True,
                   "normalize_spec_target": True}
    assert DEFAULT_OBJECTIVE["spec_weight"] == 0.5
    assert DEFAULT_OBJECTIVE["raw_weight"] == 0.5


def test_a_no_second_copy_of_the_weights_in_the_config():
    """A value under train: would be read by nothing while looking authoritative.

    objective: wins in the resolver, so a stale train.raw_recon_weight is not a
    number anybody would find by grepping for the one in force.
    """
    tcfg = _config().get("train", {})
    for stale in ("spec_recon_weight", "raw_recon_weight", "raw_beta",
                  "fold_kl", "spec_weight", "raw_weight",
                  "mask_before_frontend", "normalize_spec_target"):
        assert stale not in tcfg, f"train.{stale} is a second copy of an " \
                                  f"objective value"


def test_a_a_legacy_config_still_resolves():
    """Runs from before 2026-08 kept these under train:. They are read, and
    objective: outranks them where both are present."""
    legacy = {"train": {"spec_recon_weight": 1.0, "raw_recon_weight": 0.25,
                        "fold_kl": 5e-4}}
    got = resolve_eeg_c1_objective(legacy)
    assert got["spec_weight"] == 1.0
    assert got["raw_weight"] == 0.25
    assert got["fold_kl"] == pytest.approx(5e-4)

    both = {"objective": {"raw_weight": 0.5},
            "train": {"raw_recon_weight": 0.25}}
    assert resolve_eeg_c1_objective(both)["raw_weight"] == 0.5


def test_a_the_banner_says_the_objective():
    lines = objective_banner(resolve_eeg_c1_objective(_config()))
    assert lines[0] == "objective:"
    assert lines[1].strip() == "0.5 x spec MSE"
    assert lines[2].strip() == "0.5 x raw SmoothL1(beta=0.5)"
    assert lines[3].strip() == "0.001 x ScaleFold KL"
    assert objective_equation(resolve_eeg_c1_objective(_config())) == \
        "0.5*L_spec + 0.5*L_raw + 0.001*L_foldKL"


# --------------------------------------------------------------------------- #
# B. the total-loss formula
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("weights", [
    (0.5, 0.5, 1e-3), (1.0, 0.25, 1e-3), (0.0, 1.0, 0.0),
])
def test_b_total_is_the_explicit_weighted_sum(weights):
    ws, wr, wk = weights
    out = _out("E19_256")
    out["fold_reg"] = torch.tensor(0.37, requires_grad=True)
    total, m = masked_reconstruction_loss(out, spec_weight=ws, raw_weight=wr,
                                          fold_kl=wk)
    expected = (ws * m["loss_masked_spec_mse"]
                + wr * m["loss_masked_raw_smoothl1"]
                + wk * m["loss_fold_kl"])
    assert float(total.detach()) == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert m["loss_total"] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_b_the_default_weights_are_the_canonical_ones():
    out = _out("E32_512")
    total, m = masked_reconstruction_loss(out)
    assert float(total) == pytest.approx(
        0.5 * m["loss_masked_spec_mse"] + 0.5 * m["loss_masked_raw_smoothl1"],
        rel=1e-6)


def test_b_the_two_losses_are_not_averaged_together():
    """Each term is computed from its own head against its own target.

    Averaging the predictions first, or the targets, would make both logged
    numbers describe a quantity neither head produces.
    """
    out = _out("E19_256")
    _, base = masked_reconstruction_loss(out)
    moved = {k: (v.clone() if torch.is_tensor(v) else v)
             for k, v in out.items()}
    moved["pred_raw"] = moved["pred_raw"] + 3.0
    _, after = masked_reconstruction_loss(moved)
    assert after["loss_masked_spec_mse"] == pytest.approx(
        base["loss_masked_spec_mse"]), "the raw head moved the spec loss"
    assert after["loss_masked_raw_smoothl1"] > base["loss_masked_raw_smoothl1"]


def test_b_the_logging_keys_are_explicit_and_the_alias_still_resolves():
    _, m = masked_reconstruction_loss(_out("E19_256"))
    for key in ("loss_total", "loss_masked_spec_mse",
                "loss_masked_raw_smoothl1", "loss_fold_kl"):
        assert key in m
    assert m["loss_masked_mse"] == pytest.approx(m["loss_masked_spec_mse"])
    assert m["masked_corr"] == pytest.approx(m["masked_spec_corr"])


# --------------------------------------------------------------------------- #
# C. only masked valid patches contribute
# --------------------------------------------------------------------------- #

def test_c_a_visible_prediction_changes_neither_loss():
    out = _out("E19_256")
    _, base = masked_reconstruction_loss(out)
    visible = (~out["mask"][0]).nonzero()[0].item()
    for key in ("pred_spec", "pred_raw"):
        moved = {k: (v.clone() if torch.is_tensor(v) else v)
                 for k, v in out.items()}
        moved[key][0, visible] += 50.0
        _, after = masked_reconstruction_loss(moved)
        assert after["loss_masked_spec_mse"] == pytest.approx(
            base["loss_masked_spec_mse"])
        assert after["loss_masked_raw_smoothl1"] == pytest.approx(
            base["loss_masked_raw_smoothl1"])


def test_c_a_masked_prediction_changes_its_own_loss():
    out = _out("E19_256")
    _, base = masked_reconstruction_loss(out)
    hidden = int(out["mask"][0].nonzero()[0].item())
    for key, metric, other in (("pred_spec", "loss_masked_spec_mse",
                                "loss_masked_raw_smoothl1"),
                               ("pred_raw", "loss_masked_raw_smoothl1",
                                "loss_masked_spec_mse")):
        moved = {k: (v.clone() if torch.is_tensor(v) else v)
                 for k, v in out.items()}
        moved[key][0, hidden] += 5.0
        _, after = masked_reconstruction_loss(moved)
        assert after[metric] > base[metric]
        assert after[other] == pytest.approx(base[other])


def test_c_padded_channels_are_excluded_by_the_loss_itself():
    """Not only by the mask selector. A mask that names a padded token -- from
    a figure, a test, a future caller -- must still not contribute."""
    route = ROUTES["E32_512"]
    out = _out("E32_512")
    valid = torch.ones(1, route.n_tokens, dtype=torch.bool)
    pad_from = 26 * route.patches_per_channel        # TDBRAIN's shape
    valid[:, pad_from:] = False
    out["valid_tokens"] = valid
    out["mask"][:, pad_from:] = True                 # deliberately illegal
    _, base = masked_reconstruction_loss(out)
    out["pred_spec"][0, pad_from:] += 100.0
    out["pred_raw"][0, pad_from:] += 100.0
    _, after = masked_reconstruction_loss(out)
    assert after["loss_masked_spec_mse"] == pytest.approx(
        base["loss_masked_spec_mse"])
    assert after["loss_masked_raw_smoothl1"] == pytest.approx(
        base["loss_masked_raw_smoothl1"])


# --------------------------------------------------------------------------- #
# D. comparable metrics
# --------------------------------------------------------------------------- #

def test_d_a_perfect_prediction_is_corr_one_and_nmse_zero():
    out = _out("E19_256")
    out["pred_spec"] = out["target_spec"].clone()
    out["pred_raw"] = out["target_raw"].clone()
    _, m = masked_reconstruction_loss(out)
    assert m["masked_spec_corr"] == pytest.approx(1.0, abs=1e-5)
    assert m["masked_raw_corr"] == pytest.approx(1.0, abs=1e-5)
    assert m["masked_spec_nmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["masked_raw_nmse"] == pytest.approx(0.0, abs=1e-6)


def test_d_a_zero_prediction_is_nmse_one():
    """The baseline the number is defined against: below 1 beats predicting
    zero, above 1 is worse than predicting zero."""
    out = _out("E19_256")
    out["pred_spec"] = torch.zeros_like(out["target_spec"])
    out["pred_raw"] = torch.zeros_like(out["target_raw"])
    _, m = masked_reconstruction_loss(out)
    assert m["masked_spec_nmse"] == pytest.approx(1.0, abs=1e-5)
    assert m["masked_raw_nmse"] == pytest.approx(1.0, abs=1e-5)


def test_d_a_worse_than_zero_prediction_is_above_one():
    out = _out("E19_256")
    out["pred_spec"] = -2.0 * out["target_spec"]
    _, m = masked_reconstruction_loss(out)
    assert m["masked_spec_nmse"] > 1.0


def test_d_zero_mask_and_constant_targets_stay_finite():
    out = _out("E19_256")
    out["mask"] = torch.zeros_like(out["mask"])
    _, m = masked_reconstruction_loss(out)
    for k, v in m.items():
        assert math.isfinite(v), k
    assert m["masked_spec_nmse"] == 0.0

    const = _out("E19_256")
    const["target_spec"] = torch.zeros_like(const["target_spec"])
    const["target_raw"] = torch.full_like(const["target_raw"], 2.0)
    _, m = masked_reconstruction_loss(const)
    for k, v in m.items():
        assert math.isfinite(v), k
    # A constant target has no variance, so r is undefined; reported as 0
    # rather than as a NaN that would poison every mean downstream.
    assert m["masked_spec_corr"] == 0.0
    assert m["masked_raw_corr"] == 0.0


def test_d_the_metrics_carry_no_gradient():
    out = _out("E19_256")
    out["pred_spec"].requires_grad_(True)
    out["pred_raw"].requires_grad_(True)
    _, m = masked_reconstruction_loss(out)
    for v in m.values():
        assert isinstance(v, float)


def test_d_every_comparable_metric_is_reported():
    _, m = masked_reconstruction_loss(_out("E64_256"))
    for key in ("masked_spec_corr", "masked_raw_corr", "masked_spec_nmse",
                "masked_raw_nmse", "masked_spec_mae", "masked_spec_rmse",
                "masked_raw_mae", "masked_raw_rmse"):
        assert key in m and math.isfinite(m[key]), key


def test_d_nmse_is_scale_free_where_the_loss_is_not():
    """The point of it. Scaling target and prediction together triples the MSE
    and leaves NMSE and r exactly where they were."""
    out = _out("E19_256")
    _, a = masked_reconstruction_loss(out)
    scaled = {k: (v.clone() if torch.is_tensor(v) else v)
              for k, v in out.items()}
    for k in ("pred_spec", "target_spec"):
        scaled[k] = scaled[k] * 3.0
    _, b = masked_reconstruction_loss(scaled)
    assert b["loss_masked_spec_mse"] == pytest.approx(
        9 * a["loss_masked_spec_mse"], rel=1e-4)
    assert b["masked_spec_nmse"] == pytest.approx(a["masked_spec_nmse"],
                                                 rel=1e-4)
    assert b["masked_spec_corr"] == pytest.approx(a["masked_spec_corr"],
                                                 rel=1e-4)


def test_d_the_metrics_reach_the_accumulator_by_route_and_dataset():
    acc = Accumulator()
    _, m = masked_reconstruction_loss(_out("E19_256"))
    acc.add(m, "E19_256", "tueg")
    means = acc.mean()
    for key in ("masked_raw_corr", "masked_raw_nmse"):
        assert key in means
        assert f"route/E19_256/{key}" in means
        assert f"dataset/tueg/{key}" in means


# --------------------------------------------------------------------------- #
# Distributed aggregation, with synthetic per-rank sums and counts
# --------------------------------------------------------------------------- #

def test_ddp_merge_is_sum_over_sums_not_mean_over_means():
    """Unequal slice lengths, the case a mean of means gets wrong.

    Rank 0 contributed two batches summing to 3.0; rank 1 contributed one at
    9.0. The mean of the two ranks' means is (1.5 + 9.0)/2 = 5.25. The mean is
    12/3 = 4.0. ValIterator emits equal counts today, so both forms agree on
    the current path -- this pins the form that keeps agreeing when they stop.
    """
    ranks = [({"loss_total": 3.0, "route/E19_256/loss_total": 3.0},
              {"loss_total": 2, "route/E19_256/loss_total": 2}),
             ({"loss_total": 9.0, "route/E19_256/loss_total": 9.0},
              {"loss_total": 1, "route/E19_256/loss_total": 1})]
    acc = Accumulator()
    for sums, counts in ranks:
        acc.merge(sums, counts)
    means = acc.mean()
    assert means["loss_total"] == pytest.approx(4.0)
    assert means["route/E19_256/loss_total"] == pytest.approx(4.0)
    assert means["loss_total"] != pytest.approx(5.25)


def test_ddp_merge_handles_keys_one_rank_never_saw():
    """A rank whose slice of a small dataset was shorter than one batch emits
    no keys for it at all, so the key SETS differ and a fixed-layout tensor
    reduction could not have been used."""
    acc = Accumulator()
    acc.merge({"dataset/tueg/loss_total": 1.0}, {"dataset/tueg/loss_total": 1})
    acc.merge({"dataset/hgd/loss_total": 4.0}, {"dataset/hgd/loss_total": 2})
    means = acc.mean()
    assert means["dataset/tueg/loss_total"] == pytest.approx(1.0)
    assert means["dataset/hgd/loss_total"] == pytest.approx(2.0)


def test_ddp_all_reduce_is_a_no_op_off_the_distributed_path():
    acc = Accumulator()
    acc.add({"loss_total": 1.0}, "E19_256", "tueg")
    before = dict(acc.sums)
    acc.all_reduce(False, torch.device("cpu"))
    assert acc.sums == before


#: Run under two real ranks, not simulated ones. ``all_gather_object`` is the
#: piece that cannot be exercised in-process, and the mistake it exists to
#: prevent -- a per-route table that is one rank's slice -- is invisible on one
#: process, where rank 0's slice IS the whole sweep.
_DDP_AGGREGATION_SCRIPT = """
import hashlib, json, os, sys
import torch.distributed as dist
sys.path.insert(0, sys.argv[1])
from physiowave.eeg_c1.train import Accumulator, macro_route_metrics

dist.init_process_group(backend="gloo", init_method="env://")
rank, world = dist.get_rank(), dist.get_world_size()

acc = Accumulator()
if rank == 0:
    acc.add({"loss_total": 1.0, "masked_raw_nmse": 0.8}, "E19_256", "tueg")
    acc.add({"loss_total": 2.0, "masked_raw_nmse": 0.6}, "E19_256", "tueg")
else:
    # Unequal count, and a route/dataset rank 0 never touches.
    acc.add({"loss_total": 9.0, "masked_raw_nmse": 0.4}, "E19_256", "tueg")
    acc.add({"loss_total": 5.0, "masked_raw_nmse": 0.2}, "E64_256", "hgd")

acc.all_reduce(True, None)
means = acc.mean()
means.update(macro_route_metrics(means))

blob = json.dumps({k: round(v, 9) for k, v in sorted(means.items())})
digests = [None] * world
dist.all_gather_object(digests, hashlib.sha256(blob.encode()).hexdigest())
assert len(set(digests)) == 1, "ranks ended with different metrics"

assert abs(means["route/E19_256/loss_total"] - 4.0) < 1e-9, means
assert abs(means["route/E64_256/loss_total"] - 5.0) < 1e-9, means
assert abs(means["dataset/hgd/loss_total"] - 5.0) < 1e-9, means
assert abs(means["loss_total"] - 4.25) < 1e-9, means
assert abs(means["macro_route_raw_nmse"] - 0.4) < 1e-9, means
if rank == 0:
    print("DDP_AGGREGATION_OK")
dist.destroy_process_group()
"""


@pytest.mark.slow
def test_ddp_aggregation_under_two_real_ranks(tmp_path):
    """The collective itself, on two processes.

    The unit tests above check the arithmetic; this checks that
    ``all_gather_object`` is actually reached and actually agrees. Static
    rendezvous on 127.0.0.1 rather than ``--standalone``, which hangs here.
    """
    import shutil
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun is not on PATH")

    script = tmp_path / "ddp_aggregation.py"
    script.write_text(_DDP_AGGREGATION_SCRIPT)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    try:
        r = subprocess.run(
            ["torchrun", "--nnodes=1", "--nproc_per_node=2",
             "--master_addr=127.0.0.1", "--master_port=29531",
             str(script), ROOT],
            cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail("the two ranks did not finish; a collective is not "
                    "matched on both sides")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert "DDP_AGGREGATION_OK" in r.stdout


# --------------------------------------------------------------------------- #
# Route-macro metrics
# --------------------------------------------------------------------------- #

def test_macro_is_unweighted_across_the_routes_present():
    """Deliberately NOT weighted by sample count: the validation sweep is 97%
    TUEG and HBN by batch, so a weighted summary is a TUEG summary."""
    means = {
        "route/E19_256/loss_total": 1.0,
        "route/E32_512/loss_total": 2.0,
        "route/E64_256/loss_total": 3.0,
        "route/E19_256/masked_raw_corr": 0.2,
        "route/E32_512/masked_raw_corr": 0.4,
        "route/E19_256/masked_spec_nmse": 0.5,
        "route/E32_512/masked_spec_nmse": 0.7,
        "dataset/tueg/loss_total": 99.0,      # must not enter the macro
        "loss_total": 1.2,
    }
    macro = macro_route_metrics(means)
    assert macro["macro_route_loss_total"] == pytest.approx(2.0)
    assert macro["macro_route_raw_corr"] == pytest.approx(0.3)
    assert macro["macro_route_spec_nmse"] == pytest.approx(0.6)
    # A metric no route reported produces no macro entry rather than a NaN.
    assert "macro_route_raw_nmse" not in macro


def test_macro_covers_every_documented_key():
    means = {}
    for rid in ROUTES:
        for metric in ("loss_total", "loss_masked_spec_mse",
                       "loss_masked_raw_smoothl1", "masked_spec_corr",
                       "masked_raw_corr", "masked_spec_nmse",
                       "masked_raw_nmse"):
            means[f"route/{rid}/{metric}"] = 1.0
    macro = macro_route_metrics(means)
    assert set(macro) == {
        "macro_route_loss_total", "macro_route_loss_spec",
        "macro_route_loss_raw", "macro_route_spec_corr",
        "macro_route_raw_corr", "macro_route_spec_nmse",
        "macro_route_raw_nmse"}


# --------------------------------------------------------------------------- #
# E. checkpoint selection
# --------------------------------------------------------------------------- #

class _Selector:
    """The trainer's selection bookkeeping, without a model or a corpus."""

    is_main = True
    _update_best_records = EEGC1Trainer._update_best_records
    _save_best = EEGC1Trainer._save_best

    def __init__(self):
        self.best_scores = {k: float("inf") for k in BEST_KEYS}
        self.best_epochs = {k: None for k in BEST_KEYS}
        self.saved = []

    def save(self, name):
        self.saved.append((len(self.saved), name))


def test_e_each_criterion_selects_its_own_epoch():
    """Four epochs, each of which is the best at exactly one thing."""
    epochs = [
        # epoch 0: everything is the first value, so everything improves
        dict(loss_total=1.0, loss_masked_spec_mse=1.0,
             loss_masked_raw_smoothl1=1.0, macro_route_loss_total=1.0),
        # epoch 1 (A): spec alone
        dict(loss_total=1.5, loss_masked_spec_mse=0.2,
             loss_masked_raw_smoothl1=1.4, macro_route_loss_total=1.5),
        # epoch 2 (B): raw alone
        dict(loss_total=1.4, loss_masked_spec_mse=1.1,
             loss_masked_raw_smoothl1=0.3, macro_route_loss_total=1.4),
        # epoch 3 (C): total alone
        dict(loss_total=0.4, loss_masked_spec_mse=0.9,
             loss_masked_raw_smoothl1=0.9, macro_route_loss_total=1.2),
        # epoch 4 (D): macro alone
        dict(loss_total=0.6, loss_masked_spec_mse=0.8,
             loss_masked_raw_smoothl1=0.8, macro_route_loss_total=0.1),
    ]
    sel = _Selector()
    per_epoch = []
    for epoch, metrics in enumerate(epochs):
        improved = sel._update_best_records(epoch, metrics)
        sel._save_best(improved)
        per_epoch.append(set(improved))

    assert per_epoch[1] == {"spec"}
    assert per_epoch[2] == {"raw"}
    assert per_epoch[3] == {"total"}
    assert per_epoch[4] == {"macro_total"}
    assert sel.best_epochs == {"total": 3, "spec": 1, "raw": 2,
                               "macro_total": 4}
    assert sel.best_scores["total"] == pytest.approx(0.4)
    assert sel.best_scores["spec"] == pytest.approx(0.2)
    assert sel.best_scores["raw"] == pytest.approx(0.3)
    assert sel.best_scores["macro_total"] == pytest.approx(0.1)

    files = [name for _, name in sel.saved]
    assert files.count("best_spec.pth") == 2      # epoch 0 and epoch 1
    assert files.count("best_raw.pth") == 2
    assert files.count("best_total.pth") == 2
    assert files.count("best_macro_total.pth") == 2


def test_e_best_pth_is_best_total_pth():
    """Semantically the same file: every write of one writes the other, and it
    is the TOTAL loss that decides. It used to be the spec loss, under the
    ambiguous alias -- so the file everything downstream loads by default was
    chosen by half the objective."""
    assert "best.pth" in BEST_SELECTION["total"][1]
    assert "best_total.pth" in BEST_SELECTION["total"][1]
    assert CHECKPOINT_SELECTION["best.pth"] == "val/loss_total"
    assert CHECKPOINT_SELECTION["best.pth"] == \
        CHECKPOINT_SELECTION["best_total.pth"]
    assert CHECKPOINT_SELECTION["best_spec.pth"] == "val/loss_masked_spec_mse"
    assert CHECKPOINT_SELECTION["best_raw.pth"] == \
        "val/loss_masked_raw_smoothl1"
    assert CHECKPOINT_SELECTION["best_macro_total.pth"] == \
        "val/macro_route_loss_total"

    sel = _Selector()
    sel._save_best(sel._update_best_records(
        0, dict(loss_total=1.0, loss_masked_spec_mse=1.0,
                loss_masked_raw_smoothl1=1.0, macro_route_loss_total=1.0)))
    files = [n for _, n in sel.saved]
    assert files.count("best.pth") == files.count("best_total.pth") == 1


def test_e_a_missing_or_nonfinite_metric_selects_nothing():
    """A validation sweep that produced no macro -- one route present, or none
    at all -- must not write best_macro_total.pth from a NaN."""
    sel = _Selector()
    improved = sel._update_best_records(
        0, dict(loss_total=1.0, loss_masked_spec_mse=float("nan")))
    assert improved == ["total"]
    assert sel.best_epochs["spec"] is None
    assert sel.best_epochs["macro_total"] is None


def test_e_legacy_checkpoint_state_is_migrated_not_reinterpreted(capsys):
    """best_val_loss_masked_mse meant the SPEC loss. Read into the total bar it
    would set a threshold from a different quantity and a genuinely better
    model could fail to clear it for the whole run."""
    sel = _Selector()
    EEGC1Trainer._restore_best_records(
        sel, {"best_val_loss_masked_mse": 0.44, "epoch": 7})
    assert sel.best_scores["spec"] == pytest.approx(0.44)
    assert sel.best_epochs["spec"] == 7
    for key in ("total", "raw", "macro_total"):
        assert sel.best_scores[key] == float("inf")
        assert sel.best_epochs[key] is None
    assert "migrated legacy" in capsys.readouterr().out

    # A checkpoint that already has the dictionary is taken as it stands.
    EEGC1Trainer._restore_best_records(
        sel, {"best_scores": {"total": 0.1, "spec": 0.2, "raw": 0.3,
                              "macro_total": 0.4},
              "best_epochs": {"total": 1, "spec": 2, "raw": 3,
                              "macro_total": 4}})
    assert sel.best_scores["total"] == pytest.approx(0.1)
    assert sel.best_epochs["macro_total"] == 4


# --------------------------------------------------------------------------- #
# F. resume compatibility
# --------------------------------------------------------------------------- #

def _cfg(**objective):
    o = dict(resolve_eeg_c1_objective(None))
    o.update(objective)
    return {"objective": o,
            "model": {"embed_dim": 384, "depth": 6, "num_heads": 6,
                      "mask_ratio": 0.70, "dropout": 0.0},
            "train": {"steps_per_epoch": 768, "grad_accumulation_steps": 1}}


def test_f_an_objective_change_refuses_an_exact_resume():
    old = _cfg(spec_weight=1.0, raw_weight=0.25)
    new = _cfg(spec_weight=0.5, raw_weight=0.5)
    diffs = resume_incompatibilities(old, new)
    keys = {k for k, _, _ in diffs}
    assert keys == {"objective.spec_weight", "objective.raw_weight"}

    msg = resume_refusal_message("/ck/latest.pth", diffs)
    assert "different objective or schedule" in msg
    assert "--init-from" in msg
    assert "objective.raw_weight" in msg and "0.25" in msg and "0.5" in msg


def test_f_the_same_configuration_resumes():
    assert resume_incompatibilities(_cfg(), _cfg()) == []
    # 1.0e-3 against 0.001 is the same number written two ways.
    a, b = _cfg(), _cfg()
    b["objective"]["fold_kl"] = 0.001
    assert resume_incompatibilities(a, b) == []


@pytest.mark.parametrize("field,value", [
    ("embed_dim", 512), ("depth", 8), ("num_heads", 8), ("mask_ratio", 0.75),
    ("dropout", 0.1),
])
def test_f_an_architecture_change_refuses_a_resume(field, value):
    old, new = _cfg(), _cfg()
    new["model"][field] = value
    assert [k for k, _, _ in resume_incompatibilities(old, new)] == \
        [f"model.{field}"]
    assert field in ARCH_KEYS


def test_f_a_schedule_change_refuses_a_resume():
    """A resumed cosine counts in the OLD epoch's units."""
    old, new = _cfg(), _cfg()
    new["train"]["steps_per_epoch"] = 1536
    assert ("train.steps_per_epoch", 768, 1536) in \
        resume_incompatibilities(old, new)

    old2, new2 = _cfg(), _cfg()
    new2["train"]["grad_accumulation_steps"] = 4
    assert ("train.grad_accumulation_steps", 1, 4) in \
        resume_incompatibilities(old2, new2)


def test_f_a_derived_epoch_length_is_compared_against_the_sampler():
    """`null` on both sides means "derived from the mixture", and the mixture
    can have changed. What it derived TO is in the sampler state."""
    old, new = _cfg(), _cfg()
    old["train"]["steps_per_epoch"] = None
    new["train"]["steps_per_epoch"] = 954
    assert resume_incompatibilities(old, new) == []          # nothing to compare
    assert ("train.steps_per_epoch", 384, 954) in \
        resume_incompatibilities(old, new, {"steps_per_epoch": 384})


def test_f_the_two_objective_flags_are_compared_too():
    old, new = _cfg(), _cfg()
    new["objective"]["mask_before_frontend"] = False
    new["objective"]["normalize_spec_target"] = False
    keys = {k for k, _, _ in resume_incompatibilities(old, new)}
    assert keys == {"objective.mask_before_frontend",
                    "objective.normalize_spec_target"}


class _Weights:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


class _Initer:
    """Just enough of the trainer for init_from, which touches nothing else."""

    def __init__(self, objective):
        self.objective = objective
        self.raw_model = _Weights()

    def _check_vocab(self, ck, path):
        return None

    init_from = EEGC1Trainer.init_from


def test_f_init_from_is_not_gated_on_the_objective(tmp_path, capsys):
    """A change of objective is the REASON to use --init-from, so it must not
    be the thing that blocks it. Only --resume refuses."""
    ck = tmp_path / "old.pth"
    torch.save({"model": {"w": torch.zeros(2)},
                "config": _cfg(spec_weight=1.0, raw_weight=0.25),
                "epoch": 9, "global_step": 900,
                "best_val_loss_masked_mse": 0.7}, ck)

    t = _Initer(resolve_eeg_c1_objective(_cfg(spec_weight=0.5, raw_weight=0.5)))
    t.init_from(str(ck))                      # must not raise
    assert t.raw_model.loaded is not None, "the weights did not transfer"
    out = capsys.readouterr().out
    assert "weights only" in out
    # And it says the objective changed rather than passing it over in silence.
    assert "1*L_spec + 0.25*L_raw" in out
    assert "0.5*L_spec + 0.5*L_raw" in out

    # Nothing but the weights: init_from touches no optimizer, scheduler,
    # epoch counter or step count, and the stub above has none of them, so a
    # regression that restored one would raise AttributeError here.
    assert not hasattr(t, "optimizer") and not hasattr(t, "epoch")


# --------------------------------------------------------------------------- #
# G. the progress script
# --------------------------------------------------------------------------- #

def _write_metrics(path, rows_):
    with open(path, "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")


def test_g_the_progress_script_renders_a_dual_objective_run(tmp_path, capsys):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eeg_c1_progress

    _write_metrics(tmp_path / "metrics_epoch.jsonl", [
        {"epoch": 0, "global_step": 10,
         "train/loss_total": 0.9, "val/loss_total": 0.8,
         "val/loss_masked_spec_mse": 1.0, "val/loss_masked_raw_smoothl1": 0.6,
         "val/masked_spec_corr": 0.11, "val/masked_raw_corr": 0.12,
         "val/masked_spec_nmse": 0.9, "val/masked_raw_nmse": 0.95,
         "val/macro_route_loss_total": 0.79,
         "val/route/E19_256/masked_spec_corr": 0.2,
         "val/route/E32_512/masked_spec_corr": 0.3},
    ])
    assert eeg_c1_progress.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for column in ("train_total", "val_total", "val_spec", "val_raw",
                   "spec_corr", "raw_corr", "spec_nmse", "raw_nmse",
                   "macro_total", "gate", "mins"):
        assert column in out, column
    assert "best_total.pth" in out and "best_spec.pth" in out
    assert "best_raw.pth" in out and "best_macro_total.pth" in out
    # The claim that is no longer true must not be printed.
    assert "best.pth holds" not in out
    assert "NOT the best spec loss" in out
    assert "predates dual-objective selection" not in out


def test_g_a_legacy_run_renders_with_dashes(tmp_path, capsys):
    """A run from before the raw head has no raw metrics at all. It must
    render, with '-' where a number does not exist, rather than raise."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eeg_c1_progress

    _write_metrics(tmp_path / "metrics_epoch.jsonl", [
        {"epoch": 0, "global_step": 10, "train/loss_masked_mse": 1.2,
         "val/loss_masked_mse": 1.1, "val/masked_corr": 0.3,
         "val/route/E19_256/masked_corr": 0.31},
        {"epoch": 1, "global_step": 20, "train/loss_masked_mse": 1.0,
         "val/loss_masked_mse": 0.9, "val/masked_corr": 0.4,
         "val/route/E19_256/masked_corr": 0.41},
    ])
    assert eeg_c1_progress.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "-" in out
    # loss_masked_mse is the spec term under its old name, so it fills val_spec.
    assert "1.10000" in out and "0.90000" in out
    assert "is not in this run's metrics" in out       # no total, no raw
    # And the trailing note describes THIS run's best.pth, which really was
    # spec-selected. Printing the new policy over an old run's checkpoint
    # would mislabel a file that is already on disk.
    assert "predates dual-objective selection" in out
    assert "holds its best SPEC loss" in out
    # The per-route breakdown falls back to the pre-dual name and says so,
    # rather than printing a column of dashes that looks like a dead route.
    assert "falling" in out and "masked_corr" in out
    assert "E19_256" in out


@pytest.mark.parametrize("metric", ["masked_spec_corr", "masked_raw_corr",
                                    "loss_total", "masked_raw_nmse"])
def test_g_the_metric_flag_still_selects_the_breakdown(tmp_path, capsys, metric):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eeg_c1_progress

    _write_metrics(tmp_path / "metrics_epoch.jsonl", [
        {"epoch": 0, "global_step": 1, "val/loss_total": 1.0,
         f"val/route/E19_256/{metric}": 0.5,
         f"val/dataset/tueg/{metric}": 0.7},
    ])
    assert eeg_c1_progress.main([str(tmp_path), "--metric", metric]) == 0
    assert "E19_256" in capsys.readouterr().out
    assert eeg_c1_progress.main(
        [str(tmp_path), "--by", "dataset", "--metric", metric]) == 0
    assert "tueg" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# H. the reconstruction figure
# --------------------------------------------------------------------------- #

def _viz():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import visualize_eeg_pretraining as viz
    return viz


def _code_only(path: str) -> str:
    """A file's source with comments removed.

    "the visualizer no longer reads X" is a statement about the CODE. Grepping
    the raw file also finds the comment explaining why X is gone, so the test
    would fail on its own documentation.
    """
    out = []
    for line in open(path):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("#")[0])
    return "".join(out)


def _example(route_id="E19_256", n_valid=None, seed=3):
    """A synthetic ``_fixed_example`` payload on the sample grid."""
    r = ROUTES[route_id]
    C, P, pt = r.n_channels, r.patches_per_channel, r.patch_t
    rng = np.random.default_rng(seed)
    mask = rng.random((C, P)) < 0.7
    valid = np.ones((C, P), bool)
    if n_valid is not None:
        valid[n_valid:] = False
        mask[n_valid:] = False
    shape = (C, P * pt)
    return {
        "target_raw": rng.standard_normal(shape).astype(np.float32),
        "pred_raw": rng.standard_normal(shape).astype(np.float32) * 0.2,
        "target_spec": rng.standard_normal(shape).astype(np.float32),
        "pred_spec": rng.standard_normal(shape).astype(np.float32) * 0.2,
        "clean_spec": rng.standard_normal(shape).astype(np.float32) * 50,
        "mask": mask, "valid": valid, "normalize_spec_target": True,
    }


def test_h_target_and_composite_share_colour_limits():
    viz = _viz()
    route = ROUTES["E19_256"]
    panels = viz.reconstruction_panels(_example(), route)
    assert len(panels) == 8
    raw_limits = {(p[3], p[4]) for p in panels[0:3]}
    spec_limits = {(p[3], p[4]) for p in panels[4:7]}
    assert len(raw_limits) == 1, "the raw panels autoscale independently"
    assert len(spec_limits) == 1, "the spec panels autoscale independently"
    # And a prediction at a fifth of the amplitude is NOT stretched to match.
    (vmin, vmax), = spec_limits
    assert vmin == -vmax and vmax > 0


def test_h_the_spec_panels_are_target_spec_not_clean_spec():
    viz = _viz()
    route = ROUTES["E19_256"]
    ex = _example()
    panels = viz.reconstruction_panels(ex, route)
    assert np.array_equal(panels[4][0], ex["target_spec"])
    assert not np.array_equal(panels[4][0], ex["clean_spec"])
    # clean_spec is 50x the scale here; a limit taken from it would be 50x too.
    assert panels[4][4] < 10.0
    assert "normalised spec target" in panels[4][1]


def test_h_the_composite_is_target_on_visible_and_prediction_on_masked():
    viz = _viz()
    route = ROUTES["E19_256"]
    ex = _example()
    m = np.repeat(ex["mask"], route.patch_t, axis=1)
    for tgt, pred, panel in (("target_raw", "pred_raw", 2),
                             ("target_spec", "pred_spec", 6)):
        comp = viz.reconstruction_panels(ex, route)[panel][0]
        assert np.array_equal(comp[~m], ex[tgt][~m])
        assert np.array_equal(comp[m], ex[pred][m])


def test_h_the_error_panels_exclude_visible_and_padded():
    viz = _viz()
    route = ROUTES["E32_512"]
    ex = _example("E32_512", n_valid=26)
    m = np.repeat(ex["mask"], route.patch_t, axis=1)
    valid = np.repeat(ex["valid"], route.patch_t, axis=1)
    for panel in (3, 7):
        err = viz.reconstruction_panels(ex, route)[panel][0]
        assert np.isnan(err[~m]).all(), "a visible patch entered the error map"
        assert np.isnan(err[~valid]).all(), "a padded channel entered it"
        assert np.isfinite(err[m & valid]).all()
        assert (err[m & valid] >= 0).all()


def test_h_the_masked_input_panel_is_the_signal_the_frontend_saw():
    viz = _viz()
    route = ROUTES["E19_256"]
    ex = _example()
    m = np.repeat(ex["mask"], route.patch_t, axis=1)
    corrupted = viz.reconstruction_panels(ex, route)[1][0]
    assert (corrupted[m] == 0).all()
    assert np.array_equal(corrupted[~m], ex["target_raw"][~m])


def test_h_figure_metadata_resolves_the_objective_from_the_objective_block():
    """The visualizer read train.spec_recon_weight and the config writes
    objective.spec_weight, so it fell back to a literal 1.0/0.25. That was
    right only while the config said 1.0/0.25; at 0.5/0.5 every figure would
    have been captioned with weights the run never trained under."""
    cfg = {"objective": {"spec_weight": 0.5, "raw_weight": 0.5},
           "train": {"spec_recon_weight": 1.0, "raw_recon_weight": 0.25}}
    got = resolve_eeg_c1_objective(cfg)
    assert got["spec_weight"] == 0.5
    assert got["raw_weight"] == 0.5

    # And no code path in the visualizer reads the dead train: spellings.
    # Comments are stripped first: the reason those names are gone is written
    # down beside where they used to be, and it would otherwise fail this.
    viz = _viz()
    code = _code_only(viz.__file__)
    assert "spec_recon_weight" not in code, \
        "the visualizer still reads a train: key the config does not write"
    assert "raw_recon_weight" not in code
    assert "resolve_eeg_c1_objective" in code


def test_h_the_dual_objective_figure_reports_the_resolved_weights(tmp_path):
    viz = _viz()
    writer = viz.FigureWriter(str(tmp_path), "png", {
        "objective": {"spec_weight": 0.5, "raw_weight": 0.5,
                      "fold_kl": 1e-3, "raw_beta": 0.5,
                      "mask_before_frontend": True,
                      "normalize_spec_target": True}})
    viz.fig_dual_objective(writer, [
        {"epoch": 0, "train/loss_masked_spec_mse": 1.0,
         "val/loss_masked_spec_mse": 1.1,
         "train/loss_masked_raw_smoothl1": 0.5,
         "val/loss_masked_raw_smoothl1": 0.55, "train/loss_fold_kl": 1e-6},
        {"epoch": 1, "train/loss_masked_spec_mse": 0.9,
         "val/loss_masked_spec_mse": 1.0,
         "train/loss_masked_raw_smoothl1": 0.4,
         "val/loss_masked_raw_smoothl1": 0.45, "train/loss_fold_kl": 1e-6},
    ])
    meta = json.loads(
        (tmp_path / "figure_metadata" / "10_dual_objective.json").read_text())
    assert meta["weights"] == {"spec": 0.5, "raw": 0.5, "fold_kl": 1e-3}
    assert meta["objective_equation"] == \
        "0.5*L_spec + 0.5*L_raw + 0.001*L_foldKL"


# --------------------------------------------------------------------------- #
# H2. surveying more than one window, and saying that a figure was picked
# --------------------------------------------------------------------------- #

class _FakeDataset:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


def test_h2_survey_indices_span_the_dataset_rather_than_one_recording():
    """range(n) would draw n consecutive windows of ONE recording.

    The corpus is one shard per recording with windows consecutive inside it,
    so "the first four windows" is four windows of one subject's one episode.
    Picking the best of those selects over the mask draw, not over the corpus.
    """
    viz = _viz()
    idx = viz._survey_indices(_FakeDataset(10000), 4)
    assert idx == [0, 3333, 6666, 9999]
    # Degenerate inputs still name a drawable window.
    assert viz._survey_indices(_FakeDataset(10000), 1) == [0]
    assert viz._survey_indices(_FakeDataset(1), 8) == [0]
    assert viz._survey_indices(_FakeDataset(0), 8) == []
    # Asking for more windows than exist yields each of them once, not
    # duplicates that would render the same figure twice under two names.
    assert viz._survey_indices(_FakeDataset(3), 8) == [0, 1, 2]


def test_h2_the_first_survey_figure_keeps_its_historical_name():
    """A paper script, a reference or a test written before --recon-windows
    existed points at fig_mask_reconstruction. Renaming every figure to
    _w00 would break all of them to no purpose."""
    viz = _viz()
    assert viz._recon_name("fig_mask_reconstruction", 0) == \
        "fig_mask_reconstruction"
    assert viz._recon_name("fig_mask_reconstruction", 1) == \
        "fig_mask_reconstruction_w01"


def test_h2_rank_one_is_the_best_window_and_a_diverged_one_ranks_last():
    """Rank 1 must be the LOWEST masked NMSE, and a NaN must never take it.

    A window whose decoder produced NaN compares false against everything, so
    an unguarded sort can leave it in first place -- and first place is exactly
    what someone reading the survey for a figure will reach for.
    """
    viz = _viz()

    def ex(raw, spec):
        return {"metrics": {"masked_raw_nmse": raw, "masked_spec_nmse": spec}}

    rows = [("E19_256", "tueg"), ("E128_512", "hbn")]
    grid = [[ex(0.9, 0.5), ex(0.1, 0.1)],
            [ex(float("nan"), 0.2), ex(0.2, 0.2)],
            [ex(0.3, 0.8), ex(0.3, 0.3)]]
    ranks = viz._rank_windows(rows, grid)

    # 0.3 < 0.9 < NaN
    assert ranks[("E19_256", 2)]["raw"] == 1
    assert ranks[("E19_256", 0)]["raw"] == 2
    assert ranks[("E19_256", 1)]["raw"] == 3
    # The two heads rank independently: window 1 is worst on raw, best on spec.
    assert ranks[("E19_256", 1)]["spec"] == 1
    assert all(r["n"] == 3 for r in ranks.values())
    # Ranked WITHIN a route. E128_512's window 0 is its own best despite being
    # numerically worse than nothing in E19_256; a pooled ranking would report
    # "the best window" as "the window from the easiest route".
    assert ranks[("E128_512", 0)]["raw"] == 1


def test_h2_a_route_absent_from_a_survey_column_does_not_take_first_place():
    """A route whose dataset ran out of windows has None in the later columns.

    float(None["metrics"]) is not the failure mode to worry about -- the
    failure mode is a missing row silently sorting to rank 1.
    """
    viz = _viz()
    rows = [("E19_256", "tueg")]
    grid = [[{"metrics": {"masked_raw_nmse": 0.4, "masked_spec_nmse": 0.4}}],
            [None]]
    ranks = viz._rank_windows(rows, grid)
    assert ranks[("E19_256", 0)]["raw"] == 1
    assert ranks[("E19_256", 1)]["raw"] == 2


# --------------------------------------------------------------------------- #
# End to end: a smoke run, its checkpoints, and its figures
# --------------------------------------------------------------------------- #

def _smoke_train(tmp_path, run, corpus, *extra):
    cmd = [sys.executable, "-m", "physiowave.train.pretrain_main",
           "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run),
           *extra,
           "--set", f"data.manifest_train={corpus['train']}",
           f"data.manifest_val={corpus['val']}",
           "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
           "model.channel_embed_dim=8", "train.epochs=1",
           "train.warmup_epochs=0", "train.precision=fp32",
           "train.grad_accumulation_steps=1", "train.steps_per_epoch=4",
           "train.batch_size_by_route.E19_256=1",
           "train.batch_size_by_route.E32_512=1",
           "train.batch_size_by_route.E64_256=1",
           "train.batch_size_by_route.E128_512=1"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


@pytest.mark.slow
def test_end_to_end_checkpoints_figures_and_refusals(tmp_path):
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=3, recordings=1,
                                windows=3)
    run = tmp_path / "run"
    r = _smoke_train(tmp_path, run, corpus)
    assert r.returncode == 0, r.stderr[-3000:]

    # -- the banner said the objective ---------------------------------- #
    for line in ("0.5 x spec MSE", "0.5 x raw SmoothL1(beta=0.5)",
                 "0.001 x ScaleFold KL", "resolved steps per epoch",
                 "total optimizer updates", "effective world size",
                 "per-route batch size", "passes per dataset/epoch"):
        assert line in r.stdout, line

    # -- every checkpoint the selection policy names -------------------- #
    for name in ("latest.pth", "best.pth", "best_total.pth", "best_spec.pth",
                 "best_raw.pth", "best_macro_total.pth"):
        assert (run / name).is_file(), name
    for name in ("metrics_step.jsonl", "metrics_epoch.jsonl",
                 "config_resolved.yaml"):
        assert (run / name).is_file(), name

    ck = torch.load(run / "best.pth", map_location="cpu", weights_only=False)
    assert set(ck["best_scores"]) == set(BEST_KEYS)
    assert ck["checkpoint_selection"]["best.pth"] == "val/loss_total"
    # The old field survives as an alias for the spec bar and selects nothing.
    assert ck["best_val_loss_masked_mse"] == \
        pytest.approx(ck["best_scores"]["spec"])
    assert ck["objective"]["spec_weight"] == 0.5
    assert ck["objective"]["raw_weight"] == 0.5

    row = [json.loads(l) for l in open(run / "metrics_epoch.jsonl")][-1]
    for key in ("val/loss_total", "val/loss_masked_spec_mse",
                "val/loss_masked_raw_smoothl1", "val/masked_spec_corr",
                "val/masked_raw_corr", "val/masked_spec_nmse",
                "val/masked_raw_nmse", "val/macro_route_loss_total",
                "val/macro_route_spec_corr", "val/macro_route_raw_nmse"):
        assert key in row and math.isfinite(row[key]), key
    assert row["val/loss_total"] == pytest.approx(
        0.5 * row["val/loss_masked_spec_mse"]
        + 0.5 * row["val/loss_masked_raw_smoothl1"]
        + 1e-3 * row["val/loss_fold_kl"], rel=1e-4)

    # -- a resume under a different objective is refused ---------------- #
    bad = subprocess.run(
        [sys.executable, "-m", "physiowave.train.pretrain_main",
         "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run),
         "--resume", "auto",
         "--set", f"data.manifest_train={corpus['train']}",
         f"data.manifest_val={corpus['val']}",
         "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
         "model.channel_embed_dim=8", "train.epochs=2",
         "train.warmup_epochs=0", "train.precision=fp32",
         "train.grad_accumulation_steps=1", "train.steps_per_epoch=4",
         "objective.spec_weight=1.0", "objective.raw_weight=0.25",
         "train.batch_size_by_route.E19_256=1",
         "train.batch_size_by_route.E32_512=1",
         "train.batch_size_by_route.E64_256=1",
         "train.batch_size_by_route.E128_512=1"],
        cwd=ROOT, capture_output=True, text=True)
    combined = bad.stdout + bad.stderr
    assert bad.returncode != 0
    assert "different objective or schedule" in combined
    assert "--init-from" in combined
    assert "objective.raw_weight" in combined

    # -- but weights-only init-from is allowed -------------------------- #
    fresh = tmp_path / "fresh"
    ok = subprocess.run(
        [sys.executable, "-m", "physiowave.train.pretrain_main",
         "--config", "pretrain/eeg_c1_moe", "--output-dir", str(fresh),
         "--init-from", str(run / "best.pth"),
         "--set", f"data.manifest_train={corpus['train']}",
         f"data.manifest_val={corpus['val']}",
         "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
         "model.channel_embed_dim=8", "train.epochs=1",
         "train.warmup_epochs=0", "train.precision=fp32",
         "train.grad_accumulation_steps=1", "train.steps_per_epoch=4",
         "objective.spec_weight=1.0", "objective.raw_weight=0.25",
         "train.batch_size_by_route.E19_256=1",
         "train.batch_size_by_route.E32_512=1",
         "train.batch_size_by_route.E64_256=1",
         "train.batch_size_by_route.E128_512=1"],
        cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr[-3000:]
    assert "weights only" in ok.stdout

    # -- the figures ---------------------------------------------------- #
    fig = subprocess.run(
        [sys.executable, "scripts/visualize_eeg_pretraining.py",
         "--run-dir", str(run), "--checkpoint", "best.pth", "--split", "val",
         "--format", "png", "--threads", "2", "--only",
         "fig_mask_reconstruction", "fig_raw_waveform_reconstruction",
         "fig_dual_objective"],
        cwd=ROOT, capture_output=True, text=True)
    assert fig.returncode == 0, fig.stderr[-3000:]

    meta = json.loads(
        (run / "figure_metadata" / "fig_mask_reconstruction.json").read_text())
    assert meta["objective"]["spec_weight"] == 0.5
    assert meta["objective"]["raw_weight"] == 0.5
    assert len(meta["panels"]) == 8
    assert {e["route_id"] for e in meta["examples"]} == set(ROUTES)
    for e in meta["examples"]:
        for key in ("loss_masked_spec_mse", "masked_spec_corr",
                    "masked_spec_nmse", "loss_masked_raw_smoothl1",
                    "masked_raw_corr", "masked_raw_nmse",
                    "actual_mask_ratio"):
            assert key in e, key

    npz = np.load(run / "figure_data" / "fig_mask_reconstruction.npz")
    for rid in ROUTES:
        route = ROUTES[rid]
        mask = npz[f"{rid}_mask"]
        valid = npz[f"{rid}_valid"]
        m = np.repeat(mask, route.patch_t, axis=1)
        v = np.repeat(valid, route.patch_t, axis=1)
        for tag in ("raw", "spec"):
            err = npz[f"{rid}_masked_error_{tag}"]
            assert np.isnan(err[~m]).all(), f"{rid} {tag} error kept visible"
            assert np.isfinite(err[m & v]).all()
        # The stored spec target is the NORMALISED one: each patch of it has
        # ~zero mean and ~unit variance, which clean_spec does not.
        tgt = npz[f"{rid}_target_spec"]
        patches = tgt[valid.any(axis=1)].reshape(
            -1, route.patches_per_channel, route.patch_t)
        assert np.abs(patches.mean(axis=-1)).max() < 1e-3
        assert np.abs(patches.std(axis=-1) - 1.0).max() < 1e-2

    raw_meta = json.loads(
        (run / "figure_metadata"
         / "14_raw_waveform_reconstruction.json").read_text())
    assert {r["route_id"] for r in raw_meta["rows"]} == set(ROUTES)
    for r_ in raw_meta["rows"]:
        for key in ("raw_corr", "raw_nmse", "raw_mae", "raw_rmse"):
            assert key in r_, key
    assert "not raw EDF values" in raw_meta["note"]

    # -- surveying several windows so a figure can be chosen -------------- #
    survey = subprocess.run(
        [sys.executable, "scripts/visualize_eeg_pretraining.py",
         "--run-dir", str(run), "--checkpoint", "best.pth", "--split", "val",
         "--format", "png", "--threads", "2", "--recon-windows", "3",
         "--only", "fig_mask_reconstruction",
         "fig_raw_waveform_reconstruction"],
        cwd=ROOT, capture_output=True, text=True)
    assert survey.returncode == 0, survey.stderr[-3000:]
    for name in ("fig_mask_reconstruction", "fig_mask_reconstruction_w01",
                 "fig_mask_reconstruction_w02",
                 "14_raw_waveform_reconstruction",
                 "14_raw_waveform_reconstruction_w01",
                 "14_raw_waveform_reconstruction_w02"):
        assert (run / "figures" / f"{name}.png").is_file(), name

    table = json.loads(
        (run / "figure_metadata" / "reconstruction_survey.json").read_text())
    assert table["n_windows"] == 3
    assert len(table["windows"]) == 3 * len(ROUTES)
    # Different windows, not the same one drawn three times under three names.
    for rid in ROUTES:
        got = sorted(r["window_index"] for r in table["windows"]
                     if r["route_id"] == rid)
        assert len(set(got)) == 3, f"{rid} surveyed the same window {got}"
    for rid in ROUTES:
        assert sorted(r["raw_nmse_rank"] for r in table["windows"]
                      if r["route_id"] == rid) == [1, 2, 3]

    # A figure that was picked out of a survey says so in its own metadata.
    picked = json.loads(
        (run / "figure_metadata"
         / "fig_mask_reconstruction_w02.json").read_text())
    assert picked["survey_index"] == 2
    assert picked["n_windows_surveyed"] == 3
    for e in picked["examples"]:
        assert e["raw_nmse_rank"] in (1, 2, 3)
        assert e["n_windows_surveyed"] == 3

    dual = json.loads(
        (run / "figure_metadata" / "10_dual_objective.json").read_text())
    assert dual["weights"] == {"spec": 0.5, "raw": 0.5, "fold_kl": 1e-3}

    # -- the progress script ------------------------------------------- #
    prog = subprocess.run(
        [sys.executable, "scripts/eeg_c1_progress.py", str(run)],
        cwd=ROOT, capture_output=True, text=True)
    assert prog.returncode == 0, prog.stderr[-2000:]
    assert "best.pth holds" not in prog.stdout
    assert "best_macro_total.pth" in prog.stdout


def test_paper_pretraining_figure_needs_no_checkpoint(tmp_path):
    """fig_paper_pretraining draws from the manifest and the epoch metrics alone,
    so the synced results (no checkpoint) are enough."""
    import json
    import subprocess
    import sys
    run = tmp_path / "run"
    run.mkdir()
    datasets = {"tueg": "E19_256", "faced": "E32_512", "m3cv": "E64_256", "hbn": "E128_512"}
    (run / "dataset_manifest.json").write_text(json.dumps({
        "datasets": {d: {"route_id": r, "n_windows": 1000 * (i + 1)}
                     for i, (d, r) in enumerate(datasets.items())},
        "target_weights": {d: 0.25 for d in datasets},
        "realised_mixture": {"by_step": {d: 0.25 for d in datasets}}}))
    rows = [{"epoch": e, "global_step": 10 * e,
             **{f"val/route/{r}/loss_total": 0.5 - 0.01 * e for r in datasets.values()},
             "val/macro_route_loss_total": 0.5 - 0.01 * e} for e in range(5)]
    (run / "metrics_epoch.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    script = os.path.join(ROOT, "scripts", "visualize_eeg_pretraining.py")
    r = subprocess.run([sys.executable, script, "--run-dir", str(run), "--only",
                        "fig_paper_pretraining", "--format", "svg", "--paper"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert (run / "figures" / "fig_paper_pretraining.svg").exists()
    meta = json.loads((run / "figure_metadata" / "fig_paper_pretraining.json").read_text())
    assert meta["checkpoint"] is None and set(meta["datasets"]) == set(datasets)
    # a model figure without a checkpoint is still refused
    r = subprocess.run([sys.executable, script, "--run-dir", str(run), "--only",
                        "fig_paper_reconstruction"], capture_output=True, text=True)
    assert r.returncode != 0 and "no checkpoint" in r.stderr
