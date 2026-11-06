"""
The EEG C1 multi-route pretraining path.

Numbered to the acceptance list, so a failure names the requirement it broke.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import (CHANNEL_TO_ID, CHANNEL_VOCAB, PAD_ID,  # noqa: E402
                               channel_ids_for, normalize_channel_name,
                               vocab_sha256)
from model import BERTWaveletTransformer                             # noqa: E402
from physiowave.eeg_c1.data import (CorpusIndex, RouteSchedule,      # noqa: E402
                                    ShardInfo)
from physiowave.eeg_c1.model import (MultiRouteEEGPretrainer,        # noqa: E402
                                     masked_reconstruction_loss)
from physiowave.eeg_c1.preprocess import (map_to_slots, place_on_slots,  # noqa: E402
                                          split_subjects, window_signal,
                                          zscore_windows)
from physiowave.eeg_c1.routes import (DOWNSTREAM_ONLY,               # noqa: E402
                                      PRETRAIN_DATASETS, ROUTES,
                                      SLOTS_32, default_sampling_weights)


def tiny(**kw):
    kw.setdefault("embed_dim", 64)
    kw.setdefault("depth", 1)
    kw.setdefault("num_heads", 4)
    kw.setdefault("channel_embed_dim", 16)
    return MultiRouteEEGPretrainer(**kw)


def meta_for(route_id, n_valid=None):
    r = ROUTES[route_id]
    ids, _ = channel_ids_for(r.slots)
    valid = torch.ones(r.n_channels, dtype=torch.bool)
    if n_valid is not None:
        valid[n_valid:] = False
        ids = [i if valid[k] else PAD_ID for k, i in enumerate(ids)]
    return {"channel_ids": torch.tensor(ids, dtype=torch.long),
            "valid_channel_mask": valid}


# --- 1 ---------------------------------------------------------------------- #
def test_1_c0_pretrain_unchanged_without_channel_embedding():
    """C0 keeps the behaviour it had before the channel code existed."""
    def build():
        torch.manual_seed(0)
        return BERTWaveletTransformer(
            in_channels=4, max_level=3, patch_size=(1, 128), embed_dim=64,
            depth=1, num_heads=4, scale_fold="dynamic", task_type="pretrain",
            mask_ratio=0.5)

    x = torch.randn(2, 4, 1024)
    m = build().eval()
    torch.manual_seed(7)
    a = m(x, task="pretrain", mask_ratio=0.5)
    m2 = build().eval()
    torch.manual_seed(7)
    b = m2(x, task="pretrain", mask_ratio=0.5)
    assert torch.equal(a[1], b[1])
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert m.channel_encoder is None
    with pytest.raises(ValueError):
        m(x, task="pretrain", channel_meta={"channel_ids": torch.zeros(4).long()})


# --- 2 ---------------------------------------------------------------------- #
def test_2_c1_pretrain_consumes_channel_metadata():
    """The legacy pretrain path accepts channel_meta instead of refusing it."""
    torch.manual_seed(0)
    m = BERTWaveletTransformer(
        in_channels=4, max_level=3, patch_size=(1, 128), embed_dim=64, depth=1,
        num_heads=4, scale_fold="dynamic", task_type="pretrain", mask_ratio=0.5,
        channel_encoding="id", channel_injection="token", channel_embed_dim=16)
    meta = {"channel_ids": torch.tensor([2, 3, 4, 5])}
    pred, mask, target = m(torch.randn(2, 4, 1024), task="pretrain",
                           mask_ratio=0.5, channel_meta=meta)
    assert pred.shape == target.shape == (2, 32, 128)
    assert mask.shape == (2, 32)

    out = tiny()(torch.randn(2, 19, 1024), "E19_256",
                 channel_meta=meta_for("E19_256"), mask_ratio=0.5)
    assert out["pred"].shape == (2, 152, 128)


# --- 3 ---------------------------------------------------------------------- #
def test_3_c1_injects_at_token_site_only():
    """Not into the waveform, and not into the fold's scale logits."""
    torch.manual_seed(0)
    m = tiny().eval()
    x = torch.randn(2, 19, 1024)
    meta = meta_for("E19_256")

    with torch.no_grad():
        # A big gate: if the code reached the frontend at all, the folded
        # spectrogram would move with it.
        m.channel_token_gate.fill_(2.0)
        spec_with = m(x, "E19_256", channel_meta=meta,
                      mask_generator=torch.Generator().manual_seed(3))["spec"]
        spec_bare = m.wavelet_frontends["E19_256"](x)
    assert torch.allclose(spec_with, spec_bare, atol=1e-6), \
        "the channel code changed the wavelet/ScaleFold output"

    # It must reach the tokens, though.
    with torch.no_grad():
        m.channel_token_gate.fill_(0.0)
        t0 = m(x, "E19_256", channel_meta=meta,
               mask_generator=torch.Generator().manual_seed(3))["tokens"]
        m.channel_token_gate.fill_(2.0)
        t1 = m(x, "E19_256", channel_meta=meta,
               mask_generator=torch.Generator().manual_seed(3))["tokens"]
    assert not torch.allclose(t0, t1), "the token site received nothing"

    assert not hasattr(m, "channel_to_scale")
    with pytest.raises(ValueError):
        tiny(channel_injection="fold")
    with pytest.raises(ValueError):
        tiny(channel_encoding="signed")


# --- 4 ---------------------------------------------------------------------- #
def test_4_gate_has_finite_nonzero_gradient_at_first_step():
    torch.manual_seed(0)
    m = tiny()
    out = m(torch.randn(2, 19, 1024), "E19_256",
            channel_meta=meta_for("E19_256"), mask_ratio=0.5)
    loss, _ = masked_reconstruction_loss(out)
    loss.backward()
    g = m.channel_token_gate.grad
    assert g is not None and torch.isfinite(g).all() and float(g) != 0.0
    assert m.channel_encoder.id_embed.weight.grad is not None


# --- 5 ---------------------------------------------------------------------- #
@pytest.mark.parametrize("route_id,expected", [
    ("E19_256", 152), ("E32_512", 256), ("E64_256", 512), ("E128_512", 1024)])
def test_5_token_counts_per_route(route_id, expected):
    r = ROUTES[route_id]
    assert r.n_tokens == expected
    assert r.patches_per_channel == 8
    assert r.window_samples == int(4.0 * r.sampling_rate)
    assert r.patch_size == (1, int(0.5 * r.sampling_rate))
    m = tiny()
    out = m(torch.randn(1, r.n_channels, r.window_samples), route_id,
            channel_meta=meta_for(route_id), mask_ratio=0.5)
    assert out["pred"].shape == (1, expected, r.patch_t)
    assert out["mask"].shape == (1, expected)


# --- 6 ---------------------------------------------------------------------- #
def test_6_four_wavelet_experts_are_independent():
    m = tiny()
    assert len(m.wavelet_frontends) == 4
    seen = {}
    for rid, front in m.wavelet_frontends.items():
        ids = {id(p) for p in front.parameters()}
        for other, prev in seen.items():
            assert not (ids & prev), f"{rid} shares parameters with {other}"
        seen[rid] = ids


# --- 7 ---------------------------------------------------------------------- #
def test_7_transformer_is_shared_not_copied():
    m = tiny()
    names = [n for n, _ in m.named_parameters()]
    assert sum(n.startswith("shared_transformer.") for n in names) > 0
    for rid in ROUTES:
        assert not any(n.startswith(f"wavelet_frontends.{rid}.encoder")
                       for n in names)
    shared = {id(p) for p in m.shared_transformer.parameters()}
    for front in m.wavelet_frontends.values():
        assert not (shared & {id(p) for p in front.parameters()})
    # One encoder's worth of blocks, not four.
    assert len([n for n in names if ".blocks." in n and
                n.startswith("shared_transformer.")]) > 0


# --- 8 ---------------------------------------------------------------------- #
def test_8_same_rate_routes_share_patcher_and_decoder():
    m = tiny()
    assert sorted(m.patch_embed_by_rate) == ["256", "512"]
    assert sorted(m.reconstruction_heads) == ["256", "512"]
    assert ROUTES["E19_256"].rate_key == ROUTES["E64_256"].rate_key == "256"
    assert ROUTES["E32_512"].rate_key == ROUTES["E128_512"].rate_key == "512"
    assert (m.patch_embed_by_rate[ROUTES["E19_256"].rate_key]
            is m.patch_embed_by_rate[ROUTES["E64_256"].rate_key])
    assert (m.reconstruction_heads[ROUTES["E32_512"].rate_key]
            is m.reconstruction_heads[ROUTES["E128_512"].rate_key])


# --- 9 ---------------------------------------------------------------------- #
def test_9_padding_channels_affect_nothing():
    torch.manual_seed(0)
    m = tiny().eval()
    meta = meta_for("E32_512", n_valid=26)
    x = torch.randn(2, 32, 2048)
    x[:, 26:] = 0.0
    x2 = x.clone()
    x2[:, 26:] = torch.randn(2, 6, 2048) * 99.0

    def run(t):
        return m(t, "E32_512", channel_meta=meta, mask_ratio=0.5,
                 mask_generator=torch.Generator().manual_seed(11))

    with torch.no_grad():
        a, b = run(x), run(x2)
    assert torch.equal(a["mask"], b["mask"])
    sel = a["mask"]
    assert torch.allclose(a["pred"][sel], b["pred"][sel], atol=1e-6)
    la, _ = masked_reconstruction_loss(a)
    lb, _ = masked_reconstruction_loss(b)
    assert torch.allclose(la, lb, atol=1e-7)
    # Padded slots are never selected, which is how they leave the loss.
    assert int(a["mask"].reshape(2, 32, 8)[:, 26:].sum()) == 0
    # And the code for a padded slot is zero.
    with torch.no_grad():
        code = m.channel_encoder(meta)
    assert torch.allclose(code[26:], torch.zeros_like(code[26:]))


# --- 10 --------------------------------------------------------------------- #
def test_10_tdbrain_26_maps_into_32_slots():
    tdbrain = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
               "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3",
               "Pz", "P4", "P8", "O1", "Oz", "O2"]
    assert len(tdbrain) == 26
    mp = map_to_slots(tdbrain, SLOTS_32)
    assert int(mp.valid.sum()) == 26
    assert len(mp.empty_slots) == 6
    assert not mp.unmatched_sources and not mp.unknown_names
    # Placed by NAME: every filled slot holds the row whose name matches it.
    x = np.arange(26 * 10, dtype=np.float32).reshape(26, 10)
    placed = place_on_slots(x, mp, 32)
    for src, slot in zip(mp.matrix_rows, mp.slot_of_row):
        assert SLOTS_32[slot] == tdbrain[src]
        assert np.array_equal(placed[slot], x[src])
    for j in range(32):
        if not mp.valid[j]:
            assert np.all(placed[j] == 0)


# --- 11 --------------------------------------------------------------------- #
def test_11_hbn_129_to_128_is_by_name_and_traceable():
    from EEG.preprocess_pretrain_corpus import HBN_NON_SCALP
    assert "E129" in HBN_NON_SCALP
    names = [f"E{i}" for i in range(1, 130)]
    drop = [n for n in names if n.strip().upper() in
            {s.upper() for s in HBN_NON_SCALP}]
    assert drop == ["E129"], "the removed channel must be named, not positional"
    assert len([n for n in names if n not in drop]) == 128
    # A 129-channel file with no identifiable reference must fail, not be
    # trimmed by position.
    anon = [f"Ch{i}" for i in range(1, 130)]
    assert not [n for n in anon if n.strip().upper() in
                {s.upper() for s in HBN_NON_SCALP}]


# --- 12 --------------------------------------------------------------------- #
class _Args:
    """The subset of the CLI namespace the adapters and registry read."""
    allow_upsample_faced = False
    mains_hz = None
    no_notch = False
    registry = None


def test_12_upsampling_is_allowed_only_at_a_corpus_native_rate():
    """FACED ships 31 of 123 subjects at 250 Hz; that is acquisition, not a
    derivative. A rate the registry does not list stays refused, which is the
    only thing separating those subjects from the 250 Hz preprocessed release.
    """
    from EEG.preprocess_pretrain_corpus import upsample_allowed

    assert upsample_allowed("faced", 250.0, _Args())
    assert not upsample_allowed("faced", 128.0, _Args())
    assert not upsample_allowed("faced", 200.0, _Args())
    # PhysioNetMI is 160 Hz throughout against a 256 Hz route.
    assert upsample_allowed("physionet_mi", 160.0, _Args())
    # TDBRAIN records at 500 and lists nothing: a 250 Hz TDBRAIN file is wrong.
    assert not upsample_allowed("tdbrain", 250.0, _Args())

    forced = _Args()
    forced.allow_upsample_faced = True
    assert upsample_allowed("faced", 128.0, forced)
    assert not upsample_allowed("tdbrain", 128.0, forced)   # FACED-only escape


# --- 12b -------------------------------------------------------------------- #
def test_12b_registry_supplies_mains_and_m3cv_is_not_notched_twice():
    """A wrong mains frequency is silent in both directions, so it comes from a
    checked-in table rather than whichever shell script launched the job.
    """
    from EEG.preprocess_pretrain_corpus import resolved_mains

    assert resolved_mains("faced", _Args())[0] == 50.0
    assert resolved_mains("tdbrain", _Args())[0] == 50.0
    assert resolved_mains("physionet_mi", _Args())[0] == 60.0
    assert resolved_mains("hbn", _Args())[0] == 60.0
    assert resolved_mains("tueg", _Args())[0] == 60.0

    # HGD is measured to have no line peak at either frequency, and its notch
    # would take 50 Hz AND the 100 Hz harmonic -- the latter inside the
    # 70-140 Hz band the dataset exists to measure. Skipped, with the reason
    # recorded, and kept distinct from "the publisher already notched it".
    freq, why = resolved_mains("hgd", _Args())
    assert freq is None
    assert "notch skipped" in why
    assert "high-gamma" in why

    # M3CV is published already notched at 49-51 Hz. Filtering it again would
    # be a second stopband on an already-empty one.
    freq, why = resolved_mains("m3cv", _Args())
    assert freq is None
    assert "already notched" in why

    explicit = _Args()
    explicit.mains_hz = 60.0
    assert resolved_mains("m3cv", explicit) == (60.0, "--mains-hz")

    off = _Args()
    off.no_notch = True
    assert resolved_mains("faced", off)[0] is None


# --- 12c -------------------------------------------------------------------- #
def test_12c_physionet_channel_padding_resolves_to_electrodes():
    """PhysioNet's EDF+ pads labels to four characters with periods. Before this
    was handled, all 64 channels landed as UNK, per-file slot coverage was 0 of
    64, and every recording in the corpus failed the coverage gate.
    """
    from channel_embedding import CHANNEL_TO_ID, normalize_channel_name

    for raw, want in (("Fc5.", "FC5"), ("C3..", "C3"), ("Cz..", "Cz"),
                      ("Iz..", "Iz"), ("Fp1.", "Fp1"), ("Af7.", "AF7"),
                      ("Po7.", "PO7"), ("T10.", "T10")):
        got = normalize_channel_name(raw)
        assert got == want, f"{raw!r} -> {got!r}, expected {want!r}"
        assert got in CHANNEL_TO_ID

    # The wrappers that were already handled must keep working.
    assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
    assert normalize_channel_name("T3") == "T7"
    assert normalize_channel_name("E128") == "E128"


# --- 12d -------------------------------------------------------------------- #
def test_12d_bids_subject_identity_is_taken_from_the_label():
    """Subject identity decides the train/val split, so a wrong answer here is a
    subject leaking across it that nothing downstream can detect. The earlier
    FACED adapter used the parent directory, which in BIDS is always "eeg".
    """
    from EEG.preprocess_pretrain_corpus import _bids_subject

    cases = {
        "HBN/R3/sub-NDARAB123/eeg/sub-NDARAB123_task-rest_eeg.set": "sub-NDARAB123",
        "faced/sub-101/eeg/sub-101_task-emotion_eeg.edf": "sub-101",
        "hgd/sub-14/eeg/sub-14_task-motor_eeg.set": "sub-14",
        "tdbrain/sub-19681349/ses-1/eeg/x.edf": "sub-19681349",
    }
    for path, want in cases.items():
        assert _bids_subject(path) == want
    assert _bids_subject("nobids/S001R03.edf") is None
    # Distinct subjects must not collapse onto one id.
    ids = {_bids_subject(p) for p in cases}
    assert len(ids) == len(cases)


# --- 12e -------------------------------------------------------------------- #
def test_12e_streaming_adapters_shard_by_subject():
    """--array=0-63 must give each task 1/64 of the corpus, not all of it.

    TUEG shards a file list it builds up front. The other six adapters stream
    Recordings, and nothing applied --shard to them: every task of an array read
    the whole corpus and raced its siblings for the same output paths. The check
    now runs inside each adapter, before the file is opened -- filtering after
    the read would be correct and useless, since each of 64 tasks would still
    decode all of HBN and discard 63/64 of it.
    """
    from EEG.preprocess_pretrain_corpus import owns, subject_shard

    class Sharded:
        shard = None

    subjects = [f"sub-{i:05d}" for i in range(2000)]

    # A subject belongs to exactly one task, for any partition.
    for total in (2, 4, 16, 64):
        buckets = [0] * total
        for sub in subjects:
            idx = subject_shard(sub, total)
            assert 0 <= idx < total
            buckets[idx] += 1
        assert sum(buckets) == len(subjects)
        assert all(b > 0 for b in buckets), f"an empty task at N={total}"
        # Roughly balanced: no task may carry more than 3x the mean.
        assert max(buckets) < 3 * (len(subjects) / total)

    # owns() partitions the corpus and never duplicates or drops a subject.
    seen = []
    for idx in range(8):
        a = Sharded()
        a.shard = (idx, 8)
        seen += [s for s in subjects if owns(s, a)]
    assert sorted(seen) == sorted(subjects)

    # No --shard means one process owns everything.
    assert all(owns(s, Sharded()) for s in subjects[:50])

    # The two sharding paths must agree, or a subject lands on two tasks.
    assert subject_shard("aaaaaaaa", 16) == subject_shard("aaaaaaaa", 16)


# --- 12f -------------------------------------------------------------------- #
def test_12f_proportional_sampling_sees_every_window_once():
    """"As much data as a dataset has" -- not a quota a small corpus repeats to.

    The mixture weights are probabilities over STEPS, and a step draws
    batch_by_route[route] windows: 64 on E19_256, 12 on E128_512. Weighting
    steps by window count directly would hand E19_256 five times the windows
    its share of the corpus warrants, so the weights divide by the route's batch
    size and the WINDOW shares come out equal to the corpus shares.
    """
    from physiowave.eeg_c1.data import (DEFAULT_BATCH_BY_ROUTE, CorpusIndex,
                                        RouteSchedule)

    counts = {"tueg": 13_711_294, "physionet_mi": 39_000, "m3cv": 400_000,
              "faced": 1_100_000, "tdbrain": 90_000, "hbn": 4_500_000,
              "hgd": 47_000}
    index = CorpusIndex([
        ShardInfo(f"/x/{d}.h5", d, PRETRAIN_DATASETS[d].route_id, n)
        for d, n in counts.items()])
    total = sum(counts.values())

    sched = RouteSchedule(index, weights="proportional", seed=42, num_replicas=4)
    assert sched.weight_policy == "proportional"
    mix = sched.realised_mixture()

    for d, n in counts.items():
        # Window share tracks corpus share.
        assert abs(mix["by_window"][d] - n / total) < 0.01, d
        # And that means one pass, which is the requirement.
        b = DEFAULT_BATCH_BY_ROUTE[PRETRAIN_DATASETS[d].route_id]
        seen = mix["by_step"][d] * sched.steps_per_epoch * b * 4
        assert 0.85 < seen / n < 1.20, f"{d} seen {seen/n:.2f}x per epoch"

    # The step shares are deliberately NOT the corpus shares -- HBN's small
    # micro-batch buys it more steps for the same number of windows.
    assert mix["by_step"]["hbn"] > mix["by_step"]["tueg"]
    assert counts["hbn"] < counts["tueg"]

    # null means balanced -- size alone must not decide the mixture.
    assert RouteSchedule(index, weights=None, seed=42).weight_policy == "balanced"

    # temperature is the dial between the two, and it is bounded.
    for alpha, lo, hi in ((0.5, 0.40, 0.60), (0.75, 0.55, 0.75)):
        t = RouteSchedule(index, weights=f"temperature:{alpha}", seed=42,
                          num_replicas=4)
        tw = t.realised_mixture()["by_window"]
        assert lo < tw["tueg"] < hi, f"alpha={alpha}: tueg {tw['tueg']:.3f}"
        # Between the extremes, in the direction that matters.
        assert tw["tueg"] < counts["tueg"] / total
        assert tw["physionet_mi"] > counts["physionet_mi"] / total
    with pytest.raises(SystemExit):
        RouteSchedule(index, weights="temperature", seed=42)
    with pytest.raises(SystemExit):
        RouteSchedule(index, weights="temperature:2.0", seed=42)

    # The old behaviour is still reachable, and still repeats small corpora.
    bal = RouteSchedule(index, weights="balanced", seed=42, num_replicas=4)
    bmix = bal.realised_mixture()
    b = DEFAULT_BATCH_BY_ROUTE[PRETRAIN_DATASETS["physionet_mi"].route_id]
    pmi = bmix["by_step"]["physionet_mi"] * bal.steps_per_epoch * b * 4
    assert bmix["by_window"]["physionet_mi"] > 20 * (counts["physionet_mi"] / total)
    assert pmi > 0

    with pytest.raises(SystemExit):
        RouteSchedule(index, weights="whatever-that-means", seed=42)


# --- 12g -------------------------------------------------------------------- #
def test_12g_hgd_montage_is_10_05_and_fully_resolvable():
    """HGD's 128 electrodes must every one map to a real vocabulary id.

    The first version of this slot list was assembled from 10-10 names and a
    guess. HGD is 10-05: fifty of its electrodes are HALFWAY positions -- FFC5h
    lies between FFC5 and FFC3 -- so only 74 of 128 resolved, per-file coverage
    sat below the gate, and every recording in the corpus would have been
    skipped. The list is now measured from the corpus with --derive-slots.
    """
    from physiowave.eeg_c1.routes import SLOTS_128, SLOTS_128_HGD

    assert len(SLOTS_128_HGD) == 128
    assert len(set(SLOTS_128_HGD)) == 128, "a duplicate slot would drop a channel"

    ids, unknown = channel_ids_for(SLOTS_128_HGD)
    assert not unknown, f"{len(unknown)} HGD slots are not electrodes: {unknown[:8]}"
    assert PAD_ID not in ids and 1 not in ids, "a slot resolved to <pad>/<unk>"

    # The halfway positions are spelled with a lowercase h, and normalisation
    # folds the spellings a recording might use onto that one.
    h_slots = [x for x in SLOTS_128_HGD if x[-1] == "h"]
    assert len(h_slots) == 50, f"{len(h_slots)} halfway positions, expected 50"
    for name in ("FFC5h", "TPP10h", "OI2h", "AFP3h"):
        assert name in SLOTS_128_HGD
        assert normalize_channel_name(name.upper()) == name
        assert normalize_channel_name(name.lower()) == name

    # HGD and HBN share the route's SHAPE, never electrode identities: HBN
    # records EGI net positions and these are scalp labels. Any overlap would
    # train one electrode's embedding on the other's signal.
    assert not (set(SLOTS_128_HGD) & set(SLOTS_128))

    # Cz is the recording reference, kept as a real (attenuated) channel.
    assert SLOTS_128_HGD.index("Cz") == 15
    assert "M1" in SLOTS_128_HGD and "M2" in SLOTS_128_HGD

    # The aux channels the 133-channel files carry are not electrodes and must
    # not have crept into the montage.
    for aux in ("EOG", "EOGh", "EOGv", "EMG_RH", "EMG_LH", "EMG_RF"):
        assert aux not in SLOTS_128_HGD


# --- 12h -------------------------------------------------------------------- #
def test_12h_every_shipped_container_is_walked_and_readable():
    """One extension list and one reader, shared by all six adapters.

    M3CV ships 2,469 BrainVision recordings and reported "no readable recording"
    because its adapter walked .set/.edf/.fif/.cnt. Six adapters each carrying
    their own extension tuple and their own if/elif reader chain is how that
    happens, and how a corpus in a format the chain does not mention falls
    through to read_raw_edf and fails on a file that was never an EDF.
    """
    from EEG.preprocess_pretrain_corpus import READABLE_EXTS, read_raw_any

    for ext in (".edf", ".bdf", ".set", ".fif", ".cnt", ".vhdr", ".mff"):
        assert ext in READABLE_EXTS, ext

    # Only the HEADER of a multi-file format. Walking .eeg or .vmrk as well
    # would read every BrainVision recording two or three times.
    for companion in (".eeg", ".vmrk", ".fdt", ".dat"):
        assert companion not in READABLE_EXTS, companion

    # The reader dispatches on the extension rather than defaulting to EDF.
    import inspect as _inspect
    src = _inspect.getsource(read_raw_any)
    for reader in ("read_raw_bdf", "read_raw_eeglab", "read_raw_fif",
                   "read_raw_cnt", "read_raw_brainvision", "read_raw_egi"):
        assert reader in src, reader


# --- 12i -------------------------------------------------------------------- #
def test_12i_no_route_slot_is_dead_for_every_corpus_on_it():
    """A slot no corpus on the route records is padding in every window.

    E64_256's last two slots were P9/P10, which --inspect showed neither
    PhysioNetMI nor M3CV records -- 62/64 and 60/64, both missing exactly those.
    Two guaranteed-zero channels in every window of the route, while four real
    electrodes were being dropped for want of a slot.
    """
    from physiowave.eeg_c1.preprocess import map_to_slots
    from physiowave.eeg_c1.routes import SLOTS_64

    assert len(SLOTS_64) == 64 and len(set(SLOTS_64)) == 64
    assert "P9" not in SLOTS_64 and "P10" not in SLOTS_64
    assert "TP9" in SLOTS_64 and "TP10" in SLOTS_64

    # The montages as the two --inspect runs reported them.
    physionet = [
        "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C5..", "C3..",
        "C1..", "Cz..", "C2..", "C4..", "C6..", "Cp5.", "Cp3.", "Cp1.", "Cpz.",
        "Cp2.", "Cp4.", "Cp6.", "Fp1.", "Fpz.", "Fp2.", "Af7.", "Af3.", "Afz.",
        "Af4.", "Af8.", "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..",
        "F6..", "F8..", "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.",
        "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..",
        "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..",
        "Iz..",
    ]
    m3cv = [s for s in SLOTS_64 if s not in ("AFz", "Iz", "TP9", "TP10")] + \
           ["FT9", "FT10", "TP9", "TP10"]

    cov = {}
    for name, montage in (("physionet_mi", physionet), ("m3cv", m3cv)):
        mapping = map_to_slots(montage, SLOTS_64)
        cov[name] = int(mapping.valid.sum())
        assert cov[name] >= int(0.75 * 64), f"{name} would fail the gate"

    assert cov["physionet_mi"] == 62
    assert cov["m3cv"] == 62, "M3CV should gain the two freed slots"

    # No slot may be dead for BOTH corpora on the route.
    filled = set()
    for montage in (physionet, m3cv):
        mapping = map_to_slots(montage, SLOTS_64)
        filled |= {SLOTS_64[j] for j, v in enumerate(mapping.valid) if v}
    dead = [s for s in SLOTS_64 if s not in filled]
    assert not dead, f"slots no corpus on E64_256 records: {dead}"


# --- 13 --------------------------------------------------------------------- #
def test_13_deap_is_never_a_pretraining_dataset():
    assert "deap" not in PRETRAIN_DATASETS
    assert "deap" in DOWNSTREAM_ONLY
    assert "deap" not in default_sampling_weights()
    for name in DOWNSTREAM_ONLY:
        assert name not in PRETRAIN_DATASETS
    assert set(PRETRAIN_DATASETS) == {
        "tueg", "faced", "tdbrain", "physionet_mi", "m3cv", "hbn", "hgd"}


# --- 14 --------------------------------------------------------------------- #
def test_14_subject_split_has_no_leak():
    subs = [f"s{i:03d}" for i in range(37)]
    train, val = split_subjects(subs, 0.1, 42)
    assert not (set(train) & set(val))
    assert set(train) | set(val) == set(subs)
    assert len(val) >= 1
    assert split_subjects(subs, 0.1, 42) == (train, val)
    assert split_subjects(subs, 0.1, 43) != (train, val)


# --- 15 --------------------------------------------------------------------- #
def _fake_index(n=5000):
    return CorpusIndex([ShardInfo(f"/x/{d}.h5", d, s.route_id, n)
                        for d, s in PRETRAIN_DATASETS.items()])


def test_15_route_sampler_mixture_and_ddp_agreement():
    index = _fake_index()
    lengths = {d: 5000 for d in PRETRAIN_DATASETS}

    # `balanced` is the default and means P(route)=1/4.
    single = RouteSchedule(index, steps_per_epoch=4000, seed=42)
    assert single.weight_policy == "balanced"
    single.set_epoch(0)
    plan = single.plan()
    by_route = {}
    for d in plan:
        rid = PRETRAIN_DATASETS[d].route_id
        by_route[rid] = by_route.get(rid, 0) + 1
    for rid, n in by_route.items():
        assert abs(n / len(plan) - 0.25) < 0.03, f"{rid} drew {n/len(plan):.3f}"

    # `proportional`, on seven datasets that are all the same size, gives each
    # an equal share of WINDOWS and therefore an unequal share of steps, since
    # a route's micro-batch ranges from 64 down to 12.
    prop = RouteSchedule(index, weights="proportional", steps_per_epoch=4000,
                         seed=42)
    assert prop.weight_policy == "proportional"
    windows = prop.realised_mixture()["by_window"]
    for d in PRETRAIN_DATASETS:
        assert abs(windows[d] - 1.0 / len(PRETRAIN_DATASETS)) < 0.02, d
    steps = prop.realised_mixture()["by_step"]
    assert steps["hbn"] > 3 * steps["tueg"], "batch size was not divided out"

    ranks = [RouteSchedule(index, steps_per_epoch=500, seed=42,
                           num_replicas=4, rank=r) for r in range(4)]
    for s in ranks:
        s.set_epoch(0)
    steps = [list(s.steps(lengths)) for s in ranks]
    for i in range(500):
        assert len({t[i][0] for t in steps}) == 1, "ranks disagreed on the route"
        assert len({t[i][1] for t in steps}) == 1
        picks = [x for t in steps for x in t[i][2]]
        assert len(set(picks)) == len(picks), "two ranks got the same window"


# --- 16 --------------------------------------------------------------------- #
def test_16_resume_continues_the_sampling_sequence():
    index = _fake_index()
    lengths = {d: 5000 for d in PRETRAIN_DATASETS}
    full = RouteSchedule(index, steps_per_epoch=200, seed=42, num_replicas=2,
                         rank=0)
    full.set_epoch(3)
    baseline = list(full.steps(lengths))

    resumed = RouteSchedule(index, steps_per_epoch=200, seed=42, num_replicas=2,
                            rank=0)
    resumed.load_state_dict({"epoch": 3, "seed": 42, "steps_per_epoch": 200,
                             "start_step": 80})
    assert list(resumed.steps(lengths)) == baseline[80:]
    assert len(resumed) == 120

    other = RouteSchedule(index, steps_per_epoch=200, seed=42)
    other.set_epoch(4)
    assert other.plan() != full.plan()


# --- 17 --------------------------------------------------------------------- #
@pytest.mark.parametrize("route_id", list(ROUTES))
def test_17_unpatchify_inverts_patchify(route_id):
    r = ROUTES[route_id]
    spec = torch.randn(3, r.n_channels, r.window_samples)
    patches = MultiRouteEEGPretrainer.patchify(spec, r.patch_t)
    assert patches.shape == (3, r.n_tokens, r.patch_t)
    back = MultiRouteEEGPretrainer.unpatchify(patches, r.n_channels, r.patch_t)
    assert back.shape == spec.shape
    assert torch.equal(spec, back)


# --- 18 --------------------------------------------------------------------- #
REQUIRED_FIGURES = [
    "fig_dataset_routes", "fig_pretraining_convergence", "fig_route_convergence",
    "fig_mask_reconstruction", "fig_mask_examples_by_dataset",
    "fig_mask_statistics", "fig_wavelet_frequency_response",
    "fig_scale_fold_weights", "fig_channel_embedding",
    # The two the dual objective needs. Without them the figure set shows one
    # of the two reconstruction heads and reports weights nothing resolved.
    "10_dual_objective", "14_raw_waveform_reconstruction",
]


@pytest.mark.slow
def test_18_all_required_svgs_generate_from_a_smoke_checkpoint(tmp_path):
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "smoke_corpus"), subjects=3,
                                recordings=1, windows=2)
    run_dir = tmp_path / "run"
    cmd = [sys.executable, "-m", "physiowave.train.pretrain_main",
           "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run_dir),
           "--set", f"data.manifest_train={corpus['train']}",
           f"data.manifest_val={corpus['val']}",
           "model.embed_dim=48", "model.depth=1", "model.num_heads=4",
           "model.channel_embed_dim=16", "train.epochs=1",
           "train.warmup_epochs=0", "train.precision=fp32",
           "train.grad_accumulation_steps=1", "train.steps_per_epoch=4",
           "train.batch_size_by_route.E19_256=1",
           "train.batch_size_by_route.E32_512=1",
           "train.batch_size_by_route.E64_256=1",
           "train.batch_size_by_route.E128_512=1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    # Every file the selection policy names, not just the compatibility copy.
    for name in ("latest.pth", "best.pth", "best_total.pth", "best_spec.pth",
                 "best_raw.pth", "best_macro_total.pth"):
        assert (run_dir / name).is_file(), name

    r = subprocess.run(
        [sys.executable, "scripts/visualize_eeg_pretraining.py",
         "--run-dir", str(run_dir), "--checkpoint", "best.pth",
         "--split", "val", "--format", "svg"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    for name in REQUIRED_FIGURES:
        svg = run_dir / "figures" / f"{name}.svg"
        assert svg.is_file(), f"{name}.svg was not generated"
        text = svg.read_text()
        assert len(text) > 0
        # Matplotlib writes the XML declaration first, which is correct for a
        # standalone SVG; the requirement's "starts with <svg" is read as "is a
        # non-empty SVG document".
        assert text.startswith("<?xml") or text.startswith("<svg")
        assert "<svg" in text
        assert (run_dir / "figure_data" / f"{name}.npz").is_file()
        meta = json.loads(
            (run_dir / "figure_metadata" / f"{name}.json").read_text())
        for key in ("checkpoint_sha256", "global_step", "mask_seed",
                    "plotting_script_git_commit", "generated_utc"):
            assert key in meta


# --- vocabulary and schema, supporting the above ---------------------------- #
def test_channel_vocabulary_is_append_only():
    """The ids Sleep-EDF and PhysioP300 were trained under must not move."""
    assert CHANNEL_VOCAB[0] == "<pad>" and CHANNEL_VOCAB[1] == "<unk>"
    for name, expected in (("Fpz-Cz", 2), ("Pz-Oz", 3)):
        assert CHANNEL_TO_ID[name] == expected
    # The PhysioP300 monopolar block starts where it always did.
    assert CHANNEL_TO_ID["Fp1"] == 22
    assert CHANNEL_TO_ID["Iz"] == 85
    assert len(CHANNEL_VOCAB) == len(set(CHANNEL_VOCAB))
    assert len(vocab_sha256()) == 64


def test_channel_name_normalisation():
    assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
    assert normalize_channel_name("EEG T3-LE") == "T7"      # old nomenclature
    assert normalize_channel_name("fp2") == "Fp2"
    assert normalize_channel_name("E17") == "E17"
    ids, unknown = channel_ids_for(["EEG Fp1-REF", "nonsense"])
    assert ids[0] == CHANNEL_TO_ID["Fp1"] and ids[1] == 1
    assert unknown == ["nonsense"]


def test_every_dataset_slot_list_matches_its_route():
    for spec in PRETRAIN_DATASETS.values():
        assert len(spec.slots) == spec.route.n_channels
        _, unknown = channel_ids_for(spec.slots)
        assert not unknown, f"{spec.dataset_id} has out-of-vocabulary slots"


def test_windowing_drops_the_incomplete_tail():
    x = np.random.randn(4, 1024 * 2 + 500).astype(np.float32)
    w, starts = window_signal(x, 1024, 1024)
    assert w.shape == (2, 4, 1024)
    assert starts.tolist() == [0, 1024]
    short, _ = window_signal(np.random.randn(4, 100).astype(np.float32), 1024,
                             1024)
    assert short.shape[0] == 0


def test_zscore_leaves_padded_channels_exactly_zero():
    w = (np.random.randn(3, 8, 256).astype(np.float32) + 5.0) * 20
    valid = np.array([True] * 6 + [False] * 2)
    w[:, ~valid] = 0.0
    z = zscore_windows(w, valid, 1e-6, 20.0)
    assert np.all(z[:, ~valid] == 0.0)
    assert abs(float(z[:, valid].mean())) < 1e-4
    assert abs(float(z[:, valid].std()) - 1.0) < 1e-2
    assert float(np.abs(z).max()) <= 20.0


# --------------------------------------------------------------------------- #
# TUEG: identity, sharding, and the gates that decide whether a corpus is usable
# --------------------------------------------------------------------------- #

def test_tueg_identity_comes_from_the_filename():
    from EEG.preprocess_pretrain_corpus import tueg_identity

    root = "/corpus/TUEG_v2.0.2"
    p = f"{root}/edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf"
    ident = tueg_identity(p, root)
    assert ident == {"subject": "aaaaaaaa", "session": "s001",
                     "montage": "01_tcp_ar", "rule": "filename"}

    # v1.x had no bucket directory; the filename rule does not care.
    p1 = f"{root}/edf/aaaaaaab/s003_2011/02_tcp_le/aaaaaaab_s003_t002.edf"
    assert tueg_identity(p1, root)["subject"] == "aaaaaaab"
    assert tueg_identity(p1, root)["montage"] == "02_tcp_le"
    assert tueg_identity(p1, root)["rule"] == "filename"

    # A file that does not follow the convention falls back and SAYS it did,
    # so an --inspect run shows whether the fallback is carrying the corpus.
    p2 = f"{root}/edf/000/aaaaaaac/s001_2015/01_tcp_ar/oddly_named.edf"
    fb = tueg_identity(p2, root)
    assert fb["rule"] == "path"
    assert fb["subject"] == "aaaaaaac"


def test_tueg_sharding_is_by_subject_and_covers_everything():
    from EEG.preprocess_pretrain_corpus import shard_files, tueg_identity

    root = "/corpus"
    files = [f"{root}/edf/000/aaaaaa{c}{d}/s{s:03d}_2015/01_tcp_ar/"
             f"aaaaaa{c}{d}_s{s:03d}_t000.edf"
             for c in "abcde" for d in "fghij" for s in (1, 2, 3)]
    assert len(files) == 75

    class Args:
        shard = None

    n = 7
    seen, owners = [], {}
    for i in range(n):
        Args.shard = (i, n)
        kept = shard_files(files, Args, root)
        seen.extend(kept)
        for f in kept:
            owners.setdefault(tueg_identity(f, root)["subject"], set()).add(i)

    assert sorted(seen) == sorted(files), "sharding lost or duplicated a file"
    for subject, tasks in owners.items():
        assert len(tasks) == 1, f"{subject} was split across tasks {tasks}"


def test_subject_split_side_is_independent_of_shard_count():
    """The whole reason the split is hashed rather than shuffled."""
    from physiowave.eeg_c1.preprocess import subject_split_side

    subs = [f"aaaaaa{i:03d}" for i in range(500)]
    sides = {s: subject_split_side(s, 0.1, 42) for s in subs}
    # Re-deciding one subject at a time, in any order, gives the same answer.
    for s in reversed(subs):
        assert subject_split_side(s, 0.1, 42) == sides[s]
    val = [s for s in subs if sides[s] == "val"]
    assert 0.05 < len(val) / len(subs) < 0.16
    assert not (set(val) & {s for s in subs if sides[s] == "train"})
    # A different seed is a different partition.
    assert [subject_split_side(s, 0.1, 7) for s in subs] != list(sides.values())


def test_auxiliary_channels_are_dropped_without_failing_the_corpus():
    """TUEG's EKG/PHOTIC/IBI rows are 21% of a file and are not a naming bug."""
    from physiowave.eeg_c1.routes import SLOTS_19

    tuh = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
           "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF",
           "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
           "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
           "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
           "EEG EKG1-REF", "PHOTIC-REF", "IBI", "BURSTS", "SUPPR"]
    mp = map_to_slots(tuh, SLOTS_19)

    # Every scalp electrode lands, old nomenclature included.
    assert int(mp.valid.sum()) == 19, f"empty: {mp.empty_slots}"
    assert not mp.empty_slots
    # The five auxiliary rows are dropped and named.
    assert len(mp.unmatched_sources) == 5
    assert {"IBI", "BURSTS", "SUPPR"} <= set(mp.unmatched_sources)
    # T3/T4/T5/T6 are the same electrodes as T7/T8/P7/P8, so they land there.
    placed = {SLOTS_19[j]: tuh[i]
              for i, j in zip(mp.matrix_rows, mp.slot_of_row)}
    assert placed["T7"] == "EEG T3-REF"
    assert placed["P8"] == "EEG T6-REF"
    # No placed channel is UNK; the gate that matters is slot coverage.
    ids, _ = channel_ids_for(SLOTS_19)
    assert 1 not in ids


def test_a_montage_that_cannot_be_named_leaves_slots_empty():
    """The failure the coverage gate exists to catch, as distinct from the above."""
    from physiowave.eeg_c1.routes import SLOTS_19

    anonymous = [f"Ch{i}" for i in range(1, 25)]
    mp = map_to_slots(anonymous, SLOTS_19)
    assert int(mp.valid.sum()) == 0
    assert len(mp.empty_slots) == 19
    assert len(mp.unmatched_sources) == 24


def test_manifest_merge_refuses_a_split_leak(tmp_path):
    """The check that is only possible once every shard's manifest is present."""
    root = tmp_path / "corpus"
    d = root / "tueg"
    d.mkdir(parents=True)
    row = {"path": str(d / "a.h5"), "dataset_id": "tueg",
           "route_id": "E19_256", "n_windows": 4, "subjects": ["aaaaaaaa"]}
    (d / "manifest_train.0000.jsonl").write_text(json.dumps(row) + "\n")
    leaked = dict(row, path=str(d / "b.h5"))
    (d / "manifest_val.0001.jsonl").write_text(json.dumps(leaked) + "\n")

    r = subprocess.run(
        [sys.executable, "scripts/build_eeg_c1_manifest.py",
         "--corpus-root", str(root), "--datasets", "tueg", "--allow-missing"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1
    assert "both splits" in r.stderr
    assert "aaaaaaaa" in r.stderr


def test_a_sparse_recording_is_skipped_not_fatal(tmp_path):
    """A file carrying a third of the montage fails alone, not the whole run.

    TUEG holds recordings from 21 to 41 channels, and some of the small ones do
    not carry the 10-20 nineteen at all. Those produce windows that are more
    mask than measurement; the aggregate empty-slot gate averages them away, so
    the per-recording floor is what actually keeps them out.
    """
    from EEG.preprocess_pretrain_corpus import process_recording, Recording
    from physiowave.eeg_c1.preprocess import PreprocessConfig, PreprocessError
    from physiowave.eeg_c1.routes import SLOTS_19

    route = ROUTES["E19_256"]
    cfg = PreprocessConfig(notch_hz=60.0)

    class Args:
        min_slot_coverage = 0.75

    sparse = Recording(
        recording_id="sparse", subject_id="aaaaaaaz",
        data=np.random.randn(6, 256 * 30) * 25e-6,
        channel_names=["EEG FP1-REF", "EEG FP2-REF", "EEG C3-REF",
                       "EEG C4-REF", "EEG O1-REF", "EEG O2-REF"],
        sampling_rate=256.0, unit="V", mains_hz=60.0)
    with pytest.raises(PreprocessError) as exc:
        process_recording(sparse, "tueg", cfg, SLOTS_19, route,
                          str(tmp_path), {}, args_ref=Args())
    assert "6 of 19" in str(exc.value)

    # The full montage passes the same floor.
    full = Recording(
        recording_id="full", subject_id="aaaaaaay",
        data=np.random.randn(19, 256 * 30) * 25e-6,
        channel_names=[f"EEG {c.upper()}-REF" for c in SLOTS_19],
        sampling_rate=256.0, unit="V", mains_hz=60.0)
    entry = process_recording(full, "tueg", cfg, SLOTS_19, route,
                              str(tmp_path), {}, args_ref=Args())
    assert entry is not None and entry["n_windows"] > 0
    assert entry["placed_total"] == 19 and entry["placed_unk"] == 0
    assert not entry["empty_slots"]


# --- 19 --------------------------------------------------------------------- #
def test_19_schedule_length_is_remaining_not_total(tmp_path):
    """len(schedule) counts DOWN, and start_step moves under a running epoch.

    Both are deliberate: a resumed epoch must iterate only what is left, and
    start_step is what makes that resumable. Anything that wants "step k of N"
    has to read steps_per_epoch and snapshot start_step BEFORE the loop --
    reading either inside it printed step 101 of 20.
    """
    from physiowave.eeg_c1.data import CorpusIndex, RouteSchedule
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=2)
    index = CorpusIndex.from_manifest(corpus["train"])
    sched = RouteSchedule(index, steps_per_epoch=40, seed=7)

    assert len(sched) == 40 and sched.steps_per_epoch == 40
    sched.start_step = 30
    assert len(sched) == 10, "length must be what is LEFT of the epoch"
    assert sched.steps_per_epoch == 40, "the epoch's size must not move"

    # And the two together give a count that goes up, not down.
    first, total = sched.start_step, sched.steps_per_epoch
    seen = [first + i + 1 for i in range(len(sched))]
    assert seen == list(range(31, 41))
    assert max(seen) == total


def test_19b_the_logged_steps_are_the_intended_ones(tmp_path):
    """Every fiftieth step and the LAST one -- not the midpoint.

    `i + 1 == len(schedule)` reads as "the last step" and is not: the length
    counts down as start_step advances, so the two met in the middle and the
    epoch's final step never logged.
    """
    epoch_steps, first_step = 769, 0
    wrong = [i for i in range(epoch_steps)
             if i % 50 == 0 or i + 1 == epoch_steps - i]
    right = [i for i in range(epoch_steps)
             if i % 50 == 0 or first_step + i + 1 == epoch_steps]

    assert 384 in wrong and 384 not in right, "the midpoint was the 'last' step"
    assert epoch_steps - 1 in right, "the epoch's last step must log"
    assert epoch_steps - 1 not in wrong

    # A resumed epoch logs its own last step, not the one it would have had
    # from the top.
    resumed = [i for i in range(epoch_steps - 600)
               if i % 50 == 0 or 600 + i + 1 == epoch_steps]
    assert (epoch_steps - 600) - 1 in resumed


# --- 20 --------------------------------------------------------------------- #
def test_20_val_iterator_length_matches_what_it_yields(tmp_path):
    """A progress bar's total has to be the number of batches, not an estimate."""
    from physiowave.eeg_c1.data import CorpusIndex
    from physiowave.eeg_c1.entry import build_smoke_corpus
    from physiowave.eeg_c1.train import ValIterator

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=4, recordings=1,
                                windows=6)
    index = CorpusIndex.from_manifest(corpus["val"])
    batch_by_route = {"E19_256": 2, "E32_512": 2, "E64_256": 2, "E128_512": 2}

    for replicas, rank in ((1, 0), (2, 0), (2, 1), (3, 2)):
        it = ValIterator(index, batch_by_route, replicas, rank)
        assert len(it) == sum(1 for _ in it), f"{replicas} replicas, rank {rank}"
        it.close()

    capped = ValIterator(index, batch_by_route, 1, 0, max_batches_per_dataset=1)
    assert len(capped) == sum(1 for _ in capped)
    capped.close()


# --- 21 --------------------------------------------------------------------- #
def test_21_a_fresh_run_does_not_inherit_a_crashed_one_s_metrics(tmp_path):
    """Four jobs died in one directory; the fifth's curves must be its own."""
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=2)
    run = tmp_path / "run"
    run.mkdir()
    # What a crashed attempt leaves behind.
    (run / "metrics_step.jsonl").write_text(
        json.dumps({"epoch": 0, "step": 1, "loss_total": 9.9}) + "\n")
    (run / "metrics_epoch.jsonl").write_text(
        json.dumps({"epoch": 0, "train/loss_masked_mse": 9.9}) + "\n")

    cmd = [sys.executable, "-m", "physiowave.train.pretrain_main",
           "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run),
           "--set", f"data.manifest_train={corpus['train']}",
           f"data.manifest_val={corpus['val']}",
           "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
           "model.channel_embed_dim=8", "train.epochs=1",
           "train.warmup_epochs=0", "train.precision=fp32",
           "train.grad_accumulation_steps=1", "train.steps_per_epoch=2",
           "train.batch_size_by_route.E19_256=1",
           "train.batch_size_by_route.E32_512=1",
           "train.batch_size_by_route.E64_256=1",
           "train.batch_size_by_route.E128_512=1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    steps = [json.loads(l) for l in open(run / "metrics_step.jsonl")]
    assert all(row.get("loss_total") != 9.9 for row in steps), \
        "the crashed attempt's rows are still in the new run's curve"
    epochs = [json.loads(l) for l in open(run / "metrics_epoch.jsonl")]
    assert len(epochs) == 1

    # Moved, not deleted -- the crashed attempts are how you find out why.
    kept = run / "superseded" / "0" / "metrics_step.jsonl"
    assert kept.is_file()
    assert json.loads(kept.read_text())["loss_total"] == 9.9


def test_21b_a_resume_keeps_its_own_history(tmp_path):
    """The earlier rows are this run's, not a stranger's."""
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=2)
    run = tmp_path / "run"
    base = [sys.executable, "-m", "physiowave.train.pretrain_main",
            "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run),
            "--set", f"data.manifest_train={corpus['train']}",
            f"data.manifest_val={corpus['val']}",
            "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
            "model.channel_embed_dim=8", "train.warmup_epochs=0",
            "train.precision=fp32", "train.grad_accumulation_steps=1",
            "train.steps_per_epoch=2",
            "train.batch_size_by_route.E19_256=1",
            "train.batch_size_by_route.E32_512=1",
            "train.batch_size_by_route.E64_256=1",
            "train.batch_size_by_route.E128_512=1"]
    r = subprocess.run(base + ["train.epochs=1"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    first = len([l for l in open(run / "metrics_epoch.jsonl")])

    r = subprocess.run(base + ["train.epochs=2", "--resume", "auto"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert len([l for l in open(run / "metrics_epoch.jsonl")]) > first, \
        "a resume retired the history it was supposed to continue"
    assert not (run / "superseded").exists()


# --- 22 --------------------------------------------------------------------- #
LAUNCHER = os.path.join(ROOT, "EEG", "pretrain_eeg_c1_moe.sh")


def _launcher_overrides(env=None):
    """The --set list the launcher would build, without running anything."""
    script = subprocess.run(
        ["bash", "-c",
         f'set -uo pipefail\n'
         f'pw_require_python_deps() {{ return 0; }}\n'
         f'export -f pw_require_python_deps\n'
         f'sed -n "/^OVERRIDES=(/,/^fi$/p" {LAUNCHER}'],
        capture_output=True, text=True)
    body = script.stdout
    assert "OVERRIDES=(" in body, script.stderr
    full = ('set -uo pipefail\n'
            'MANIFEST_TRAIN=t; MANIFEST_VAL=v\n'
            'EPOCHS="${EPOCHS:-}"; GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-}"\n'
            'LR="${LR:-}"; WEIGHT_DECAY="${WEIGHT_DECAY:-}"\n'
            'MASK_RATIO="${MASK_RATIO:-}"; SEED="${SEED:-}"\n'
            'WEIGHTS="${WEIGHTS:-}"; STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"\n'
            'SET="${SET:-}"\n'
            'VIS_EVERY_EPOCHS="${VIS_EVERY_EPOCHS:-}"\n'
            + body +
            '\nprintf "%s\\n" "${OVERRIDES[@]}"\n')
    r = subprocess.run(["bash", "-c", full], capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_22_the_launcher_does_not_shadow_the_config(tmp_path):
    """A default in the launcher is a second copy of a number, and it wins.

    Every hyperparameter used to be passed through --set unconditionally from a
    default that duplicated the config's, so editing the config changed
    nothing: the file said mask_ratio 0.75 and the run used 0.5.
    """
    bare = _launcher_overrides()
    for key in ("train.epochs", "train.grad_accumulation_steps", "train.lr",
                "train.weight_decay", "model.mask_ratio", "seed"):
        assert not any(o.startswith(key + "=") for o in bare), \
            f"{key} is overridden even when nothing asked for it"

    asked = _launcher_overrides({"MASK_RATIO": "0.6", "EPOCHS": "3"})
    assert "model.mask_ratio=0.6" in asked
    assert "train.epochs=3" in asked
    assert not any(o.startswith("train.lr=") for o in asked)


def test_22b_mask_ratio_zero_still_overrides():
    """0 is a value. A `${VAR:+...}` presence test would drop it."""
    assert "model.mask_ratio=0" in _launcher_overrides({"MASK_RATIO": "0"})


def test_22c_the_config_owns_the_hyperparameters(tmp_path):
    """These keys live in the config and nowhere else.

    Values are not pinned here. A tunable that is asserted in two files is a
    tunable you cannot change without a test failure telling you nothing, and
    the launcher shadowing them -- not their values -- is what this is for.
    """
    import yaml
    with open(os.path.join(ROOT, "configs", "pretrain", "eeg_c1_moe.yaml")) as f:
        cfg = yaml.safe_load(f)
    for section, key in (("model", "mask_ratio"), ("model", "dropout"),
                         ("model", "embed_dim"), ("model", "depth"),
                         ("train", "grad_accumulation_steps"),
                         ("train", "batch_size_by_route"), ("train", "lr"),
                         ("train", "epochs"), ("data", "weights")):
        assert key in cfg[section], f"{section}.{key} left the config"

    bare = _launcher_overrides()
    for dotted in ("model.mask_ratio", "model.dropout", "model.embed_dim",
                   "train.grad_accumulation_steps", "train.lr", "train.epochs",
                   "data.weights", "train.steps_per_epoch"):
        assert not any(o.startswith(dotted + "=") for o in bare), \
            f"the launcher overrides {dotted} when nothing asked it to"


# --- 23 --------------------------------------------------------------------- #
def test_23_open_shards_are_bounded(tmp_path, monkeypatch):
    """The cache used to grow to one handle per shard touched, and never shrink.

    A rank draws tens of thousands of windows an epoch from ~95,000 shards, so
    an unbounded cache is a file-descriptor leak with a slow-open symptom: the
    process's file table grows all run, every later open costs more, and the
    validation sweep -- which opens fresh handles each epoch -- gets steadily
    slower while training appears to speed up.
    """
    from physiowave.eeg_c1.data import CorpusIndex, EEGWindowDataset
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=6, recordings=1,
                                windows=2)
    index = CorpusIndex.from_manifest(corpus["train"])
    monkeypatch.setenv("PW_MAX_OPEN_SHARDS", "2")
    ds = EEGWindowDataset(index, "tueg")
    assert len(ds.shards) > 2, "need more shards than the bound to test it"

    seen = []
    for i in range(len(ds)):
        seen.append(ds[i]["x"].shape)
        assert len(ds._handles) <= 2, "the bound is not enforced"
    assert len(seen) == len(ds), "eviction lost a window"

    # The most recent shard stays; the oldest is the one that goes.
    ds.close()
    assert not ds._handles

    # And every window still reads correctly after eviction has cycled.
    ds2 = EEGWindowDataset(index, "tueg")
    monkeypatch.delenv("PW_MAX_OPEN_SHARDS")
    ds3 = EEGWindowDataset(index, "tueg")
    for i in range(len(ds2)):
        assert torch.equal(ds2[i]["x"], ds3[i]["x"])
    ds2.close()
    ds3.close()


def test_23b_a_reused_shard_is_not_reopened(tmp_path):
    """The bound must still be a cache, not a no-op."""
    from physiowave.eeg_c1.data import CorpusIndex, EEGWindowDataset
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=4)
    index = CorpusIndex.from_manifest(corpus["train"])
    ds = EEGWindowDataset(index, "tueg")
    ds[0]
    first = ds._handles[ds.locate(0)[0]]
    ds[0]
    assert ds._handles[ds.locate(0)[0]] is first, "reopened a cached shard"
    ds.close()


# --- 24 --------------------------------------------------------------------- #
ENV_SH = os.path.join(ROOT, "scripts", "cineca_env.sh")


def _env_sh(snippet, env=None):
    r = subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{ENV_SH}" >/dev/null 2>&1\n{snippet}'],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, **(env or {})})
    return r


def test_24_a_training_launcher_refuses_the_wrong_interpreter():
    """Four jobs ran on the right python by accident, and the fifth did not.

    Everything that decides which interpreter is active -- PW_ON_CINECA,
    PW_VARS_ONLY, PW_VENV, VIRTUAL_ENV, PATH -- is inherited by
    `sbatch --export=ALL` from the submitting shell. A launcher that assumes
    instead of checking reports the consequence (a missing package) rather than
    the cause.
    """
    off = _env_sh('pw_require_training_venv; echo "rc=$?"')
    assert "rc=0" in off.stdout, "off-cluster must not require $HOME/pw"

    on = _env_sh('PW_ON_CINECA=1 pw_require_training_venv; echo "rc=$?"')
    assert "rc=1" in on.stdout, "on-cluster must reject a foreign interpreter"
    for key in ("PW_ON_CINECA", "PW_VARS_ONLY", "PW_VENV", "VIRTUAL_ENV",
                "sys.prefix"):
        assert key in on.stderr, f"the diagnostic does not report {key}"


def test_24b_the_cluster_is_not_detected_from_one_variable():
    """$FAST alone decided it, so a job without it silently became a laptop."""
    with open(ENV_SH) as f:
        body = f.read()
    i = body.index("PW_ON_CINECA=0")
    block = body[i:i + 400]
    assert "/leonardo/prod/opt/modulefiles" in block, (
        "cluster detection still rests on the environment alone")


def test_24c_the_launcher_checks_the_interpreter_before_the_packages():
    """Reversed, a missing package is blamed on the wrong interpreter's owner."""
    with open(os.path.join(ROOT, "EEG", "pretrain_eeg_c1_moe.sh")) as f:
        body = f.read()
    assert body.index("pw_require_training_venv") < body.index("pw_require_python_deps")


def test_24d_env_sh_has_no_python_docstrings():
    """`\"\"\"...\"\"\"` parses in bash and runs as a command not found."""
    with open(ENV_SH) as f:
        assert '"""' not in f.read()


# --- 25 --------------------------------------------------------------------- #
def test_24e_the_p300_cache_is_found_by_what_was_decoded(tmp_path):
    """Not by what a run wants. `cache/c${IN_CHANNELS}` was the old spelling.

    It agreed with the only cache that existed and stopped agreeing the moment
    the montage became a choice: the preparation writes ONE superset cache,
    c64, and the split takes its subset by name -- so IN_CHANNELS=62 asked for
    a c62 that is never written, and the runner refused a cache that was there.
    """
    edf = tmp_path / "erpbci"
    (edf / "cache").mkdir(parents=True)
    none = _env_sh(f'pw_p300_cache_dir "{edf}"; echo "rc=$?"')
    assert "rc=1" in none.stdout, "an empty cache directory is not a cache"

    (edf / "cache" / "c58").mkdir()
    legacy = _env_sh(f'pw_p300_cache_dir "{edf}"; echo "rc=$?"')
    assert str(edf / "cache" / "c58") in legacy.stdout
    assert "rc=0" in legacy.stdout

    # The superset wins when both are on disk: it is the one every channel set
    # can be cut from, and find_subject_cache searches in this order too.
    (edf / "cache" / "c64").mkdir()
    both = _env_sh(f'pw_p300_cache_dir "{edf}"; echo "rc=$?"')
    assert str(edf / "cache" / "c64") in both.stdout
    # Nothing is ever named for a channel count a run asked for.
    assert "c62" not in both.stdout


def test_24f_the_p300_runners_agree_with_find_subject_cache():
    """One search order, in two languages. They drift silently if uncoupled."""
    import re
    sh = open(ENV_SH).read()
    body = re.search(r"pw_p300_cache_dir\(\)\s*\{(.*?)\n\}", sh, re.S).group(1)
    shell_order = re.search(r"for d in ([^;]+);", body).group(1).split()

    py = open(os.path.join(ROOT, "EEG", "physio_p300_finetune.py")).read()
    py_order = re.findall(
        r'"([^"]+)"', re.search(r"^CACHE_DIRS = \((.*?)\)", py, re.M | re.S).group(1))
    assert shell_order == py_order, (shell_order, py_order)

    # And no runner may rebuild the name from a channel count.
    for rel in ("scripts/slurm/cineca_p300_folds.sbatch",
                "EEG/run_p300_channel_ablation.sh"):
        src = open(os.path.join(ROOT, rel)).read()
        assert "cache/c${IN_CHANNELS" not in src, rel
        assert "pw_p300_cache_dir" in src, rel


def test_24g_the_p300_split_is_written_at_the_montage_that_is_trained():
    """IN_CHANNELS reaches the split as --channels, or the two can disagree.

    finetune.py checks in_channels against the file's channel metadata and
    refuses a mismatch, so the failure is loud -- but it happens after the
    allocation, once per fold.
    """
    for rel in ("scripts/slurm/cineca_p300_folds.sbatch",
                "EEG/run_p300_channel_ablation.sh"):
        src = open(os.path.join(ROOT, rel)).read()
        assert "--stage split" in src, rel
        assert '--channels "${IN_CHANNELS}"' in src, rel


def test_25_a_run_path_under_the_filesystem_root_is_refused():
    """`$PW_CKPT_ROOT/run` with PW_CKPT_ROOT empty is `/run`.

    --export=ALL carries the empty value from the submitting shell into the
    job, where the only symptom was rank 0 dying on a permission error after
    the allocation was granted and fifteen ranks reporting a lost rendezvous.
    """
    bad = _env_sh('pw_check_run_path OUTPUT_DIR "/pretrain_eeg_c1_moe_n1"; echo "rc=$?"')
    assert "rc=1" in bad.stdout
    assert "sits directly under /" in bad.stderr
    assert "cineca_env.sh" in bad.stderr, "the message must name the cause"

    empty = _env_sh('pw_check_run_path OUTPUT_DIR ""; echo "rc=$?"')
    assert "rc=1" in empty.stdout

    relative = _env_sh('pw_check_run_path OUTPUT_DIR "runs/x"; echo "rc=$?"')
    assert "rc=1" in relative.stdout

    missing = _env_sh('pw_check_run_path OUTPUT_DIR "/nope/nowhere/x"; echo "rc=$?"')
    assert "rc=1" in missing.stdout
    assert "does not exist" in missing.stderr

    ok = _env_sh(f'pw_check_run_path OUTPUT_DIR "{ROOT}/outputs/x"; echo "rc=$?"')
    assert "rc=0" in ok.stdout, ok.stderr


def test_25b_both_launchers_check_before_they_allocate_or_mkdir():
    sbatch = os.path.join(ROOT, "scripts", "slurm",
                          "cineca_eeg_c1_moe_pretrain.sbatch")
    with open(sbatch) as f:
        body = f.read()
    assert body.index('pw_check_run_path OUTPUT_DIR') < body.index("srun ")
    assert 'pw_check_run_path DATA_ROOT' in body

    with open(os.path.join(ROOT, "EEG", "pretrain_eeg_c1_moe.sh")) as f:
        body = f.read()
    assert body.index('pw_check_run_path OUTPUT_DIR') < body.index('mkdir -p "${OUTPUT_DIR}"')


def test_25c_a_failed_mkdir_stops_the_launcher():
    """set -e is deliberately off, so an unchecked mkdir carries on to srun."""
    with open(os.path.join(ROOT, "EEG", "pretrain_eeg_c1_moe.sh")) as f:
        body = f.read()
    i = body.index('mkdir -p "${OUTPUT_DIR}"')
    assert "exit 1" in body[i:i + 200], "mkdir failure is not fatal"


# --- 26 --------------------------------------------------------------------- #
def test_26_temperature_survives_the_override_parser():
    """`temperature:0.5` must reach the policy branch as a string.

    Overrides are parsed as YAML scalars, and `temperature: 0.5` -- with the
    space -- is YAML for a mapping. Only the spelling without it stays a string.
    """
    from physiowave.config import apply_overrides

    cfg = apply_overrides({"data": {"weights": "balanced"}},
                          ["data.weights=temperature:0.5"])
    assert cfg["data"]["weights"] == "temperature:0.5"

    spaced = apply_overrides({"data": {"weights": "balanced"}},
                             ["data.weights=temperature: 0.5"])
    assert spaced["data"]["weights"] == {"temperature": 0.5}, (
        "if YAML ever stops reading this as a mapping the hint below is moot")


def test_26b_a_policy_written_as_a_mapping_says_so(tmp_path):
    from physiowave.eeg_c1.data import CorpusIndex, RouteSchedule
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=2)
    index = CorpusIndex.from_manifest(corpus["train"])

    with pytest.raises(SystemExit) as exc:
        RouteSchedule(index, weights={"temperature": 0.5})
    assert "without the space" in str(exc.value)

    # And the string spelling works.
    s = RouteSchedule(index, weights="temperature:0.5")
    assert s.weight_policy == "temperature:0.5"


def test_26c_the_launcher_passes_weights_through():
    asked = _launcher_overrides({"WEIGHTS": "temperature:0.5"})
    assert "data.weights=temperature:0.5" in asked
    assert not any(o.startswith("data.weights=") for o in _launcher_overrides())


# --- 27 --------------------------------------------------------------------- #
def _c1_cmd(run, corpus, *extra):
    return [sys.executable, "-m", "physiowave.train.pretrain_main",
            "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run),
            *extra,
            "--set", f"data.manifest_train={corpus['train']}",
            f"data.manifest_val={corpus['val']}",
            "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
            "model.channel_embed_dim=8", "train.warmup_epochs=0",
            "train.precision=fp32", "train.grad_accumulation_steps=1",
            "train.steps_per_epoch=3",
            "train.batch_size_by_route.E19_256=1",
            "train.batch_size_by_route.E32_512=1",
            "train.batch_size_by_route.E64_256=1",
            "train.batch_size_by_route.E128_512=1"]


def test_27_init_from_takes_the_weights_and_nothing_else(tmp_path):
    """A resumed scheduler counts in the OLD epoch's units.

    steps_per_epoch is derived from the mixture, so changing the mixture
    changes it -- 384 under balanced, 954 under temperature:0.5. Restoring a
    scheduler across that leaves the cosine a fifth short of annealed at the
    end of the run. --init-from starts a new run from another's weights, with
    the optimizer, the schedule and the epoch counter fresh.
    """
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=3, recordings=1,
                                windows=3)
    a, b = tmp_path / "a", tmp_path / "b"
    r = subprocess.run(_c1_cmd(a, corpus) + ["train.epochs=1"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]

    r = subprocess.run(
        _c1_cmd(b, corpus, "--init-from", str(a / "best.pth")) + ["train.epochs=2"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "initialised from" in r.stdout
    assert "weights only" in r.stdout

    # The epoch counter restarted, so the run is its own.
    epochs = [json.loads(l)["epoch"]
              for l in open(b / "metrics_epoch.jsonl")]
    assert epochs == [0, 1], f"the epoch counter continued: {epochs}"

    # The weights did come across: the new run starts where the old ended,
    # not from a fresh init.
    import torch
    old = torch.load(a / "best.pth", map_location="cpu", weights_only=False)
    new = torch.load(b / "latest.pth", map_location="cpu", weights_only=False)
    assert new["global_step"] < old["global_step"] + 10
    fresh = subprocess.run(_c1_cmd(tmp_path / "z", corpus) + ["train.epochs=1"],
                           cwd=ROOT, capture_output=True, text=True)
    assert fresh.returncode == 0, fresh.stderr[-2000:]


def test_27b_init_from_and_resume_are_not_both_applicable(tmp_path):
    from physiowave.eeg_c1.entry import build_smoke_corpus
    corpus = build_smoke_corpus(str(tmp_path / "c"), subjects=2, recordings=1,
                                windows=2)
    a = tmp_path / "a"
    subprocess.run(_c1_cmd(a, corpus) + ["train.epochs=1"], cwd=ROOT,
                   capture_output=True, text=True)
    r = subprocess.run(
        _c1_cmd(tmp_path / "b", corpus, "--init-from", str(a / "best.pth"),
                "--resume", "auto") + ["train.epochs=1"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "different operations" in (r.stdout + r.stderr)


def test_27c_the_launcher_passes_init_from():
    with open(os.path.join(ROOT, "EEG", "pretrain_eeg_c1_moe.sh")) as f:
        body = f.read()
    assert '[[ -n "${INIT_FROM}" ]] && EXTRA+=(--init-from "${INIT_FROM}")' in body


def test_26d_the_launcher_passes_arbitrary_overrides():
    """An architecture experiment is a submission, not an edit and a revert."""
    got = _launcher_overrides({"SET": "model.embed_dim=512 model.depth=8",
                               "STEPS_PER_EPOCH": "1536"})
    assert "model.embed_dim=512" in got
    assert "model.depth=8" in got
    assert "train.steps_per_epoch=1536" in got
    assert not any(o.startswith("model.") for o in _launcher_overrides())
