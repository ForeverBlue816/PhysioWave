# ECG C1 downstream evaluation

The pretrained ECG C1 encoder (`docs/ecg_c1_pretraining.md`) is evaluated on
the paper's ECG tasks, plus the PTB-XL multi-label task that ECG foundation
model papers report. Every task runs in three modes:

- **ft**: the pretrained encoder, fine-tuned. This is the main row.
- **probe**: the encoder frozen; only the classification head is trained.
- **scratch**: the same model with no pretrained weights. This is the control
  that every pretrained number is read against.

## Commands (Leonardo)

```bash
cd ~/PhysioWave-T && git pull && source $HOME/pw/bin/activate

# 1. data (login node, ~10 GB, resumable; CPSC 2018 from Kaggle, needs the Kaggle token)
bash scripts/download_ecg_downstream.sh all
bash scripts/download_ecg_downstream.sh status

# 2. ONE job, one node, 4 GPUs, 6 h: GPU i builds task i's split if needed,
#    then runs on it, one after the other: ft at lr 1e-4, 3e-5 and 1e-5,
#    probe at 1e-3, scratch at 1e-4 (ptbxl, cpsc2018, chapman, ptbxl_super).
#    Resubmitting skips finished runs. Per mode, the table reports the run
#    with the best VALIDATION score.
sbatch scripts/slurm/cineca_ecg_downstream.sbatch

# 3. the table -- the job writes it to $PW_CKPT_ROOT/ecg_downstream/summary.{txt,md,json};
#    to put it in the repository:
python scripts/collect_ecg_downstream.py --root $PW_CKPT_ROOT/ecg_downstream \
    --markdown docs/runs/ecg_c1_moe/downstream.md
```

`PRETRAINED` defaults to `$PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth`, the
checkpoint with the lowest validation loss. Use `TAG=` together with `LR=`,
`EPOCHS=` or `SEED=` to run a variant next to the default rows.

## Tasks

| task | data | classes | split | paper metric |
|---|---|---|---|---|
| `ptbxl` | PTB-XL 1.0.3, 21,799 records | 5 superclasses, single label | strat_fold 1–8 / 9 / 10 | accuracy |
| `ptbxl_super` | PTB-XL 1.0.3 | 5 superclasses, multi-label | strat_fold 1–8 / 9 / 10 | macro AUROC (literature) |
| `cpsc2018` | Challenge 2021 `cpsc_2018` + `cpsc_2018_extra` | 9 official classes, multi-label | record-level 70/20/10, seed 42 | F1-micro |
| `chapman` | PhysioNet ecg-arrhythmia 1.0.0 (45,152 records) | SB / AFIB / GSVT / SR, multi-label | record-level 70/20/10, seed 42 | F1-micro |

### Label definitions

These are the paper's, taken from `ECG/ptbxl_finetune.py`,
`ECG/cpsc_multilabel.py` and `ECG/shaoxing_multilabel.py`.

- **`ptbxl`**: diagnostic statements with likelihood ≥ 80. When a record has
  several, the label is the latest of NORM < MI < STTC < CD < HYP.
- **`ptbxl_super`**: every diagnostic statement at any likelihood, as in
  Strodthoff et al. (2021). This gives 21,388 labelled records.
- **`cpsc2018` and `chapman`**: classes are matched by SNOMED code.

In every task, records with none of the classes are left out. PTB-XL's folds
are patient-disjoint. CPSC and Chapman carry one record per patient.
`split.json` shows that no subject crosses splits.

### Preprocessing

Downstream records are preprocessed exactly as the pretraining corpus was:

1. convert to mV;
2. remove DC;
3. 50 Hz notch;
4. 0.5 Hz high-pass;
5. resample to 500 Hz;
6. place leads by name in the L12_500 slots;
7. cut 10 s windows;
8. normalise with one scale per window across the 12 leads;
9. clip.

A 12-lead, 500 Hz, 10 s window is exactly the pretraining route L12_500.
Because of that, the pretrained wavelet frontend and patch embedding load
together with the transformer and channel encoder: the whole encoder
transfers. On EEG usually only the transformer does.

Two departures from pretraining, both deliberate:

- **No quality filter.** A benchmark's test set must not lose the records a
  filter dislikes.
- **Short CPSC records are padded, not dropped.** CPSC records are 6–60 s.
  They are cut into 10 s windows with a 5 s stride, and a record shorter than
  10 s is padded to 10 s.

## Training and scoring

`physiowave/train/finetune_main.py` with `configs/finetune/ecg_c1_<task>.yaml`:

| setting | value |
|---|---|
| model | 512 / 9 / 8, the pretraining encoder |
| pooling | mean pool |
| epochs | 30 |
| lr | 1e-4, warmup 10%, cosine schedule |
| weight decay | 0.05 |
| batch size | 64 |
| early stopping | patience 10 |
| precision | bf16 |

**Multi-label tasks** use one sigmoid per class and BCE loss.

**Checkpoint selection.** The checkpoint is selected on validation:

- accuracy for `ptbxl`, the paper's metric;
- macro AUROC for the multi-label tasks.

The test set is scored once, at that checkpoint.

**Multi-label F1.** F1 depends on a threshold, so per-class thresholds that
maximise F1 are chosen on validation and applied unchanged to test. These
results are reported as `*_tuned`. The fixed-0.5 F1 is reported beside them.

**CPSC is scored per record.** A record's window probabilities are averaged,
which is how the challenge scored it. Window-level scores are kept under
`window_*` in `results.json`.

## About the paper's (v1) numbers

The collector prints PhysioWave v1's numbers beside the C1 rows. They are not
like-for-like, and the table says so:

- v1 scored 4.1 s windows, not records;
- it used min-max or z-score normalisation;
- it used a fixed multi-label threshold;
- it used its own random splits.

The C1 rows score each record once, on a split nothing was selected on.

## Files

| file | what |
|---|---|
| `scripts/download_ecg_downstream.sh` | the three downloads (S3 for PTB-XL and Chapman; PhysioNet's Kaggle copies for CPSC, or `CPSC_SOURCE=physionet` for parallel PhysioNet HTTP) |
| `scripts/fetch_physionet_http.py` | parallel, resumable WFDB mirror from a PhysioNet HTTP directory |
| `ECG/ecg_downstream_prep.py` | raw → `{train,val,test}.h5` + `split.json` |
| `configs/finetune/ecg_c1_*.yaml` | per-task model and training settings |
| `ECG/finetune_ecg_c1.sh` | one run (TASK, MODE); exports the L12_500 encoder from a pretraining checkpoint |
| `scripts/slurm/cineca_ecg_downstream.sbatch` | everything in one job: 1 node, 4 GPUs (one task each), 6 h |
| `scripts/collect_ecg_downstream.py` | the results table (text, JSON, Markdown) |
| `physiowave/ecg_c1/downstream.py` | the ECG downstream model (EEG's with the ECG routes, lead vocabulary and wavelets) |
