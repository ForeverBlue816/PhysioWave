# Running PhysioWave on CINECA Leonardo

Everything site-specific lives in [`scripts/cineca_env.sh`](../scripts/cineca_env.sh),
which every launcher sources. Off-cluster the same file falls back to
repository-relative paths, so the launch scripts work unchanged on a laptop.

## Your account and directories (once)

The repository names no account, user or personal directory. Put yours in
`~/.physiowave_env`, which `cineca_env.sh` sources, and the account also in
`~/.bashrc`, because `sbatch` reads it from the shell that submits:

```bash
cat > ~/.physiowave_env <<'EOF'
export PW_FAST_ROOT="$FAST/<your directory>"      # checkpoints, caches, manifests
export SBATCH_ACCOUNT=<account> SALLOC_ACCOUNT=<account>
EOF
echo 'export SBATCH_ACCOUNT=<account> SALLOC_ACCOUNT=<account>' >> ~/.bashrc
```

Without `PW_FAST_ROOT`, checkpoints go to `$FAST/$USER`. The job scripts carry
no `#SBATCH -A` line; pass `-A <account>` to `sbatch` if you prefer that to the
variable.

## Storage layout

| What | Where | Why |
|---|---|---|
| Code | `$HOME/PhysioWave-T` | backed up; 50 GB quota |
| Training venv | `$HOME/pw` | built on top of the `cineca-ai` module |
| Data-prep venv | `$HOME/pwprep` | `mne` / `wfdb`, kept away from the training env |
| Checkpoints, logs | `$PW_FAST_ROOT/runs` | permanent, project-shared, low-latency I/O |
| SSL + preprocess caches | `$PW_FAST_ROOT/cache` | rebuilt rarely, read constantly |
| Manifests | `$PW_FAST_ROOT/manifests` | small, must outlive a job |
| Training corpora | `$SCRATCH/bio/{eeg,ecg,emg}` | large, rebuildable |

> `$SCRATCH` is **temporary**: files are purged 40 days after creation. Only the
> HDF5 corpora live there, because they can be rebuilt from the raw archives.
> Anything you cannot rebuild belongs in `$FAST`.

Inside each modality directory, one subdirectory per registry dataset id:

```
$SCRATCH/bio/eeg/tueg/*.h5      $SCRATCH/bio/ecg/mimic_iv_ecg/*.h5
$SCRATCH/bio/eeg/siena/*.h5     $SCRATCH/bio/ecg/ptbxl/{train,val,test}.h5
$SCRATCH/bio/eeg/tuab/tuab_{train,val,test}.h5
                                $SCRATCH/bio/emg/ninapro_db6/*.h5
                                $SCRATCH/bio/emg/epn612/{train,val,test}.h5
```

The ids are the keys of `REGISTRY` in
[`physiowave/data/registry.py`](../physiowave/data/registry.py).

## Which environments you need

Two. One is enough to *train*; the split exists because `mne` drags in a newer
`numpy`/`scipy`/`matplotlib` that would shadow the ones `cineca-ai` compiled for
Leonardo. Data preparation is a one-off CPU job that never runs alongside
training, so it gets its own env.

Actual third-party imports across the whole repository:

- **hard:** `torch numpy scipy h5py pandas PyWavelets scikit-learn pyyaml tqdm`
- **soft** (guarded by `try/except`): `tensorboard`, `omegaconf`
- **data prep only:** `mne` (EEG EDF), `wfdb` (ECG)
- **tests only:** `pytest`, `ruff`

`requirements.txt` additionally lists `transformers`, `accelerate`, `datasets`,
`evaluate`, `peft`, `matplotlib` and `seaborn`, which **nothing imports**. Do not
install them here: `accelerate` and `peft` depend on `torch`, pip cannot see the
module-provided torch, and it will pull a ~2.5 GB pip wheel that shadows the
Leonardo build and destroys multi-node NCCL performance.

## Install order

```bash
# 1. Log in (see the group's CINECA tutorial for the STEP client)
ssh <user>@login.leonardo.cineca.it

# 2. Code
mkdir -p $HOME && cd $HOME
git clone <repo> PhysioWave-T

# 3. Modules -- BEFORE creating the venv
module load profile/deeplrn
module av cineca-ai                 # confirm 4.3.0 exists
module load cineca-ai/4.3.0

# 4. Verify the module's interpreter before building anything on top of it
python -V                                                    # need >= 3.10
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; assert hasattr(torch.amp,'GradScaler')"
```

That last assertion is a hard gate:
[`pretrain_main.py`](../physiowave/train/pretrain_main.py) constructs
`torch.amp.GradScaler("cuda", ...)` unconditionally, which only exists from
torch 2.3. If it fails, pick a newer `cineca-ai` from `module av`.

```bash
# 5. Training env
cd $HOME && python -m venv pw --system-site-packages && source pw/bin/activate

# 6. Install only what the module does not already provide
for m in numpy scipy h5py pandas sklearn yaml tqdm pywt; do
  python -c "import $m" 2>/dev/null && echo "OK   $m" || echo "MISS $m"
done
pip install --no-cache-dir --no-deps PyWavelets     # usually the only gap
pip install --no-cache-dir nvitop                   # GPU monitoring

# 7. Confirm torch was NOT replaced -- must still point into .../cineca-ai/...
python -c "import torch; print(torch.__file__)"

# 8. Data-prep env (separate, does not use cineca-ai)
deactivate && module purge
module load python/3.11
cd $HOME && python -m venv pwprep && source pwprep/bin/activate
pip install --no-cache-dir mne wfdb h5py numpy scipy pandas tqdm
```

## Two environments never share a shell

`cineca-ai` and the standalone `python/3.11` module pull in different builds of
the same dependency (`bzip2/1.0.8-ib3znej` vs `bzip2/1.0.8-gp5wcz5`), which the
module system reports as a conflict. Switching between the training and the
data-prep environment therefore means starting from `module purge`:

```bash
# training
module purge && module load profile/deeplrn cineca-ai/4.3.0
source $HOME/pw/bin/activate

# data preparation
module purge && module load python/3.11
source $HOME/pwprep/bin/activate
```

## Never launch through `torchrun`

`torch.distributed.run` spawns its workers with `sys.executable`. The `torchrun`
console script on PATH belongs to the cineca-ai environment, so its
`sys.executable` is the *module's* interpreter and the workers cannot see
anything pip installed into the venv -- `pywt` among them -- even with the venv
active in the parent shell. The launch scripts always use

```bash
python -m torch.distributed.run ...
```

so the venv's interpreter is the parent and the workers inherit its
site-packages. `scripts/cineca_env.sh` exposes this as `${PW_TORCHRUN[@]}`.

Note that `torchrun` is absent from PATH on the login node but present on a
compute node, so preferring it fails only where the GPUs are.

## Smoke test (login node, CPU, free)

```bash
module load profile/deeplrn cineca-ai/4.3.0 && source $HOME/pw/bin/activate
cd $HOME/PhysioWave-T
pip install --no-cache-dir --no-deps pytest && pytest tests/ -q
PYTHON=python bash scripts/run_tpami.sh smoke
PYTHON=python bash scripts/run_tpami.sh all --dry-run
```

## Interactive GPU debugging

```bash
srun --nodes=1 --gpus=4 --ntasks-per-node=4 --cpus-per-task=8 \
     --account <account> -p boost_usr_prod --pty /bin/bash
cd $HOME/PhysioWave-T && source scripts/cineca_env.sh
bash EEG/pretrain_eeg.sh > ~/eeg.log 2>&1 & nvitop
```

Four tasks of eight CPUs is CINECA's shape for a GPU job, one task per GPU.
`--pty` attaches the terminal to task 0 only, so this is a single interactive
shell and a `torchrun --nproc_per_node=4` typed into it is a single torchrun.

Note that task 0's cgroup is the eight CPUs of *its* task, not the node's
thirty-two. Four ranks each with `NUM_WORKERS=8` would then put thirty-two
dataloader workers on eight cores. Either lower `NUM_WORKERS`, or ask for the
shape the sbatch scripts use, which gives one task the whole node:

```bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 \
     --mem=494000 --account <account> -p boost_usr_prod --pty /bin/bash
```

### When srun refuses

```
srun: error: Unable to allocate resources: More processors requested than permitted
```

This is an association or QOS limit, and it counts what you are *already*
holding. Check for a live allocation before rewriting the request:

```bash
squeue -u $USER
sacctmgr show assoc user=$USER format=account,partition,qos,grptres%30,maxtres%30
sacctmgr show qos format=name,maxtresperuser%30,maxtresperjob%30
```

Billing is by CPU hours with one GPU counted as eight CPUs, so a four-GPU node
bills 32 CPU-hours per wall-clock hour either way.

Preparation scripts (`EEG/download_p300.py`, `EEG/physio_p300_finetune.py`)
need no GPU. Decoding on a compute node avoids the login node's cgroup killing
the workers, and a serial partition is both cheaper and easier to schedule than
`boost_usr_prod`.

## Batch jobs

```bash
# Pretraining, one modality per job
sbatch --export=ALL,MODALITY=eeg,EPOCHS=100  scripts/slurm/cineca_pretrain.sbatch
sbatch --export=ALL,MODALITY=ecg             scripts/slurm/cineca_pretrain.sbatch
sbatch --export=ALL,MODALITY=semg            scripts/slurm/cineca_pretrain.sbatch

# RALF fusion; encoders are discovered from $PW_FAST_ROOT/runs automatically
sbatch scripts/slurm/cineca_fusion.sbatch

# Downstream fine-tuning, one node
sbatch --export=ALL,MODALITY=emg,TASK=db5,IN_CHANNELS=16,NUM_CLASSES=53,EPOCHS=100 \
       scripts/slurm/cineca_finetune.sbatch
```

Both templates refuse to start when the configured corpus has no `*.h5` files,
rather than falling back to the synthetic smoke dataset — a silent fallback would
spend a 24 h four-node reservation producing numbers that are not results.

Override anything through `--export`: `PW_CKPT_ROOT`, `PW_DATA_EEG`, `DATASETS`,
`PW_VENV`, `PROJECT_DIR`, `EPOCHS`, `BATCH_SIZE`, `PRECISION`, `EXTRA`.

## NinaPro DB5 (EMG fine-tuning)

`EMG/db5_finetune.py` converts the raw `.mat` files. Two traps it handles:
the `exercise` field inside the `.mat` disagrees with the file name (E1 reads
3, E2 reads 1), and `restimulus` restarts at 1 in every exercise file, so
concatenating without an offset merges 52 movements into 23.

```bash
# download (login node -- compute nodes have no internet)
mkdir -p $SCRATCH/bio/emg/db5/raw && cd $SCRATCH/bio/emg/db5/raw
for n in $(seq 1 10); do
  curl -fL --retry 3 -O "https://ninapro.hevs.ch/files/DB5_Preproc/s${n}.zip"
  unzip -q -o "s${n}.zip" && rm "s${n}.zip"
done

# convert (pwprep environment: needs scipy for .mat)
python EMG/db5_finetune.py --root $SCRATCH/bio/emg/db5/raw \
                           --out-dir $SCRATCH/bio/emg/db5 \
                           --window 512 --stride 128

# fine-tune (pw environment, inside a GPU allocation)
TASK=db5 IN_CHANNELS=16 NUM_CLASSES=53 NUM_GPUS=4 bash EMG/finetune_emg.sh
```

`--window` must be a multiple of the model's `patch_size`; `model.py` asserts
`T % patch_size == 0`. 512 samples is 2.56 s at 200 Hz and fits inside 99% of
the movement segments.

## Monitoring

```bash
squeue --me
scancel <JOBID>
```
