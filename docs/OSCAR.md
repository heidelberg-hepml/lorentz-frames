# Training on Brown's Oscar cluster (CCV)

A follow-along recipe for this repo on [Oscar](https://docs.ccv.brown.edu/oscar): start at
SSH, end at a results table. Oscar-specific facts (directories, partitions, `interact`,
module workflow) follow the CCV documentation; the generic SLURM+Apptainer variant lives in
[`SLURM.md`](SLURM.md), and the science workflow (which model, which knobs, seeds, tables)
in [`GUIDE.md`](../GUIDE.md).

## 0. Connect

```bash
ssh your-brown-username@ssh.ccv.brown.edu       # replace your-brown-username; Brown credentials (same as Canvas)
```

You land on a **login node** (`[you@login00X ~]$`). Login nodes are for file management,
editing, installs, and *submitting* jobs only — do **not** run trainings, tests, or
`find_lr.py` on them (heavy processes get killed). Compute happens through `interact`
(interactive session on a compute node) or `sbatch` (batch job).

## 1. Know the three directories (this determines where everything goes)

| dir | path | size | properties | use it for |
|---|---|---|---|---|
| home | `~` | 100 GB, per-user | many-small-files optimized, snapshots | **repo clone + venv** |
| data | `~/data/<group>` | ≥256 GB, per-group | big-file reads, backed up, **permanent** | **the dataset + finished runs** |
| scratch | `~/scratch` | 512 GB soft / 12 TB hard | fast big-file I/O, **files unread for 30 days are PURGED** | **live `runs/` output** |

Check your quotas any time with `checkquota`. The CCV-recommended pattern is exactly what
we set up below: read inputs from `~/data`, write outputs to `~/scratch`, copy keepers back
to `~/data` when a run finishes (step 9). Mind the *inode* quota too (a venv is ~50k files — fine in
home, don't put it in data).

> **Scratch purge is per-file by atime** (last read). A 3-seed campaign finishes well inside
> 30 days, but if you pause mid-campaign, `find ~/scratch -atime +25` shows what's at risk —
> copy checkpoints you care about to `~/data` (step 9).

## 2. One-time setup (on the login node — this part is allowed there)

Python ≥ 3.10 and a CUDA-tuned torch both come from the **NGC PyTorch container module**:
`module load ngc-pytorch-container/25.08-py3-ayk4` is *supposed* to set
`$NGC_PYTORCH_CONTAINER` to the 25.08 image, and every python command in this guide runs
inside it via `apptainer exec --nv "$NGC_PYTORCH_CONTAINER" …`. Bare `python` on Oscar is
the system 3.9 — never use it for this repo (too old, and no torch).

> **Verify the image — the module has shipped mis-labelled.** On Oscar the `25.08-py3-ayk4`
> modulefile has `setenv`'d `$NGC_PYTORCH_CONTAINER` to the **24.03** sif
> (check: `module show ngc-pytorch-container/25.08-py3-ayk4 | grep NGC_PYTORCH_CONTAINER`).
> The 24.03 image silently breaks this repo — its python has no `ensurepip` (so
> `python -m venv` fails and pip leaks into `~/.local`), and its torch 2.3 is too old for
> `torch-geometric >= 2.6` (`torch.compiler has no attribute is_compiling` at import). So do
> **not** trust the module's value: resolve the real 25.08 sif and hard-stop if it isn't one
> (the block below does this). If it's still broken when you read this, file a CCV ticket.

```bash
# repo + venv live in home
cd ~
git clone https://github.com/t0mnt/GTagger-experiments.git
cd GTagger-experiments

module load ngc-pytorch-container/25.08-py3-ayk4
echo "$NGC_PYTORCH_CONTAINER"        # -> the image the module chose (may be WRONG, see above)

# Trust the resolved sif, not the module: if it isn't a 25.08 image, point at the real one
# directly; then hard-stop if we STILL don't have 25.08 (everything below would fail on 24.03).
case "$NGC_PYTORCH_CONTAINER" in
  *25.08*) : ;;   # module happened to be correct -- keep it
  *) export NGC_PYTORCH_CONTAINER="$(ls /oscar/rt/sw/external/ngc-pytorch-container/25.08-py3/*.sif | head -1)" ;;
esac
case "$NGC_PYTORCH_CONTAINER" in
  *25.08*) echo "using $NGC_PYTORCH_CONTAINER" ;;
  *) echo "ERROR: no 25.08 image resolved ($NGC_PYTORCH_CONTAINER); fix the module / path first"; return 1 2>/dev/null || exit 1 ;;
esac

# containers auto-mount only $HOME and /tmp; bind the data/scratch trees so the
# symlinks wired up below keep working inside the container
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

# make both permanent, for new shells and for the sbatch scripts below. Persist the RESOLVED
# sif AFTER the module load so it overrides the module's (possibly wrong) setenv in new shells.
echo 'module load ngc-pytorch-container/25.08-py3-ayk4' >> ~/.bashrc
echo "export NGC_PYTORCH_CONTAINER=\"$NGC_PYTORCH_CONTAINER\"" >> ~/.bashrc
echo 'export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"' >> ~/.bashrc

# a venv that INHERITS the container's stack (torch, torch-geometric, numpy, ...),
# created from INSIDE the container; pip adds only what the image lacks.
# Filtered out of requirements.txt before installing:
#   - torch:    keep the container's CUDA-tuned build (a pip torch would clobber it)
#   - xformers: not in the image; skipped by default (opt back in later -- see the
#     "Opting back into xformers" note below, a venv-only change)
#   - the lgatr/lloca [xformers-attention] extras -> plain lgatr/lloca (same reason)
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc '
  # fail fast: a missing ensurepip means the WRONG image (24.03) -- do not let pip
  # silently fall back to a ~/.local user-install against the container python.
  python -m venv --system-site-packages venv || { echo "venv creation FAILED (no ensurepip -> not the 25.08 image); aborting"; exit 1; }
  source venv/bin/activate
  pip install --upgrade pip
  pip install -e .
  sed -E -e "/^torch[>=<]/d" -e "/^xformers/d" -e "s/\[xformers-attention\]//" \
      requirements.txt > /tmp/reqs-oscar.txt
  pip install -r /tmp/reqs-oscar.txt
'

# sanity: torch must still be the CONTAINER's own build -- a local +cuXXX / nvXX.XX string
# (e.g. 2.11.0+cu130 on the 25.08 image, or 2.3.0a0...nv24.3 on 24.03), NOT a plain pip wheel;
# torch-geometric should now be >= 2.6
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python -c "import torch, torch_geometric, numpy; \
             print(torch.__version__, torch_geometric.__version__, numpy.__version__)"
'
```

Notes:
- **torch-geometric**: the image ships 2.3.1 but the repo needs ≥ 2.6.0 (its ptr-only
  `MeanAggregation` calls raise `NotImplementedError` on ≤ 2.5 — verified by test suite),
  so pip installs a newer one *into the venv*, shadowing the container's copy. Safe:
  PyG ≥ 2.4 is pure Python and does not pull in a new torch.
- **numpy**: the repo pins < 2.0 (weaver compatibility); if the image carries numpy 2.x,
  pip shadows it in the venv the same way. The sanity line above shows what won.
- **xformers-free is fine**, with one override to remember: all 8 hybrids and most
  baselines never touch xformers (the hybrids' L-GATr stages use dense masks → native
  torch attention, the GPS models plain torch attention, and learned frames use
  `equivectors: equimlp`, already the default; GUIDE §7). Only the four attention
  baselines whose configs pin `attention_backend: xformers` — `tag_transformer`,
  `tag_top_transformer`, `tag_lgatr`, `tag_slim` — would crash on a GPU without it.
  For those, override the backend: `model.attention_backend=flash` if the NGC image
  ships flash-attn (`apptainer exec "$NGC_PYTORCH_CONTAINER" python -c "import flash_attn"`
  — it usually does), else `=flex` (pure torch, slower, and its torch.compile path is
  version-sensitive). Either way, validate the override with one §3-style quick run
  on gpu-debug before a real job.
- **Opting back into xformers** (to run `tag_lgatr`/`tag_slim` on their default backend
  instead of the override above). This is a **venv-only change** — no re-clone, no wipe,
  nothing outside the venv is touched; worst case `rm -rf venv` and redo this step:

  ```bash
  apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
    source venv/bin/activate
    # --no-deps is the whole trick: xformers pins its own torch, and without the flag
    # pip drags that torch into the venv, SHADOWING the container CUDA build (same
    # failure class as the ~/.local leak). --no-deps installs only xformers itself.
    pip install --no-deps -U xformers
    python -m xformers.info | head -20   # want: memory_efficient_attention available,
                                         # and the torch line matching the container build
  '
  ```

  If `xformers.info` errors (undefined symbols = wheel built against a different torch
  ABI than the NGC build), fall back to a source build inside the container — it has the
  full CUDA toolchain: `TORCH_CUDA_ARCH_LIST=<your GPU arch, e.g. 8.0 for A100>
  pip install --no-deps --no-build-isolation xformers` (slow, ~20 min). Either way,
  finish with one §3-style quick run of `tag_slim` before a real job.

Now wire the directories per §1 — dataset into `data`, run output into `scratch`:

```bash
# dataset -> ~/data (permanent, backed up). Set your allocation ONCE (find it: `ls ~/data/`);
# $USER fills in automatically. NB: don't paste raw <angle-bracket> placeholders -- the shell
# reads `<` as a redirect and errors with `bash: ...: No such file or directory`.
GROUP=your-allocation            # <-- replace with your ~/data group dir (from `ls ~/data/`)
mkdir -p ~/data/$GROUP/$USER/gtagger
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py toptagging'   # ~1.5 GB download (file mgmt: login node OK)
mv data/toptagging_full.npz ~/data/$GROUP/$USER/gtagger/
ln -s ~/data/$GROUP/$USER/gtagger/toptagging_full.npz data/toptagging_full.npz

# run output -> ~/scratch (fast, purged; we copy keepers back at the end)
mkdir -p ~/scratch/gtagger_runs
ln -s ~/scratch/gtagger_runs runs
```

**Dataset placement rule of thumb** — three tiers by size and replaceability: the
tiny `data/*_mini.npz` smoke files ship with the repo and stay in **home**; the
1.5 GB `toptagging_full.npz` lives in **`~/data`** (permanent, backed up, symlinked
above); anything JetClass-sized (§2.1 JetClass, §2.2 TopTagXL — ~190 GB ROOT trees)
goes on **`~/scratch`** behind a symlink, accepting the 30-day-idle purge trade
since a redownload is cheaper than burning the backed-up quota (§1).

### 2.1 JetClass (only if you run the jctagging campaign)

The JetClass dataset is ~190 GB of tars extracting to roughly as much again — too big
for home, and it strains a group's `~/data` quota, so **scratch** is the natural home.
The purge risk manages itself while you train: the streaming loader reads the ROOT
files continuously, refreshing their atime — an *active* campaign is purge-safe, but
files idle > 30 days are at risk (§1).

```bash
# download + extract (hours — run in a CPU interact session, not on the login node)
interact -n 4 -m 16g -t 12:00:00
mkdir -p ~/scratch/jetclass && ln -s ~/scratch/jetclass ~/GTagger-experiments/data/JetClass
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py jetclass'
rm ~/scratch/jetclass/*.tar     # reclaim the ~190 GB of tars once extraction finished
exit
```

Training swaps the config name and recipe in the *same* §4/§5 commands (science:
GUIDE §5.1 — shared epochs=5, wd=0; fill each `jc_<hybrid>.yaml`'s `???` from the
jctagging sweep, not the top-tagging one):

```bash
# §4 becomes:  python find_lr.py -cn jctagging model=tag_<hybrid> save=false +lr_find.find_batch_size=true
# train.sh:    python run.py -cp config -cn jctagging model=tag_<hybrid> training=jc_<hybrid> gpus=1
```

### 2.2 TopTagXL (only if you run the toptagxl campaign)

Same scratch treatment as JetClass (it is another ~100M-jet ROOT tree with the same
streaming loader, so the same size and atime/purge reasoning applies). The collector
reads the file list + md5 checksums from Zenodo record 10878355's API at download
time, then verifies and extracts exactly like §2.1:

```bash
interact -n 4 -m 16g -t 12:00:00
mkdir -p ~/scratch/toptagxl && ln -s ~/scratch/toptagxl ~/GTagger-experiments/data/toptagxl
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python data/collect_data.py toptagxl'
rm ~/scratch/toptagxl/*.tar     # reclaim the tar space once extraction finished
exit
```

Commands swap exactly as in §2.1: `-cn toptagxl` + `training=xl_<hybrid>`, with the
`???` knobs filled from a `find_lr.py -cn toptagxl` sweep (science: GUIDE §5.2 —
binary task on JetClass-wide inputs, shared epochs=5, wd=0; shrink
`data.val_files_range` before training, the shipped default is a 10M-jet
validation pass).

## 3. Smoke-test on a compute node

Never on the login node — grab a short interactive CPU session for the tests, then a
GPU-debug session for the model smoke:

```bash
# CPU: the invariance/equivariance suites (~6 min). No --nv on a CPU node.
interact -n 4 -m 16g -t 00:30:00
cd ~/GTagger-experiments
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && pytest tests/experiments/test_tag_equivariance.py tests/experiments/test_tag_invariance.py -q'
exit

# GPU: one tiny training end-to-end (gpu-debug = short wait, short cap)
interact -q gpu-debug -g 1 -n 4 -m 20g -t 00:30:00
cd ~/GTagger-experiments
nvidia-smi                                   # confirm you see a GPU (host side)
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python run.py -cp config_quick -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS save=false gpus=1
'
exit
```

If `torch.cuda.is_available()` is `False` inside the container, the usual causes are a
missing `--nv` flag or not actually being on a GPU node (`nvidia-smi` on the host settles
which) — the torch build itself comes from the NGC image and is known-good.

## 4. Find batch size + LR per model (GPU interactive)

One session per model you plan to train (or chain them in one longer session):

```bash
interact -q gpu -g 1 -n 8 -m 48g -t 02:00:00     # add -f <feature> to pin a GPU type; `nodes gpu` lists them
cd ~/GTagger-experiments
apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc '
  source venv/bin/activate
  python find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
      save=false +lr_find.find_batch_size=true
'
#  ->  reuse with:  training.batchsize=<N> training.lr=<lr>
```

Fill each printed pair into that model's `config/training/top_<Model>.yaml` (they are the
only `???` keys — the shared recipe pins epochs=20, AdamW, warmup-cosine; GUIDE §5–6).

## 5. Submit the real training

Two files: the sbatch header wraps a payload script that runs inside the container
(the `apptainer run --nv "$NGC_PYTORCH_CONTAINER" script.sh` pattern).

`train.sh` (the payload — one per model, or parametrize `$MODEL`; `chmod +x train.sh`):

```bash
#!/bin/bash
source ~/GTagger-experiments/venv/bin/activate
cd ~/GTagger-experiments

python run.py -cp config -cn toptagging \
    model=tag_LorentzNetLGATrSlimGraphGPS \
    training=top_LorentzNetLGATrSlimGraphGPS \
    data.dataset=full gpus=1
# -cp config is REQUIRED: run.py defaults to the tiny config_quick tree,
# which has no top_<Model> training recipes.
```

`train.sbatch`:

```bash
#!/bin/bash
#SBATCH -J gtagger
#SBATCH -p gpu                    # partition; `allq gpu` shows load. gpu-he needs High-End priority
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00               # raise for the heavy CGENN-GPS (~a day/trial on a top GPU)
#SBATCH -o slurm-%j.out
# #SBATCH -a <account>            # only if you belong to a condo/priority account (see `condos`)
# #SBATCH -f ampere               # optionally pin a GPU architecture/feature

module load ngc-pytorch-container/25.08-py3-ayk4
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"

apptainer run --nv "$NGC_PYTORCH_CONTAINER" ~/GTagger-experiments/train.sh
```

```bash
sbatch train.sbatch
myq                       # your queue; `squeue -u $USER -t PENDING --start` estimates start time
tail -f slurm-<jobid>.out # or runs/<exp>/<run>/out_0.log once it starts
myjobinfo                 # time/memory actually used after it finishes
scancel <jobid>           # if needed
```

Each finished run prints its `table test: … \\` row into the log (GUIDE §4).

## 6. Seeds (3 trials → mean ± std)

After trial 1 finishes, submit the same run twice more as **fresh-trial warm starts**
(never plain warm starts — those reload the trained model and its finished scheduler;
GUIDE §8). In `train.sh`, replace the `python run.py` line with:

```bash
python run.py -cp ~/GTagger-experiments/runs/EXPNAME/RUNNAME -cn config \
    warm_start_idx=PREV_RUN_IDX warm_start_load=false   # substitute EXPNAME/RUNNAME/PREV_RUN_IDX
```

(`run_idx` is 0 for the first run, 1 after the first warm start, …; the saved `config.yaml`
in the run dir carries everything else.) The run's table row consolidates to
`[N trials] $mean ± std$` automatically.

## 7. The full campaign (which models, and which need the LR finder)

The study's grid is the 8 hybrids. **All 8 need §4** (their recipes deliberately leave
`batchsize`/`lr` as `???`); everything else in their shared recipe is already decided:

```bash
MODELS="tag_PlainGraphTrans tag_PlainGraphGPS \
        tag_ParticleNetParTGraphTrans tag_ParticleNetParTGraphGPS \
        tag_CGENNLGATrGraphTrans tag_CGENNLGATrGraphGPS \
        tag_LorentzNetLGATrSlimGraphTrans tag_LorentzNetLGATrSlimGraphGPS"

# in a GPU interact session (§4): one sweep per model, fill each top_<Model>.yaml
for M in $MODELS; do
  apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc \
    "source venv/bin/activate && python find_lr.py -cn toptagging model=$M save=false +lr_find.find_batch_size=true"
done

# then one sbatch per model (§5), then 2 more fresh-trial seeds each (§6)
```

Once the recipes are filled, shake down the config axes before (or alongside) the seed
runs, in this order — PlainGraphGPS PE/SE variants first (`model.net.use_edge_attr`,
`model.net.use_rwse`, `model.net.norm=batch|layer` — confirm each trains), then every
model under both graph metrics (`model.net.knn_metric=deltaR|minkowski`), then the
LLoCa models (Plain / ParticleNet-ParT) under PD frames (`model/framesnet=learnedpd`).
See GUIDE §6's shakedown note for the reasoning.

The **baseline reference rows** (`tag_ParT`, `tag_particlenet`, `tag_lgatr`, `tag_slim`,
`tag_lorentznet`, `tag_transformer`, …) do **not** need the LR finder — they run under
their published recipes, which already pin lr/batchsize/budget:

```bash
# same train.sh/train.sbatch pattern as §5, with the payload line:
python run.py -cp config -cn toptagging model=tag_ParT training=top_ParT data.dataset=full gpus=1
# likewise: tag_lgatr+top_lgatr, tag_slim+top_slim, tag_lorentznet+top_lorentznet, ...
# xformers isn't installed (§2), so the four attention baselines need a backend override
# (flash if the image has flash-attn, else flex -- see the §2 note; validate on gpu-debug):
#   model=tag_lgatr training=top_lgatr model.attention_backend=flash
# (same for tag_slim, tag_transformer, tag_top_transformer; all other rows run as-is)
```

(Heads-up on wall time: order the queue submissions cheapest-first; `CGENNLGATrGraphGPS`
is the expensive one — budget ~a day per trial on a top GPU — while the slim models are
orders of magnitude lighter.)

## 8. The comparison table

```bash
apptainer exec "$NGC_PYTORCH_CONTAINER" bash -lc \
  'source venv/bin/activate && python aggregate_table.py --runs runs --split test --out comparison.tex'
```

## 9. Save what matters (scratch purges!)

```bash
# finished runs you want to keep -> data (permanent, backed up). Set GROUP as in step 2
# (`ls ~/data/`) and replace EXPNAME with your run's exp_name; $USER fills in.
GROUP=your-allocation
cp -r ~/scratch/gtagger_runs/EXPNAME ~/data/$GROUP/$USER/gtagger/runs_keep/
```

Do this at the end of the campaign (and for any long pause > ~3 weeks). `comparison.tex`,
the `table_metrics_*.json` files, `out_*.log`, and the best-model checkpoints are the
irreplaceable parts.

## Quick reference

| task | command |
|---|---|
| run anything in the container | `apptainer exec --nv "$NGC_PYTORCH_CONTAINER" bash -lc 'source venv/bin/activate && <cmd>'` |
| my jobs / all GPU jobs | `myq` / `allq gpu` |
| GPU types available | `nodes gpu` |
| quotas | `checkquota` |
| condo limits (if any) | `condos` |
| interactive CPU / GPU | `interact -n 4 -m 16g -t 01:00:00` / `interact -q gpu -g 1` |
| scratch purge check | `find ~/scratch -atime +25` |

Alternatives to raw SSH that CCV supports, if you prefer them: Open OnDemand (browser
terminal + Jupyter at the CCV portal) and VS Code Remote-SSH (docs: "Remote IDE").
