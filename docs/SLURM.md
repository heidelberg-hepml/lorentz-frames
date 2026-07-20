# Training on a SLURM + Apptainer cluster

A minimal recipe for the common HPC setup where **PyTorch is only available inside
an Apptainer/Singularity image** loaded as a module (you can't `pip install torch`
on the login node). The trick is a `--system-site-packages` venv so the repo's
dependencies install *on top of* the container's torch instead of clobbering it.

Replace the `<...>` placeholders with your cluster's values (module name, image
path, partition, GPU spec, account).

## 1. One-time setup (on a login node)

```bash
git clone https://github.com/<you>/GTagger-experiments && cd GTagger-experiments
module load apptainer                       # or: module load singularity
IMG=<path/to/pytorch.sif>                    # your PyTorch container

# a venv that INHERITS the container's torch (so pip won't reinstall/clobber it)
apptainer exec "$IMG" python -m venv --system-site-packages venv

# install the repo + deps inside the container, but NOT torch (the container owns it)
# and not xformers (hard to build against a container torch; we run without it)
apptainer exec "$IMG" bash -lc '
  source venv/bin/activate
  pip install -e .
  sed -E -e "/^torch[>=<]/d" -e "/^xformers/d" -e "s/\[xformers-attention\]//" \
      requirements.txt > /tmp/reqs.txt
  pip install -r /tmp/reqs.txt
'
```

Notes:
- The filter drops only the `torch` and `xformers` lines (and the lgatr/lloca
  `[xformers-attention]` extras) so pip keeps the container's CUDA-matched torch.
  `torch-geometric` stays IN the install even if the container ships one: the repo
  needs ≥ 2.6.0 (its ptr-only `MeanAggregation` calls raise `NotImplementedError`
  on ≤ 2.5), and containers often carry older. pip's copy in the venv shadows the
  container's, safely — PyG ≥ 2.4 is pure Python and does not pull in a new torch.
- **xformers-free is fine**, with one override to remember: all 8 hybrids and most
  baselines never touch xformers (the hybrids' L-GATr stages use dense masks →
  native torch attention, the GPS models plain torch attention, and learned frames
  use `model/framesnet/equivectors=equimlp`, already the default). Only the four
  attention baselines whose configs pin `attention_backend: xformers`
  (`tag_transformer`, `tag_top_transformer`, `tag_lgatr`, `tag_slim`) would crash
  on a GPU without it — run those with `model.attention_backend=flash` (if the
  container ships flash-attn; NGC images usually do) or `=flex` (pure torch), and
  validate the override with a quick config_quick run first. If your cluster can
  build xformers against the container torch, keep the lines in instead.

## 2. Get the data

```bash
apptainer exec "$IMG" bash -lc 'source venv/bin/activate && python data/collect_data.py toptagging'
# -> data/toptagging_full.npz  (~1.5 GB)
```

## 3. Smoke-test on a GPU node

```bash
srun --partition=<gpu-partition> --gres=gpu:1 --time=00:20:00 --pty bash
module load apptainer
apptainer exec --nv "$IMG" bash -lc '
  source venv/bin/activate
  python run.py -cp config_quick -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS save=false gpus=1
'
```

`--nv` exposes the GPU to the container. If your `$HOME`/scratch isn't auto-mounted,
add `--bind <data_dir>:<data_dir>`.

## 4. Find lr + batch size, then train (sbatch)

First (interactively or as a short job) size the batch and lr:

```bash
apptainer exec --nv "$IMG" bash -lc '
  source venv/bin/activate
  python find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
      save=false +lr_find.find_batch_size=true
'   # prints:  ->  reuse with:  training.batchsize=<N> training.lr=<lr>
```

Fill those into `config/training/top_<Model>.yaml`, then submit `train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=lloca
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
# #SBATCH --account=<account>

module load apptainer
IMG=<path/to/pytorch.sif>

srun apptainer exec --nv --bind "$PWD:$PWD" --pwd "$PWD" "$IMG" bash -lc '
  source venv/bin/activate
  python run.py -cp config -cn toptagging \
      model=tag_LorentzNetLGATrSlimGraphGPS \
      training=top_LorentzNetLGATrSlimGraphGPS \
      data.dataset=full gpus=1
'
# -cp config is REQUIRED here: run.py defaults to the tiny config_quick tree,
# which has no top_<Model> training recipes (the command would fail at composition).
```

```bash
sbatch train.sbatch
```

JetClass instead of top-tagging: fetch the ~190 GB dataset with
`python data/collect_data.py jetclass` (put it on big-file storage and symlink
`data/JetClass` there), then swap `-cn toptagging` → `-cn jctagging` and
`training=top_<Model>` → `training=jc_<Model>` in the same commands (GUIDE §5.1).

TopTagXL works the same way: fetch with `python data/collect_data.py toptagxl`
(also ~JetClass-sized; the file list + md5 checksums come from Zenodo record
10878355's API at download time) onto big-file storage with `data/toptagxl`
symlinked there, then `-cn toptagxl` + `training=xl_<Model>`, seeding each
`xl_<Model>.yaml`'s `???` from the swept `jc_` values and confirming with a
`find_lr.py -cn toptagxl` sweep (GUIDE §5.2 — including why to shrink
`data.val_files_range` before training).

## 5. Multiple seeds, and the table

A single submission is one trial. For 3 seeds, submit the **same** run twice more as
**fresh-trial warm starts** — point `-cp`/`-cn` at the saved run config and pass
`warm_start_idx=<prev run_idx> warm_start_load=false` — so each trial trains from a
fresh initialization and the row consolidates to `mean ± std` in that run directory.
(A plain warm start, `warm_start_load=true` default, RELOADS the trained model and its
finished scheduler — that's for eval-reload / continue-training, not seeds; see
`GUIDE.md` §8.) Across *different* models, collect the rows afterwards:

```bash
python aggregate_table.py --runs runs --split test --out comparison.tex
```

(See `GUIDE.md` §8 for the trial/warm-start mechanics, and §6 for the lr/weight-decay
guidance.)
