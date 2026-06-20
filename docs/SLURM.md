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
apptainer exec "$IMG" bash -lc '
  source venv/bin/activate
  pip install -e .
  grep -vE "^torch([>=<]|-geometric|-ema)?\b" requirements.txt > /tmp/reqs.txt
  pip install -r /tmp/reqs.txt
'
```

Notes:
- `requirements.txt` lists `torch`, `torch-geometric`, `torch-ema` and `xformers`.
  The container usually already provides a CUDA-matched torch (and often
  torch-geometric); the `grep -vE` above drops the torch lines so pip keeps the
  container's build. If torch-geometric is *not* in the container, drop only
  `^torch>` / `^torch-ema` and let pip install `torch-geometric`.
- **xformers** is pulled in via `lgatr[xformers-attention]` / `lloca[xformers-attention]`
  and can be hard to build. If it won't install, you can install `lgatr`/`lloca`
  without the extra and run xformers-free: the GraphGPS non-equivariant models use
  plain torch attention, the equivariant ones fall back to non-xformers L-GATr
  backends, and learned frames work with `model/framesnet/equivectors=equimlp`.

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
      training=tag_gtagger_and_friends_default save=false +lr_find.find_batch_size=true
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
  python run.py \
      model=tag_LorentzNetLGATrSlimGraphGPS \
      training=top_LorentzNetLGATrSlimGraphGPS \
      data.dataset=full gpus=1
'
```

```bash
sbatch train.sbatch
```

## 5. Multiple seeds, and the table

A single submission is one trial. For 3 seeds, submit the **same** run twice more
as warm starts (same `exp_name`/`run_name`) so the row consolidates to `mean ± std`
in that run directory. Across *different* models, collect the rows afterwards:

```bash
python aggregate_table.py --runs runs --split test --out comparison.tex
```

(See `GUIDE.md` §8 for the trial/warm-start mechanics, and §6 for the lr/weight-decay
guidance.)
