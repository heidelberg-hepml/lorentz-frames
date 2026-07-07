# Training on Brown's Oscar cluster (CCV)

A follow-along recipe for this repo on [Oscar](https://docs.ccv.brown.edu/oscar): start at
SSH, end at a results table. Oscar-specific facts (directories, partitions, `interact`,
module workflow) follow the CCV documentation; the generic SLURM+Apptainer variant lives in
[`SLURM.md`](SLURM.md), and the science workflow (which model, which knobs, seeds, tables)
in [`GUIDE.md`](../GUIDE.md).

## 0. Connect

```bash
ssh <your-brown-username>@ssh.ccv.brown.edu     # Brown credentials (same as Canvas)
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

```bash
# repo + venv live in home
cd ~
git clone https://github.com/t0mnt/GTagger-experiments.git
cd GTagger-experiments

# the repo needs python >= 3.10; Oscar's system python is older, so load a module
module avail python          # pick the newest 3.1x
module load python/3.11<TAB-complete-the-exact-name>
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt   # pip torch wheels bundle their own CUDA -- no cuda module needed to install
```

Notes:
- If **xformers** fights the resolver, it's optional here: the non-equivariant GPS models use
  plain torch attention, and learned frames work xformers-free with
  `model/framesnet/equivectors=equimlp` (GUIDE §7).
- Make the module load automatic for jobs: you'll repeat `module load python/3.11…` in every
  sbatch script below (or add it to `~/.bashrc`).

Now wire the directories per §1 — dataset into `data`, run output into `scratch`:

```bash
# dataset -> ~/data (permanent, backed up). <group> = your PI's group dir under ~/data
mkdir -p ~/data/<group>/<you>/gtagger
python data/collect_data.py toptagging                       # ~1.5 GB download (file mgmt: login node OK)
mv data/toptagging_full.npz ~/data/<group>/<you>/gtagger/
ln -s ~/data/<group>/<you>/gtagger/toptagging_full.npz data/toptagging_full.npz

# run output -> ~/scratch (fast, purged; we copy keepers back at the end)
mkdir -p ~/scratch/gtagger_runs
ln -s ~/scratch/gtagger_runs runs
```

(The tiny `data/*_mini.npz` smoke files ship with the repo and stay in home.)

## 3. Smoke-test on a compute node

Never on the login node — grab a short interactive CPU session for the tests, then a
GPU-debug session for the model smoke:

```bash
# CPU: the invariance/equivariance suites (~6 min)
interact -n 4 -m 16g -t 00:30:00
source ~/GTagger-experiments/venv/bin/activate && cd ~/GTagger-experiments
pytest tests/experiments/test_tag_equivariance.py tests/experiments/test_tag_invariance.py -q
exit

# GPU: one tiny training end-to-end (gpu-debug = short wait, short cap)
interact -q gpu-debug -g 1 -n 4 -m 20g -t 00:30:00
source ~/GTagger-experiments/venv/bin/activate && cd ~/GTagger-experiments
nvidia-smi                                   # confirm you see a GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python run.py -cp config_quick -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS save=false gpus=1
exit
```

If `torch.cuda.is_available()` is `False`, your torch wheel/driver mismatch: rebuild the venv
from a GPU node following CCV's
[framework-install recipe](https://docs.ccv.brown.edu/oscar/gpu-computing/installing-frameworks-pytorch-tensorflow-jax)
(`interact -q gpu -g 1`, `module purge && unset LD_LIBRARY_PATH`, recreate the venv there).

## 4. Find batch size + LR per model (GPU interactive)

One session per model you plan to train (or chain them in one longer session):

```bash
interact -q gpu -g 1 -n 8 -m 48g -t 02:00:00     # add -f <feature> to pin a GPU type; `nodes gpu` lists them
source ~/GTagger-experiments/venv/bin/activate && cd ~/GTagger-experiments
python find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
    save=false +lr_find.find_batch_size=true
#  ->  reuse with:  training.batchsize=<N> training.lr=<lr>
```

Fill each printed pair into that model's `config/training/top_<Model>.yaml` (they are the
only `???` keys — the shared recipe pins epochs=20, AdamW, warmup-cosine; GUIDE §5–6).

## 5. Submit the real training

`train.sbatch` (one per model, or parametrize `$MODEL`):

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

module load python/3.11<exact-name>
source ~/GTagger-experiments/venv/bin/activate
cd ~/GTagger-experiments

python run.py -cp config -cn toptagging \
    model=tag_LorentzNetLGATrSlimGraphGPS \
    training=top_LorentzNetLGATrSlimGraphGPS \
    data.dataset=full gpus=1
# -cp config is REQUIRED: run.py defaults to the tiny config_quick tree,
# which has no top_<Model> training recipes.
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
GUIDE §8). In the sbatch, replace the `python run.py` line with:

```bash
python run.py -cp ~/GTagger-experiments/runs/<exp_name>/<run_name> -cn config \
    warm_start_idx=<prev run_idx> warm_start_load=false
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
  python find_lr.py -cn toptagging model=$M save=false +lr_find.find_batch_size=true
done

# then one sbatch per model (§5), then 2 more fresh-trial seeds each (§6)
```

The **baseline reference rows** (`tag_ParT`, `tag_particlenet`, `tag_lgatr`, `tag_slim`,
`tag_lorentznet`, `tag_transformer`, …) do **not** need the LR finder — they run under
their published recipes, which already pin lr/batchsize/budget:

```bash
python run.py -cp config -cn toptagging model=tag_ParT training=top_ParT data.dataset=full gpus=1
# likewise: tag_lgatr+top_lgatr, tag_slim+top_slim, tag_lorentznet+top_lorentznet, ...
```

(Heads-up on wall time: order the queue submissions cheapest-first; `CGENNLGATrGraphGPS`
is the expensive one — budget ~a day per trial on a top GPU — while the slim models are
orders of magnitude lighter.)

## 8. The comparison table

```bash
python aggregate_table.py --runs runs --split test --out comparison.tex
```

## 9. Save what matters (scratch purges!)

```bash
# finished runs you want to keep -> data (permanent, backed up)
cp -r ~/scratch/gtagger_runs/<exp_name> ~/data/<group>/<you>/gtagger/runs_keep/
```

Do this at the end of the campaign (and for any long pause > ~3 weeks). `comparison.tex`,
the `table_metrics_*.json` files, `out_*.log`, and the best-model checkpoints are the
irreplaceable parts.

## Quick reference

| task | command |
|---|---|
| my jobs / all GPU jobs | `myq` / `allq gpu` |
| GPU types available | `nodes gpu` |
| quotas | `checkquota` |
| condo limits (if any) | `condos` |
| interactive CPU / GPU | `interact -n 4 -m 16g -t 01:00:00` / `interact -q gpu -g 1` |
| scratch purge check | `find ~/scratch -atime +25` |

Alternatives to raw SSH that CCV supports, if you prefer them: Open OnDemand (browser
terminal + Jupyter at the CCV portal) and VS Code Remote-SSH (docs: "Remote IDE").
