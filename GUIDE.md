# Using this repo (with the graph-transformer hybrids)

A practical walkthrough for someone who has just cloned the repo and wants to
train models — especially the GraphTrans / GraphGPS hybrid taggers added on top of
the LLoCa baselines. For the upstream paper-reproduction commands see
[`REPRODUCE.md`](REPRODUCE.md); for the method see the papers linked in
[`README.md`](README.md).

---

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Get the top-tagging dataset (~1.5 GB → `data/toptagging_full.npz`):

```bash
python data/collect_data.py toptagging
```

Smoke-test the install on the tiny datasets shipped under `data/` (no GPU needed):

```bash
pytest tests/experiments/test_tag_equivariance.py -q     # 32 invariance checks
python run.py -cp config_quick -cn toptagging save=false # one quick training
```

`config_quick/` mirrors `config/` with tiny models/data — ideal for sanity checks
and reading along with print statements. `config/` is the real training setup.

---

## 2. Repo layout

| path | what |
|---|---|
| `run.py` | entry point: builds an experiment from a hydra config and trains/evaluates it |
| `find_lr.py` | LR range test + optional GPU batch-size finder (see §6) |
| `config/` | real configs; `config_quick/` the tiny mirror |
| `config/model/tag_*.yaml` | one file per tagger (model definition only) |
| `config/training/top_*.yaml` | training budgets / optimizers / schedules |
| `config/model/framesnet/` | LLoCa frame predictors (for non-equivariant models) |
| `experiments/baselines/` | the network implementations |
| `experiments/tagging/wrappers.py` | the wrapper that adapts each net to the tagging pipeline |
| `tests/experiments/` | `test_tag_equivariance.py`, `test_tag_invariance.py`, `test_tag_flops.py` |

---

## 3. The model zoo

Selected with `model=tag_<name>`. Two families of hybrids were added, each in a
2×2 grid of {graph backbone} × {GraphTrans = sequential GNN→transformer, GraphGPS
= interleaved GNN‖attention per layer}:

| backbone | GraphTrans | GraphGPS | equivariance |
|---|---|---|---|
| plain MPNN + torch-MHA | `tag_PlainGraphTrans` | `tag_PlainGraphGPS` | non-equiv → LLoCa frames |
| ParticleNet EdgeConv + ParT attn | `tag_ParticleNetParTGraphTrans` | `tag_ParticleNetParTGraphGPS` | non-equiv → LLoCa frames |
| CGENN + L-GATr | `tag_CGENNLGATrGraphTrans` | `tag_CGENNLGATrGraphGPS` | **equivariant by construction** |
| LorentzNet + L-GATr-slim | `tag_LorentzNetLGATrSlimGraphTrans` | `tag_LorentzNetLGATrSlimGraphGPS` | **equivariant by construction** |

Plus the upstream baselines: `tag_ParT`, `tag_particlenet`, `tag_transformer`,
`tag_graphnet`, `tag_lgatr`, `tag_lorentznet`, `tag_MIParT`, `tag_pelican_fair`, …

**Equivariance comes from one of two routes**, and it determines whether you set a
framesnet:

- **Internally equivariant** (CGENN, LorentzNet-slim, L-GATr, pelican): equivariant
  by construction, run on `framesnet=identity` (the default in their configs). Do
  *not* give them a learned framesnet.
- **Non-equivariant + LLoCa** (Plain, ParticleNet-ParT, ParT, transformer, graphnet):
  made Lorentz-equivariant by canonicalizing inputs into a learned local frame. Set
  `model/framesnet=learnedpd` (or `learnedso13`, …) to enable it; `identity` (default)
  gives the plain non-equivariant baseline.

---

## 4. Running a training

```bash
# a non-equivariant hybrid, made equivariant with learned frames
python run.py model=tag_PlainGraphGPS model/framesnet=learnedso13

# an internally-equivariant hybrid (identity frames; nothing to set)
python run.py model=tag_LorentzNetLGATrSlimGraphGPS

# the hybrid's own recipe (inherits tag_gts_and_friends_default), full data, a GPU
python run.py model=tag_ParticleNetParTGraphGPS training=top_ParticleNetParTGraphGPS \
    data.dataset=full gpus=1
```

Useful overrides: `data.dataset={full,mini}`, `training.iterations=…`,
`training.batchsize=…`, `training.lr=…`, `gpus=N`, `save={true,false}`,
`model.net.knn_metric={deltaR,minkowski}`, `model.net.num_blocks=…`.

Each run prints a paste-ready LaTeX table row at the end:
`table test: <Model> & <frames> & <iters> [N trials] & <params> & <acc> & <auc> &
<rej03> & <rej05> & <rej08> & <time>s & <flops> & <kNN>` (the `[N trials]` tag and
`mean ± std` cells appear once a run directory holds more than one trial, §8).

---

## 5. Configs: model vs training

A `config/model/tag_*.yaml` is **model definition only** — it has no LR, optimizer
or budget. Those come from the **training** config, selected separately. The
top-tagging default is now `tag_gts_and_friends_default` (AdamW, epochs=20,
CosineAnnealingWarmup, wd=0.01) — the shared GT-hybrid recipe — so the new models run
correctly with no `training=` at all (you just fill `lr`/`batchsize` from
`find_lr.py`). Pass `training=top_<baseline>` to run a baseline under its own tuned
recipe (e.g. `top_transformer` = Lion, lr=3e-5, 300k iters) — see §7.

The 8 GT hybrids share one recipe: each `config/training/top_<hybrid>.yaml`
`defaults: [tag_gts_and_friends_default]` (AdamW, **epochs=20**,
CosineAnnealingWarmup, shared `weight_decay=0.01`, validate once/epoch) and only fills
its own `batchsize` + `lr` from `find_lr.py` — that shared budget is what makes the
hybrid-vs-hybrid table fair. The upstream baselines keep their own recipes as
reference rows — `top_ParT` (Ranger, lr=1e-3, 20 epochs), `top_lorentznet` (AdamW,
lr=1e-3, 35 epochs), `top_lgatr` (Lion, lr=3e-4, wd=0.2), `top_particlenet` (lr=1e-2)
— or point them at `tag_gts_and_friends_default` to put them on the same budget.

### 5.1 JetClass

The JetClass task mirrors the top-tagging setup one-to-one, selected with
`-cn jctagging` (dataset: `python data/collect_data.py jetclass` — the full 100M-jet
training set, streamed from ROOT files, so mind the disk). Two differences to know:

- **Wider inputs.** `features: default` adds 10 particle-ID / trajectory scalars, so
  the non-equivariant models see `in_channels = 7 + 10 = 17` (wired automatically by
  `init_physics`).
- **JetClass recipes drop weight decay** (`weight_decay: 0`, matching `jc_ParT`).

The training-config structure repeats the §5 pattern: the task default is
`jc_gts_and_friends_default` (AdamW, CosineAnnealingWarmup, **epochs=5** — the
ParT-standard exposure, 1M steps × batch 512 ≈ 5 passes of the 100M train set —
validate once per nominal epoch), and each hybrid has a `config/training/jc_<hybrid>.yaml`
whose `batchsize`/`lr` are `???` until you fill them from the LR finder **on the
jctagging task** (don't copy the top-tagging values — the wider inputs move both the
memory ceiling and the loss-vs-lr curve):

```bash
python find_lr.py -cn jctagging model=tag_CGENNLGATrGraphGPS save=false \
    +lr_find.find_batch_size=true
# -> fill training.batchsize / training.lr into config/training/jc_CGENNLGATrGraphGPS.yaml
```

(The default `num_iter=300` sweep length is right on JetClass too — it samples the lr
ramp, not the dataset, so the 100M-jet size changes nothing; raising it only biases the
suggestion low, per the finder's docstring. No JetClass-specific overrides are needed.)

Note the `???` markers are for humans — hydra cannot enforce them here (an OmegaConf
`???` never overrides a value inherited from `tag_default`), so an unfilled recipe
silently trains at the 512 / 1e-3 fallback instead of erroring. The JetClass baseline
reference rows keep their published budgets: `jc_ParT` (Ranger, 1M iterations),
`jc_MIParT`, `jc_lgatr` (Lion, bs=512), `jc_transformer`. To cut cost, shrink
`data.train_files_range` — the same 5 passes then apply to the subset; don't lower
the epochs.

---

## 6. Choosing hyperparameters

**Learning rate (and GPU batch size) — `find_lr.py`.** Runs a Leslie-Smith LR
range test under the *training config's* optimizer + param-group ratios (but with
weight-decay and grad-clipping switched off for the sweep — they'd only mask the
divergence) and reports a robust `loss-min/10` peak LR — a safe peak for the
warmup→cosine schedule (it never builds the scheduler; it ramps the LR by hand from 1e-7). The default training
is now `tag_gts_and_friends_default` (AdamW, clip=1.0, wd=0.01), so for the GT
hybrids you sweep with nothing extra. Pass `training=<recipe>` only to match a
*different* optimizer — the LR scale is optimizer-specific, so a Lion baseline (e.g.
`training=top_transformer`) must be swept under Lion. `find_lr.py` now defaults to the real `config/` tree
(full data); add `data.dataset=mini` for a quick trial.

```bash
# LR only (default training is the shared AdamW gts-and-friends recipe)
python find_lr.py -cn toptagging model=tag_CGENNLGATrGraphGPS save=false

# on a GPU: fit the batch size first, then sweep the LR at that size
python find_lr.py -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \
    save=false +lr_find.find_batch_size=true
```

With `+lr_find.find_batch_size=true` it doubles the batch size until CUDA OOM
(running a full train step, so the probe includes optimizer-state memory) and keeps
the largest fitting power of two (`bs_safety=1.0` default; set `<1` to trade the
power of two for headroom), then prints the batch size and LR, e.g.
`-> reuse with: training.batchsize=2048 training.lr=3.1e-04`. Verify the batch size
with a short real run first (it probes one batch, and jets vary in size). Knobs:
`+lr_find.{bs_start,bs_max,bs_safety,num_iter,end_lr}` — keep `num_iter` short (~300;
a longer sweep biases the suggestion lower, it doesn't sharpen it). For models that expose a
`knn_metric`, the sweep pins **`deltaR`** by default so the suggested LRs are comparable across
the family (the LR scale is metric-independent — the model still *trains* under its own
configured metric); pass `+lr_find.force_knn_metric=keep` to sweep each model's own metric
instead (or `=minkowski` to pin that).

**Weight decay.** No automated finder — it can't be range-tested like the LR (its
effect emerges over a full run), so sweep `weight_decay=0,0.01,0.05` (Hydra multirun)
on one model and apply the winner to all. The GT hybrids ship a shared
**`weight_decay: 0.01`** (AdamW) in `tag_gts_and_friends_default`; one value for
the whole family keeps the comparison about architecture. With decoupled decay on
normalized weights it acts mostly as an effective-LR / weight-norm knob
(scale-invariant), so a single value is fair across GNN and transformer parts alike;
norms, biases and class tokens are already excluded (the `ndim<=1` param group) and
framesnets keep `weight_decay_framesnet=0`. The Lion baselines are the exception —
Lion's decay also scales with LR, so the L-GATr (`wd=0.2`, lr=3e-4) and slim
(`wd=2`, lr=3e-5) recipes are the same `lr × wd ≈ 6e-5`; for a Lion run set
`wd ≈ 6e-5 / lr`, not a copied raw number.

**Budget / epochs.** Early stopping is on (`es_patience`), so the iteration count
is an upper bound — but its patience is large, so in practice the budget *is* the
cap. The GT hybrids encode the fair choice in `tag_gts_and_friends_default`:
**epochs=20** (equal data exposure — derived per model as `epochs × batches_per_epoch`,
not one model's ad-hoc 20-epochs / 200k-iters) and **validate once per epoch** so
best-val checkpointing has equal granularity across the family. Check the val curve
converged; the repo always reports the best-validation checkpoint, so over-budgeting
only costs compute, not accuracy.

**Best-checkpoint metric — why `loss` is the default** (`best_model_metric` in
`tag_default`; `accuracy` is the alternative). Both are measured at the same validation
checkpoints — the difference is in what each number can *resolve*. Accuracy is a
thresholded count, `accuracy_score(labels, round(sigmoid(logits)))`: it only changes when
a jet's score crosses 0.5, so it takes values k/N only, and near convergence — where
decision flips are rare — successive checkpoints tie or differ by counting noise
(σ ≈ √(p(1−p)/N)), leaving the argmax to chance. Val loss aggregates every jet's
continuous margin, so it separates checkpoints accuracy sees as identical (better-calibrated
scores with zero flips), and it tracks the threshold-free report metrics (AUC, rejections
at ε_S = 0.3/0.5/0.8) that the table actually shows. Selection by accuracy optimizes the
one working point (0.5) the paper doesn't report, with a coarser ruler.

**After the sweeps — shakedown order for the config axes.** Once every model's
`batchsize`/`lr` are filled in, exercise the variant knobs in this order:

1. **PlainGraphGPS PE/SE variants first** (cheapest model, most toggles — confirm each
   trains): `model.net.use_edge_attr=true|false` (relative Minkowski edge PE),
   `model.net.use_rwse=true|false` (+ `model.net.rwse_k=K`), `model.net.norm=batch|layer`.
2. **Then every model under both graph metrics**: `model.net.knn_metric=deltaR|minkowski`
   (minkowski is the Lorentz-invariant graph; deltaR the eta–phi one).
3. **Then the LLoCa-canonicalized models under PD frames**: `model/framesnet=learnedpd`
   on the Plain and ParticleNet-ParT hybrids (the internally-equivariant CGENN /
   LorentzNet hybrids stay on `identity`).

(The full knob catalogue lives in `todo.md` §3 and `docs/ablations.md`.)

---

## 7. Frames, xformers, and avoiding it

The built-in Transformer / L-GATr taggers and the `lgatr` frame predictor use
xformers' memory-efficient attention (saves ~2× RAM on variable-length jets); on
an H100 you normally just `pip install xformers` and it's the recommended backend.
The new **GraphGPS non-equivariant** models use plain `torch.nn.MultiheadAttention`,
so they need no xformers at all. If you do want a learned framesnet without
xformers, use the **MLP frame predictor**:

```bash
python run.py model=tag_PlainGraphGPS model/framesnet=learnedpd \
    model/framesnet/equivectors=equimlp     # MLP frames, no xformers (vs =lgatr)
```

(`equivectors` ∈ {`equimlp`, `pelican`, `lgatr`}; `equimlp` is the lightest and
xformers-free.) The internally-equivariant hybrids use identity frames and never
touch xformers in the framesnet — their L-GATr stages attend with dense masks
(native torch attention), so all 8 hybrids run on a GPU without xformers installed.
The only models that don't are the four baselines whose configs pin
`attention_backend: xformers` (`tag_transformer`, `tag_top_transformer`,
`tag_lgatr`, `tag_slim`); on an xformers-free install, run those with
`model.attention_backend=flash` (needs the flash-attn package; NGC/cluster
containers usually ship it) or `=flex` (pure-torch FlexAttention — slower, and
its torch.compile path is version-sensitive, so smoke-test it on your GPU first).

---

## 8. Multiple trials and the results table

- **One `run.py` invocation = one trial** (`run_idx=0`) and emits one table row.
- **Several trials of the *same* model** accumulate into `mean ± std` automatically:
  re-run the *same* experiment as a **fresh-trial warm start** — point `-cp`/`-cn` at the
  saved run config and pass `warm_start_idx=<prev run_idx> warm_start_load=false`. It
  increments `run_idx`, shares the run directory, appends to
  `runs/<exp>/<run>/table_metrics_*.json`, and starts from a **new random initialization**
  with a fresh optimizer/scheduler. The final row then reads
  `… & <iters> [N trials] & $acc ± σ$ & …`.
  **Do NOT use a plain warm start (the `warm_start_load=true` default) for trials**: that
  reloads the previous model *and* the finished scheduler, so the "trial" is a correlated
  continuation of the same training — and the reloaded cosine steps past `T_max`, ramping
  the lr back *up* toward its maximum over the run. Plain warm starts are for eval-reload
  and deliberate continue-training (with `training.scheduler_scale`) only.
  Seed bookkeeping: with the default `seed=null` every trial draws a fresh init
  automatically; if you pin `seed`, vary it per trial (`seed=1`, `seed=2`, …) or all
  trials are identical. (Batch *order* is sampler-seeded and identical across trials
  either way.)
- **Different models do *not* merge** into one table — each lands in its own run
  directory with its own row. To build a comparison table, run
  `python aggregate_table.py --runs runs --split test --out comparison.tex` (it collects each
  run's row into one LaTeX table; its `COLUMNS` includes `frames` and `kNN`), or do it by hand
  from the printed `table test:` lines (`grep "table test:" runs/*/*/out_0.log`).

For 3 seeds of a model: launch the run, then fresh-trial warm-start it twice more (same
`exp_name`/`run_name`, `warm_start_load=false`). For the heavy `CGENNLGATrGraphGPS`
(~4.5e11 FLOPs/jet, ~a day per trial on an H100) budget accordingly; the slim model is
~300× lighter.

---

## 9. Tests

```bash
pytest tests/experiments/test_tag_equivariance.py -q   # invariance (32 cases)
pytest tests/experiments/test_tag_flops.py -q -s       # FLOPs + param counts
```

`test_tag_equivariance.py` asserts three properties on the `config_quick` models:
azimuthal invariance for every hybrid (Minkowski kNN), full SO(3)/Lorentz
invariance for the internally-equivariant ones (spurions off, fully connected,
float64), and LLoCa-frame invariance for the canonicalized ones under both learned
frames (`learnedpd` and `learnedso13`; `learnedpd` carries a looser float64 bound for
its polar-decomposition boost-precision floor). The unit-test workflow
(`.github/workflows/tests.yaml`) now runs the equivariance + invariance suites, but it
only triggers on the `ready for review` label — so still run them locally as your gate.
(`test_tag_flops.py` stays out of CI: its learned-frame cases need a CUDA-matched
xformers build.)

---

## 10. Gotchas

- **The default training recipe** (`tag_gts_and_friends_default`, AdamW) now
  fits the new models — just set an LR/batchsize from `find_lr.py` (§5/§6). Baselines
  still need their own `training=top_<baseline>` (e.g. Lion for the transformer).
- **`use_float64`** is `false` in production (float32); the equivariance tests flip
  it on for the exact-invariance checks. The kNN distance computations follow the
  run dtype.
- **kNN graphs are slightly discontinuous** (a transform can flip a near-tied
  neighbour), so as-configured models are azimuthally invariant only to ~1e-3; this
  is inherent to every kNN GNN and vanishes with learned frames or a fully connected
  graph. It does not affect training.
- **`norm: batch` vs `layer`** on the non-equivariant GPS models: `batch` is the
  GraphGPS default; `layer` is the padding-safe alternative for variable jet sizes.
  The equivariant GPS models use the geometry-native norm (EquiLayerNorm / RMSNorm)
  and cannot use BatchNorm on their vector/multivector streams.
