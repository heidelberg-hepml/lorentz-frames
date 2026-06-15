# TODO — outstanding work

A running checklist for finishing the graph-transformer hybrid study and connecting it to
a paper. Grouped by "before training", "open design decisions", and "paper release".

---

## 1. Before training — fill in the training configs

The 8 hybrid recipes are skeletons with required `???` keys:
`config/training/top_{Plain,ParticleNetParT,CGENNLGATr,LorentzNetLGATrSlim}{GraphTrans,GraphGPS}.yaml`.

For each model, fill `iterations`, `batchsize`, `lr` (optionally `weight_decay`, `optimizer`,
`scheduler`):

- [ ] `batchsize` ← `find_lr.py +lr_find.find_batch_size=true` (largest power-of-two that fits the H100).
- [ ] `lr` ← `find_lr.py` (reported loss-min / 10).
- [ ] `iterations` ← `epochs * ceil(num_train_jets / batchsize)` (see §2: pick one epoch budget for all).
- [ ] `weight_decay` ← tune on val ∈ {0, 0.01, 0.05, 0.1} for AdamW (ParT-style 0.01 is a fine start).
- [ ] Decide the shared `scheduler` (see §2) and set it in `tag_default.yaml` (or per-recipe).

## 2. Training-recipe decisions (fairness)

**Scheduler.** Recommend a single shared **cosine-annealing-with-warmup** schedule for the whole
comparison set (hybrids + baselines), tuning only lr/batchsize/weight_decay per model — so the
comparison isolates the architecture, not the recipe. Warmup matters for the transformer-heavy
hybrids. The repo's `CosineAnnealingLR` has no warmup; options:
- [ ] use `OneCycleLR` (`onecycle_pct_start` ≈ 0.05–0.10 gives a short warmup + cosine decay), **or**
- [ ] add a dedicated `LinearWarmup → CosineAnnealingLR` scheduler to `base_experiment._init_scheduler`
      (cleaner "cosine + warmup"; ~5–10% warmup). *(I can add this on request.)*

**Epochs vs iterations.** Everything is configured in *iterations* (`T_max = iterations * scheduler_scale`),
but ParT/ParticleNet calibrate that to ~20 epochs while L-GATr uses a fixed 200k-iter budget. For a
fair comparison fix one **epoch budget** (data exposure) for all models and derive
`iterations = epochs * ceil(num_train_jets / batchsize)` per model; rely on early stopping
(`es_patience`) to stop converged models early. This guarantees the hybrids see the full dataset the
same number of times as the baselines.
- [ ] Pick the epoch budget (e.g. ParT-standard ~20–30 for top-tagging; raise if the hybrids underfit).
- [ ] Re-express the baseline recipes (`top_ParT`, `top_particlenet`, `top_lgatr`) in the same epoch
      budget for the head-to-head table (keep the published-recipe numbers as a separate reference row).

## 3. Ablations — CLI recipes (for the paper's ablation tables)

All via Hydra overrides on `run.py` (use `-cp config` for the full configs). Every override is
recorded per-run in `config.yaml` + the flattened MLflow params, so any sweep is reconstructable
from the run dir. **Surfaced in the results table** (`aggregate_table.py` `COLUMNS`): only `frames`
(framesnet) and `kNN` (`knn_metric`); everything else (knn_k, num_layers/num_blocks, bias,
pair_input_dim, use_rwse, use_edge_attr, …) lives only in config.yaml / MLflow. To put a knob in the
head-to-head table, add it to `aggregate_table.py`'s `COLUMNS` string **and** the per-run `table …:`
log line that the regex reads.

- **kNN graph (all networks).** count `model.net.knn_k=K` (CGENN uses `model.net.k=K`); metric
  `model.net.knn_metric=deltaR|minkowski`; fully-connected = k ≥ P−1 (`9999`, or `model.net.k=null`
  for CGENN). minkowski is the Lorentz-invariant graph (needed for full-group invariance); deltaR is
  the eta–phi graph.
- **LLoCa on/off (the non-equivariant backbones: Plain, ParticleNet-ParT).** on =
  `model/framesnet=learnedso13` (learned SO(1,3) frames → tensorial transport engaged); off / "do
  nothing" = `model/framesnet=identity` (no-op, bit-identical plain backbone). Symmetry-budget
  variants: `learnedso3` (rotations), `learnedso2`, `learnedz`, `learnedrest`, `learnedpd`;
  `randomlorentz` is the data-augmentation baseline. (CGENN / LorentzNet are already internally
  equivariant → leave on `identity`.)
- **ParT pairwise bias (ParticleNet-ParT GraphTrans + GraphGPS).** `model.net.bias=true|false`;
  `model.net.pair_input_dim=1|4|5|7` selects how many QCD interaction features (1=lnΔ; 4=+ln kT,
  ln z, ln m²; 5=+lnΔs²; 7=+cosθ,Δy,Δφ — see `pairwise_lv_fts`). The learned weights compensate, so
  the bias stays compatible with the frame transport.
- **GraphGPS PE/SE (Plain GraphGPS).** relative edge PE `model.net.use_edge_attr=true|false`
  (Minkowski log|(pᵢ+pⱼ)²|); structural encoding `model.net.use_rwse=true|false`
  (+`model.net.rwse_k=K`); norm `model.net.norm=batch|layer`. CGENN GraphGPS relative edge features:
  `model.net.use_explicit_edge_features=true|false`.
- **Depth (transformer / GPS blocks).** `model.net.num_layers=N` (Plain, ParticleNet-ParT) /
  `model.net.num_blocks=N` (CGENN, LorentzNet). The depth curve is the "can the transformer
  compensate for a weaker GNN" story → a performance/efficiency section (room to discuss BigBird /
  sparse attention and the flex / xformers / flash backends the L-GATr stack already supports).

Other knobs worth a sweep: width/capacity (`hidden_*_channels`, `dim`, `gnn_dims`, `embed_dim`);
input-skip (`model.net.use_input_concat`); residual-symmetry spurions on the equivariant models
(`model.net.beam_spurion`, `model.net.add_time_spurion`); dropout. Depth and width move the param
count (a table column) — pair them with FLOPs/time for a fair efficiency plot.

## 4. Open design decisions / discrepancies

- [x] **CGENN-LGATr GraphGPS local branch had no edge features** — fixed: it now injects the same
      static relative-momentum edge multivectors `[pᵢ−pⱼ, rawᵢ, rawⱼ]` as the GraphTrans cousin
      (`use_explicit_edge_features`, default on). Equivariance 3/3.
- [x] CLS readout frame: **jet frame** (covariant, boost into the jet rest frame). Decided.
- [x] LLoCa transport made **strictly additive** (identity frames bit-identical to the plain backbone).
- [x] Scheduler: shared **CosineAnnealingWarmup** available; **early termination off** (`es_patience=null`),
      best-validation checkpoint still reported.

### Audit findings (full GraphTrans-vs-GraphGPS sweep) — remaining, low priority
- [ ] **Local-branch dropout is inconsistent across the GraphGPS family.** Plain + CGENN GPS apply an
      external `Dropout` to the local-MPNN output (`Norm(Dropout(MPNN(X)) + X)`); LorentzNet + ParticleNet
      GPS apply **none** (their GNN owns an internal residual, so the layer adds only the external Norm).
      The residual difference is *deliberate* (avoids a double residual), but the dropout is dropped as a
      side effect. No-op at the default `dropout_prob=0`, so it only matters if dropout is enabled — decide
      whether the four local branches should match.
- [ ] **LorentzNet GraphGPS never zeroes padded slots between its layers** (only at the final pool), so the
      shared `LorentzNetKNNBlock`'s BatchNorms accumulate over nonzero padded state across the 10 layers
      (GraphTrans zeroes after its GNN stack). Logits are unaffected (readout is masked) but BN running
      stats drift — cosmetic; zero padded slots per layer if you want exact parity.
- [ ] **(latent, both LorentzNet variants)** `phi_e` BatchNorm in `LorentzNetKNNBlock` normalises over
      *invalid* edges too — the edge mask is applied only *after* `phi_e`. Pre-existing, shared by both
      variants (not a Trans-vs-GPS divergence); mask before `phi_e` for cleanliness.
- [x] **"LorentzNet mean"** (scalar message aggregation was mean, should be sum) — already fixed in the
      shared block (`h_msg = m.sum(-1)`, commit `8a7b5fc`) and inherited by GraphGPS; both now match
      official LorentzNet (sum scalars / mean vectors).

## 5. Paper release — branding / identity (only the maintainer has these)

Critical (still point at the upstream LLoCa project):
- [ ] `README.md` — title ("Lorentz Local Canonicalization"), arXiv badges (2505.20280 / 2508.14898),
      author list + `heidelberg-hepml/*` links, the BibTeX block.
- [ ] `reproduce.md` — clone URL `heidelberg-hepml/lloca-experiments` + `cd lloca-experiments`,
      upstream arXiv references; **replace the manual JetClass-download line with
      `python data/collect_data.py jetclass`** (now automated).
- [ ] `LICENSE` — copyright currently lists the upstream LLoCa authors; add your authors / mark derivative.

Minor (stale strings / metadata):
- [ ] `pyproject.toml` — add an `authors` field (name is already `gtagger-experiments`).
- [ ] add a `CITATION.cff` for the new paper.
- [ ] `experiments/base_experiment.py:262` — `path_code = os.path.join(self.cfg.base_dir, "lloca")`
      hardcodes "lloca" for the saved-source dir → project name.
- [ ] `docs/SLURM.md:79` — `#SBATCH --job-name=lloca`.
- [ ] `config/{toptagging,jctagging,ttbar}.yaml` + `config_quick/*` — debug `exp_name`s
      (`topt_local_debug`, `jc_debug`, `ttbar_debug`).
- [ ] `config/model/tag_CGENNLGATrGraphTrans.yaml` — incomplete `#should be` comment (cosmetic).
- [ ] `tests/helpers/equivariance.py:4` — upstream attribution comment; fine to keep as a credit.
- [ ] **Defork** the GitHub repo when publishing (a fork is hidden from search / awkward to Zenodo-archive);
      keep the upstream attribution in README + LICENSE.

## 6. Done (for reference)

- 2×2×2 hybrid family ({Plain, ParticleNet-ParT, CGENN-LGATr, LorentzNet-LGATr-slim} × {GraphTrans, GraphGPS}).
- Faithful LLoCa tensorial message-passing for the ParticleNet-ParT **and Plain** hybrids (MPNN/EdgeConv
  `change_local_frame` + `LLoCaAttention`), **additive** (identity frames bit-identical: 0 added params),
  jet-frame class token (GraphTrans) / invariant mean-pool (GraphGPS), rapidity clamp.
- Equivariance suite (24/24, incl. full Lorentz boost under learned `so(1,3)` frames).
- `find_lr.py` batch-size finder; `aggregate_table.py`; `data/collect_data.py jetclass`; `GUIDE.md`; `docs/SLURM.md`.
