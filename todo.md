# TODO — outstanding work

A running checklist for finishing the graph-transformer hybrid study and connecting it to
a paper. Grouped by "before training", "open design decisions", and "paper release".

---

## 1. Before training — fill in the training configs

The 8 hybrid recipes are skeletons with required `???` keys:
`config/training/top_{Plain,ParticleNetParT,CGENNLGATr,LorentzNetLGATrSlim}{GraphTrans,GraphGPS}.yaml`.

The 8 GT recipes now inherit `tag_gts_and_friends_default` (shared `epochs=20` + `scheduler=CosineAnnealingWarmup`);
only **`batchsize`, `lr`** remain `???` per model (optionally `weight_decay`):

- [ ] `batchsize` ← `find_lr.py +lr_find.find_batch_size=true` (largest power-of-two that fits the H100).
- [ ] `lr` ← `find_lr.py` (reported loss-min / 10).
      (`weight_decay` tuning moved to `docs/ablations.md` "Training-side minor tunes" —
      the shared 0.01 ships as the decided default.)
- [x] `epochs` (shared data-exposure budget) and `scheduler` are **decided** in `tag_gts_and_friends_default`
      (see §2); `iterations` is auto-derived at runtime (`_resolve_epoch_budget`).

## 2. Training-recipe decisions (fairness)

**Scheduler — DECIDED: `CosineAnnealingWarmup`** (set in `tag_gts_and_friends_default`), shared across the GT
hybrids, tuning only lr/batchsize/weight_decay per model — warmup matters for the transformer/
equivariant layers, and one shared schedule isolates architecture for the hybrid-vs-hybrid table.
`OneCycleLR` is the repo-proven alternative (it is warmup→cosine too) but its warmup is cosine-shaped
and it cycles AdamW's β₁ by default — minor confounds. The published **baselines** (ParT/ParticleNet/
L-GATr) keep their own recipes as reference rows (you can't out-tune the originals); optionally re-run
them under the shared schedule for one apples-to-apples row. Annealing to ~0 is desirable (the
end-of-training low-lr phase gives the best final val metric); per-module heterogeneity is handled by
warmup (peak) + AdamW + `lr_factor_framesnet`, not by raising the floor. Set a small
`cosanneal_eta_min` (e.g. 1e-6) only as a hedge against a slightly over-long schedule.

**Epochs vs iterations.** Now automated: set `training.epochs` and `iterations` is derived per model as
`epochs * len(train_loader)` (the exact batch count — reflects batchsize, subsampling, drop_last). This
equalizes **data exposure** (the standard fairness axis); note equal epochs ≠ equal gradient *updates*
(a larger-batch model gets fewer steps), and each model still anneals fully over its own iteration count.
- [x] Epoch budget **decided: `epochs=20`** (ParT-standard) in `tag_gts_and_friends_default`, shared by all 8 GT
      hybrids; bump to ~30 if they underfit (CLI: `training.epochs=30`).
- [ ] Keep the baselines' published-recipe numbers as a separate reference row in the table.
- [ ] **Paper — methods sentence for the budget.** Something like: *"All hybrid taggers are trained
      for an equal data exposure of 20 epochs on the top-tagging dataset (5 epochs on JetClass,
      the ParT-standard exposure). Because batch sizes are tuned per model, the iteration count is
      derived at run time as epochs × batches/epoch, and each model's warmup–cosine schedule
      anneals over its own iteration count. Equal epochs implies unequal optimizer-step counts
      across batch sizes; we follow the community convention of fixing data exposure (ParT: 20
      epochs; LorentzNet, PELICAN: 35). Baseline reference rows are trained under their published
      recipes."* Optionally cite the recipe drift this replaces (published iteration counts
      correspond to 20.5 / 21.3 / 32 epochs for ParT / L-GATr / the Lion transformer).

**Best-checkpoint metric.** `best_model_metric` (in `tag_default`): `loss` (default, lowest val loss) or
`accuracy` (highest val accuracy). Selection-by-loss and -by-accuracy usually track but can diverge late;
the toggle only changes which checkpoint `es_load_best_model` keeps/reports.

## 3. Ablations — CLI recipes (for the paper's ablation tables)

All via Hydra overrides on `run.py` (use `-cp config` for the full configs). Every override is
recorded per-run in `config.yaml` + the flattened MLflow params, so any sweep is reconstructable
from the run dir. **Surfaced in the results table** (`aggregate_table.py` `COLUMNS`): only `frames`
(framesnet) and `kNN` (`knn_metric`); everything else (knn_k, num_layers/num_blocks, bias,
pair_input_dim, use_rwse, use_edge_attr, …) lives only in config.yaml / MLflow. Rows are grouped into ONE
table per task (toptagging / toptagxl / jctagging — different metric columns; `exp_type` read
from each run's config.yaml; JetClass emits an aggregator-compatible row too). To put a knob in
the head-to-head table, add it to `aggregate_table.py`'s per-task `COLUMNS` legend **and** the
per-run `table …:` log line that the regex reads.

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
  `model.net.pair_input_dim=1|4|5|8` selects how many QCD interaction features (1=lnΔ; 4=+ln kT,
  ln z, ln m²; 5=+lnΔs²; 8=+cosθ,Δy,Δφ — see `pairwise_lv_fts`; the weaver feature ladder jumps
  5→8 when adding cosθ/Δy/Δφ, so 6/7 are not valid — `assert len(outputs)==num_outputs` enforces
  this). The learned weights compensate, so the bias stays compatible with the frame transport.
- **GraphGPS PE/SE (Plain GraphGPS).** ParT-pair edge features `model.net.use_edge_attr=true|false`
  (lnΔ, ln kT, ln z, ln m² routed through the MPNN edge channel, ParticleNeXt-style); structural encoding `model.net.use_rwse=true|false`
  (+`model.net.rwse_k=K`); norm `model.net.norm=batch|layer`. CGENN GraphGPS relative edge features:
  `model.net.use_explicit_edge_features=true|false`.
  - [ ] Add general RWSE support to all GPS models (currently PlainGraphGPS-only) if it shows
        success on Plain. Note on `rwse_k`: GraphGPS uses k=20 on sparse molecular graphs
        (ZINC, ~23 nodes, diameter ~10); jet kNN is dense and small-diameter so the walk mixes
        fast and ~4–8 steps likely capture the useful return-probability structure (higher k adds
        near-saturated, redundant dims) — sweep `{4,8,16}` on Plain before generalizing.
- **Depth (transformer / GPS blocks).** `model.net.num_layers=N` (Plain, ParticleNet-ParT) /
  `model.net.num_blocks=N` (CGENN, LorentzNet). The depth curve is the "can the transformer
  compensate for a weaker GNN" story → a performance/efficiency section (room to discuss BigBird /
  sparse attention and the flex / xformers / flash backends the L-GATr stack already supports).

Other knobs worth a sweep: width/capacity (`hidden_*_channels`, `dim`, `gnn_dims`, `embed_dim`);
input-skip (`model.net.use_input_concat`); residual-symmetry spurions on the equivariant models
(`model.net.beam_spurion`, `model.net.add_time_spurion`); dropout — incl. the uniform
`model.net.attn_dropout` knob on all four GPS models (attention-weights dropout via sdpa;
GraphGPS ships 0.5, jet lineage ships none — a one-override family row on top-tagging). Depth and width move the param
count (a table column) — pair them with FLOPs/time for a fair efficiency plot.

**Off by design: global spectral PE/SE (LapPE / SignNet / eigenvalue SE).** A LapPE node-encoder
now EXISTS behind `use_lappe` (+`lappe_k`) on PlainGraphGPS — implemented exactly as the
demonstrate-the-negative-result toggle described below — but it ships OFF, and the default-off
choice is deliberate — worth a sentence in the paper. (1) A jet has *no canonical graph*:
the kNN adjacency/Laplacian is something we construct (eta–phi or minkowski kNN, and rebuilt in
feature space for EdgeConv), so anything read off its spectrum encodes our graph-building choice, not
the physics. (2) A jet is *not position-blind*: particles carry (Δη, Δφ, pT, E) — a physically
meaningful absolute PE that LapPE would only try to reconstruct (this is exactly why ParT ships no
positional encoding). (3) Under LLoCa a PE must be a *Lorentz invariant* to preserve invariance, but
LapPE eigenvectors have sign/basis ambiguity **and** graph dependence, so they are not clean
invariants (SignNet exists only to patch the sign ambiguity). The encodings that transfer are the
relative QCD pairwise features (lnΔ, ln kT, ln z, ln m²) and, on a *static* graph, RWSE — both already
exposed above. If a reviewer wants the negative result demonstrated, run PlainGraphGPS with
`model.net.use_lappe=true` (the toggle is implemented; sign-flip augmentation handles the
eigenvector ambiguity) — expected to show it doesn't help, per the argument above.

## 3b. Paper points (observations from the build/audit)

- GPS attention *rescues* a symmetry-breaking local graph: spurions off (float64, Lorentz),
  deltaR-vs-minkowski kNN breaks invariance ~14× more in GraphTrans (3.9e-3) than GraphGPS
  (2.8e-4; minkowski ≈ machine-exact both) — the invariant global L-GATr branch runs parallel
  in GPS and absorbs a non-invariant metric.
- ParticleNeXt edge features vs ParT pairwise bias = same encoding at two sites; in GPS one is
  likely redundant, and which one localizes the pairwise signal (per Spinner). See ablations.md.
- Directed kNN for message passing, symmetrized graph for RWSE/LapPE — the GraphGPS/MNIST
  convention (parity point for the "faithful port" claim).
- RWSE k for jets is short (~8) vs GraphGPS's 20 — dense small-diameter jet kNN mixes fast.

## 4. Open design decisions / discrepancies

- [x] **CGENN-LGATr GraphGPS local branch under-fed vs its GraphTrans cousin** — fixed in two
      passes: it now injects, under `use_explicit_edge_features` (default on), all three static
      signals the GraphTrans CGENN stage does — the relative-momentum edge multivectors
      `[pᵢ−pⱼ, rawᵢ, rawⱼ]`, **and** the raw mv / raw scalar inputs re-injected as per-node
      attributes (`node_attr_x` in `theta_x`, `node_attr_h` in `theta_h`) every layer. (The first
      pass added only the edge features; the node attributes were the missing two-thirds.)
      Equivariance 3/3 (xy-rotation + full-group rotation + Lorentz boost).
- [x] CLS readout frame: **jet frame** (covariant, boost into the jet rest frame). Decided.
- [x] LLoCa transport made **strictly additive** (identity frames bit-identical to the plain backbone).
- [x] Scheduler: shared **CosineAnnealingWarmup** available; **early termination off** (`es_patience=null`),
      best-validation checkpoint still reported.

### Resolved audit trail -> docs/audit-ledger.md

The detailed audit findings formerly here (boost_jet ordering + rotation-frame decision with
its measurements; the property-based sweep incl. the BatchNorm-over-padding faithfulness
verdict; the GraphTrans-vs-GPS dropout ledger; the JetClass/plots/trials infrastructure sweep;
the jet_frames pre-publication sweep) moved to `docs/audit-ledger.md` -- they are resolved or
decided, and several back the paper's fidelity claims. Still OPEN from those sweeps:

- [ ] `torch.cuda.amp.autocast(...)` deprecation in 4 baseline files (mechanical migration).
- [ ] ParT-GPS float attn_mask + bool key_padding_mask deprecation (merge masks before torch
      makes it fatal).
- [ ] xformers pin note for the SLURM target (docs/SLURM.md install step).
- [ ] learnedpd boost-precision-floor methods sentence for the paper.
- [ ] LorentzNet GPS: zero padded slots between layers for exact BN-running-stat parity
      (cosmetic; logits unaffected).
- [ ] LorentzNetKNNBlock phi_e BN normalises invalid edges (pre-existing, both variants).
- [ ] JetClass: fill the 8 `jc_<Hybrid>.yaml` `???` batchsize/lr from find_lr on jctagging.
- [ ] Rejection-metric convention differs top vs JetClass (one methods sentence, or unify).

## 5. Paper release — branding / identity (only the maintainer has these)

Critical (still point at the upstream LLoCa project):
- [ ] `README.md` — title ("Lorentz Local Canonicalization"), arXiv badges (2505.20280 / 2508.14898),
      author list + `heidelberg-hepml/*` links, the BibTeX block.
- [ ] `reproduce.md` — clone URL `heidelberg-hepml/lloca-experiments` + `cd lloca-experiments`,
      upstream arXiv references; **replace the manual JetClass-download line with
      `python data/collect_data.py jetclass`** (now automated).
- [ ] `REPRODUCE.md` — stale xformers claim: says running LLoCa/L-GATr taggers without xformers
      "requires modifying the data embedding and attention mask construction". No longer true —
      `model.attention_backend=flash|flex` does it as a config override (GUIDE §7, docs/OSCAR.md §2
      note). Rewrite the paragraph; upstream PR comment about it planned separately.
- [ ] `LICENSE` — copyright currently lists the upstream LLoCa authors; add your authors / mark derivative.
- [ ] **Humanize the prose** in the assistant-drafted texts/files before publication — `GUIDE.md`,
      `docs/{OSCAR,SLURM,ablations,diffs}.md`, this todo, the longer code comments: pass for
      personal voice, trim the em-dash-heavy style, keep the technical content.

Minor (stale strings / metadata):
- [ ] `pyproject.toml` — add an `authors` field (name is already `gtagger-experiments`).
- [ ] add a `CITATION.cff` for the new paper.
- [ ] `experiments/base_experiment.py:299` — `path_code = os.path.join(self.cfg.base_dir, "lloca")`
      hardcodes "lloca" for the saved-source dir → project name.
- [ ] `docs/OSCAR.md` §2 — fix the `.sif` discrepancy once CCV remedies it: the
      `ngc-pytorch-container/25.08-py3-ayk4` module `setenv`s the 24.03 sif; drop the resolve-the-real-25.08-sif workaround when the module is corrected.
- [ ] `docs/SLURM.md:79` — `#SBATCH --job-name=lloca`.
- [ ] `config/{toptagging,jctagging,ttbar}.yaml` + `config_quick/*` — debug `exp_name`s
      (`topt_local_debug`, `jc_debug`, `ttbar_debug`).
- [ ] `config/model/tag_CGENNLGATrGraphTrans.yaml` — incomplete `#should be` comment (cosmetic).
- [ ] `tests/helpers/equivariance.py:4` — upstream attribution comment; fine to keep as a credit.
- [x] **stale `tagging_features_framesnet` overrides** — the data key was renamed to
      `tagging_features`, but the old name lingered in run-command examples that would error under
      Hydra (struct mode rejects unknown-key overrides): `REPRODUCE.md` lines 227–229 & 234–235
      (×5) and `.github/workflows/experiments_tagging.yaml:39` (×1). Renamed each to
      `data.tagging_features=…` (the workflow one would otherwise have failed the tagging CI job).
- [ ] **Defork** the GitHub repo when publishing (a fork is hidden from search / awkward to Zenodo-archive);
      keep the upstream attribution in README + LICENSE.

## 6. Done (for reference)

- 2×2×2 hybrid family ({Plain, ParticleNet-ParT, CGENN-LGATr, LorentzNet-LGATr-slim} × {GraphTrans, GraphGPS}).
- Faithful LLoCa tensorial message-passing for the ParticleNet-ParT **and Plain** hybrids (MPNN/EdgeConv
  `change_local_frame` + `LLoCaAttention`), **additive** (identity frames bit-identical: 0 added params),
  jet-frame class token (GraphTrans) / invariant mean-pool (GraphGPS), rapidity clamp.
- Equivariance suite (24/24, incl. full Lorentz boost under learned `so(1,3)` frames).
- `find_lr.py` batch-size finder; `aggregate_table.py`; `data/collect_data.py jetclass`; `GUIDE.md`; `docs/SLURM.md`.
