# How this fork differs from upstream (lloca-experiments)

A quick aside, one line per difference. Deliberate faithful *quirks of official model
implementations* (CGENN's padded-batch-max readout, LorentzNet's dead LGEB dropout kwarg,
ParT's cls-block-only dropout zeros) are kept, commented in code, and not listed here.

## Added
- Models: the 8 GraphTrans/GraphGPS hybrids ({Plain, ParticleNet-ParT, CGENN–L-GATr,
  LorentzNet–L-GATr-slim} × {GraphTrans, GraphGPS}) with configs, quick-configs and tests.
- Tools: `find_lr.py` (LR range test + GPU batch-size finder), `aggregate_table.py`,
  `data/collect_data.py {jetclass, toptagxl}` (download + md5-verify + extract).
- Recipes: shared family defaults (`tag_/jc_gts_and_friends_default`) + per-model
  `top_/jc_<hybrid>.yaml`; both task configs default to the family recipe (was `jc_ParT`-style).
- Trials workflow: fresh-trial warm starts (`warm_start_load=false`), per-run
  `table_metrics_*.json` accumulation, automatic `[N trials] mean ± std` table rows.
- Epoch budget: `training.epochs` → iterations derived at runtime (`_resolve_epoch_budget`);
  `CosineAnnealingWarmup` (linear warmup → cosine) scheduler option.
- Table row extended with model name, trials tag, train time, per-jet FLOPs, kNN metric;
  `aggregate_table.py` emits one table per task (top/XL/JetClass columns differ) and
  JetClass emits an aggregator-compatible row.
- Guardrail warnings: hybrid training at the unswept 512/1e-3 fallback; `seed` set together
  with a fresh trial; end-of-training loss-vs-accuracy checkpoint-selection cross-check.
- Docs: `GUIDE.md`, `docs/{SLURM,OSCAR,ablations,diffs}.md`, `todo.md` ledger.

## Changed
- `es_patience` 100 → `null`: no early termination (train the full budget; the best-validation
  checkpoint is still saved/restored/reported) — a significant plateau-allowance shift.
- Best-checkpoint restore re-pairs the EMA shadow with the restored weights (also submitted
  upstream); `best_model_metric` toggle (loss/accuracy) added for the selection metric.
- ParT's weight-decay grouping extended from the hardcoded `{"cls_token"}` to
  `net.no_weight_decay()`, covering the CLS-token hybrids.
- Packaging: project renamed `gtagger-experiments` in `pyproject.toml`.
- Deps: `lloca[xformers-attention]>=1.3.6` (`jet_frames` uses `fn.mass_regularize`, first in 1.3.6).
- CI runs the tagging equivariance+invariance suites (upstream removed its broken test line
  without a replacement, leaving tagging uncovered). The expected residual symmetry group
  (SO(2)-about-beam vs full Lorentz) is now derived from each model's spurion keys
  (`residual_symmetry_group`), keyed off the unified `beam_spurion`/`add_time_spurion` names,
  so the asserted group tracks the config instead of a hand-maintained per-model list.
- `save` defaults true in `config/` (best-val weights kept as `model_run{idx}.pt`);
  `validate_every_n_epochs_min` sentinel for once-per-epoch validation.
- DDP additionally wraps the framesnet (upstream plans a different DDP rework; multi-GPU
  remains not-recommended by upstream).
- Debug-friendly extras: `warm_start_load` non-sticky in saved configs, selection-history
  logging, extended run-summary logging.

## Fixed here first, since adopted upstream (#92 etc.)
- CGENN wrapper dense-frame edge construction; MIParT rapidity clamp; score.pdf storing
  sigmoid probabilities; `tag_lorentznet` `n_scalar` wiring; dataset mask `.all→.any`;
  `torch-geometric>=2.6` pin.
  (The `jc_lgatr` `tag_gatr→top_lgatr` recipe-base rename was fixed on `main` directly,
  NOT here — it arrives via the merge, so it is not a fork-first fix.)

## Fixed here, offered upstream (post-audit round)
- `for_inference` single-logit heads use sigmoid (softmax over a 1-wide dim is constant 1.0);
  guard/softmax on dim 1 (segmentation-safe). `pairwise_lv_fts` clamps delta_r2 BEFORE the
  sqrt (sqrt(0) backward is NaN -> poisoned learned-frames grads on bit-identical pairs).
  `boost_jet` forced off for pure-rotation frames (LearnedSO3/SO2; measured set).
- Embedding order: tagging features (deta/dphi/dr, log pt) are computed BEFORE the optional
  jet-rest-frame boost. Post-boost the jet has pt~0, so pt_jet clamps and eta_jet/phi_jet read
  off a numerically-zero vector -> dphi/deta rotation-unstable, breaking even SO(2) invariance
  (xyrotation max MSE O(1e2..1e4) -> O(1e-8)). The mass regulator also excludes spurions
  (`& ~is_spurion`), so a spacelike beam spurion (m^2=-1) is not forced lightlike (which would
  void the spacelike-beam ablation).

## Fixed here (input pipeline & training robustness)
- CGENN `MVLayerNorm` gain reshaped (1, C) -> (C,) so it falls under the optimizer's ndim<=1
  weight-decay exemption like every other norm gain; the official 2-d shape was silently
  weight-decayed -- an undocumented regularization asymmetry hitting only the CGENN hybrids
  (broadcasting unchanged: unsqueezed to (1, C) at use).
- Streaming loaders: `infinity_mode` (file re-cycling) is TRAIN-only -- on val/test it looped
  forever; `steps_per_epoch` bounds the train epoch (JetClass + TopTagXL).
- Finetuning: `ema_decay` read from the config top level (`cfg.ema_decay`), not
  `cfg.training.ema_decay` (the old path crashed in struct mode); a fresh-trial warm start
  (`warm_start_load=false`) re-loads the pretrained backbone for an independent finetune trial.
- miniweaver hardening: a corrupt/truncated ROOT read logs file+traceback instead of a silent
  `None` (md5 covers only the tar download, not a scratch copy, so silent skips skewed class
  balance across epochs); a glob matching no files warns instead of silently shrinking the
  requested file range; the "Nothing to load for worker N" error is parenthesized so it
  actually reports the worker id.

## Conventions this fork sets (upstream has no stance)
- Hybrid-family fairness: shared AdamW/schedule/budget, per-model batchsize+lr from the LR
  finder; dropout kept per-reference (ParT-side blocks 0.1, GPS and L-GATr sides 0/none);
  `use_pre_activation_pair: false` on the PNParT hybrids (published-ParT parity); a uniform
  `attn_dropout: 0.0` knob on all four GPS models (sdpa dropout_p; equivariance-safe);
  Plain-GPS `use_edge_attr` = the ParT pair features MPNN-routed (ParticleNeXt-style
  routing ablation), OFF by default so Plain stays the bare backbone.
- JetClass recipes: `weight_decay: 0`, `epochs: 5` (ParT-standard exposure), per-model
  re-sweep of batchsize/lr on the jctagging task.

## Disclosures for the methods section (per-reference choices, not bugs)
- **Head depth is per-reference, not unified.** The four GraphTrans hybrids classify with a
  single Linear from the CLS token (the official GraphTrans head); the four GraphGPS hybrids
  use a 2-layer SAN-style MLP after mean-pool (the official GraphGPS `SANGraphHead`). Both are
  faithful to their lineage, so head capacity co-varies with the GT-vs-GPS axis by design.
- **`tag_particlenet` runs `use_fusion: true`** (weaver's default + what the hybrids use),
  ~172k params above the LLoCa-paper ParticleNet baseline row. Deliberate; note it when
  comparing to published ParticleNet numbers.
- **The four non-equivariant hybrids hardcode `tagging_features="all"`** in `TaggerWrapper`,
  so the `data.tagging_features` ablation moves only the equivariant rows (headline table
  unaffected).
- **`deta` uses an unconditional sign flip** (`-(eta_i - eta_jet)`), not weaver's
  hemisphere-dependent flip -- internally consistent, but the input pipeline is not
  weaver-verbatim on this one feature.
