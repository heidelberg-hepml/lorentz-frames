# How this fork differs from upstream (lloca-experiments)

A quick aside, one line per difference. Deliberate faithful *quirks of official model
implementations* (CGENN's padded-batch-max readout, LorentzNet's dead LGEB dropout kwarg,
ParT's cls-block-only dropout zeros) are kept, commented in code, and not listed here.

## Added
- Models: the 8 GraphTrans/GraphGPS hybrids ({Plain, ParticleNet-ParT, CGENN–L-GATr,
  LorentzNet–L-GATr-slim} × {GraphTrans, GraphGPS}) with configs, quick-configs and tests.
- Tools: `find_lr.py` (LR range test + GPU batch-size finder), `aggregate_table.py`,
  `data/collect_data.py jetclass` (download+verify+extract).
- Recipes: shared family defaults (`tag_/jc_gts_and_friends_default`) + per-model
  `top_/jc_<hybrid>.yaml`; both task configs default to the family recipe (was `jc_ParT`-style).
- Trials workflow: fresh-trial warm starts (`warm_start_load=false`), per-run
  `table_metrics_*.json` accumulation, automatic `[N trials] mean ± std` table rows.
- Epoch budget: `training.epochs` → iterations derived at runtime (`_resolve_epoch_budget`);
  `CosineAnnealingWarmup` (linear warmup → cosine) scheduler option.
- Table row extended with model name, trials tag, train time, per-jet FLOPs, kNN metric.
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
- CI runs the tagging equivariance+invariance suites (upstream removed its broken test line
  without a replacement, leaving tagging uncovered).
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

## Conventions this fork sets (upstream has no stance)
- Hybrid-family fairness: shared AdamW/schedule/budget, per-model batchsize+lr from the LR
  finder; dropout kept per-reference (ParT-side blocks 0.1, GPS and L-GATr sides 0/none).
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
