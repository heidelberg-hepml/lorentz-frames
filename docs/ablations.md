# Minor ablations not done

A list of minor ablations not done that can potentially minorly improve performance but are
unlikely to and not the focus of this research. Everything here is conceptually sound (nothing
that breaks equivariance or the model's design contract); most are existing config toggles,
the rest are one-line hooks. The *headline* ablations (kNN metric/count, LLoCa frames on/off,
ParT pairwise-bias features, PE/SE, depth) live in `todo.md` §3 and are not repeated here.

## Bridges and readout tokens (GraphTrans family)

- CGENN GraphTrans bridge: gate the linear bridge (`MVLinear`/`Linear`) with an equivariant
  nonlinearity (MVSiLU / gated EquiLinear) instead of a plain linear map; or deepen it to a
  2-layer bridge MLP.
- Add an equivariant norm (EquiLayerNorm) after the CGENN→L-GATr bridge (the L-GATr blocks are
  pre-norm, so this is redundant in principle — hence minor).
- Plain/PNP GraphTrans: drop or move `bridge_norm` (LayerNorm after the bridge; also redundant
  under pre-norm blocks).
- Class token init: zeros vs `trunc_normal(0.02)` vs plain `normal(0.02)` — the four GraphTrans
  hybrids currently mix these conventions.
- Class token: learnable scalar (current) vs the pure-L-GATr convention of a **fixed** one-hot
  indicator scalar with zero multivector.
- CLS placement: prepended through all blocks (current, L-GATr convention) vs ParT-style
  class-attention blocks only at the end.
- Readout: concat CLS + masked mean-pool (instead of CLS only) for the GraphTrans models.
- CGENN GraphTrans readout: extract full invariants (grade norms, deliberately-added parity-odd
  pseudoscalar) from the CLS multivector instead of `extract_scalar`'s grade-0 only.
- LorentzNet GraphGPS: re-add the pooled per-node ‖v‖² readout (the leaner scalar-only readout
  matches pure LorentzNet; this is the deliberate reversal).
- CGENN GraphGPS: grade-0-only readout (the pre-audit lean variant) vs the current full
  `get_invariants`.

## Residual / norm / gate micro-choices

- `cgenn_residual` true/false in the CGENN GraphTrans stage (reference `tag_cgenn` runs without
  residuals; the toggle exists).
- `cgenn_normalization_init` null vs 0 (NormalizationLayer off/on inside the geometric-product
  layers; toggle exists, reference runs without).
- `use_phi_m` off (LorentzNet per-edge sigmoid gate; toggle exists — off leaves soft attention
  entirely to the transformer, the "GraphGPS division of labour").
- `use_node_attr` off (LorentzNet hybrids; toggle exists — off is the pre-audit underfed variant,
  useful to quantify what the per-layer raw-scalar re-injection buys).
- A `drop_local` hook on the internal-residual GPS local branches (LorentzNet, ParticleNet) so all
  four local branches see dropout if dropout is ever enabled (currently only Plain/CGENN GPS do).
- `norm: layer` instead of `batch` on the non-equivariant GPS models (padding-safe alternative;
  toggle exists).
- Post-norm instead of pre-norm in the Plain transformer blocks.
- Re-zero padded slots between LorentzNet-GPS layers (cosmetic; only BN running stats see them).

## Graph construction and aggregation

- Static vs dynamic (per-layer feature-space) kNN for the Plain models — Plain is static,
  ParticleNet-ParT is dynamic; swapping either isolates the graph-rebuilding choice.
- Symmetrized (undirected) vs directed receiver-based kNN edges in the CGENN edge builder.
- Deliberate self-loops (some GNN recipes include them; the audit removed the *accidental* ones).
- Growing-k schedule across GNN layers (DGCNN-style) instead of a fixed k.
- Aggregation: `cgenn_aggregation` sum vs mean (toggle exists); Plain MPNN mean → sum or max;
  EdgeConv mean → max (original DGCNN uses max; ParticleNet chose mean).
- `use_explicit_edge_features` off for CGENN GraphGPS (toggle exists; quantifies the edge/node
  re-injection).
- Richer Minkowski edge features for Plain-GPS `use_edge_attr` (full ParT 4/7-feature set through
  the MPNN edge channel instead of only log|(pᵢ+pⱼ)²|).
- LorentzNet `c_weight` sweep (1e-3 / 5e-3 / 1e-2).

## Capacity and shape (kept per-reference in the study; unify-or-sweep as ablations)

- FFN ratio 2 vs 4 across the Trans/GPS pairs (currently per-reference: ParT 4, GraphGPS 2).
- GELU vs ReLU unification across the non-equivariant pair (currently per-reference).
- Dropout 0.1 vs 0.0 family-wide (per-reference now; the most likely of this list to actually
  matter at a fixed 20-epoch budget).
- GPS attention dropout 0.5 (GraphGPS uses it on some datasets) vs the current 0.
- `num_heads` 4/8/16; `head_scale` off; `multi_query` on (L-GATr attention).
- `head_layers` 1/2/3 for the SAN-style GPS heads; unify their GELU (equivariant) vs ReLU
  (non-equivariant) activation.
- GNN:transformer depth ratio at fixed total (2:10 / 3:10 / 4:8) and blocks 8/10/12.
- LorentzNet-GPS shared width: towards-GNN midpoint (~84 s / 24 v) vs the current
  towards-transformer 96/32 (the config notes the alternative).
- `attn_reps` composition for the LLoCa transport (e.g. `12x0n+1x1n`, `4x0n+3x1n`, a `1x2n`
  tensor channel) at fixed embed_dim; same for the EdgeConv/MPNN `hidden_reps_list` split.
- `increase_hidden_channels_attention/_mlp` 2 vs 4 (lgatr's own default is 4 for the MLP).
- LorentzNet hybrid widths: `n_v_hidden` 8/16/32, `n_h_hidden` 72 vs 96.
- `concat_original` / `use_input_concat` off (raw-input skip at the bridge; toggles exist).
- `use_fusion` off for ParticleNet-ParT GraphTrans (toggle exists; the baseline row runs true).
- `use_fts_bn` off (input BatchNorm on the non-equivariant models).
- `add_fourmomenta_backbone` on (feed local four-momenta as extra scalar channels; wrapper toggle
  exists — off is the reference convention).
- `use_pre_activation_pair` false for the PNP hybrids, aligning with the repo's `tag_ParT` row
  (currently true = weaver default; the two differ in whether the pair bias passes a final GELU).
- `remove_self_pair` true in the pair embedding.

## Symmetry-breaking inputs

- Spurion variants on the equivariant hybrids: `beam_mirror` off, `spacelike`/`timelike` beam
  forms, single vs two beams, `spurion_scale` ≠ 1 (model-level analogues of the data-level knobs).
- LorentzNet hybrids: `use_time_spurion` / `use_beam_spurion` individually off.
- `data.tagging_features` zinvariant/so3invariant/null rows for the equivariant hybrids (note the
  non-equivariant four silently keep "all" — the documented D1 caveat).

## Training-side minor tunes

- EMA of weights for eval (`ema=true`, decay 0.999) — classic small, free gain; note the
  best-checkpoint reload currently keeps the end-of-training EMA shadow (see todo).
- `weight_decay` {0, 0.05, 0.1} beyond the shared 0.01; AdamW betas/eps.
- Warmup fraction `warmup_pct_start` 0.01/0.1 and a small `cosanneal_eta_min` (1e-6) hedge.
- `epochs` 30/35 vs the shared 20 (`training.epochs=30` is a one-flag change).
- `best_model_metric: accuracy` vs `loss` for checkpoint selection (toggle exists).
- OneCycleLR vs CosineAnnealingWarmup (repo-proven alternative; cycles β₁ by default — minor
  confound noted in todo).
- Optimizer family swap for the hybrids (Lion / Ranger at rescaled lr·wd) vs the shared AdamW.
- Label smoothing on the BCE loss (not implemented; one-line).
- Gradient clip 0.5/5.0/off vs the standard 1.0.
- More than 3 fresh-trial seeds per row (tightens the error bars, changes no mean).

Deliberately excluded as *conceptually broken* (not "minor"): learnable **vector/multivector**
class tokens (pick a direction → break equivariance), BatchNorm over multivector components,
LapPE as an invariant PE under learned frames (sign/basis-ambiguous — kept only as the expected
negative result via `use_lappe`), and re-adding `add_tagging_features_framesnet` (upstream
residual-symmetry infrastructure, deliberately not resurrected).
