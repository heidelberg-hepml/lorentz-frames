# Minor ablations not done

A list of minor ablations not done that can potentially minorly improve performance but are
unlikely to and not the focus of this research. Everything here is conceptually sound (nothing
that breaks equivariance or the model's design contract); most are existing config toggles,
the rest are one-line hooks. The *headline* ablations (kNN metric/count, LLoCa frames on/off,
ParT pairwise-bias features, PE/SE, depth) live in `todo.md` §3 and are not repeated here.

## Bridges and readout tokens (GraphTrans family)

- CGENN GraphTrans bridge: gate the linear bridge (`MVLinear`/`Linear`) with an equivariant
  nonlinearity (MVSiLU / gated EquiLinear) instead of a plain linear map; CGENN phi_x 
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
- Retune the kNN `k`? The family ships small (`knn_k`/CGENN `k` at recipe defaults); a fair-k
  for jets is unsettled — jet kNN graphs are dense and small-diameter, unlike the sparse
  molecular graphs GNN-PE work is tuned on. Sweep `knn_k` (and CGENN `k`) jointly with the
  deltaR/minkowski metric, since the two interact (metric changes which neighbours k selects).
- Aggregation: `cgenn_aggregation` sum vs mean (toggle exists); Plain MPNN mean → sum or max;
  EdgeConv mean → max (original DGCNN uses max; ParticleNet chose mean).
- `use_explicit_edge_features` off for CGENN GraphGPS (toggle exists; quantifies the edge/node
  re-injection).
- DONE: Plain-GPS `use_edge_attr` now feeds the full ParT 4-feature pair set through the MPNN
  edge channel (ParticleNeXt-style; was a single log|(pᵢ+pⱼ)²| invariant). Remaining variant (through
  the MPNN edge channel instead of only log|(pᵢ+pⱼ)²|).
- **ParticleNeXt-style edge features vs ParT-style pairwise attention bias — two relative
  pairwise encodings, and stacking both.** `use_edge_attr` routes the pair features into the
  MPNN edge channel (ParticleNeXt); `bias`/`pair_input_dim` routes the same QCD invariants into
  the attention logits (ParT). They encode the same information at different sites, so in the GPS
  models (local MPNN ‖ global attention) one may be **redundant**. Per Jonas Spinner, both
  compensate for the non-equivariant backbone's lack of Lorentz invariance — so measuring *which*
  is redundant here isolates whether the MPNN or the attention is the load-bearing site for the
  pairwise signal, and could motivate future **non-hybrid** research on single-site pairwise
  routing (hybrids willing).
- LorentzNet `c_weight` sweep (1e-3 / 5e-3 / 1e-2).

## Capacity and shape (kept per-reference in the study; unify-or-sweep as ablations)

- FFN ratio 2 vs 4 across the Trans/GPS pairs (currently per-reference: ParT 4, GraphGPS 2).
- GELU vs ReLU unification across the non-equivariant pair (currently per-reference).
- Dropout 0.1 vs 0.0 family-wide (per-reference now; the most likely of this list to actually
  matter at a fixed 20-epoch budget).
- **Attention-weight dropout on the GPS models** (`attn_dropout=0.5` for the GraphGPS-native
  value, `0.1` as the softer point). Official GraphGPS uses 0.5 essentially everywhere at our
  model scale — small benchmarks (12k–45k graphs) AND the base PCQM4Mv2-full config
  (3.7M graphs, 5 layers, ~6M params); only the scaled-up entries (GPSmedium ~20M,
  GPSdeep 16 layers) trade attn_dropout down to 0.1 while *raising* residual dropout 0→0.1.
  So the driver is model capacity, not dataset size: at 1.2–2.5M params the official recipe
  says 0.5, making it the primary ablation value (PlainGraphGPS first, cheapest); the
  campaign default stays 0. Equivariance-safe everywhere: attention probabilities are
  Lorentz invariants. Non-equivariant GPS = config-only (keys exist), and the LLoCa
  transport path already forwards `dropout_p` to the native sdpa backend; the embedded
  L-GATr stacks in the hybrids also dispatch native (dense `attn_mask`), so they accept
  `dropout_p` on stock lgatr too. Only the *pure* LGATr/Transformer taggers default to the
  xformers backend, which needs a `dropout_p`→`p` rename upstream (flex/varlen have no
  dropout at all — small lgatr+lloca PRs, drafted in the campaign notes).
- `num_heads` 4/8/16; `head_scale` off; `multi_query` on (L-GATr attention).
- `head_layers` 1/2/3 for the SAN-style GPS heads; unify their GELU (equivariant) vs ReLU
  (non-equivariant) activation.
- GNN:transformer depth ratio at fixed total (2:10 / 3:10 / 4:8) and blocks 8/10/12.
- **Freeze the GNN stage (GraphTrans family).** Train with the GNN frozen at random init
  (or freeze a pretrained GNN and train only the transformer) to test whether the local
  message-passing does *learned* work or just supplies a fixed relative-structure feature the
  transformer reads off. Cheap, and it directly quantifies the GNN's marginal contribution —
  the counterpart to the depth-ratio sweep: depth asks "how much GNN", freezing asks "does the
  GNN need to be trained at all". A strong frozen-GNN result would sharpen the "can the
  transformer compensate for a weaker GNN" story toward "the transformer does the heavy lifting".
- LorentzNet-GPS shared width: towards-GNN midpoint (~84 s / 24 v) vs the current
  towards-transformer 96/32 (the config notes the alternative).
- `attn_reps` composition for the LLoCa transport (e.g. `12x0n+1x1n`, `4x0n+3x1n`, a `1x2n`
  tensor channel) at fixed embed_dim; same for the EdgeConv/MPNN `hidden_reps_list` split.
- `increase_hidden_channels_attention/_mlp` 2 vs 4 (lgatr's own default is 4 for the MLP).
- LorentzNet hybrid widths: `n_v_hidden` 8/16/32, `n_h_hidden` 72 vs 96.
- `concat_original` / `use_input_concat` off (raw-input skip at the bridge; toggles exist).
- **`use_fusion` (ParticleNet family) — unresolved consistency axis.** State across the family:
  weaver's ParticleNet defaults it ON; the `tag_particlenet` baseline row now runs it OFF
  (reverted to match upstream / the LLoCa-paper published ParticleNet so the baseline reproduces
  the literature number — ~172-205k params lighter than fusion-on); the ParticleNet-ParT
  **GraphTrans** hybrid's EdgeConv backbone runs it ON (`use_fusion: true`); the ParticleNet-ParT
  **GraphGPS** hybrid has no fusion at all (its multi-scale growth is the 10 interleaved GPS
  layers). So the setting is not unified and there is no single "right" value: fusion
  concatenates all EdgeConv block outputs, natural for the 3-block standalone ParticleNet but
  largely redundant with a 10-layer GPS stack. TO DECIDE before the paper: either (a) turn the
  GraphTrans hybrid's fusion OFF for cross-family consistency, or (b) keep it ON as the
  faithful-ParticleNet-backbone choice and add a fusion-ON `tag_particlenet` control row so the
  baseline still has a matched partner. The toggle exists on the baseline and the GraphTrans
  hybrid; the GPS hybrid has nothing to toggle.
- `use_fts_bn` off (input BatchNorm on the non-equivariant models).
- `add_fourmomenta_backbone` on (feed local four-momenta as extra scalar channels; wrapper toggle
  exists — off is the reference convention).
- `use_pre_activation_pair` false for the PNP hybrids, aligning with the repo's `tag_ParT` row
  (currently true = weaver default; the two differ in whether the pair bias passes a final GELU).
- `remove_self_pair` true in the pair embedding.

## GraphGPS recipe parity (cross-checked against the official rampasek/GraphGPS configs)

Where this repo's GPS family deviates from the recipe the official configs actually run, and
what the official repo's own precedent says about each axis. (attn_dropout, the largest
deviation, is covered in "Capacity and shape" above.)

- **PE/SE encodings.** Official GraphGPS node-encodes in essentially every config — including
  datasets with *no real edge features*: MalNet-Tiny ships LapPE by default (plus `+RWSE`,
  `+SignNet` variants and a `-noPE` ablation) with constant `DummyEdge` edge inputs. The
  crucial context: MalNet nodes are *anonymous* (function-call graphs; node features are
  synthesized via LocalDegreeProfile), so encodings are the only signal — whereas jet
  constituents arrive with rich per-node kinematics, which is why PE/SE stays off by default
  here (headline PE/SE ablation in todo §3: RWSE first, LapPE as the expected negative).
  If LapPE ever shows signal, **SignNet** (LapPE with a sign-invariant DeepSets/MLP encoder
  instead of training-time sign flips) is the canonical next step — not currently implemented.
- **Edge features.** Official GatedGCN *requires* edge inputs, so edge-free datasets get
  constant `DummyEdge` embeddings. This repo's local branches instead derive edge information
  from node differences (EdgeConv) or the physically-motivated Minkowski invariant
  (`use_edge_attr`) — a strict upgrade over constants, not a fidelity gap; worth one methods
  sentence.
- **Local module.** Official uses CustomGatedGCN in essentially every config; this repo swaps
  in EdgeConv / plain MPNN / CGENN / LGEB — deliberately, that *is* the research axis. A
  GatedGCN local arm on PlainGraphGPS would be the "faithful GPS" control if a reviewer asks
  what the unmodified recipe scores on jets.

## Symmetry-breaking inputs

- Spurion variants on the equivariant hybrids: `beam_mirror` off, `spacelike`/`timelike` beam
  forms, single vs two beams, `spurion_scale` ≠ 1 (model-level analogues of the data-level knobs).
- LorentzNet hybrids: `add_time_spurion` / `beam_spurion` individually off.
- `data.tagging_features` zinvariant/so3invariant/null rows for the equivariant hybrids. NB: the
  four non-equivariant hybrids (`TaggerWrapper` subclasses) hardcode `tagging_features="all"`
  internally, so this knob changes ONLY the equivariant rows; the non-equivariant headline rows
  are unaffected (see the disclosures list in `docs/diffs.md`).

## Training-side minor tunes

- EMA of weights for eval (`ema=true`, decay 0.999) — classic small, free gain (the
  best-checkpoint reload now correctly re-pairs the EMA shadow).
- `weight_decay` sweep {0, 0.01, 0.05, 0.1} on ONE mid-cost hybrid, freeze the winner
  family-wide (moved from todo §1; the shared 0.01 ships as the default until then);
  AdamW betas/eps.
- Warmup fraction `warmup_pct_start` 0.01/0.1 and a small `cosanneal_eta_min` (1e-6) hedge.
- `epochs` 30/35 vs the shared 20 (`training.epochs=30` is a one-flag change).
- `best_model_metric: accuracy` vs `loss` for checkpoint selection (toggle exists).
- OneCycleLR vs CosineAnnealingWarmup (repo-proven alternative; cycles β₁ by default — minor
  confound noted in todo).
- Optimizer family swap for the hybrids (Lion / Ranger at rescaled lr·wd) vs the shared AdamW.
- Label smoothing on the BCE loss (not implemented; one-line).
- Gradient clip 0.5/5.0/off vs the standard 1.0.
- More than 3 fresh-trial seeds per row (tightens the error bars, changes no mean).

## New tricks (modern transformer/CS innovations, none physics-motivated)

Bags-of-tricks from the post-ParT transformer literature. All are drop-in replacements
inside the non-equivariant transformer stages (Plain / ParticleNet-ParT); for the
equivariant stages each needs an equivariance check first (grade-wise application is
usually the fix, as lgatr already does for its own norm).

- **SwiGLU** FFN (gated `SiLU(xW₁)·xW₂` → W₃) instead of the plain 2-layer GELU/ReLU FFN
  — the LLaMA-era default, usually a small free gain at matched params (shrink the hidden
  ratio 4 → 8/3 to compensate the third matrix).
- **Gated attention**: element-wise output gate on the attention branch
  (`x + g(x) ⊙ Attn(x)`, sigmoid or SiLU gate) — stabilizes deep stacks and is the same
  mechanism `use_phi_m` already gives the LorentzNet stage; this would add it to the
  transformer side.
- **RMSNorm everywhere**, not just where the L-GATr-slim stack already uses it — swap the
  `nn.LayerNorm`s in the Plain/PNP transformer blocks and GPS heads (cheaper, usually
  neutral-to-positive; keep BatchNorm out of it, that's a separate axis).
- **QK normalization** (L2-normalize or RMSNorm queries/keys before the dot product, with
  a learned temperature) — kills attention-logit blowups at high lr; pairs well with the
  warmup-cosine recipe and might let find_lr pick a higher peak.
- **Attention-logit soft-capping** (`cap·tanh(logits/cap)`, Gemma-style) — same failure
  mode as QK-norm, cheaper, mutually exclusive with it in practice.
- **LayerScale / residual-branch scaling** — per-channel learnable scale on each residual
  branch, init ~1e-2. Note the ParT blocks already ship weaver's variant
  (`scale_fc/scale_attn/scale_heads/scale_resids`, all currently true): ablating those
  OFF is the zero-cost version of this axis.
- **DropPath / stochastic depth** (drop whole residual branches at rate ~0.05–0.1) — the
  ViT-era regularizer that often beats plain dropout at fixed budget; interacts with the
  dropout decision above.
- **ReZero / zero-init residual gates** (init the residual branch scale at exactly 0) —
  overlaps with LayerScale; pick one.
- **Register tokens** (a few extra learnable no-readout tokens alongside the CLS) — ViT
  finding that they absorb attention-sink artifacts; trivially equivariance-safe if kept
  scalar-grade like the CLS.
- **Muon / second-order-ish optimizers** for the transformer stage — the current-gen
  optimizer family beyond Lion; a training-side swap, not an architecture change.
  
  ### **GNN family**
  
- **PNA-style multi-aggregation (and other GNN aggregator tricks)** — replace the single
  mean/max aggregation in the MPNN/EdgeConv local branches with the PNA combination
  (mean+max+min+std, degree-scaled); the fixed-k kNN graphs make the degree scalers
  trivial, so it reduces to concatenated multi-aggregation. Same family: softmax/powermean
  aggregation, learnable aggregation temperature.
- **GraphNorm** on the local-branch node features — normalizes with a learnable mean-scale
  per graph, designed exactly for the BatchNorm-over-variable-graphs pathology the GPS
  models flirt with. NOTE: GraphNorm is **non-equivariant** (it shifts/scales per-feature
  across a jet's particles), so it is only a drop-in for the non-equivariant models —
  and thus currently less promising than it could become; an invariant-statistics variant
  (norms-only, grade-wise) would be the research version.
- Virtual node value that intializes CLS token for GraphTrans, connecting some global GNN output to transformer (primary global net)
- DropNode simulating pileup, but in my opinion goes too far by degrading the input, still could be promising since unexplored and could be made to mimick Soft Drop (Larkoski et al.)


Deliberately excluded from this list: **RoPE / ALiBi / any positional encoding along the
token axis** (jets are unordered sets — a sequence position is physics-meaningless; the
kinematic features already carry the real geometry), and **Mixup/CutMix-style input
interpolation** (a mixed jet is not a physical jet; label smoothing above is the sane
sibling).

These are deliberately untouched knobs: they are here for researchers interested in
fine-tuning these models or using them in their own experiments. If you want to pursue
research applying any of these tricks to the tagging setting, I'd be interested in
collaborating — open an issue or get in touch.

Deliberately excluded as *conceptually broken* (not "minor"): learnable **vector/multivector**
class tokens (pick a direction → break equivariance), BatchNorm over multivector components,
LapPE as an invariant PE under learned frames (sign/basis-ambiguous — kept only as the expected
negative result via `use_lappe`), and re-adding `add_tagging_features_framesnet` (upstream
residual-symmetry infrastructure, deliberately not resurrected).
