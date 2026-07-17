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
  `model.net.pair_input_dim=1|4|5|8` selects how many QCD interaction features (1=lnΔ; 4=+ln kT,
  ln z, ln m²; 5=+lnΔs²; 8=+cosθ,Δy,Δφ — see `pairwise_lv_fts`; the weaver feature ladder jumps
  5→8 when adding cosθ/Δy/Δφ, so 6/7 are not valid — `assert len(outputs)==num_outputs` enforces
  this). The learned weights compensate, so the bias stays compatible with the frame transport.
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

### `boost_jet` — feature/boost ordering + rotation-frame interaction (DECISION NEEDED)

**Background.** `data.boost_jet` (default **`true`** on top-tagging via `config/toptagging.yaml`;
`false` on the base `tagging.yaml`, inherited by JetClass/TopTagXL) boosts every jet to its own
rest frame inside `embed_tagging_data` before the backbone sees it — "to avoid large boosts,"
i.e. a numerical-stability aid for the framesnet's frame prediction. `init_physics` forces it
**off** for equivariant models and for identity-frames non-equivariant models, so `boost_jet=true`
is reached by exactly the **non-equivariant + learned-frames rows** (the LLoCa-on canonicalized
models: ParT/transformer/graphnet baselines and the four non-equivariant hybrids). Two loci
consume the boost; the audit's R-A1 flagged both.

**Locus 1 — FIXED (embedding-level features → framesnet). Correctness bug, done.**
`embed_tagging_data` used to compute the 7 tagging features (`log_pt, log_energy, log_pt_rel,
log_energy_rel, dphi, deta, dr`) *after* the boost. In the rest frame the jet has ~0 three-momentum,
so `pt_jet` hits its clamp and `φ_jet/η_jet` come from `atan2/eta` of a numerically-zero vector →
the 4 jet-relative features (`log_pt_rel, dphi, deta, dr`) were measured against an arbitrary axis
built from float residuals of the boost. That axis is **not** covariant, so these features — which
feed the **framesnet** as scalar inputs (`scalars_withspurions = cat([scalars, tagging_features])`,
wrappers.py) — broke the model's Lorentz invariance. Measured: `xyrotation` end-to-end output
max-MSE **O(10²–10⁴)** before, **O(10⁻⁸)** after; feature distributions also mismatched the
hardcoded lab-frame standardization constants (`log_pt_rel` mean ~12 vs the code's −4.7). **Fix
(committed):** compute the features in the lab frame first, *then* boost the momenta for the
backbone (`boost_jet=false` paths bit-identical). Invariance suite 24/24 green after.
  - **Comparability sub-decision:** this changes trained-model results for every learned-frames
    row. Upstream's published LLoCa numbers were trained with the old (degenerate) features, so if
    bit-comparability with their table matters, gate the reorder behind a flag / versioned note.
    Otherwise accept it and re-baseline (recommended — the old behavior is simply wrong).

**Locus 2 — NOT fixed (backbone-level recompute). The open decision.**
The `TaggerWrapper` backbone separately recomputes *local* tagging features from the **boosted**
momenta (`get_tagging_features(fourmomenta_local, jet_local, "all")`). The embedding-level reorder
does not reach this, and it can't be patched mechanically: computing local features from *pre*-boost
momenta while the backbone sees *post*-boost momenta would be a cross-frame subtraction (features
describing a different frame than the tokens) — likely worse. The only self-consistent option is a
config choice: **turn `boost_jet` off for the frames where it misbehaves.**

*Which frames misbehave (measured — determine empirically, the set is subtle):* a frame that
**fixes the time axis** cannot restore the boosted jet's momentum, so it stays at `(M,0,0,0)` and
the local jet-relative features degenerate. Transverse `pt` of the wrapper-local jet, by frame:

| frame | wrapper-local jet pt | strands? |
|---|---|---|
| `learnedpd`, `learnedso13` (full Lorentz) | ~10²–10³ | no |
| `learnedso3` (SO(3) rotation) | ~1e-11 | **yes** |
| `learnedso2` (SO(2) about beam) | ~1e-11 | **yes** |
| `learnedz` (z-boost) | real \|p_z\| | no (degenerate only in the transverse plane) |
| `learnedrest` (contains a boost) | ~6e5 | no |

So the earlier guess "(so3, so2, rest)" was wrong to include `rest` (it boosts). The pure-rotation
frames **`so3`, `so2`** strand the jet; `z` only transversely.

*What actually degenerates (measured — NOT "dead constants"):* every channel still varies across
constituents. `log_pt_rel` becomes a shifted **exact duplicate** of `log_pt` (corr 1.0 → one wasted
channel); `dphi/deta/dr` become the constituent's **absolute** local angles at the wrong
standardization scale (`dr` mean ~2.1 vs expected ~0.2). It is **NOT an invariance break** (features
computed in the canonical frame stay invariant even when degenerate — which is why 24/24 still
passes), and it touches only these specific ablation configs.

**Physics (this is the decisive argument, and why the time vector matters).** A time direction is
provided to essentially every model: `add_time_reference: true` adds a time spurion `[1,0,0,0]`
alongside the two beam spurions `[1,0,0,±1]` — a direct input token for the equivariant models and a
framesnet input for the LLoCa models. Rotation-only frames (`so3`/`so2`) exist **precisely** for the
regime where boosts are physically meaningful — the regime the beam/time reference *defines*.
`boost_jet` then boosts each jet to rest, i.e. it **discards exactly the boost information those
frames were chosen to preserve.** So `boost_jet` + pure-rotation frames is not just numerically
degenerate — it is **self-defeating**: you picked rotation-only canonicalization to keep boost info,
then boosted it away. `boost_jet=false` for `so3`/`so2` is therefore the *physically consistent*
choice, independent of the numerical degeneration.

**The remaining tradeoff.** `boost_jet`'s original job is framesnet numerical stability (small
boosts are easier to predict frames for); turning it off feeds the framesnet lab-frame momenta.
For 500–1000 GeV top-tagging jets this is a modest-boost regime, so the stability cost is likely
minor — but it is a real feature-quality/physics-consistency **vs** framesnet-stability tradeoff,
and it only affects the `so3`/`so2` ablation rows. (Aside, not part of this decision: `boost_jet`
also boosts the beam/time **spurions** per-jet — turning fixed lab references into jet-rest-frame
ones — for *all* frames; covariant so probably fine, but note it if you revisit the spurion design.)

**Decision — pick one for the `so3`/`so2` (and transverse-`z`) ablation rows:**
  - [ ] **(a, recommended) Force `boost_jet=false` for pure-rotation frames** in `init_physics`
        (determine the exact set empirically, not by name — `z` needs a transverse-only carve-out or
        just leave `z` as-is). Physically consistent; well-defined features; small stability cost.
  - [ ] **(b) Leave as-is and document** the residual (redundant/off-scale local features on those
        rows; not an invariance break). Zero code; the ablation rows are mildly under-fed.
  - [ ] **(c) Drop `so3`/`so2` from the symmetry-budget ablation** if the interaction makes them
        uninterpretable — but they're the point of that ablation, so (a) is better.
  **Not upstream-exclusive — the identical latent interaction is present on upstream** (same
  `TaggerWrapper` recompute from boosted momenta; `learnedso3.yaml`/`learnedso2.yaml` ship there;
  `init_physics` keeps `boost_jet=true` for non-equivariant **learned-frames** rows — the
  `boost_jet=False` fallback fires only for *identity* frames). So `model=tag_ParT
  model/framesnet=learnedso3` on top-tagging strands the jet on upstream too. It is kept OFF the
  upstream `original-repo-fixes` PR because it is a **design tradeoff (feature-quality vs framesnet
  stability), not a correctness bug** — no invariance break, features stay invariant-but-degenerate —
  NOT because upstream is immune. Upstream's headline results simply use full-Lorentz `learnedpd`
  frames, which don't strand; their so3/so2 subgroup ablations (if run on top-tagging with the
  default `boost_jet=true`) hit the same mild degeneration. If the decision below is taken, the
  `init_physics` fix is equally applicable upstream and could be a follow-up PR there.

### Audit findings (property-based sweep — permutation / mask / determinism / degenerate jets)
- [x] **BatchNorm-over-padding is FAITHFUL to official ParticleNet/ParT — verified, do NOT "fix".**
      The property test showed a real effect: in *training*, padding the same jet to more columns
      shifts the logits by up to 0.18 (ParticleNet-ParT GraphTrans), 0.08 (Plain GraphTrans), ~1e-3
      (the two GPS), because the input `bn_fts` (`nn.BatchNorm1d`, all 4 channels-first models) and the
      EdgeConv/MPNN `BatchNorm2d/1d` (Plain + ParticleNet GraphTrans) compute statistics over the
      zero-padded slots. **Checked against the references**: weaver ParticleNet does
      `self.bn_fts(features).masked_fill(padding_mask, 0)` (BN over the full padded tensor, mask
      *after*) with unmasked EdgeConv `BatchNorm2d`; weaver ParT's `Embed.input_bn` is a
      `nn.BatchNorm1d` over `(batch, channels, seq_len)` with no pre-mask (zeroed only after embed).
      Our ports reproduce both exactly, so this is intended fidelity, not a bug — masking it would
      DIVERGE from the architectures being compared. (The GraphGPS per-layer `MaskedNorm` is likewise
      faithful to the *GraphGPS* recipe's masked BatchNorm; each backbone matches its own lineage.)
      Eval is bit-exact (running stats). Nothing to change; documented for the paper's fidelity claim.
- [x] **Verified clean (all 8, float64):** determinism (bit-exact), permutation-invariance over
      particles (~1e-16), padded-VALUE leakage in eval (bit-exact), padding-COUNT invariance in eval
      (bit-exact), finite logits on degenerate 1-particle jets, **gradient coverage** (every trainable
      param reached by the loss), **batch-composition independence** (a jet's logits identical alone vs
      batched, ~1e-16 — no cross-jet leakage), and **identical-particle / collinear jets finite**.
      Set-symmetry, eval-time masking, batch isolation and numerics are sound across the family. (The
      `embed_tagging_data` in-place-`ptr` footgun found en route is documented at `embedding.py:103`.)
- [x] **Benign dead vector-path in `LorentzNetLGATrSlimGraphTrans`** (grad-coverage check). The
      `lgatr` LGATrSlim's `linear_out` multivector weights and the *last* block's MLP multivector
      weights get no gradient — they feed only the deliberately-discarded vector output
      (`out_v_channels=1`, "vector sink"; the model reads `_, s_out`). Intended, not a bug; a few wasted
      multivector params. (Earlier blocks' vector weights are live — vectors reach later scalars via
      attention. The GPS sibling routes differently and has none.)

### Audit findings (full GraphTrans-vs-GraphGPS sweep) — remaining, low priority
- [ ] **Dropout is inconsistent across the hybrid family — two layers to the decision (checked):**
      (a) *GPS local branch (latent)*: Plain + CGENN GPS apply an external `Dropout` to the local-MPNN
      output; LorentzNet + ParticleNet GPS apply none. No-op at shipped defaults — ALL FOUR GPS configs
      ship dropout 0/None, so at defaults the GPS family is behaviorally equal; only matters if the
      dropout ablation is ever run.
      (b) *GraphTrans transformer stage (LIVE)*: `tag_PlainGraphTrans` ships `dropout: 0.1` and
      `ParticleNetParTGraphTrans` gets 0.1/0.1/0.1 from its ParT-block class defaults (config sets no
      dropout keys), while CGENN/LorentzNet GraphTrans run dropout-free (`dropout_prob=None`). So the
      two non-equivariant GT hybrids train WITH dropout and the six other hybrids without — a live
      regularization asymmetry across both comparison axes (GT-vs-GPS and equivariant-vs-not).
      Each stage is *faithful to its source*: ParT blocks publish 0.1 — and the repo's `tag_ParT`
      reference row DOES train its 8 main blocks at 0.1 (its config zeroes only `cls_block_params`,
      weaver's own convention for the class-attention blocks) — L-GATr publishes none, and pure
      `tag_cgenn`/`tag_lorentznet` use 0.2 on their classification heads only (LorentzNet's LGEB
      dropout kwarg is dead code). So the consistent chains are: ParT-baseline 0.1 ↔ PNParT-GT 0.1
      (faithful), and L-GATr-none ↔ equivariant hybrids none (faithful). The one axis where dropout
      is confounded with the comparison is **GT (0.1) vs GPS (0) within the two non-equivariant
      backbones**. Decision: treat dropout as part of the reference block definition (per-reference,
      like FFN ratio and GELU/ReLU — keep as-is + one methods sentence + the existing family-wide
      dropout ablation row), OR harmonize the family to 0 (zeroing PNParT-GT breaks its faithfulness
      to the ParT baseline row it is directly compared against).

### Audit findings (infrastructure sweep: JetClass path / plots / trials)
- [x] **`jc_gts_and_friends_default` added** and `config/jctagging.yaml` now defaults to it (was
      `jc_ParT` — the same recipe-inheritance trap the top tree fixed). ParT-standard `epochs: 5`
      (1M steps x 512 = ~5 passes of 100M), CosineAnnealingWarmup, wd 0 (JetClass convention),
      validate once per nominal epoch.
- [x] **per-model `jc_<Hybrid>.yaml` recipes added (all 8)**, mirroring `top_<Hybrid>` on
      `jc_gts_and_friends_default`. See GUIDE §5.1. (`jc_lgatr`'s broken `tag_gatr` base was
      fixed on `main` directly — merges in cleanly.)
- [ ] **JetClass: fill the 8 `jc_<Hybrid>.yaml` `???` batchsize/lr** from
      `find_lr.py -cn jctagging model=tag_<hybrid> save=false +lr_find.find_batch_size=true`
      before the JetClass campaign (don't copy top values — inputs are 7+10 channels; and note
      an unfilled `???` silently runs at the 512/1e-3 fallback instead of erroring).
- [ ] **Rejection-metric convention differs between experiments** (pre-existing): top-tagging uses the
      nearest-ROC-point (`argmin |tpr - epsS|`), JetClass uses `scipy.interp1d` interpolation. One
      methods sentence, or unify.
- [x] **best-checkpoint restore now re-pairs the EMA**: the end of `train()` loads the checkpoint's
      `"ema"` alongside `"model"` (when `ema: true`), so the `_ema` eval uses the EMA shadow that
      belongs to the restored best-validation checkpoint instead of the end-of-training one.

### Pre-publication audit (session: jet_frames + GT-family sanity sweep)

Training-readiness verified across all 8 GT hybrids (real `config/`): forward + backward + AdamW step
crash-free; param counts 1.16–2.53M (LorentzNet 1.83/2.46M, CGENN GNN 248k — the earlier fixes held;
small later deltas: the audit's node_attr re-injection adds +1.5k/+6.7k to the LorentzNet hybrids and
the official-CGENN knob flip removes the NormalizationLayer params, so counts are now 1.15/1.90/1.83/2.46M
for CGENN-Trans/CGENN-GPS/LN-Trans/LN-GPS);
**zero dead input channels** in either the four-momentum path or the 7 `tagging_features` (the general
form of the CGENN `node_attr` check — CGENN comes back balanced). PDFrames runs end-to-end on the 4
non-equivariant hybrids (Plain × {Trans, GPS}, ParticleNet-ParT × {Trans, GPS}); the 4 internally-equivariant
hybrids (CGENN, LorentzNet) **assert `IdentityFrames`** by design (`self.framesnet = framesnet  # not actually
used`; `IdentityFrames` is 0-param and never referenced — a perfect no-op). The `/20` momentum rescale is
inherited from the standalone references (equivariant backbones rescale manually; non-equivariant ones
canonicalize via `TaggerWrapper` + BatchNorm). Equivariance 32/32 (both frames), `test_amplitudes` fixed,
training smoke (`PlainGraphTrans + learnedpd`) ran end-to-end through evaluation.

- [ ] **`torch.cuda.amp.autocast(...)` deprecation in 4 active baseline files** (plaingraphtrans.py:285,
      plaingraphgps.py:322, particlenetpartgraphgps.py:223, particlenettransformer.py:792; mipart.py
      has 2 more in commented-out code). `FutureWarning` today, error in some future torch. Mechanical
      migration to `torch.amp.autocast('cuda', ...)`; will not change current numerics.
- [ ] **ParT-GPS mixed-type attention mask deprecation** at particlenetpartgraphgps.py:115 — float
      `attn_mask=attn_bias` paired with bool `key_padding_mask` triggers torch's "mismatched
      key_padding_mask and attn_mask is deprecated" warning. Functionally correct today (padding still
      goes to −∞); future-fatal. Fix: merge the bool padding mask into the float bias (`bias.masked_fill(pad, -inf)`)
      and pass a single float `attn_mask`.
- [ ] **`xformers` env note for the SLURM target** — the installed wheel must match the cluster's
      torch+python or the L-GATr `lgatr` equivectors silently fall back / fail to load (this is the same
      class as the 9 environment-only FLOPs failures in this dev container). Pin a known-good
      (torch, xformers) pair in `docs/SLURM.md` under the install step; matters only for runs that
      actually use `lgatr` equivectors.
- [ ] **Precision-floor note for the paper** — `learnedpd` carries a higher boost-precision floor than
      `learnedso13` (float64, polar decomposition divides by energy). Measured at ~1e-4 absolute (kNN, 10
      boosts) on the GT hybrids — far below any true symmetry break and consistent with the standalone
      baselines (ParT ~1e-4, ParticleNet ~1e-7 same conditions). The test file already encodes per-frame
      tolerances; one sentence in the methods section would head off reviewer questions.

- [x] **jet_frames lloca-compat fix** — `TaggerWrapper.jet_frames` always uses the 4d orthogonalizer
      but reused the framesnet's `ortho_kwargs` (which the PD family keys as `eps_reg`, the 3d name).
      Translate the key for the 4d call so any framesnet works.
- [x] **jet_frames missing `num_graphs`** — set-level equivectors (`pelican`) need it; mirror the main
      framesnet path. `equimlp` absorbs it via `**kwargs`.
- [x] **`test_tag_equivariance.py::test_lloca_frame_invariance`** now parametrizes over both `learnedpd`
      and `learnedso13` (LLoCa's recommended default is PD; `so13` was the only frame tested before, which
      hid the jet_frames bug). Per-frame tolerances (`so13` ≤ 1e-3, `pd` ≤ 2e-2) — 16/16.
- [x] **`test_tag_invariance.py::test_amplitudes`** had a stale config key (`data.tagging_features_framesnet=null`,
      removed upstream by `f08f7df`/`a45da1b`) — every case failed at config composition before any model ran,
      so the baseline-under-frames check was silently dead. Aligned to `data.tagging_features=null`; 16/16.

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
- [ ] `experiments/base_experiment.py:262` — `path_code = os.path.join(self.cfg.base_dir, "lloca")`
      hardcodes "lloca" for the saved-source dir → project name.
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
