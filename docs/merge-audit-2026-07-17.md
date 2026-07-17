# Final mega-merge audit — peaceful-ptolemy → main (2026-07-17)

Scope: the full tree diff `origin/main...claude/peaceful-ptolemy-qIXQd` (77 files, ~6.9k added
lines), every upstream commit on `main` since the June fork point, the `original-repo-fixes`
branch (rated per the maintainer's request), plus empirical verification in a fresh CPU
container (torch 2.13, torch-geometric 2.8, lloca 1.3.6, lgatr 1.4.4, hydra 1.3.4, xformers
0.0.35 CPU). Everything below was verified by reading the code AND, where marked, by running
it. Known issues already ledgered in `todo.md` / code comments are not re-reported.

**Test evidence collected:** `test_tag_equivariance.py` 32/32 PASS; `test_tag_invariance.py`
24/24 PASS (but see B5 — that suite asserts nothing); `test_tag_flops.py` 53 pass / 19 fail,
every failure environmental (18 × the known lloca-xformers assert on a CUDA-less install, 1 ×
`tag_pelican_fair` torch.compile in this container). End-to-end smokes: default quick training;
`tag_PlainGraphGPS` with `training.epochs=1` (budget 157 = 1 × len(loader) derived correctly,
validated once/epoch, best checkpoint restored); documented fresh-trial warm start (run_idx
increments, fresh init confirmed by diverging metrics, `[2 trials] mean ± std` row emitted;
`table_metrics_test.json` accumulates correctly). `collect_data.py jetclass`: extraction layout
and all 13 md5s verified identical to the official `jet-universe/particle_transformer`
downloader.

---

## A. Merge mechanics — handle these AT the merge

**A1. The histories are unrelated — this will not be a normal merge.** peaceful-ptolemy's root
(3fb972b, June 19) is a snapshot with no common ancestor with `main` (`git merge-base` is
empty). `git merge` needs `--allow-unrelated-histories` and every differing file becomes a
two-sided conflict with no base. Anything resolved "take branch side" silently reverts upstream
work. Concretely, taking the branch tree wholesale reverts A2–A4 below. Recommended: merge with
`-X theirs`-style resolution *file by file*, or replace main's tree and then re-apply the A2–A4
deltas; then merge/rebase `original-repo-fixes` (section C) on top.

**A2. `config/training/jc_lgatr.yaml` is broken on the branch** — still `defaults: [tag_gatr]`;
no `tag_gatr.yaml` exists, so `training=jc_lgatr` fails Hydra composition (verified:
`MissingConfigException`). Main fixed this via #91+#93 (`top_lgatr`). todo.md's "merges in
cleanly" assumption does not hold for unrelated histories — this must be resolved to main's
side by hand.

**A3. `requirements.txt` lost `wget`, which the branch's own code imports.** Upstream added
`wget` (b896347) because `data/collect_data.py` imports it at module level (branch lines 5,
52, 90 — the branch even extended its use for JetClass). A by-the-book fresh install
(GUIDE §1) then fails at the very first quickstart step `python data/collect_data.py
toptagging` with `ModuleNotFoundError: No module named 'wget'`. Add `wget` back (or port the
downloader to `requests`/`urllib`, which upstream's comment half-suggests anyway).

**A4. REPRODUCE.md / run.py regressions vs main:** the five upstream-added
`model.framesnet.gamma_max=3` flags on the ttbar generator reproduction commands (516d5df,
"Add missing…") are absent from the branch's REPRODUCE.md, as is the red "Multi-GPU not
supported" warning box (c562a97) and run.py's multi-GPU warning print. The branch predates
these; a branch-side merge resolution drops them.

## B. Live workflow traps (novel findings, empirically confirmed)

**B1. `boost_jet` + learned frames: the SO(2)-invariance of every learned-frames row is
broken by degenerate tagging features — on the real config, today.**
`config/toptagging.yaml` ships `data.boost_jet: true`. `embed_tagging_data` boosts all momenta
into the jet rest frame *first* and computes the 7 tagging features *afterwards*, so the jet
used as reference is ≈ (M, 0, 0, 0) and `phi_jet`/`eta_jet` are atan2/asinh of numerical noise.
`init_physics` force-disables `boost_jet` for the internally-equivariant family and for
identity-frames runs, so the *only* live consumers are exactly the **learned-framesnet rows**
(Plain/ParticleNetParT hybrids and the ParT/transformer/particlenet/graphnet baselines under
`learnedpd`/`learnedso13`/…): their framesnet equivectors eat the degenerate features.
Measured here (float64, quick tree, `boost_jet=true`): `dphi` shifts up to **5.76** and `dr` up
to **2.09** under a *pure azimuthal rotation* (the docstring promises "all features are
SO(2)-invariant"), and `PlainGraphGPS + learnedso13` logits move by **1.46**. The repo's own
equivariance tests can't see this because every learned-frames test sets
`data.tagging_features=null`. This is upstream-inherited (main has `boost_jet: true` too — it
affects the published LLoCa pipeline, not just this branch) and is exactly what
`original-repo-fixes` fixes by computing features before the boost — that fix is **correct and
necessary** (see C). JetClass is unaffected (`boost_jet` false there).

**B2. The default-training swap silently changes or breaks every REPRODUCE.md tagging
command.** `config/toptagging.yaml` now defaults to `tag_gts_and_friends_default` (AdamW,
lr 1e-3, epochs 20) instead of `top_transformer` (Lion, 3e-5, 300k iters), and
`config/jctagging.yaml` to `jc_gts_and_friends_default` instead of `jc_ParT` (Ranger, 1M
iters). The swap itself is documented in diffs.md — but its side effect is not: all REPRODUCE
baseline commands that pass no `training=` (Table 5 `tag_top_transformer`; every JetClass
Table 3/4/9 row) now compose to the *wrong recipe* and will not reproduce the upstream papers.
The one command that does pass `training=` (`training=jc_lgatr`) errors instead (A2). If
REPRODUCE.md is kept as upstream-paper documentation, each of its commands needs an explicit
`training=top_transformer` / `training=jc_ParT` (or a note).

**B3. An explicit `training.iterations=N` on the CLI is silently ignored whenever the composed
recipe carries an epoch budget** — which both task defaults now do. Verified: `python run.py
-cn toptagging training.iterations=1000` composes to `epochs=20, iterations=1000` and
`_resolve_epoch_budget` overwrites 1000 with `20 × len(train_loader)` (≈47k on full data —
the REPRODUCE timing-estimate command now runs ~47× longer than documented, and GUIDE §4's own
"useful override: `training.iterations=…`" no longer works on the defaults). Suggested fix:
give a *non-null* `iterations` precedence over `epochs` (config-family defaults still work,
CLI ergonomics restored), or at minimum log a WARNING when a non-null iterations value is
overwritten.

**B4. All four GitHub workflows are permanently skipped — the CI the fork claims was never
running.** Every job is gated `if: github.event.label.name == 'ready for review'`, but
`on.pull_request` uses the default activity types (opened/synchronize/reopened), which never
include `labeled`; on push/dispatch events `github.event.label` doesn't exist either. So the
condition is false for every event that can trigger the workflow: the tests job (including the
equivariance-suite line this branch added, and the tagging job whose stale key the branch
fixed) has never executed. diffs.md's "CI runs the tagging equivariance+invariance suites" and
GUIDE §9's "only triggers on the ready-for-review label" are both wrong in practice.
`original-repo-fixes` addresses this (see C — rated: right direction, still gappy).

**B5. `test_tag_invariance.py` contains no assertions.** It computes invariance MSEs and
*prints* them; the 24 "passed" cases only prove composition+forward don't crash. The branch's
new `test_tag_equivariance.py` does assert (32 real checks — good), but any green-CI claim
resting on the invariance suite is hollow, and it is the suite named in tests.yaml.
`original-repo-fixes` converts it to real asserts with per-jet transforms (rated correct).

## C. Rating of `original-repo-fixes` (86956a2) — requested review

The branch is a curated port of original-repo bugs onto clean main for an upstream PR. Overall
verdict: **high quality — 12 of 13 items verified correct, one directionally-correct but
incomplete. Crucially, NONE of these fixes are contained in peaceful-ptolemy** (they were
ported onto main, and the histories are unrelated), and they touch the same files the branch
rewrites (embedding.py, experiment.py, wrappers.py, base_experiment.py, workflows, tests,
config_quick/jctagging.yaml) — so the merge plan must include a deliberate second step:
re-apply/merge `original-repo-fixes` after the ptolemy merge and resolve conflicts.

| Fix | Verdict | Present in ptolemy? |
|---|---|---|
| embedding.py: features before boost | **Correct** — empirically reproduced the bug it fixes (B1) | **No — bug live** (learned-frames rows) |
| embedding.py: mass_reg skips spurions | **Correct** (spacelike beam m²=−1 < mass_reg² always; silently made lightlike, voiding the beam ablation) | No — bug present (latent: beam ablations) |
| wrappers.py: ParticleNetWrapper `[4,5]` → extra_scalars offset | **Correct** (JetClass PID one-hots landed in the (phi,eta) kNN slots) | No — bug present (JetClass `tag_particlenet` rows) |
| experiment.py: `param_groups is None` guard | **Correct** — and ptolemy *amplified* the clobber by adding 4 hybrid names to the match list, so a finetune of any GraphTrans hybrid would clobber its backbone/head lr split too | No — bug present + wider |
| finetune ema_decay top-level read | **Correct** (`cfg.training.ema_decay` doesn't exist; struct-mode crash) | No — bug present (finetune+ema) |
| base_experiment: `scaler.update()` before skip-return | **Correct** (classic AMP scale-freeze; latent: needs `max_grad_norm` + amp) | No — bug present (latent) |
| workflows: subscribe `labeled` + dispatch escape | **Directionally correct, incomplete**: after the label lands, later `synchronize` pushes still skip (label empty on those events), and pushes to main still skip. Prefer `contains(github.event.pull_request.labels.*.name, 'ready for review')` on all PR event types | No — CI fully dead (B4) |
| test_tag_invariance: real asserts + per-jet transforms | **Correct** (per-jet transforms would catch cross-jet-leakage bugs of exactly the fixed CGENN class) | No — print-only (B5) |
| test_tag_flops: skip env failures | Reasonable | No |
| miniweaver: ternary parens / read-failure logs / missing-glob logs | Plausible-correct (not deeply re-verified) | No |
| jetclassexperiment: infinity_mode train-only | **Correct** (with `steps_per_epoch` set, val/test loaders loop forever) | No — bug present (latent) |
| jetclassexperiment: resolved-file-count log | Good hygiene | No |
| config_quick/jctagging: file ranges [122,123]→existing per-split files | **Correct** (previous ranges point at files that exist only in val_5M → zero train/test files) | No — quick jc broken |

## D. Findings in the new hybrid/infra code (three deep-review passes + spot verification)

The heavy machinery was independently re-verified and is **sound**: ParT block + pairwise
features bit-identical (0.0) to the mipart/weaver reference; LLoCa transport bit-identical to
lloca's reference attention and invariant to a global frame right-factor at 1e-15/float64;
kNN builders brute-force-matched on ragged batches (no cross-jet/padded/self edges — the fixed
upstream CGENN bug class is absent everywhere); CGENN dense-edge fix semantically identical to
upstream #92; spurion channels never leak into graphs/pools; masked means divide by true
counts; wd-grouping names match `named_parameters()`; all 16 model configs instantiate with
every key consumed. Remaining findings, none critical:

- **learnedrest kNN degeneracy (major if that ablation is run):** the Plain/PNParT wrappers
  build kNN `points` from *canonicalized* momenta; `LearnedRestFrames` maps every particle to
  (m,0,0,0), so deltaR (and minkowski) distances become pure float noise — the todo §3
  "symmetry-budget variants" ablation would silently train on a physics-free graph for
  `learnedrest`. (`learnedpd`/`learnedso13`/identity verified unaffected.)
- **`jet_frames` assumes ≥2 equivectors:** `framesnet=learnedso2` (n_vectors=1) crashes the two
  GraphTrans hybrids' readout-frame construction (loud shape error) — another todo §3 variant.
- **`attn_reps=null` + any learned framesnet crashes** both PNParT backbones
  (`NoneType.prepare_frames`) — the documented "non-tensorial" ablation only works on identity.
- **PNParT-GPS layer-0 `minkowski` + `v=None` crashes** (GraphTrans silently falls back to
  points instead) — unreachable via shipped wrappers, but an API asymmetry.
- **`use_pre_activation_pair: true`** in both PNParT hybrids diverges from `tag_ParT`'s
  `false` (and weaver/MIParT training configs) — an undocumented ParT-fidelity divergence in
  the pairwise-bias head; GT-vs-GPS is internally consistent.
- **PlainGraphTrans FFN residual is undropped** (`x + ffn(...)` with dropout only inside),
  while the ParT reference block drops after fc2 — one more unlisted dropout asymmetry beyond
  the ledgered 0.1-vs-0 note.
- **"Isolates the fusion" overclaim:** plaingraphgps's docstring/config comments say
  GT-vs-GPS isolates interleaved-vs-sequential fusion, but the pair also differs in activation
  (gelu/relu), ffn_ratio (4/2), norm scheme (preLN vs masked-BN), readout (CLS+jet-frame vs
  mean-pool), MPNN internals, and message-passing rounds (3 vs 10) — each faithful to its
  lineage per the documented philosophy, but the aggregate attribution sentence overstates;
  worth one methods-section caveat.
- **GPS `**kwargs` swallow** (both equivariant GPS classes): a typo'd `model.net.<key>=` on a
  GPS model silently no-ops instead of erroring (the GT siblings raise).
- **Quick-tree drift:** `config_quick tag_LorentzNetLGATrSlimGraphTrans` lacks
  `mlp_ratio/attn_ratio/num_layers_mlp`, so quick tests run mlp_ratio 2 while the real model
  is 4 (equivariance unaffected); quick `tag_CGENNLGATrGraphGPS` omits
  `use_explicit_edge_features` (harmless — class default true).
- **Cosmetic:** stray debug comment `#tests say no` (CGENNLGATrGraphTransHybrid.py:1205);
  LN-GT module title still says "no per-edge attention gate"/"gateless" though `use_phi_m`
  defaults true; `for_inference: true` with 1-class output softmaxes to constant 1.0 (weaver
  inheritance, latent); `use_edge_attr=true` ablation feeds unstandardized log|(pᵢ+pⱼ)²|
  (range ≈ ±18) into messages; an eval-only rerun (`train=false`) prints the *persisted*
  table rows rather than the just-computed metrics; `_count_flops` on the train-split row uses
  a shuffled first jet (test-split rows are deterministic and comparable); GUIDE §1's
  "no GPU needed" smoke command still requires xformers (the quick default model is the
  transformer — the §7 xformers-free story only covers the hybrids).

## E. External-package hazard (not this repo's code, affects baseline rows)

lloca 1.3.6's `ParticleTransformer.forward` calls `self.attention.prepare_frames(frames)`
*before* `self.trimmer(x, v, mask, uu)` permutes (training) / truncates tokens, and the trimmer
does not reorder frames. `tag_ParT` ships `trim: true`, so **learned-frames tag_ParT rows**
train with frames misaligned against the shuffled token order (identity frames unaffected;
the hybrids are immune — they deliberately removed the trimmer). Verify independently before
citing an LLoCa-ParT baseline row, and consider `model.net.trim=false` for those runs (or an
upstream lloca fix).

## F. Deliberate divergences double-checked and fine (no action)

`use_fusion: true` in tag_particlenet (matches weaver/official ParticleNet; commit-documented;
note it differs from the upstream-LLoCa-paper config when comparing reference rows).
`es_patience: null` shared by baselines+hybrids (disclosed). Epoch-budget derivation on
JetClass (`len(SimpleIterDataset)` = files × events_per_file → the ParT 1M-step exposure at
bs 512, subset-scaling as documented). LapPE is actually implemented behind `use_lappe`
(todo §3 "not implemented" is stale). The `???`-fallback guardrail warning works as designed.

## Bottom line

The new-model code and the fairness machinery are in materially better shape than a typical
research branch — three independent adversarial passes plus bit-exact numeric verification
found no critical defect in the 8 hybrids, and the fresh-trial/epoch-budget/table pipeline
works end-to-end. What stands between this branch and a clean merge is **integration, not the
models**: (1) the unrelated-history merge must not revert main's fixes (A2–A4), (2)
`original-repo-fixes` — which is accurate — must land too, especially the tagging-features
reorder (B1) that currently invalidates every learned-frames row on the real config, and (3)
the default-recipe swap's REPRODUCE/CLI side effects (B2/B3) and the dead CI gate (B4/B5)
should be fixed before results from this tree back a publication.
