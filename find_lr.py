"""
    python find_lr.py -cp config -cn jctagging model=tag_transformer save=false
    python find_lr.py -cp config -cn toptagging model=tag_transformer save=false
    python find_lr.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd save=false

The task is selected with `-cn` (toptagging / jctagging / amplitudes / ttbar / ...)
and the dataset within a task with the usual data overrides (e.g. `data.dataset=mini`
for top tagging); the sweep simply cycles that task's training dataloader, so a
larger `+lr_find.num_iter` samples more of the data.

The recommended learning rate is reported as `loss-min / 10` (a robust peak lr for
an annealed / one-cycle schedule); the steepest-descent point is also printed.

Prefer `loss-min / 10`: it is stable in `num_iter` because it tracks the
edge-of-divergence lr that the loss landscape fixes. The steepest-descent point is
NOT -- a longer sweep lets ordinary training progress (not the lr) dominate the loss
drop and biases it toward low lr (davidtvs/pytorch-lr-finder#68). Keep `num_iter`
short (300 is deliberate); if a suggestion looks unstable, lower it, don't raise it.

It reuses the experiment's own `_batch_loss`, optimizer, scaler and dataloader,
so the measured loss-vs-lr curve reflects the lr-scale-determining setup: optimizer
type, betas/eps, param-group ratios (`lr_factor_framesnet`), batchsize and amp. Two
regularizers are deliberately switched off for the sweep -- `weight_decay` (inert over
~300 steps) and gradient clipping (it would cap the step and mask the high-lr
divergence the test needs) -- so the curve is the raw loss-vs-lr response. The base
`training.lr` is *ignored*; only the inter-group lr ratios are preserved.

The optimizer (type/betas) and param-groups come from the chosen *training* config; the task
defaults (e.g. `toptagging`) now select `tag_gts_and_friends_default` (AdamW), so
the GT hybrids sweep correctly with just `model=...`. Because the suggested lr is
optimizer-specific, pass `training=top_<baseline>` to sweep a baseline under its own
optimizer (e.g. `training=top_transformer` for the Lion transformer).
 
The test follows the Leslie-Smith / fastai recipe: exponentially ramp the lr over
a few hundred batches, record an EMA-smoothed training loss, stop early if the
loss diverges, then report `loss-min / 10` (steepest-descent is printed too, but see
the num_iter caveat above).
 
Pass `save=false` so no run directory is created. Tune the sweep on the CLI under
`lr_find.*` (these keys are not in the base config, so the `+` prefix is REQUIRED --
a bare `lr_find.num_iter=…` raises a Hydra ConfigCompositionException):
 
    +lr_find.start_lr=1e-7   lowest lr in the sweep                 (default 1e-7)
    +lr_find.end_lr=1e1      highest lr in the sweep                (default 1e1)
    +lr_find.num_iter=300    number of batches in the sweep         (default 300)
                             (dataset-size independent: 300 is right for JetClass too --
                             the sweep samples the lr ramp, not the dataset, and raising
                             num_iter biases the suggestion low, see the note above)
    +lr_find.beta=0.98       EMA factor for loss smoothing          (default 0.98)
    +lr_find.diverge=5.0     stop when smoothed_loss > diverge*best (default 5.0)
    +lr_find.skip_start=10    points dropped from the start          (default 10)
    +lr_find.skip_end=5      points dropped from the end            (default 5)
    +lr_find.output=lr_finder.png   plot path                       (default lr_finder.png)
    +lr_find.force_knn_metric=deltaR  pin the kNN metric for models that have one (default
                                    'deltaR'; 'keep' uses the model's own, 'minkowski' sweeps that).
                                    The lr scale is metric-independent, so the suggestion still
                                    transfers to a run trained under the model's configured metric.

On a GPU you can also auto-size the batch first, then sweep the lr at that size
(so the suggested lr matches the regime you will train in):

    +lr_find.find_batch_size=true   double the batchsize until CUDA OOM (full step)
    +lr_find.bs_start=16            smallest batchsize tried               (default 16)
    +lr_find.bs_max=16384           largest batchsize tried                (default 16384)
    +lr_find.bs_safety=1.0          fraction of the largest fit to use     (default 1.0,
                                    i.e. the largest fitting power of two; <1 adds
                                    headroom but breaks the power of two)

e.g.  python find_lr.py -cp config -cn toptagging model=tag_LorentzNetLGATrSlimGraphGPS \\
          save=false +lr_find.find_batch_size=true
prints both the GPU-fit batchsize and the suggested lr. (On CPU the batch-size
search is a no-op.) Verify the printed batchsize with a short real run before a
long job -- it measures one fwd+bwd, not a full training trajectory.
"""
 
import os
 
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
 
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment
from experiments.logger import LOGGER
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment
 
CONSTRUCTORS = {
    "toptagging": TopTaggingExperiment,
    "toptaggingft": TopTaggingFineTuneExperiment,
    "toptagxl": TopTagXLExperiment,
    "jctagging": JetClassTaggingExperiment,
    "amplitudes": AmplitudeExperiment,
    "amplitudesxl": AmplitudeXLExperiment,
    "ttbar": ttbarExperiment,
}
 
DEFAULTS = dict(
    start_lr=1e-7,
    end_lr=1e1,
    num_iter=300,
    beta=0.98,
    diverge=5.0,
    skip_start=10,
    skip_end=5,
    output="lr_finder.png",
    # pin the kNN graph metric for the sweep when the model exposes a `knn_metric` choice
    # (no-op for models without one). The lr SCALE is fixed by the optimizer/batchsize/amp,
    # not the graph metric, so pinning one metric only makes the suggested lr comparable
    # across models -- the model still TRAINS under its own configured metric. 'deltaR' is the
    # default pin (every model can build it); set 'keep' to sweep each model's own metric.
    force_knn_metric="deltaR",
    # optional GPU batch-size search (CUDA only; no-op on CPU)
    find_batch_size=False,
    bs_start=16,       # smallest batchsize tried
    bs_max=16384,      # largest batchsize tried
    bs_safety=1.0,     # fraction of the largest fitting batchsize to use (1.0 keeps a power of two)
)
 
 
def build_experiment(cfg):
    """Construct and partially initialize an experiment (no scheduler, no training).
 
    Mirrors BaseExperiment.full_run() up to the point where the optimizer and
    scaler exist, which is all the range test needs. Matches the init sequence
    used in the equivariance tests.
    """
    try:
        constructor = CONSTRUCTORS[cfg.exp_type]
    except KeyError as err:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented") from err
 
    exp = constructor(cfg, rank=0, world_size=1)
    exp._init()  # device, logger, (run dir is skipped when save=false)
    exp.init_physics()  # wire data dims -> model dims
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_optimizer()
    exp._init_scaler()
 
    exp.model.to(exp.device)
    return exp
 
 
def _is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def find_max_batch_size(exp, start, max_cap, safety):
    """Doubling search for the largest batchsize that survives a full training step.

    At each candidate it runs a real fwd + bwd + optimizer step (scaler + gradient
    clipping as in real training -- unlike ``range_test``, which deliberately omits
    clipping so it does not mask divergence), so the measured memory reflects what
    training actually uses -- not just a fwd+bwd lower bound. The search doubles
    until a CUDA OOM, so the largest size that fits is a power of two; with the
    default ``safety=1.0`` that power of two is returned unchanged (a fractional
    ``safety`` trades GPU utilisation / the power-of-two for headroom and is only
    worth it if you see OOM later from jets larger than those in the probe batch).

    CUDA only (returns the configured batchsize on CPU). The optimizer step mutates
    the model + optimizer state, so the caller MUST re-initialise before the lr sweep.

    NOTE: it probes one batch per size; verify the chosen batchsize with a short
    real run before launching a multi-day job.
    """
    if not torch.cuda.is_available():
        LOGGER.info("No CUDA device -> skipping batch-size search (keeping configured batchsize).")
        return int(exp.cfg.training.batchsize)

    exp.model.train()
    last_ok, bs = None, int(start)
    LOGGER.info("Searching for the largest batchsize that fits a full training step:")
    while bs <= max_cap:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with open_dict(exp.cfg):
                exp.cfg.training.batchsize = bs
            exp._init_dataloader()
            data = next(iter(exp.train_loader))
            loss, _ = exp._batch_loss(data)
            exp.optimizer.zero_grad(set_to_none=True)
            exp.scaler.scale(loss).backward()
            exp.scaler.unscale_(exp.optimizer)
            if exp.cfg.training.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    exp.model.parameters(), exp.cfg.training.clip_grad_norm,
                    error_if_nonfinite=False,
                )
            exp.scaler.step(exp.optimizer)
            exp.scaler.update()
            peak = torch.cuda.max_memory_allocated() / 1e9
            LOGGER.info(f"  batchsize {bs:6d}: OK  (peak {peak:.1f} GB)")
            last_ok = bs
            bs *= 2
        except RuntimeError as err:
            if not _is_oom(err):
                raise
            LOGGER.info(f"  batchsize {bs:6d}: OOM")
            torch.cuda.empty_cache()
            break

    if last_ok is None:
        LOGGER.warning(f"Even batchsize {start} does not fit; keeping {start}.")
        return int(start)
    chosen = max(int(start), int(last_ok * safety))
    note = (
        "the largest fitting power of two"
        if safety >= 1.0
        else f"{safety:.0%} of {last_ok} (NOT a power of two)"
    )
    LOGGER.info(f"Largest fitting batchsize {last_ok} -> using {chosen} ({note}).")
    return chosen


def _cycle(loader):
    while True:
        yield from loader
 
 
def range_test(exp, start_lr, end_lr, num_iter, beta, diverge):
    """Exponentially ramp the lr and record the EMA-smoothed training loss."""
    optimizer, scaler = exp.optimizer, exp.scaler
 
    # preserve the relative lr ratios between param groups (net vs framesnet vs ...)
    base_lr0 = optimizer.param_groups[0]["lr"]
    base_ratios = [pg["lr"] / base_lr0 for pg in optimizer.param_groups]
    gamma = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))

    # The range test measures the RAW loss-vs-lr response, so neutralize the two training
    # regularizers that would only distort it (neither sets the lr SCALE -- that is the
    # optimizer type + batchsize): weight_decay is inert over a ~300-step sweep (the
    # weight-norm equilibrium needs thousands of steps), and grad clipping would cap the
    # step and can mask the high-lr divergence the test must see (a no-op for AdamW/Lion,
    # which renormalize, but not for plain SGD -> off keeps the test optimizer-agnostic).
    for pg in optimizer.param_groups:
        pg["weight_decay"] = 0.0
 
    lrs, losses = [], []
    avg_loss, best_loss = 0.0, float("inf")
 
    exp.model.train()
    iterator = iter(_cycle(exp.train_loader))
    log_every = max(1, num_iter // 20)
 
    for step in range(num_iter):
        lr = start_lr * gamma**step
        for pg, ratio in zip(optimizer.param_groups, base_ratios):
            pg["lr"] = lr * ratio
 
        data = next(iterator)
        loss, _ = exp._batch_loss(data)
 
        # update step -- NO grad clipping on purpose: it would cap the step and hide the
        # high-lr divergence the test needs to graph (see the note above the loop).
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        loss_value = loss.detach().item()
        if not np.isfinite(loss_value):
            LOGGER.warning(f"Non-finite loss at step {step} (lr={lr:.2e}); stopping.")
            break
 
        # bias-corrected EMA of the loss
        avg_loss = beta * avg_loss + (1.0 - beta) * loss_value
        smoothed = avg_loss / (1.0 - beta ** (step + 1))
        lrs.append(lr)
        losses.append(smoothed)
 
        if step == 0 or smoothed < best_loss:
            best_loss = smoothed
        if smoothed > diverge * best_loss:
            LOGGER.info(f"Loss diverged at step {step} (lr={lr:.2e}); stopping early.")
            break
 
        if step % log_every == 0:
            LOGGER.info(f"step {step:4d}  lr={lr:.2e}  smoothed_loss={smoothed:.4f}")
 
    return np.array(lrs), np.array(losses)
 
 
def suggest_lr(lrs, losses, skip_start, skip_end, beta=0.98):
    """Two heuristics: `loss-min/10` (robust, the recommended one) and the
    steepest-descent point (lr at the minimum gradient of loss vs log(lr)).

    The EMA used to smooth the loss leaves a high-variance transient over its
    ~1/(1-beta) warmup window; if the gradient search sees it, the "steepest"
    point collapses onto that early dip (e.g. ~1e-7). We therefore skip the
    warmup window before searching for the steepest point. `loss-min` is taken
    over the whole trimmed curve since it is unaffected by the warmup.
    """
    n = len(lrs)
    if n <= skip_start + skip_end + 2:
        skip_start, skip_end = 0, 0
    end = n - skip_end
    lr_trim = lrs[skip_start:end]
    loss_trim = losses[skip_start:end]

    # steepest descent, ignoring the EMA warmup transient
    warmup = min(int(round(1.0 / (1.0 - beta))), max(0, (end - skip_start) // 3))
    lr_grad = lrs[skip_start + warmup : end]
    loss_grad = losses[skip_start + warmup : end]
    if len(lr_grad) >= 2:
        gradients = np.gradient(loss_grad, np.log(lr_grad))
        steepest = float(lr_grad[int(np.argmin(gradients))])
    else:
        steepest = float(lr_trim[int(np.argmin(loss_trim))])

    min_loss_lr = float(lr_trim[int(np.argmin(loss_trim))])
    return steepest, min_loss_lr, lr_trim, loss_trim
 
 
def make_plot(lrs, losses, steepest, min_loss_lr, output, title="LR range test"):
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lrs, losses, color="#1f77b4")
    ax.axvline(steepest, color="#d62728", ls="--", label=f"steepest: {steepest:.2e}")
    ax.axvline(
        min_loss_lr / 10.0,
        color="#2ca02c",
        ls=":",
        label=f"loss-min / 10: {min_loss_lr / 10.0:.2e}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed training loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    LOGGER.info(f"Saved LR-finder plot to {os.path.abspath(output)}")
 
 
# Default to the real config/ tree so a bare run matches training (clipping on, full data);
# pass `-cp config_quick ... data.dataset=mini` for a fast smoke test of the finder itself.
@hydra.main(config_path="config", config_name="toptagging", version_base=None)
def main(cfg):
    # LR finding is single-process and never trains / evaluates / saves a model
    cfg.train = False
    cfg.evaluate = False
    cfg.plot = False
 
    params = dict(DEFAULTS)
    lr_find = OmegaConf.select(cfg, "lr_find", default=None)
    if lr_find is not None:
        overrides = OmegaConf.to_container(lr_find, resolve=True)
        params.update({k: v for k, v in overrides.items() if v is not None})

    # Pin the kNN graph metric for the sweep when the model exposes one. GT/hybrid models carry
    # `model.net.knn_metric` ('deltaR' = eta-phi L2, 'minkowski' = Lorentz interval); models
    # without it (ParT, transformer, ...) are untouched. The lr SCALE is set by the optimizer +
    # batchsize + amp, not the graph metric, so pinning one metric only makes the suggested lr
    # comparable across models -- the model still trains under its own configured metric. The
    # wrappers always build the (eta, phi) points, so pinning 'deltaR' is input-safe. Pass
    # +lr_find.force_knn_metric=keep to sweep each model's own metric (or =minkowski to pin that).
    forced_metric = params["force_knn_metric"]
    if isinstance(forced_metric, str) and forced_metric.lower() in ("keep", "none", "off", ""):
        forced_metric = None
    current_metric = OmegaConf.select(cfg, "model.net.knn_metric", default=None)
    if forced_metric is not None and current_metric is not None and current_metric != forced_metric:
        with open_dict(cfg):
            cfg.model.net.knn_metric = forced_metric
        LOGGER.info(
            f"LR sweep: forcing model.net.knn_metric '{current_metric}' -> '{forced_metric}' "
            f"(metric-independent lr scale; +lr_find.force_knn_metric=keep to retain '{current_metric}')."
        )

    exp = build_experiment(cfg)

    # optional: size the batch to the GPU before the lr sweep, then sweep at that
    # batchsize so the suggested lr matches the regime you will actually train in
    if params["find_batch_size"]:
        bs = find_max_batch_size(exp, params["bs_start"], params["bs_max"], params["bs_safety"])
        with open_dict(cfg):
            cfg.training.batchsize = bs
        # the search ran optimizer steps -> rebuild a clean model/optimizer/scaler so
        # the lr sweep starts from a fresh init (as real training would)
        exp.init_model()
        exp._init_optimizer()
        exp._init_scaler()
        exp.model.to(exp.device)
        exp._init_dataloader()

    # name the plot/npz after the model + batchsize so repeated sweeps don't clobber each other
    model_name = (OmegaConf.select(cfg, "model.net._target_", default="") or "").rsplit(".", 1)[-1] or "model"
    if params["output"] == DEFAULTS["output"]:
        params["output"] = f"lr_finder_{model_name}_bs{cfg.training.batchsize}.png"

    LOGGER.info(
        f"Running LR range test: {params['start_lr']:.1e} -> {params['end_lr']:.1e} "
        f"over <= {params['num_iter']} batches (batchsize={cfg.training.batchsize})"
    )
 
    lrs, losses = range_test(
        exp,
        start_lr=params["start_lr"],
        end_lr=params["end_lr"],
        num_iter=params["num_iter"],
        beta=params["beta"],
        diverge=params["diverge"],
    )
    if len(lrs) < 3:
        LOGGER.error(
            "Not enough points collected before divergence; "
            "try a smaller start_lr, a larger diverge, or more num_iter."
        )
        return
 
    steepest, min_loss_lr, _, _ = suggest_lr(
        lrs,
        losses,
        skip_start=params["skip_start"],
        skip_end=params["skip_end"],
        beta=params["beta"],
    )
    make_plot(
        lrs, losses, steepest, min_loss_lr, params["output"],
        title=f"LR range test - {model_name} (bs={cfg.training.batchsize})",
    )
    np.savez(os.path.splitext(params["output"])[0] + ".npz", lr=lrs, loss=losses)

    # loss-min/10 is the robust recommendation (peak lr for an annealed schedule);
    # the steepest-descent point is reported as a usually-similar lower bound.
    suggested = min_loss_lr / 10.0
    bs = cfg.training.batchsize
    LOGGER.info("=" * 64)
    if params["find_batch_size"]:
        LOGGER.info(f"Batchsize (fit to GPU):          {bs}")
    LOGGER.info(f"Suggested lr (loss-min / 10):    {suggested:.2e}   [recommended]")
    LOGGER.info(f"Suggested lr (steepest descent): {steepest:.2e}")
    reuse = f"training.lr={suggested:.2e}"
    if params["find_batch_size"]:
        reuse = f"training.batchsize={bs} " + reuse
    LOGGER.info(f"  ->  reuse with:  {reuse}")
    LOGGER.info("=" * 64)
 
 
if __name__ == "__main__":
    main()
 
