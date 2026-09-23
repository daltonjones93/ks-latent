"""Stage-1/Stage-2 training loops (brief §5.1-5.2).

Follows the MPS engineering rules (brief §1.3): the whole training tensor is
moved to `device` once before the epoch loop, batches are drawn via an
index-shuffle over that on-device tensor (no `DataLoader`), and the epoch
loss is accumulated as a tensor with a single `.item()` call per epoch
rather than per step.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ks_latent.config import RegConfig, Stage1TrainingConfig, Stage2TrainingConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator, _polynomial_term_indices
from ks_latent.models.spectral_field import decode_from_spectrum, encode_to_spectrum
from ks_latent.training.data import gather_windows, index_shuffle_batches, make_window_index, roll_batch
from ks_latent.training.losses import (
    decorr_var_loss,
    horizon_weighted_latent_loss,
    horizon_weights,
    kernel_unstable_floor_loss,
    latent_self_spectrum_lowpass_loss,
    local_field_channel_mean_loss,
    logdet_barrier_loss,
    low_pass_spectral_loss,
    propagator_graded_spectrum_shape_loss,
    propagator_multistep_spectrum_shape_loss,
    propagator_local_expansion_floor_loss,
    propagator_spectrum_shape_loss,
    reconstruction_loss,
    reference_mode_energy,
    reference_temporal_separation,
    spatial_coherence_loss,
    spatial_energy_floor_loss,
    spectral_shape_floor_loss,
    temporal_expansion_floor_loss,
    temporal_smoothness_loss,
    variance_floor_loss,
)
from ks_latent.training.regularizer import LatentIndexPenalty


def pde_head_poly_coeffs(pde_head: AuxPropagator | LatentPropagator) -> torch.Tensor:
    """Returns the raw `(1, n_terms)` weight of a `pde_head`'s own
    `poly_coeffs: nn.Linear(n_terms, 1)` -- the SAME tensor whose entries
    are this project's directly-interpretable PDE coefficients (see
    `_SpectralPDEDeltaBody`'s docstring). Added 2026-09-10, user-directed:
    "I wonder if we could impose some sparsity using l1 norm on the
    polynomial coefficients of the pde" -- used by `w_pde_coeff_l1` in both
    `Stage1TrainingConfig`/`Stage2TrainingConfig` to penalize
    `weight.abs().sum()`.

    Handles both bodies that can carry a `pde_head`: `_SpectralPDEDeltaBody`
    (backbone="spectral_pde", `poly_coeffs` directly on `.body`) and
    `_SpectralPDERawDeltaBody` (backbone="spectral_pde_raw", `poly_coeffs`
    one level down at `.body.inner`, see that class's docstring). Raises
    (fail loudly, brief ground rule 2) if `pde_head` was not built with
    `field_kind="polynomial"` -- there is no coefficient vector to
    sparsify for `field_kind="mlp"`, so a nonzero `w_pde_coeff_l1` in that
    configuration is a real misconfiguration, not a no-op."""
    body = pde_head.body
    if hasattr(body, "poly_coeffs"):
        return body.poly_coeffs.weight
    if hasattr(body, "inner") and hasattr(body.inner, "poly_coeffs"):
        return body.inner.poly_coeffs.weight
    raise ValueError(
        "w_pde_coeff_l1 > 0 requires pde_head to have been built with "
        "field_kind='polynomial' (no poly_coeffs found on pde_head.body or "
        "pde_head.body.inner) -- got a pde_head whose body has no polynomial "
        "coefficient vector to sparsify."
    )


def pde_head_linear_coeffs(pde_head: AuxPropagator | LatentPropagator) -> torch.Tensor:
    """Same as `pde_head_poly_coeffs`, but returns only the entries for
    LINEAR (single-derivative-index) terms -- tuples of length 1 in
    `_polynomial_term_indices`'s own enumeration (`w`, `w_x`, `w_xx`, ...)
    -- EXCLUDING both the constant term (length 0) and every multi-factor
    product/power term (length >=2). Basis-invariant for these terms
    specifically: `T_1(x)=x` under `field_kind="chebyshev"` too, so a
    length-1 term means the exact same thing (the raw, undenormalized
    coefficient on that single derivative channel) regardless of
    `field_kind`.

    Added 2026-09-11, user-directed: "please try using the w_pde_coeff_l1
    machinery to penalize single index terms" -- motivated by this
    session's finding that every closure whose LARGEST-magnitude
    coefficient was one of these linear terms produced simple, near-
    sinusoidal-looking standalone rollouts (exact Fourier eigenmodes of a
    constant-coefficient linear operator on a periodic domain), while the
    two runs whose largest term was genuinely nonlinear (`w*w_x`,
    `w_x*w_xx`) produced visibly richer dynamics. Used by
    `w_pde_coeff_l1` when `cfg.pde_coeff_l1_linear_only=True` to penalize
    ONLY these coefficients, pressuring the fit toward explaining the
    dynamics via nonlinear coupling instead."""
    weight = pde_head_poly_coeffs(pde_head)  # (1, n_terms)
    body = pde_head.body
    inner = body.inner if hasattr(body, "inner") else body
    n_vars = inner.max_order + 1
    term_indices = _polynomial_term_indices(n_vars, inner.poly_degree, inner.poly_max_term_order)
    mask = torch.tensor(
        [len(idx) == 1 for idx in term_indices], device=weight.device, dtype=torch.bool
    )
    if not mask.any():
        raise ValueError(
            "pde_coeff_l1_linear_only=True requires at least one linear (single-index) "
            f"term to survive poly_max_term_order={inner.poly_max_term_order!r} filtering "
            f"at max_order={inner.max_order!r} -- none found, nothing to penalize."
        )
    return weight[:, mask]


def pde_head_nonlinear_output(pde_head: AuxPropagator | LatentPropagator, z: torch.Tensor) -> torch.Tensor:
    """`z`: `(N, d_latent)` -> `(N, d_latent)`, `pde_head.body`'s own
    `nonlinear_field(z)` -- the closure's output restricted to genuinely
    NONLINEAR (product, length>=2) terms only, with the constant and every
    single-derivative linear term masked to zero. Works uniformly for
    both `backbone="spectral_pde"` (`_SpectralPDEDeltaBody.nonlinear_field`
    directly) and `backbone="spectral_pde_raw"` (`_SpectralPDERawDeltaBody.
    nonlinear_field`, which internally self-FFTs `z` before delegating) --
    both expose the same `nonlinear_field` method name, so no `.inner`
    dispatch is needed here (contrast `pde_head_poly_coeffs`, which does
    need it since raw's `poly_coeffs` lives one level down).

    Added 2026-09-12, user-directed: "directly replicate iLED's
    stabilization mechanism" (arXiv:2309.05812) -- see `_SpectralPDEDeltaBody.
    nonlinear_field`'s docstring for the full motivation. Used by
    `w_pde_nonlinear_l2` to penalize `pde_head_nonlinear_output(pde_head,
    z).pow(2).mean()`, the analogue of iLED's `||Psi_1(z,h)||^2` term.
    Raises (fail loudly) if `pde_head` was not built with
    `field_kind="polynomial"`/`"chebyshev"` -- there is no linear/nonlinear
    split to isolate for `field_kind="mlp"`."""
    body = pde_head.body
    if hasattr(body, "nonlinear_field"):
        return body.nonlinear_field(z)
    raise ValueError(
        "w_pde_nonlinear_l2 > 0 requires pde_head to have been built with "
        "field_kind='polynomial'/'chebyshev' (no nonlinear_field found on pde_head.body) -- "
        f"got a pde_head whose body ({type(body).__name__}) has no nonlinear/linear split."
    )


def pde_head_field_output(pde_head: AuxPropagator | LatentPropagator, z: torch.Tensor) -> torch.Tensor:
    """`z`: `(N, d_latent)` -> `(N, d_latent)`, `pde_head.body`'s own FULL
    `field(z)` output -- every term, linear and nonlinear, unlike
    `pde_head_nonlinear_output`'s deliberately-masked subset. Works
    uniformly for both `backbone="spectral_pde"` (`_SpectralPDEDeltaBody.
    field`, always present) and `backbone="spectral_pde_raw"`
    (`_SpectralPDERawDeltaBody.field`, added 2026-09-12 alongside this
    helper) -- both expose the same `field` method name.

    Added 2026-09-12, user-directed: "can you recommend a fix to prevent
    drift toward mode-0" (Section 164's standalone rollout: broadband at
    t=0, 72-86% concentrated in the self-FFT DC/mean mode by t=54). True
    KS conserves `int u dx` EXACTLY -- every term in `-u*u_x - u_xx -
    u_xxxx` is a total x-derivative, which integrates to zero over a
    periodic domain. A freely-fit polynomial closure has no such
    guarantee: measured directly on Section 164's own trained pde_head,
    `field(z).mean(dim=-1)` (the predicted dz/dt's own spatial mean, i.e.
    its self-FFT DC component) across 500 real states was `-0.00031 +-
    0.0045` -- small but systematically nonzero, not centered at zero.
    Two concrete measured sources: the bias/intercept (0.0065, a flat
    additive offset with no counterpart in true KS) and the `w_x*w_x`
    term (coefficient -0.043, the largest non-total-derivative term in
    the fit) -- since `w_x^2 >= 0` pointwise everywhere, its contribution
    to the mean never oscillates out, unlike a genuine total-derivative
    term (e.g. `w_xx*w_xxx = d/dx(w_xx^2/2)`, which integrates to exactly
    zero and contributes nothing here). Used by `w_pde_mean_conservation`
    to penalize `pde_head_field_output(pde_head, z).mean(dim=-1).pow(2).
    mean()` directly, rather than hand-suppressing specific terms --
    mirrors true KS's own exact conservation law as a soft prior on the
    LEARNED closure's functional form."""
    body = pde_head.body
    if hasattr(body, "field"):
        return body.field(z)
    raise ValueError(
        "w_pde_mean_conservation > 0 requires pde_head to have been built with "
        "field_kind='polynomial'/'chebyshev'/'mlp' on a spectral_pde/spectral_pde_raw body "
        f"-- got a pde_head whose body ({type(body).__name__}) has no field() method."
    )


def warmup_cosine_lr_lambda(epochs: int, warmup_epochs: int, lr_min_factor: float):
    """Linear warmup over `warmup_epochs`, then cosine decay to a floor of
    `lr_min_factor` (not to zero), over `epochs` total (added 2026-08-29,
    ported from a reference implementation's `_lr_lambda`; see
    `Stage1TrainingConfig`'s docstring for why). One call of the returned
    function per *epoch* (this codebase steps its scheduler once per epoch,
    not per optimizer step, unlike the per-step reference)."""
    warmup_epochs = max(0, warmup_epochs)

    def f(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        denom = max(1, epochs - warmup_epochs)
        prog = min((epoch - warmup_epochs) / denom, 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return lr_min_factor + (1 - lr_min_factor) * cos

    return f


@dataclass
class Stage1Result:
    train_history: list[dict]
    val_recon_final: float


def eval_stage1_reconstruction(
    ae: KSAutoencoderPatched, trajectories: torch.Tensor, window: int, device: torch.device
) -> float:
    """Mean reconstruction MSE over every window in `trajectories` (no
    augmentation, deterministic), for the §18 "Stage-1 val reconstruction"
    target."""
    ae.eval()
    trajectories = trajectories.to(device)
    n_runs, T, NX = trajectories.shape
    index = make_window_index(n_runs, T, window, device=device)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_idx in index_shuffle_batches(index.shape[0], 512):
            idx = index[batch_idx]
            windows = gather_windows(trajectories, idx, window)
            u_flat = windows.reshape(-1, NX)
            u_hat, _ = ae(u_flat)
            total += reconstruction_loss(u_hat, u_flat).item() * u_flat.shape[0]
            count += u_flat.shape[0]
    ae.train()
    return total / count


def linear_decay(epoch: int, epochs: int, start: float, end: float) -> float:
    """Linearly interpolate from `start` (epoch 0) to `end` (epoch
    `epochs - 1`), held at `end` for any epoch beyond that (shouldn't
    occur in practice, but keeps the function total). `epochs <= 1`
    returns `end` outright -- there is no "epoch 0" vs. "final epoch" to
    interpolate between. Used for Stage 1's optional
    `w_var_end`/`w_logdet_end`/`w_spatial_end` regularizer-weight decay
    (`Stage1TrainingConfig`'s docstring)."""
    if epochs <= 1:
        return end
    frac = min(epoch / (epochs - 1), 1.0)
    return start + frac * (end - start)


def train_stage1(
    ae: KSAutoencoderPatched,
    aux: AuxPropagator,
    train_trajectories: torch.Tensor,
    val_trajectories: torch.Tensor,
    cfg: Stage1TrainingConfig,
    device: torch.device,
    reg_cfg: RegConfig | None = None,
    verbose: bool = False,
    log_every: int = 1,
    checkpoint_every: int = 0,
    on_epoch_end: Callable[[int], None] | None = None,
    amp: bool = False,
    pde_head: AuxPropagator | None = None,
) -> Stage1Result:
    """`train_trajectories`/`val_trajectories`: `(n_runs, T, NX)`, already
    normalized (brief §3.3).

    `checkpoint_every`/`on_epoch_end` (added 2026-08-31, user-directed
    after an unattended run was lost to an unexpected machine crash with
    no way to recover it -- "please save a checkpoint every 10 epochs"):
    `checkpoint_every=0` (default) is off, unchanged behavior. When set,
    `on_epoch_end(epoch)` is called after every `checkpoint_every`-th
    completed epoch (`epoch` 0-indexed, called after epochs
    `checkpoint_every-1, 2*checkpoint_every-1, ...`). This function does
    not know how to write a checkpoint file itself (it has no `ae_cfg`/
    `aux_cfg`/output-path knowledge, all of which live in the calling
    script) -- `on_epoch_end` is a closure the caller builds that saves
    whatever it wants using the SAME `ae`/`aux` objects passed into this
    function (mutated in place during training, so the closure needs no
    arguments beyond the epoch number).

    `verbose` (added 2026-08-29): print one line per `log_every` epochs
    (epoch, loss, recon, k_now, wall time) -- see `train_stage2`'s
    docstring for why this was added.

    Window size and the `L_pred` rollout both depend on `aux.mode` (brief
    §5.2 addendum): `"two_step"` needs 2 history snapshots + `k_pred`
    future targets; `"markovian"` needs only 1 history snapshot (the
    two-step history is exactly what it's testing whether we can drop), so
    `L_recon`'s snapshot count shrinks with it -- both modes flow through
    the same `aux.rollout(...)` call, `LatentPropagator.rollout` already
    ignores `z_prev` internally in `"markovian"` mode. `"history"` (added
    2026-08-29, user-directed) needs `aux.cfg.n_history` snapshots and
    flows through `aux.rollout_history(...)` instead -- see
    `PropagatorConfig`'s docstring.

    `cfg.k_pred_max > cfg.k_pred` (added 2026-08-29, default off) ramps the
    `L_pred` rollout length from `cfg.k_pred` to `cfg.k_pred_max` over
    `cfg.k_pred_warmup_epochs` epochs -- see `Stage1TrainingConfig`'s
    docstring. The training window is sized for the largest rollout the
    curriculum will ever reach (`n_history + max(k_pred, k_pred_max)`),
    even on early epochs using a shorter `k_now`.

    `reg_cfg` (added 2026-08-29, default `None` = off): optional banded
    latent-index-smoothness + off-band decorrelation penalty, see
    `RegConfig`'s docstring. When given and active, it is applied to every
    encoded latent in the window and, if `reg_cfg.apply_to_propagated`, to
    `aux`'s own propagated outputs too -- otherwise those are unconstrained
    and inherit index structure only implicitly, from being fit to targets
    that have it (see the ported reference project's `train.py` docstring
    for the same argument made about its own propagator).

    `amp` (added 2026-09-01, user-directed: "we should be training in
    bfloat16 or float16, that will be way more efficient"): wraps each
    batch's forward pass + loss computation (everything from the
    windows/rollout through the final `loss` value, NOT the
    backward/optimizer step) in `torch.autocast(device_type=device.type,
    dtype=torch.bfloat16)`. `bfloat16` specifically (not `float16`):
    same exponent range as `float32` (unlike `float16`'s narrower range),
    so it needs no `GradScaler`/loss-scaling machinery to avoid gradient
    underflow -- confirmed working cleanly on this project's actual MPS
    device (`torch==2.13.0`: `Conv1d`, `scaled_dot_product_attention`,
    and plain `Linear` layers all run and backward correctly under MPS
    `bfloat16` autocast). Backward/`opt.step()` stay outside the
    `autocast` context (the standard idiom) -- gradients and optimizer
    state remain `float32` regardless of `amp`, only the forward
    activations use the lower-precision dtype. Default `False` (off,
    unchanged behavior) since this changes numerics for any existing
    reproducibility comparisons; opt in explicitly per run.

    `pde_head` (added 2026-09-08, user-directed: "joint training seems
    pretty smart. can you implement this idea" -- see
    docs/sine_transform_pde_plan.md's "hybrid" discussion): an optional
    SECOND propagator, always `backbone="spectral_pde"` (the constrained,
    interpretable local-derivative-MLP form), trained ALONGSIDE `aux`
    (which stays whatever free/unconstrained backbone -- e.g. `"mlp"` --
    is actually driving `l_pred`) via a `cfg.w_pde_distill`-weighted
    single-step DISTILLATION loss, not by asking it to produce its own
    multi-step rollout. Concretely: `g_target = z_pred[:, 0]` (the
    already-computed FIRST step of `aux`'s own rollout -- no extra forward
    pass through `aux`) vs. `pde_head.step_one(z_start)` (a single,
    ordinary forward pass -- `pde_head` is never asked to integrate its
    own multi-step rollout under gradient pressure, which is exactly what
    made backbone="spectral_pde" so hard to train as the PRIMARY
    propagator: see this session's Section 101-106 findings). This is
    deliberately NOT the earlier, already-failed `fit_latent_pde.py`
    SINDy approach (post-hoc regression on a FROZEN checkpoint,
    `R^2~0.005`, collapsed to `D_KY=0` -- see the plan doc's §4.2): here
    the encoder and `pde_head` co-adapt jointly, from the start,
    specifically pressured toward local-PDE-fittability, rather than a
    fixed representation optimized for something else being fit after the
    fact. `None` (default) is off, unchanged behavior -- `cfg.w_pde_distill`
    also gates whether the term is added even if `pde_head` is given,
    matching every other regularizer's own convention in this loop.

    `cfg.pde_distill_detach_target` (added 2026-09-08, user-directed: "I
    think the loss should be mutual for stage 1 training too. We always
    want the propagator to have dynamics that can be easily modeled by
    the pde_head right?"): `True` (default) detaches `g_target` -- no
    gradient flows back into `aux` from this term, so it stays free to fit
    the real dynamics however it wants, uncorrupted by pressure to look
    local; gradient reaches only `pde_head`'s own parameters (it learns to
    imitate `aux`'s realized one-step dynamics) AND, through `p_pred`'s
    dependence on `z_start`, the ENCODER (pressuring it toward a
    representation the local-derivative-MLP form can actually fit).
    `False` makes it MUTUAL -- `g_target` stays attached, so gradient ALSO
    reaches `aux` directly, genuinely pressuring the propagator itself
    (not just the encoder) toward dynamics `pde_head` can describe. Real
    risk, named directly: this is exactly the "pressure toward simplicity"
    mechanism behind this project's own H-PROP finding (every propagator
    pressured to look simple/local has collapsed to a fixed point) -- if
    `pde_head` is weak or lazily fit (plausible early in training, near-
    identity at `zero_init`), a mutual loss could pull `aux` toward
    matching that impoverished target instead of real dynamics. Monitor
    `l_pred`/Gate 3 `D_KY` specifically when this is `False`.
    """
    ae.to(device)
    aux.to(device)
    if pde_head is not None:
        pde_head.to(device)
    train_trajectories = train_trajectories.to(device)
    n_runs, T, NX = train_trajectories.shape
    if aux.mode == "history":
        n_history = aux.cfg.n_history
    elif aux.mode == "two_step":
        n_history = 2
    else:
        n_history = 1
    k_pred_cap = max(cfg.k_pred, cfg.k_pred_max)
    ramp_active = cfg.k_pred_max > cfg.k_pred
    window = n_history + k_pred_cap
    if cfg.w_pde_distill_real_rollout > 0 and cfg.pde_distill_real_rollout_k > k_pred_cap:
        raise ValueError(
            f"pde_distill_real_rollout_k={cfg.pde_distill_real_rollout_k!r} exceeds "
            f"max(k_pred, k_pred_max)={k_pred_cap!r} -- the training window has no real "
            "future states beyond that to supply as rollout targets."
        )
    window_index = make_window_index(n_runs, T, window, device=device)

    d_latent = ae.cfg.d_latent
    reg = LatentIndexPenalty(reg_cfg, d_latent).to(device) if reg_cfg is not None else None
    reg_active = reg is not None and reg.active

    p_ref = None
    if cfg.w_shape_floor > 0:
        # Precomputed ONCE from real training data (encoder_kind="spectral_field"
        # only, duck-typed via ae.cfg.K -- same convention as w_lowpass's
        # generic ae.cfg.K/.L read above) -- a fixed, non-learned reference,
        # not recomputed per batch. See spectral_shape_floor_loss's docstring.
        p_ref = reference_mode_energy(train_trajectories.reshape(-1, NX), ae.cfg.K).to(device)

    z_temporal_floor_ref = None
    w_temporal_floor_ref = None
    if cfg.w_temporal_floor_z > 0 or cfg.w_temporal_floor_w > 0:
        # Precomputed ONCE from real training data via the SAME fixed
        # transform used everywhere else -- entirely independent of the
        # current encoder (see reference_temporal_separation's docstring).
        # encoder_kind="spectral_field" only (duck-typed via ae.cfg.K/.N_w).
        z_true_all = encode_to_spectrum(train_trajectories, ae.cfg.K)  # (n_runs, T, 2K)
        if cfg.w_temporal_floor_z > 0:
            z_temporal_floor_ref = reference_temporal_separation(
                z_true_all, cfg.temporal_floor_lag
            ).to(device)
        if cfg.w_temporal_floor_w > 0:
            w_true_all = decode_from_spectrum(z_true_all, ae.cfg.K, ae.cfg.N_w)  # (n_runs, T, N_w)
            w_temporal_floor_ref = reference_temporal_separation(
                w_true_all, cfg.temporal_floor_lag
            ).to(device)

    spectrum_shape_graded_ref = None
    if cfg.w_spectrum_shape_graded > 0:
        # Loaded ONCE from a file, not recomputed -- see
        # Stage1TrainingConfig.w_spectrum_shape_graded's docstring.
        # Independent of the current model entirely (unlike the
        # encoder-anchored floors above).
        spectrum_shape_graded_ref = torch.from_numpy(
            np.load(cfg.spectrum_shape_graded_reference_path)
        ).float().to(device)

    params = list(ae.parameters()) + list(aux.parameters())
    if pde_head is not None:
        params = params + list(pde_head.parameters())
    # `stable_linear_lr_factor` (Section 174): if `aux` exposes any
    # `poly_stable_linear_terms` raw parameters, give them their own much
    # smaller LR param group instead of the shared `cfg.lr` -- see
    # `Stage1TrainingConfig.stable_linear_lr_factor`'s docstring.
    stable_params = aux.stable_linear_raw_parameters() if hasattr(aux, "stable_linear_raw_parameters") else []
    if stable_params:
        stable_ids = {id(p) for p in stable_params}
        other_params = [p for p in params if id(p) not in stable_ids]
        opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg.lr},
                {"params": stable_params, "lr": cfg.lr * cfg.stable_linear_lr_factor},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lr_lambda(cfg.epochs, cfg.warmup_epochs, cfg.lr_min_factor)
    )

    history = []
    for epoch in range(cfg.epochs):
        epoch_t0 = time.time()
        k_now = (
            k_curriculum(epoch, cfg.k_pred, cfg.k_pred_max, cfg.k_pred_warmup_epochs)
            if ramp_active
            else cfg.k_pred
        )
        w_var_now = (
            linear_decay(epoch, cfg.epochs, cfg.w_var, cfg.w_var_end)
            if cfg.w_var_end is not None
            else cfg.w_var
        )
        w_logdet_now = (
            linear_decay(epoch, cfg.epochs, cfg.w_logdet, cfg.w_logdet_end)
            if cfg.w_logdet_end is not None
            else cfg.w_logdet
        )
        w_spatial_now = (
            linear_decay(epoch, cfg.epochs, cfg.w_spatial, cfg.w_spatial_end)
            if cfg.w_spatial_end is not None
            else cfg.w_spatial
        )
        if cfg.physics_prior_correction_warmup_epochs > 0 and hasattr(aux.body, "correction_scale"):
            # See Stage1TrainingConfig.physics_prior_correction_warmup_epochs's
            # docstring -- homotopy schedule between "exactly the true KS
            # equation" (0.0) and "full learned correction" (1.0).
            aux.body.correction_scale = linear_decay(
                epoch, cfg.physics_prior_correction_warmup_epochs, 0.0, 1.0
            )
        w_pred_now = (
            linear_decay(epoch, cfg.w_pred_warmup_epochs, 0.0, cfg.w_pred)
            if cfg.w_pred_warmup_epochs > 0
            else cfg.w_pred
        )
        k_roll_now = (
            k_curriculum(epoch, 1, cfg.pde_distill_real_rollout_k, cfg.pde_distill_real_rollout_warmup_epochs)
            if cfg.w_pde_distill_real_rollout > 0
            else 0
        )
        k_energy_now = (
            k_curriculum(epoch, 1, cfg.pde_energy_floor_rollout_k, cfg.pde_energy_floor_warmup_epochs)
            if pde_head is not None and cfg.w_pde_energy_floor > 0
            else 0
        )
        k_prop_energy_now = (
            k_curriculum(epoch, 1, cfg.prop_energy_floor_rollout_k, cfg.prop_energy_floor_warmup_epochs)
            if cfg.w_prop_energy_floor > 0
            else 0
        )
        batches = index_shuffle_batches(window_index.shape[0], cfg.batch_size)
        epoch_loss = torch.zeros((), device=device)
        epoch_recon = torch.zeros((), device=device)
        epoch_pde_distill = torch.zeros((), device=device)
        epoch_pde_distill_real = torch.zeros((), device=device)
        epoch_pde_distill_real_rollout = torch.zeros((), device=device)
        epoch_pde_nonlinear = torch.zeros((), device=device)
        epoch_pde_mean = torch.zeros((), device=device)
        epoch_pde_energy = torch.zeros((), device=device)
        epoch_prop_energy = torch.zeros((), device=device)
        epoch_pde_coeff_l1 = torch.zeros((), device=device)
        n_batches = 0
        local_expansion_floor_applied_this_epoch = False
        spectrum_shape_applied_this_epoch = False
        spectrum_shape_graded_applied_this_epoch = False
        spectrum_shape_multistep_applied_this_epoch = False
        pde_spectrum_shape_applied_this_epoch = False
        pde_spectrum_shape_multistep_applied_this_epoch = False
        pde_spectrum_shape_self_applied_this_epoch = False
        pde_spectrum_shape_multistep_self_applied_this_epoch = False
        for batch_idx in batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                idx = window_index[batch_idx]
                windows = gather_windows(train_trajectories, idx, window)  # (b, window, NX)
                b = windows.shape[0]
                shifts = torch.randint(0, NX, (b,), device=device)
                windows = roll_batch(windows, shifts)

                u_flat = windows.reshape(b * window, NX)
                z = ae.encode(u_flat)
                if cfg.noise_std > 0.0:
                    # Reconstruction path only: L_recon = MSE(D(E(x+eps)), x).
                    # Everything downstream of `z` (the prediction rollout, both
                    # regularizer terms) uses the *clean* encoding, so their
                    # numbers mean the same thing at every noise level.
                    z_noisy = ae.encode(u_flat + cfg.noise_std * torch.randn_like(u_flat))
                    u_hat = ae.decode(z_noisy)
                else:
                    u_hat = ae.decode(z)
                l_recon = reconstruction_loss(u_hat, u_flat)

                z_win = z.view(b, window, -1)
                u_targets = windows[:, n_history : n_history + k_now]

                if aux.mode == "history":
                    z_hist_in = z_win[:, :n_history]  # (b, n_history, d)
                    z_pred = aux.rollout_history(z_hist_in, k_now)  # (b, k_now, d)
                else:
                    if aux.mode == "two_step":
                        z_prev_in, z_curr_in = z_win[:, 0], z_win[:, 1]
                    else:
                        z_prev_in = z_curr_in = z_win[:, 0]  # z_prev_in ignored by aux.rollout
                    z_pred = aux.rollout(z_prev_in, z_curr_in, k_now)  # (b, k_now, d)
                pred_losses = [
                    reconstruction_loss(ae.decode(z_pred[:, k]), u_targets[:, k])
                    for k in range(k_now)
                ]
                l_pred = sum(pred_losses) / len(pred_losses)

                l_decorr, l_var = decorr_var_loss(z)
                loss = (
                    cfg.w_recon * l_recon
                    + w_pred_now * l_pred
                    + cfg.w_decorr * l_decorr
                    + w_var_now * l_var
                )
                if w_spatial_now > 0:
                    # See ks_latent.training.losses.spatial_coherence_loss's
                    # docstring for the RegConfig.lambda_z precedent/risk this
                    # carries -- off by default (cfg.w_spatial == 0).
                    l_spatial = spatial_coherence_loss(
                        z, bandwidth=cfg.spatial_bandwidth, signed=cfg.spatial_signed
                    )
                    loss = loss + w_spatial_now * l_spatial
                if cfg.w_prop_energy_floor > 0:
                    # See Stage1TrainingConfig.w_prop_energy_floor's
                    # docstring -- UNSUPERVISED (no real target beyond the
                    # starting state), applied to aux ITSELF (gradient
                    # reaches aux's own parameters fully; the starting
                    # state is detached so the encoder is not pulled by
                    # this term). k_prop_energy_now is the per-epoch
                    # RAMPED horizon (prop_energy_floor_warmup_epochs's
                    # docstring explains why the ramp is not optional).
                    if aux.mode == "history":
                        e_hist_prop = z_hist_in.detach()
                        z_prop_energy_roll = aux.rollout_history(e_hist_prop, k_prop_energy_now)
                    else:
                        e_start_prop = z_curr_in.detach()
                        z_prop_energy_roll = aux.rollout(e_start_prop, e_start_prop, k_prop_energy_now)
                    if torch.isfinite(z_prop_energy_roll).all():
                        l_prop_energy_floor = spatial_energy_floor_loss(
                            z_prop_energy_roll.reshape(-1, z_prop_energy_roll.shape[-1]),
                            cfg.prop_energy_floor_gamma,
                        )
                        loss = loss + cfg.w_prop_energy_floor * l_prop_energy_floor
                    else:
                        # Same reasoning as w_pde_energy_floor's identical
                        # guard (Section 170's own postmortem) -- skip this
                        # batch's contribution rather than let a transient
                        # explosion poison the whole loss.
                        print(
                            f"  [stage1] WARNING: aux energy-floor rollout produced "
                            f"non-finite values at k_prop_energy_now={k_prop_energy_now} "
                            f"(epoch {epoch}) -- skipping this batch's prop-energy-floor "
                            f"contribution.",
                            flush=True,
                        )
                        l_prop_energy_floor = torch.zeros((), device=z_prop_energy_roll.device)
                if cfg.w_kernel_unstable_floor > 0:
                    # See Stage1TrainingConfig.w_kernel_unstable_floor's
                    # docstring -- a pure function of aux's own kernel
                    # parameters (A/width/nu/mu), NOT of this batch's data
                    # at all (kernel_Lhat() takes no arguments), so this
                    # runs every batch at negligible cost regardless.
                    # Only has an effect if aux's body exposes kernel_Lhat
                    # (field_kind="forced_burgers" with
                    # burgers_kernel_instability=True) -- silently a no-op
                    # otherwise (nothing to regularize).
                    Lhat = aux.kernel_Lhat()
                    if Lhat is not None:
                        l_kernel_unstable = kernel_unstable_floor_loss(
                            Lhat, cfg.kernel_unstable_target_modes, cfg.kernel_unstable_margin
                        )
                        loss = loss + cfg.w_kernel_unstable_floor * l_kernel_unstable
                if pde_head is not None and cfg.w_pde_distill > 0:
                    # See train_stage1's own docstring for pde_head's full
                    # rationale. z_start: the same starting state aux's own
                    # rollout used to produce z_pred[:, 0] -- z_hist_in's
                    # newest snapshot in "history" mode, z_curr_in otherwise.
                    z_start = z_hist_in[:, -1] if aux.mode == "history" else z_curr_in
                    g_target = z_pred[:, 0]
                    if cfg.pde_distill_detach_target:
                        g_target = g_target.detach()
                    p_pred = pde_head.step_one(z_start)
                    l_pde_distill = reconstruction_loss(p_pred, g_target)
                    loss = loss + cfg.w_pde_distill * l_pde_distill
                if pde_head is not None and cfg.w_pde_distill_real > 0:
                    # See Stage1TrainingConfig.w_pde_distill_real's
                    # docstring -- every consecutive REAL pair already in
                    # z_win (no new forward pass), independent of aux's
                    # own predictions entirely.
                    real_start = z_win[:, :-1].reshape(-1, z_win.shape[-1])
                    real_next = z_win[:, 1:].reshape(-1, z_win.shape[-1])
                    if cfg.pde_distill_real_detach:
                        real_start = real_start.detach()
                        real_next = real_next.detach()
                    p_pred_real = pde_head.step_one(real_start)
                    l_pde_distill_real = reconstruction_loss(p_pred_real, real_next)
                    loss = loss + cfg.w_pde_distill_real * l_pde_distill_real
                if pde_head is not None and cfg.w_pde_distill_real_rollout > 0:
                    # See Stage1TrainingConfig.w_pde_distill_real_rollout's
                    # docstring (iLED, arXiv:2309.05812) -- pde_head's OWN
                    # chained rollout from a REAL starting state, compared
                    # at every step to REAL future states already in
                    # z_win (never aux/propagator's own predictions).
                    # k_roll_now (per-epoch, ramped -- see
                    # pde_distill_real_rollout_warmup_epochs's docstring)
                    # NOT the raw config target -- applying the full
                    # target horizon from epoch 0 against an undertrained
                    # pde_head can diverge on the very first batch.
                    k_roll = k_roll_now
                    r_start = z_win[:, n_history - 1]
                    if cfg.pde_distill_real_detach:
                        r_start = r_start.detach()
                    p_pred_rollout = pde_head.rollout(r_start, r_start, k_roll)  # (b, k_roll, d)
                    r_targets = z_win[:, n_history : n_history + k_roll]
                    if cfg.pde_distill_real_detach:
                        r_targets = r_targets.detach()
                    l_pde_distill_real_rollout = ((p_pred_rollout - r_targets) ** 2).mean()
                    loss = loss + cfg.w_pde_distill_real_rollout * l_pde_distill_real_rollout
                if pde_head is not None and cfg.w_pde_nonlinear_l2 > 0:
                    # See Stage1TrainingConfig.w_pde_nonlinear_l2's
                    # docstring (iLED, arXiv:2309.05812) -- a prior on
                    # pde_head's OWN nonlinear closure alone, evaluated on
                    # real detached states, never mutual with the encoder.
                    z_for_nonlinear = z_win.reshape(-1, z_win.shape[-1]).detach()
                    l_pde_nonlinear = pde_head_nonlinear_output(pde_head, z_for_nonlinear).pow(2).mean()
                    loss = loss + cfg.w_pde_nonlinear_l2 * l_pde_nonlinear
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape > 0
                    and not pde_spectrum_shape_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape's
                    # docstring -- pde_head analogue of w_spectrum_shape,
                    # same expensive/once-per-epoch convention, evaluated
                    # on real detached states (never mutual with the
                    # encoder), pde_head.step_one only (never the main aux
                    # propagator).
                    pde_spectrum_shape_applied_this_epoch = True
                    z_pool_pde = z_win.reshape(-1, z_win.shape[-1]).detach()
                    n_sample = min(cfg.pde_spectrum_shape_n_samples, z_pool_pde.shape[0])
                    sample_idx = torch.randperm(z_pool_pde.shape[0], device=device)[:n_sample]
                    z_sample_pde = z_pool_pde[sample_idx].float()
                    l_pde_spectrum_shape = propagator_spectrum_shape_loss(
                        pde_head.step_one, z_sample_pde,
                        n_expand=cfg.pde_spectrum_shape_n_expand,
                        expand_target=cfg.pde_spectrum_shape_expand_target,
                        contract_floor=cfg.pde_spectrum_shape_contract_floor,
                        two_sided=cfg.pde_spectrum_shape_two_sided,
                    )
                    loss = loss + cfg.w_pde_spectrum_shape * l_pde_spectrum_shape
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_multistep > 0
                    and not pde_spectrum_shape_multistep_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape_multistep's
                    # docstring -- pde_head analogue of
                    # w_spectrum_shape_multistep (Section 186), same
                    # composed-k-step-Jacobian mechanism, pde_head.step_one
                    # only.
                    pde_spectrum_shape_multistep_applied_this_epoch = True
                    z_pool_pde = z_win.reshape(-1, z_win.shape[-1]).detach()
                    n_sample = min(cfg.pde_spectrum_shape_multistep_n_samples, z_pool_pde.shape[0])
                    sample_idx = torch.randperm(z_pool_pde.shape[0], device=device)[:n_sample]
                    z_sample_pde = z_pool_pde[sample_idx].float()
                    l_pde_spectrum_shape_multistep = propagator_multistep_spectrum_shape_loss(
                        pde_head.step_one, z_sample_pde,
                        k=cfg.pde_spectrum_shape_multistep_k,
                        n_expand=cfg.pde_spectrum_shape_n_expand,
                        expand_target=cfg.pde_spectrum_shape_expand_target,
                        contract_floor=cfg.pde_spectrum_shape_contract_floor,
                    )
                    loss = loss + cfg.w_pde_spectrum_shape_multistep * l_pde_spectrum_shape_multistep
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_self > 0
                    and not pde_spectrum_shape_self_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape_self's
                    # docstring (Section 193) -- SELF-rollout-sampled
                    # analogue of w_pde_spectrum_shape: evaluated at states
                    # pde_head's OWN free rollout actually visits, not real
                    # data.
                    pde_spectrum_shape_self_applied_this_epoch = True
                    z0_pool = z_win[:, 0].reshape(-1, z_win.shape[-1]).detach()
                    n_sample = min(cfg.pde_spectrum_shape_self_n_samples, z0_pool.shape[0])
                    sample_idx = torch.randperm(z0_pool.shape[0], device=device)[:n_sample]
                    z0_self = z0_pool[sample_idx].float()
                    z_self_states = _pde_head_self_rollout_states(
                        pde_head, z0_self, cfg.pde_spectrum_shape_self_rollout_k
                    )
                    if z_self_states is not None:
                        l_pde_spectrum_shape_self = propagator_spectrum_shape_loss(
                            pde_head.step_one, z_self_states,
                            n_expand=cfg.pde_spectrum_shape_n_expand,
                            expand_target=cfg.pde_spectrum_shape_expand_target,
                            contract_floor=cfg.pde_spectrum_shape_contract_floor,
                            two_sided=cfg.pde_spectrum_shape_two_sided,
                        )
                        loss = loss + cfg.w_pde_spectrum_shape_self * l_pde_spectrum_shape_self
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_multistep_self > 0
                    and not pde_spectrum_shape_multistep_self_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape_
                    # multistep_self's docstring (Section 193) -- SELF-
                    # rollout-sampled analogue of
                    # w_pde_spectrum_shape_multistep.
                    pde_spectrum_shape_multistep_self_applied_this_epoch = True
                    z0_pool = z_win[:, 0].reshape(-1, z_win.shape[-1]).detach()
                    n_sample = min(cfg.pde_spectrum_shape_self_n_samples, z0_pool.shape[0])
                    sample_idx = torch.randperm(z0_pool.shape[0], device=device)[:n_sample]
                    z0_self = z0_pool[sample_idx].float()
                    z_self_states = _pde_head_self_rollout_states(
                        pde_head, z0_self, cfg.pde_spectrum_shape_self_rollout_k
                    )
                    if z_self_states is not None:
                        l_pde_spectrum_shape_multistep_self = propagator_multistep_spectrum_shape_loss(
                            pde_head.step_one, z_self_states,
                            k=cfg.pde_spectrum_shape_multistep_k,
                            n_expand=cfg.pde_spectrum_shape_n_expand,
                            expand_target=cfg.pde_spectrum_shape_expand_target,
                            contract_floor=cfg.pde_spectrum_shape_contract_floor,
                        )
                        loss = loss + (
                            cfg.w_pde_spectrum_shape_multistep_self * l_pde_spectrum_shape_multistep_self
                        )
                if pde_head is not None and cfg.w_pde_mean_conservation > 0:
                    # See Stage1TrainingConfig.w_pde_mean_conservation's
                    # docstring -- mirrors true KS's exact int(u)dx
                    # conservation law, evaluated on real detached states,
                    # never mutual with the encoder.
                    z_for_mean = z_win.reshape(-1, z_win.shape[-1]).detach()
                    l_pde_mean = pde_head_field_output(pde_head, z_for_mean).mean(dim=-1).pow(2).mean()
                    loss = loss + cfg.w_pde_mean_conservation * l_pde_mean
                if pde_head is not None and cfg.w_pde_energy_floor > 0:
                    # See Stage1TrainingConfig.w_pde_energy_floor's
                    # docstring -- UNSUPERVISED (no real target beyond
                    # the starting state), detached from the encoder,
                    # k_energy_now is the per-epoch RAMPED horizon (see
                    # pde_energy_floor_warmup_epochs's docstring for why
                    # the ramp is not optional).
                    e_start = z_win[:, n_history - 1].detach()
                    z_energy_roll = pde_head.rollout(e_start, e_start, k_energy_now)  # (b, k_energy, d)
                    if torch.isfinite(z_energy_roll).all():
                        l_pde_energy_floor = spatial_energy_floor_loss(
                            z_energy_roll.reshape(-1, z_energy_roll.shape[-1]), cfg.pde_energy_floor_gamma
                        )
                        loss = loss + cfg.w_pde_energy_floor * l_pde_energy_floor
                    else:
                        # A partially-converged pde_head's own unsupervised
                        # rollout can transiently diverge within k_energy_now
                        # steps even after the curriculum ramp fully engages
                        # -- grad clipping only protects against large-but-
                        # FINITE gradients, not a forward pass that's already
                        # inf/nan (found empirically, Section 170's own
                        # postmortem: corrupted ~100 epochs silently before
                        # becoming visible in the once-per-10-epoch log).
                        # Skip this batch's contribution entirely (zero, not
                        # connected to the exploded rollout) rather than let
                        # it poison the whole loss and permanently corrupt
                        # the optimizer state. Frequent occurrence is itself
                        # informative (signals k_energy_now/gamma/weight are
                        # mismatched to how well-conditioned pde_head is at
                        # that point in training) -- printed every time
                        # rather than silently, since it should be rare.
                        print(
                            f"  [stage1] WARNING: pde_head energy-floor rollout produced "
                            f"non-finite values at k_energy_now={k_energy_now} (epoch {epoch}) "
                            f"-- skipping this batch's energy-floor contribution.",
                            flush=True,
                        )
                        l_pde_energy_floor = torch.zeros((), device=z_energy_roll.device)
                if pde_head is not None and cfg.w_pde_coeff_l1 > 0:
                    # See Stage1TrainingConfig.w_pde_coeff_l1's/
                    # pde_coeff_l1_linear_only's docstrings.
                    if cfg.pde_coeff_l1_linear_only:
                        l_pde_coeff_l1 = pde_head_linear_coeffs(pde_head).abs().sum()
                    else:
                        l_pde_coeff_l1 = pde_head_poly_coeffs(pde_head).abs().sum()
                    loss = loss + cfg.w_pde_coeff_l1 * l_pde_coeff_l1
                if cfg.w_var_floor > 0:
                    l_var_floor = variance_floor_loss(z, gamma=cfg.var_floor_gamma)
                    loss = loss + cfg.w_var_floor * l_var_floor
                if w_logdet_now > 0:
                    l_logdet = logdet_barrier_loss(z, eps=cfg.logdet_eps)
                    loss = loss + w_logdet_now * l_logdet
                if cfg.w_logdet_physical > 0:
                    # See Stage1TrainingConfig.w_logdet_physical's docstring --
                    # same logdet_barrier_loss mechanism as w_logdet above, but
                    # applied to the PHYSICAL field decode_from_spectrum(z)
                    # reconstructs, not z (the rFFT coefficients) directly.
                    # encoder_kind="spectral_field" only (ae.cfg.K/.N_w duck-typed,
                    # same convention as w_lowpass's ae.cfg.K/.L read).
                    w_states = decode_from_spectrum(z, ae.cfg.K, ae.cfg.N_w)
                    l_logdet_physical = logdet_barrier_loss(w_states, eps=cfg.logdet_physical_eps)
                    loss = loss + cfg.w_logdet_physical * l_logdet_physical
                if cfg.w_smooth > 0:
                    l_smooth = temporal_smoothness_loss(z_win, curvature_weight=cfg.smooth_curvature_weight)
                    loss = loss + cfg.w_smooth * l_smooth
                if cfg.w_lowpass > 0:
                    # encoder_kind="spectral_field" only -- ae.cfg.K/.L are
                    # only present on SpectralFieldAutoencoderConfig, same
                    # duck-typed generic-attribute-read convention as
                    # `d_latent = ae.cfg.d_latent` above.
                    l_lowpass = low_pass_spectral_loss(z, ae.cfg.K, ae.cfg.L, power=cfg.lowpass_power)
                    loss = loss + cfg.w_lowpass * l_lowpass
                if cfg.w_lowpass_rollout > 0:
                    # See Stage1TrainingConfig.w_lowpass_rollout's docstring --
                    # unlike w_lowpass (encoder's own z), this targets the
                    # AUX PROPAGATOR's own short rolled-out z_pred (already
                    # computed above whenever --full-propagator is active).
                    l_lowpass_rollout = low_pass_spectral_loss(
                        z_pred.reshape(-1, z_pred.shape[-1]), ae.cfg.K, ae.cfg.L,
                        power=cfg.lowpass_rollout_power,
                    )
                    loss = loss + cfg.w_lowpass_rollout * l_lowpass_rollout
                if cfg.w_z_lowpass > 0:
                    # See Stage1TrainingConfig.w_z_lowpass's docstring --
                    # ANY encoder's raw z (unlike w_lowpass above, which
                    # requires encoder_kind="spectral_field").
                    zK = cfg.z_lowpass_K if cfg.z_lowpass_K is not None else d_latent // 2 + 1
                    zL = cfg.z_lowpass_L if cfg.z_lowpass_L is not None else float(d_latent)
                    l_z_lowpass = latent_self_spectrum_lowpass_loss(z, zK, zL, power=cfg.z_lowpass_power)
                    loss = loss + cfg.w_z_lowpass * l_z_lowpass
                if cfg.w_z_lowpass_rollout > 0:
                    # See Stage1TrainingConfig.w_z_lowpass's docstring --
                    # same as w_z_lowpass but on the (--full-propagator)
                    # aux propagator's own rolled-out z_pred, mirroring
                    # w_lowpass_rollout's convention.
                    zK = cfg.z_lowpass_K if cfg.z_lowpass_K is not None else d_latent // 2 + 1
                    zL = cfg.z_lowpass_L if cfg.z_lowpass_L is not None else float(d_latent)
                    l_z_lowpass_rollout = latent_self_spectrum_lowpass_loss(
                        z_pred.reshape(-1, z_pred.shape[-1]), zK, zL, power=cfg.z_lowpass_rollout_power
                    )
                    loss = loss + cfg.w_z_lowpass_rollout * l_z_lowpass_rollout
                if cfg.w_channel_mean > 0:
                    # See Stage1TrainingConfig.w_channel_mean's docstring --
                    # encoder_kind="local_field" only (ae.cfg.n_sites/
                    # .local_channels duck-typed, same convention as other
                    # encoder-specific terms here).
                    l_channel_mean = local_field_channel_mean_loss(
                        z, ae.cfg.n_sites, ae.cfg.local_channels
                    )
                    loss = loss + cfg.w_channel_mean * l_channel_mean
                if cfg.w_logdet_physical_rollout > 0:
                    # See Stage1TrainingConfig.w_logdet_physical_rollout's
                    # docstring -- Stage-1 analogue of w_logdet_physical, but
                    # targeting the AUX PROPAGATOR's own short rolled-out
                    # z_pred (mapped to physical space) instead of the
                    # encoder's real z.
                    w_pred_states = decode_from_spectrum(
                        z_pred.reshape(-1, z_pred.shape[-1]), ae.cfg.K, ae.cfg.N_w
                    )
                    l_logdet_physical_rollout = logdet_barrier_loss(
                        w_pred_states, eps=cfg.logdet_physical_rollout_eps
                    )
                    loss = loss + cfg.w_logdet_physical_rollout * l_logdet_physical_rollout
                if cfg.w_var_physical > 0:
                    # See Stage1TrainingConfig.w_var_physical's docstring --
                    # decorr_var_loss's l_var term (w_var's own mechanism),
                    # applied to decode_from_spectrum(z) (the encoder's real
                    # physical-space field) instead of z directly.
                    w_states_var = decode_from_spectrum(z, ae.cfg.K, ae.cfg.N_w)
                    _, l_var_physical = decorr_var_loss(w_states_var)
                    loss = loss + cfg.w_var_physical * l_var_physical
                if cfg.w_shape_floor > 0:
                    l_shape_floor = spectral_shape_floor_loss(z, ae.cfg.K, p_ref)
                    loss = loss + cfg.w_shape_floor * l_shape_floor
                if cfg.w_temporal_floor_z > 0:
                    # See Stage1TrainingConfig.w_temporal_floor_z's docstring --
                    # one-sided floor on real states' own separation in z-space
                    # at cfg.temporal_floor_lag real steps apart.
                    l_temporal_floor_z = temporal_expansion_floor_loss(
                        z_win, cfg.temporal_floor_lag, z_temporal_floor_ref
                    )
                    loss = loss + cfg.w_temporal_floor_z * l_temporal_floor_z
                if cfg.w_temporal_floor_w > 0:
                    # Same mechanism, in PHYSICAL space (decode_from_spectrum(z)).
                    w_win = decode_from_spectrum(
                        z_win.reshape(-1, d_latent), ae.cfg.K, ae.cfg.N_w
                    ).reshape(b, window, ae.cfg.N_w)
                    l_temporal_floor_w = temporal_expansion_floor_loss(
                        w_win, cfg.temporal_floor_lag, w_temporal_floor_ref
                    )
                    loss = loss + cfg.w_temporal_floor_w * l_temporal_floor_w
                if reg_active and epoch >= reg_cfg.start_epoch:
                    loss = loss + reg(z_win)
                    if reg_cfg.apply_to_propagated:
                        loss = loss + reg(z_pred)

            if (
                cfg.w_local_expansion_floor > 0
                and not local_expansion_floor_applied_this_epoch
                and aux.mode == "markovian"
            ):
                # See Stage1TrainingConfig.w_local_expansion_floor's docstring --
                # EXPENSIVE (torch.func.vmap(jacrev(...))), so applied only
                # ONCE per epoch, on a small subsample, and deliberately
                # OUTSIDE the autocast block above (explicit float32, same
                # convention as logdet_barrier_loss's own bfloat16-under-
                # autocast workaround -- torch.func transforms composed with
                # autocast are not reliably supported).
                local_expansion_floor_applied_this_epoch = True
                n_sample = min(cfg.local_expansion_floor_n_samples, z.shape[0])
                sample_idx = torch.randperm(z.shape[0], device=device)[:n_sample]
                z_sample = z[sample_idx].float()
                l_local_expansion = propagator_local_expansion_floor_loss(
                    aux.step_one, z_sample, floor=cfg.local_expansion_floor_value
                )
                loss = loss + cfg.w_local_expansion_floor * l_local_expansion

            if (
                cfg.w_spectrum_shape > 0
                and not spectrum_shape_applied_this_epoch
                and aux.mode == "markovian"
            ):
                # See Stage1TrainingConfig.w_spectrum_shape's docstring --
                # same expensive/once-per-epoch/outside-autocast convention
                # as w_local_expansion_floor (shares its underlying
                # Jacobian computation).
                spectrum_shape_applied_this_epoch = True
                n_sample = min(cfg.spectrum_shape_n_samples, z.shape[0])
                sample_idx = torch.randperm(z.shape[0], device=device)[:n_sample]
                z_sample = z[sample_idx].float()
                l_spectrum_shape = propagator_spectrum_shape_loss(
                    aux.step_one, z_sample,
                    n_expand=cfg.spectrum_shape_n_expand,
                    expand_target=cfg.spectrum_shape_expand_target,
                    contract_floor=cfg.spectrum_shape_contract_floor,
                    two_sided=cfg.spectrum_shape_two_sided,
                )
                loss = loss + cfg.w_spectrum_shape * l_spectrum_shape

            if (
                cfg.w_spectrum_shape_graded > 0
                and not spectrum_shape_graded_applied_this_epoch
                and aux.mode == "markovian"
            ):
                # See Stage1TrainingConfig.w_spectrum_shape_graded's
                # docstring -- same expensive/once-per-epoch convention as
                # w_spectrum_shape, using the per-rank reference loaded
                # above instead of a two-group split.
                spectrum_shape_graded_applied_this_epoch = True
                n_sample = min(cfg.spectrum_shape_graded_n_samples, z.shape[0])
                sample_idx = torch.randperm(z.shape[0], device=device)[:n_sample]
                z_sample = z[sample_idx].float()
                l_spectrum_shape_graded = propagator_graded_spectrum_shape_loss(
                    aux.step_one, z_sample, spectrum_shape_graded_ref,
                )
                loss = loss + cfg.w_spectrum_shape_graded * l_spectrum_shape_graded

            if (
                cfg.w_spectrum_shape_multistep > 0
                and not spectrum_shape_multistep_applied_this_epoch
                and aux.mode == "markovian"
            ):
                # See Stage1TrainingConfig.w_spectrum_shape_multistep's
                # docstring (Section 186) -- constrains the COMPOSED k-step
                # Jacobian's spectrum, not the one-step Jacobian, to catch
                # persistent expansive-subspace self-feeding that a
                # one-step check cannot see. Same expensive/once-per-epoch
                # convention, smaller default sample count (k sequential
                # applications of step_fn before the one Jacobian call).
                spectrum_shape_multistep_applied_this_epoch = True
                n_sample = min(cfg.spectrum_shape_multistep_n_samples, z.shape[0])
                sample_idx = torch.randperm(z.shape[0], device=device)[:n_sample]
                z_sample = z[sample_idx].float()
                l_spectrum_shape_multistep = propagator_multistep_spectrum_shape_loss(
                    aux.step_one, z_sample,
                    k=cfg.spectrum_shape_multistep_k,
                    n_expand=cfg.spectrum_shape_n_expand,
                    expand_target=cfg.spectrum_shape_expand_target,
                    contract_floor=cfg.spectrum_shape_contract_floor,
                )
                loss = loss + cfg.w_spectrum_shape_multistep * l_spectrum_shape_multistep

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            epoch_loss += loss.detach()
            epoch_recon += l_recon.detach()
            if pde_head is not None and cfg.w_pde_distill > 0:
                epoch_pde_distill += l_pde_distill.detach()
            if pde_head is not None and cfg.w_pde_distill_real > 0:
                epoch_pde_distill_real += l_pde_distill_real.detach()
            if pde_head is not None and cfg.w_pde_distill_real_rollout > 0:
                epoch_pde_distill_real_rollout += l_pde_distill_real_rollout.detach()
            if pde_head is not None and cfg.w_pde_nonlinear_l2 > 0:
                epoch_pde_nonlinear += l_pde_nonlinear.detach()
            if pde_head is not None and cfg.w_pde_mean_conservation > 0:
                epoch_pde_mean += l_pde_mean.detach()
            if pde_head is not None and cfg.w_pde_energy_floor > 0:
                epoch_pde_energy += l_pde_energy_floor.detach()
            if cfg.w_prop_energy_floor > 0:
                epoch_prop_energy += l_prop_energy_floor.detach()
            if pde_head is not None and cfg.w_pde_coeff_l1 > 0:
                epoch_pde_coeff_l1 += l_pde_coeff_l1.detach()
            n_batches += 1
        sched.step()
        epoch_seconds = time.time() - epoch_t0
        decay_active = (
            cfg.w_var_end is not None or cfg.w_logdet_end is not None or cfg.w_spatial_end is not None
        )
        pde_distill_active = pde_head is not None and cfg.w_pde_distill > 0
        pde_distill_mean = (epoch_pde_distill / n_batches).item() if pde_distill_active else None
        pde_distill_real_active = pde_head is not None and cfg.w_pde_distill_real > 0
        pde_distill_real_mean = (epoch_pde_distill_real / n_batches).item() if pde_distill_real_active else None
        pde_distill_real_rollout_active = pde_head is not None and cfg.w_pde_distill_real_rollout > 0
        pde_distill_real_rollout_mean = (
            (epoch_pde_distill_real_rollout / n_batches).item() if pde_distill_real_rollout_active else None
        )
        pde_nonlinear_active = pde_head is not None and cfg.w_pde_nonlinear_l2 > 0
        pde_nonlinear_mean = (epoch_pde_nonlinear / n_batches).item() if pde_nonlinear_active else None
        pde_mean_active = pde_head is not None and cfg.w_pde_mean_conservation > 0
        pde_mean_mean = (epoch_pde_mean / n_batches).item() if pde_mean_active else None
        pde_energy_active = pde_head is not None and cfg.w_pde_energy_floor > 0
        pde_energy_mean = (epoch_pde_energy / n_batches).item() if pde_energy_active else None
        prop_energy_active = cfg.w_prop_energy_floor > 0
        prop_energy_mean = (epoch_prop_energy / n_batches).item() if prop_energy_active else None
        pde_coeff_l1_active = pde_head is not None and cfg.w_pde_coeff_l1 > 0
        pde_coeff_l1_mean = (epoch_pde_coeff_l1 / n_batches).item() if pde_coeff_l1_active else None
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            decay_suffix = (
                f"  w_var {w_var_now:.4g}  w_logdet {w_logdet_now:.4g}  w_spatial {w_spatial_now:.4g}"
                if decay_active
                else ""
            )
            # `pde_distill` (added 2026-09-08, user-directed: "can we report
            # the pde loss in the logs for training, just to see if it's
            # going down") -- the RAW (unweighted) l_pde_distill, so its
            # trend is comparable across different --w-pde-distill weights.
            pde_suffix = f"  pde_distill {pde_distill_mean:.6f}" if pde_distill_active else ""
            if pde_distill_real_active:
                pde_suffix += f"  pde_distill_real {pde_distill_real_mean:.6f}"
            if pde_distill_real_rollout_active:
                pde_suffix += f"  pde_distill_real_rollout {pde_distill_real_rollout_mean:.6f}"
            if pde_nonlinear_active:
                pde_suffix += f"  pde_nonlinear {pde_nonlinear_mean:.6f}"
            if pde_mean_active:
                pde_suffix += f"  pde_mean {pde_mean_mean:.6f}"
            if pde_energy_active:
                pde_suffix += f"  pde_energy {pde_energy_mean:.6f}"
            if prop_energy_active:
                pde_suffix += f"  prop_energy {prop_energy_mean:.6f}"
            if pde_coeff_l1_active:
                pde_suffix += f"  pde_coeff_l1 {pde_coeff_l1_mean:.6f}"
            print(
                f"  [stage1] epoch {epoch:3d}/{cfg.epochs}  k {k_now:2d}  "
                f"lr {sched.get_last_lr()[0]:.2e}  loss {(epoch_loss / n_batches).item():.6f}  "
                f"recon {(epoch_recon / n_batches).item():.6f}{decay_suffix}{pde_suffix}  [{epoch_seconds:.1f}s]",
                flush=True,
            )
        history.append(
            {
                "epoch": epoch,
                "loss": (epoch_loss / n_batches).item(),
                "recon": (epoch_recon / n_batches).item(),
                **({"pde_distill": pde_distill_mean} if pde_distill_active else {}),
                **({"pde_distill_real": pde_distill_real_mean} if pde_distill_real_active else {}),
                **(
                    {"pde_distill_real_rollout": pde_distill_real_rollout_mean}
                    if pde_distill_real_rollout_active
                    else {}
                ),
                **({"pde_nonlinear": pde_nonlinear_mean} if pde_nonlinear_active else {}),
                **({"pde_mean": pde_mean_mean} if pde_mean_active else {}),
                **({"pde_energy": pde_energy_mean} if pde_energy_active else {}),
                **({"prop_energy": prop_energy_mean} if prop_energy_active else {}),
                **({"pde_coeff_l1": pde_coeff_l1_mean} if pde_coeff_l1_active else {}),
                "k_now": k_now,
                "seconds": epoch_seconds,
                **(
                    {"w_var": w_var_now, "w_logdet": w_logdet_now, "w_spatial": w_spatial_now}
                    if decay_active
                    else {}
                ),
            }
        )
        if checkpoint_every > 0 and on_epoch_end is not None and (epoch + 1) % checkpoint_every == 0:
            on_epoch_end(epoch)

    val_recon = eval_stage1_reconstruction(ae, val_trajectories, window, device)
    return Stage1Result(train_history=history, val_recon_final=val_recon)


def encode_dataset_with_shifts(
    ae: KSAutoencoderPatched, trajectories: torch.Tensor, shifts: list[int], device: torch.device
) -> torch.Tensor:
    """Encode every trajectory under each cyclic shift in `shifts` (brief
    §4.5/§5.2: shift augmentation in latent space). Returns `(n_runs *
    len(shifts), T, d_latent)`.

    MPS correctness fallback (found 2026-09-23, Lorenz-96 Section 199 Stage
    2 nan bug): with `torch.use_deterministic_algorithms(True,
    warn_only=True)` set (every entry point does this via
    `ks_latent.utils.seeding.set_seed`, per ground rule 3), encoding a
    large batch (e.g. this project's 50-trajectory L96 train split, ~100k
    rows through a windowed-attention ViT encoder) through `ae.encode` on
    MPS was reproduced 8/8 times to silently emit `nan` for a clean subset
    of rows (168/256 rows fully nan, 88/256 fully clean -- not a
    numerically-noisy-but-finite result, genuine `nan`), while the
    bitwise-identical computation on CPU was clean every time. The exact
    failing kernel inside the attention stack was not pinned down (the
    `enc_attn_mask`/`dec_attn_mask` buffers themselves were checked and are
    NOT corrupted), consistent with the `index_put_with_accumulate_mps`
    "does not have a deterministic implementation" warning this project has
    seen throughout MPS training. Same spirit as brief §1.3 rule 6 (`torch.fft`
    on MPS: test for support, route to CPU on failure) -- not a blind
    try/except (ground rule 2): each shift's chunk is checked for
    finiteness, and only a genuinely non-finite MPS result triggers a CPU
    recompute of that chunk; if CPU *also* comes back non-finite, that is a
    real bug (bad checkpoint, bad data) and this raises rather than
    swallowing it.
    """
    ae.eval()
    trajectories = trajectories.to(device)
    n_runs, T, NX = trajectories.shape
    cpu_ae = None
    out = []
    with torch.no_grad():
        for c in shifts:
            shifted = torch.roll(trajectories, shifts=c, dims=-1)
            flat = shifted.reshape(n_runs * T, NX)
            z = ae.encode(flat)
            if device.type == "mps" and not torch.isfinite(z).all():
                if cpu_ae is None:
                    cpu_ae = deepcopy(ae).to("cpu").eval()
                z_cpu = cpu_ae.encode(flat.to("cpu"))
                if not torch.isfinite(z_cpu).all():
                    raise RuntimeError(
                        f"encode_dataset_with_shifts: ae.encode produced non-finite "
                        f"output on CPU too (shift={c}), so this is not the known "
                        f"MPS deterministic-algorithms bug -- a real upstream bug "
                        f"(bad checkpoint or bad data) must be investigated."
                    )
                z = z_cpu.to(device)
            out.append(z.view(n_runs, T, -1))
    ae.train()
    return torch.cat(out, dim=0)


def k_curriculum(
    epoch: int,
    k_min: int,
    k_max: int,
    warmup_epochs: int,
    k_mid: int | None = None,
    mid_epochs: int = 0,
) -> int:
    """Ramp the rollout length from `k_min` to `k_max` over the first
    `warmup_epochs` epochs, then hold at `k_max`. `warmup_epochs <= 0` holds
    at `k_max` from epoch 0 (no ramp). Shared by Stage 2's existing
    K-curriculum (`k_min=2`) and Stage 1's optional multistep-rollout
    curriculum (`Stage1TrainingConfig.k_pred_max`, added 2026-08-29).

    `k_mid`/`mid_epochs` (added 2026-08-29, user-directed): split the ramp
    into two linear segments -- `k_min` -> `k_mid` over the first
    `mid_epochs` epochs, then `k_mid` -> `k_max` over the remaining
    `warmup_epochs - mid_epochs` epochs -- instead of one linear ramp
    straight through. `k_mid=None` (default) is the original single-segment
    behavior. Motivated by the plateau seen training the
    vit/attn_window=4/linear propagator (best_val_kmax_mse stopped improving
    once k hit 16 at epoch 8/20 of a straight 2->16 ramp): dwell longer on
    short horizons (e.g. k=2..6) before the harder long-horizon regime,
    rather than a uniform per-k dwell that a single linear ramp gives."""
    if k_mid is None:
        frac = min(epoch / warmup_epochs, 1.0) if warmup_epochs > 0 else 1.0
        return k_min + int(math.floor(frac * (k_max - k_min) + 0.5))
    if epoch < mid_epochs:
        frac = epoch / mid_epochs if mid_epochs > 0 else 1.0
        return k_min + int(math.floor(frac * (k_mid - k_min) + 0.5))
    stage2_epochs = max(warmup_epochs - mid_epochs, 1)
    frac = min((epoch - mid_epochs) / stage2_epochs, 1.0)
    return k_mid + int(math.floor(frac * (k_max - k_mid) + 0.5))


def _k_curriculum(
    epoch: int, k_max: int, warmup_epochs: int, k_mid: int | None = None, mid_epochs: int = 0
) -> int:
    return k_curriculum(epoch, 2, k_max, warmup_epochs, k_mid, mid_epochs)


def _linear_anneal(epoch: int, total_epochs: int, start: float, end: float) -> float:
    frac = epoch / max(total_epochs - 1, 1)
    return start + frac * (end - start)


def _pde_head_self_rollout_states(
    pde_head: LatentPropagator, z0: torch.Tensor, k: int
) -> torch.Tensor | None:
    """`z0`: `(B, d)` real starting states, already detached. Rolls
    `pde_head` forward `k` steps under `torch.no_grad()` (cheap -- no
    backprop through the rollout chain, matching every other once-per-
    epoch Jacobian regularizer's "evaluate AT a batch of points"
    convention) and returns the FINAL state of each self-generated
    rollout, fully detached -- the state pool `w_pde_spectrum_shape_self`/
    `w_pde_spectrum_shape_multistep_self` (Section 193, see those fields'
    docstrings in `Stage1TrainingConfig` for the full motivation: the
    real-data-anchored versions never sample from where pde_head's OWN
    free rollout actually goes). Returns `None` if the self-rollout goes
    non-finite (plausible mid-training, an unsupervised free-running
    rollout of a partially-converged closure) -- same "skip this batch's
    contribution rather than poison the whole loss" discipline
    `w_pde_energy_floor` already uses for exactly this failure mode."""
    with torch.no_grad():
        self_roll = pde_head.rollout(z0, z0, k)
        z_self = self_roll[:, -1].detach()
    if not torch.isfinite(z_self).all():
        return None
    return z_self


def _propagator_spectrum_shape_self_pool(
    propagator: LatentPropagator,
    windows: torch.Tensor,
    n_hist: int,
    k: int,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]] | None:
    """Generalizes `_pde_head_self_rollout_states` to the MAIN Stage-2
    propagator (Section 203 rerun -- see `Stage2TrainingConfig.
    w_spectrum_shape_self`'s docstring for the full motivation), and to
    `mode="history"` (which `_pde_head_self_rollout_states`'s `.rollout`
    call cannot fit through -- `LatentPropagator._require_not_history`
    raises -- since `pde_head` is always `mode="markovian"` and never
    needed this).

    Dispatches on `propagator.mode`:
    - `"markovian"`: reuses `_pde_head_self_rollout_states` (identical
      mechanism, `propagator` in place of `pde_head`) and returns
      `propagator.step_one` (already `(B,d) -> (B,d)`) as the step
      function to shape.
    - `"history"`: rolls `propagator.rollout_history` forward `k` steps
      under `torch.no_grad()` from a real `(B, n_hist, d)` starting
      window, then takes the TRAILING `n_hist`-length window of
      `cat([z_hist0, self_roll])` as the evaluation point (the actual
      `step_history` input shape). `_propagator_step_jacobian_singular_
      values` (shared with every other spectrum-shape loss) requires a
      FLAT per-sample vector to differentiate through
      `torch.func.vmap(jacrev(...))` correctly -- so the returned pool is
      flattened to `(n_samples, n_hist*d)` and paired with a `step_fn`
      closure that reshapes back to `(B, n_hist, d)` before calling
      `propagator.step_history`, then flattens again is NOT needed since
      `step_history` already returns `(B, d)`. This is the same
      flatten-for-jacrev trick `LatentPropagator.step_history` itself
      uses internally for `backbone="mlp"` (`z_hist.reshape(B, -1)`).

    Returns `None` if the self-rollout goes non-finite (same discipline
    as `_pde_head_self_rollout_states`) -- caller must skip this batch's
    contribution rather than poison the whole loss."""
    B_full = windows.shape[0]
    n_sample = min(n_samples, B_full)
    sample_idx = torch.randperm(B_full, device=device)[:n_sample]
    if propagator.mode == "history":
        z_hist0 = windows[sample_idx, :n_hist].detach()
        with torch.no_grad():
            self_roll = propagator.rollout_history(z_hist0, k)
            full = torch.cat([z_hist0, self_roll], dim=1)
            z_hist_self = full[:, -n_hist:].detach()
        if not torch.isfinite(z_hist_self).all():
            return None
        d = z_hist_self.shape[-1]
        z_pool = z_hist_self.reshape(n_sample, -1).float()

        def step_fn(z_flat: torch.Tensor) -> torch.Tensor:
            bsz = z_flat.shape[0]
            return propagator.step_history(z_flat.reshape(bsz, n_hist, d))

        return z_pool, step_fn
    z0 = windows[sample_idx, 0].detach()
    z_self = _pde_head_self_rollout_states(propagator, z0, k)
    if z_self is None:
        return None
    return z_self.float(), propagator.step_one


@dataclass
class Stage2Result:
    train_history: list[dict]
    best_val_kmax_mse: float
    best_state_dict: dict


def _propagator_history_len(propagator: LatentPropagator) -> int:
    """Number of leading states a window must supply before the k_max
    prediction targets -- 2 for `"two_step"`/`"markovian"` (see
    `LatentPropagator.step`), or `cfg.n_history` for `mode="history"`
    (see `.step_history`)."""
    return propagator.cfg.n_history if propagator.mode == "history" else 2


def eval_stage2_kmax(
    propagator: LatentPropagator,
    sequences: torch.Tensor,
    k_max: int,
    device: torch.device,
    compare_k: int | None = None,
) -> float | tuple[float, float]:
    """Returns the `k_max`-pooled MSE (see `Stage2TrainingConfig.
    compare_k`'s docstring for why this isn't comparable across different
    `k_max` values). `compare_k` set: ALSO pools the MSE over just the
    first `compare_k` steps of the SAME rollout (no extra propagator
    calls) and returns `(kmax_mse, compare_mse)` instead of a bare
    float -- lets two runs at different `k_max` be compared on equal
    footing at a shared, smaller horizon."""
    propagator.eval()
    sequences = sequences.to(device)
    n_runs, T, d = sequences.shape
    n_hist = _propagator_history_len(propagator)
    index = make_window_index(n_runs, T, k_max + n_hist, device=device)
    total, count = 0.0, 0
    compare_total = 0.0
    with torch.no_grad():
        for batch_idx in index_shuffle_batches(index.shape[0], 512):
            idx = index[batch_idx]
            windows = gather_windows(sequences, idx, k_max + n_hist)
            z_true = windows[:, n_hist:]
            if propagator.mode == "history":
                z_pred = propagator.rollout_history(windows[:, :n_hist], k_max)
            else:
                z_pred = propagator.rollout(windows[:, 0], windows[:, 1], k_max)
            mse = ((z_pred - z_true) ** 2).mean().item()
            total += mse * windows.shape[0]
            count += windows.shape[0]
            if compare_k is not None:
                compare_mse = ((z_pred[:, :compare_k] - z_true[:, :compare_k]) ** 2).mean().item()
                compare_total += compare_mse * windows.shape[0]
    propagator.train()
    kmax_mse = total / count
    if compare_k is None:
        return kmax_mse
    return kmax_mse, compare_total / count


def train_stage2(
    propagator: LatentPropagator,
    train_sequences: torch.Tensor,
    val_sequences: torch.Tensor,
    cfg: Stage2TrainingConfig,
    device: torch.device,
    verbose: bool = False,
    log_every: int = 1,
    checkpoint_every: int = 0,
    on_epoch_end: Callable[[int], None] | None = None,
    amp: bool = False,
    pde_head: AuxPropagator | None = None,
    freeze_propagator: bool = False,
) -> Stage2Result:
    """`train_sequences`/`val_sequences`: `(n_seqs, T, d_latent)`, from
    `encode_dataset_with_shifts` on a frozen Stage-1 AE.

    `freeze_propagator` (added 2026-09-08, user-directed: "should we add a
    phase 3 that refines the pde with the propagator fixed? basically how
    should we extract the pde?" -- see docs/sine_transform_pde_plan.md
    §23): a "Phase 3" mode -- `propagator` is kept in `.eval()` and
    EXCLUDED from the optimizer entirely (only `pde_head`'s parameters are
    trained), with `propagator`'s own rollout computed under `torch.
    no_grad()` (no wasted graph/memory -- it isn't being updated
    regardless). This is the actual answer to "how should we extract the
    pde": Phases 1-2's `pde_head` losses exist to SHAPE the encoder/
    propagator toward being PDE-describable while they're still being
    fit to real data; once `propagator` is fully trained and fixed, the
    cleanest way to get the best possible PDE fit to its (now fixed)
    dynamics is a dedicated final refinement stage with no competing
    pressure on `propagator` at all -- ordinary teacher-frozen
    distillation. `cfg.pde_distill_detach_target`/the mutual-vs-detached
    distinction is moot here (`propagator` isn't in the optimizer, so no
    gradient reaches it regardless of whether the target is detached) --
    both `cfg.w_pde_distill` (option A) and `cfg.w_pde_rollout` (option B)
    can be set more aggressively than in Phases 1-2, since there is no
    more collapse risk to `propagator` to protect against; `pde_head` is
    the only thing that can still fail (an inherently under-parameterized
    or ill-suited local PDE form simply not fitting well), which is
    exactly the honest measurement this phase is FOR. Requires `pde_head`
    to be given (raises `ValueError` otherwise -- nothing to train).

    `amp` (added 2026-09-01): see `train_stage1`'s `amp` docstring section
    -- identical mechanism (bfloat16 autocast around the forward pass +
    loss only), applied here to the propagator's rollout instead.

    `verbose` (added 2026-08-29, user-directed): print one line per
    `log_every` epochs (epoch, k_now, loss, val_kmax_mse, wall time for
    that epoch). Added after a real gap was noticed the hard way: neither
    `train_stage1` nor `train_stage2` printed anything mid-run, so a slow
    (e.g. attention-heavy `vit`-backbone) training job gave no visibility
    into whether it was progressing or stuck short of waiting for it to
    finish entirely.

    `checkpoint_every`/`on_epoch_end` (added 2026-08-31): see
    `train_stage1`'s docstring for the full motivation and contract -- same
    convention here, called with the CURRENT (not best-so-far) `propagator`
    state after every `checkpoint_every`-th completed epoch.

    `pde_head` (added 2026-09-08, user-directed: "is the pde_head trained
    during phase 2 as well. we should try to get pde rollout to have
    decent performance" then "please implement both option a and b, and
    we can try both. at the end of the day, we're really just training
    the propagator right, the pde_head training is acting as a
    regularization term" -- see docs/sine_transform_pde_plan.md §21):
    continues training `pde_head` (a `backbone="spectral_pde_raw"`
    propagator, same object Stage 1 built/trained if continuing from
    there) ALONGSIDE `propagator` here.

    **This is a DELIBERATELY DIFFERENT design from Stage 1's own
    `pde_head` mechanism** (`train_stage1`'s docstring): there, the
    distillation target (`aux`'s own realized step) is DETACHED, so
    gradient reaches only `pde_head` and, through `z`, the ENCODER --
    `aux` stays protected while the encoder co-adapts. Here, Stage 2 never
    touches the encoder at all (`train_sequences` is already-encoded,
    frozen `z`) -- so a detached target would make this term pointless as
    a "regularizer on the propagator" (the user's own framing): there
    would be nothing left for it to shape except `pde_head` in isolation.
    So here the loss is MUTUAL (`propagator`'s own `z_pred` is NOT
    detached): gradient flows into BOTH `propagator` and `pde_head`,
    genuinely regularizing the propagator itself toward dynamics a local
    PDE can also explain, while `pde_head` simultaneously learns to track
    it. Gated by two independent weights (either or both may be active):

    - `cfg.w_pde_distill` ("option A", the safe extension of Stage 1's own
      mechanism): at EVERY one of the `k_now` steps of `propagator`'s own
      realized rollout, `pde_head.step_one` is evaluated from the SAME
      state `propagator` started that step from, and compared to
      `propagator`'s own realized next state. `pde_head`'s own gradient is
      still only ever a single-step regression -- never backprop through
      `pde_head`'s own multi-step chain -- so this stays numerically safe
      regardless of `k_now`.
    - `cfg.w_pde_rollout` ("option B", the riskier direct approach): runs
      `pde_head.rollout` autoregressively for the SAME `k_now` steps,
      chained through `pde_head` itself, and compares the WHOLE trajectory
      to `propagator`'s own. This is the thing that made
      `backbone="spectral_pde"` "incredibly hard to train" as a PRIMARY
      propagator in Sections 101-106 (stiffness, collapse) -- offered
      here specifically so both can be tried and compared, per user
      direction, now that Stage 1's distillation may have already
      pressured `z` toward being more locally-PDE-fittable than those
      earlier attempts started from.

    `None` (default) is off, unchanged behavior."""
    propagator.to(device)
    if pde_head is not None:
        pde_head.to(device)
    train_sequences = train_sequences.to(device)
    n_seqs, T, d = train_sequences.shape
    n_hist = _propagator_history_len(propagator)
    window = cfg.k_max + n_hist
    if pde_head is not None and cfg.w_pde_distill_real_rollout > 0 and cfg.pde_distill_real_rollout_k > cfg.k_max:
        raise ValueError(
            f"pde_distill_real_rollout_k={cfg.pde_distill_real_rollout_k!r} exceeds "
            f"k_max={cfg.k_max!r} -- the training window has no real future states beyond "
            "that to supply as rollout targets."
        )
    window_index = make_window_index(n_seqs, T, window, device=device)

    if freeze_propagator and pde_head is None:
        raise ValueError("freeze_propagator=True requires a non-None pde_head -- nothing to train.")
    if freeze_propagator:
        propagator.eval()
        for p in propagator.parameters():
            p.requires_grad_(False)

    spectrum_shape_graded_ref = None
    if cfg.w_spectrum_shape_graded > 0:
        # See Stage2TrainingConfig.w_spectrum_shape_graded's docstring --
        # loaded ONCE from a file, not recomputed.
        spectrum_shape_graded_ref = torch.from_numpy(
            np.load(cfg.spectrum_shape_graded_reference_path)
        ).float().to(device)

    params = [] if freeze_propagator else list(propagator.parameters())
    if pde_head is not None:
        params = params + list(pde_head.parameters())
    # See `train_stage1`'s identical block -- `stable_linear_lr_factor`
    # (Section 174) gives any `poly_stable_linear_terms` raw parameters
    # their own much smaller LR param group.
    stable_params = (
        propagator.stable_linear_raw_parameters()
        if not freeze_propagator and hasattr(propagator, "stable_linear_raw_parameters")
        else []
    )
    if stable_params:
        stable_ids = {id(p) for p in stable_params}
        other_params = [p for p in params if id(p) not in stable_ids]
        opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg.lr},
                {"params": stable_params, "lr": cfg.lr * cfg.stable_linear_lr_factor},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine_lr_lambda(cfg.epochs, cfg.warmup_epochs, cfg.lr_min_factor)
    )

    varmatch_target: torch.Tensor | float = 1.0
    if cfg.w_varmatch > 0 and cfg.w_varmatch_adaptive:
        # Real per-channel latent variance from the AE's own encoded training
        # data, in place of decorr_var_loss's hardcoded uniform target of 1 --
        # see Stage2TrainingConfig.w_varmatch_adaptive's docstring.
        varmatch_target = train_sequences.reshape(-1, d).var(dim=0)

    history = []
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in propagator.state_dict().items()}
    for epoch in range(cfg.epochs):
        epoch_t0 = time.time()
        k_now = _k_curriculum(epoch, cfg.k_max, cfg.k_warmup_epochs, cfg.k_mid, cfg.k_mid_epochs)
        noise_in = _linear_anneal(epoch, cfg.epochs, cfg.noise_in_start, cfg.noise_in_end)
        noise_step = _linear_anneal(epoch, cfg.epochs, cfg.noise_step_start, cfg.noise_step_end)
        weights = horizon_weights(k_now, cfg.gamma, device=device, dtype=train_sequences.dtype)
        k_roll_now = (
            k_curriculum(epoch, 1, cfg.pde_distill_real_rollout_k, cfg.pde_distill_real_rollout_warmup_epochs)
            if pde_head is not None and cfg.w_pde_distill_real_rollout > 0
            else 0
        )
        k_energy_now = (
            k_curriculum(epoch, 1, cfg.pde_energy_floor_rollout_k, cfg.pde_energy_floor_warmup_epochs)
            if pde_head is not None and cfg.w_pde_energy_floor > 0
            else 0
        )
        k_prop_energy_now = (
            k_curriculum(epoch, 1, cfg.prop_energy_floor_rollout_k, cfg.prop_energy_floor_warmup_epochs)
            if cfg.w_prop_energy_floor > 0
            else 0
        )

        batches = index_shuffle_batches(window_index.shape[0], cfg.batch_size)
        epoch_loss = torch.zeros((), device=device)
        epoch_pde_distill = torch.zeros((), device=device)
        epoch_pde_rollout = torch.zeros((), device=device)
        epoch_pde_distill_real = torch.zeros((), device=device)
        epoch_pde_distill_real_rollout = torch.zeros((), device=device)
        epoch_pde_nonlinear = torch.zeros((), device=device)
        epoch_pde_mean = torch.zeros((), device=device)
        epoch_pde_energy = torch.zeros((), device=device)
        epoch_prop_energy = torch.zeros((), device=device)
        epoch_pde_coeff_l1 = torch.zeros((), device=device)
        n_batches = 0
        spectrum_shape_applied_this_epoch = False
        spectrum_shape_self_applied_this_epoch = False
        spectrum_shape_graded_applied_this_epoch = False
        spectrum_shape_multistep_applied_this_epoch = False
        pde_spectrum_shape_applied_this_epoch = False
        pde_spectrum_shape_multistep_applied_this_epoch = False
        pde_spectrum_shape_self_applied_this_epoch = False
        pde_spectrum_shape_multistep_self_applied_this_epoch = False
        for batch_idx in batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                idx = window_index[batch_idx]
                windows = gather_windows(train_sequences, idx, window)
                z_true = windows[:, n_hist : n_hist + k_now]

                if propagator.mode == "history":
                    z_hist = windows[:, :n_hist] + noise_in * torch.randn_like(windows[:, :n_hist])
                    z_pred = propagator.rollout_history(z_hist, k_now, step_noise=noise_step)
                else:
                    z_prev = windows[:, 0] + noise_in * torch.randn_like(windows[:, 0])
                    z_curr = windows[:, 1] + noise_in * torch.randn_like(windows[:, 1])
                    z_pred = propagator.rollout(z_prev, z_curr, k_now, step_noise=noise_step)
                loss = horizon_weighted_latent_loss(z_pred, z_true, weights, kind=cfg.latent_loss)
                if cfg.w_varmatch > 0:
                    # Penalize the rolled-out predictions' per-dimension variance (across the
                    # batch and all k_now rollout steps) for deviating from var_target -- see
                    # Stage2TrainingConfig's docstring (targets the fixed-point-collapse failure
                    # mode: Var -> 0 as different ICs converge to the same point).
                    _, l_var = decorr_var_loss(z_pred.reshape(-1, d), var_target=varmatch_target)
                    loss = loss + cfg.w_varmatch * l_var
                if cfg.w_prop_energy_floor > 0:
                    # See Stage1TrainingConfig.w_prop_energy_floor's
                    # docstring -- UNSUPERVISED (no real target, no
                    # window-size constraint), applied to the propagator
                    # ITSELF via its own ramped (k_prop_energy_now)
                    # autoregressive rollout from a real detached starting
                    # state. Unlike w_varmatch above (bounded by k_now,
                    # i.e. by k_max -- found empirically, Sections 169/171,
                    # to be blind to a collapse rate too slow to show up
                    # within that horizon), this term's horizon is
                    # independent of k_now/k_max entirely.
                    if propagator.mode == "history":
                        e_hist_prop = windows[:, :n_hist].detach()
                        z_prop_energy_roll = propagator.rollout_history(e_hist_prop, k_prop_energy_now)
                    else:
                        e_start_prop = windows[:, n_hist - 1].detach()
                        z_prop_energy_roll = propagator.rollout(e_start_prop, e_start_prop, k_prop_energy_now)
                    if torch.isfinite(z_prop_energy_roll).all():
                        l_prop_energy_floor = spatial_energy_floor_loss(
                            z_prop_energy_roll.reshape(-1, z_prop_energy_roll.shape[-1]),
                            cfg.prop_energy_floor_gamma,
                        )
                        loss = loss + cfg.w_prop_energy_floor * l_prop_energy_floor
                    else:
                        # Same reasoning as w_pde_energy_floor's identical
                        # guard (Section 170's own postmortem).
                        print(
                            f"  [stage2] WARNING: propagator energy-floor rollout produced "
                            f"non-finite values at k_prop_energy_now={k_prop_energy_now} "
                            f"(epoch {epoch}) -- skipping this batch's prop-energy-floor "
                            f"contribution.",
                            flush=True,
                        )
                        l_prop_energy_floor = torch.zeros((), device=z_prop_energy_roll.device)
                if cfg.w_kernel_unstable_floor > 0:
                    # See Stage1TrainingConfig.w_kernel_unstable_floor's
                    # docstring -- a pure function of propagator's own
                    # kernel parameters, not of this batch's data at all.
                    Lhat = propagator.kernel_Lhat()
                    if Lhat is not None:
                        l_kernel_unstable = kernel_unstable_floor_loss(
                            Lhat, cfg.kernel_unstable_target_modes, cfg.kernel_unstable_margin
                        )
                        loss = loss + cfg.w_kernel_unstable_floor * l_kernel_unstable
                if cfg.w_spatial > 0:
                    # Stage-2 analogue of Stage1TrainingConfig.w_spatial -- see
                    # Stage2TrainingConfig.w_spatial's docstring.
                    l_spatial = spatial_coherence_loss(
                    z_pred.reshape(-1, d), bandwidth=cfg.spatial_bandwidth, signed=cfg.spatial_signed
                )
                    loss = loss + cfg.w_spatial * l_spatial
                if cfg.w_lowpass_rollout > 0:
                    # See Stage2TrainingConfig.w_lowpass_rollout's docstring --
                    # unlike w_lowpass (Stage 1, encoder's own z), this targets
                    # the PROPAGATOR's own rolled-out z_pred. K/L are duck-typed
                    # from propagator.cfg (only meaningful for
                    # backbone="spectral_pde"; guaranteed present whenever this
                    # weight is nonzero by Stage2TrainingConfig.__post_init__).
                    K = propagator.cfg.spectral_K
                    L = propagator.cfg.spectral_L
                    l_lowpass_rollout = low_pass_spectral_loss(
                        z_pred.reshape(-1, d), K, L, power=cfg.lowpass_rollout_power
                    )
                    loss = loss + cfg.w_lowpass_rollout * l_lowpass_rollout
                if cfg.w_z_lowpass_rollout > 0:
                    # See Stage1TrainingConfig.w_z_lowpass's docstring for
                    # the full mechanism (ANY encoder's raw z, unlike
                    # w_lowpass_rollout above which requires
                    # encoder_kind="spectral_field"). No encoder here to
                    # apply a non-rollout term to -- rollout-only in Stage 2.
                    zK = cfg.z_lowpass_K if cfg.z_lowpass_K is not None else d // 2 + 1
                    zL = cfg.z_lowpass_L if cfg.z_lowpass_L is not None else float(d)
                    l_z_lowpass_rollout = latent_self_spectrum_lowpass_loss(
                        z_pred.reshape(-1, d), zK, zL, power=cfg.z_lowpass_rollout_power
                    )
                    loss = loss + cfg.w_z_lowpass_rollout * l_z_lowpass_rollout
                if cfg.w_logdet_rollout > 0:
                    # See Stage2TrainingConfig.w_logdet_rollout's docstring --
                    # maps the PROPAGATOR's own rolled-out z_pred to physical
                    # space (decode_from_spectrum, the exact irfft) before
                    # applying the full-covariance log-det anti-collapse
                    # barrier, so this reacts to the field's own shape
                    # collapsing, not just its spectral coefficients.
                    K = propagator.cfg.spectral_K
                    N_w = propagator.cfg.spectral_N_w
                    w_states = decode_from_spectrum(z_pred.reshape(-1, d), K, N_w)
                    l_logdet_rollout = logdet_barrier_loss(w_states, eps=cfg.logdet_rollout_eps)
                    loss = loss + cfg.w_logdet_rollout * l_logdet_rollout
                if cfg.w_logdet_rollout_latent > 0:
                    # See Stage2TrainingConfig.w_logdet_rollout_latent's
                    # docstring -- mirrors Stage 1's own w_logdet exactly
                    # (logdet_barrier_loss applied directly to a batch of
                    # latent vectors), just on the propagator's own
                    # rolled-out z_pred instead of the encoder's raw
                    # output. General to any backbone (no decode step).
                    l_logdet_rollout_latent = logdet_barrier_loss(
                        z_pred.reshape(-1, d), eps=cfg.logdet_rollout_latent_eps
                    )
                    loss = loss + cfg.w_logdet_rollout_latent * l_logdet_rollout_latent
                if pde_head is not None and (cfg.w_pde_distill > 0 or cfg.w_pde_rollout > 0):
                    # See train_stage2's own docstring for the full mechanism
                    # and why this is MUTUAL (z_pred NOT detached), unlike
                    # Stage 1's own pde_head loss.
                    z_start = z_hist[:, -1] if propagator.mode == "history" else z_curr
                    if cfg.w_pde_distill > 0:
                        # Option A: single-step match at EVERY position along
                        # propagator's own realized rollout -- pde_head's own
                        # gradient stays a single-step regression regardless
                        # of k_now (numerically safe).
                        z_states = torch.cat([z_start.unsqueeze(1), z_pred[:, :-1]], dim=1)
                        p_pred = pde_head.step_one(z_states.reshape(-1, d)).reshape(z_pred.shape)
                        l_pde_distill = ((p_pred - z_pred) ** 2).mean()
                        loss = loss + cfg.w_pde_distill * l_pde_distill
                    if cfg.w_pde_rollout > 0:
                        # Option B: pde_head's OWN autoregressive k_now-step
                        # rollout vs. propagator's -- the riskier direct
                        # approach, see the docstring's Sections 101-106 note.
                        z_pde_rollout = pde_head.rollout(z_start, z_start, k_now)
                        l_pde_rollout = ((z_pde_rollout - z_pred) ** 2).mean()
                        loss = loss + cfg.w_pde_rollout * l_pde_rollout
                if pde_head is not None and cfg.w_pde_distill_real > 0:
                    # See Stage1TrainingConfig.w_pde_distill_real's
                    # docstring -- here `windows` is the FROZEN real
                    # encoded sequence (no dependence on the propagator at
                    # all), so this term is completely isolated from
                    # whatever the propagator is doing this step.
                    real_start = windows[:, :-1].reshape(-1, d)
                    real_next = windows[:, 1:].reshape(-1, d)
                    if cfg.pde_distill_real_detach:
                        real_start = real_start.detach()
                        real_next = real_next.detach()
                    p_pred_real = pde_head.step_one(real_start)
                    l_pde_distill_real = reconstruction_loss(p_pred_real, real_next)
                    loss = loss + cfg.w_pde_distill_real * l_pde_distill_real
                if pde_head is not None and cfg.w_pde_distill_real_rollout > 0:
                    # See Stage1TrainingConfig.w_pde_distill_real_rollout's
                    # docstring (iLED, arXiv:2309.05812) -- pde_head's OWN
                    # chained rollout from a REAL starting state, compared
                    # at every step to REAL future states already in
                    # `windows` (never propagator's own predictions).
                    # k_roll_now (per-epoch, ramped -- see
                    # pde_distill_real_rollout_warmup_epochs's docstring),
                    # not the raw config target -- see Stage 1's own
                    # identical comment for the empirical reason.
                    k_roll = k_roll_now
                    r_start = windows[:, n_hist - 1]
                    if cfg.pde_distill_real_detach:
                        r_start = r_start.detach()
                    p_pred_rollout = pde_head.rollout(r_start, r_start, k_roll)  # (b, k_roll, d)
                    r_targets = windows[:, n_hist : n_hist + k_roll]
                    if cfg.pde_distill_real_detach:
                        r_targets = r_targets.detach()
                    l_pde_distill_real_rollout = ((p_pred_rollout - r_targets) ** 2).mean()
                    loss = loss + cfg.w_pde_distill_real_rollout * l_pde_distill_real_rollout
                if pde_head is not None and cfg.w_pde_nonlinear_l2 > 0:
                    # See Stage1TrainingConfig.w_pde_nonlinear_l2's
                    # docstring (iLED, arXiv:2309.05812) -- a prior on
                    # pde_head's OWN nonlinear closure alone, evaluated on
                    # real detached states, never mutual with propagator.
                    z_for_nonlinear = windows.reshape(-1, d).detach()
                    l_pde_nonlinear = pde_head_nonlinear_output(pde_head, z_for_nonlinear).pow(2).mean()
                    loss = loss + cfg.w_pde_nonlinear_l2 * l_pde_nonlinear
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape > 0
                    and not pde_spectrum_shape_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage2TrainingConfig.w_pde_spectrum_shape's
                    # docstring -- pde_head analogue of w_spectrum_shape,
                    # same expensive/once-per-epoch convention, evaluated
                    # on real detached states, pde_head.step_one only.
                    pde_spectrum_shape_applied_this_epoch = True
                    z_pool_pde = windows.reshape(-1, d).detach()
                    n_sample = min(cfg.pde_spectrum_shape_n_samples, z_pool_pde.shape[0])
                    sample_idx = torch.randperm(z_pool_pde.shape[0], device=device)[:n_sample]
                    z_sample_pde = z_pool_pde[sample_idx].float()
                    l_pde_spectrum_shape = propagator_spectrum_shape_loss(
                        pde_head.step_one, z_sample_pde,
                        n_expand=cfg.pde_spectrum_shape_n_expand,
                        expand_target=cfg.pde_spectrum_shape_expand_target,
                        contract_floor=cfg.pde_spectrum_shape_contract_floor,
                        two_sided=cfg.pde_spectrum_shape_two_sided,
                    )
                    loss = loss + cfg.w_pde_spectrum_shape * l_pde_spectrum_shape
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_multistep > 0
                    and not pde_spectrum_shape_multistep_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage2TrainingConfig.w_pde_spectrum_shape_multistep's
                    # docstring -- pde_head analogue of
                    # w_spectrum_shape_multistep, pde_head.step_one only.
                    pde_spectrum_shape_multistep_applied_this_epoch = True
                    z_pool_pde = windows.reshape(-1, d).detach()
                    n_sample = min(cfg.pde_spectrum_shape_multistep_n_samples, z_pool_pde.shape[0])
                    sample_idx = torch.randperm(z_pool_pde.shape[0], device=device)[:n_sample]
                    z_sample_pde = z_pool_pde[sample_idx].float()
                    l_pde_spectrum_shape_multistep = propagator_multistep_spectrum_shape_loss(
                        pde_head.step_one, z_sample_pde,
                        k=cfg.pde_spectrum_shape_multistep_k,
                        n_expand=cfg.pde_spectrum_shape_n_expand,
                        expand_target=cfg.pde_spectrum_shape_expand_target,
                        contract_floor=cfg.pde_spectrum_shape_contract_floor,
                    )
                    loss = loss + cfg.w_pde_spectrum_shape_multistep * l_pde_spectrum_shape_multistep
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_self > 0
                    and not pde_spectrum_shape_self_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape_self's
                    # docstring (Section 193) -- SELF-rollout-sampled
                    # analogue of w_pde_spectrum_shape.
                    pde_spectrum_shape_self_applied_this_epoch = True
                    z0_pool = windows[:, 0].reshape(-1, d).detach()
                    n_sample = min(cfg.pde_spectrum_shape_self_n_samples, z0_pool.shape[0])
                    sample_idx = torch.randperm(z0_pool.shape[0], device=device)[:n_sample]
                    z0_self = z0_pool[sample_idx].float()
                    z_self_states = _pde_head_self_rollout_states(
                        pde_head, z0_self, cfg.pde_spectrum_shape_self_rollout_k
                    )
                    if z_self_states is not None:
                        l_pde_spectrum_shape_self = propagator_spectrum_shape_loss(
                            pde_head.step_one, z_self_states,
                            n_expand=cfg.pde_spectrum_shape_n_expand,
                            expand_target=cfg.pde_spectrum_shape_expand_target,
                            contract_floor=cfg.pde_spectrum_shape_contract_floor,
                            two_sided=cfg.pde_spectrum_shape_two_sided,
                        )
                        loss = loss + cfg.w_pde_spectrum_shape_self * l_pde_spectrum_shape_self
                if (
                    pde_head is not None
                    and cfg.w_pde_spectrum_shape_multistep_self > 0
                    and not pde_spectrum_shape_multistep_self_applied_this_epoch
                    and pde_head.mode == "markovian"
                ):
                    # See Stage1TrainingConfig.w_pde_spectrum_shape_
                    # multistep_self's docstring (Section 193) -- SELF-
                    # rollout-sampled analogue of
                    # w_pde_spectrum_shape_multistep.
                    pde_spectrum_shape_multistep_self_applied_this_epoch = True
                    z0_pool = windows[:, 0].reshape(-1, d).detach()
                    n_sample = min(cfg.pde_spectrum_shape_self_n_samples, z0_pool.shape[0])
                    sample_idx = torch.randperm(z0_pool.shape[0], device=device)[:n_sample]
                    z0_self = z0_pool[sample_idx].float()
                    z_self_states = _pde_head_self_rollout_states(
                        pde_head, z0_self, cfg.pde_spectrum_shape_self_rollout_k
                    )
                    if z_self_states is not None:
                        l_pde_spectrum_shape_multistep_self = propagator_multistep_spectrum_shape_loss(
                            pde_head.step_one, z_self_states,
                            k=cfg.pde_spectrum_shape_multistep_k,
                            n_expand=cfg.pde_spectrum_shape_n_expand,
                            expand_target=cfg.pde_spectrum_shape_expand_target,
                            contract_floor=cfg.pde_spectrum_shape_contract_floor,
                        )
                        loss = loss + (
                            cfg.w_pde_spectrum_shape_multistep_self * l_pde_spectrum_shape_multistep_self
                        )
                if pde_head is not None and cfg.w_pde_mean_conservation > 0:
                    # See Stage1TrainingConfig.w_pde_mean_conservation's
                    # docstring -- mirrors true KS's exact int(u)dx
                    # conservation law, evaluated on real detached states,
                    # never mutual with the propagator.
                    z_for_mean = windows.reshape(-1, d).detach()
                    l_pde_mean = pde_head_field_output(pde_head, z_for_mean).mean(dim=-1).pow(2).mean()
                    loss = loss + cfg.w_pde_mean_conservation * l_pde_mean
                if pde_head is not None and cfg.w_pde_energy_floor > 0:
                    # See Stage1TrainingConfig.w_pde_energy_floor's
                    # docstring -- UNSUPERVISED (no real target beyond
                    # the starting state), detached from the propagator,
                    # k_energy_now is the per-epoch RAMPED horizon.
                    e_start = windows[:, n_hist - 1].detach()
                    z_energy_roll = pde_head.rollout(e_start, e_start, k_energy_now)  # (b, k_energy, d)
                    if torch.isfinite(z_energy_roll).all():
                        l_pde_energy_floor = spatial_energy_floor_loss(
                            z_energy_roll.reshape(-1, z_energy_roll.shape[-1]), cfg.pde_energy_floor_gamma
                        )
                        loss = loss + cfg.w_pde_energy_floor * l_pde_energy_floor
                    else:
                        # See train_stage1's identical guard -- same
                        # reasoning (Section 170's own postmortem).
                        print(
                            f"  [stage2] WARNING: pde_head energy-floor rollout produced "
                            f"non-finite values at k_energy_now={k_energy_now} (epoch {epoch}) "
                            f"-- skipping this batch's energy-floor contribution.",
                            flush=True,
                        )
                        l_pde_energy_floor = torch.zeros((), device=z_energy_roll.device)
                if pde_head is not None and cfg.w_pde_coeff_l1 > 0:
                    # See Stage2TrainingConfig.w_pde_coeff_l1's/
                    # pde_coeff_l1_linear_only's docstrings.
                    if cfg.pde_coeff_l1_linear_only:
                        l_pde_coeff_l1 = pde_head_linear_coeffs(pde_head).abs().sum()
                    else:
                        l_pde_coeff_l1 = pde_head_poly_coeffs(pde_head).abs().sum()
                    loss = loss + cfg.w_pde_coeff_l1 * l_pde_coeff_l1

            if (
                cfg.w_spectrum_shape > 0
                and not spectrum_shape_applied_this_epoch
                and propagator.mode == "markovian"
                and not freeze_propagator
            ):
                # See Stage2TrainingConfig.w_spectrum_shape's docstring --
                # same expensive/once-per-epoch/outside-autocast convention
                # as Stage 1's copy, but evaluated on propagator.step_one
                # directly (no rollout) using real, UNNOISED encoded states
                # drawn from this batch's own windows.
                spectrum_shape_applied_this_epoch = True
                z_pool = windows.reshape(-1, d)
                n_sample = min(cfg.spectrum_shape_n_samples, z_pool.shape[0])
                sample_idx = torch.randperm(z_pool.shape[0], device=device)[:n_sample]
                z_sample = z_pool[sample_idx].float()
                l_spectrum_shape = propagator_spectrum_shape_loss(
                    propagator.step_one, z_sample,
                    n_expand=cfg.spectrum_shape_n_expand,
                    expand_target=cfg.spectrum_shape_expand_target,
                    contract_floor=cfg.spectrum_shape_contract_floor,
                    two_sided=cfg.spectrum_shape_two_sided,
                )
                loss = loss + cfg.w_spectrum_shape * l_spectrum_shape

            if (
                cfg.w_spectrum_shape_self > 0
                and not spectrum_shape_self_applied_this_epoch
                and propagator.mode in ("markovian", "history")
                and not freeze_propagator
            ):
                # See Stage2TrainingConfig.w_spectrum_shape_self's
                # docstring -- SELF-rollout-sampled analogue of
                # w_spectrum_shape, applied to the MAIN propagator (the
                # candidate left untried after Section 203's rerun: see
                # docs/OPEN_QUESTIONS.md). Same once-per-epoch/outside-
                # autocast convention; `_propagator_spectrum_shape_self_
                # pool` dispatches on propagator.mode and returns None on
                # a non-finite self-rollout (skip this batch's
                # contribution, same discipline as w_pde_energy_floor).
                spectrum_shape_self_applied_this_epoch = True
                pool = _propagator_spectrum_shape_self_pool(
                    propagator, windows, n_hist, cfg.spectrum_shape_self_rollout_k,
                    cfg.spectrum_shape_self_n_samples, device,
                )
                if pool is not None:
                    z_pool_self, step_fn_self = pool
                    l_spectrum_shape_self = propagator_spectrum_shape_loss(
                        step_fn_self, z_pool_self,
                        n_expand=cfg.spectrum_shape_n_expand,
                        expand_target=cfg.spectrum_shape_expand_target,
                        contract_floor=cfg.spectrum_shape_contract_floor,
                        two_sided=cfg.spectrum_shape_two_sided,
                    )
                    loss = loss + cfg.w_spectrum_shape_self * l_spectrum_shape_self

            if (
                cfg.w_spectrum_shape_graded > 0
                and not spectrum_shape_graded_applied_this_epoch
                and propagator.mode == "markovian"
                and not freeze_propagator
            ):
                # See Stage2TrainingConfig.w_spectrum_shape_graded's
                # docstring -- same convention as w_spectrum_shape above,
                # using the per-rank reference loaded at the top of this
                # function instead of a two-group split.
                spectrum_shape_graded_applied_this_epoch = True
                z_pool = windows.reshape(-1, d)
                n_sample = min(cfg.spectrum_shape_graded_n_samples, z_pool.shape[0])
                sample_idx = torch.randperm(z_pool.shape[0], device=device)[:n_sample]
                z_sample = z_pool[sample_idx].float()
                l_spectrum_shape_graded = propagator_graded_spectrum_shape_loss(
                    propagator.step_one, z_sample, spectrum_shape_graded_ref,
                )
                loss = loss + cfg.w_spectrum_shape_graded * l_spectrum_shape_graded

            if (
                cfg.w_spectrum_shape_multistep > 0
                and not spectrum_shape_multistep_applied_this_epoch
                and propagator.mode == "markovian"
                and not freeze_propagator
            ):
                # See Stage2TrainingConfig.w_spectrum_shape_multistep's
                # docstring (Section 186) -- same convention as
                # w_spectrum_shape above, but constrains the COMPOSED
                # k-step Jacobian's spectrum rather than the one-step
                # Jacobian's, to catch persistent expansive-subspace
                # self-feeding a one-step check cannot see.
                spectrum_shape_multistep_applied_this_epoch = True
                z_pool = windows.reshape(-1, d)
                n_sample = min(cfg.spectrum_shape_multistep_n_samples, z_pool.shape[0])
                sample_idx = torch.randperm(z_pool.shape[0], device=device)[:n_sample]
                z_sample = z_pool[sample_idx].float()
                l_spectrum_shape_multistep = propagator_multistep_spectrum_shape_loss(
                    propagator.step_one, z_sample,
                    k=cfg.spectrum_shape_multistep_k,
                    n_expand=cfg.spectrum_shape_n_expand,
                    expand_target=cfg.spectrum_shape_expand_target,
                    contract_floor=cfg.spectrum_shape_contract_floor,
                )
                loss = loss + cfg.w_spectrum_shape_multistep * l_spectrum_shape_multistep

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if params:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            epoch_loss += loss.detach()
            if pde_head is not None and cfg.w_pde_distill > 0:
                epoch_pde_distill += l_pde_distill.detach()
            if pde_head is not None and cfg.w_pde_rollout > 0:
                epoch_pde_rollout += l_pde_rollout.detach()
            if pde_head is not None and cfg.w_pde_distill_real > 0:
                epoch_pde_distill_real += l_pde_distill_real.detach()
            if pde_head is not None and cfg.w_pde_distill_real_rollout > 0:
                epoch_pde_distill_real_rollout += l_pde_distill_real_rollout.detach()
            if pde_head is not None and cfg.w_pde_nonlinear_l2 > 0:
                epoch_pde_nonlinear += l_pde_nonlinear.detach()
            if pde_head is not None and cfg.w_pde_mean_conservation > 0:
                epoch_pde_mean += l_pde_mean.detach()
            if pde_head is not None and cfg.w_pde_energy_floor > 0:
                epoch_pde_energy += l_pde_energy_floor.detach()
            if cfg.w_prop_energy_floor > 0:
                epoch_prop_energy += l_prop_energy_floor.detach()
            if pde_head is not None and cfg.w_pde_coeff_l1 > 0:
                epoch_pde_coeff_l1 += l_pde_coeff_l1.detach()
            n_batches += 1
        sched.step()

        eval_result = eval_stage2_kmax(propagator, val_sequences, cfg.k_max, device, compare_k=cfg.compare_k)
        if freeze_propagator:
            # eval_stage2_kmax unconditionally restores .train() on exit --
            # undo that here so a frozen propagator's dropout (if any) stays
            # OFF for the rest of training too, matching a genuinely fixed
            # teacher rather than a stochastic one.
            propagator.eval()
        val_kmax_mse, val_compare_mse = eval_result if cfg.compare_k is not None else (eval_result, None)
        if val_kmax_mse < best_val:
            best_val = val_kmax_mse
            best_state = {k: v.clone() for k, v in propagator.state_dict().items()}
        epoch_seconds = time.time() - epoch_t0
        pde_distill_active = pde_head is not None and cfg.w_pde_distill > 0
        pde_rollout_active = pde_head is not None and cfg.w_pde_rollout > 0
        pde_distill_mean = (epoch_pde_distill / n_batches).item() if pde_distill_active else None
        pde_rollout_mean = (epoch_pde_rollout / n_batches).item() if pde_rollout_active else None
        pde_distill_real_active = pde_head is not None and cfg.w_pde_distill_real > 0
        pde_distill_real_mean = (epoch_pde_distill_real / n_batches).item() if pde_distill_real_active else None
        pde_distill_real_rollout_active = pde_head is not None and cfg.w_pde_distill_real_rollout > 0
        pde_distill_real_rollout_mean = (
            (epoch_pde_distill_real_rollout / n_batches).item() if pde_distill_real_rollout_active else None
        )
        pde_nonlinear_active = pde_head is not None and cfg.w_pde_nonlinear_l2 > 0
        pde_nonlinear_mean = (epoch_pde_nonlinear / n_batches).item() if pde_nonlinear_active else None
        pde_mean_active = pde_head is not None and cfg.w_pde_mean_conservation > 0
        pde_mean_mean = (epoch_pde_mean / n_batches).item() if pde_mean_active else None
        pde_energy_active = pde_head is not None and cfg.w_pde_energy_floor > 0
        pde_energy_mean = (epoch_pde_energy / n_batches).item() if pde_energy_active else None
        prop_energy_active = cfg.w_prop_energy_floor > 0
        prop_energy_mean = (epoch_prop_energy / n_batches).item() if prop_energy_active else None
        pde_coeff_l1_active = pde_head is not None and cfg.w_pde_coeff_l1 > 0
        pde_coeff_l1_mean = (epoch_pde_coeff_l1 / n_batches).item() if pde_coeff_l1_active else None
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            compare_str = (
                f"  val_k{cfg.compare_k}_mse {val_compare_mse:.6f}" if cfg.compare_k is not None else ""
            )
            # `pde_distill`/`pde_rollout` (added 2026-09-08, user-directed:
            # "can we report the pde loss in the logs for training, just to
            # see if it's going down") -- RAW (unweighted) losses, so their
            # trend is comparable across different weight settings.
            pde_suffix = ""
            if pde_distill_active:
                pde_suffix += f"  pde_distill {pde_distill_mean:.6f}"
            if pde_rollout_active:
                pde_suffix += f"  pde_rollout {pde_rollout_mean:.6f}"
            if pde_distill_real_active:
                pde_suffix += f"  pde_distill_real {pde_distill_real_mean:.6f}"
            if pde_distill_real_rollout_active:
                pde_suffix += f"  pde_distill_real_rollout {pde_distill_real_rollout_mean:.6f}"
            if pde_nonlinear_active:
                pde_suffix += f"  pde_nonlinear {pde_nonlinear_mean:.6f}"
            if pde_mean_active:
                pde_suffix += f"  pde_mean {pde_mean_mean:.6f}"
            if pde_energy_active:
                pde_suffix += f"  pde_energy {pde_energy_mean:.6f}"
            if prop_energy_active:
                pde_suffix += f"  prop_energy {prop_energy_mean:.6f}"
            if pde_coeff_l1_active:
                pde_suffix += f"  pde_coeff_l1 {pde_coeff_l1_mean:.6f}"
            print(
                f"  [stage2] epoch {epoch:3d}/{cfg.epochs}  k {k_now:2d}  "
                f"lr {sched.get_last_lr()[0]:.2e}  loss {(epoch_loss / n_batches).item():.6f}  "
                f"noise_step {noise_step:.4f}  val_kmax_mse {val_kmax_mse:.6f}{compare_str}{pde_suffix}  "
                f"[{epoch_seconds:.1f}s]",
                flush=True,
            )

        history_entry = {
            "epoch": epoch,
            "loss": (epoch_loss / n_batches).item(),
            "k_now": k_now,
            "noise_in": noise_in,
            "noise_step": noise_step,
            "seconds": epoch_seconds,
            "val_kmax_mse": val_kmax_mse,
        }
        if cfg.compare_k is not None:
            history_entry[f"val_k{cfg.compare_k}_mse"] = val_compare_mse
        if pde_distill_active:
            history_entry["pde_distill"] = pde_distill_mean
        if pde_rollout_active:
            history_entry["pde_rollout"] = pde_rollout_mean
        if pde_distill_real_active:
            history_entry["pde_distill_real"] = pde_distill_real_mean
        if pde_distill_real_rollout_active:
            history_entry["pde_distill_real_rollout"] = pde_distill_real_rollout_mean
        if pde_nonlinear_active:
            history_entry["pde_nonlinear"] = pde_nonlinear_mean
        if pde_mean_active:
            history_entry["pde_mean"] = pde_mean_mean
        if pde_energy_active:
            history_entry["pde_energy"] = pde_energy_mean
        if prop_energy_active:
            history_entry["prop_energy"] = prop_energy_mean
        if pde_coeff_l1_active:
            history_entry["pde_coeff_l1"] = pde_coeff_l1_mean
        history.append(history_entry)
        if checkpoint_every > 0 and on_epoch_end is not None and (epoch + 1) % checkpoint_every == 0:
            on_epoch_end(epoch)

    propagator.load_state_dict(best_state)
    return Stage2Result(train_history=history, best_val_kmax_mse=best_val, best_state_dict=best_state)
