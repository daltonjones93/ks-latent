"""Stage-1/Stage-2 loss terms (brief §5.1-5.2, PROJECT_HANDOFF.md).

`L_pred` targets are physical-space ground truth, never latent-space
targets. A latent-space target `z*_{n+k} = E(u_{n+k})` moves as the encoder
trains, which lets the encoder cheat by collapsing to an easy-to-predict
(e.g. near-constant) representation that also satisfies its own
ever-shifting target -- a real degeneracy observed in earlier iterations of
this project (`docs/ML_for_KS_writeup.md` §3.4/§6). Decoding the rollout
back to physical space before comparing removes that degree of freedom: the
ground truth `u_{n+k}` is fixed throughout training.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap

from ks_latent.models.spectral_field import encode_to_spectrum, rfft_wavenumbers


def reconstruction_loss(u_hat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Mean over all elements of `(u_hat - u)**2`; equals the brief's "mean
    over snapshots of ||D(E(u))-u||^2 / NX" once snapshots are flattened
    into the batch dimension."""
    return F.mse_loss(u_hat, u)


def batch_covariance(z: torch.Tensor) -> torch.Tensor:
    """Sample covariance of a `(B, d)` batch of latent vectors."""
    zc = z - z.mean(dim=0, keepdim=True)
    B = z.shape[0]
    return (zc.T @ zc) / (B - 1)


def decorr_var_loss(
    z: torch.Tensor, var_target: torch.Tensor | float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """`L_decorr = (1/d^2) sum_{i!=k} C_ik^2`, `L_var = (1/d) sum (C_ii-var_target_i)^2`.

    `var_target` (added 2026-08-31, user-directed -- Phase 2 architecture
    doc Section 35, option 7): per-dimension target for `l_var`, default
    `1.0` (original behavior, uniform across all dims). Added because a
    HARDCODED uniform target of `1` is only correct if the input `z` is
    already unit-variance per channel; `Stage2TrainingConfig.w_varmatch`
    applies this to a Stage-1 AE's raw latent `z_pred`, whose real
    per-channel variance can range over many orders of magnitude (observed
    ~6 down to ~2e-6 in one AE) -- for any channel far from variance 1,
    the uniform target actively fights the primary latent-matching loss
    instead of complementing it. Pass the AE's own real per-channel
    variance (e.g. `train_sequences.var(dim=(0,1))`) as `var_target` to
    preserve the original anti-fixed-point-collapse motivation (matching
    real cross-initial-condition diversity) without that conflict."""
    d = z.shape[1]
    cov = batch_covariance(z)
    diag = torch.diagonal(cov)
    off_diag_sq_sum = (cov**2).sum() - (diag**2).sum()
    l_decorr = off_diag_sq_sum / (d * d)
    l_var = ((diag - var_target) ** 2).mean()
    return l_decorr, l_var


def variance_floor_loss(z: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Per-channel variance-FLOOR anti-collapse regularizer (VICReg-style;
    added 2026-08-31, user-directed -- Phase 2 architecture doc Section 35
    option 3a, following the discovery that extended joint Stage-1
    fine-tuning/training can collapse one latent channel's variance many
    orders of magnitude below the rest (observed `~2.4e-6` vs. a top
    eigenvalue of `~6`), wrecking Stage-2 propagator fitting at long
    rollout horizons despite BETTER structural (D4/D6/D7) diagnostics).

    `L = mean_i( relu(gamma - std_i) ^ 2 )`, `std_i = std(z[:, i])` over the
    batch. Zero whenever every channel's std is already `>= gamma`; grows
    quadratically as a channel's std falls below `gamma`. Deliberately NOT
    the same mechanism as `decorr_var_loss`'s existing `l_var` term
    (`Stage1TrainingConfig.w_var`, already on by default at `0.01`): that
    term pulls EVERY channel toward variance exactly `1`, including
    channels that naturally want to be much larger (the observed spectrum
    ranges up to `~6`) -- so it fights healthy strong channels too, not
    just collapsing weak ones, and its gradient on any one channel is
    diluted by the `mean` over all `d` channels. A hinge-style floor pays
    NOTHING for a channel already at or above `gamma` (no matter how far
    above), so it doesn't compete with reconstruction's own preference for
    that channel's natural scale, and its full gradient magnitude applies
    to whichever channels are actually near collapse. Choose `gamma` well
    below `1` (e.g. `0.05-0.2`) -- it's a floor against total collapse, not
    a target to pull every channel toward.

    Does NOT address a purely correlation-driven joint eigenvalue collapse
    (a channel with healthy marginal variance but near-total linear
    dependence on another) -- see `logdet_barrier_loss` for that case."""
    std = z.std(dim=0, unbiased=True)
    return torch.relu(gamma - std).pow(2).mean()


def spatial_energy_floor_loss(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """PER-SAMPLE spatial-energy-FLOOR anti-decay regularizer (added
    2026-09-13, user-directed: viewing pde_head's own free-running smooth-
    field GIF (Section 167), "I also see the decay in the pde trajectory
    towards 0... would it be worth trying to fix the maximum energy in
    the pde in physical space?... maybe we want to try to force the model
    to always have an energetic component").

    `L = mean( relu(gamma - std_across_ring(z)) ^ 2 )`, where
    `std_across_ring(z) = z.std(dim=-1)` is EACH sample's own std ACROSS
    its `d_latent`/ring index (a spatial-energy measure of that ONE
    state), not across the batch. This is a DIFFERENT axis, and a
    different failure mode, from `variance_floor_loss`
    (per-CHANNEL std across the BATCH, guarding against one latent
    dimension collapsing relative to the others) and `decorr_var_loss`'s
    `l_var`/`Stage2TrainingConfig.w_varmatch` (per-channel variance
    across DIFFERENT initial conditions, guarding against distinct
    trajectories converging to the same fixed point). Here, a SINGLE
    trajectory's own spatial profile can decay toward spatially-uniform
    (std -> 0) as it evolves in time, which none of those other terms
    would ever penalize (they operate on different axes entirely) --
    exactly what was observed in Section 167's own standalone pde_head
    rollout: no divergence, no mode-0 lock-in (both already fixed), but
    the smoothed field's own peak amplitude decayed steadily over 200
    steps (real data's own spatial std averages ~0.97, minimum ~0.53,
    measured directly on Section 167's real encoded z -- `gamma` should
    sit clearly below that range, not at it).

    Same hinge-floor construction as `variance_floor_loss` (pays nothing
    once std_across_ring is already `>= gamma`, so it doesn't fight
    genuine decay toward a smaller-but-still-energetic attractor, only
    decay toward near-zero) -- applied to whatever rollout tensor the
    caller passes in (shape `(..., d_latent)`, reduced over the last
    dim; callers reshape a `(B, k, d)` rollout to `(B*k, d)` first so
    every step of the rollout is penalized, not just the final one)."""
    spatial_std = z.std(dim=-1, unbiased=True)
    return torch.relu(gamma - spatial_std).pow(2).mean()


def logdet_barrier_loss(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Full-covariance log-det anti-collapse regularizer (added 2026-08-31,
    user-directed -- Phase 2 architecture doc Section 35 option 3b, "more
    principled, catches correlation-driven collapse too"). `L =
    -logdet(Cov(z) + eps*I) / d`, to be MINIMIZED -- equivalent (up to an
    additive constant) to minimizing the negative differential entropy of
    a Gaussian fit to `z`'s per-batch distribution, so minimizing it
    pushes the WHOLE covariance eigenspectrum away from zero, not just the
    diagonal. Unlike `variance_floor_loss` (which only constrains MARGINAL
    per-channel variance and can miss a channel that has normal individual
    variance but is nearly a linear combination of others, hence a small
    JOINT eigenvalue), this term reacts to the smallest eigenvalue of the
    full covariance regardless of whether it comes from one channel's own
    variance or from cross-channel redundancy -- the more general, but
    also more novel/untested-in-this-codebase, of the two mechanisms
    (no established local precedent the way `variance_floor_loss`'s
    VICReg-style form has). `eps` regularizes the covariance to keep
    `slogdet` well-defined/finite even from a small batch or a batch with
    a genuinely singular covariance; it does not need tuning to match any
    particular scale -- it only needs to be small relative to a healthy
    channel's variance. Divided by `d` to keep the loss's scale roughly
    comparable across different `d_latent`, matching this module's other
    per-dimension-normalized terms (`l_decorr`, `l_var`).

    Explicitly upcasts to `float32` before `slogdet` (added 2026-09-01,
    found via a real crash under `--amp`): unlike `softmax`/`layer_norm`,
    `torch.linalg.slogdet` is not on autocast's automatic upcast list --
    it raises outright (`Low precision dtypes not supported`) on a
    `bfloat16` input rather than silently losing precision, so this needs
    an explicit cast rather than relying on autocast's usual per-op
    policy."""
    d = z.shape[1]
    z = z.float()
    cov = batch_covariance(z)
    cov_reg = cov + eps * torch.eye(d, device=z.device, dtype=z.dtype)
    _, logdet = torch.linalg.slogdet(cov_reg)
    return -logdet / d


def spatial_coherence_loss(
    z: torch.Tensor, bandwidth: float = 3.0, eps: float = 1e-3, signed: bool = False
) -> torch.Tensor:
    """Differentiable training-time version of D7 (`ks_latent.analysis.
    diagnostics.same_time_channel_correlation` -- user-directed 2026-08-31:
    "turn the D7 diagnostic into a loss ... push the model towards creating
    spatial coherence"). `z`: `(B, d)`, a batch of latent vectors from
    `encode()`. Returns `1 - bandedness_score`, to be MINIMIZED (so
    `bandedness_score`, in `[0, 1]`, is maximized).

    IMPORTANT PRECEDENT -- read before enabling at any real weight:
    `RegConfig.lambda_z`/`lambda_decorr` (`ks_latent/training/
    regularizer.py`) already implements a closely related idea --
    `BandedSmoothness` pulls index-nearby coordinates toward equal VALUES,
    `OffBandDecorrelation` explicitly counteracts the resulting collapse-
    to-a-constant degenerate optimum by penalizing far-pair correlation --
    and that TWO-PART, collapse-resistant-by-design mechanism is
    nonetheless documented (`RegConfig`'s own docstring) as collapsing the
    latent in practice at `lambda_z >= 5e-3` in the reference project. This
    function is NOT that mechanism (it rewards CORRELATION structure across
    a batch, not per-sample VALUE proximity -- see the distinction below),
    but the same class of risk (rewarding local structure can be cheaply
    satisfied by making nearby channels partially redundant, at a real cost
    to the latent's effective dimensionality) plausibly still applies. Test
    at a small weight with full Gate 3/4 monitoring (Lyapunov spectrum
    especially) before trusting any result from this at all -- do not
    enable it as a default, and do not assume it is safe merely because it
    isn't the identical mechanism already known to fail.

    Mechanism: build the batch's empirical `|Pearson correlation|` matrix
    `A` (same statistic D7 uses, but differentiable and computed fresh
    every batch, not from a large offline sample), weight it by a fixed
    circular-band Gaussian kernel `W` (same `w(dist) = exp(-dist^2 /
    (2*bandwidth^2))` D3/D6/D7's `bandedness()` uses) in the CURRENT, FIXED
    latent index order (no Fiedler search -- a loss needs the index itself,
    not some permutation of it, to become meaningful). Score ONLY the
    OFF-DIAGONAL entries: `bandedness_score = sum_{i!=j}(A*W) /
    sum_{i!=j}(A)` -- i.e. of whatever off-diagonal correlation mass
    exists, how concentrated is it near the diagonal versus far.

    A first version of this function normalized by `sum(A)` INCLUDING the
    diagonal, which is a real bug, not a style choice -- caught by
    `tests/unit/test_losses_spatial_coherence.py`'s very first test:
    `W(dist=0)=1` but `W(dist>0)<1`, so under that normalization ANY
    off-diagonal correlation at all -- even to an immediate neighbor --
    makes the ratio WORSE than having none, since the diagonal's fixed
    contribution (`d*1` to both numerator and denominator) gets diluted by
    off-diagonal terms that add less to the numerator (`A_ij * W_ij < A_ij`)
    than to the denominator (`A_ij`). That version's true global optimum
    was "zero correlation everywhere" -- pure decorrelation, which
    `w_decorr` already does -- not locality at all. Excluding the diagonal
    fixes this: `sum_{i!=j}` no longer has that fixed dominant term, so the
    ratio genuinely measures near-vs-far CONCENTRATION of whatever
    off-diagonal mass exists, and is well-defined (via `eps`-regularization
    against the `mean(W)` baseline) even when there is none.

    Collapse-safety, re-verified with the off-diagonal-only version: a
    FULLY COLLAPSED/redundant `z` (every channel identical, so `A_ij=1`
    for every `i!=j` too, not just the diagonal) scores exactly `mean(W)`
    over off-diagonal pairs -- a mediocre, not maximal, value (the maximum
    requires off-diagonal correlation concentrated at the SMALLEST
    distances specifically, which uniform full collapse is not). This does
    not prove the loss is safe overall (locally-redundant-but-not-fully-
    collapsed optima remain possible, a softer version of the same class
    of risk `lambda_z` fell into) -- it only rules out the single specific
    degenerate optimum that sank the closest existing mechanism.

    `signed` (added 2026-09-01, user-directed, backs D8 -- see
    `ks_latent.analysis.diagnostics.signed_bandedness`): default `False`
    uses `A = |C|` as above (D7's statistic -- rewards coupling regardless
    of sign, so an anti-correlated near-neighbor scores identically to a
    correlated one). `True` uses `A = C` (signed) instead and a DIFFERENT
    formula -- `score = weighted_mean(A_off, W) - unweighted_mean(A_off)`
    (the Gaussian-circular-band-weighted mean of the off-diagonal signed
    correlations, minus their plain mean), returning `-score` (to
    minimize) rather than `1 - score`. This is NOT just "drop the abs from
    the existing ratio": an earlier attempt did exactly that and produces
    a mechanism failure caught by this module's own tests (see
    `signed_bandedness`'s docstring for the exact counterexample) -- a
    global sign bias in `A` can make the unsigned formula's numerator AND
    denominator both negative, so their ratio reports spuriously HIGH
    coherence for a matrix that's actually anti-coherent near the
    diagonal. The weighted-minus-unweighted-mean form doesn't have that
    failure mode (a global sign bias shifts both terms together and
    cancels out of the difference), so it's used here too, for the same
    reason `spatial_coherence_loss`(signed=True) and D8 should agree on
    what "signed local coherence" means. Genuinely different optimum from
    the unsigned loss, not merely a stricter version of it -- test at a
    small weight with full Gate 3/4 monitoring, same as the unsigned
    version, before trusting a result from this."""
    d = z.shape[1]
    z_c = z - z.mean(dim=0, keepdim=True)
    std = z_c.std(dim=0, keepdim=True).clamp_min(1e-6)
    z_n = z_c / std
    C = (z_n.T @ z_n) / z.shape[0]
    off_diag = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    idx = torch.arange(d, device=z.device, dtype=torch.float32)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    dist = torch.minimum(diff, d - diff)
    W = torch.exp(-(dist**2) / (2.0 * bandwidth**2))
    if signed:
        A_off = C * off_diag
        W_off = W * off_diag
        weighted_mean = (A_off * W_off).sum() / W_off.sum()
        unweighted_mean = A_off.sum() / off_diag.sum()
        return -(weighted_mean - unweighted_mean)
    A = C.abs()
    A_off = A * off_diag
    numer = (A_off * W).sum()
    denom = A_off.sum()
    baseline = (W * off_diag).mean()  # mean(W) over off-diagonal pairs
    # eps-regularized: when there is ~no off-diagonal correlation at all
    # (denom -> 0), the score falls back to the uninformative `baseline`
    # rather than being an arbitrary/unstable 0/0.
    bandedness_score = (numer + eps * baseline) / (denom + eps)
    return 1.0 - bandedness_score


def temporal_smoothness_loss(z_win: torch.Tensor, curvature_weight: float = 1.0) -> torch.Tensor:
    """Differentiable training-time version of
    `scripts/analyze_latent_smoothness.py`'s encoded-real-trajectory step
    size / curvature measures (added 2026-09-04, user-directed: "is there
    a way to encode the test function here analyze_latent_smoothness.py
    as a regularization term ... implement that" -- following the
    finding that Section 82's GIF-visible "huge jumps" were confirmed
    quantitatively by that diagnostic, and that `w_spatial`/D7/D8
    regularize a DIFFERENT axis entirely -- CROSS-SECTIONAL correlation
    between latent-index channels at a single instant, with no term
    involving `z_t` vs `z_{t+1}` at all).

    `z_win`: `(B, window, d)`, a batch of CONSECUTIVE real encoded
    states -- exactly `train_stage1`'s existing `z_win` (already
    available every batch for `L_pred`, no new data loading needed).
    `L = mean(||z_{t+1}-z_t||^2) / d + curvature_weight *
    mean(||z_{t+1}-2*z_t+z_{t-1}||^2) / d` (the second term contributes
    exactly `0` if `window < 3`, e.g. `mode="markovian"` with
    `k_pred=1`). Divided by `d` to keep the loss's scale roughly
    comparable across different `d_latent`, matching this module's other
    per-dimension-normalized terms (`l_decorr`, `l_var`,
    `logdet_barrier_loss`).

    Deliberately NOT normalized by each trajectory's own RMS latent
    radius (unlike the read-only diagnostic script's version, which
    needs that to compare ACROSS sections with very different overall
    latent scales) -- here there is only one model being trained, and
    keeping the loss in absolute units lets it compose additively with
    `w_recon`/`w_var`/`w_var_floor` exactly like every other regularizer
    in this module, rather than introducing a second, gradient-visible
    scale-dependence that could let the model satisfy this loss by
    inflating overall latent variance instead of actually smoothing the
    trajectory.

    COLLAPSE RISK (read before enabling at any real weight, same caution
    as `spatial_coherence_loss`'s and `variance_floor_loss`'s
    docstrings): minimizing raw step size in absolute terms is trivially
    satisfied by an encoder that maps every physical state to nearly the
    same latent vector -- unlike `spatial_coherence_loss` (which rewards
    a CORRELATION structure, not proximity to a fixed point), this term's
    global optimum in isolation genuinely IS latent collapse. It is only
    safe to enable alongside the existing anti-collapse pressure
    (`w_recon` always active; `w_var`/`w_var_floor`/`w_logdet` as
    needed) that already keeps every other regularizer in this module
    honest -- test at a small `w_smooth` with full Gate 3/4 monitoring
    before trusting any result, same as every other regularizer here."""
    d = z_win.shape[-1]
    step = z_win[:, 1:] - z_win[:, :-1]
    l_step = step.pow(2).sum(dim=-1).mean() / d
    if z_win.shape[1] < 3:
        return l_step
    curv = z_win[:, 2:] - 2.0 * z_win[:, 1:-1] + z_win[:, :-2]
    l_curv = curv.pow(2).sum(dim=-1).mean() / d
    return l_step + curvature_weight * l_curv


def low_pass_spectral_loss(z: torch.Tensor, K: int, L: float, power: float = 1.0) -> torch.Tensor:
    """`encoder_kind="spectral_field"` only (added 2026-09-06, see
    docs/sine_transform_pde_plan.md, user-directed: "add a regularizer that
    acts as a low pass filter. Namely, it penalizes the higher frequency
    terms of z proportional to their frequency").

    `z`: `(B, 2*K)`, the SAME truncated-rFFT convention
    `ks_latent.models.spectral_field.encode_to_spectrum` produces (`K` real
    parts concatenated with `K` imaginary parts of the kept rFFT modes).
    `L`: the physical domain length the modes' wavenumbers are computed
    against (must match the training dataset's own `KSConfig.L`).

    `loss = mean_batch[ sum_k k^power * |z_k|^2 ]`, `k = 2*pi*m/L` the
    physical angular wavenumber of mode `m` (`ks_latent.models.
    spectral_field.rfft_wavenumbers`). `power=1` matches the literal
    "proportional to frequency" request; `power=2` is the classical `H^1`
    Sobolev seminorm (`sum_k k^2*|z_k|^2 = integral (dw/dx)^2 dx` by
    Parseval's theorem) -- both are exposed via `Stage1TrainingConfig.
    lowpass_power` rather than assuming either is obviously the right
    choice.

    On top of the HARD truncation to `K` modes this architecture already
    imposes (dropped modes are exactly, not approximately, absent), this is
    a SOFT penalty pushing energy toward the lowest few of the kept modes --
    a second, finer-grained smoothness lever, not a redundant one. This
    mechanism is mathematically the same operation as the dealiasing/
    hyperviscosity filters classical pseudo-spectral PDE solvers apply to
    suppress spurious high-wavenumber energy from pointwise nonlinear
    products (see the plan doc's §2.2-2.3) -- it serves both the
    "smooth enough to fit a PDE to" goal and a classical numerical-
    stability role at once."""
    real, imag = z[:, :K], z[:, K : 2 * K]
    energy = real.pow(2) + imag.pow(2)  # (B, K), |z_k|^2 per mode
    k = rfft_wavenumbers(K, L, device=z.device, dtype=z.dtype)  # (K,)
    weight = k.pow(power)
    return (energy * weight[None, :]).sum(dim=-1).mean()


def latent_self_spectrum_lowpass_loss(z: torch.Tensor, K: int, L: float, power: float = 1.0) -> torch.Tensor:
    """Generalizes `low_pass_spectral_loss` (which requires `z` to already
    BE a truncated rFFT spectrum, `encoder_kind="spectral_field"` only) to
    the raw latent `z: (B, d_latent)` of ANY encoder -- added 2026-09-11,
    user-directed: "I would like to penalize higher frequencies in the
    fourier transform of the latent space. This may help make the pde
    dynamics more easily fit and smooth."

    Directly motivated by Sections 146-148: a `backbone="spectral_pde_raw"`
    `pde_head` (which treats `z`'s own `d_latent` index as a periodic ring
    and takes ITS OWN self-FFT via `encode_to_spectrum`, see that
    function's docstring and `_SpectralPDERawDeltaBody`) trained MUTUALLY
    against a `masked_mlp_expand` propagator STALLED when the pde_head's
    own self-FFT was truncated to the lowest half of `z`'s spectrum
    (`--pde-K 24`, Section 146/147), while training healthily at the full
    spectrum (`--pde-K` unset, Section 148) -- strongly suggesting `z`'s
    own high-self-frequency content is real signal the pde_head needs, not
    safely-discardable noise, AT LEAST for that specific (masked, narrow-
    receptive-field) propagator. This loss instead attacks the DYNAMICS
    directly, penalizing the RAW propagator/encoder for producing a `z`
    whose own self-spectrum has high-frequency content in the first
    place, rather than truncating what any one pde_head is allowed to see
    downstream -- a difference with real consequences: `--pde-K`
    truncation is a downstream architectural constraint the encoder/
    propagator have no visibility into or pressure to accommodate, while
    this loss provides an actual TRAINING gradient pushing `z` itself
    toward the smoother self-spectrum a low-order polynomial closure can
    fit well.

    Mechanism: computes `z`'s own truncated self-FFT via
    `ks_latent.models.spectral_field.encode_to_spectrum(z, K)` (the SAME
    "native index as space" convention `spectral_pde_raw`/
    `spatial_coherence_loss` already use -- treating the `d_latent` index
    as a periodic ring of assumed circumference `L`), then applies the
    IDENTICAL frequency-weighted energy penalty `low_pass_spectral_loss`
    already implements to that self-spectrum. `K`/`L` here describe THIS
    self-FFT (typically `d_latent//2+1`/`d_latent`, matching a
    `spectral_pde_raw` pde_head's own defaults so the encoder is
    pressured toward exactly the spectral shape that pde_head will see),
    NOT any encoder-specific spectral truncation the encoder itself might
    separately impose (e.g. `encoder_kind="spectral_field"`'s own `K`/`L`,
    an unrelated, physical-space FFT of `u` -- these two are conceptually
    different transforms that happen to share a truncation parameter
    name)."""
    z_hat = encode_to_spectrum(z, K)
    return low_pass_spectral_loss(z_hat, K, L, power=power)


def local_field_channel_mean_loss(z: torch.Tensor, n_sites: int, local_channels: int) -> torch.Tensor:
    """`encoder_kind="local_field"` only (added 2026-09-11, user-directed:
    "implement the root fix"). Penalizes each RESIDUAL channel (1..
    local_channels-1 -- channel 0 is the gauge-anchored physical average,
    excluded) for having a nonzero mean ACROSS SITES, per sample:
    `mean_i[ (mean_site z_field[:, :, c])^2 ]` over residual channels `c`.

    Motivation, found directly on a real trained checkpoint (Section 153):
    channel 0 (anchor) has mean 0.000 across sites, but the two free
    residual channels had drifted to means of 7.494 and 8.607 -- an order
    of magnitude larger than genuine per-site physical variation (std
    ~0.8-1.3). Nothing in this project's OTHER anti-collapse regularizers
    (w_var/w_logdet/w_decorr) constrains this -- they all act on SCALE/
    covariance, never on MEAN. Because `z` is flattened SITE-MAJOR (period
    `local_channels` in the raw index), that offset is a near-pure
    period-`local_channels` signal, which aliases onto EXACTLY one
    self-FFT mode (`n_sites`, e.g. 32 for `n_sites=32`) -- confirmed
    empirically to be the SAME dominant mode a trained pde_head's own
    standalone rollout collapses onto (>98% of its energy, for every one
    of 10 different real initial conditions tested), a pure flattening
    ARTIFACT unrelated to real KS structure.

    Deliberately a TRAINING-TIME LOSS, not an architectural (forward-pass)
    fix: a first attempt subtracted each residual channel's own per-sample
    across-site mean directly inside `KSAutoencoderLocalField.encode()`,
    which is a GLOBAL (all-`n_sites`) reduction -- it made every output
    site depend on every input site, destroying the encoder's core
    architectural locality guarantee (caught immediately by
    `test_receptive_field_bounded_and_matches_analytic_formula`). Setting
    `enc_out`'s own Conv1d bias to `False` (a purely local, parameter-
    level fix -- verified NOT to break that test) was ALSO tried and
    verified empirically NOT SUFFICIENT on its own: GELU is not a
    zero-mean-preserving nonlinearity, so a persistent per-channel offset
    re-emerges through the conv stack regardless of the final layer's own
    bias. A soft, loss-level penalty is the only way to discourage this
    representation WITHOUT touching the forward computation (and thus
    without touching locality) -- it only shapes what the encoder LEARNS
    to output, via gradient descent, exactly like every other regularizer
    in this training loop."""
    z_field = z.reshape(z.shape[0], n_sites, local_channels)
    residual_mean = z_field[:, :, 1:].mean(dim=1)  # (B, local_channels-1)
    return residual_mean.pow(2).mean()


def reference_mode_energy(u: torch.Tensor, K: int) -> torch.Tensor:
    """`encoder_kind="spectral_field"` only (added 2026-09-07, see
    docs/sine_transform_pde_plan.md, user-directed: "I really just want to
    come up with a regularizer to prevent the latent state from
    collapsing" -- following the finding that `w_var`/`w_logdet`/`w_decorr`
    all fight the ALREADY-CORRECT natural spectral energy decay of a real
    KS field, since they have no way to distinguish "healthy decay" from
    "pathological collapse").

    `u`: `(B, NX)` REAL physical-space snapshots (from the training data
    itself, not anything the model produces) -- computes the empirical
    per-mode energy of the TRUE field's own low-`K`-mode spectrum, giving a
    physically-grounded reference for what a non-collapsed latent's own
    per-mode energy PROPORTIONS should look like, calibrated from the real
    PDE rather than guessed. `u`'s own `rfft` mode `m` sits at the exact
    same physical wavenumber `k=2*pi*m/L` regardless of `u`'s own grid
    resolution `NX` (as long as `NX` resolves mode `K-1` without aliasing,
    trivially true here) -- the SAME physical wavenumbers `z`'s own kept
    modes represent, so this needs no rescaling to compare against `z`.

    Returns `(K,)`, normalized to sum to 1 (a proportion-of-kept-energy
    profile, not an absolute target -- `z`'s own overall scale is governed
    by `w_recon`/etc elsewhere; only the RELATIVE distribution across the
    `K` kept modes is meaningful here). See `spectral_shape_floor_loss`'s
    docstring for why this is used as a one-sided FLOOR, not an exact
    target to match."""
    coeffs = torch.fft.rfft(u.float(), dim=-1)[:, :K]
    energy = (coeffs.real.pow(2) + coeffs.imag.pow(2)).mean(dim=0)  # (K,)
    return energy / energy.sum().clamp_min(1e-12)


def spectral_shape_floor_loss(
    z: torch.Tensor, K: int, p_ref: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """`encoder_kind="spectral_field"` only (added 2026-09-07, see
    docs/sine_transform_pde_plan.md). `p_ref`: `(K,)`, the real-data-
    derived reference proportions from `reference_mode_energy` (precomputed
    ONCE from real training data, a fixed, non-learned target -- see
    `Stage1TrainingConfig.w_shape_floor`'s docstring for how it's wired
    into `train_stage1`).

    Answers a real subtlety directly: `u` has far more physical modes than
    `z` keeps (`K`), so when the encoder compresses `u -> w -> z`, genuine
    high-frequency structure from `u` that gets discarded doesn't just
    vanish -- the encoder plausibly needs to FOLD some of that information
    into the low modes it does keep, so `z`'s kept modes may legitimately
    need MORE energy than the real data's own bare low-`K`-mode slice
    alone would suggest, not the same amount. An exact shape-match penalty
    would fight exactly this necessary compensation. Instead, this is a
    ONE-SIDED floor: penalize a mode's proportional share of `z`'s own
    energy falling BELOW its real-data-derived reference share; never
    penalize it for exceeding that share. Extra "compensation" energy the
    encoder needs can freely land in any kept mode, above its own floor,
    completely unpenalized -- only genuine collapse (a mode falling short
    of what the real physics says it needs at minimum) is penalized.

    `loss = sum_k relu(log(p_ref_k + eps) - log(p_z_k + eps))^2`, `p_z`
    `z`'s own current per-mode energy proportions (same computation
    `low_pass_spectral_loss` uses internally), normalized the same way as
    `p_ref`."""
    real, imag = z[:, :K], z[:, K : 2 * K]
    energy = real.pow(2) + imag.pow(2)  # (B, K)
    p_z = energy.mean(dim=0)
    p_z = p_z / p_z.sum().clamp_min(1e-12)
    deficit = torch.log(p_ref + eps) - torch.log(p_z + eps)
    return F.relu(deficit).pow(2).sum()


def reference_temporal_separation(states: torch.Tensor, lag: int) -> torch.Tensor:
    """`states`: `(n_runs, T, D)` REAL data (from the training set itself,
    not anything the model produces) -- returns a scalar, the mean
    Euclidean distance between real states `lag` steps apart:
    `mean(||states[:, lag:] - states[:, :-lag]||)`. Used as a fixed,
    non-learned reference for `temporal_expansion_floor_loss` -- see that
    function's docstring and `Stage1TrainingConfig.w_temporal_floor_z`/
    `w_temporal_floor_w`'s for the full motivation. `T` must be `> lag`."""
    if states.shape[1] <= lag:
        raise ValueError(f"states.shape[1]={states.shape[1]!r} must be > lag={lag!r}")
    diffs = states[:, lag:] - states[:, :-lag]
    return diffs.norm(dim=-1).mean()


def temporal_expansion_floor_loss(
    states: torch.Tensor, lag: int, sep_ref: torch.Tensor
) -> torch.Tensor:
    """`states`: `(B, window, D)` -- the ENCODER's own real encoded
    windows (added 2026-09-09, user-directed: "why don't we just have a
    schedule that slowly ramps up w_pred loss" experiments led to a
    decisive diagnostic (Section 124): even with a propagator that is
    LITERALLY the exact analytic KS equation the whole time (zero learned
    dynamics at all), joint training under real `w_pred` pressure still
    collapsed the rollout to a fixed point. Since there was no learned
    dynamics to blame, this implicated the ENCODER itself: nothing in
    plain reconstruction loss stops it from mapping temporally-adjacent,
    causally-connected real states to NEARLY THE SAME `z` -- which
    trivially minimizes prediction error against ANY propagator
    (including a perfectly exact one) regardless of whether the real
    underlying dynamics are chaotic. `w_logdet_physical` (a BATCH-level,
    cross-snapshot covariance-rank check) did not fix this, because a
    batch mixing many unrelated snapshots can show full aggregate
    diversity while still flattening any SPECIFIC short real trajectory
    segment -- a different, more local granularity of collapse than a
    global covariance check can see.

    This is a direct, TEMPORAL-STRUCTURE-SPECIFIC fix: a ONE-SIDED floor
    (same design convention as `spectral_shape_floor_loss` -- calibrated
    from real data, never penalizes exceeding it) on how far apart, in
    representation space, real states `lag` real steps apart are allowed
    to become. `sep_ref` (from `reference_temporal_separation`, computed
    ONCE from real ground-truth data via the SAME fixed transform used
    everywhere else, entirely independent of the current encoder) is the
    TRUE system's own mean separation at this lag; `states` here supplies
    the ENCODER's own CURRENT real encoded windows at the same lag.
    Deliberately the opposite direction from `temporal_smoothness_loss`
    (which REWARDS small step-to-step differences -- exactly wrong for
    this failure mode): here, real states drifting apart over time is
    required, not penalized; only insufficient separation is penalized.

    `loss = mean(relu(sep_ref - ||states[:,lag:] - states[:,:-lag]||)^2)`.
    Works identically whether `states` is `z` (rFFT coefficients) or `w`
    (`decode_from_spectrum(z)`, the physical field) -- see
    `Stage1TrainingConfig.w_temporal_floor_z`/`w_temporal_floor_w`'s
    docstrings for the two scopings this project distinguishes elsewhere
    (`w_logdet` vs `w_logdet_physical`, `w_var` vs `w_var_physical`)."""
    if states.shape[1] <= lag:
        raise ValueError(f"states.shape[1]={states.shape[1]!r} must be > lag={lag!r}")
    diffs = states[:, lag:] - states[:, :-lag]
    actual_sep = diffs.norm(dim=-1)
    return F.relu(sep_ref - actual_sep).pow(2).mean()


def _propagator_step_jacobian_singular_values(
    step_fn: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor,
) -> torch.Tensor:
    """Shared helper for `propagator_local_expansion_floor_loss` and
    `propagator_spectrum_shape_loss`: the propagator's own per-sample
    step-Jacobian FULL singular-value spectrum (descending), via
    `torch.func.vmap(jacrev(...))`. `step_fn`: a single-step markovian map
    `(B, d) -> (B, d)`; `z`: `(B, d)` real encoded states. Returns `(B, d)`.
    `torch.linalg.svdvals` has no MPS kernel (as of this writing) -- moved
    to CPU for just this op (a differentiable device transfer, cheap given
    `d` is small, ~tens of dims) rather than failing outright or forcing
    the whole training run onto CPU -- see
    `propagator_local_expansion_floor_loss`'s docstring for how this was
    found (a real training-time MPS crash, not a hypothetical)."""

    def f(z_single: torch.Tensor) -> torch.Tensor:
        return step_fn(z_single.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(f))(z)  # (B, d, d)
    return torch.linalg.svdvals(J.cpu()).to(J.device)  # (B, d), descending order


def propagator_local_expansion_floor_loss(
    step_fn: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor, floor: float = 1.0,
) -> torch.Tensor:
    """`step_fn`: a single-step markovian map `(B, d) -> (B, d)` (e.g.
    `aux.step_one`); `z`: `(B, d)` real encoded states to evaluate at
    (typically a subsample of the current batch, for cost). Returns a
    DIFFERENTIABLE, one-sided floor on the propagator's own per-sample
    step-Jacobian spectral norm (largest singular value) -- the exact same
    quantity `ks_latent.analysis.diagnostics.propagator_step_jacobian_
    spectral_norms` measures post-hoc for Gate 4's D9 diagnostic
    (`prop_jacobian_med`/`prop_jacobian_p95`), turned into a training-time
    regularizer via `torch.func.vmap(jacrev(...))` (batched, differentiable
    -- gradients flow back through the Jacobian computation itself into
    whatever produced `z`, e.g. the encoder).

    Added 2026-09-09, user-directed, after a decisive question ("we could
    try a rollout variant, but it's the propagator already collapsing in
    stage 1? so it's sort of irrelevant?") clarified what's actually being
    tested: with `physics_prior`'s learned correction pinned near zero
    (Section 124-126's diagnostic setup), the propagator's OWN step has no
    free parameters left to adjust -- but the ENCODER still chooses WHERE
    in z-space real states land, and a fixed nonlinear map can be locally
    expansive in one region of phase space and contractive in another.
    This is the first regularizer in this whole investigation that
    actually queries the DYNAMICS (not just real-data separation, which
    `temporal_expansion_floor_loss` checks, or batch-level covariance rank,
    which `w_logdet_physical` checks) -- it directly rewards the encoder
    for placing real states where the EXISTING dynamics (learned or not)
    happen to expand rather than contract, the literal definition of a
    positive local Lyapunov exponent.

    `floor` defaults to `1.0` (the literal expand-vs-contract boundary --
    unlike `temporal_expansion_floor_loss`, this needs no real-data
    calibration at all, since "does not contract" is an absolute,
    principled criterion, not a dataset-specific scale).

    `loss = mean(relu(floor - top_singular_value)^2)`, one-sided (same
    convention as every other floor in this module -- never penalizes a
    sample for expanding MORE than the floor).

    LIMITATION found the hard way (Section 130, 2026-09-10): this only
    constrains the SINGLE largest singular value. With a shared MLP
    producing all `d` output dimensions, training found the cheapest way
    to satisfy this floor is to elevate exactly that one direction and let
    the other `d-1` decay far below it -- measured directly on a real
    Section 130 checkpoint: top 5 singular values `[1.30, 1.19, 1.16,
    1.13, 1.05]`, bottom 5 `[0.60, 0.57, 0.56, 0.54, 0.52]`, only 5 of 44
    `>= 1.0`, per-step volume-change factor `8.9e-5` (`sum(log(sv)) =
    -9.32`) -- catastrophic net contraction despite this floor being
    satisfied throughout training. `propagator_spectrum_shape_loss` (added
    directly in response to this finding) fixes the actual failure mode by
    shaping the WHOLE spectrum instead of just the top entry."""

    sv = _propagator_step_jacobian_singular_values(step_fn, z)
    top_sv = sv[:, 0]
    return F.relu(floor - top_sv).pow(2).mean()


def propagator_spectrum_shape_loss(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    n_expand: int = 11,
    expand_target: float = 1.5,
    contract_floor: float = 0.7,
    two_sided: bool = False,
) -> torch.Tensor:
    """Added 2026-09-10, Section 131, user-directed: "can we use the
    regularizer to force some singular vectors to have expansive values
    around 1.5 and others to have contracting values?" -- a direct
    response to `propagator_local_expansion_floor_loss`'s own diagnosed
    failure mode (see its docstring): constraining only the top singular
    value leaves the other `d-1` free to collapse toward zero, which is
    exactly what training did. This shapes the FULL spectrum instead: the
    top `n_expand` singular values get a one-sided EXPANSIVE floor (toward
    `expand_target`, e.g. `1.5`, matching this project's own benchmark for
    a genuinely chaotic propagator -- Section 85's median was 1.60), and
    the remaining `d - n_expand` get a separate, LOWER one-sided floor
    (`contract_floor`, e.g. `0.7`) that still permits net contraction
    (KS is dissipative -- most directions SHOULD contract) but prevents
    the runaway per-step volume collapse (`8.9e-5` in the Section 130
    checkpoint above) that an unconstrained tail produces.

    `n_expand=11` defaults to this project's own established `L=100`
    replication target for the number of positive Lyapunov exponents
    (`docs/REPLICATION_LOG.md`/this project's `CLAUDE.md` §18: "Positive
    latent exponents: 11, +-2") -- not an arbitrary split, a direct attempt
    to shape the LEARNED propagator's Jacobian spectrum toward the shape
    the TRUE attractor's own Lyapunov spectrum is already known to have,
    rather than picking a split with no external anchor.

    `loss = mean(relu(expand_target - sv[:, :n_expand])^2) +
    mean(relu(contract_floor - sv[:, n_expand:])^2)` -- both one-sided
    (same convention as every floor in this module: never penalizes a
    singular value for exceeding its own target). Raises if
    `n_expand not in [1, d-1]` (needs both a nonempty expansive group and a
    nonempty contracting group to be a meaningful split).

    `two_sided=True` (added 2026-09-18, Section 186, user-directed after a
    decisive diagnostic on Section 185's own trained checkpoint: the
    one-sided floor above is structurally incapable of correcting the
    ACTUAL observed failure. Section 185's top-15 measured singular values
    (`[4.65, 3.94, 3.72, 3.47, 3.46, 3.15, 3.05, 2.75, 1.85, 1.63, 1.58,
    1.31, 1.25, 1.16, 1.16]`) were already 3-4x ABOVE `expand_target=1.1`,
    so `relu(expand_target - top)` was already exactly 0 everywhere in the
    expansive group -- zero gradient, zero corrective pressure, for the
    entire 200-epoch run. `two_sided` replaces the one-sided hinge with a
    plain squared-error MATCH, `(sv - target)^2`, in both groups: a
    singular value ABOVE its target now costs exactly as much as one
    below it. This is the mechanism that can actually pull an
    already-excessive top singular value back down toward the real,
    measured Lyapunov-derived rate, which the one-sided version never
    could."""

    sv = _propagator_step_jacobian_singular_values(step_fn, z)  # (B, d), descending
    d = sv.shape[1]
    if not (1 <= n_expand < d):
        raise ValueError(f"n_expand must be in [1, d-1] (d={d!r}), got n_expand={n_expand!r}")
    top = sv[:, :n_expand]
    rest = sv[:, n_expand:]
    if two_sided:
        l_expand = (top - expand_target).pow(2).mean()
        l_contract_floor = (rest - contract_floor).pow(2).mean()
    else:
        l_expand = F.relu(expand_target - top).pow(2).mean()
        l_contract_floor = F.relu(contract_floor - rest).pow(2).mean()
    return l_expand + l_contract_floor


def propagator_multistep_spectrum_shape_loss(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    k: int,
    n_expand: int,
    expand_target: float,
    contract_floor: float,
) -> torch.Tensor:
    """Added 2026-09-18, Section 186, user-directed: "is there a way to
    force the pde's singular vectors ... to go from expansive to
    contractive and back again? ... if the expansive singular vectors all
    feed into other expansive singular vectors in the evolution of the
    system, we will see runaway growth."

    `propagator_spectrum_shape_loss` (one-step or `two_sided`) constrains
    only the ONE-STEP Jacobian's singular VALUES. That is not the same
    quantity as whether the dynamics are actually chaotic-but-bounded: a
    map can have a "reasonable-looking" one-step spectrum at every single
    point along its trajectory and still blow up, if the specific singular
    VECTORS that are expansive at step `n` keep landing back in an
    expansive subspace at step `n+1` (persistent, non-mixing alignment)
    instead of rotating into a contracting direction (genuine Oseledets /
    covariant-Lyapunov-vector mixing, the actual mechanism that keeps real
    chaotic systems bounded despite having positive Lyapunov exponents).
    Autodiff has no cheap differentiable handle on singular VECTORS
    themselves (no stable gradient through a degenerate/ill-conditioned
    eigenbasis), so this targets the same failure mode indirectly but
    precisely: it constrains the singular VALUES of the COMPOSED `k`-step
    Jacobian `d(z_{n+k})/d(z_n)` (built by chaining `step_fn` `k` times
    before differentiating once), not the one-step Jacobian.

    Why this catches persistent self-feeding: if the top `n_expand`
    one-step singular values are each `s` and the corresponding directions
    keep re-expanding the SAME subspace every step with no mixing, the
    k-step composed top singular value is `s^k` -- for Section 185's own
    measured `s~4.65`, `k=10` gives `~2.7e6`. If instead the expansive
    directions properly rotate/mix into contracting ones (genuine bounded
    chaos), the k-step top singular value converges toward the REAL
    asymptotic Lyapunov rate, `exp(lambda_1 * k * dt_snap)` -- which is
    exactly `expand_target ** k` here, since `expand_target` (e.g. `1.1`)
    was itself derived as `exp(lambda_1 * dt_snap)` from Section 85's own
    measured spectrum (see `propagator_spectrum_shape_loss`'s docstring).
    So `(k-step top sv) - expand_target**k`, penalized with a plain
    squared error (two-sided, matching `two_sided=True` above -- a ceiling
    is the entire point here, not just a floor), directly penalizes
    exactly the compounding-via-self-feeding pathology, using only
    quantities autodiff can already differentiate. The bottom group gets
    the same two-sided treatment toward `contract_floor ** k`, so the
    composed map is prevented from either exploding OR collapsing to
    near-singular over `k` steps.

    `step_fn`: a single-step markovian map `(B, d) -> (B, d)` (e.g.
    `aux.step_one`), NOT already composed -- composition happens inside
    this function. `k`: number of steps to compose (recommend matching the
    training curriculum's `k_pred_max`, since that is the horizon the rest
    of the loss already cares about getting right). More expensive than
    the one-step version (autodiff must trace `k` sequential applications
    of `step_fn` before the single Jacobian call) -- use a smaller
    `z`-sample than the one-step version if cost matters; still computed
    at most once per epoch, matching this module's established pattern for
    every Jacobian-based regularizer."""

    def step_k(z_in: torch.Tensor) -> torch.Tensor:
        out = z_in
        for _ in range(k):
            out = step_fn(out)
        return out

    sv = _propagator_step_jacobian_singular_values(step_k, z)  # (B, d), descending
    d = sv.shape[1]
    if not (1 <= n_expand < d):
        raise ValueError(f"n_expand must be in [1, d-1] (d={d!r}), got n_expand={n_expand!r}")
    top = sv[:, :n_expand]
    rest = sv[:, n_expand:]
    l_expand = (top - expand_target**k).pow(2).mean()
    l_contract = (rest - contract_floor**k).pow(2).mean()
    return l_expand + l_contract


def propagator_graded_spectrum_shape_loss(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    target_spectrum: torch.Tensor,
) -> torch.Tensor:
    """Added 2026-09-10, Section 134, a per-RANK generalization of
    `propagator_spectrum_shape_loss`'s two-group floor. That function's
    own docstring documents its diagnosed limitation (Section 130: only
    constraining the top singular value let the rest collapse); this
    function's motivation is a SECOND, more subtle limitation found next
    (Section 131-133): even the two-group version assumes the top
    `n_expand` singular values should all sit near one flat
    `expand_target` and the rest near one flat `contract_floor`. A robust
    (200-sample) per-rank measurement of Section 85's own real, already-
    validated propagator showed this is wrong -- the true spectrum is a
    smooth graded decline across every rank, not two plateaus:

        rank:    0      1      2      3      4      5      6
        median: 1.602  1.497  1.422  1.367  1.326  1.284  1.243  ...
        rank:    10     11     12    (continuing down through the bottom
        median: 1.079  1.018  0.955   31 ranks: median 0.509, down to
                                       0.15-0.3 at the very bottom)

    `target_spectrum`: `(d,)`, descending, typically this exact kind of
    empirical per-rank MEDIAN from a trusted checkpoint's own real
    spectrum (see `ks_latent.analysis.diagnostics.
    propagator_step_jacobian_full_spectrum`, which computes it). Stated
    plainly: this is a transfer from one trusted, already-validated
    model's own measured spectrum to another, NOT literal ground truth
    from the raw PDE -- the `d`-dim latent embedding is model-dependent,
    so there is no model-independent "true" target spectrum expressible
    in these coordinates. It is the best available empirical anchor this
    project has for "what does a genuinely non-collapsed propagator's
    Jacobian spectrum look like at this dimensionality," not a physical
    law.

    `loss = mean(relu(target_spectrum - sv)^2)`, one-sided per rank (same
    convention as every floor in this module). Raises if
    `target_spectrum`'s length doesn't match `d`."""

    sv = _propagator_step_jacobian_singular_values(step_fn, z)  # (B, d), descending
    d = sv.shape[1]
    if target_spectrum.shape[0] != d:
        raise ValueError(
            f"target_spectrum length {target_spectrum.shape[0]!r} must match d={d!r}"
        )
    target = target_spectrum.to(device=sv.device, dtype=sv.dtype).unsqueeze(0)  # (1, d)
    return F.relu(target - sv).pow(2).mean()


def horizon_weights(k_max: int, gamma: float = 1.0, device=None, dtype=None) -> torch.Tensor:
    """`w_h = gamma^h / sum(gamma^h)` for `h = 1..k_max` (brief §5.2; gamma=1
    is uniform weighting, the "latest configuration" per the handoff)."""
    h = torch.arange(1, k_max + 1, device=device, dtype=dtype)
    w = gamma**h
    return w / w.sum()


def horizon_weighted_mse(
    z_pred: torch.Tensor, z_true: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """`z_pred`, `z_true`: `(B, K, d)`; `weights`: `(K,)`. Per-horizon MSE
    (mean over batch and latent dim), weighted and summed over horizon."""
    per_horizon_mse = ((z_pred - z_true) ** 2).mean(dim=(0, 2))  # (K,)
    return (weights * per_horizon_mse).sum()


def horizon_weighted_latent_loss(
    z_pred: torch.Tensor, z_true: torch.Tensor, weights: torch.Tensor, kind: str = "l2"
) -> torch.Tensor:
    """Like `horizon_weighted_mse`, generalized to `kind in {"l2", "l1"}`
    (added 2026-08-29, ported from a reference implementation -- see
    `Stage2TrainingConfig.latent_loss`'s docstring: L1 is less dominated by
    the rare large increments a chaotic latent trajectory produces, which
    the reference project measured as visibly better short-time tracking).
    `kind="l2"` reproduces `horizon_weighted_mse` exactly."""
    if kind == "l2":
        per_horizon = ((z_pred - z_true) ** 2).mean(dim=(0, 2))
    elif kind == "l1":
        per_horizon = (z_pred - z_true).abs().mean(dim=(0, 2))
    else:
        raise ValueError(f"kind must be 'l2' or 'l1', got {kind!r}")
    return (weights * per_horizon).sum()


def kernel_unstable_floor_loss(Lhat: torch.Tensor, target_modes: int, margin: float = 0.0) -> torch.Tensor:
    """Hinge penalty encouraging AT LEAST `target_modes` low-wavenumber
    Fourier modes to stay genuinely unstable (`Lhat(k) >= margin`), added
    2026-09-16, Section 183, user-directed: "we should also increase the
    regularizer that tries to keep at least 13 unstable modes in the pde,
    or if that doesn't exist, implement it" -- no such regularizer
    existed. `Lhat`: `(K,)`, from `_SpectralPDEDeltaBody.kernel_Lhat()` --
    a pure function of the model's own A/width/nu/mu parameters,
    independent of any batch of real data, so this loss term needs no
    forward pass over data at all to compute.

    Motivation: Sections 175-179 (learnable A/beta, ordinary k-step MSE
    loss) consistently drove A toward MORE damping (never discovering
    instability is useful for genuine long-run dynamics -- a purely
    local, short-horizon-favorable direction under that loss). Sections
    180-182 (A/beta architecturally FIXED unstable) confirmed the
    destabilizing mechanism itself is not the problem in isolation, but
    ALSO that hard-fixing the value doesn't by itself produce a genuine
    balance -- the system either needs to find its OWN accommodation
    given a floor guaranteeing SOME minimum instability always survives
    training, rather than a single frozen value.

    `L = mean( relu(margin - Lhat[1:target_modes+1]) ^ 2 )` -- excludes
    index 0 (the DC/mean mode, whose Lhat is architecturally EXACTLY 0
    always, regardless of any parameter -- see `kernel_Lhat`'s own
    docstring on why: the k^2 prefactor vanishes at k=0). Indices
    `1..target_modes` (inclusive, `target_modes` modes total) must ALL
    clear `margin` for this loss to reach exactly 0 -- a stronger,
    per-mode guarantee than merely counting how many modes happen to be
    positive on average, so it directly targets "at least this many
    unstable modes" rather than a softer aggregate proxy."""
    target_slice = Lhat[1 : target_modes + 1]
    return torch.relu(margin - target_slice).pow(2).mean()
