"""Structure diagnostics D1-D5 on an existing (frozen) checkpoint, no
retraining (brief §8, main notes §3), plus D6 (temporal coherence
structure, user-directed 2026-08-30) and D7 (same-time channel
correlation, user-directed 2026-08-31) -- neither part of the original
brief, see each section's own docstring below.

Each diagnostic function returns a small dataclass with the raw arrays plus
whatever summary numbers the brief asks for; `scripts/run_diagnostics.py`
assembles them all into `diagnostics_report.md` (Gate 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pywt
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from torch.func import jacrev, vmap

from ks_latent.analysis.dimension import two_nn_dimension
from ks_latent.utils.circstats import CircularStats, circular_distance, circular_weighted_stats

# ---------------------------------------------------------------------------
# D1: sensitivity maps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityMapResult:
    S: np.ndarray  # (d, NX)
    per_latent: list[CircularStats]
    order_by_centroid: np.ndarray  # (d,) indices sorting latents by centroid


def _sensitivity_map(fn_single: Callable[[torch.Tensor], torch.Tensor], samples: torch.Tensor) -> np.ndarray:
    """`E_samples |d fn/d input|`, `fn_single: (in_dim,) -> (out_dim,)`,
    `samples: (B, in_dim)`. Returns `(out_dim, in_dim)`."""
    jac = vmap(jacrev(fn_single))(samples)  # (B, out_dim, in_dim)
    return jac.abs().mean(dim=0).detach().numpy()


def decoder_sensitivity_map(
    decoder: Callable[[torch.Tensor], torch.Tensor], z_samples: torch.Tensor, L: float
) -> SensitivityMapResult:
    """`S[k, x] = E_z |d D(z)(x) / d z_k|` (brief §8 D1). `decoder` maps a
    single `(d,)` latent vector to a `(NX,)` field."""
    S_xk = _sensitivity_map(decoder, z_samples)  # (NX, d)
    S = S_xk.T  # (d, NX)
    return _summarize_sensitivity_map(S, L)


def encoder_sensitivity_map(
    encoder: Callable[[torch.Tensor], torch.Tensor], u_samples: torch.Tensor, L: float
) -> SensitivityMapResult:
    """Encoder analogue: `|d z_k / d u(x)|`. `encoder` maps a single `(NX,)`
    field to a `(d,)` latent vector."""
    S = _sensitivity_map(encoder, u_samples)  # (d, NX)
    return _summarize_sensitivity_map(S, L)


def _summarize_sensitivity_map(S: np.ndarray, L: float) -> SensitivityMapResult:
    d, NX = S.shape
    x = np.arange(NX) * L / NX
    per_latent = [circular_weighted_stats(x, S[k], L) for k in range(d)]
    order = np.argsort([s.centroid for s in per_latent])
    return SensitivityMapResult(S=S, per_latent=per_latent, order_by_centroid=order)


# ---------------------------------------------------------------------------
# D2: wavenumber content
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WavenumberContentResult:
    spectral_centroid: np.ndarray  # (d,) characteristic |k| per latent
    spectral_bandwidth: np.ndarray  # (d,) spread of |k| per latent
    wavelet_energy_entropy: np.ndarray  # (d,) joint space-scale localization (lower = more localized)


def wavenumber_content(S: np.ndarray, L: float, wavelet: str = "db4") -> WavenumberContentResult:
    """FFT each row of the sensitivity map `S` (`(d, NX)`) for a spectral
    centroid/bandwidth, plus a wavelet-packet energy-entropy joint
    space-scale localization measure (addendum §14.4: Wittenberg & Holmes
    1999 find KS localized in *both* real and Fourier space, so a measure
    beyond the two extremes is needed)."""
    d, NX = S.shape
    k = 2.0 * np.pi * np.fft.fftfreq(NX, d=L / NX)
    k_abs = np.abs(k)

    centroids = np.empty(d)
    bandwidths = np.empty(d)
    entropies = np.empty(d)
    for i in range(d):
        power = np.abs(np.fft.fft(S[i])) ** 2
        total = power.sum()
        centroid = float(np.sum(k_abs * power) / total)
        bandwidth = float(np.sqrt(np.sum(((k_abs - centroid) ** 2) * power) / total))
        centroids[i] = centroid
        bandwidths[i] = bandwidth

        coeffs = pywt.wavedec(S[i], wavelet, mode="periodization")
        energies = np.concatenate([c**2 for c in coeffs])
        p = energies / energies.sum()
        p = p[p > 0]
        max_entropy = np.log(len(energies))
        entropies[i] = float(-(p * np.log(p)).sum() / max_entropy) if max_entropy > 0 else 0.0

    return WavenumberContentResult(
        spectral_centroid=centroids, spectral_bandwidth=bandwidths, wavelet_energy_entropy=entropies
    )


# ---------------------------------------------------------------------------
# D3: Jacobian coupling graph, spectral seriation, bandedness p-value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouplingGraphResult:
    A_jacobian: np.ndarray  # (d, d), E|d z_{n+1,k} / d z_{n,l}|
    A_distance_corr: np.ndarray  # (d, d), nonlinear alternative
    A_mutual_info: np.ndarray  # (d, d), k-NN mutual information
    permutation: np.ndarray  # (d,), the seriation (Fiedler-vector) ordering
    bandedness_observed: float
    bandedness_p_value: float


def jacobian_coupling(
    propagator_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_prev_samples: torch.Tensor,
    z_curr_samples: torch.Tensor,
) -> np.ndarray:
    """`A[k,l] = E |d z_{n+1,k} / d z_{n,l}|` (brief §8 D3), holding
    `z_{n-1}` fixed at its sampled value (the propagator's Jacobian is taken
    w.r.t. its *second* argument only, matching Phase 4's single_state
    convention)."""

    def step_wrt_curr(z_curr_single: torch.Tensor, z_prev_single: torch.Tensor) -> torch.Tensor:
        return propagator_step(z_prev_single.unsqueeze(0), z_curr_single.unsqueeze(0)).squeeze(0)

    def jac_one(z_prev_single, z_curr_single):
        return jacrev(step_wrt_curr)(z_curr_single, z_prev_single)

    jac = vmap(jac_one)(z_prev_samples, z_curr_samples)  # (B, d, d)
    return jac.abs().mean(dim=0).detach().numpy()


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation (Szekely, Rizzo & Bakirov 2007, Ann. Statist.
    35:2769): a nonlinear dependence measure, zero iff independent."""
    n = len(x)
    a = squareform(pdist(x.reshape(-1, 1)))
    b = squareform(pdist(y.reshape(-1, 1)))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).sum() / n**2
    dvar_x2 = (A * A).sum() / n**2
    dvar_y2 = (B * B).sum() / n**2
    denom = np.sqrt(dvar_x2 * dvar_y2)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def nonlinear_coupling(Z_next: np.ndarray, Z_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance correlation and k-NN mutual information between every pair
    `(z_{n+1,k}, z_{n,l})`, as a linearization-free cross-check on
    `jacobian_coupling` (brief §8 D3). `Z_next`, `Z_curr`: `(B, d)`."""
    d = Z_curr.shape[1]
    A_dcor = np.empty((d, d))
    for k in range(d):
        for l in range(d):
            A_dcor[k, l] = _distance_correlation(Z_next[:, k], Z_curr[:, l])
    A_mi = np.empty((d, d))
    for k in range(d):
        mi = mutual_info_regression(Z_curr, Z_next[:, k], random_state=0)
        A_mi[k, :] = mi
    return A_dcor, A_mi


def _fiedler_permutation(A: np.ndarray) -> np.ndarray:
    """Spectral seriation via the Fiedler vector of the graph Laplacian of
    the symmetrized coupling matrix."""
    W = (A + A.T) / 2.0
    np.fill_diagonal(W, 0.0)
    degree = W.sum(axis=1)
    Lap = np.diag(degree) - W
    eigvals, eigvecs = np.linalg.eigh(Lap)
    order = np.argsort(eigvals)
    fiedler = eigvecs[:, order[1]]  # second-smallest eigenvalue's eigenvector
    return np.argsort(fiedler)


def bandedness(A: np.ndarray, permutation: np.ndarray, bandwidth: float = 3.0) -> float:
    """`sum_ij A[pi_i, pi_j] * w(circular_dist(i,j))`, normalized by
    `sum(A)`, `w(dist) = exp(-dist^2 / (2*bandwidth^2))` (brief §8 D3).
    A picture is not a result -- this is the scalar that gets a p-value."""
    d = A.shape[0]
    A_perm = A[np.ix_(permutation, permutation)]
    i, j = np.meshgrid(np.arange(d), np.arange(d), indexing="ij")
    dist = circular_distance(i, j, d)
    w = np.exp(-(dist**2) / (2.0 * bandwidth**2))
    total = A.sum()
    if total <= 0:
        return 0.0
    return float((A_perm * w).sum() / total)


def bandedness_p_value(
    A: np.ndarray, n_null: int = 1000, bandwidth: float = 3.0, seed: int = 0
) -> tuple[np.ndarray, float, float]:
    """Seriate `A` via its Fiedler vector, score its bandedness, and compare
    against `n_null` random permutations of the same matrix (brief §8 D3:
    "compared against a null distribution of >= 1000 random permutations ->
    report a p-value"). Returns `(permutation, observed_score, p_value)`.
    """
    d = A.shape[0]
    permutation = _fiedler_permutation(A)
    observed = bandedness(A, permutation, bandwidth)

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_null)
    for i in range(n_null):
        perm = rng.permutation(d)
        null_scores[i] = bandedness(A, perm, bandwidth)
    p_value = float(np.mean(null_scores >= observed))
    return permutation, observed, p_value


def bandedness_p_value_entry_shuffle(
    A: np.ndarray, n_null: int = 500, bandwidth: float = 3.0, seed: int = 0
) -> tuple[np.ndarray, float, float]:
    """A CORRECTLY calibrated alternative to `bandedness_p_value` at
    real-world `d` (discovered 2026-08-31 building D7 below -- see that
    section's CALIBRATION WARNING for the full story).
    `bandedness_p_value`'s null scores the Fiedler-OPTIMIZED ordering of
    `A` against RANDOM (unoptimized) orderings of the SAME `A` -- since the
    Fiedler vector is specifically chosen to concentrate `A`'s largest
    entries near the diagonal, it will nearly always beat an unoptimized
    ordering, EVEN on pure symmetric noise (empirically: `p<0.05` in 6/8
    seeds on `d=44` noise matrices). This function instead shuffles `A`'s
    off-diagonal VALUES among themselves (preserving symmetry, the
    diagonal, and the exact multiset of observed magnitudes -- so the
    comparison isn't confounded by a different total mass/scale between
    real and null, which an earlier attempt at fixing this using
    column-shuffled raw-data surrogates ran into) and RE-OPTIMIZES a fresh
    Fiedler ordering for each shuffled surrogate before scoring it -- so the
    null distribution reflects "best achievable bandedness with no true
    spatial structure, same value distribution, same `d`," not merely "a
    random relabeling of the real matrix." Verified: ~5% false-positive
    rate on pure-noise `d=44` matrices (0/6 seeds `p<0.05` in the check
    that motivated this), versus badly inflated for the original.
    Returns `(permutation, observed_score, p_value)`, same signature as
    `bandedness_p_value`."""
    d = A.shape[0]
    permutation = _fiedler_permutation(A)
    observed = bandedness(A, permutation, bandwidth)

    iu = np.triu_indices(d, k=1)
    vals = A[iu].copy()
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_null)
    for i in range(n_null):
        shuffled = rng.permutation(vals)
        A_null = np.zeros_like(A)
        A_null[iu] = shuffled
        A_null = A_null + A_null.T
        np.fill_diagonal(A_null, np.diag(A))
        perm_null = _fiedler_permutation(A_null)
        null_scores[i] = bandedness(A_null, perm_null, bandwidth)
    p_value = float(np.mean(null_scores >= observed))
    return permutation, observed, p_value


def coupling_graph_diagnostic(
    propagator_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_prev_samples: torch.Tensor,
    z_curr_samples: torch.Tensor,
    n_null: int = 1000,
    seed: int = 0,
) -> CouplingGraphResult:
    A_jac = jacobian_coupling(propagator_step, z_prev_samples, z_curr_samples)
    with torch.no_grad():
        Z_next = torch.stack(
            [propagator_step(z_prev_samples[i : i + 1], z_curr_samples[i : i + 1]).squeeze(0)
             for i in range(z_curr_samples.shape[0])]
        ).numpy()
    A_dcor, A_mi = nonlinear_coupling(Z_next, z_curr_samples.numpy())
    permutation, observed, p_value = bandedness_p_value_entry_shuffle(A_jac, n_null=n_null, seed=seed)
    return CouplingGraphResult(
        A_jacobian=A_jac, A_distance_corr=A_dcor, A_mutual_info=A_mi,
        permutation=permutation, bandedness_observed=observed, bandedness_p_value=p_value,
    )


def jacobian_coupling_history(
    propagator_step_history: Callable[[torch.Tensor], torch.Tensor],
    z_hist_samples: torch.Tensor,  # (B, n_history, d)
) -> np.ndarray:
    """`mode="history"` counterpart of `jacobian_coupling` (added
    2026-08-29, user-directed): `A[k,l] = E |d z_{n+1,k} / d z_{n,l}|`,
    holding every *older* state in the history fixed at its sampled value
    -- the Jacobian is taken only w.r.t. the most recent (current) state,
    matching `jacobian_coupling`'s existing convention."""

    def step_wrt_curr(z_curr_single: torch.Tensor, z_older_single: torch.Tensor) -> torch.Tensor:
        z_hist_single = torch.cat([z_older_single, z_curr_single.unsqueeze(0)], dim=0)
        return propagator_step_history(z_hist_single.unsqueeze(0)).squeeze(0)

    def jac_one(z_hist_single: torch.Tensor) -> torch.Tensor:
        z_older_single, z_curr_single = z_hist_single[:-1], z_hist_single[-1]
        return jacrev(step_wrt_curr)(z_curr_single, z_older_single)

    jac = vmap(jac_one)(z_hist_samples)  # (B, d, d)
    return jac.abs().mean(dim=0).detach().numpy()


def coupling_graph_diagnostic_history(
    propagator_step_history: Callable[[torch.Tensor], torch.Tensor],
    z_hist_samples: torch.Tensor,  # (B, n_history, d)
    n_null: int = 1000,
    seed: int = 0,
) -> CouplingGraphResult:
    """`mode="history"` counterpart of `coupling_graph_diagnostic` (added
    2026-08-29, user-directed)."""
    A_jac = jacobian_coupling_history(propagator_step_history, z_hist_samples)
    with torch.no_grad():
        Z_next = propagator_step_history(z_hist_samples).numpy()
    A_dcor, A_mi = nonlinear_coupling(Z_next, z_hist_samples[:, -1].numpy())
    permutation, observed, p_value = bandedness_p_value_entry_shuffle(A_jac, n_null=n_null, seed=seed)
    return CouplingGraphResult(
        A_jacobian=A_jac, A_distance_corr=A_dcor, A_mutual_info=A_mi,
        permutation=permutation, bandedness_observed=observed, bandedness_p_value=p_value,
    )


# ---------------------------------------------------------------------------
# D4: translation representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationRepResult:
    shifts: np.ndarray  # (n_shifts,)
    R_matrices: list[np.ndarray]  # (d, d) real matrix per shift
    relative_residuals: np.ndarray  # (n_shifts,)
    eigenvalues: list[np.ndarray]  # complex (d,) per shift
    group_property_shifts: list[tuple[int, int]]
    group_property_errors: np.ndarray  # ||R(c1)R(c2) - R(c1+c2)|| / ||R(c1+c2)||


def _fit_R(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form least squares `R = argmin_R ||Y - X R^T||_F^2` (brief §8
    D4), via the normal equations `R = (Y^T X)(X^T X)^-1`. Returns `(R,
    relative_residual)`."""
    XtX = X.T @ X
    R = (Y.T @ X) @ np.linalg.pinv(XtX)
    residual = np.linalg.norm(Y - X @ R.T) / np.linalg.norm(Y)
    return R, float(residual)


def translation_representation(
    encoder: Callable[[torch.Tensor], torch.Tensor],
    u_samples: torch.Tensor,
    shifts: list[int],
    NX: int,
) -> TranslationRepResult:
    """For each shift `c`, fit `R(c) = argmin_R E_u ||E(roll(u,c)) - R
    E(u)||^2` in closed form, then check the group property
    `R(c1)R(c2) ~ R(c1+c2)` and the eigenvalues of each `R(c)` (brief §8
    D4). If `E` is genuinely (approximately) equivariant, `R(c)` is
    (approximately) a rotation and its eigenvalues take the form
    `e^{i k_j c}`, handing over latent wavenumbers `k_j` with zero
    retraining."""
    with torch.no_grad():
        X = encoder(u_samples).numpy()

    R_matrices, residuals, eigvals = [], [], []
    with torch.no_grad():
        for c in shifts:
            u_shift = torch.roll(u_samples, shifts=c, dims=-1)
            Y = encoder(u_shift).numpy()
            R, residual = _fit_R(X, Y)
            R_matrices.append(R)
            residuals.append(residual)
            eigvals.append(np.linalg.eigvals(R))

    group_pairs, group_errors = [], []
    shift_set = set(shifts)
    for i, c1 in enumerate(shifts):
        for j, c2 in enumerate(shifts):
            c12 = (c1 + c2) % NX
            if c12 in shift_set and c12 != 0:
                k = shifts.index(c12)
                lhs = R_matrices[i] @ R_matrices[j]
                rhs = R_matrices[k]
                err = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
                group_pairs.append((c1, c2))
                group_errors.append(err)

    return TranslationRepResult(
        shifts=np.array(shifts),
        R_matrices=R_matrices,
        relative_residuals=np.array(residuals),
        eigenvalues=eigvals,
        group_property_shifts=group_pairs,
        group_property_errors=np.array(group_errors),
    )


# ---------------------------------------------------------------------------
# D5: local intrinsic dimension vs. patch length
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalDimensionResult:
    lengths: np.ndarray
    d_local: np.ndarray
    slope: float
    slope_ci: tuple[float, float]
    intercept: float


def extract_patches(
    points: np.ndarray, length: int, rng: np.random.Generator, start: int | None = None
) -> np.ndarray:
    """A length-`length` periodic window from each row of `points` (`(N,
    NX)`), starting at `start` if given, else an independent random offset
    per row."""
    N, NX = points.shape
    starts = np.full(N, start, dtype=int) if start is not None else rng.integers(0, NX, size=N)
    idx = (starts[:, None] + np.arange(length)[None, :]) % NX
    return np.take_along_axis(points, idx, axis=1)


def local_dimension_vs_length(
    points: np.ndarray,
    lengths: list[int],
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> LocalDimensionResult:
    """Two-NN dimension of physical patches of each length in `lengths`
    (brief §8 D5). Extensivity at `L=100` predicts `d_local(l) ~ 0.226*l +
    c`; fits the slope with a confidence interval via `scipy.stats.linregress`."""
    from scipy import stats as scipy_stats

    d_local = np.array(
        [two_nn_dimension(extract_patches(points, ell, rng)).dimension for ell in lengths]
    )
    lengths_arr = np.asarray(lengths, dtype=float)
    fit = scipy_stats.linregress(lengths_arr, d_local)
    t_val = scipy_stats.t.ppf((1 + confidence) / 2, df=len(lengths) - 2)
    ci = (fit.slope - t_val * fit.stderr, fit.slope + t_val * fit.stderr)
    return LocalDimensionResult(
        lengths=lengths_arr, d_local=d_local, slope=float(fit.slope), slope_ci=ci, intercept=float(fit.intercept)
    )


# ---------------------------------------------------------------------------
# D6: temporal coherence structure (user-directed 2026-08-30, not part of
# the original brief's D1-D5 -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
# Section 8)
# ---------------------------------------------------------------------------
#
# Motivation: D3's Jacobian-bandedness diagnostic measures whether the
# PROPAGATOR's learned map is local in latent-index space. That is blind to
# a real but non-local coupling structure by construction: an FNO layer's
# spectral conv is a shift-equivariant (circulant) operator regardless of
# how many Fourier modes it keeps, so a very broadband (not "banded")
# Jacobian is fully consistent with genuinely respecting the latent's ring
# topology -- bandedness specifically detects a narrow local kernel, not
# "any operator that respects the ring's symmetry." Measured directly:
# D3's own bandedness score on the FNO+ViT propagator's Jacobian dropped
# from 0.99 (plain vit propagator) to 0.35 (still significant vs random,
# but much weaker) even after Fiedler-seriating to the BEST possible
# circular ordering -- consistent with "broadband but still structured,"
# not conclusive either way.
#
# This diagnostic instead asks a data-only question, independent of any
# propagator: does the RAW ENCODED TRAJECTORY z(t) show an emergent
# low-dimensional (ideally near-ring) organization across its d channels,
# based on how similarly they behave *dynamically* over time? A same-time
# cross-covariance approach (the original, abandoned seriation attempt
# mentioned in `jacobian_coupling`'s docstring) cannot find this: L_decorr
# already forces E[z_i(t) z_j(t)] (over the sample/dataset distribution) to
# be approximately diagonal by construction, so there is nothing left to
# find there. A *lagged, windowed* correlation is a different statistic
# L_decorr does not constrain -- it asks whether z_i and z_j move together
# ALONG a single trajectory through time (possibly with a short lag,
# reflecting a finite propagation speed the way spatially-nearby points in
# the true PDE field would be correlated with a short time lag), not
# whether they are correlated across independent samples at one instant.
# This is the same class of technique used for "functional connectivity"
# networks in neuroscience and coherent-structure detection from unlabelled
# sensor arrays in fluid mechanics: build a similarity matrix from
# lagged/windowed correlation, then find a low-dimensional spatial
# embedding via the graph Laplacian's leading eigenvectors (the same
# spectral-seriation machinery `_fiedler_permutation`/`bandedness_p_value`
# above already use for D3, reused here as-is on this new matrix).


def local_temporal_coherence(Z: np.ndarray, window: int, max_lag: int, step: int | None = None) -> np.ndarray:
    """`Z`: `(T, d)` for a single encoded trajectory, or `(n_runs, T, d)` for
    several (their windows are pooled together). Returns a symmetric `(d,
    d)` matrix `C`, where `C[i, j]` is the median, over many overlapping
    length-`window` windows and every trajectory, of `max_{|tau|<=max_lag}
    |corr(z_i(t0:t0+window), z_j(t0+tau:t0+window+tau))|` -- the best-lag
    windowed correlation between channels `i` and `j`, aggregated robustly
    (median, not mean, since a chaotic trajectory's local coherence is
    plausibly non-stationary -- a few high-coherence windows should not be
    able to dominate the summary the way a mean would let them).

    `window` must be `> 2 * max_lag` (every window needs room to shift by
    `max_lag` in either direction and still fit inside the trajectory).
    `step` (window stride) defaults to `window // 2`.

    Calibration note (verified empirically, see
    `tests/unit/test_diagnostics_d6.py`): `window` should be `>>
    2*max_lag+1` (the number of lags scanned), not merely `>`. Taking the
    max over many lags of a correlation estimated from too few samples is a
    multiple-comparisons problem with a real, systematic positive bias --
    `window=50, max_lag=10` gives ~0.3 "coherence" for pairs of PURE
    independent noise (a false-positive rate for `bandedness_p_value` well
    above nominal); `window=200, max_lag=5` does not. Use a generous
    `window` relative to `max_lag` in practice.
    """
    if Z.ndim == 2:
        Z = Z[None]
    n_runs, T, d = Z.shape
    if window <= 2 * max_lag:
        raise ValueError(f"window={window} must be > 2*max_lag={2 * max_lag}")
    if step is None:
        step = max(window // 2, 1)
    starts = list(range(max_lag, T - window - max_lag + 1, step))
    if not starts:
        raise ValueError(
            f"no valid windows: need T={T} >= window={window} + 2*max_lag={2 * max_lag}"
        )
    lags = range(-max_lag, max_lag + 1)
    per_window = []
    for r in range(n_runs):
        z = Z[r]
        for t0 in starts:
            best = np.zeros((d, d))
            for tau in lags:
                a = z[t0 : t0 + window]
                b = z[t0 + tau : t0 + window + tau]
                a_c = a - a.mean(axis=0, keepdims=True)
                b_c = b - b.mean(axis=0, keepdims=True)
                a_std = a_c / (a_c.std(axis=0, keepdims=True) + 1e-12)
                b_std = b_c / (b_c.std(axis=0, keepdims=True) + 1e-12)
                corr = (a_std.T @ b_std) / window  # (d, d): corr[i,j] = corr(a_i, b_j)
                best = np.maximum(best, np.abs(corr))
            per_window.append(best)
    stacked = np.stack(per_window, axis=0)  # (n_windows_total, d, d)
    C = np.median(stacked, axis=0)
    return (C + C.T) / 2.0


def laplacian_eigenmap(A: np.ndarray, n_components: int = 2) -> np.ndarray:
    """`n_components` leading non-trivial eigenvectors of the graph
    Laplacian of the symmetrized `A` (generalizes `_fiedler_permutation`,
    which uses only the single leading one, to a full low-dimensional
    embedding coordinate per channel). Returns `(d, n_components)`: row `k`
    is channel `k`'s coordinate in the discovered space. A clean embedding
    that visibly collapses onto (close to) a 1D curve/ring in this space is
    the direct, visualizable evidence of emergent spatial structure this
    diagnostic is for; a diffuse, high-rank-looking cloud is evidence
    against it."""
    W = (A + A.T) / 2.0
    d = W.shape[0]
    np.fill_diagonal(W, 0.0)
    degree = W.sum(axis=1)
    Lap = np.diag(degree) - W
    eigvals, eigvecs = np.linalg.eigh(Lap)
    order = np.argsort(eigvals)
    # Skip the trivial constant eigenvector (eigenvalue 0, order[0]).
    n = min(n_components, d - 1)
    return eigvecs[:, order[1 : 1 + n]]


@dataclass(frozen=True)
class TemporalCoherenceResult:
    C: np.ndarray  # (d, d)
    permutation: np.ndarray  # Fiedler-seriated channel ordering
    bandedness_observed: float
    bandedness_p_value: float
    embedding: np.ndarray  # (d, n_components)


def temporal_coherence_diagnostic(
    Z: np.ndarray,
    window: int,
    max_lag: int,
    step: int | None = None,
    n_null: int = 1000,
    seed: int = 0,
    n_components: int = 2,
) -> TemporalCoherenceResult:
    """Full D6 pipeline: build the lagged-coherence matrix, Fiedler-seriate
    it and score its bandedness against `n_null` random permutations (reusing
    `bandedness_p_value` verbatim -- same statistical machinery as D3, a
    different input matrix), and compute a `n_components`-dimensional
    spectral embedding of the channels. `Z`: see `local_temporal_coherence`."""
    C = local_temporal_coherence(Z, window, max_lag, step=step)
    permutation, observed, p_value = bandedness_p_value_entry_shuffle(C, n_null=n_null, seed=seed)
    embedding = laplacian_eigenmap(C, n_components=n_components)
    return TemporalCoherenceResult(
        C=C, permutation=permutation, bandedness_observed=observed,
        bandedness_p_value=p_value, embedding=embedding,
    )


# ---------------------------------------------------------------------------
# D7: same-time (no lag, no propagator) channel correlation
# ---------------------------------------------------------------------------
#
# User-directed 2026-08-31, after correctly pushing back on D6: "I thought
# D6 was for temporal coherence, not spatial? I want latent variables to
# have some local structure where neighboring latent variables vary
# together." D6 (above) is indeed a TEMPORAL (lagged, across time)
# statistic; this section is the genuinely SAME-INSTANT one -- does
# encode(u(t))_i and encode(u(t))_j covary across the real data distribution
# at a single time, for channels `i`, `j` near each other under some
# permutation? This is precisely the naive covariance-seriation idea this
# file's own D3 docstring dismisses ("approximately I by construction
# because of L_decorr, so it could not have found anything") -- but that
# claim was never empirically checked against a real trained checkpoint.
# It was checked here (2026-08-31) and found to be WRONG as stated: on
# `stage1_ae_patched_full_history3_fullprop_wvar005_tw16_mlpaux_200ep.pt`,
# the real same-time |correlation| matrix has mean off-diagonal magnitude
# ~0.12 and max ~0.46 -- not remotely close to the identity. The actual
# mechanism is `Stage1TrainingConfig.w_decorr` (default 0.01, always on in
# every training run in this document), NOT `RegConfig.lambda_decorr`
# (off by default, never used in this document, easy to confuse the two by
# name) -- and `w_decorr=0.01` is evidently too weak to fully flatten the
# off-diagonal structure in practice, leaving real signal for this
# diagnostic to find.
#
# CALIBRATION WARNING, discovered 2026-08-31 while building this section
# (see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22 for the full
# writeup): `bandedness_p_value` -- the shared statistical core used
# VERBATIM by D3, D6, and (initially) this section -- is badly miscalibrated
# at `d=44` (the real latent dimension used everywhere in this project; the
# existing D3/D6 calibration tests only ever checked `d=12`). Its null
# permutes the SAME observed matrix `A` under random relabelings and asks
# "does the Fiedler-optimized ordering score higher than a random one of
# this same A" -- but the Fiedler vector is SPECIFICALLY optimized to
# concentrate `A`'s largest entries near the diagonal, so it will nearly
# always beat an unoptimized random ordering, EVEN WHEN `A` is pure
# symmetric noise with no true structure at all: direct test on `d=44`
# noise matrices gave `p<0.05` in 6 of 8 seeds (see the empirical check in
# Section 22). This is a multiple-comparisons/adaptive-search bias, not a
# coding bug -- the null needs to account for the fact that the ordering
# itself was CHOSEN to maximize the score, which `bandedness_p_value`'s
# same-matrix-permutation null does not do.
#
# `same_time_coupling_diagnostic` below uses `bandedness_p_value_
# entry_shuffle` (defined next to `bandedness_p_value` above) instead of
# `bandedness_p_value` for this reason -- an EARLIER attempt at fixing this,
# shuffling raw-data COLUMNS of `Z` and recomputing correlation from
# scratch for each null sample, was itself flawed: it destroys real
# structure but also changes the null matrix's TOTAL correlation mass
# relative to the real one (shuffled columns give near-zero off-diagonal
# correlation almost everywhere, so the null's `A.sum()` shrinks toward
# just the diagonal, which trivially inflates ITS bandedness score under
# `bandedness()`'s sum-normalized formula -- this gave p=1.0 on a matrix
# with GENUINE strong planted structure, the opposite of what was wanted).
# The entry-VALUE-shuffle null fixes this by construction (same value
# multiset in both real and null, only their (i,j) placement differs), and
# was verified to have both correct calibration (0/6 false positives on
# `d=44` noise) AND real power (recovers planted ring structure at `d=44`
# with `p=0.0000`; weaker but present at `d=12`).
#
# D3 and D6 still use the ORIGINAL, miscalibrated `bandedness_p_value` and
# have NOT been fixed here (D3's matrix is a propagator Jacobian and D6's a
# lagged-correlation matrix -- the same entry-value-shuffle fix should
# apply to both just as directly as it did here, since it only needs the
# derived (d,d) matrix, not the raw data, but doing so is left for a
# follow-up rather than done in the same pass as this section). Every
# "significant" D3/D6 p-value reported earlier in this document should be
# treated with real skepticism until recalibrated the same way -- though
# the ACTUAL permutation each one discovered may still be informative even
# if its p-value overstates significance; Sections 20/21's trained-
# propagator Gate-3 results (does a masked_mlp actually avoid collapse
# using that permutation) are a more direct, calibration-independent test
# of that.


@dataclass(frozen=True)
class SameTimeCouplingResult:
    A: np.ndarray  # (d, d), |correlation(z_i, z_j)| across real samples, same time
    permutation: np.ndarray
    bandedness_observed: float
    bandedness_p_value: float
    embedding: np.ndarray  # (d, n_components)


def _corrcoef_nan_safe(Z: np.ndarray) -> np.ndarray:
    """`np.corrcoef`, with any `nan` entries replaced by `0` (added
    2026-09-07, caught by `encoder_kind="spectral_field"`'s always-zero
    DC-imaginary latent channel -- see
    `ks_latent.models.spectral_field`'s module docstring -- crashing
    downstream Fiedler-permutation eigendecomposition in
    `bandedness_p_value_entry_shuffle`, since `np.corrcoef` divides by each
    channel's own stddev, which is EXACTLY zero for a channel with no
    variance at all, producing nan rows/cols). A zero-variance channel has
    no meaningful linear relationship with anything -- `0` correlation is
    the natural, safe degenerate convention, and lets every downstream
    diagnostic (bandedness scoring, Fiedler ordering, spectral embedding)
    run on a well-formed matrix instead of aborting. Any encoder
    architecture with a dead/collapsed latent channel could hit this, not
    just this one -- exactly the kind of state these diagnostics exist to
    detect and report, not crash on."""
    corr = np.corrcoef(Z, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def same_time_channel_correlation(Z: np.ndarray) -> np.ndarray:
    """`Z`: `(n_samples, d)` real encoded snapshots (any mix of times/
    trajectories, treated as i.i.d. samples -- no propagator, no lag,
    unlike D3/D6). Returns the `(d, d)` `|Pearson correlation|` matrix."""
    return np.abs(_corrcoef_nan_safe(Z))


def same_time_coupling_diagnostic(
    Z: np.ndarray, n_null: int = 500, seed: int = 0, n_components: int = 2, bandwidth: float = 3.0,
) -> SameTimeCouplingResult:
    """Full D7 pipeline, using `bandedness_p_value_entry_shuffle` (see the
    CALIBRATION WARNING above -- do NOT swap this for plain
    `bandedness_p_value`, which is miscalibrated at `d=44`): build the
    same-time `|correlation|` matrix, then score its bandedness against a
    properly-recalibrated null, and compute a spectral embedding. This is
    the most direct test of "do neighboring latent variables vary
    together" -- no propagator, no time lag, just the raw encoded data's own
    same-instant statistics."""
    A = same_time_channel_correlation(Z)
    permutation, observed, p_value = bandedness_p_value_entry_shuffle(
        A, n_null=n_null, bandwidth=bandwidth, seed=seed
    )
    embedding = laplacian_eigenmap(A, n_components=n_components)
    return SameTimeCouplingResult(
        A=A, permutation=permutation, bandedness_observed=observed,
        bandedness_p_value=p_value, embedding=embedding,
    )


@dataclass
class SameTimeCouplingSignedResult:
    A: np.ndarray  # (d, d), SIGNED correlation(z_i, z_j), same time
    bandedness_observed: float
    bandedness_p_value: float


def same_time_channel_correlation_signed(Z: np.ndarray) -> np.ndarray:
    """`Z`: `(n_samples, d)`, same convention as
    `same_time_channel_correlation` -- returns the raw SIGNED `(d, d)`
    Pearson correlation matrix (no `abs`)."""
    return _corrcoef_nan_safe(Z)


def signed_bandedness(A: np.ndarray, bandwidth: float = 3.0) -> float:
    """D8's scalar (added 2026-09-01, user-directed: "another diagnostic
    D8 that uses signed correlation"). In the CURRENT, FIXED latent index
    order (no Fiedler permutation search, unlike D7 -- that assumes
    non-negative edge weights, which a signed correlation matrix does not
    satisfy): a genuinely different question from D7's "does *some*
    channel ordering reveal coupling structure" -- D8 asks "does the
    ACTUAL, in-use ordering exhibit SAME-SIGN local coherence," which is
    exactly what a signed `w_spatial` term (Stage1/
    Stage2TrainingConfig.spatial_signed) optimizes for directly, and which
    D7 (built on `|correlation|`) cannot distinguish from anti-correlated
    local coupling.

    `score = weighted_mean(A_off, W) - unweighted_mean(A_off)`: the
    Gaussian-circular-band-weighted mean of the off-diagonal SIGNED
    correlations, minus their plain (unweighted) mean -- i.e. "are
    near-diagonal pairs MORE positively correlated than the average pair,
    on net." Deliberately NOT `spatial_coherence_loss`'s unsigned
    `numer/denom` ratio ported naively to a signed `A`: an EARLIER version
    of this function did exactly that and was caught failing its own unit
    test (`test_d8_distinguishes_anti_correlated_local_structure_that_
    d7_cannot`) -- an alternating-sign local structure (immediate
    neighbors systematically ANTI-correlated) makes both the ratio's
    numerator and denominator strongly NEGATIVE, and a negative-over-
    negative ratio reports a spuriously HIGH POSITIVE score, exactly
    backwards from "significant same-sign coherence." The weighted-minus-
    unweighted-mean form has no such failure mode: both terms are means of
    the SAME (unbounded, fixed-denominator) sum, so an overall sign bias
    in `A` shifts both terms together and cancels out of the difference,
    leaving only the genuine near-vs-far CONTRAST that "local coherence"
    is supposed to measure. NOT bounded to `[0, 1]` (or any fixed range)
    the way D7's `abs`-based score is -- report alongside its own null
    (`signed_bandedness_p_value`), not compared numerically against D7."""
    d = A.shape[0]
    idx = np.arange(d)
    dist = np.minimum(np.abs(idx[:, None] - idx[None, :]), d - np.abs(idx[:, None] - idx[None, :]))
    W = np.exp(-(dist**2) / (2.0 * bandwidth**2))
    off_diag = 1.0 - np.eye(d)
    A_off = A * off_diag
    W_off = W * off_diag
    weighted_mean = (A_off * W_off).sum() / W_off.sum()
    unweighted_mean = A_off.sum() / off_diag.sum()
    return float(weighted_mean - unweighted_mean)


def signed_bandedness_p_value(
    A: np.ndarray, n_null: int = 500, bandwidth: float = 3.0, seed: int = 0
) -> tuple[float, float]:
    """D8's null: random PERMUTATIONS of the channel labels applied to the
    SAME signed matrix (not `bandedness_p_value_entry_shuffle`'s entry-
    shuffle-then-Fiedler-reseriate, and not a Laplacian/Fiedler search at
    all -- `_fiedler_permutation` assumes non-negative edge weights for
    its graph-Laplacian construction, which a signed correlation matrix
    does not satisfy). Answers "is the REAL, fixed index order's signed
    bandedness better than a random relabeling of the same channels'
    pairwise signed correlations would typically achieve" -- the natural
    null for a FIXED-order statistic, as opposed to D7's "is the best-
    achievable-under-any-permutation bandedness better than chance."
    Returns `(observed, p_value)`."""
    d = A.shape[0]
    observed = signed_bandedness(A, bandwidth)
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_null)
    for i in range(n_null):
        perm = rng.permutation(d)
        A_perm = A[np.ix_(perm, perm)]
        null_scores[i] = signed_bandedness(A_perm, bandwidth)
    p_value = float(np.mean(null_scores >= observed))
    return observed, p_value


def same_time_coupling_diagnostic_signed(
    Z: np.ndarray, n_null: int = 500, seed: int = 0, bandwidth: float = 3.0,
) -> SameTimeCouplingSignedResult:
    """Full D8 pipeline -- see `signed_bandedness`/`signed_bandedness_p_value`
    docstrings for the mechanism and how it differs from D7."""
    A = same_time_channel_correlation_signed(Z)
    observed, p_value = signed_bandedness_p_value(A, n_null=n_null, bandwidth=bandwidth, seed=seed)
    return SameTimeCouplingSignedResult(A=A, bandedness_observed=observed, bandedness_p_value=p_value)


# ---------------------------------------------------------------------------
# D9: real-trajectory smoothness (step size / curvature / propagator
# Jacobian norm) -- added 2026-09-05, user-directed ("please run
# visualizations, smoothness diagnostic + Gate 3/4 (in the future add
# smoothness diagnostic to gate 4)"). Originally a one-off comparison
# script (`scripts/analyze_latent_smoothness.py`, built the same session
# to quantify a purely visual "huge jumps in the GIF" observation on
# Section 82); moved here so both that script and Gate 4
# (`scripts/run_diagnostics.py`) share one implementation instead of
# duplicating it. See `ks_latent.training.losses.temporal_smoothness_loss`
# for the differentiable training-time analogue of the step/curvature half
# of this diagnostic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothnessResult:
    """Real-trajectory step size / curvature (encoder-only, propagator-
    independent) plus the trained propagator's own step-Jacobian spectral
    norm distribution (sampled at real points on the attractor). All eight
    scalar fields are median/95th-percentile pairs; see
    `encoded_trajectory_smoothness`/`propagator_step_jacobian_spectral_norms`
    for the exact definitions. `step_norm`/`curv_norm` are normalized by
    each trajectory's own RMS latent radius, so different checkpoints'
    differing overall latent scales stay comparable -- `propagator_jacobian`
    is deliberately NOT normalized this way (it's already scale-relative,
    a ratio of output to input perturbation size)."""

    step_med: float
    step_p95: float
    step_norm_med: float
    step_norm_p95: float
    curv_med: float
    curv_p95: float
    curv_norm_med: float
    curv_norm_p95: float
    propagator_jacobian_med: float
    propagator_jacobian_p95: float
    propagator_jacobian_supported: bool
    propagator_jacobian_skip_reason: str


def encoded_trajectory_smoothness(z_seq: np.ndarray) -> dict:
    """`z_seq`: `(n_runs, T, d_latent)` real encoded trajectories. Returns
    step-size (`||z_{t+1}-z_t||`) and curvature (`||z_{t+1}-2*z_t+z_{t-1}||`)
    percentiles, raw and normalized by each trajectory's own RMS latent
    radius (`sqrt(mean(||z||^2))` over that run)."""
    step = np.linalg.norm(z_seq[:, 1:] - z_seq[:, :-1], axis=-1)  # (n, T-1)
    curv = np.linalg.norm(
        z_seq[:, 2:] - 2 * z_seq[:, 1:-1] + z_seq[:, :-2], axis=-1
    )  # (n, T-2)
    rms_radius = np.sqrt((z_seq**2).sum(-1).mean(-1))  # (n,)
    step_norm = step / rms_radius[:, None]
    curv_norm = curv / rms_radius[:, None]
    return {
        "step_med": np.median(step), "step_p95": np.percentile(step, 95),
        "step_norm_med": np.median(step_norm), "step_norm_p95": np.percentile(step_norm, 95),
        "curv_med": np.median(curv), "curv_p95": np.percentile(curv, 95),
        "curv_norm_med": np.median(curv_norm), "curv_norm_p95": np.percentile(curv_norm, 95),
    }


def propagator_step_jacobian_spectral_norms(
    prop, z_seq: torch.Tensor, n_samples: int = 200, seed: int = 0
) -> np.ndarray:
    """`z_seq`: `(n_runs, T, d_latent)` real encoded trajectories (torch).
    Samples `n_samples` real (history-window, next-state) points and
    returns the spectral norm (largest singular value) of the trained
    propagator's own step Jacobian at each -- the local Lipschitz constant
    of the LEARNED dynamics, evaluated ON the attractor rather than at
    random off-manifold points. Supports `mode in ("markovian", "history")`
    only -- raises `ValueError` for `"two_step"` (not yet implemented;
    callers should catch this and skip, not treat it as a hard failure)."""
    prop = prop.to("cpu").eval()
    mode = prop.cfg.mode
    n_runs, T, d = z_seq.shape
    rng = np.random.default_rng(seed)
    norms = []
    if mode == "markovian":
        def f(z):
            return prop.step_one(z.unsqueeze(0)).squeeze(0)

        run_idx = rng.integers(0, n_runs, size=n_samples)
        t_idx = rng.integers(0, T, size=n_samples)
        for r, t in zip(run_idx, t_idx):
            J = jacrev(f)(z_seq[r, t])  # (d, d)
            norms.append(torch.linalg.svdvals(J)[0].item())
    elif mode == "history":
        n_hist = prop.cfg.n_history

        def f(flat):
            return prop.step_history(flat.view(1, n_hist, d)).squeeze(0)

        run_idx = rng.integers(0, n_runs, size=n_samples)
        t_idx = rng.integers(n_hist - 1, T, size=n_samples)
        for r, t in zip(run_idx, t_idx):
            hist = z_seq[r, t - n_hist + 1 : t + 1].reshape(-1)  # (n_hist*d,)
            J = jacrev(f)(hist)  # (d, n_hist*d)
            norms.append(torch.linalg.svdvals(J)[0].item())
    else:
        raise ValueError(f"propagator_step_jacobian_spectral_norms: unsupported mode {mode!r}")
    return np.array(norms)


def propagator_step_jacobian_full_spectrum(
    prop, z_seq: torch.Tensor, n_samples: int = 200, seed: int = 0
) -> np.ndarray:
    """Like `propagator_step_jacobian_spectral_norms`, but returns the FULL
    sorted-descending singular-value spectrum at each sampled point (not
    just the top entry): `(n_samples, d)`. Added 2026-09-10, Section 134,
    to calibrate `ks_latent.training.losses.propagator_graded_spectrum_shape_loss`'s
    per-rank reference target from a trusted checkpoint's own real
    spectrum -- the median across samples at each rank is a robust
    empirical target, unlike a single point's spectrum (which is noisy: a
    single-point check this session initially suggested the real spectrum
    was close to flat near 1.0-1.1, and only a proper 200-sample per-rank
    median revealed it is actually a smooth graded decline -- rank 0
    median 1.602, rank 12 median 0.955, continuing down to 0.15-0.3 at the
    bottom ranks -- nothing like two flat groups). Same mode support
    (`"markovian"`/`"history"`, raises for `"two_step"`) as
    `propagator_step_jacobian_spectral_norms`."""
    prop = prop.to("cpu").eval()
    mode = prop.cfg.mode
    n_runs, T, d = z_seq.shape
    rng = np.random.default_rng(seed)
    spectra = []
    if mode == "markovian":
        def f(z):
            return prop.step_one(z.unsqueeze(0)).squeeze(0)

        run_idx = rng.integers(0, n_runs, size=n_samples)
        t_idx = rng.integers(0, T, size=n_samples)
        for r, t in zip(run_idx, t_idx):
            J = jacrev(f)(z_seq[r, t])  # (d, d)
            spectra.append(torch.linalg.svdvals(J).detach().numpy())
    elif mode == "history":
        n_hist = prop.cfg.n_history

        def f(flat):
            return prop.step_history(flat.view(1, n_hist, d)).squeeze(0)

        run_idx = rng.integers(0, n_runs, size=n_samples)
        t_idx = rng.integers(n_hist - 1, T, size=n_samples)
        for r, t in zip(run_idx, t_idx):
            hist = z_seq[r, t - n_hist + 1 : t + 1].reshape(-1)  # (n_hist*d,)
            J = jacrev(f)(hist)  # (d, n_hist*d)
            spectra.append(torch.linalg.svdvals(J).detach().numpy())  # min(d, n_hist*d) = d values
    else:
        raise ValueError(f"propagator_step_jacobian_full_spectrum: unsupported mode {mode!r}")
    return np.stack(spectra)  # (n_samples, d)


def smoothness_diagnostic(
    z_seq_np: np.ndarray, z_seq_torch: torch.Tensor, prop, n_samples: int = 200, seed: int = 0,
) -> SmoothnessResult:
    """Full D9 pipeline: `encoded_trajectory_smoothness` (always computed)
    plus `propagator_step_jacobian_spectral_norms` (skipped gracefully,
    with `propagator_jacobian_supported=False`, for propagator modes that
    don't support it yet -- e.g. `mode="two_step"` -- rather than raising
    and aborting the rest of Gate 4)."""
    enc = encoded_trajectory_smoothness(z_seq_np)
    try:
        jac_norms = propagator_step_jacobian_spectral_norms(prop, z_seq_torch, n_samples=n_samples, seed=seed)
        jac_med, jac_p95 = float(np.median(jac_norms)), float(np.percentile(jac_norms, 95))
        supported, skip_reason = True, ""
    except ValueError as e:
        jac_med, jac_p95 = float("nan"), float("nan")
        supported, skip_reason = False, str(e)
    return SmoothnessResult(
        step_med=enc["step_med"], step_p95=enc["step_p95"],
        step_norm_med=enc["step_norm_med"], step_norm_p95=enc["step_norm_p95"],
        curv_med=enc["curv_med"], curv_p95=enc["curv_p95"],
        curv_norm_med=enc["curv_norm_med"], curv_norm_p95=enc["curv_norm_p95"],
        propagator_jacobian_med=jac_med, propagator_jacobian_p95=jac_p95,
        propagator_jacobian_supported=supported, propagator_jacobian_skip_reason=skip_reason,
    )
