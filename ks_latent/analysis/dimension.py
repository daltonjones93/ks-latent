"""Intrinsic dimension estimators (brief §6.1).

Two-NN was implemented in Phase 1, where it is needed to demonstrate the
temporal-correlation failure mode that motivates `AttractorPointDataset`
(brief §3.3: "thinning does not fix it"). Correlation dimension and
diffusion maps are added here (Phase 4), along with the full d in
{2,5,10,15,20,22} sphere/torus validation sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class TwoNNResult:
    dimension: float
    mu: np.ndarray  # r2/r1 ratio per point
    n_points: int


def two_nn_dimension(X: np.ndarray, discard_fraction: float = 0.1) -> TwoNNResult:
    """Two-NN intrinsic dimension estimator (Facco, Pigolotti, Rosasco &
    Laio 2017, "Estimating the intrinsic dimension of datasets by a minimal
    neighborhood information", Sci. Rep. 7:12140).

    Under local-uniform-density assumptions, `mu_i = r2_i / r1_i` (ratio of
    distances to the first and second nearest neighbor) follows a Pareto
    distribution with shape `d`: `P(mu > x) = x**-d`. Fits
    `-log(1 - F(mu)) = d * log(mu)` through the origin (no intercept) over
    the central `1 - discard_fraction` of the empirical CDF, using the
    midpoint convention `F_i = (i - 0.5) / N`.

    Known downward bias at high true dimension (the brief documents this
    reading true d=22 as roughly 18-19); this is a property of the
    estimator, not a bug -- do not "fix" it by recalibrating the fit.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 10:
        raise ValueError(f"two_nn_dimension needs at least 10 points, got {N}")

    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)
    r1, r2 = dists[:, 1], dists[:, 2]
    if np.any(r1 <= 0.0):
        raise ValueError(
            "two_nn_dimension: found duplicate points (nearest-neighbor distance "
            "0). Remove duplicates before estimating dimension."
        )
    mu = r2 / r1

    mu_sorted = np.sort(mu)
    F = (np.arange(1, N + 1) - 0.5) / N
    x = np.log(mu_sorted)
    y = -np.log(1.0 - F)

    lo = int(np.floor(0.5 * discard_fraction * N))
    hi = int(np.ceil((1.0 - 0.5 * discard_fraction) * N))
    x_c, y_c = x[lo:hi], y[lo:hi]

    d = float(np.sum(x_c * y_c) / np.sum(x_c * x_c))
    return TwoNNResult(dimension=d, mu=mu, n_points=N)


def two_nn_convergence_curve(
    X: np.ndarray,
    sample_sizes: list[int],
    n_repeats: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[int, tuple[float, float]]:
    """Mean/std of the two-NN estimate as a function of sample size.

    "An estimate that has not visibly plateaued is not trustworthy" (brief
    §6.1) -- call this before reporting any dimension number from a new
    dataset and check the curve has flattened by the largest `sample_sizes`
    entry.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = X.shape[0]
    out: dict[int, tuple[float, float]] = {}
    for n in sample_sizes:
        if n > N:
            raise ValueError(f"sample size {n} exceeds available points {N}")
        estimates = []
        for _ in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            estimates.append(two_nn_dimension(X[idx]).dimension)
        out[n] = (float(np.mean(estimates)), float(np.std(estimates)))
    return out


@dataclass(frozen=True)
class CorrelationDimResult:
    dimension: float
    radii: np.ndarray
    log_correlation_sum: np.ndarray
    fit_mask: np.ndarray


def correlation_sum(X: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Grassberger-Procaccia correlation sum `C(r)` at each radius in `radii`:
    the fraction of point pairs closer than `r`."""
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    tree = cKDTree(X)
    # count_neighbors counts every ordered pair (including self-pairs, each
    # unordered pair twice); convert to the standard unordered-pair fraction.
    counts = np.asarray(tree.count_neighbors(tree, radii), dtype=np.float64)
    pair_counts = (counts - N) / 2.0
    return pair_counts / (N * (N - 1) / 2.0)


def correlation_dimension(
    X: np.ndarray,
    n_radii: int = 30,
    fit_quantiles: tuple[float, float] = (0.2, 0.7),
) -> CorrelationDimResult:
    """Grassberger-Procaccia correlation dimension (Grassberger & Procaccia
    1983, Physica D 9:189): slope of `log C(r)` vs `log r` over a scaling
    region in the middle of the radius range.

    This is a **lower bound** on the true (e.g. two-NN or box-counting)
    dimension in general -- document it as such when reporting, don't treat
    a discrepancy with two-NN as evidence one of them is wrong.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N < 20:
        raise ValueError(f"correlation_dimension needs at least 20 points, got {N}")

    dists = pdist(X[: min(N, 500)])  # small subsample just to pick a radius range
    r_min, r_max = np.quantile(dists, [0.01, 0.9])
    radii = np.geomspace(r_min, r_max, n_radii)

    C = correlation_sum(X, radii)
    valid = C > 0
    log_r, log_C = np.log(radii[valid]), np.log(C[valid])

    lo_q, hi_q = fit_quantiles
    lo, hi = int(lo_q * len(log_r)), int(hi_q * len(log_r))
    if hi - lo < 2:
        raise ValueError("Not enough valid radii in the fit window; widen fit_quantiles.")
    fit_mask = np.zeros(len(log_r), dtype=bool)
    fit_mask[lo:hi] = True

    slope, _ = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
    return CorrelationDimResult(
        dimension=float(slope), radii=radii[valid], log_correlation_sum=log_C, fit_mask=fit_mask
    )


@dataclass(frozen=True)
class DiffusionMapResult:
    eigenvalues: np.ndarray  # descending, trivial eigenvalue ~1 already dropped
    eigenvectors: np.ndarray  # (N, n_components)
    embedding: np.ndarray  # eigenvectors scaled by eigenvalues**t


def median_sqdist_bandwidth(X: np.ndarray, sample_size: int = 500, rng=None) -> float:
    """Median-heuristic squared-distance bandwidth for the diffusion kernel."""
    if rng is None:
        rng = np.random.default_rng(0)
    N = X.shape[0]
    idx = rng.choice(N, size=min(sample_size, N), replace=False)
    return float(np.median(pdist(X[idx], "sqeuclidean")))


def diffusion_maps(
    X: np.ndarray, epsilon: float | None = None, n_components: int = 10, alpha: float = 1.0, t: int = 1
) -> DiffusionMapResult:
    """Diffusion maps (Coifman & Lafon 2006, Appl. Comput. Harmon. Anal.
    21:5-30): alpha-normalized Gaussian-kernel Markov chain, eigendecomposed
    via its symmetric conjugate for numerical stability.

    Only a diagnostic/visualization tool here (brief §6.1: "with spectrum
    plotting") -- unlike two-NN/correlation dimension it doesn't return a
    single "dimension" number; look at the eigenvalue spectrum shape (a
    smooth decay with no gap, vs. a handful of dominant modes) and at
    whether the leading eigenvectors correlate with known structure.

    O(N^2) memory/time (dense pairwise distances) -- subsample large point
    clouds before calling this.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if epsilon is None:
        epsilon = median_sqdist_bandwidth(X)

    D2 = squareform(pdist(X, "sqeuclidean"))
    K = np.exp(-D2 / epsilon)
    if alpha != 0:
        d = K.sum(axis=1)
        K = K / np.outer(d**alpha, d**alpha)
    d2 = K.sum(axis=1)
    d2_inv_sqrt = 1.0 / np.sqrt(d2)
    # P = diag(1/d2) @ K is the Markov matrix; symmetrize via
    # Ms = diag(d2^-1/2) @ K @ diag(d2^-1/2), similar to P (same eigenvalues,
    # eigenvectors related by psi = d2^-1/2 * eigvecs_of_Ms).
    Ms = d2_inv_sqrt[:, None] * K * d2_inv_sqrt[None, :]
    eigvals, eigvecs_sym = np.linalg.eigh(Ms)
    order = np.argsort(eigvals)[::-1]
    n_keep = min(n_components + 1, N)
    eigvals = eigvals[order][:n_keep]
    eigvecs_sym = eigvecs_sym[:, order][:, :n_keep]

    psi = d2_inv_sqrt[:, None] * eigvecs_sym
    psi = psi / np.linalg.norm(psi, axis=0, keepdims=True)  # normalize embedding vectors

    # Drop the trivial top eigenvalue (~1, constant eigenvector).
    eigenvalues = eigvals[1:]
    eigenvectors = psi[:, 1:]
    embedding = eigenvectors * (eigenvalues**t)[None, :]
    return DiffusionMapResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors, embedding=embedding)


def sample_sphere(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points on the surface of the unit d-sphere, embedded in R^(d+1)."""
    x = rng.normal(size=(n, d + 1))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_torus(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points on the flat d-torus: d independent angles, each embedded as
    (cos, sin), giving an embedding dimension of 2d for intrinsic dimension d."""
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, d))
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
