"""Multivariate normality tests for small ensembles (Z20 E2, Paper II App. D).

`mardia(X)` returns Mardia's (1970) multivariate skewness b1 and kurtosis b2 of the rows of `X`
(`N` samples, `p` dimensions), with the sample covariance (divisor N):

    g_ij = (x_i - xbar)^T S^-1 (x_j - xbar),   b1 = (1/N^2) sum_ij g_ij^3,   b2 = (1/N) sum_i g_ii^2.

For Gaussian data E[b2] -> p(p+2) and N b1 / 6 -> chi^2 with p(p+1)(p+2)/6 degrees of freedom; at
the ensemble sizes used here the asymptotics are poor, so `mc_pvalues` calibrates both statistics
by Monte Carlo: Gaussian samples of the same N and p (the statistics are affine invariant, so the
covariance does not matter), passed through the same optional preprocessing (`project`).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def mardia(X: np.ndarray) -> tuple[float, float]:
    N, p = X.shape
    if N <= p + 1:
        raise ValueError(f"need N > p + 1 samples, got N={N}, p={p}")
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / N
    w = np.linalg.eigvalsh(S)
    if w.min() <= 1e-12 * w.max():
        raise np.linalg.LinAlgError(f"singular sample covariance (condition number {w.max() / max(w.min(), 1e-300):.3e})")
    G = Xc @ np.linalg.solve(S, Xc.T)
    return float((G ** 3).sum() / N ** 2), float((np.diag(G) ** 2).sum() / N)


def mc_pvalues(stat: tuple[float, float], N: int, p: int, n_mc: int, rng: np.random.Generator,
               project: Callable[[np.ndarray], np.ndarray] | None = None, dim_in: int | None = None) -> tuple[float, float]:
    """Two p-values (skewness: one-sided upper; kurtosis: two-sided) of `stat` against `n_mc` Gaussian
    samples of `N` x `dim_in` (default `p`), each passed through `project` (default identity) first."""
    b1s, b2s = np.empty(n_mc), np.empty(n_mc)
    for k in range(n_mc):
        Z = rng.standard_normal((N, dim_in or p))
        b1s[k], b2s[k] = mardia(project(Z) if project else Z)
    p1 = (1 + np.sum(b1s >= stat[0])) / (n_mc + 1)
    lo, hi = np.sum(b2s <= stat[1]), np.sum(b2s >= stat[1])
    p2 = min(1.0, 2 * (1 + min(lo, hi)) / (n_mc + 1))
    return float(p1), float(p2)
