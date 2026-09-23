"""Circular statistics (brief §8 D1: "the domain is periodic and linear
statistics are wrong").

Weighted circular mean/spread for a periodic domain, via the standard
resultant-vector construction (Mardia & Jupp, *Directional Statistics*,
2000): treat positions on a period-`L` circle as angles `theta = 2*pi*x/L`,
and use the mean resultant vector `(mean cos, mean sin)` rather than a plain
weighted average of `x` (which is wrong wherever the weight mass straddles
the wrap-around point).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircularStats:
    centroid: float  # position, same units as x (i.e. in [0, L))
    resultant_length: float  # R in [0, 1]; 1 = a delta function, 0 = uniform
    spread: float  # circular standard deviation, sqrt(-2 ln R)


def circular_weighted_stats(x: np.ndarray, weights: np.ndarray, L: float) -> CircularStats:
    """`x`: positions in `[0, L)`. `weights`: non-negative weights (e.g. a
    sensitivity profile); need not sum to 1.

    All-zero `weights` (added 2026-09-07, caught by
    `decoder_sensitivity_map` crashing outright on a genuinely DEAD latent
    channel -- `encoder_kind="spectral_field"`'s DC-imaginary slot, which
    is mathematically always exactly zero and therefore has EXACTLY zero
    decoder sensitivity everywhere, see `ks_latent.models.spectral_field`'s
    module docstring): this is a real, meaningful, reportable diagnostic
    state (a channel with no measurable spatial influence at all) that
    should be surfaced, not treated as an error that aborts the whole
    report -- any encoder architecture with a collapsed/dead latent
    channel could hit this, not just this one. Returns `resultant_length=
    0.0` (no concentration, consistent with "no measurable direction"),
    `centroid=nan` (genuinely undefined -- there is no meaningful position
    to report when nothing depends on this channel at all), `spread=inf`
    (maximally diffuse). Negative weights remain a hard error (a real bug,
    not a legitimate diagnostic state)."""
    if np.any(weights < 0):
        raise ValueError("circular_weighted_stats: weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        return CircularStats(centroid=float("nan"), resultant_length=0.0, spread=float("inf"))
    theta = 2.0 * np.pi * x / L
    c = np.sum(weights * np.cos(theta)) / total
    s = np.sum(weights * np.sin(theta)) / total
    R = float(np.hypot(c, s))
    R = min(R, 1.0)  # guard against roundoff pushing R fractionally above 1
    centroid_theta = np.arctan2(s, c) % (2.0 * np.pi)
    centroid = centroid_theta / (2.0 * np.pi) * L
    spread = float(np.sqrt(-2.0 * np.log(max(R, 1e-300))))
    return CircularStats(centroid=float(centroid), resultant_length=R, spread=spread)


def circular_distance(i: np.ndarray | int, j: np.ndarray | int, n: int) -> np.ndarray | int:
    """Shortest distance between indices `i`, `j` on a ring of `n` sites."""
    d = np.abs(np.asarray(i) - np.asarray(j)) % n
    return np.minimum(d, n - d)
