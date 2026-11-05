"""Gust-front and convective-scale verification metrics for the CM1 squall line (1D in x).

Why not "the easternmost cold cell": a new convective cell ahead of the line drops a small cold pool
of its own, and the easternmost-cell front then jumps by kilometres, at different times in the truth
and the forecast. Errors become heavy-tailed and dominated by detector noise. The metrics here:

1. `main_pool_front_km` - the leading (east) edge of the MAIN cold pool. Cold surface cells
   (theta' < thresh) form runs; runs separated by gaps narrower than `merge_gap_km` are one pool
   (small warm holes inside the pool); runs narrower than `min_width_km` are ignored. Among pools
   with at least `mass_frac` of the largest pool's cold mass (sum of -theta' over the run), the
   easternmost is the main pool. (Taking simply the largest flips between two comparable pools of a
   young storm, found on the original members: 8 spurious jumps of ~11 km in the first 90 min.)
   A new cell ahead of the line moves the front only once its cold pool merges with the line or
   grows to a sizeable fraction of it: a real advance (discrete propagation).
2. `displacement_km` - threshold-free signed position error: the shift s that best aligns the
   forecast's surface theta' with the truth (min mean squared difference) in a window around the true
   front, refined below one cell by a parabola through the minimum. s > 0: the forecast is east of the
   truth (ahead).
3. `fss` - the Fractions Skill Score (Roberts & Lean 2008, Mon. Wea. Rev.): binary exceedance fields
   smoothed by a moving window of n cells, FSS = 1 - MSE(f_fc, f_ob) / (mean f_fc^2 + mean f_ob^2).
   1 is perfect, 0 no overlap at that scale; scores rise with n for a displaced but otherwise right
   forecast, so the scale at which FSS passes 0.5 + f_ob/2 is the "useful scale".
All functions take numpy arrays, rows of length NX (cells of `dx_km`).
"""
from __future__ import annotations

import numpy as np

DX_KM = 0.25


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of the True runs of a 1D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def main_pool_front_km(th_sfc: np.ndarray, thresh: float = -1.0, min_width_km: float = 2.0,
                       merge_gap_km: float = 2.0, mass_frac: float = 0.25, dx_km: float = DX_KM) -> np.ndarray:
    """East edge (km) of the main cold pool, per row of `(..., NX)`; NaN where there is none."""
    a = np.asarray(th_sfc, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    out = np.full(flat.shape[0], np.nan)
    gap, wmin = int(round(merge_gap_km / dx_km)), int(round(min_width_km / dx_km))
    for r, row in enumerate(flat):
        runs = _runs(row < thresh)
        if not runs:
            continue
        merged = [list(runs[0])]
        for s, e in runs[1:]:
            if s - merged[-1][1] - 1 < gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        pools = [(s, e, float(-np.minimum(row[s:e + 1], 0.0).sum())) for s, e in merged if e - s + 1 >= wmin]
        if not pools:
            continue
        big = max(m for _, _, m in pools)
        east = max(e for s, e, m in pools if m >= mass_frac * big)
        out[r] = (east + 1) * dx_km  # east edge of the last cold cell of the main pool
    return out.reshape(a.shape[:-1])


def displacement_km(fc: np.ndarray, ob: np.ndarray, center_km: float, window_km: float = 40.0,
                    max_shift_km: float = 15.0, dx_km: float = DX_KM) -> float:
    """Signed shift (km) of `fc` relative to `ob` (1D rows): the s minimizing mean (fc(x+s) - ob(x))^2
    over a window centred on `center_km`; s > 0 means the forecast feature lies east of the truth's.
    NaN if the window does not fit in the domain."""
    fc, ob = np.asarray(fc, np.float64), np.asarray(ob, np.float64)
    NX = ob.shape[-1]
    if not np.isfinite(center_km):
        return float("nan")
    c, h, S = int(round(center_km / dx_km)), int(round(window_km / 2 / dx_km)), int(round(max_shift_km / dx_km))
    lo, hi = c - h, c + h
    lo, hi = max(lo, S), min(hi, NX - S)
    if hi - lo < 8:
        return float("nan")
    x = np.arange(lo, hi)
    shifts = np.arange(-S, S + 1)
    err = np.array([np.mean((fc[x + s] - ob[x]) ** 2) for s in shifts])
    k = int(np.argmin(err))
    sub = 0.0
    if 0 < k < len(err) - 1:  # parabola through the minimum and its neighbours
        den = err[k - 1] - 2 * err[k] + err[k + 1]
        if den > 0:
            sub = 0.5 * (err[k - 1] - err[k + 1]) / den
    return float((shifts[k] + sub) * dx_km)


def _moving_fraction(binary: np.ndarray, n: int) -> np.ndarray:
    """Fraction of True cells in a centred window of n cells (fewer cells near the open edges)."""
    b = np.asarray(binary, np.float64)
    if n <= 1:
        return b
    k = np.ones(n)
    num = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), -1, b)
    den = np.convolve(np.ones(b.shape[-1]), k, mode="same")
    return num / den


def fss(fc: np.ndarray, ob: np.ndarray, threshold: float, scales_cells) -> dict:
    """Fractions Skill Score of `fc` against `ob` (same shape, last axis x) for each window size."""
    bf, bo = np.asarray(fc) > threshold, np.asarray(ob) > threshold
    out = {}
    for n in scales_cells:
        ff, fo = _moving_fraction(bf, n), _moving_fraction(bo, n)
        ref = np.mean(ff ** 2) + np.mean(fo ** 2)
        out[int(n)] = float("nan") if ref == 0 else float(1.0 - np.mean((ff - fo) ** 2) / ref)
    return out
