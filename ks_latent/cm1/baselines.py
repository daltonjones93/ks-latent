"""Equation-free forecast baselines for the CM1 surrogate (plan H1).

Advection extrapolation (Lagrangian persistence), the nowcasting standard: the latest state is translated
along x at the speed it moved over the last `lag` frames, estimated by maximizing the correlation between
the state `lag` frames ago, shifted, and the latest state. Two variants:

  global     one shift for the whole column state (the storm moves as a unit);
  per_level  one shift per model level, for all variables at that level jointly (the anvil, the cold pool
             and the inflow move differently).

Shifts are whole cells per `lag` frames, so the speed is resolved to 1 / lag cells per frame; the forecast at
lead n moves the field by round(n * shift / lag) cells. Columns that enter through an edge take the edge
column of the latest state (the far field is quiescent). All fields are the normalized column state
`(C, NX)` with C = V * Z, variable-major (`data.ColumnData.full`).
"""
from __future__ import annotations

import torch


def _shift(u: torch.Tensor, s: int) -> torch.Tensor:
    """Translate (..., NX) by s cells (s > 0: to larger x), edge values filling the vacated cells."""
    if s == 0:
        return u.clone()
    NX = u.shape[-1]
    s = max(-NX + 1, min(NX - 1, s))
    if s > 0:
        return torch.cat([u[..., :1].expand(*u.shape[:-1], s), u[..., :NX - s]], dim=-1)
    return torch.cat([u[..., -s:], u[..., -1:].expand(*u.shape[:-1], -s)], dim=-1)


def best_shift(prev: torch.Tensor, last: torch.Tensor, max_shift: int) -> int:
    """Integer s in [-max_shift, max_shift] minimizing the mean squared difference between prev shifted
    by s and last, on the columns both cover (edge-filled cells excluded). Searched in order of |s|, and a
    larger shift must be better by a relative 1e-9, so a featureless level keeps s = 0."""
    NX = prev.shape[-1]
    best, best_e = 0, float("inf")
    for s in sorted(range(-max_shift, max_shift + 1), key=lambda v: (abs(v), v)):
        a = _shift(prev, s)
        lo, hi = max(0, s), NX + min(0, s)
        e = float(((a[..., lo:hi] - last[..., lo:hi]) ** 2).mean())
        if e < best_e * (1 - 1e-9) - 1e-30:
            best, best_e = s, e
    return best


def advection_forecast(hist: torch.Tensor, n_lead: int, n_levels: int, lag: int = 5, max_shift: int = 40,
                       mode: str = "per_level") -> tuple[torch.Tensor, list[int]]:
    """`hist` (lag + 1, C, NX): frames t0 - lag ... t0. Returns the forecast (n_lead, C, NX) for leads
    1..n_lead and the shift(s) per `lag` frames that it used."""
    if hist.shape[0] != lag + 1:
        raise ValueError(f"need lag + 1 = {lag + 1} frames, got {hist.shape[0]}")
    prev, last = hist[0], hist[-1]
    C, NX = last.shape
    V = C // n_levels
    if V * n_levels != C:
        raise ValueError(f"{C} channels are not a whole number of {n_levels}-level variables")
    if mode == "global":
        shifts = [best_shift(prev, last, max_shift)] * n_levels
    elif mode == "per_level":
        lv = lambda u, k: u.reshape(V, n_levels, NX)[:, k]
        shifts = [best_shift(lv(prev, k), lv(last, k), max_shift) for k in range(n_levels)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out = torch.empty(n_lead, V, n_levels, NX, dtype=last.dtype, device=last.device)
    L = last.reshape(V, n_levels, NX)
    for n in range(1, n_lead + 1):
        for k in range(n_levels):
            out[n - 1, :, k] = _shift(L[:, k], round(n * shifts[k] / lag))
    return out.reshape(n_lead, C, NX), shifts
