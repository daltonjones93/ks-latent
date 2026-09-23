"""RMSE normalization (brief §7: "divide the vector norm by sqrt(d) for
both free and DA runs -> per-dimension RMSE consistent with the spread").

A single shared function so free-run and DA RMSE can never silently drift
apart onto different conventions.
"""

from __future__ import annotations

import torch


def per_dim_rmse(error: torch.Tensor) -> float:
    """`||error||_2 / sqrt(d)` for a `(d,)` error vector, or per-row for a
    `(..., d)` batch (returns the mean over the batch)."""
    d = error.shape[-1]
    norm = error.norm(dim=-1)
    return float((norm / d**0.5).mean())
