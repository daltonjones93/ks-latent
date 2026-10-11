"""Shifted tilings on rectangular periodic grids (Paper II, Kolmogorov DA, SK3).

The 2D KS experiments (`scripts/exp_h9_tiled_da.py`) work on square domains with the helpers inline;
the Kolmogorov strip is rectangular (`NX` x `NY`, `NX` growing with the channel width, `NY` fixed),
so the same three pieces are factored out here and tested:

- `tiling_offsets`: the grid-point offsets of K tilings, optionally only along x (Kolmogorov flow
  is translation-invariant in x but only by whole forcing periods in y, so a y-shift by half a patch
  shows the encoder patches at forcing phases it never saw in training).
- `blend_weights`: separable Hann weights, one per tiling, vanishing on that tiling's own patch
  edges and normalized to a partition of unity on the rectangle (exactly 1 before normalization for
  offsets {0, p/2} along an axis).
- `frame_obs_sites`: for a regular observation network, the observed pixels in tiling o's frame
  (the field rolled by -o) and the site that owns each, in the original observation order, so every
  tiling assimilates the same observation vector.
"""
from __future__ import annotations

import numpy as np
import torch


def tiling_offsets(patch: int, n_tilings: int, axes: str = "xy") -> list[tuple[int, int]]:
    """Offsets (ox, oy) in grid points. `axes="xy"`: 1, 2 (diagonal) or 4 (half-patch grid) tilings;
    `axes="x"`: 1, 2 or 4 tilings shifted along x only, by multiples of patch / n_tilings."""
    hp = patch // 2
    if axes == "xy":
        table = {1: [(0, 0)], 2: [(0, 0), (hp, hp)], 4: [(0, 0), (hp, 0), (0, hp), (hp, hp)]}
        if n_tilings not in table:
            raise ValueError(f"axes='xy' supports 1, 2 or 4 tilings, got {n_tilings}")
        return table[n_tilings]
    if axes == "x":
        if patch % n_tilings:
            raise ValueError(f"patch {patch} not divisible by {n_tilings} tilings")
        return [(k * patch // n_tilings, 0) for k in range(n_tilings)]
    raise ValueError(f"axes must be 'xy' or 'x', got {axes!r}")


def hann_1d(n: int, patch: int, offset: int) -> np.ndarray:
    """sin^2 weight over each patch of a lattice shifted by `offset`: zero-free, smallest at that
    lattice's patch edges, largest at its patch centres."""
    t = (np.arange(n) - offset) % patch
    return np.sin(np.pi * (t + 0.5) / patch) ** 2


def blend_weights(NX: int, NY: int, patch: int, offsets: list[tuple[int, int]],
                  uniform: bool = False) -> list[np.ndarray]:
    """One `(NX, NY)` weight per tiling, summing to one at every grid point."""
    if uniform:
        return [np.full((NX, NY), 1.0 / len(offsets)) for _ in offsets]
    W = [hann_1d(NX, patch, ox)[:, None] * hann_1d(NY, patch, oy)[None, :] for ox, oy in offsets]
    S = sum(W)
    return [w / S for w in W]


def frame_obs_sites(NX: int, NY: int, patch: int, stride: int, offset: tuple[int, int]):
    """Observed rows/columns of tiling `offset`'s frame and the owning site of every observation,
    site-major (x outer, y inner), in the original observation order `field[::stride, ::stride]`."""
    nsy = NY // patch
    xg, yg = np.arange(0, NX, stride), np.arange(0, NY, stride)
    ix, iy = (xg - offset[0]) % NX, (yg - offset[1]) % NY
    site = ((ix[:, None] // patch) * nsy + (iy[None, :] // patch)).reshape(-1)
    return torch.as_tensor(ix), torch.as_tensor(iy), torch.as_tensor(site, dtype=torch.int64)
