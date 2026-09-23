r"""Optional banded latent-index-smoothness + off-band decorrelation penalty.

Ported 2026-08-29 from a reference implementation
(`/Users/daltonjones/Documents/experiments/ks_latent/regularizer.py`) found
to train with much lower loss / higher reconstruction accuracy than this
project's original recipe; see CLAUDE_CODE_BRIEF.md §5.1 "Ported
improvements" addendum for the full comparison. Off by default
(`RegConfig.lambda_z == 0`) -- this is a new capability, not a change to the
validated canonical recipe.

What this penalizes
--------------------
For a batch of latent vectors `z` (`d_latent`,), the banded term is

    L_band = mean_batch z^T B z / d_latent

with `B` the graph Laplacian of the path graph on latent indices, edge
`(i, i+m)` weighted `decay**(m-1)` for `m = 1..bandwidth`. This is `sum_m a_m
sum_j (z_{j+m} - z_j)^2` -- it penalizes coordinates *near in index* for
being *far apart in value*, which is what gives the flat latent index
physical meaning: adjacency in the index stops being arbitrary. `B` is
symmetric, banded, and positive semi-definite by construction (a Laplacian
is a sum of PSD rank-one terms), so the penalty can never be driven
unbounded below.

The catch: the Laplacian's null space is the constant vectors, so `L_band`
alone is minimized at `z_1 = z_2 = ... = z_d`, i.e. representation collapse
-- exactly the failure mode this codebase already spent significant effort
escaping (see docs/RESULTS.md "Collapse follow-up"). `L_band` can express
"nearby indices should agree" but not "distant indices should stay
different", so `OffBandDecorrelation` supplies the missing half: the mean
squared correlation between coordinate pairs more than `bandwidth` apart,
which collapse maximizes (every such pair reaches correlation 1). The two
terms together have a non-degenerate optimum -- locally smooth in the
index, full rank overall -- that neither has alone. It cannot be folded into
`B` itself: rewarding *difference* at distance requires positive
off-diagonal entries, which makes `B` indefinite and the penalty unbounded
below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ks_latent.config import RegConfig


def band_weights(bandwidth: int, decay: float) -> np.ndarray:
    """`a_m = decay**(m-1)` for `m = 1..bandwidth` (`a_1 = 1`)."""
    if bandwidth < 1:
        raise ValueError(f"bandwidth must be >= 1, got {bandwidth}")
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    m = np.arange(1, bandwidth + 1)
    return decay ** (m - 1.0)


def laplacian_B(d_latent: int, bandwidth: int, decay: float) -> np.ndarray:
    """Graph Laplacian of the banded path graph on `d_latent` indices; PSD
    by construction."""
    if bandwidth >= d_latent:
        raise ValueError(f"bandwidth {bandwidth} must be < d_latent {d_latent}")
    a = band_weights(bandwidth, decay)
    B = np.zeros((d_latent, d_latent))
    for m, am in enumerate(a, start=1):
        for i in range(d_latent - m):
            B[i, i] += am
            B[i + m, i + m] += am
            B[i, i + m] -= am
            B[i + m, i] -= am
    return B


def psd_report(B: np.ndarray) -> dict:
    """Symmetry and eigenvalue diagnostics, for provenance/logging."""
    asym = float(np.abs(B - B.T).max())
    w = np.linalg.eigvalsh((B + B.T) / 2.0)
    return {
        "asymmetry": asym,
        "min_eig": float(w.min()),
        "max_eig": float(w.max()),
        "n_negative": int((w < -1e-10).sum()),
        "rank": int((w > 1e-10).sum()),
    }


class BandedSmoothness(nn.Module):
    """`mean_batch z^T B z / d_latent` for a fixed banded Laplacian `B`.

    Input `Z`: `(batch, d_latent)` or `(batch, window, d_latent)` (the
    window axis, if present, is folded into the batch -- the penalty acts on
    each latent vector independently)."""

    def __init__(self, cfg: RegConfig, d_latent: int):
        super().__init__()
        B = laplacian_B(d_latent, cfg.bandwidth, cfg.decay)
        self.report = psd_report(B)
        self.d_latent = d_latent
        self.register_buffer("B", torch.as_tensor(B, dtype=torch.float32))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        z = Z.reshape(-1, self.d_latent)
        quad = torch.einsum("bj,jk,bk->b", z, self.B, z)
        return quad.mean() / self.d_latent


def offband_mask(d_latent: int, bandwidth: int) -> np.ndarray:
    """Boolean `(d_latent, d_latent)` mask selecting pairs `|i-k| > bandwidth`."""
    idx = np.arange(d_latent)
    sep = np.abs(idx[:, None] - idx[None, :])
    return sep > bandwidth


class OffBandDecorrelation(nn.Module):
    """Mean squared correlation between latent coordinates more than
    `bandwidth` apart in index -- the anti-collapse complement of
    `BandedSmoothness`. Scale-free (correlation, not covariance) so it
    cannot be satisfied by shrinking `z`, only by genuinely decorrelating
    distant coordinates. Input `Z` same shape convention as
    `BandedSmoothness`."""

    def __init__(self, d_latent: int, bandwidth: int, eps: float = 1e-6):
        super().__init__()
        if bandwidth >= d_latent:
            raise ValueError(
                f"bandwidth {bandwidth} leaves no pairs with |i-k| > bandwidth "
                f"at d_latent {d_latent}"
            )
        self.d_latent = d_latent
        self.eps = eps
        self.register_buffer("mask", torch.as_tensor(offband_mask(d_latent, bandwidth)))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        z = Z.reshape(-1, self.d_latent)
        if z.shape[0] < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        zc = z - z.mean(dim=0, keepdim=True)
        sd = zc.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(self.eps)
        zn = zc / sd
        corr = (zn.transpose(0, 1) @ zn) / zn.shape[0]
        return (corr[self.mask] ** 2).mean()


class LatentIndexPenalty(nn.Module):
    """`lambda_z * BandedSmoothness(Z) + lambda_decorr * OffBandDecorrelation(Z)`,
    bundled so every call site applies the same pair with the same weights.
    Either half is skipped (left at its identically-zero default) when its
    weight is `0.0`, so the cost is only paid when the term is actually used.
    """

    def __init__(self, cfg: RegConfig, d_latent: int):
        super().__init__()
        self.cfg = cfg
        self.banded = BandedSmoothness(cfg, d_latent) if cfg.lambda_z != 0.0 else None
        self.decorr = (
            OffBandDecorrelation(d_latent, cfg.bandwidth) if cfg.lambda_decorr != 0.0 else None
        )

    @property
    def active(self) -> bool:
        return self.banded is not None or self.decorr is not None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((), device=Z.device, dtype=Z.dtype)
        b = self.banded(Z) if self.banded is not None else zero
        d = self.decorr(Z) if self.decorr is not None else zero
        return self.cfg.lambda_z * b + self.cfg.lambda_decorr * d
