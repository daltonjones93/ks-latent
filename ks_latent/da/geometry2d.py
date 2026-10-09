"""Deployment grid for a trained 2D local-field model.

A weight-shared local model is only meaningful at the grid spacing it was
trained on: each site covers `patch` grid points, so a different spacing
changes the physical size of a site. The deployment grid is therefore
derived from the training dataset's spacing, never from the checkpoint's
stored crop size (which is a training-time window, not a domain).
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py


@dataclass(frozen=True)
class DeploymentGrid2D:
    L: float          # physical side of the square periodic domain
    dx: float         # grid spacing (equals the training spacing)
    NX: int           # grid points per side
    n_sites: int      # sites per side
    patch: int        # grid points per site per side
    site_length: float


def training_dx(dataset_path: str) -> float:
    """Grid spacing of the training dataset; requires isotropic spacing."""
    with h5py.File(dataset_path, "r") as f:
        md = dict(f["metadata"].attrs)
    dx, dy = float(md["Lx"]) / int(md["NX"]), float(md["Ly"]) / int(md["NY"])
    if abs(dx - dy) > 1e-12 * max(dx, dy):
        raise ValueError(f"anisotropic training grid: dx={dx}, dy={dy}")
    return dx


def deployment_grid_2d(L: float, train_dx: float, patch: int) -> DeploymentGrid2D:
    """Square periodic domain of side `L` at the training spacing.

    Raises if `L` is not a whole number of sites, since the model cannot be
    placed on such a domain without changing the site size."""
    site_length = train_dx * patch
    n_sites_f = L / site_length
    n_sites = int(round(n_sites_f))
    if n_sites < 1 or abs(n_sites_f - n_sites) > 1e-9 * max(1.0, n_sites_f):
        raise ValueError(
            f"L={L} is not a whole number of sites of length {site_length} "
            f"(train_dx={train_dx}, patch={patch}); got {n_sites_f} sites"
        )
    return DeploymentGrid2D(L=L, dx=train_dx, NX=n_sites * patch, n_sites=n_sites,
                            patch=patch, site_length=site_length)
