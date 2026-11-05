"""Load the 2D CM1 squall-line runs in cm1data/ as column states.

Each run is an x-z slice (nx=1600, dx=250 m, nz=62 stretched levels, open
lateral boundaries, grid moving at umove=12.5 m/s). A "column state" at site
i is the stack of VARS over all 62 levels.

Transforms:
  u, th, qv, prs -> perturbation from the far-field t=0 column (x index BASE_I)
  hydrometeors -> q_liq = qc+qr, q_ice = qi+qs+qg, then sqrt(g/kg): bounded,
                  very skewed fields made less skewed; zero stays zero
Normalization is one scalar std per variable (not per level), so quiet upper
levels do not dominate the error.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ks_latent.cm1.io import BASE_I, load_grads_run, transform  # noqa: F401
from ks_latent.cm1.io import load_netcdf_run as _load_netcdf_run

ROOT = Path(__file__).resolve().parents[2]
CM1 = ROOT / "cm1data" / "cm1r21.1"
CACHE = ROOT / "artifacts" / "cm1"
MEMBERS = [f"run_e{i}" for i in range(1, 11)]
VARS = ["u", "w", "th", "qv", "prs", "qliq", "qice"]


def load_run(run: str) -> dict:
    """Return dict with 'X' (T, V, Z, NX) float32 physical-transformed fields,
    plus coordinates and the raw surface reflectivity 'cref' (T, NX)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{run}_cols.npz"
    if f.exists():
        return dict(np.load(f))
    out = _load_netcdf_run(CM1 / run / "cm1out.nc")
    np.savez(f, **out)
    return out



