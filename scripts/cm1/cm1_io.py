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

import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CM1 = ROOT / "cm1data" / "cm1r21.1"
CACHE = ROOT / "artifacts" / "cm1"
MEMBERS = [f"run_e{i}" for i in range(1, 11)]
VARS = ["u", "w", "th", "qv", "prs", "qliq", "qice"]
BASE_I = 1500  # far-field column at t=0 (bubble is at x=200 km, radius 10 km)


def load_run(run: str) -> dict:
    """Return dict with 'X' (T, V, Z, NX) float32 physical-transformed fields,
    plus coordinates and the raw surface reflectivity 'cref' (T, NX)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{run}_cols.npz"
    if f.exists():
        return dict(np.load(f))
    ds = nc.Dataset(CM1 / run / "cm1out.nc")
    ds.set_auto_mask(False)
    g = lambda k: ds[k][:, :, 0, :].astype(np.float64)
    th, qv, prs = g("th"), g("qv"), g("prs")
    base = lambda a: a[0, :, BASE_I][None, :, None]
    qliq = g("qc") + g("qr")
    qice = g("qi") + g("qs") + g("qg")
    X = np.stack(
        [
            g("uinterp") - base(g("uinterp")),
            g("winterp"),
            th - base(th),
            (qv - base(qv)) * 1e3,          # g/kg
            (prs - base(prs)) / 100.0,      # hPa
            np.sqrt(np.clip(qliq, 0, None) * 1e3),
            np.sqrt(np.clip(qice, 0, None) * 1e3),
        ],
        axis=1,
    ).astype(np.float32)
    out = dict(
        X=X,
        t=ds["time"][:].astype(np.float64),
        x=ds["xh"][:].astype(np.float64),
        z=ds["zh"][:].astype(np.float64),
        cref=ds["cref"][:, 0, :].astype(np.float32),
        dbz=ds["dbz"][:, :, 0, :].astype(np.float32),
    )
    ds.close()
    np.savez(f, **out)
    return out
