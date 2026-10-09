"""Task A2 (docs/full_paper_tasks.md): float32 vs float64 tangent propagation for one
model Lyapunov spectrum per system, everything else identical to the paper's evaluators
(exp_crop_eval.py, exp_crop_eval_ks2d.py, eval_1d_transfer.py): IC seed 0, standardized IC,
single_state, n_directions = d_latent, 2000 steps, QR every 10, warm-up 200.

Usage: python scripts/lyap_dtype_control.py {ks1d,ks2d,kolmogorov} [IC_SEED] [DTYPES]
DTYPES: comma list, default "float32,float64".
Writes artifacts/experiments/lyap_dtype_control/<system>[_ic<seed>].json (all spectra).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig, KSConfig2D
from ks_latent.models import (
    load_autoencoder_checkpoint_resized,
    load_autoencoder_checkpoint_resized_2d,
    load_propagator_checkpoint_resized,
    load_propagator_checkpoint_resized_2d,
)
from ks_latent.solver.kolmogorov import spinup_kolmogorov
from ks_latent.solver.ks import KSConfig, spinup
from ks_latent.solver.ks2d import spinup2d

OUT = Path("artifacts/experiments/lyap_dtype_control")
system = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dtypes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["float32", "float64"]

if system == "ks1d":
    ae_path = "artifacts/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt"
    prop_path = "artifacts/stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt"
    with h5py.File("artifacts/datasets/stage1_trajectories_dtsnap1.h5", "r") as f:
        mean = float(f["metadata"].attrs["normalization_mean"]); std = float(f["metadata"].attrs["normalization_std"])
    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized(ae_path, new_n_sites=16, new_NX=256)
    d = ae_cfg.n_sites * ae_cfg.local_channels
    prop, _, _ = load_propagator_checkpoint_resized(prop_path, new_d_latent=d, new_n_tokens=16)
    cfg = KSConfig(L=100.0, NX=256, dt=0.05, snapshot_every=20, spinup_time=500.0, seed=seed)
    from ks_latent.solver.ks import integrate
    u = integrate(spinup(cfg, np.random.default_rng(seed)), cfg, n_steps=cfg.snapshot_every)
    with torch.no_grad():
        z01 = ae.encode(torch.tensor((u[0:2] - mean) / std, dtype=torch.float32)).numpy()
    pair, dt_snap, desc = z01, 1.0, "1D KS, Section-224 model, L=100"
else:
    if system == "ks2d":
        ae_path = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"
        prop_path = "artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt"
        nsx = nsy = 8; NX = NY = 128
        cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=NX, NY=NY, spinup_time=200.0, seed=seed)
        u0_raw = spinup2d(cfg, np.random.default_rng(seed)); dt_snap = cfg.dt * cfg.snapshot_every
        desc = "2D KS, p7b_ks2d, 44^2"
    elif system == "kolmogorov":
        ae_path = "artifacts/stage1_ae_kolmogorov_crop_c19_e0d0_c6.pt"
        prop_path = "artifacts/stage2_prop_kolmogorov_crop_c19_e0d0_c6.pt"
        ck0 = torch.load(ae_path, map_location="cpu", weights_only=False)
        with h5py.File(ck0["train_dataset"], "r") as f:
            md = dict(f["metadata"].attrs)
        nsx, nsy, NX, NY = 8, 8, 128, 128
        cfg = KolmogorovConfig(Lx=2 * math.pi, Ly=2 * math.pi, NX=NX, NY=NY, nu=float(md["nu"]), alpha=float(md["alpha"]),
                               forcing_n=int(md["forcing_n"]), dt=float(md["dt"]), spinup_time=100.0, seed=seed)
        u0_raw = spinup_kolmogorov(cfg, np.random.default_rng(seed)); dt_snap = float(md["dt_snap"])
        desc = f"Kolmogorov alpha={md['alpha']}, c19_e0d0_c6, 1x"
    else:
        raise SystemExit(f"unknown system {system}")
    ck = torch.load(ae_path, map_location="cpu", weights_only=False)
    mean, std = ck["normalization_mean"], ck["normalization_std"]
    ae = load_autoencoder_checkpoint_resized_2d(ae_path, new_n_sites_x=nsx, new_n_sites_y=nsy, new_NX=NX, new_NY=NY)[0].eval()
    prop, pcfg, _ = load_propagator_checkpoint_resized_2d(prop_path, new_n_sites_x=nsx, new_n_sites_y=nsy)
    d = pcfg.d_latent
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(((u0_raw - mean) / std).reshape(1, -1), dtype=torch.float32))[0].numpy()
    pair = np.stack([z0, z0]).astype(np.float64)

prop.eval()
OUT.mkdir(parents=True, exist_ok=True)
rec = {"system": system, "description": desc, "d_latent": d, "ic_seed": seed, "ae_checkpoint": ae_path,
       "prop_checkpoint": prop_path, "dt_snap": dt_snap, "runs": {}}
for name in dtypes:
    dtype = {"float32": torch.float32, "float64": torch.float64}[name]
    t0 = time.time()
    res = lyapunov_spectrum_latent_propagator(
        prop, pair, mode="single_state", n_directions=d, n_steps=2000, qr_every=10, dt_snap=dt_snap,
        warmup_steps=200, seed=seed, max_abs_state=1e4 if system == "ks1d" else 1e3, tangent_dtype=dtype,
    )
    rec["runs"][name] = {"D_KY": res.kaplan_yorke_dimension, "lambda1": float(res.exponents[0]),
                         "n_positive": int(res.n_positive), "exponents": [float(x) for x in res.exponents],
                         "wall_s": time.time() - t0}
    print(system, name, rec["runs"][name]["D_KY"], rec["runs"][name]["n_positive"], rec["runs"][name]["lambda1"],
          f"{rec['runs'][name]['wall_s']:.0f}s", flush=True)
if {"float32", "float64"} <= set(rec["runs"]):
    r32, r64 = rec["runs"]["float32"], rec["runs"]["float64"]
    rec["rel_diff_D_KY"] = abs(r64["D_KY"] - r32["D_KY"]) / r64["D_KY"] if r64["D_KY"] else None
    rec["n_positive_equal"] = r32["n_positive"] == r64["n_positive"]
    print("rel diff D_KY:", rec["rel_diff_D_KY"], "n_pos equal:", rec["n_positive_equal"], flush=True)
name = system if seed == 0 else f"{system}_ic{seed}"
if len(sys.argv) > 3 and sys.argv[3] == "float32":
    name += "_float32"  # single-dtype float32 run: keep it apart from the float64 file for the same IC
(OUT / f"{name}.json").write_text(json.dumps(rec, indent=2))
