"""Accuracy study (2026-10-08): reconstruction error of 2D KS autoencoders by wavenumber band at
44^2 (deployment grid, 20 truth snapshots 2.5 time units apart from the cached spin-up, IC seed 0).
Usage: python scripts/exp_recon_bands.py p7b_ks2d p7g10_ks2d ...  (tags of stage1 checkpoints)"""
import sys
from pathlib import Path

import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.models import load_autoencoder_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d

BANDS = [(0, 0.6), (0.6, 1.14), (1.14, 1.6), (1.6, 99)]
u0 = np.load("artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed0.npy")
cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=128, NY=128, dt=0.02, snapshot_every=125, spinup_time=200.0, seed=0)
U = integrate2d(u0, cfg, n_steps=20 * 125)[1:]  # (20, 128, 128) physical
k = 2 * np.pi * np.fft.fftfreq(128, d=44 / 128); K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
print("tag            total RMSE | error variance by |k| band " + " ".join(f"[{a},{b})" for a, b in BANDS))
for tag in sys.argv[1:]:
    p = f"artifacts/stage1_ae_kolmogorov_crop_{tag}.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(44.0, training_dx(ck["train_dataset"]), patch)
    ae = load_autoencoder_checkpoint_resized_2d(p, new_n_sites_x=g.n_sites, new_n_sites_y=g.n_sites,
                                                new_NX=g.NX, new_NY=g.NX)[0].eval()
    m, s = float(ck["normalization_mean"]), float(ck["normalization_std"])
    with torch.no_grad():
        R = ae.decode(ae.encode(torch.tensor((U.reshape(20, -1) - m) / s, dtype=torch.float32))).numpy()
    E = R.reshape(U.shape) * s + m - U
    P = np.mean(np.abs(np.fft.fft2(E)) ** 2, axis=0) / 128 ** 4
    bands = [P[(K >= a) & (K < b)].sum() for a, b in BANDS]
    print(f"{tag:14s} {np.sqrt(np.mean(E ** 2)):.3f}      | " + " ".join(f"{x:.3f}".ljust(10) for x in bands))
