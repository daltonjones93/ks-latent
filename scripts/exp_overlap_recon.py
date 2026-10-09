"""Overlapping tilings without retraining (2026-10-09): reconstruction floor of the per-patch 2D KS
autoencoder when the field is encoded/decoded on K tilings shifted by a fraction of a patch and the
decodings are blended with a partition of unity that vanishes at each tiling's patch edges.

Blend weights: for tiling k with offset (ox, oy), w_k(x, y) = b(x - ox) b(y - oy) with
b(t) = sin^2(pi (t mod p + 0.5) / p) (a Hann window over each patch, p = 16 grid points). With the
two offsets 0 and p/2 per axis, sin^2 + cos^2 = 1, so the four 2D weights sum to 1 exactly.
Usage: python scripts/exp_overlap_recon.py [ae_tag]
"""
import sys

import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.models import load_autoencoder_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d

tag = sys.argv[1] if len(sys.argv) > 1 else "p7b_ks2d"
p = f"artifacts/stage1_ae_kolmogorov_crop_{tag}.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
g = deployment_grid_2d(44.0, training_dx(ck["train_dataset"]), patch)
NX = g.NX
ae = load_autoencoder_checkpoint_resized_2d(p, new_n_sites_x=g.n_sites, new_n_sites_y=g.n_sites, new_NX=NX, new_NY=NX)[0].eval()
m, s = float(ck["normalization_mean"]), float(ck["normalization_std"])
u0 = np.load("artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed0.npy")
cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=NX, NY=NX, dt=0.02, snapshot_every=125, spinup_time=200.0, seed=0)
U = integrate2d(u0, cfg, n_steps=20 * 125)[1:]
Us = (U - m) / s

def recon(V, ox, oy):
    """Encode/decode with the tiling shifted by (ox, oy) grid points; return in the original frame."""
    W = np.roll(V, (-ox, -oy), axis=(1, 2))
    with torch.no_grad():
        R = ae.decode(ae.encode(torch.tensor(W.reshape(len(W), -1), dtype=torch.float32))).numpy().reshape(W.shape)
    return np.roll(R, (ox, oy), axis=(1, 2))

def hann1d(off):
    t = (np.arange(NX) - off) % patch
    return np.sin(np.pi * (t + 0.5) / patch) ** 2

k = 2 * np.pi * np.fft.fftfreq(NX, d=44 / NX); K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
BANDS = [(0, 0.6), (0.6, 1.14), (1.14, 1.6), (1.6, 99)]
def report(name, R):
    E = (R - Us) * s
    P = np.mean(np.abs(np.fft.fft2(E)) ** 2, axis=0) / NX ** 4
    b = [P[(K >= lo) & (K < hi)].sum() for lo, hi in BANDS]
    print(f"{name:42s} RMSE {np.sqrt(np.mean(E ** 2)):.3f} | bands " + " ".join(f"{x:.3f}" for x in b))

h = patch // 2
R00 = recon(Us, 0, 0); report("1 tiling (current)", R00)
tilings = {(0, 0): R00, (h, 0): recon(Us, h, 0), (0, h): recon(Us, 0, h), (h, h): recon(Us, h, h)}
report("4 tilings, plain average", sum(tilings.values()) / 4)
Wsum = 0; Rb = 0
for (ox, oy), R in tilings.items():
    w = hann1d(ox)[:, None] * hann1d(oy)[None, :]
    Rb = Rb + w * R; Wsum = Wsum + w
print("   (Hann weights sum to 1: max dev %.1e)" % np.abs(Wsum - 1).max())
report("4 tilings, Hann partition of unity", Rb / Wsum)
q = patch // 4
t16 = {(ox, oy): recon(Us, ox, oy) for ox in range(0, patch, q) for oy in range(0, patch, q)}
report("16 tilings (quarter-patch), plain average", sum(t16.values()) / 16)
