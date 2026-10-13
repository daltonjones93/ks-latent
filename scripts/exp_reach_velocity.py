"""Truth-side reach test (Block Z10, 2026-10-13): how much of the advecting velocity is set by vorticity within a
radius R?

Kolmogorov flow is integrated in vorticity form; the velocity that advects the vorticity is u = K * omega
(Biot-Savart, K ~ 1/r in 2D), so it depends on vorticity arbitrarily far away. A local latent model with reach R
sees only vorticity within R. For true fields at 1x/2x/3x box sizes we rebuild the velocity from vorticity within R,
u_R = (K 1{|r| < R}) * omega, with K the periodic Biot-Savart kernel of that box (inverse FFT of (i k_y, -i k_x)/k^2,
min-image distance), and report the unexplained fraction F(R) = <|u - u_R|^2> / <|u|^2> and its absolute size.

Prediction (the vorticity-nonlocality reading of the alpha = 0.05 small-box failure): at fixed R (model reach,
~2.4-2.8 physical units; at most pi = half the 1x box for any small-box model), F grows with box size at
alpha = 0.05 and 0.025, and not (or much less) at alpha = 0.1. Larger R in a larger box shows how far a model would
need to see (the "larger windows" question; only route (b), window training, can provide R > pi).
Usage: python scripts/exp_reach_velocity.py --alpha 0.05 [--scales 1,2,3 --seeds 0,1 --T 40]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ks_latent.config import KolmogorovConfig
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov

OUT = Path("artifacts/experiments/reach_velocity")
SITE = 2 * math.pi / 8  # one latent site (16 grid points at dx = 2 pi / 128)
RADII = [1.0, 2.0, 3.0, 3.5, 4.0, 6.0, 8.0, 12.0]  # in sites; 4 sites = pi = half the 1x box


def kernels(N: int, L: float):
    k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    hx, hy = 1j * KY / K2, -1j * KX / K2  # u_hat = hx * w_hat, v_hat = hy * w_hat (sign irrelevant to F)
    hx[0, 0] = hy[0, 0] = 0.0
    x = np.arange(N) * L / N
    d1 = np.minimum(x, L - x)
    dist = np.sqrt(d1[:, None] ** 2 + d1[None, :] ** 2)
    return hx, hy, np.real(np.fft.ifft2(hx)), np.real(np.fft.ifft2(hy)), dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--scales", default="1,2,3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args()
    res = {"alpha": a.alpha, "site": SITE, "radii_sites": RADII, "T": a.T, "runs": {}}
    for S in [int(s) for s in a.scales.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            L, N = 2 * math.pi * S, 128 * S
            cfg = KolmogorovConfig(Lx=L, Ly=L, NX=N, NY=N, alpha=a.alpha, spinup_time=100.0, seed=seed)
            u0 = spinup_kolmogorov(cfg, np.random.default_rng(seed))
            w = integrate_kolmogorov(u0, cfg, n_steps=int(round(a.T / cfg.dt)),
                                     snapshot_every=int(round(a.every / cfg.dt)))[1:]
            hx, hy, kx, ky, dist = kernels(N, L)
            W = np.fft.fft2(w)
            u, v = np.real(np.fft.ifft2(W * hx)), np.real(np.fft.ifft2(W * hy))
            tot = float((u ** 2 + v ** 2).mean())
            row = {"u_var": tot, "F": {}, "resid_var": {}}
            for r in RADII:
                R = r * SITE
                if R > dist.max():
                    continue
                m = dist < R
                ur = np.real(np.fft.ifft2(W * np.fft.fft2(kx * m)))
                vr = np.real(np.fft.ifft2(W * np.fft.fft2(ky * m)))
                e = float(((u - ur) ** 2 + (v - vr) ** 2).mean())
                row["F"][f"{r:g}"], row["resid_var"][f"{r:g}"] = e / tot, e
            row["wall_s"] = time.time() - t0
            res["runs"][f"{S}x_seed{seed}"] = row
            print(f"[reach] a{a.alpha:g} {S}x seed{seed} |u|^2 {tot:.4f} F(R sites) "
                  + " ".join(f"{k}:{v:.3f}" for k, v in row["F"].items()) + f" [{row['wall_s']:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reach_velocity_a{a.alpha:g}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
