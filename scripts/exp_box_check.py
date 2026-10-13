"""Truth-side box-size check (Block Z9, 2026-10-13): is a small periodic box a representative training domain?

Runs the true PDE (no model) in the 1x box and a box twice as large, several ICs, and compares the time-averaged
spectral density per mode, Phi(k) = <|u_k|^2> / (NX NY)^2 * A / (2 pi)^2 (A = box area; kinetic energy
|w_k|^2 / k^2 for Kolmogorov vorticity), in shells of width k1/2 (k1 = 2 pi / L_1x, the 1x box's lowest
wavenumber). A box is representative when the larger box's extra, longer-wavelength modes (|k| = k1/2) carry the
same density as the 1x box's longest modes (|k| = k1) -- a flat low-k spectrum, as in KS -- rather than more
(energy piling up at scales the small box cannot hold, as in an arrested inverse cascade). The band share used in
Z8 (energy fraction below a fixed |k|) mixes this density with the number of modes, which grows with the box.

Reports per box: Phi by shell, the share below 2.5 k1 (Z8's band, roughly), and the ratios
  R_new  = Phi_2x(k1/2) / Phi_1x(k1)   (density at the new scale vs the small box's largest scale)
  R_same = Phi_2x(k1)   / Phi_1x(k1)   (the same scale in both boxes)
Usage: python scripts/exp_box_check.py --system kolm --alpha 0.05 --seeds 0,1,2
       python scripts/exp_box_check.py --system ks2d --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ks_latent.config import KolmogorovConfig, KSConfig2D
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov
from ks_latent.solver.ks2d import integrate2d, spinup2d

OUT = Path("artifacts/experiments/box_check")


def shell_density(w: np.ndarray, L: float, k1: float, kinetic: bool) -> dict:
    T, NX, NY = w.shape
    k = np.fft.fftfreq(NX, d=L / NX) * 2 * np.pi
    K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    P = np.abs(np.fft.fft2(w)) ** 2 / (NX * NY) ** 2
    if kinetic:
        P = P / np.where(K > 0, K ** 2, 1.0)
    P = P.mean(axis=0)
    P[0, 0] = 0.0
    b = np.round(2 * K / k1) / 2  # shell centre in units of k1, step 1/2
    out = {}
    for s in np.unique(b):
        if s == 0 or s > 8:
            continue
        m = b == s
        out[f"{s:g}"] = float(P[m].mean() * L * L / (2 * np.pi) ** 2)
    out["share_lt_2.5k1"] = float(P[(K > 0) & (K < 2.5 * k1)].sum() / P.sum())
    out["variance"] = float(P.sum()) if not kinetic else float((np.abs(np.fft.fft2(w)) ** 2).mean(axis=0)[1:].sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["kolm", "ks2d"], required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--scales", default="1,2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--T", type=float, default=100.0, help="averaging window after spin-up")
    a = ap.parse_args()
    L1 = 2 * math.pi if a.system == "kolm" else 44.0
    k1 = 2 * math.pi / L1
    tag = f"kolm_a{a.alpha:g}" if a.system == "kolm" else "ks2d"
    res = {"system": a.system, "alpha": a.alpha if a.system == "kolm" else None, "k1": k1, "T": a.T, "runs": {}}
    for S in [int(s) for s in a.scales.split(",")]:
        for seed in [int(s) for s in a.seeds.split(",")]:
            t0 = time.time()
            L, N = L1 * S, 128 * S
            if a.system == "kolm":
                cfg = KolmogorovConfig(Lx=L, Ly=L, NX=N, NY=N, alpha=a.alpha, spinup_time=100.0, seed=seed)
                u0 = spinup_kolmogorov(cfg, np.random.default_rng(seed))
                every = int(round(0.5 / cfg.dt))
                w = integrate_kolmogorov(u0, cfg, n_steps=int(round(a.T / cfg.dt)), snapshot_every=every)[1:]
            else:
                cfg = KSConfig2D(Lx=L, Ly=L, NX=N, NY=N, spinup_time=200.0, seed=seed)
                cache = Path(f"artifacts/cache/ks2d_spinup_L{L:g}_NX{N}_dt0.02_T200_seed{seed}.npy")
                u0 = np.load(cache) if cache.exists() else spinup2d(cfg, np.random.default_rng(seed))
                every = int(round(0.5 / cfg.dt))
                w = integrate2d(u0, cfg, n_steps=int(round(a.T / cfg.dt)), snapshot_every=every)[1:]
            r = shell_density(w, L, k1, kinetic=a.system == "kolm")
            r["wall_s"] = time.time() - t0
            res["runs"][f"{S}x_seed{seed}"] = r
            print(f"[box] {tag} {S}x seed{seed}: Phi(0.5)={r.get('0.5', float('nan')):.4g} Phi(1)={r['1']:.4g} "
                  f"Phi(1.5)={r.get('1.5', float('nan')):.4g} Phi(2)={r['2']:.4g} share<2.5k1={r['share_lt_2.5k1']:.4f} "
                  f"[{r['wall_s']:.0f}s]", flush=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    phi = lambda S, s: [res["runs"][f"{S}x_seed{q}"][s] for q in seeds]
    if "2x_seed%d" % seeds[0] in res["runs"]:
        p1 = np.mean(phi(1, "1"))
        res["R_new"] = [x / p1 for x in phi(2, "0.5")]
        res["R_same"] = [x / p1 for x in phi(2, "1")]
        res["share_1x"], res["share_2x"] = phi(1, "share_lt_2.5k1"), phi(2, "share_lt_2.5k1")
        print(f"[box] {tag} R_new (Phi_2x(k1/2)/Phi_1x(k1)) per IC {np.round(res['R_new'], 3).tolist()} mean "
              f"{np.mean(res['R_new']):.3f}; R_same {np.round(res['R_same'], 3).tolist()} mean {np.mean(res['R_same']):.3f}; "
              f"share<2.5k1 1x {np.mean(res['share_1x']):.4f} 2x {np.mean(res['share_2x']):.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"box_check_{tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
