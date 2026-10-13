"""Lorenz-96 transfer evaluation of a window-trained per-site local latent model (Z16).

For each ring size N the script (i) integrates the TRUE system (RK4 dt 0.01, snapshots every 0.1) from
spun-up initial conditions and computes its full Lyapunov spectrum (Benettin with the exact sparse
tangent, QR every snapshot); (ii) rolls the model out from the encoded true initial state and reports
the forecast error as a function of lead time (valid time in Lyapunov times), long-run statistics
(mean, std, energy spectrum, one-point PDF moments); and (iii) computes the model's Lyapunov spectrum
with the latent-map Benettin (single state, float32 tangent). One JSON per (model, N).

Usage: python scripts/eval_l96_transfer.py --tag l96_crop_c3 --N 48 64 128 256 [--n-ic 8]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import Lorenz96Config
from ks_latent.models import load_autoencoder_checkpoint_resized, load_propagator_checkpoint_resized
from ks_latent.solver.lorenz96 import integrate, l96_rhs, spinup

OUT = Path("artifacts/experiments/l96")
F, DT_SNAP = 8.0, 0.1


def tangent_rhs(x, dX, F):
    """J(x) dX for dX (k, N): df_i = x_{i-1}(dX_{i+1} - dX_{i-2}) + (x_{i+1} - x_{i-2}) dX_{i-1} - dX_i."""
    xm1, xp1, xm2 = np.roll(x, 1), np.roll(x, -1), np.roll(x, 2)
    return (xm1 * (np.roll(dX, -1, axis=-1) - np.roll(dX, 2, axis=-1))
            + (xp1 - xm2) * np.roll(dX, 1, axis=-1) - dX)


def true_lyapunov(x0, n_snap, warmup, k, dt=0.01, every=10, seed=0):
    N = x0.shape[0]
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((N, k)))[0].T  # (k, N)
    x = x0.copy()
    sums = np.zeros(k)
    cnt = 0
    for s in range(n_snap):
        for _ in range(every):
            def d(st, tg):
                return l96_rhs(st, F), tangent_rhs(st, tg, F)
            k1s, k1t = d(x, Q)
            k2s, k2t = d(x + 0.5 * dt * k1s, Q + 0.5 * dt * k1t)
            k3s, k3t = d(x + 0.5 * dt * k2s, Q + 0.5 * dt * k2t)
            k4s, k4t = d(x + dt * k3s, Q + dt * k3t)
            x = x + dt / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
            Q = Q + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        q, r = np.linalg.qr(Q.T)
        Q = q.T
        if s >= warmup:
            sums += np.log(np.abs(np.diag(r)))
            cnt += 1
    return sums / (cnt * DT_SNAP)


def ky_dim(lam):
    c = np.cumsum(lam)
    j = np.where(c >= 0)[0]
    if len(j) == 0:
        return 0.0
    j = j[-1]
    return float(j + 1 + (c[j] / abs(lam[j + 1]) if j + 1 < len(lam) else 0.0))


def energy_spectrum(x):  # (T, N)
    return (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(0) / x.shape[1] ** 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--N", type=int, nargs="+", required=True)
    p.add_argument("--n-ic", type=int, default=8)
    p.add_argument("--n-roll", type=int, default=600)
    p.add_argument("--n-lyap", type=int, default=3000)
    p.add_argument("--ic-seed", type=int, default=0)
    p.add_argument("--no-model-lyap", action="store_true")
    a = p.parse_args()
    ae_path = f"artifacts/stage1_ae_patched_full_{a.tag}.pt"
    prop_path = f"artifacts/stage2_prop_patched_full_{a.tag}.pt"
    ck = torch.load(ae_path, map_location="cpu", weights_only=False)
    mean, std = ck["normalization_mean"], ck["normalization_std"]
    assert ck["l96_F"] == F and abs(ck["l96_dt_snap"] - DT_SNAP) < 1e-12
    OUT.mkdir(parents=True, exist_ok=True)
    for N in a.N:
        t0 = time.time()
        cfg = Lorenz96Config(N=N, F=F, dt=0.01, snapshot_every=10, spinup_time=50.0)
        ics = [spinup(cfg, np.random.default_rng(1000 * a.ic_seed + i)) for i in range(a.n_ic)]
        truth = np.stack([integrate(x, cfg, a.n_roll * 10) for x in ics])  # (B, T+1, N)
        # truth Lyapunov spectrum (first IC)
        k = N
        lam = true_lyapunov(ics[0], a.n_lyap, 300, k)
        dky_true = ky_dim(lam)
        # model
        patch = ck["ae_config"].NX // ck["ae_config"].n_sites
        assert N % patch == 0
        ae, ae_cfg, _ = load_autoencoder_checkpoint_resized(ae_path, new_n_sites=N // patch, new_NX=N)
        d = ae_cfg.n_sites * ae_cfg.local_channels
        prop, _, _ = load_propagator_checkpoint_resized(prop_path, new_d_latent=d, new_n_tokens=N // patch)
        ae.eval(); prop.eval()
        with torch.no_grad():
            xn = torch.tensor((truth - mean) / std, dtype=torch.float32)  # (B,T+1,N)
            z0 = ae.encode(xn[:, 0])
            zr = prop.rollout(z0, z0, k=a.n_roll)  # (B,k,d)
            zall = torch.cat([z0[:, None], zr], 1)
            B = zall.shape[0]
            xm = ae.decode(zall.reshape(B * (a.n_roll + 1), d)).reshape(B, a.n_roll + 1, N).numpy() * std + mean
            xrec = ae.decode(ae.encode(xn[:, 0])).numpy() * std + mean
        finite = bool(np.isfinite(xm).all())
        rec_rmse = float(np.sqrt(np.mean((xrec - truth[:, 0]) ** 2)))
        # forecast error vs lead time, normalized by the climatological std
        sig = truth[:, 100:].std()
        err = np.sqrt(((xm - truth) ** 2).mean(-1)) / sig  # (B,T+1)
        lt = lam[0] * DT_SNAP  # per-step growth of the leading exponent
        thr = 0.5
        valid = []
        for b in range(B):
            idx = np.where(err[b] > thr)[0]
            valid.append(float((idx[0] if len(idx) else a.n_roll) * DT_SNAP * lam[0]))
        half = slice(a.n_roll // 2, None)
        rec = {"tag": a.tag, "N": N, "d_latent": d, "n_ic": a.n_ic, "finite": finite,
               "recon_rmse_phys": rec_rmse, "lambda1_true": float(lam[0]), "D_KY_true": dky_true,
               "n_pos_true": int((lam > 0).sum()), "truth_std": float(truth[:, 100:].std()),
               "truth_mean": float(truth[:, 100:].mean()),
               "model_mean": float(xm[:, half].mean()) if finite else None,
               "model_std": float(xm[:, half].std()) if finite else None,
               "err_by_lead": {str(i): float(err[:, i].mean()) for i in (1, 5, 10, 20, 30, 50)},
               "valid_time_lt_mean": float(np.mean(valid)), "valid_time_lt_each": valid,
               "spec_true": energy_spectrum(truth[:, 300:].reshape(-1, N)).tolist(),
               "spec_model": energy_spectrum(xm[:, half].reshape(-1, N)).tolist() if finite else None,
               "lyap_true": [float(v) for v in lam]}
        if not a.no_model_lyap and finite:
            nd = min(d, N + 8)
            try:
                with torch.no_grad():
                    z01 = ae.encode(xn[0, 100:102]).numpy()
                res = lyapunov_spectrum_latent_propagator(prop, z01, mode="single_state", n_directions=nd,
                                                          n_steps=2000, qr_every=10, dt_snap=DT_SNAP,
                                                          warmup_steps=200, seed=0, max_abs_state=1e4)
                ex = res.exponents
                rec.update(D_KY_model=float(res.kaplan_yorke_dimension), lambda1_model=float(ex[0]),
                           n_pos_model=int(res.n_positive), lyap_model=[float(v) for v in ex])
                rec["D_KY_err_pct"] = 100 * (rec["D_KY_model"] / dky_true - 1)
            except (RuntimeError, ValueError) as e:
                rec["lyap_error"] = f"{type(e).__name__}: {e}"
        rec["wall_s"] = time.time() - t0
        (OUT / f"{a.tag}_N{N}_ic{a.ic_seed}.json").write_text(json.dumps(rec, indent=1))
        print(f"[{a.tag} N={N}] truth lam1 {lam[0]:.3f} D_KY {dky_true:.1f} (n+ {rec['n_pos_true']}) | model "
              f"D_KY {rec.get('D_KY_model')} lam1 {rec.get('lambda1_model')} n+ {rec.get('n_pos_model')} | "
              f"valid {rec['valid_time_lt_mean']:.2f} LT, std {rec['model_std']} vs {rec['truth_std']:.3f}, "
              f"rec rmse {rec_rmse:.4f} ({rec['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
