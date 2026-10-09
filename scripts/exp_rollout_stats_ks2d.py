"""Task A9 (docs/full_paper_tasks.md): free-rollout statistics of a crop-trained 2D KS model,
the 2D-KS port of scripts/exp_rollout_stats_kolmogorov.py.

The diagnostic is meant to be TRUTH-FREE (usable for model selection without Lyapunov truths):
the model's long free rollout is compared with the TRAINING DATA (crops' parent domain).
For validation it also reports the true PDE from the same initial condition.

Per box size (44*scale)^2 and IC seed:
  * field std (physical units) in the early and late halves of the rollout (drift check);
  * energy shares in |k| bands [0,0.35), [0.35,0.7), [0.7,1.05), [1.05,1.4), [1.4,inf)
    (most unstable |k| = 1/sqrt(2));
  * mean |z| per latent channel, early vs late.
Reference statistics of the training data (wide 264^2 domain, every 10th snapshot of the 4
training trajectories) are computed once and stored in the same JSON.

Usage: python scripts/exp_rollout_stats_ks2d.py --tag p7b_ks2d --scales 1,1.5,2 --seeds 0,1
Writes artifacts/experiments/rollout_stats/<tag>_ks2d_n<steps>.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d

OUT = Path("artifacts/experiments/rollout_stats")
EDGES = [0, 0.35, 0.7, 1.05, 1.4, np.inf]


def band_shares(u: np.ndarray, L: float) -> list[float]:
    """u: (T, N, N) physical field. Time-averaged energy share per |k| band (k > 0)."""
    n = u.shape[1]
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    k2 = 2 * np.pi * np.fft.rfftfreq(n, d=L / n)
    K = np.sqrt(k1[:, None] ** 2 + k2[None, :] ** 2)
    w = np.full(K.shape, 2.0); w[:, 0] = 1.0
    if n % 2 == 0:
        w[:, -1] = 1.0
    p = np.zeros(K.shape)
    for t in range(u.shape[0]):
        p += np.abs(np.fft.rfft2(u[t])) ** 2
    p *= w
    p[K == 0] = 0.0
    tot = p.sum()
    return [float(p[(K >= lo) & (K < hi)].sum() / tot) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def stats(u: np.ndarray, L: float) -> dict:
    h = u.shape[0] // 2
    return {"std_early": float(u[:h].std()), "std_late": float(u[h:].std()),
            "max_abs": float(np.abs(u).max()),
            "band_shares_early": band_shares(u[:h], L), "band_shares_late": band_shares(u[h:], L)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p7b_ks2d")
    ap.add_argument("--scales", default="1,1.5,2")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n-steps", type=int, default=1000, help="model steps (dt_snap = 0.1 each)")
    ap.add_argument("--discard", type=int, default=100)
    ap.add_argument("--no-truth", action="store_true")
    a = ap.parse_args()
    ae_p, pr_p = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt", f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(ae_p, map_location="cpu", weights_only=False)
    mean, std, ds = float(ck["normalization_mean"]), float(ck["normalization_std"]), ck["train_dataset"]

    # Truth-free reference: the training data themselves (raw, physical units).
    with h5py.File(ds, "r") as f:
        md = dict(f["metadata"].attrs)
        n_train = int(md.get("n_train", 4))
        Lw = float(md.get("Lx", 264.0))
        traj = f["trajectories_raw"] if "trajectories_raw" in f else f["trajectories"]
        raw = "trajectories_raw" in f
        snaps = np.concatenate([np.asarray(traj[i, ::10]) for i in range(n_train)], 0).astype(np.float64)
    nw = int(round(np.sqrt(snaps.shape[-1]))) if snaps.ndim == 2 else snaps.shape[-1]
    snaps = snaps.reshape(-1, nw, nw)
    if not raw:
        snaps = snaps * std + mean
    ref = {"dataset": ds, "L": Lw, "n_snapshots": int(snaps.shape[0]), "std": float(snaps.std()),
           "band_shares": band_shares(snaps, Lw)}
    print(f"[ref] training data std {ref['std']:.3f} bands {[round(x, 3) for x in ref['band_shares']]}", flush=True)

    res = {"tag": a.tag, "ae_checkpoint": ae_p, "prop_checkpoint": pr_p, "n_steps": a.n_steps,
           "discard": a.discard, "dt_snap": 0.1, "band_edges": [e if np.isfinite(e) else None for e in EDGES],
           "training_data_reference": ref, "runs": {},
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    for s in [float(x) for x in a.scales.split(",")]:
        L = 44.0 * s; n = int(round(128 * s)); ns = int(round(8 * s))
        ae = load_autoencoder_checkpoint_resized_2d(ae_p, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=n, new_NY=n)[0].eval()
        prop = load_propagator_checkpoint_resized_2d(pr_p, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
        c = ae.cfg.local_channels
        for seed in [int(x) for x in a.seeds.split(",")]:
            t0 = time.time()
            cfg = KSConfig2D(Lx=L, Ly=L, NX=n, NY=n, spinup_time=200.0, seed=seed)
            cache = Path(f"artifacts/cache/ks2d_spinup_L{L:g}_NX{n}_dt0.02_T200_seed{seed}.npy")
            u0 = np.load(cache) if cache.exists() else spinup2d(cfg, np.random.default_rng(seed))
            with torch.no_grad():
                z = ae.encode(torch.tensor(((u0 - mean) / std).reshape(1, -1), dtype=torch.float32))
                roll = prop.rollout(z, z, k=a.n_steps)[0]
                dec = torch.cat([ae.decode(roll[i:i + 100]) for i in range(0, a.n_steps, 100)]).numpy()
            um = dec.reshape(-1, n, n).astype(np.float64)[a.discard:] * std + mean
            zs = roll.numpy().reshape(a.n_steps, ns, ns, c)[a.discard:]
            h = zs.shape[0] // 2
            entry = {"L": L, "seed": seed, "spinup_cache_used": cache.exists(), "model": stats(um, L),
                     "latent_abs_mean_early": np.abs(zs[:h]).mean(axis=(0, 1, 2)).tolist(),
                     "latent_abs_mean_late": np.abs(zs[h:]).mean(axis=(0, 1, 2)).tolist()}
            if not a.no_truth:
                ut = integrate2d(u0, cfg, n_steps=a.n_steps * cfg.snapshot_every)[1:][a.discard:]
                entry["truth"] = stats(ut, L)
            entry["wall_s"] = time.time() - t0
            res["runs"][f"{s:g}_seed{seed}"] = entry
            m = entry["model"]; t = entry.get("truth", {})
            print(f"[{a.tag} {L:g}^2 seed{seed}] model std {m['std_early']:.3f}->{m['std_late']:.3f} "
                  f"bands(late) {[round(x, 3) for x in m['band_shares_late']]}"
                  + (f" | truth std {t['std_early']:.3f}->{t['std_late']:.3f} bands {[round(x, 3) for x in t['band_shares_late']]}" if t else "")
                  + f" ({entry['wall_s']:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}_ks2d_n{a.n_steps}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
