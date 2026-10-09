"""Task A10 (docs/full_paper_tasks.md): truth-free selection of the main-text Kolmogorov
alpha = 0.1 model. The rule is pre-registered in docs/paper1/decisions.md (A10 entry) and is
implemented here exactly; no Lyapunov truth is read anywhere in this script.

  python scripts/select_alpha01_model.py eval --tag k_xs01_v7      # one candidate -> JSON
  python scripts/select_alpha01_model.py decide                    # apply the rule to all JSONs

Step 1 (eligibility): free rollouts, 1000 steps from IC seeds 0, 1 on 1x/2x/3x boxes;
statistics over steps 100-1000: vorticity std and kinetic-energy share in |k| in [3, 5],
each averaged over the two ICs, within 15% of the training-data value at every size.
Step 2 (selection): normalized forecast error MSE/Var on the held-out validation trajectory,
model deployed on the full periodic 32pi x 2pi domain, starts t = 0, 10, ..., 180 snapshots,
free rollout, k = 8 (also k = 1, 16 reported). Lowest wins; < 2% apart -> smaller max
step-1 deviation wins.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.config import KolmogorovConfig
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.kolmogorov import spinup_kolmogorov

CANDIDATES = ["k_xs01", "k_xs01_v2", "k_xs01_v3", "k_xs01_v4", "k_xs01_v5", "k_xs01_v7"]
DATASET = "artifacts/datasets/kolmogorov_wide16_alpha0.1.h5"
OUT = Path("artifacts/experiments/a10_selection")
TOL, TIE = 0.15, 0.02
N_STEPS, DISCARD, SEEDS, SCALES = 1000, 100, (0, 1), (1, 2, 3)
STARTS, KS = list(range(0, 181, 10)), (1, 8, 16)


def scale_local_stats(w: np.ndarray, Lx: float, Ly: float) -> dict:
    """w: (T, NX, NY) physical vorticity. std and kinetic-energy share in |k| in [3, 5]."""
    _, nx, ny = w.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    K = np.sqrt(K2)
    inv = np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1.0), 0.0)
    band = (K >= 3) & (K <= 5)
    tot = forc = 0.0
    for t in range(w.shape[0]):
        p = np.abs(np.fft.fft2(w[t])) ** 2 * inv
        tot += p.sum(); forc += p[band].sum()
    return {"std": float(w.std()), "forcing_share": float(forc / tot)}


def load_model(tag, nsx, nsy, nx, ny):
    ae_p, pr_p = f"artifacts/stage1_ae_kolmogorov_crop_{tag}.pt", f"artifacts/stage2_prop_kolmogorov_crop_{tag}.pt"
    ck = torch.load(ae_p, map_location="cpu", weights_only=False)
    ae = load_autoencoder_checkpoint_resized_2d(ae_p, new_n_sites_x=nsx, new_n_sites_y=nsy, new_NX=nx, new_NY=ny)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(pr_p, new_n_sites_x=nsx, new_n_sites_y=nsy)[0].eval()
    return ae, prop, float(ck["normalization_mean"]), float(ck["normalization_std"])


def training_reference() -> dict:
    path = OUT / "training_reference.json"
    if path.exists():
        return json.loads(path.read_text())
    with h5py.File(DATASET, "r") as f:
        md = dict(f["metadata"].attrs)
        snaps = np.concatenate([np.asarray(f["trajectories"][i, ::10]) for i in range(int(md["n_train"]))], 0)
    w = snaps.reshape(-1, int(md["NX"]), int(md["NY"])).astype(np.float64)
    w = w * float(md["normalization_std"]) + float(md["normalization_mean"])
    ref = scale_local_stats(w, float(md["Lx"]), float(md["Ly"])) | {"n_snapshots": int(w.shape[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2))
    return ref


def evaluate(tag: str) -> None:
    t0 = time.time()
    with h5py.File(DATASET, "r") as f:
        md = dict(f["metadata"].attrs)
        val = np.asarray(f["trajectories"][int(md["n_train"])]).astype(np.float64)  # standardized
    nu, alpha, n_f, dt, dt_snap = float(md["nu"]), float(md["alpha"]), int(md["forcing_n"]), float(md["dt"]), float(md["dt_snap"])
    rec = {"tag": tag, "eligibility": {}, "validation": {}}

    # ---- step 1: free-rollout statistics on 1x/2x/3x boxes ----
    for S in SCALES:
        nx, ny, nsx, nsy = 128 * S, 128, 8 * S, 8
        ae, prop, mean, std = load_model(tag, nsx, nsy, nx, ny)
        per_ic = []
        for seed in SEEDS:
            cfg = KolmogorovConfig(Lx=2 * math.pi * S, Ly=2 * math.pi, NX=nx, NY=ny, nu=nu, alpha=alpha,
                                   forcing_n=n_f, dt=dt, spinup_time=100.0, seed=seed)
            cache = Path(f"artifacts/cache/kolm_spinup_a{alpha:g}_x{S}_seed{seed}.npy")
            if cache.exists():
                u0 = np.load(cache)
            else:
                u0 = spinup_kolmogorov(cfg, np.random.default_rng(seed))
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, u0)
            with torch.no_grad():
                z = ae.encode(torch.tensor(((u0 - mean) / std).reshape(1, -1), dtype=torch.float32))
                roll = prop.rollout(z, z, k=N_STEPS)[0]
                w = torch.cat([ae.decode(roll[i:i + 100]) for i in range(0, N_STEPS, 100)]).numpy()
            w = w.reshape(-1, nx, ny)[DISCARD:].astype(np.float64) * std + mean
            per_ic.append(scale_local_stats(w, cfg.Lx, cfg.Ly) | {"seed": seed, "finite": bool(np.isfinite(w).all())})
        rec["eligibility"][f"{S}x"] = {"per_ic": per_ic,
                                       "std": float(np.mean([r["std"] for r in per_ic])),
                                       "forcing_share": float(np.mean([r["forcing_share"] for r in per_ic]))}

    # ---- step 2: forecast error on the held-out validation trajectory (full 32pi x 2pi domain) ----
    nx, ny = int(md["NX"]), int(md["NY"])
    ae, prop, mean, std = load_model(tag, nx // 16, ny // 16, nx, ny)
    v = val.reshape(-1, nx, ny)  # standardized with the same statistics as the checkpoint
    var_phys = float((v * std).var())
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(v[STARTS].reshape(len(STARTS), -1), dtype=torch.float32))
        roll = prop.rollout(z0, z0, k=max(KS))  # (B, k, d)
        for k in KS:
            pred = ae.decode(roll[:, k - 1]).numpy().reshape(len(STARTS), nx, ny).astype(np.float64)
            tgt = v[[s + k for s in STARTS]]
            mse = ((pred - tgt) * std) ** 2
            rec["validation"][f"k{k}"] = {"normalized_mse": float(mse.mean() / var_phys),
                                          "per_start": [float(m.mean() / var_phys) for m in mse]}
    rec["wall_s"] = time.time() - t0
    rec["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(tag, {s: (round(e["std"], 3), round(e["forcing_share"], 3)) for s, e in rec["eligibility"].items()},
          {k: round(v["normalized_mse"], 4) for k, v in rec["validation"].items()}, f"{rec['wall_s']:.0f}s", flush=True)


def decide() -> None:
    ref = training_reference()
    rows = []
    for tag in CANDIDATES:
        r = json.loads((OUT / f"{tag}.json").read_text())
        devs = {}
        for s, e in r["eligibility"].items():
            devs[s] = {"std": e["std"] / ref["std"] - 1, "forcing_share": e["forcing_share"] / ref["forcing_share"] - 1}
        maxdev = max(abs(x) for d in devs.values() for x in d.values())
        rows.append({"tag": tag, "eligible": maxdev <= TOL, "max_abs_deviation": maxdev, "deviations": devs,
                     "val_k1": r["validation"]["k1"]["normalized_mse"], "val_k8": r["validation"]["k8"]["normalized_mse"],
                     "val_k16": r["validation"]["k16"]["normalized_mse"]})
    pool = [r for r in rows if r["eligible"]] or rows
    pool.sort(key=lambda r: r["val_k8"])
    best = pool[0]
    tied = [r for r in pool[1:] if (r["val_k8"] - best["val_k8"]) / best["val_k8"] < TIE]
    if tied:
        best = min([best] + tied, key=lambda r: r["max_abs_deviation"])
    out = {"rule": "docs/paper1/decisions.md, A10 pre-registration", "training_reference": ref,
           "tolerance": TOL, "tie_threshold": TIE, "any_eligible": any(r["eligible"] for r in rows),
           "candidates": rows, "tied_with_best": [r["tag"] for r in tied], "selected": best["tag"]}
    (OUT / "decision.json").write_text(json.dumps(out, indent=2))
    print(f"training data: std {ref['std']:.3f}, forcing share {ref['forcing_share']:.3f}")
    print(f"{'tag':12s} {'eligible':8s} {'max|dev|':>8s} {'val k1':>8s} {'val k8':>8s} {'val k16':>8s}")
    for r in sorted(rows, key=lambda r: r["val_k8"]):
        print(f"{r['tag']:12s} {str(r['eligible']):8s} {r['max_abs_deviation']:8.3f} {r['val_k1']:8.4f} "
              f"{r['val_k8']:8.4f} {r['val_k16']:8.4f}")
    print("selected:", best["tag"], "| tied:", [r["tag"] for r in tied])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "decide", "reference"])
    ap.add_argument("--tag", choices=CANDIDATES)
    a = ap.parse_args()
    if a.mode == "eval":
        evaluate(a.tag)
    elif a.mode == "reference":
        print(training_reference())
    else:
        decide()
