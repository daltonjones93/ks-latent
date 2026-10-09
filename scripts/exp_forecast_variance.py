"""Task C7(b) (docs/full_paper_tasks.md): variance of free forecasts vs lead time (Lemma 3).

For each model, on the held-out validation trajectory of its training dataset (model deployed
on the full periodic training domain), forecasts start from the encoded truth at
t = 0, 5, ..., 100 snapshots and run freely for k = 1..100 steps. Reported per lead k:
  ratio(k) = mean_starts Var_x(decoded forecast) / mean_starts Var_x(truth at the same time),
  nmse(k)  = mean_starts mean_x (forecast - truth)^2 / Var(truth).
Lemma 3: a squared-error-trained map approaches the conditional mean at leads it was trained
on, whose variance is the truth's minus the conditional variance, so ratio(k) < 1 and falls
with lead in proportion to how far out it was trained. The domain is wide (16 base boxes),
so the spatial variance of one forecast stands in for the ensemble variance.

  python scripts/exp_forecast_variance.py --alpha 0.05 --tags c19_e0d0_c6 k_xs005_noreg
Writes artifacts/experiments/forecast_variance/<tag>.json
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

from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d

OUT = Path("artifacts/experiments/forecast_variance")
STARTS, KMAX = list(range(0, 101, 5)), 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    ds = f"artifacts/datasets/kolmogorov_wide16_alpha{a.alpha:g}.h5"
    with h5py.File(ds, "r") as f:
        md = dict(f["metadata"].attrs)
        val = np.asarray(f["trajectories"][int(md["n_train"])]).astype(np.float64)  # standardized
    nx, ny = int(md["NX"]), int(md["NY"])
    v = val.reshape(-1, nx, ny)
    var_t = float(v.var())
    for tag in a.tags:
        t0 = time.time()
        ae_p, pr_p = f"artifacts/stage1_ae_kolmogorov_crop_{tag}.pt", f"artifacts/stage2_prop_kolmogorov_crop_{tag}.pt"
        ck = torch.load(ae_p, map_location="cpu", weights_only=False)
        assert ck["train_dataset"].endswith(Path(ds).name), (ck["train_dataset"], ds)
        k_max = torch.load(pr_p, map_location="cpu", weights_only=False)["crop_prop_config"]["k_max"]
        ae = load_autoencoder_checkpoint_resized_2d(ae_p, new_n_sites_x=nx // 16, new_n_sites_y=ny // 16,
                                                    new_NX=nx, new_NY=ny)[0].eval()
        prop = load_propagator_checkpoint_resized_2d(pr_p, new_n_sites_x=nx // 16, new_n_sites_y=ny // 16)[0].eval()
        ratio, nmse = np.zeros(KMAX), np.zeros(KMAX)
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(v[STARTS].reshape(len(STARTS), -1), dtype=torch.float32))
            roll = prop.rollout(z0, z0, k=KMAX)  # (B, k, d)
            for k in range(1, KMAX + 1):
                pred = ae.decode(roll[:, k - 1]).numpy().reshape(len(STARTS), nx, ny).astype(np.float64)
                tgt = v[[s + k for s in STARTS]]
                ratio[k - 1] = pred.var(axis=(1, 2)).mean() / tgt.var(axis=(1, 2)).mean()
                nmse[k - 1] = ((pred - tgt) ** 2).mean() / var_t
            recon = ae.decode(z0).numpy().reshape(len(STARTS), nx, ny)
        rec = {"tag": tag, "alpha": a.alpha, "k_max": int(k_max), "dataset": ds, "starts": STARTS,
               "dt_snap": float(md["dt_snap"]), "lead_steps": list(range(1, KMAX + 1)),
               "variance_ratio": ratio.tolist(), "nmse": nmse.tolist(),
               "recon_variance_ratio": float(recon.var(axis=(1, 2)).mean() / v[STARTS].var(axis=(1, 2)).mean()),
               "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               "wall_s": time.time() - t0}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2))
        print(f"{tag} k_max={k_max}: ratio k=1,8,16,32,64,100: "
              + " ".join(f"{ratio[k - 1]:.3f}" for k in (1, 8, 16, 32, 64, 100))
              + f" | nmse k=8 {nmse[7]:.3f} ({rec['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
