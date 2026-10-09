"""C4 (docs/ks_local_field_ltransfer.tex Sec. capacity_experiments):
zero-training fusion test of the consensus hypothesis.

During rollout, apply the re-projection z <- E(D(z)) every m propagator
steps. The Lyapunov spectrum is that of the composite map E.D.P^m, with
m*dt_snap physical time per composite step (D_KY is dimensionless, so this
only affects lambda units). m = 0 means no re-projection (the baseline).

All ICs are normalized with the training split's stats (see
scripts/exp_ic_normalization_check.py for why this matters). Same Lyapunov
settings as exp_g3 (n_directions=d_latent, QR every 10 steps, warmup 200,
2000 steps), with the step counts divided by m so every run covers the same
2000 propagator steps of physical time.

Pre-registered rule: consensus-supporting if 2x transfer error improves
while native-size D_KY stays within 10% of the m=0 native value;
uninterpretable if native D_KY moves by more than 10%.

Preemptable, one task at a time. Usage:
  python scripts/exp_c4_reprojection.py [--summarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KolmogorovConfig
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.kolmogorov import spinup_kolmogorov

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "c4"

MODELS = {
    "s231_a005": dict(
        dataset=A / "datasets/kolmogorov_trajectories_default.h5",
        ae=A / "stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt",
        prop=A / "stage2_prop_kolmogorov_full_section231_kolmogorov_localmlp2d_window3_warmstart_k12_300ep.pt",
        truth={1: 23.26, 2: 56.84},
    ),
    "s233_a01": dict(
        dataset=A / "datasets/kolmogorov_trajectories_alpha0.1.h5",
        ae=A / "stage1_ae_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3.pt",
        prop=A / "stage2_prop_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3_warmstart_k12_300ep.pt",
        truth={1: 15.88, 2: 30.34},
    ),
}
SCALES = (1, 2)
MS = (0, 1, 4, 16)


class Reprojected(nn.Module):
    """Composite map z -> E(D(P^m(z))); m=0 is the plain propagator."""

    def __init__(self, prop, ae, m: int):
        super().__init__()
        self.prop, self.ae, self.m = prop, ae, m
        self.cfg = prop.cfg

    def step(self, z_prev, z_curr):
        if self.m == 0:
            return self.prop.step(z_prev, z_curr)
        z = z_curr
        for _ in range(self.m):
            z = self.prop.step(z, z)
        return self.ae.encode(self.ae.decode(z))


def result_path(name, scale, m):
    return OUT / f"{name}_x{scale}_m{m}.json"


def run_one(name, scale, m, seed=0):
    spec = MODELS[name]
    with h5py.File(spec["dataset"], "r") as f:
        md = dict(f["metadata"].attrs)
    mean, std = float(md["normalization_mean"]), float(md["normalization_std"])
    cfg0 = torch.load(spec["ae"], map_location="cpu", weights_only=False)["ae_config"]
    nsx, nsy = cfg0.n_sites_x * scale, cfg0.n_sites_y
    NX, NY = cfg0.NX * scale, cfg0.NY
    ae = load_autoencoder_checkpoint_resized_2d(str(spec["ae"]), new_n_sites_x=nsx, new_n_sites_y=nsy,
                                                new_NX=NX, new_NY=NY)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(str(spec["prop"]), new_n_sites_x=nsx, new_n_sites_y=nsy)[0].eval()
    cfg = KolmogorovConfig(Lx=float(md["Lx"]) * scale, Ly=float(md["Ly"]), NX=NX, NY=NY, nu=float(md["nu"]),
                           alpha=float(md["alpha"]), forcing_n=int(md["forcing_n"]), dt=float(md["dt"]),
                           spinup_time=100.0, seed=seed)
    u0 = (spinup_kolmogorov(cfg, np.random.default_rng(seed)) - mean) / std
    t0 = time.time()
    with torch.no_grad():
        z0 = ae.encode(torch.tensor(u0.reshape(1, -1), dtype=torch.float32))[0].numpy().astype(np.float64)
    model = Reprojected(prop, ae, m)
    mm = max(m, 1)  # fixed total propagator steps (2000) and physical time across m
    d = prop.cfg.d_latent
    res = lyapunov_spectrum_latent_propagator(
        model, np.stack([z0, z0]), mode="single_state", n_directions=d, n_steps=2000 // mm,
        qr_every=max(1, 10 // mm), dt_snap=float(md["dt_snap"]) * mm, warmup_steps=200 // mm,
        seed=seed, max_abs_state=1e3,
    )
    truth = spec["truth"][scale]
    out = {
        "model": name, "scale": scale, "m": m, "seed": seed, "d_latent": d,
        "model_D_KY": res.kaplan_yorke_dimension, "model_lambda1": float(res.exponents[0]),
        "model_n_positive": int(res.n_positive), "true_D_KY": truth,
        "rel_err_pct": abs(res.kaplan_yorke_dimension - truth) / truth * 100.0,
        "ic": "normalized", "wall_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path(name, scale, m).write_text(json.dumps(out, indent=2))
    return out


def summarize():
    rows, verdict = [], {}
    for name in MODELS:
        native0 = result_path(name, 1, 0)
        base2 = result_path(name, 2, 0)
        for m in MS:
            r1, r2 = result_path(name, 1, m), result_path(name, 2, m)
            row = {"model": name, "m": m}
            if r1.exists():
                row["x1"] = json.loads(r1.read_text())
            if r2.exists():
                row["x2"] = json.loads(r2.read_text())
            rows.append(row)
            if m and r1.exists() and r2.exists() and native0.exists() and base2.exists():
                n0 = json.loads(native0.read_text())["model_D_KY"]
                b2 = json.loads(base2.read_text())["rel_err_pct"]
                native_shift = abs(row["x1"]["model_D_KY"] - n0) / n0 * 100.0
                improved = row["x2"]["rel_err_pct"] < b2
                verdict[f"{name}_m{m}"] = (
                    "UNINTERPRETABLE (native D_KY moved %.0f%%)" % native_shift if native_shift > 10
                    else ("CONSENSUS-SUPPORTING" if improved else "NO IMPROVEMENT")
                )
    out = {"rows": rows, "verdict": verdict}
    (OUT / "c4_summary.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        x1, x2 = r.get("x1", {}), r.get("x2", {})
        print(f"{r['model']:10s} m={r['m']:2d}  x1 D_KY={x1.get('model_D_KY', float('nan')):6.2f}  "
              f"x2 D_KY={x2.get('model_D_KY', float('nan')):6.2f} err={x2.get('rel_err_pct', float('nan')):5.1f}%")
    print(json.dumps(verdict, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if not args.summarize:
        try:
            os.nice(19)
        except OSError:
            pass
        for name in MODELS:
            for m in MS:
                for scale in SCALES:
                    if result_path(name, scale, m).exists():
                        continue
                    print(f"[c4 {time.strftime('%H:%M:%S')}] {name} x{scale} m={m}", flush=True)
                    r = run_one(name, scale, m)
                    print(f"   D_KY={r['model_D_KY']:.2f} (truth {r['true_D_KY']}) err={r['rel_err_pct']:.1f}% "
                          f"[{r['wall_s']:.0f}s]", flush=True)
    summarize()


if __name__ == "__main__":
    main()
