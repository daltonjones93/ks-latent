"""C0 (docs/ks_local_field_ltransfer.tex Sec. capacity): capacity audit of
existing checkpoints, no training.

For each (model, transfer size): integrate the TRUE PDE from 3 independent
spun-up ICs, normalize with the training split's stats, encode, and also
roll the model's own propagator forward from each encoded IC. On both latent
sets measure:
  - per-site participation ratio of the c x c channel covariance (pooled over
    sites and time) -- the pre-registered C0 statistic -- plus the same on the
    channel CORRELATION matrix (scale-free; covariance PR is dominated by
    whichever channel has the largest variance) and a two-NN intrinsic
    dimension of the per-site c-vectors (nonlinear);
  - global participation ratio over all d_latent channels;
  - rollout/true total-variance ratio (a direct collapse indicator).

Pre-registered rule: if per-site rank < c/2 at every FAILING transfer,
capacity is ruled out as the primary cause.

Preemptable: one JSON per (model, size); reruns skip finished entries.
Usage: python scripts/exp_c0_capacity_audit.py [--summarize]
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

from ks_latent.analysis.dimension import two_nn_dimension
from ks_latent.config import KolmogorovConfig, KSConfig, KSConfig2D
from ks_latent.models import (
    load_autoencoder_checkpoint_resized,
    load_autoencoder_checkpoint_resized_2d,
    load_propagator_checkpoint_resized,
    load_propagator_checkpoint_resized_2d,
    load_propagator_checkpoint_resized_2d_biotsavart,
    load_propagator_checkpoint_resized_2d_multiscale,
)
from ks_latent.solver.kolmogorov import integrate_kolmogorov, spinup_kolmogorov
from ks_latent.solver.ks import integrate, spinup
from ks_latent.solver.ks2d import integrate2d, spinup2d

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "c0"
N_IC = 3
N_SNAP = 100
N_ROLL = 600
ROLL_DISCARD = 200

K05 = A / "datasets/kolmogorov_trajectories_default.h5"
K10 = A / "datasets/kolmogorov_trajectories_alpha0.1.h5"
KS2 = A / "datasets/ks2d_trajectories_default.h5"
KS1 = A / "datasets/stage1_trajectories_dtsnap1.h5"
AE231 = A / "stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt"

# name: (system, dataset, ae, prop, scales, true D_KY by scale, transfer error by scale)
MODELS = {
    "ks1d_s224": ("ks1d", KS1, A / "stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt",
                  A / "stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt",
                  (1, 2, 4), {1: 22.14, 2: 44.61, 4: 89.13}, {2: "<2%", 4: "<2%"}),
    "ks2d_s230": ("ks2d", KS2, A / "stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt",
                  A / "stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt",
                  (1, 1.5), {1: 85.6, 1.5: 192.5}, {1.5: "77%"}),
    "kolm01_s233": ("kolmogorov", K10, A / "stage1_ae_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3.pt",
                    A / "stage2_prop_kolmogorov_full_section233_kolmogorov_alpha0p1_localmlp2d_window3_warmstart_k12_300ep.pt",
                    (1, 2, 3), {1: 15.9, 2: 30.3, 3: 56.4}, {2: "4.5%", 3: "19.4%"}),
    "kolm005_s231": ("kolmogorov", K05, AE231,
                     A / "stage2_prop_kolmogorov_full_section231_kolmogorov_localmlp2d_window3_warmstart_k12_300ep.pt",
                     (1, 2), {1: 23.26, 2: 56.8}, {2: "50%"}),
    "kolm005_d2": ("kolmogorov", K05, AE231,
                   A / "stage2_prop_kolmogorov_full_section236_kolmogorov_alpha0p05_localmlp2d_biotsavart_warmstart_k12_300ep.pt",
                   (1, 2), {1: 23.26, 2: 56.8}, {2: "95%"}),
    "kolm005_d3": ("kolmogorov", K05, AE231,
                   A / "stage2_prop_kolmogorov_full_section237_kolmogorov_alpha0p05_multiscale_local_mlp2d_warmstart_k12_300ep.pt",
                   (1, 2), {1: 23.26, 2: 56.8}, {2: "99%"}),
}


def load_prop_2d(path, nsx, nsy):
    backbone = getattr(torch.load(path, map_location="cpu", weights_only=False)["prop_config"], "backbone", None)
    loader = {
        "local_mlp_2d_biotsavart": load_propagator_checkpoint_resized_2d_biotsavart,
        "multiscale_local_mlp_2d": load_propagator_checkpoint_resized_2d_multiscale,
    }.get(backbone, load_propagator_checkpoint_resized_2d)
    return loader(path, new_n_sites_x=nsx, new_n_sites_y=nsy)[0]


def build(system, dataset, ae_path, prop_path, scale):
    """Returns (ae, prop, c, site_shape, truth_fn, norm)."""
    with h5py.File(dataset, "r") as f:
        md = dict(f["metadata"].attrs)
    mean, std = float(md["normalization_mean"]), float(md["normalization_std"])
    cfg0 = torch.load(ae_path, map_location="cpu", weights_only=False)["ae_config"]
    if system == "ks1d":
        ns, NX = int(round(cfg0.n_sites * scale)), int(round(cfg0.NX * scale))
        ae = load_autoencoder_checkpoint_resized(str(ae_path), new_n_sites=ns, new_NX=NX)[0]
        prop = load_propagator_checkpoint_resized(str(prop_path), new_d_latent=ns * cfg0.local_channels,
                                                  new_n_tokens=ns)[0]
        cfg = KSConfig(L=float(md["L"]) * scale, NX=NX, dt=0.05, snapshot_every=round(float(md["dt_snap"]) / 0.05),
                       spinup_time=500.0)

        def truth(rng):
            u0 = spinup(cfg, rng)
            return integrate(u0, cfg, n_steps=N_SNAP * cfg.snapshot_every)[1:]
        return ae, prop, cfg0.local_channels, (ns,), truth, (mean, std)

    nsx, nsy = int(round(cfg0.n_sites_x * scale)), cfg0.n_sites_y
    NX, NY = int(round(cfg0.NX * scale)), cfg0.NY
    Lx, Ly = float(md["Lx"]) * scale, float(md["Ly"])
    if system == "ks2d":  # isotropic: both axes scale
        nsy, NY, Ly = int(round(cfg0.n_sites_y * scale)), int(round(cfg0.NY * scale)), float(md["Ly"]) * scale
    ae = load_autoencoder_checkpoint_resized_2d(str(ae_path), new_n_sites_x=nsx, new_n_sites_y=nsy,
                                                new_NX=NX, new_NY=NY)[0]
    prop = load_prop_2d(str(prop_path), nsx, nsy)
    if system == "kolmogorov":
        cfg = KolmogorovConfig(Lx=Lx, Ly=Ly, NX=NX, NY=NY, nu=float(md["nu"]), alpha=float(md["alpha"]),
                               forcing_n=int(md["forcing_n"]), dt=float(md["dt"]), spinup_time=100.0)
        spin, integ = spinup_kolmogorov, integrate_kolmogorov
    else:
        cfg = KSConfig2D(Lx=Lx, Ly=Ly, NX=NX, NY=NY, spinup_time=200.0)
        spin, integ = spinup2d, integrate2d

    def truth(rng):
        u0 = spin(cfg, rng)
        return integ(u0, cfg, n_steps=N_SNAP * cfg.snapshot_every)[1:].reshape(N_SNAP, -1)
    return ae, prop, cfg0.local_channels, (nsx, nsy), truth, (mean, std)


def pr(C):
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def metrics(Z, c, rng):
    """Z: (N, d_latent), site-major."""
    sites = Z.reshape(-1, c)
    Cs = np.cov(sites, rowvar=False)
    sd = np.sqrt(np.diag(Cs))
    sub = sites[rng.choice(len(sites), size=min(20000, len(sites)), replace=False)]
    sub = sub + 1e-9 * rng.standard_normal(sub.shape)  # break exact ties
    return {
        "per_site_pr_cov": pr(Cs),
        "per_site_pr_corr": pr(Cs / np.outer(sd, sd)),
        "per_site_twonn": float(two_nn_dimension(sub).dimension),
        "per_site_channel_std": sd.tolist(),
        "global_pr_cov": pr(np.cov(Z, rowvar=False)),
        "n_samples": int(Z.shape[0]),
        "total_variance": float(np.trace(np.cov(Z, rowvar=False))),
    }


def run_one(name, scale):
    system, dataset, ae_p, prop_p, _, dky, err = MODELS[name]
    out_path = OUT / f"{name}_x{scale:g}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    t0 = time.time()
    ae, prop, c, site_shape, truth_fn, (mean, std) = build(system, dataset, ae_p, prop_p, scale)
    ae.eval(); prop.eval()
    Z_true, Z_roll = [], []
    for ic in range(N_IC):
        rng = np.random.default_rng(1000 + ic)
        u = (truth_fn(rng) - mean) / std
        with torch.no_grad():
            z = ae.encode(torch.tensor(u, dtype=torch.float32))
            Z_true.append(z.numpy())
            roll = prop.rollout(z[:1], z[:1], k=N_ROLL)[0]
            Z_roll.append(roll[ROLL_DISCARD:].numpy())
    Z_true, Z_roll = np.concatenate(Z_true), np.concatenate(Z_roll)
    rng = np.random.default_rng(0)
    res = {
        "model": name, "system": system, "scale": scale, "c": c, "site_shape": site_shape,
        "d_latent": int(Z_true.shape[1]), "true_D_KY": dky.get(scale), "transfer_error": err.get(scale),
        "true": metrics(Z_true, c, rng),
        "rollout": metrics(Z_roll, c, rng) if np.isfinite(Z_roll).all() else {"nonfinite": True},
        "wall_s": time.time() - t0,
    }
    if "total_variance" in res["rollout"]:
        res["rollout_over_true_variance"] = res["rollout"]["total_variance"] / res["true"]["total_variance"]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    return res


def summarize():
    rows = []
    for name, spec in MODELS.items():
        for s in spec[4]:
            p = OUT / f"{name}_x{s:g}.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
    hdr = (f"{'model':14s} {'x':>4s} {'d':>4s} {'DKY':>6s} {'err':>6s} | "
           f"{'PRcov':>5s} {'PRcor':>5s} {'2NN':>5s} {'gPR':>6s} | "
           f"{'PRcov':>5s} {'PRcor':>5s} {'2NN':>5s} {'gPR':>6s} {'var':>6s}")
    print("                                       | --- true-encoded ---       | --- model rollout ---")
    print(hdr)
    for r in rows:
        t, o = r["true"], r["rollout"]
        rv = r.get("rollout_over_true_variance", float("nan"))
        if o.get("nonfinite"):
            ro = "nonfinite"
        else:
            ro = (f"{o['per_site_pr_cov']:5.2f} {o['per_site_pr_corr']:5.2f} {o['per_site_twonn']:5.2f} "
                  f"{o['global_pr_cov']:6.1f} {rv:6.3f}")
        print(f"{r['model']:14s} {r['scale']:4g} {r['d_latent']:4d} {str(r['true_D_KY']):>6s} "
              f"{str(r['transfer_error'] or '-'):>6s} | {t['per_site_pr_cov']:5.2f} {t['per_site_pr_corr']:5.2f} "
              f"{t['per_site_twonn']:5.2f} {t['global_pr_cov']:6.1f} | {ro}")
    (OUT / "c0_summary.json").write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if not args.summarize:
        try:
            os.nice(19)
        except OSError:
            pass
        for name, spec in MODELS.items():
            for s in spec[4]:
                print(f"[c0 {time.strftime('%H:%M:%S')}] {name} x{s:g}", flush=True)
                r = run_one(name, s)
                print(f"   true PRcov={r['true']['per_site_pr_cov']:.2f} 2NN={r['true']['per_site_twonn']:.2f}  "
                      f"rollout/true var={r.get('rollout_over_true_variance')}", flush=True)
    summarize()


if __name__ == "__main__":
    main()
