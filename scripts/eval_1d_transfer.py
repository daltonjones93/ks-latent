"""Task A1 (docs/full_paper_tasks.md): zero-shot transfer evaluation of a 1D local-field
model at a set of domain lengths, with STANDARDIZED initial conditions (dataset mean/std),
writing one JSON per (model, L, IC seed, IC convention) with the full Lyapunov spectrum,
decoded-rollout statistics and time-averaged energy spectra (truth and model).

Replaces the hard-coded MODEL_TRANSFERRED_D_KY values of exp_e1 and the raw-IC encoding of
run_ltransfer_test.py. Protocol otherwise identical to the record's Phase F2 / E1 / E4:
truth IC = spinup(KSConfig(L, NX=256*L/100, dt=0.05, snapshot_every=20, spinup 500), rng(seed));
Lyapunov: single_state, n_directions = d_latent, 2000 steps, qr_every 10, warmup 200, seed 0,
dt_snap 1.0.

Usage:
  python scripts/eval_1d_transfer.py --tag section224 --L 25 50 100 200 400 --ic-seed 0 --ic normalized
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

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.models import load_autoencoder_checkpoint_resized, load_propagator_checkpoint_resized
from ks_latent.solver.ks import KSConfig, integrate, spinup, wavenumbers

MODELS = {
    "section224": (
        "artifacts/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt",
        "artifacts/stage2_prop_patched_full_section224_ks_localfield_d48_localmlp_warmstart_k12_300ep.pt",
    ),
    "e4_seed1": (
        "artifacts/stage1_ae_patched_full_exp_e4_seed1_localfield_d48_localmlp.pt",
        "artifacts/stage2_prop_patched_full_exp_e4_seed1_localfield_d48_localmlp_warmstart_k12_300ep.pt",
    ),
    "e4_seed2": (
        "artifacts/stage1_ae_patched_full_exp_e4_seed2_localfield_d48_localmlp.pt",
        "artifacts/stage2_prop_patched_full_exp_e4_seed2_localfield_d48_localmlp_warmstart_k12_300ep.pt",
    ),
}
DATASET = "artifacts/datasets/stage1_trajectories_dtsnap1.h5"
OLD_L = 100.0
OUT = Path("artifacts/experiments/e1_model")


def energy_spectrum(traj: np.ndarray) -> np.ndarray:
    """Time-averaged |u_hat(k)|^2 / NX^2 over rows of a (T, NX) trajectory, k >= 0 half."""
    nx = traj.shape[1]
    spec = np.mean(np.abs(np.fft.rfft(traj, axis=1)) ** 2, axis=0) / nx**2
    return spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", choices=list(MODELS), required=True)
    p.add_argument("--L", type=float, nargs="+", required=True)
    p.add_argument("--ic-seed", type=int, default=0)
    p.add_argument("--ic", choices=["normalized", "raw"], default="normalized")
    p.add_argument("--n-lyap-steps", type=int, default=2000)
    p.add_argument("--n-roll", type=int, default=400)
    p.add_argument("--tangent-dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    ae_path, prop_path = MODELS[args.tag]
    with h5py.File(DATASET, "r") as f:
        mean = float(f["metadata"].attrs["normalization_mean"])
        std = float(f["metadata"].attrs["normalization_std"])
    old_cfg = torch.load(ae_path, map_location="cpu", weights_only=False)["ae_config"]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # provenance only
        sha = "unknown"
    OUT.mkdir(parents=True, exist_ok=True)

    for L in args.L:
        t0 = time.time()
        scale = L / OLD_L
        n_sites = int(round(old_cfg.n_sites * scale))
        nx = int(round(old_cfg.NX * scale))
        assert nx % n_sites == 0
        ae, ae_cfg, _ = load_autoencoder_checkpoint_resized(ae_path, new_n_sites=n_sites, new_NX=nx)
        d = ae_cfg.n_sites * ae_cfg.local_channels
        prop, _, _ = load_propagator_checkpoint_resized(prop_path, new_d_latent=d, new_n_tokens=n_sites)
        ae.eval(); prop.eval()

        ks_cfg = KSConfig(L=L, NX=nx, dt=0.05, snapshot_every=20, spinup_time=500.0, seed=args.ic_seed)
        u0 = spinup(ks_cfg, np.random.default_rng(args.ic_seed))
        truth = integrate(u0, ks_cfg, n_steps=args.n_roll * ks_cfg.snapshot_every)  # (n_roll+1, nx), raw
        enc_in = (truth - mean) / std if args.ic == "normalized" else truth

        with torch.no_grad():
            z0 = ae.encode(torch.tensor(enc_in[0:1], dtype=torch.float32))
            z_roll = prop.rollout(z0, z0, k=args.n_roll)[0]
            # The decoder always outputs standardized units (it was trained on (u-mean)/std),
            # whatever was fed to the encoder; convert to physical units in both conventions.
            # (run_ltransfer_test.py compared standardized decoder output with raw truth, which is
            # the origin of the record's "~20% amplitude undershoot": 2.62 * 1.316 = 3.45.)
            u_model = ae.decode(torch.cat([z0, z_roll], 0)).numpy().astype(np.float64) * std + mean
            z01 = ae.encode(torch.tensor(enc_in[0:2], dtype=torch.float32)).numpy()

        finite = bool(np.isfinite(u_model).all())
        half = slice(args.n_roll // 2, None)  # statistics over the second half of the rollout
        k = wavenumbers(L, nx)[: nx // 2 + 1]
        spec_true = energy_spectrum(truth[half])
        spec_model = energy_spectrum(u_model[half]) if finite else None

        lyap_kwargs = dict(mode="single_state", n_directions=d, n_steps=args.n_lyap_steps, qr_every=10,
                           dt_snap=1.0, warmup_steps=200, seed=0, max_abs_state=1e4)
        if args.tangent_dtype == "float64":
            lyap_kwargs["tangent_dtype"] = torch.float64
        try:
            res = lyapunov_spectrum_latent_propagator(prop, z01, **lyap_kwargs)
            ex, dky, npos, err = res.exponents, res.kaplan_yorke_dimension, res.n_positive, None
        except (RuntimeError, ValueError) as e:  # recorded, not swallowed: the JSON carries the error
            ex, dky, npos, err = None, None, None, f"{type(e).__name__}: {e}"

        rec = {
            "model": args.tag, "ae_checkpoint": ae_path, "prop_checkpoint": prop_path,
            "L": L, "NX": nx, "n_sites": n_sites, "d_latent": d, "ic_seed": args.ic_seed, "ic": args.ic,
            "normalization_mean": mean, "normalization_std": std,
            "tangent_dtype": args.tangent_dtype, "n_lyap_steps": args.n_lyap_steps,
            "D_KY": dky, "D_KY_over_L": (dky / L) if dky is not None else None,
            "lambda1": float(ex[0]) if ex is not None else None, "n_positive": npos,
            "exponents": [float(x) for x in ex] if ex is not None else None, "error": err,
            "rollout_finite": finite,
            "max_abs_u_true": float(np.abs(truth).max()),
            "max_abs_u_model": float(np.abs(u_model).max()) if finite else None,
            "std_u_true": float(truth[half].std()),
            "std_u_model": float(u_model[half].std()) if finite else None,
            "k": k.tolist(), "energy_spectrum_true": spec_true.tolist(),
            "energy_spectrum_model": spec_model.tolist() if spec_model is not None else None,
            "k_peak_true": float(k[1:][np.argmax(spec_true[1:])]),
            "k_peak_model": float(k[1:][np.argmax(spec_model[1:])]) if spec_model is not None else None,
            "git_sha": sha, "wall_s": time.time() - t0,
        }
        name = f"{args.tag}_L{L:g}_ic{args.ic_seed}_{args.ic}" + ("_f64" if args.tangent_dtype == "float64" else "")
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=2))
        print(f"[{name}] D_KY={dky} n_pos={npos}/{d} lambda1={rec['lambda1']} "
              f"k_peak true/model={rec['k_peak_true']:.3f}/{rec['k_peak_model']} "
              f"max|u| true/model={rec['max_abs_u_true']:.2f}/{rec['max_abs_u_model']} ({rec['wall_s']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
