"""Task I30(d) (2026-10-09): does the learned propagator degrade on DA analysis states?

Hypothesis (user): repeated analyses put latent states where the propagator was not trained (its
Stage B input noise annealed to 0.02 latent units), so its forecasts from analysis states are worse
than from encoded true states, which would drive the slow divergence at sparse observations.
Training with larger input noise would then help.

Test. One latent filter (12-channel model, one tiling, stride 16, RTPS 0.5, as in I21) is cycled for
`--n-cycles`. At every `--every`-th cycle, for three kinds of starting state z:
  analysis : the analysis mean;
  truth    : E(u_t), the encoded truth;
  pert     : E(u_t) + isotropic noise with the same rms as the analysis mean's latent error;
the propagator's one-cycle forecast M^n(z) is compared with the "true" forecast from the same
state, E(PDE_T(D(z))): decode, integrate the 2D KS PDE for one cycle (T = n dt_snap), encode. The
same definition is used for all three, so decoder and encoder errors enter each equally.
model error(z) = rms over latent coordinates of M^n(z) - E(PDE_T(D(z))).

Reading:
  analysis ~ truth            -> no degradation; the hypothesis is rejected.
  analysis ~ pert >> truth    -> the propagator degrades for ANY perturbation of that size; training
                                 with input noise of that size is the indicated fix.
  analysis >> pert            -> analysis states are worse than random perturbations of equal size
                                 (structured, filter-specific); noise training may not cover them.
Writes artifacts/experiments/da2d/i30_degradation_<out>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d
from ks_latent.utils.seeding import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p7c12_ks2d")
    ap.add_argument("--n-cycles", type=int, default=200)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--obs-stride", type=int, default=16)
    ap.add_argument("--rtps", type=float, default=0.5)
    ap.add_argument("--gc-c", type=float, default=1.5)
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--model-noise-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()
    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    PR = f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(44.0, training_dx(ck["train_dataset"]), patch)
    NX, ns = g.NX, g.n_sites
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
    std = float(ck["normalization_std"])
    n = 5
    cfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=a.seed)
    u0 = np.load(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt0.02_T200_seed{a.seed}.npy")
    truth = integrate2d(u0, cfg, n_steps=n * a.n_cycles * cfg.snapshot_every)  # field units
    steps_cycle = n * cfg.snapshot_every

    enc = lambda F: ae.encode(torch.tensor(F.reshape(len(F), -1) / std, dtype=torch.float32))  # noqa: E731
    dec = lambda Z: ae.decode(Z).double().numpy().reshape(len(Z), NX, NX) * std  # noqa: E731
    rms = lambda v: float(v.pow(2).mean().sqrt())  # noqa: E731

    def model_err(z: torch.Tensor) -> float:
        with torch.no_grad():
            zm = z.clone(); zp = z.clone()
            for _ in range(n):
                zn = prop.step(zp, zm); zp, zm = zm, zn
            u = dec(z)[0]
            u_T = integrate2d(u - u.mean(), cfg, n_steps=steps_cycle, snapshot_every=steps_cycle)[-1]
            return rms(zm - enc(u_T[None]))

    s = a.obs_stride
    site_of_obs = obs_site_map_2d(NX, NX, ns, ns, s, s)
    h = lambda z: ae.decode(z.to(torch.float32).unsqueeze(0)).squeeze(0).reshape(NX, NX)[::s, ::s].reshape(-1).double()  # noqa: E731
    R_diag = torch.full((site_of_obs.shape[0],), 0.1 ** 2, dtype=torch.float64)
    with torch.no_grad():
        z0 = enc(truth[:1])[0]
    d = z0.shape[0]; c = d // (ns * ns)
    gen = torch.Generator().manual_seed(a.seed)
    Zc = z0 + 0.1 * torch.randn(a.n_ensemble, d, generator=gen); Zp = Zc.clone()
    rng = np.random.default_rng(a.seed + 1)
    rec = {k: [] for k in ("cycle", "z_err", "analysis", "truth", "pert")}
    for cyc in range(a.n_cycles):
        fc = forecast_ensemble(prop, Zp, Zc, n, a.model_noise_std)
        Zp, Zf = fc.z_prev_final, fc.z_curr_final
        u = truth[(cyc + 1) * n]
        y = torch.tensor((u[::s, ::s] / std + 0.1 * rng.standard_normal((NX // s, NX // s))).reshape(-1))
        filt = LETKFLocal2D(h, R_diag, site_of_obs, ns, ns, c, a.gc_c, seed=a.seed * 1000 + cyc, rtps=a.rtps)
        Zc = filt.analyze(Zf.double(), y).Z_analysis.float()
        if cyc % a.every == a.every - 1:
            with torch.no_grad():
                za = Zc.mean(0, keepdim=True); zt = enc(u[None])
            e = rms(za - zt)
            zpert = zt + e * torch.randn(zt.shape, generator=gen)
            rec["cycle"].append(cyc + 1); rec["z_err"].append(e)
            rec["analysis"].append(model_err(za)); rec["truth"].append(model_err(zt)); rec["pert"].append(model_err(zpert))
            print(f"cycle {cyc + 1}: z_err {e:.3f} | model error from analysis {rec['analysis'][-1]:.4f} "
                  f"truth {rec['truth'][-1]:.4f} pert {rec['pert'][-1]:.4f}", flush=True)
    summ = {k: float(np.mean(v)) for k, v in rec.items() if k != "cycle"}
    late = [i for i, cy in enumerate(rec["cycle"]) if cy > a.n_cycles // 2]
    summ_late = {k: float(np.mean(np.array(v)[late])) for k, v in rec.items() if k != "cycle"}
    out = {"args": vars(a), "mean": summ, "mean_second_half": summ_late, "per_check": rec, "wall_s": time.time() - t0}
    Path(f"artifacts/experiments/da2d/i30_degradation_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, "all:", {k: round(v, 4) for k, v in summ.items()}, "| 2nd half:",
          {k: round(v, 4) for k, v in summ_late.items()}, f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
