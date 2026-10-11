"""Block I (2026-10-09): how Gaussian are the variables each filter works in?

Climatological marginals of the 2D KS field u (grid points) and of the per-patch latent
coordinates z (p7b_ks2d, 6 channels per site), from 20 well-separated snapshots of the 44^2 truth
(same trajectory as the DA runs). Reports skewness and excess kurtosis (0 for a Gaussian), and,
for the DA-relevant question, the same statistics of the PRIOR ENSEMBLE: forecast-ensemble
deviations from their mean, physical (decoded) vs latent, pooled over grid points / coordinates.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kurtosis, skew

from ks_latent.config import KSConfig2D
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d

import os  # model via KS_TAG (2026-10-10); default p7b_ks2d keeps the original output
TAG_ = os.environ.get("KS_TAG", "p7b_ks2d"); SUFFIX_ = "" if TAG_ == "p7b_ks2d" else f"_{TAG_}"
AE = f"artifacts/stage1_ae_kolmogorov_crop_{TAG_}.pt"; PR = f"artifacts/stage2_prop_kolmogorov_crop_{TAG_}.pt"
ck = torch.load(AE, map_location="cpu", weights_only=False)
patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
g = deployment_grid_2d(44.0, training_dx(ck["train_dataset"]), patch)
NX, ns = g.NX, g.n_sites
ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
std = float(ck["normalization_std"])
u0 = np.load("artifacts/cache/ks2d_spinup_L44_NX128_dt0.02_T200_seed0.npy")
cfg = KSConfig2D(Lx=44.0, Ly=44.0, NX=NX, NY=NX, dt=0.02, snapshot_every=250, spinup_time=200.0, seed=0)
U = integrate2d(u0, cfg, n_steps=20 * 250)[1:] / std  # 20 snapshots, 5 time units apart
with torch.no_grad():
    Z = ae.encode(torch.tensor(U.reshape(20, -1), dtype=torch.float32)).numpy()
c = Z.shape[1] // (ns * ns)
out = {}
def stats(name, x):
    x = np.asarray(x)
    out[name] = {"skew_abs_mean": float(np.mean(np.abs(skew(x, axis=0)))),
                 "exkurt_mean": float(np.mean(kurtosis(x, axis=0))),
                 "pooled_skew": float(skew(x.ravel())), "pooled_exkurt": float(kurtosis(x.ravel()))}
# climatology: pool sites (statistically homogeneous): u over grid points; z per channel over sites
stats("clim_u", U.reshape(-1, 1))
stats("clim_z", Z.reshape(20, ns * ns, c).reshape(-1, c))
# prior ensemble: 64 members started near truth, 5 propagator steps (one DA cycle)
gen = torch.Generator().manual_seed(0)
Zc = torch.tensor(Z[:4]).repeat_interleave(16, 0) + 0.1 * torch.randn(64, Z.shape[1], generator=gen)
with torch.no_grad():
    fc = forecast_ensemble(prop, Zc.clone(), Zc, 5, 0.02)
    Zf = fc.z_curr_final.numpy()
    Uf = ae.decode(fc.z_curr_final).numpy()
for k in range(4):
    zk = Zf[16 * k:16 * k + 16]; uk = Uf[16 * k:16 * k + 16]
    stats(f"prior_z_{k}", zk - zk.mean(0)); stats(f"prior_u_{k}", uk - uk.mean(0))
for kind in ("prior_z", "prior_u"):
    ks = [v for k, v in out.items() if k.startswith(kind + "_")]
    out[kind] = {m: float(np.mean([d[m] for d in ks])) for m in ks[0]}
    for k in [k for k in list(out) if k.startswith(kind + "_")]:
        del out[k]
Path(f"artifacts/experiments/da2d/gaussianity_L44{SUFFIX_}.json").write_text(json.dumps(out, indent=2))
for k, v in out.items():
    print(k, {m: round(x, 3) for m, x in v.items()})
