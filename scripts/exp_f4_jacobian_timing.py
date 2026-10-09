"""Task F4 (docs/full_paper_tasks.md): cost of the latent observation Jacobian, dense reverse mode
vs the coloured forward-mode construction, at the corrected 2D KS deployment.

Supersedes exp_t1_jacobian_timing.py, which sized the domain from the crop-training grid and
scaled the observation stride with L. Here: deployment grid at the training spacing
(`deployment_grid_2d`), fixed observation stride 8 (2.75 length units, as in every DA run),
float32 model as in the filter (h returns float64, as in cycling.py), z = encoding-sized random
state. Per size: wall time (median of 3 after a warm-up) of
  * jacrev  -- dense (m, d) by reverse mode, chunked; run up to 132^2 (memory beyond),
  * coloured -- the compact (m, c) form the per-site filter uses: c forward-mode JVPs,
and the max |difference| between the two where both are computed. 4 threads.
Writes artifacts/experiments/da2d/jacobian_timing_F4.json.
"""
import json
import statistics as st
import time

import torch
from torch.func import jacrev

from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import coloured_obs_jacobian_2d_compact, obs_site_map_2d
from ks_latent.models import load_autoencoder_checkpoint_resized_2d

torch.set_num_threads(4)
AE = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"
STRIDE = 8
SIZES = [44, 66, 88, 132, 220, 330, 440]
JACREV_MAX = 132
ck = torch.load(AE, map_location="cpu", weights_only=False)
patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
dx = training_dx(ck["train_dataset"])


def timed(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return st.median(ts), out


res = []
for L in SIZES:
    g = deployment_grid_2d(float(L), dx, patch)
    ae, cfg, _ = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=g.n_sites, new_n_sites_y=g.n_sites,
                                                        new_NX=g.NX, new_NY=g.NX)
    ae.eval()
    c = cfg.local_channels
    d = g.n_sites ** 2 * c
    site = obs_site_map_2d(g.NX, g.NX, g.n_sites, g.n_sites, STRIDE, STRIDE)
    m = site.shape[0]

    def h(z):
        return ae.decode(z.to(torch.float32).unsqueeze(0))[0].reshape(g.NX, g.NX)[::STRIDE, ::STRIDE].reshape(-1).to(
            torch.float64)

    z = torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t_col, Jc = timed(lambda: coloured_obs_jacobian_2d_compact(h, z, site, c).detach())
    rec = {"L": L, "NX": g.NX, "n_sites": g.n_sites, "d": d, "m": m, "channels": c,
           "t_coloured_s": t_col, "t_jacrev_s": None, "max_abs_diff": None}
    if L <= JACREV_MAX:
        t_rev, J = timed(lambda: jacrev(h, chunk_size=256)(z).detach())
        cols = site.unsqueeze(1) * c + torch.arange(c).unsqueeze(0)
        Jfromc = torch.zeros_like(J)
        Jfromc[torch.arange(m).unsqueeze(1), cols] = Jc
        rec.update(t_jacrev_s=t_rev, max_abs_diff=float((J - Jfromc).abs().max()),
                   max_abs=float(J.abs().max()))
    res.append(rec)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    del ae
with open("artifacts/experiments/da2d/jacobian_timing_F4.json", "w") as f:
    json.dump(res, f, indent=2)
