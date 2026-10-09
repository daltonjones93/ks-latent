"""P7: zero-shot evaluation of a 2D-crop-trained 2D KS model on periodic square boxes
of side scale*44 (8*scale sites per axis). IC normalized with the wide training set's
stats (stored in the checkpoint). Lyapunov settings as exp_g3/exp_crop_eval."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.config import KSConfig2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import spinup2d

TRUTH = {1.0: 81.77, 1.5: 192.5}  # direct truth runs (P7 step 1, G1)
p = argparse.ArgumentParser()
p.add_argument("--ae-ckpt", required=True); p.add_argument("--prop-ckpt", required=True)
p.add_argument("--scale", type=float, required=True); p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", required=True)
p.add_argument("--n-directions", type=int, default=None, help="default: all d_latent")
a = p.parse_args()
ck = torch.load(a.ae_ckpt, map_location="cpu", weights_only=False)
mean, std = ck["normalization_mean"], ck["normalization_std"]
ns, N = int(round(8 * a.scale)), int(round(128 * a.scale))
ae = load_autoencoder_checkpoint_resized_2d(a.ae_ckpt, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=N, new_NY=N)[0].eval()
prop, pcfg, _ = load_propagator_checkpoint_resized_2d(a.prop_ckpt, new_n_sites_x=ns, new_n_sites_y=ns)
prop.eval()
cfg = KSConfig2D(Lx=44.0 * a.scale, Ly=44.0 * a.scale, NX=N, NY=N, spinup_time=200.0, seed=a.seed)
u0 = (spinup2d(cfg, np.random.default_rng(a.seed)) - mean) / std
t0 = time.time()
with torch.no_grad():
    z0 = ae.encode(torch.tensor(u0.reshape(1, -1), dtype=torch.float32))[0].numpy().astype(np.float64)
res = lyapunov_spectrum_latent_propagator(prop, np.stack([z0, z0]), mode="single_state",
                                          n_directions=a.n_directions or pcfg.d_latent,
                                          n_steps=2000, qr_every=10, dt_snap=cfg.dt * cfg.snapshot_every,
                                          warmup_steps=200, seed=a.seed, max_abs_state=1e3)
truth = TRUTH.get(a.scale)
out = {"system": "ks2d", "scale": a.scale, "seed": a.seed, "d_latent": pcfg.d_latent,
       "n_directions": a.n_directions or pcfg.d_latent,
       "model_D_KY": res.kaplan_yorke_dimension, "model_lambda1": float(res.exponents[0]),
       "model_n_positive": int(res.n_positive), "true_D_KY": truth,
       "rel_err_pct": abs(res.kaplan_yorke_dimension - truth) / truth * 100 if truth else None,
       "ae_ckpt": a.ae_ckpt, "prop_ckpt": a.prop_ckpt, "ic": "normalized (wide-set stats)", "wall_s": time.time() - t0}
Path("artifacts/experiments/crop").mkdir(parents=True, exist_ok=True)
Path(f"artifacts/experiments/crop/{a.tag}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2), flush=True)
