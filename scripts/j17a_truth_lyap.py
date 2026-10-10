"""J17A A3 (2026-10-10): truth Lyapunov spectrum / D_KY of u_t = KS(u) - beta P[u'^3] at 44^2, with the
settings of the standard-KS truth run behind exp_crop_eval_ks2d.py's TRUTH table (G1: total 40, warmup 20,
QR every 1 time unit, dt 0.02, spinup 200, seed 0). Writes artifacts/experiments/da2d/j17a_truth_lyap_beta<b>.json."""
import argparse, json, time
import numpy as np
from ks_latent.analysis.lyapunov import lyapunov_spectrum_ks2d
from ks_latent.config import KSConfig2D
p = argparse.ArgumentParser()
p.add_argument("--cubic", type=float, required=True)
p.add_argument("--n-directions", type=int, default=150)  # as the standard-KS G1 run (g1_2dks_transfer_native44_n150.json)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, spinup_time=200., seed=a.seed)
t0 = time.time()
r = lyapunov_spectrum_ks2d(cfg, n_directions=a.n_directions, total_time=40., qr_interval=1., warmup_time=20.,
                           seed=a.seed, max_abs_state=200., cubic=a.cubic)
out = {"cubic": a.cubic, "seed": a.seed, "n_directions": a.n_directions, "D_KY": r.kaplan_yorke_dimension,
       "lambda1": float(r.exponents[0]), "n_positive": int(r.n_positive), "standard_KS_D_KY": 81.77,
       "saturated": bool(r.kaplan_yorke_dimension > 0.95 * a.n_directions), "wall_s": time.time() - t0,
       "exponents": [float(x) for x in r.exponents]}
open(f"artifacts/experiments/da2d/j17a_truth_lyap_beta{a.cubic:g}.json", "w").write(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "exponents"})
