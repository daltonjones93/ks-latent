"""J17S (2026-10-10): truth Lyapunov spectrum / D_KY at 44^2 of a candidate truth (`--kind ganiso|aniso`,
`--level`), with the settings of the standard-KS truth run (G1: 150 directions, total 40, warmup 20, QR
every 1 time unit, dt 0.02, spinup 200, seed 0; standard KS gives D_KY 81.8). Writes
artifacts/experiments/da2d/j17s_lyap_<kind><level>.json."""
import argparse, json, time
from ks_latent.analysis.lyapunov import lyapunov_spectrum_ks2d
from ks_latent.config import KSConfig2D
p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["ganiso", "aniso"], required=True)
p.add_argument("--level", type=float, required=True)
p.add_argument("--n-directions", type=int, default=150)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
cfg = KSConfig2D(Lx=44., Ly=44., NX=128, NY=128, dt=0.02, spinup_time=200., seed=a.seed)
t0 = time.time()
r = lyapunov_spectrum_ks2d(cfg, n_directions=a.n_directions, total_time=40., qr_interval=1., warmup_time=20.,
                           seed=a.seed, max_abs_state=200., **{a.kind: a.level})
out = {"kind": a.kind, "level": a.level, "seed": a.seed, "n_directions": a.n_directions,
       "D_KY": r.kaplan_yorke_dimension, "lambda1": float(r.exponents[0]), "n_positive": int(r.n_positive),
       "standard_KS_D_KY": 81.77, "saturated": bool(r.kaplan_yorke_dimension > 0.95 * a.n_directions),
       "wall_s": time.time() - t0, "exponents": [float(x) for x in r.exponents]}
open(f"artifacts/experiments/da2d/j17s_lyap_{a.kind}{a.level:g}.json", "w").write(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "exponents"})
