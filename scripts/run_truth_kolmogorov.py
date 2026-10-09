"""True-PDE Kaplan-Yorke dimension of Kolmogorov flow at Lx = scale*2pi, Ly = 2pi
(same grid spacing/nu/forcing as every dataset), one seed. Preemptable via the
Benettin checkpoint. Writes artifacts/experiments/crop/truth_a{alpha}_x{scale}_seed{seed}.json.
Usage: python scripts/run_truth_kolmogorov.py ALPHA SCALE SEED N_DIRECTIONS [SPINUP_TIME]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, math, sys, time
from pathlib import Path
from ks_latent.analysis.lyapunov import kaplan_yorke_dimension, lyapunov_spectrum_kolmogorov
from ks_latent.config import KolmogorovConfig
alpha, scale, seed, ndir = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spinup = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
# 2026-10-05: optional averaging window (argv[6] total_time, argv[7] warmup_time); defaults keep the
# original 40/20 and the original output name. Non-default windows get a _T<total> suffix.
total_time = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
warmup_time = float(sys.argv[7]) if len(sys.argv) > 7 else 20.0
# 2026-10-06 (task A3, docs/paper1_tasks.md): optional argv[8], comma-separated physical
# times (measured after warmup, same convention as `lyapunov_spectrum_kolmogorov`'s own
# `record_times` docstring) at which to snapshot the running D_KY -- lets a rerun's
# convergence be checked directly (has the running average actually flattened by the end
# of the averaging window?) instead of only ever seeing the final value, which is all the
# original 2x truths (56.84/38.68/53.89, 40-unit window) recorded.
record_times = [float(t) for t in sys.argv[8].split(",")] if len(sys.argv) > 8 else None
tag = f"truth_a{alpha:g}_x{scale}_seed{seed}".replace(".", "p") + ("" if total_time == 40.0 else f"_T{total_time:g}")
out = Path("artifacts/experiments/crop"); out.mkdir(parents=True, exist_ok=True)
cfg = KolmogorovConfig(Lx=2 * math.pi * scale, Ly=2 * math.pi, NX=128 * scale, NY=128, nu=0.01, alpha=alpha,
                       forcing_n=4, dt=0.01, spinup_time=spinup, seed=seed)
t0 = time.time()
res = lyapunov_spectrum_kolmogorov(cfg, n_directions=ndir, total_time=total_time, qr_interval=0.5, warmup_time=warmup_time,
                                   seed=seed, max_abs_state=500.0, checkpoint_path=str(out / f"{tag}.ckpt.npz"),
                                   checkpoint_every_time=4.0, record_times=record_times)
ex = res.exponents
dky = kaplan_yorke_dimension(ex)
running_dky = None
if res.partial_exponents is not None:
    running_dky = {str(t): kaplan_yorke_dimension(e) for t, e in sorted(res.partial_exponents.items())}
    print("running D_KY by time:", running_dky, flush=True)
(out / f"{tag}.json").write_text(json.dumps({
    "alpha": alpha, "scale": scale, "seed": seed, "n_directions": ndir, "spinup_time": spinup,
    "total_time": total_time, "warmup_time": warmup_time, "D_KY": dky,
    "lambda1": float(ex[0]), "n_positive": int((ex > 0).sum()), "exponents": [float(x) for x in ex],
    "running_D_KY_by_time": running_dky,
    "wall_s_this_session": time.time() - t0}, indent=2))
print("done", tag, dky, "n_pos", int((ex > 0).sum()), "of", ndir, flush=True)
if int((ex > 0).sum()) > 0.8 * ndir:
    print("WARNING: n_positive close to n_directions; D_KY may be truncated -- rerun with more directions", flush=True)
