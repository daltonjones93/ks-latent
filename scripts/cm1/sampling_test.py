"""Is 5-min output too coarse? (1-min test run of member 1's setup.)

Storm columns (cref > 10 dBZ), state compressed to the same 32 column PCs as
the locality test (PCA and per-variable std from members 1-7). For the 1-min
run we compute
  D(dt)  = E |x(t+dt) - x(t)|^2   (structure function, dt = 1..10 min)
           smooth evolution: D ~ dt^2; noise-like jumps: D(1 min) already ~ D(5 min)
  rho(dt) = corr( x(t+dt)-x(t), x(t+2dt)-x(t+dt) )   (successive-increment correlation)
           -0.5 is the signature of a component that decorrelates within one interval
and the same per-PC, to see which column patterns are under-sampled.
The 5-min numbers are also computed from the original 6 h member files for reference.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cm1_io as c

CTL = Path(sys.argv[1])  # .../cm1out_s.ctl of the 1-min run
K = 32
POOL = c.MEMBERS[:7]
runs = {r: c.load_run(r) for r in POOL}
std = np.array([np.concatenate([runs[r]["X"][:, v].ravel() for r in POOL]).std() for v in range(len(c.VARS))])

def cols(X):
    X = X / std[None, :, None, None]
    return X.transpose(0, 3, 1, 2).reshape(X.shape[0], X.shape[3], -1)

sub = np.concatenate([cols(runs[r]["X"])[:, ::4].reshape(-1, len(c.VARS) * 62) for r in POOL])
mu = sub.mean(0)
P = np.linalg.svd(sub - mu, full_matrices=False)[2][:K].T

def stats(p, act, step):
    """p: (T, NX, K) PCs at uniform spacing; step: interval in frames."""
    D = []
    rho_pc = np.zeros(K); n = 0
    a = p[step:] - p[:-step]                      # increments over `step`
    m = act[:-step]
    D = float((a ** 2).sum(-1)[m].mean())
    a1, a2 = a[:-step], a[step:]                  # successive increments
    mm = m[:-step] & act[2 * step:]
    rho = float((a1 * a2).sum(-1)[mm].sum() / np.sqrt((a1**2).sum(-1)[mm].sum() * (a2**2).sum(-1)[mm].sum()))
    rho_pc = ((a1 * a2)[mm].sum(0) / np.sqrt((a1**2)[mm].sum(0) * (a2**2)[mm].sum(0)))
    var_pc = (a ** 2)[m].mean(0)
    return D, rho, rho_pc, var_pc

d = c.load_grads_run(CTL)
t = d["t"]; dt_out = float(np.median(np.diff(t)))
assert abs(dt_out - 60) < 1e-6, f"expected 1-min output, got {dt_out}"
p = ((cols(d["X"]) - mu) @ P)
act = d["cref"] > 10
start = int(np.searchsorted(t, 1800))           # use t >= 30 min (convection present)
p, act = p[start:], act[start:]
res = {"t_range_min": [float(t[start] / 60), float(t[-1] / 60)], "active_cols_mean": float(act.sum(1).mean()),
       "by_dt_min": {}}
for step in [1, 2, 3, 5, 10]:
    D, rho, rho_pc, var_pc = stats(p, act, step)
    res["by_dt_min"][step] = {"D": D, "rho_successive": rho,
                              "rho_pc_first8": rho_pc[:8].round(2).tolist(),
                              "frac_var_in_pcs_with_rho_below_-0.3": float(var_pc[rho_pc < -0.3].sum() / var_pc.sum())}
D1, D5 = res["by_dt_min"][1]["D"], res["by_dt_min"][5]["D"]
res["D1_over_D5"] = D1 / D5
res["note"] = "D1/D5 = 0.04 for smooth (dt^2) evolution, 0.2 for random-walk, ~1 for white noise"
# reference: original 5-min member files over the same time window and over 6 h
ref = {}
for r in POOL:
    pr = (cols(runs[r]["X"]) - mu) @ P
    ar = runs[r]["cref"] > 10
    i0, i1 = 6, 19                                 # 30-90 min
    ref.setdefault("30-90min", []).append(stats(pr[i0:i1], ar[i0:i1], 1)[1])
    ref.setdefault("0-6h", []).append(stats(pr, ar, 1)[1])
res["original_5min_rho"] = {k: float(np.mean(v)) for k, v in ref.items()}
out = c.CACHE / "sampling_test.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
