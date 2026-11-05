"""B1: is the Cashew build the same model as the one that made the original members?

Compares a Cashew member run from member 1's exact setup (`mem000`: same sounding, same namelist,
same random noise, 1-min output) with the original `run_e1` (5-min output) at matching times.
Same model => identical at t = 0, then tiny build-level differences (compiler, flags) that grow
the way chaotic member differences grow, never a different storm. Reference: the difference between
two *different* original members at the same time.

Run where both are available:
  python validate_cashew_build.py --mem000 DIR/members/mem000/cm1out_s.ctl --run-e1 run_e1_cols.npz \
      --out validate.json
`--run-e1` is the cached transform of the original run (`scripts/cm1/cm1_io.load_run`, float32).
"""
import argparse
import json

import numpy as np

from ks_latent.cm1.data import VARS
from ks_latent.cm1.io import load_grads_run


def front_km(th_sfc, thresh=-1.0, dx=0.25):
    cold = np.where(th_sfc < thresh)[0]
    return float((cold.max() + 0.5) * dx) if cold.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem000", required=True)
    ap.add_argument("--run-e1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new = load_grads_run(a.mem000)
    old = dict(np.load(a.run_e1))
    t_old, t_new = old["t"], new["t"]
    idx = [int(np.where(np.isclose(t_new, t))[0][0]) for t in t_old if np.isclose(t_new, t).any()]
    if len(idx) != len(t_old):
        raise ValueError(f"only {len(idx)} of {len(t_old)} original times found in mem000")
    Xo, Xn = old["X"].astype(np.float64), new["X"][idx].astype(np.float64)  # (T, V, Z, NX)
    scale = np.sqrt((Xo ** 2).mean(axis=(0, 2, 3)))                        # per-variable RMS
    d = (Xn - Xo) / scale[None, :, None, None]
    ith = VARS.index("th")
    out = {"t_s": t_old.tolist(),
           "bitwise_equal_t0": bool(np.array_equal(Xn[0], Xo[0])),
           "nmse": (d ** 2).mean(axis=(1, 2, 3)).tolist(),
           "rms_phys_by_var": {v: (np.sqrt(((Xn - Xo)[:, i] ** 2).mean(axis=(1, 2)))).tolist() for i, v in enumerate(VARS)},
           "max_abs_th_K": np.abs((Xn - Xo)[:, ith]).max(axis=(1, 2)).tolist(),
           "front_km_new": [front_km(Xn[k, ith, 0]) for k in range(len(idx))],
           "front_km_old": [front_km(Xo[k, ith, 0]) for k in range(len(idx))],
           "rain_proxy_new": new["cref"][idx].mean(axis=1).tolist(),
           "rain_proxy_old": old["cref"].mean(axis=1).tolist(),
           "scale": dict(zip(VARS, scale.tolist()))}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for k in (0, 1, 2, 6, 12, 24, 36, 72):
        print(f"t={t_old[k] / 60:5.0f} min  nMSE {out['nmse'][k]:.2e}  max|dtheta| {out['max_abs_th_K'][k]:.3g} K  "
              f"front {out['front_km_new'][k]:.1f} vs {out['front_km_old'][k]:.1f} km")
    print("bitwise equal at t=0:", out["bitwise_equal_t0"])


if __name__ == "__main__":
    main()
