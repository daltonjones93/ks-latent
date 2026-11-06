"""H1(a): advection extrapolation against the surrogate, on exactly the cases of a model's results file.

Reads the start times, lead count and test storms from a train_cm1.py results file (results.json or a
results_<tag>.json re-score), forecasts each case by advection extrapolation (`ks_latent.cm1.baselines`,
global and per-level shifts), and prints skill vs persistence (nMSE ratio; normalized units, all channels,
the whole domain, as `forecast_report` scores the model) next to the model's own skill from the same file.
  python scripts/cm1/advection_baseline.py --run ~/cm1_runs/full_v1_stagec \
      --results ~/cm1_runs/full_v1_stagec/results_h2_lead120.json --source cache:A+B+C --out h1a.json
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import member_files  # noqa: E402
from ks_latent.cm1.baselines import advection_forecast  # noqa: E402
from ks_latent.cm1.data import ColumnData, Normalizer, load_member  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402

LEADS_MIN = (5, 30, 60, 90, 120)


@dataclass
class AdvectionConfig:
    lag: int
    max_shift: int
    modes: tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (its normalizer.json)")
    ap.add_argument("--results", required=True, help="the model's results file: cases and model skill")
    ap.add_argument("--source", required=True)
    ap.add_argument("--lag", type=int, default=5, help="frames over which the motion is estimated")
    ap.add_argument("--max-shift", type=int, default=40, help="cells per lag")
    ap.add_argument("--modes", nargs="+", default=["global", "per_level"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    run = Path(a.run).expanduser()
    norm = Normalizer.from_dict(json.loads((run / "normalizer.json").read_text()))
    res = json.loads(Path(a.results).expanduser().read_text())
    fc_m = res["forecast"]
    t0s, n_lead = fc_m["t0s"], fc_m["n_lead"]
    names = res.get("eval_test") or res["split"]["test"]
    files = member_files(a.source)
    data = ColumnData([load_member(files[n]) for n in names], norm, torch.device("cpu"))
    Z = data.Z
    dt = data.dt
    rows = {m: [] for m in a.modes}
    pers, shifts = [], {m: [] for m in a.modes}
    for t0 in t0s:
        if t0 - a.lag < 0:
            raise ValueError(f"start frame {t0} < lag {a.lag}")
        for mi in range(len(names)):
            if t0 + n_lead >= data.T[mi]:
                continue
            u = data.full(mi, slice(t0 - a.lag, t0 + n_lead + 1)).double()        # (lag + 1 + L, C, NX)
            hist, tr = u[:a.lag + 1], u[a.lag + 1:]
            pers.append(((hist[-1][None] - tr) ** 2).mean(dim=(1, 2)).numpy())
            for mode in a.modes:
                f, sh = advection_forecast(hist, n_lead, Z, lag=a.lag, max_shift=a.max_shift, mode=mode)
                rows[mode].append(((f - tr) ** 2).mean(dim=(1, 2)).numpy())
                shifts[mode].append(sh)
    pe = np.mean(pers, 0)
    out = {"results": str(a.results), "t0s": t0s, "n_lead": n_lead, "members": names, "lag": a.lag,
           "nmse_pers": pe.tolist(), "modes": {}}
    lead_idx = {L: int(round(L * 60 / dt)) - 1 for L in LEADS_MIN if int(round(L * 60 / dt)) <= n_lead}
    model_skill = np.array(fc_m["nmse"]) / np.array(fc_m["nmse_pers"])
    hdr = "lead (min)            " + " ".join(f"{L:>7d}" for L in lead_idx)
    print(f"[H1a] {len(pers)} cases ({len(names)} storms x {len(t0s)} start times), skill = nMSE / persistence nMSE")
    print(hdr)
    print("model (results file)  " + " ".join(f"{model_skill[i]:7.3f}" for i in lead_idx.values()))
    for mode in a.modes:
        nm = np.mean(rows[mode], 0)
        sk = nm / pe
        sh = np.array(shifts[mode])
        out["modes"][mode] = {"nmse": nm.tolist(), "skill": sk.tolist(),
                              "shift_cells_per_lag_mean": float(sh.mean()), "shift_cells_per_lag_absmax": int(np.abs(sh).max())}
        print(f"advection {mode:11s} " + " ".join(f"{sk[i]:7.3f}" for i in lead_idx.values())
              + f"   (mean shift {sh.mean():+.2f} cells / {a.lag} frames)")
    Path(a.out).expanduser().write_text(json.dumps(out, indent=1))
    write_provenance(Path(a.out).expanduser(), config=AdvectionConfig(a.lag, a.max_shift, tuple(a.modes)), seed=0,
                     device="cpu", wall_time_s=time.time() - t_start, extra={"args": vars(a), "normalizer": norm.to_dict()})


if __name__ == "__main__":
    main()
