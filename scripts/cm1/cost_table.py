"""N6 / G9: cost table. Wall time of the surrogate's forecasts and of a full latent DA cycle, against CM1.

Times, on the given device, after a warm-up:
  forecast   N members x 60 one-minute steps (one forecast hour), propagator only, and decoded at the end
  cycle      one DA cycle of the headline setup (5 propagator steps for N members + decoding + observation
             operators + per-site LETKF), averaged over --cycles cycles of run_twin on one truth
CM1's cost is not re-measured here: pass the measured core-minutes per simulated hour per member
(--cm1-core-min-per-hour; Cashew logs: 72-93 min per 6-h 2D member on one core, i.e. ~13-16).

  python scripts/cm1/cost_table.py --run ~/cm1_runs/full_v1_stagec --source cache:A+A2+B --out ~/cm1_da/cost_gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, _forecast, run_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-members", type=int, default=32)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cm1-core-min-per-hour", type=float, default=14.5)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: leave as is)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = get_device(a.device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
    ck = torch.load(run / "refl_head.pt", map_location=dev)
    head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev).eval()
    head.load_state_dict(ck["state_dict"])
    files = member_files(a.source)
    names = split["train"][: a.n_members]
    others = ColumnData([load_member(files[n]) for n in names], norm, dev)
    truth = ColumnData([load_member(files[split["test"][0]])], norm, dev)
    z_truth = encode_members(ae, truth)[0]
    t0 = int(round(3600 / others.dt))
    Z0 = torch.stack([z[t0] for z in encode_members(ae, others)])
    halo = cfg["stage_b"]["halo_sites"]
    res = {"device": str(dev), "n_members": len(names), "latent_shape": list(Z0.shape[1:]), "threads": torch.get_num_threads()}
    with torch.no_grad():
        _forecast(prop, Z0, z_truth, t0, 5, halo, 0.0, torch.Generator().manual_seed(0), hold=t0)   # warm-up
        ts, td = [], []
        for r in range(a.repeats):
            sync(dev); t = time.perf_counter()
            Z = _forecast(prop, Z0, z_truth, t0, 60, halo, 0.0, torch.Generator().manual_seed(r), hold=t0)
            sync(dev); ts.append(time.perf_counter() - t)
            t = time.perf_counter(); ae.decode(Z); sync(dev); td.append(time.perf_counter() - t)
    res["forecast_hour_s"] = float(np.median(ts))
    res["decode_ensemble_s"] = float(np.median(td))
    tc = TwinConfig(t0=t0, n_cycles=a.cycles, steps_per_cycle=5, halo=halo, model_noise=0.1, refl_every_cells=8,
                    vr_tilts=(0.5, 2.0, 5.0), boundary="persistence", letkf=LETKFConfig(loc_km=5.0, rtps=0.9))
    run_twin(ae, prop, truth, z_truth, Z0, norm, TwinConfig(**{**tc.__dict__, "n_cycles": 1}), refl_head=head)  # warm-up
    sync(dev); t = time.perf_counter()
    run_twin(ae, prop, truth, z_truth, Z0, norm, tc, refl_head=head)
    sync(dev)
    res["da_cycle_s"] = (time.perf_counter() - t) / a.cycles          # includes scoring, an upper bound
    res["da_3h_experiment_s"] = res["da_cycle_s"] * 36
    cm1_s = a.cm1_core_min_per_hour * 60 * len(names)                   # one forecast hour, N members, core-seconds
    res["cm1_forecast_hour_core_s"] = cm1_s
    res["speedup_forecast_vs_cm1_core"] = cm1_s / res["forecast_hour_s"]
    (out / "cost.json").write_text(json.dumps(res, indent=1))
    write_provenance(out / "cost.json", config=tc, seed=0, device=str(dev), wall_time_s=0.0, extra={"args": vars(a)})
    print(f"[cost] {json.dumps(res)}")


if __name__ == "__main__":
    main()
