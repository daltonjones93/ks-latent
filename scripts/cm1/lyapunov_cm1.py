"""L1 and L5: finite-time Lyapunov spectra of the CM1 latent propagator, and perturbation spreading.

For each truth storm (encoded CM1 trajectory) and window, the full finite-time spectrum of the latent map
(boundary sites held), in two linearizations (`ks_latent.cm1.lyapunov`): along the propagator's own run
("own") and along the encoded CM1 trajectory ("truth"). Reported per window and mode, over storms: lambda_1
(per min), the error doubling time ln 2 / lambda_1, the number of positive exponents, the Kaplan-Yorke
dimension, plus running estimates for a convergence check. L5: a one-site perturbation pushed through the
tangent map along the truth, in the storm (the cold-pool minimum) and in the quiescent far field; front speed
in km/min against the data-only reach (8-12 km per 5 min, i.e. 1.6-2.4 km/min) and the network's
numerical maximum (reach sites x site width per step).
  python scripts/cm1/lyapunov_cm1.py --run ~/cm1_runs/full_v1_stagec --source cache:A+B+C --out ~/cm1_da/l1
"""
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
from ks_latent.cm1.lyapunov import latent_spectrum, spreading  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truths", nargs="*", default=[],
                    help="default: the first 3 ens_A2 and the first 3 ens_B test storms of the run")
    ap.add_argument("--windows", nargs="+", default=["60:60", "120:60"], help="start:length in minutes")
    ap.add_argument("--warmup-min", type=float, default=16.0)
    ap.add_argument("--modes", nargs="+", default=["truth", "own"], choices=["truth", "own"])
    ap.add_argument("--n-directions", type=int, default=0, help="0: the full spectrum")
    ap.add_argument("--qr-every", type=int, default=2)
    ap.add_argument("--spread-steps", type=int, default=30)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    if prop.cfg.noise_channels:
        raise ValueError("use a deterministic propagator (noise_channels = 0)")
    halo, patch = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    site_km = patch * 0.25
    files = member_files(a.source)
    truths = a.truths or ([n for n in split["test"] if n.startswith("ens_A2_")][:3]
                          + [n for n in split["test"] if n.startswith("ens_B_")][:3])
    seen = set(split["train"]) | set(split["val"])
    if not truths or any(t in seen for t in truths):
        raise ValueError(f"truths must be unseen storms: {truths}")
    res = {"args": vars(a), "truths": truths, "spectra": [], "spreading": []}
    for tn in truths:
        d = ColumnData([load_member(files[tn])], norm, dev)
        z = encode_members(ae, d)[0].double()                     # (T, P, c); float64 for the tangent map
        prop_d = prop.double()
        dt_min = d.dt / 60.0
        warm = int(round(a.warmup_min / dt_min)) // a.qr_every * a.qr_every
        for w in a.windows:
            w0, wl = (float(v) for v in w.split(":"))
            n = int(round(wl / dt_min)) // a.qr_every * a.qr_every
            t0 = int(round(w0 / dt_min)) - warm
            for mode in a.modes:
                tc = time.time()
                r = latent_spectrum(prop_d, z, t0, n, halo, mode=mode, warmup=warm,
                                    n_directions=a.n_directions or None, qr_every=a.qr_every, dt_min=dt_min,
                                    record_steps=[k for k in (n // 4, n // 2, n) if k % a.qr_every == 0 and k > 0])
                ex = r.exponents
                rec = {"truth": tn, "window": w, "mode": mode, "lambda1_per_min": float(ex[0]),
                       "doubling_min": float(np.log(2) / ex[0]) if ex[0] > 0 else float("inf"),
                       "n_positive": r.n_positive, "d_ky": r.kaplan_yorke_dimension, "dim": int(ex.size),
                       "sum": float(ex.sum()), "top": ex[:300].tolist(),
                       "running_lambda1": {str(k): float(v[0]) for k, v in (r.partial_exponents or {}).items()},
                       "running_n_positive": {str(k): int((v > 0).sum()) for k, v in (r.partial_exponents or {}).items()}}
                res["spectra"].append(rec)
                print(f"[L1] {tn} window {w} {mode:5s}: lambda1 {ex[0]:.4f}/min (doubling {rec['doubling_min']:.1f} min), "
                      f"{r.n_positive} positive of {ex.size}, D_KY {r.kaplan_yorke_dimension:.1f}  [{time.time() - tc:.0f}s]", flush=True)
                (out / "l1.json").write_text(json.dumps(res, indent=1))
        # L5: spreading in the storm (surface cold-pool minimum) and in the far field
        ith = norm.vars.index("th") * d.Z
        t5 = int(round(float(a.windows[0].split(":")[0]) / dt_min))
        th_sfc = d.full(0, t5, device="cpu")[0, ith].numpy()
        storm_site = int(np.argmin(th_sfc) // patch)
        P = z.shape[1]
        storm_site = min(max(storm_site, halo + 5), P - halo - 6)
        for label, site in (("storm", storm_site), ("far_field", halo + 30)):
            s = spreading(prop_d, z, t5, a.spread_steps, halo, site)
            s.update({"truth": tn, "where": label,
                      "speed_km_per_min": {k: v * site_km / dt_min for k, v in s["speed_sites_per_step"].items()}})
            res["spreading"].append(s)
            print(f"[L5] {tn} {label:9s} site {site}: front speed (km/min) " +
                  ", ".join(f"thr {k}: {v:.2f}" for k, v in s["speed_km_per_min"].items()), flush=True)
        (out / "l1.json").write_text(json.dumps(res, indent=1))
    # summary over storms
    print("\n[L1] mean +- sd over storms:")
    summ = {}
    for w in a.windows:
        for mode in a.modes:
            rs = [r for r in res["spectra"] if r["window"] == w and r["mode"] == mode]
            g = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
            l1, npos, dky = g("lambda1_per_min"), g("n_positive"), g("d_ky")
            summ[f"{w}_{mode}"] = {"lambda1": l1, "n_positive": npos, "d_ky": dky}
            print(f"   window {w:7s} {mode:5s}: lambda1 {l1[0]:.4f} +- {l1[1]:.4f}/min (doubling {np.log(2) / l1[0]:.1f} min), "
                  f"positive {npos[0]:.0f} +- {npos[1]:.0f}, D_KY {dky[0]:.0f} +- {dky[1]:.0f}")
    for where in ("storm", "far_field"):
        ss = [s["speed_km_per_min"]["0.001"] for s in res["spreading"] if s["where"] == where]
        print(f"[L5] {where:9s}: front speed at threshold 1e-3: {np.nanmean(ss):.2f} +- {np.nanstd(ss):.2f} km/min "
              f"(data-only reach 1.6-2.4; network maximum {prop.cfg.reach_sites * site_km / dt_min:.1f})")
    res["summary"] = summ
    (out / "l1.json").write_text(json.dumps(res, indent=1))
    write_provenance(out / "l1.json", config=prop.cfg, seed=0, device=str(dev), wall_time_s=time.time() - t_start,
                     extra={"args": vars(a)})
    print(f"[L1] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
