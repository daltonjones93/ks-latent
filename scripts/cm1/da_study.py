"""G1/G2/G2b/G3: the latent DA study over many truths, with equation-free baselines and error bars.

For each truth (members the model never saw, e.g. ens_A2 for a model trained on ens_A) and each
observation-noise seed, runs the same observation set with every --modes entry:
  latent          the latent LETKF with the learned surrogate (the method)
  physical        same surrogate forecasts, LETKF on the decoded columns, re-encoded (G2b, latent vs physical)
  enoi_surrogate  EnOI: surrogate forecast of the analysis mean + static latent anomalies of the training members (G2a)
  enoi_phys       classic EnOI, no machine learning: persistence of the physical analysis + static anomalies of
                  the training archive's true columns, LETKF per cell (G2a/G2b: the equation-free, ML-free baseline)
The EnOI static ensembles are a climatological archive, as in operational EnOI: every training member at
frames within --enoi-window-min of the analysis time, every --enoi-step-min, with the covariance scaled by
--enoi-alpha and no RTPS. (A first version with only the 7-28 members at the exact time failed: a rank-N
covariance from storms at different positions plus a background that never moves let spurious increments
accumulate, nMSE > 1.)
plus the no-DA ensemble (surrogate, no analysis) and the autoencoder floor. Scores are second-half
means; across truths it reports mean +- 95% CI (normal approximation, 1.96 sd / sqrt(n); n = truths,
seeds averaged within a truth) and paired differences against no DA and against `latent`. With
--forecast-from, free forecasts from the analyses after those cycles are scored per lead (G3), with persistence
of the analysis and reflectivity >= 35 dBZ FSS. G2b baselines (same LETKF and observations, other forecast
models): `persist` (members do not evolve; spread from --model-noise) and `analog` (local nearest-neighbour
forecasts from the training storms' latent archive, `ks_latent.cm1.analog`).

  python scripts/cm1/da_study.py --run ~/cm1_runs/d34_c32_kuniform --source cache:~/cm1_cache+~/cm1_cache_A2 \\
      --truth-prefix ens_A2_ --n-truths 10 --seeds 0 1 2 --out ~/cm1_da/g1_v1
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
from ks_latent.cm1.data import VARS, ColumnData, _natural_key, load_member  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, run_twin, summarize_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402

MODES = {"latent": {}, "physical": {"analysis_space": "physical"},
         "enoi_surrogate": {"ensemble_mode": "enoi"},
         "enoi_phys": {"ensemble_mode": "enoi_phys", "forecast_model": "persistence", "model_noise": 0.0},
         "obs_only": {"ensemble_mode": "oi_clim", "forecast_model": "persistence", "model_noise": 0.0},
         "persist": {"forecast_model": "persistence"},     # G2b: same LETKF, members do not evolve (needs noise)
         "analog": {"forecast_model": "analog"}}           # G2b: same LETKF, local analog forecasts (ks_latent.cm1.analog)
ENOI_MODES = ("enoi_surrogate", "enoi_phys", "obs_only")    # static-archive ensembles: covariance scaled by --enoi-alpha, no RTPS
METRICS = ("nmse_mean_dec", "th_obs_rmse", "th_obs_spread", "front_abs_disp_km_median", "refl_echo_rmse",
           "vr_obs_rmse")


def compact(s: dict) -> dict:
    out = {}
    for stage in ("prior", "analysis"):
        if stage in s:
            d = {k: s[stage][k] for k in METRICS if k in s[stage]}
            d.update({f"rmse_{v}": x for v, x in s[stage]["rmse_phys"].items()})
            for k in ("rank_hist", "rank_hist_pert", "ranks_cycles", "ranks_pert_cycles"):   # claim 2
                if stage == "prior" and k in s[stage]:
                    d[k] = s[stage][k]
            out[stage] = d
    return out


def ci(x) -> dict:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    h = 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "ci95": float(h), "n": int(x.size)}


def forecast_curve(r: dict) -> dict:
    """Mean over the forecast_from cycles of the per-lead forecast scores."""
    fcs = [c["forecast"] for c in r["cycles"] if "forecast" in c]
    if not fcs:
        return {}
    L = min(len(f) for f in fcs)
    keys = [k for k in fcs[0][0] if k != "lead_frames"]
    return {k: np.nanmean([[abs(f[j][k]) if k.startswith("front_disp_km") else f[j][k] for j in range(L)] for f in fcs], 0).tolist()
            for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truths", nargs="*", default=[])
    ap.add_argument("--truth-prefix", default="", help="add the first --n-truths members whose name starts with this")
    ap.add_argument("--n-truths", type=int, default=10)
    ap.add_argument("--include-test", action="store_true", help="also use the run's own test members as truths")
    ap.add_argument("--tuning", action="store_true",
                    help="tune DA settings on the run's VALIDATION storms (used only for checkpoint selection, never "
                         "trained on), so the test storms never influence a setting; all truths must be validation storms")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--obs-set", default="th+refl+vr")
    ap.add_argument("--obs-km", type=float, default=2.0)
    ap.add_argument("--sigma-obs", type=float, default=0.5)
    ap.add_argument("--loc-km", type=float, default=5.0)
    ap.add_argument("--rtps", type=float, default=0.9)
    ap.add_argument("--model-noise", type=float, default=0.05)
    ap.add_argument("--refl-km", type=float, default=2.0)
    ap.add_argument("--sigma-refl", type=float, default=5.0)
    ap.add_argument("--vr-tilts", type=float, nargs="+", default=[0.5, 2.0, 5.0])
    ap.add_argument("--sigma-vr", type=float, default=2.0)
    ap.add_argument("--radar-x-km", type=float, default=330.0)
    ap.add_argument("--boundary", choices=["truth", "persistence"], default="truth",
                    help="twin halo: the truth every step, or held at the truth at t0 (H0: no future information)")
    ap.add_argument("--n-members", type=int, default=0)
    ap.add_argument("--random-members", action="store_true",
                    help="draw the --n-members ensemble members at random (seed 0) from the training members, "
                         "rather than the first ones in name order (which favours one ensemble)")
    ap.add_argument("--t0-min", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--cycle-min", type=float, default=5.0)
    ap.add_argument("--forecast-from", type=int, nargs="*", default=[11, 17, 23, 29],
                    help="cycle indices after which to run free forecasts (default: analyses at 2, 2.5, 3, 3.5 h)")
    ap.add_argument("--forecast-minutes", type=float, default=60.0)
    ap.add_argument("--enoi-window-min", type=float, default=30.0,
                    help="EnOI static ensemble: training members at frames within +- this of the analysis time")
    ap.add_argument("--enoi-step-min", type=float, default=10.0)
    ap.add_argument("--enoi-alpha", type=float, default=0.5, help="EnOI covariance scaling (alpha < 1)")
    ap.add_argument("--analog-k", type=int, default=10, help="G2b analog: neighbours")
    ap.add_argument("--analog-radius", type=int, default=1, help="G2b analog: patch half-width in sites")
    ap.add_argument("--analog-mean", action="store_true", help="G2b analog: mean of the k increments (default: one drawn)")
    ap.add_argument("--analog-members", type=int, default=0, help="G2b analog: archive from the first n training storms (0: all)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    refl_head = None
    if "refl" in a.obs_set:
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        refl_head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev)
        refl_head.load_state_dict(ck["state_dict"]); refl_head.eval()
    files = member_files(a.source)
    if a.random_members and a.n_members:
        ens_names = sorted(np.random.default_rng(0).choice(split["train"], a.n_members, replace=False).tolist(),
                           key=_natural_key)
    else:
        ens_names = split["train"][: a.n_members or None]
    seen = set(split["train"]) | set(split["val"])
    truths = list(a.truths)
    if a.truth_prefix:
        truths += sorted([n for n in files if n.startswith(a.truth_prefix)], key=_natural_key)[: a.n_truths]
    if a.include_test:
        truths += split["test"]
    if a.tuning:
        truths = truths or list(split["val"])
        bad = [t for t in truths if t not in split["val"]]
        if bad or not truths:
            raise ValueError(f"--tuning truths must be validation storms (got {bad or 'none'})")
        print(f"[study] TUNING on validation storms {truths}: not for reporting as test results")
    else:
        bad = [t for t in truths if t in seen]
        if bad or not truths:
            raise ValueError(f"truths must be members the model never saw (got {bad or 'none'})")
    stoch = getattr(prop, "cfg", None) is not None and prop.cfg.noise_channels > 0
    print(f"[study] {run.name}: {len(truths)} truths, seeds {a.seeds}, modes {a.modes}, {len(ens_names)} members, "
          f"model noise {a.model_noise}, {'stochastic' if stoch else 'deterministic'} propagator")
    others = ColumnData([load_member(files[n]) for n in ens_names], norm, dev)
    dt = others.dt
    t0, spc = int(round(a.t0_min * 60 / dt)), int(round(a.cycle_min * 60 / dt))
    n_cycles = int(round(a.hours * 60 / a.cycle_min))
    z_others = encode_members(ae, others)                            # for Z0 and the EnOI static anomalies
    static_phys = lambda t: torch.stack([others.full(m, f, device=dev)[0] for m in range(len(ens_names)) for f in frames(t)])
    Z0 = torch.stack([z[t0] for z in z_others])
    W, S = int(round(a.enoi_window_min * 60 / dt)), max(1, int(round(a.enoi_step_min * 60 / dt)))
    Tm = min(others.T)
    frames = lambda t: [f for f in range(t - W, t + W + 1, S) if 0 <= f < Tm]
    static = lambda t: torch.stack([z[f] for z in z_others for f in frames(t)])
    base = dict(t0=t0, n_cycles=n_cycles, steps_per_cycle=spc, sigma_obs=a.sigma_obs, halo=cfg["stage_b"]["halo_sites"],
                obs_every_cells=int(round(a.obs_km / 0.25)), model_noise=a.model_noise,
                assim_theta=("th" in a.obs_set or "mesonet" in a.obs_set),
                surface_vars=("th", "u", "prs", "qv") if "mesonet" in a.obs_set else ("th",),
                refl_every_cells=int(round(a.refl_km / 0.25)) if "refl" in a.obs_set else 0, sigma_refl=a.sigma_refl,
                vr_tilts=tuple(a.vr_tilts) if "vr" in a.obs_set else (), sigma_vr=a.sigma_vr, radar_x_km=a.radar_x_km,
                forecast_from=tuple(a.forecast_from), forecast_frames=int(round(a.forecast_minutes * 60 / dt)),
                letkf=LETKFConfig(loc_km=a.loc_km, rtps=a.rtps), boundary=a.boundary)
    analog = None
    if "analog" in a.modes:   # archive: every training storm the surrogate was trained on (same data budget)
        from ks_latent.cm1.analog import AnalogConfig, AnalogPropagator
        names = split["train"][: a.analog_members or None]
        zl = []
        for n in names:
            zl += [z.cpu() for z in encode_members(ae, ColumnData([load_member(files[n])], norm, dev))]
        analog = AnalogPropagator(zl, AnalogConfig(radius=a.analog_radius, k=a.analog_k, stochastic=not a.analog_mean), dev)
        del zl
        print(f"[study] analog archive: {len(names)} training storms, {len(analog.keys)} patches "
              f"(radius {a.analog_radius}, k {a.analog_k}, {'mean' if a.analog_mean else 'one drawn'})", flush=True)
    rows = {}      # truth -> {"free": compact, "floor": x, mode: [compact per seed], "fc_" + mode: [curve per seed]}
    for tn in truths:
        truth = ColumnData([load_member(files[tn])], norm, dev)
        if truth.dt != dt:
            raise ValueError(f"{tn}: frame spacing differs from the ensemble's")
        z_truth = encode_members(ae, truth)[0]
        with torch.no_grad():
            floor = [float(((ae(u)[0] - u) ** 2).mean()) for u in
                     (truth.full(0, t0 + (k + 1) * spc, device=dev) for k in range(n_cycles // 2, n_cycles))]
        free = run_twin(ae, prop, truth, z_truth, Z0, norm, TwinConfig(**{**base, "model_noise": 0.0}),
                        assimilate=False, refl_head=refl_head)
        rows[tn] = {"floor": float(np.mean(floor)), "free": compact(summarize_twin(free)), "fc_free": forecast_curve(free)}
        for mode in a.modes:
            rows[tn][mode], rows[tn]["fc_" + mode] = [], []
            for sd in a.seeds:
                kw = {**base, **MODES[mode], "seed": sd}
                if mode in ENOI_MODES:
                    kw["letkf"] = LETKFConfig(loc_km=a.loc_km, inflation=a.enoi_alpha, rtps=0.0)
                r = run_twin(ae, analog if mode == "analog" else prop, truth, z_truth, Z0, norm, TwinConfig(**kw),
                             refl_head=refl_head, static_ens=static, static_phys=static_phys)
                rows[tn][mode].append(compact(summarize_twin(r)))
                rows[tn]["fc_" + mode].append(forecast_curve(r))
        fr = rows[tn]["free"]["prior"]["nmse_mean_dec"]
        print(f"[study] {tn}: no DA {fr:.3f}  " + "  ".join(
            f"{m} {np.mean([x['prior']['nmse_mean_dec'] for x in rows[tn][m]]):.3f}" for m in a.modes)
            + f"  floor {rows[tn]['floor']:.3f}  [{time.time() - t_start:.0f}s]", flush=True)
    # aggregate over truths (seed-averaged within a truth)
    agg = {"n_truths": len(truths), "truths": truths, "seeds": a.seeds, "modes": {}}
    per_truth = lambda m, st, k: [np.mean([x[st][k] for x in rows[t][m] if k in x[st]]) for t in truths]
    free_of = lambda st, k: [rows[t]["free"]["prior"][k] for t in truths]
    keys = ["nmse_mean_dec", "front_abs_disp_km_median", "th_obs_rmse", "refl_echo_rmse", "vr_obs_rmse",
            "th_obs_spread"] + [f"rmse_{v}" for v in norm.vars]
    for m in a.modes:
        agg["modes"][m] = {}
        for st in ("prior", "analysis"):
            d = {}
            for k in keys:
                if k not in rows[truths[0]][m][0][st]:
                    continue
                x = np.array(per_truth(m, st, k))
                d[k] = ci(x)
                if k in rows[truths[0]]["free"]["prior"]:
                    d[k + "_minus_noDA"] = ci(x - np.array(free_of(st, k)))
                if m != "latent" and "latent" in a.modes and k in rows[truths[0]]["latent"][0][st]:
                    d[k + "_minus_latent"] = ci(x - np.array(per_truth("latent", st, k)))
            agg["modes"][m][st] = d
        fcs = [np.mean([np.array(c["nmse_mean_dec"]) for c in rows[t]["fc_" + m] if c], 0) for t in truths
               if rows[t]["fc_" + m] and rows[t]["fc_" + m][0]]
        if fcs:
            agg["modes"][m]["forecast_nmse_by_lead"] = np.mean(fcs, 0).tolist()
            # G3: every per-lead forecast score (incl. persistence of the analysis and reflectivity FSS), mean over
            # truths of the seed means; FSS is nan where neither field has echoes >= 35 dBZ (nan-means)
            fk = list(rows[truths[0]]["fc_" + m][0])
            agg["modes"][m]["forecast_by_lead"] = {k: np.nanmean([np.nanmean([np.array(c[k]) for c in rows[t]["fc_" + m]], 0)
                                                                 for t in truths], 0).tolist() for k in fk}
    agg["free"] = {k: ci(free_of("prior", k)) for k in keys if k in rows[truths[0]]["free"]["prior"]}
    ffc = [np.array(rows[t]["fc_free"]["nmse_mean_dec"]) for t in truths if rows[t]["fc_free"]]
    if ffc:
        agg["free"]["forecast_nmse_by_lead"] = np.mean(ffc, 0).tolist()
        agg["free"]["forecast_by_lead"] = {k: np.nanmean([np.array(rows[t]["fc_free"][k]) for t in truths], 0).tolist()
                                           for k in rows[truths[0]]["fc_free"]}
    agg["floor"] = ci([rows[t]["floor"] for t in truths])
    agg["frac_truths_latent_beats_noDA"] = float(np.mean(np.array(per_truth("latent", "prior", "nmse_mean_dec"))
                                                         < np.array(free_of("prior", "nmse_mean_dec")))) if "latent" in a.modes else None
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "study.json").write_text(json.dumps(agg, indent=1))
    write_provenance(out / "study.json", config=TwinConfig(**base), seed=0, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"args": vars(a)})
    f = agg["free"]["nmse_mean_dec"]
    print(f"\n[study] forecast nMSE, mean +- 95% CI over {len(truths)} truths (second half of {n_cycles} cycles):")
    print(f"   no DA            {f['mean']:.3f} +- {f['ci95']:.3f}    floor {agg['floor']['mean']:.3f}")
    for m in a.modes:
        p, d = agg["modes"][m]["prior"]["nmse_mean_dec"], agg["modes"][m]["prior"]["nmse_mean_dec_minus_noDA"]
        an = agg["modes"][m]["analysis"]["nmse_mean_dec"]
        fd = agg["modes"][m]["prior"].get("front_abs_disp_km_median", {"mean": float("nan")})
        print(f"   {m:16s} {p['mean']:.3f} +- {p['ci95']:.3f}  (vs no DA {d['mean']:+.3f} +- {d['ci95']:.3f})  "
              f"analysis {an['mean']:.3f}  front |disp| {fd['mean']:.2f} km")
    fr = agg["frac_truths_latent_beats_noDA"]
    print(f"[study] " + (f"latent beats no DA on {100 * fr:.0f}% of truths; " if fr is not None else "")
          + f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
