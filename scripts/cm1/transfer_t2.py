"""T2: transfer to real two-storm CM1 runs (800 km, two squall lines 400 km apart), no retraining.

The model was trained on single storms in 400 km domains. Here it runs on the 800 km two-storm truths
(`make_ensemble.py --nx 3200 --two-storms 400000`), and is compared with itself on single-storm test storms
of the same (control) environment, at the same start times and leads, boundaries held at the start.
Unlike the pilot (T1), the truths are real CM1 solutions everywhere, including where the storms' outflows
and gravity waves meet (after ~3 h), so later start times test the model on interacting storms it never saw.

(a) Forecasts: skill (nMSE / persistence nMSE, whole domain) by lead for start times every hour, two-storm
    truths against single-storm truths; the ratio of their errors by start time.
(b) DA: the latent LETKF at ensemble sizes N, with 5 km localization and without (10 000 km), theta' +
    reflectivity. Two-storm ensemble members are two training storms side by side (the training archive
    only has single storms); single-storm references use N training storms. Error ratio, two-storm mean /
    single-storm mean (different truths, so unpaired), per N and localization.
  python scripts/cm1/transfer_t2.py --run ~/cm1_runs/full_v1_stagec --source cache:A+A2+B --t2 ~/cm1_cache_T2 --out ~/cm1_da/t2
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
from transfer_pilot import forecast_nmse, join  # noqa: E402
from ks_latent.cm1.data import ColumnData, _natural_key, load_cached_members, load_member  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, run_twin, summarize_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True, help="caches of the single-storm ensembles")
    ap.add_argument("--t2", required=True, help="cache directory of the two-storm runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-prefix", default="ens_A2_", help="single-storm references: the run's test storms with this prefix")
    ap.add_argument("--fc-t0-min", type=float, nargs="+", default=[60.0, 120.0, 180.0, 240.0])
    ap.add_argument("--fc-lead-min", type=float, default=60.0)
    ap.add_argument("--ens-sizes", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--locs", type=float, nargs="+", default=[5.0, 10000.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-truths", type=int, default=8)
    ap.add_argument("--model-noise", type=float, default=0.1)
    ap.add_argument("--rtps", type=float, default=0.9)
    ap.add_argument("--t0-min", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--skip-da", action="store_true")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    two = load_cached_members(Path(a.t2).expanduser())[: a.n_truths]
    single_names = [n for n in split["test"] if n.startswith(a.single_prefix)][: a.n_truths]
    if not single_names:
        raise ValueError(f"no test storms with prefix {a.single_prefix}")
    singles = [load_member(files[n]) for n in single_names]
    dt = two[0].dt
    print(f"[T2] {len(two)} two-storm truths ({two[0].X.shape[1]} cells), {len(singles)} single-storm references "
          f"({singles[0].X.shape[1]} cells); model {run.name}", flush=True)
    rep = {"two_storm": [m.name for m in two], "single": single_names, "args": vars(a), "forecast": {}, "da": []}

    # ---- (a) forecasts ----
    nl = int(round(a.fc_lead_min * 60 / dt))
    def fc_skill(members):
        res = {}
        for t0m in a.fc_t0_min:
            t0 = int(round(t0m * 60 / dt))
            sk = []
            for m in members:
                if t0 + nl >= m.X.shape[0]:
                    continue
                d = ColumnData([m], norm, dev)
                z = encode_members(ae, d)[0]
                fc, pe = forecast_nmse(ae, prop, d, z, [t0], nl, halo, np.ones(d.NX, bool))
                sk.append((fc, pe))
            if sk:
                res[str(t0m)] = {"nmse": np.mean([s[0] for s in sk], 0).tolist(), "pers": np.mean([s[1] for s in sk], 0).tolist(),
                                 "skill_by_case": [(s[0] / s[1]).tolist() for s in sk]}
        return res
    rep["forecast"]["two_storm"] = fc_skill(two)
    rep["forecast"]["single"] = fc_skill(singles)
    L = [i for i in (4, 29, nl - 1) if i < nl]
    print("\n[T2 fc] skill (nMSE / persistence) at 5 / 30 / 60 min, and the ratio of errors two-storm / single:")
    for t0m in a.fc_t0_min:
        k = str(t0m)
        if k not in rep["forecast"]["two_storm"] or k not in rep["forecast"]["single"]:
            continue
        f2, f1 = rep["forecast"]["two_storm"][k], rep["forecast"]["single"][k]
        s2 = np.array(f2["nmse"]) / np.array(f2["pers"]); s1 = np.array(f1["nmse"]) / np.array(f1["pers"])
        r = np.array(f2["nmse"]) / np.array(f1["nmse"])
        print(f"   start {t0m:4.0f} min: two-storm " + " ".join("%.3f" % s2[i] for i in L) + "   single " +
              " ".join("%.3f" % s1[i] for i in L) + "   error ratio " + " ".join("%.2f" % r[i] for i in L), flush=True)
    (out / "t2.json").write_text(json.dumps(rep, indent=1))
    if a.skip_da:
        return

    # ---- (b) DA ----
    rh = None
    if (run / "refl_head.pt").exists():
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        rh = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev).eval(); rh.load_state_dict(ck["state_dict"])
    else:
        print("  WARNING: no refl_head.pt in the run: assimilating theta' only", flush=True)
    pool = sorted(split["train"], key=_natural_key)
    cache = {}
    member = lambda n: cache.setdefault(n, load_member(files[n]))
    t0 = int(round(a.t0_min * 60 / dt)); spc = int(round(5 * 60 / dt)); n_cyc = int(round(a.hours * 12))
    def twin(truth_m, ens_members, loc, sd):
        d = ColumnData([truth_m], norm, dev)
        z = encode_members(ae, d)[0]
        Z0 = torch.stack([zz[t0] for zz in encode_members(ae, ColumnData(ens_members, norm, dev))])
        tc = TwinConfig(t0=t0, n_cycles=n_cyc, steps_per_cycle=spc, halo=halo, obs_every_cells=8, sigma_obs=0.5,
                        model_noise=a.model_noise, refl_every_cells=8 if rh is not None else 0, sigma_refl=5.0,
                        letkf=LETKFConfig(loc_km=loc, rtps=a.rtps), boundary="persistence", forecast_from=(), seed=sd)
        return summarize_twin(run_twin(ae, prop, d, z, Z0, norm, tc, refl_head=rh))["prior"]["nmse_mean_dec"]
    for sd in a.seeds:
        perm = np.random.default_rng(2000 + sd).permutation(pool).tolist()
        for N in a.ens_sizes:
            s1, s2 = perm[:N], perm[N:2 * N]
            joined = [join(member(x), member(y)) for x, y in zip(s1, s2)]
            for loc in a.locs:
                e2 = [twin(m, joined, loc, sd) for m in two]
                e1 = [twin(m, [member(x) for x in s1], loc, sd) for m in singles]
                row = {"seed": sd, "N": N, "loc_km": loc, "two_storm": e2, "single": e1,
                       "ratio": float(np.mean(e2) / np.mean(e1))}
                rep["da"].append(row)
                print(f"[T2 da] seed {sd} N {N:3d} loc {loc:7.0f} km: two-storm {np.mean(e2):.4f}  single {np.mean(e1):.4f}  "
                      f"ratio {row['ratio']:.3f}  [{time.time() - t_start:.0f}s]", flush=True)
                (out / "t2.json").write_text(json.dumps(rep, indent=1))
    print("\n[T2] DA error ratio, two-storm / single-storm (unpaired; mean over seeds):")
    for loc in a.locs:
        for N in a.ens_sizes:
            r = [x for x in rep["da"] if x["N"] == N and x["loc_km"] == loc]
            m2, h2 = ci(np.concatenate([x["two_storm"] for x in r])); m1, h1 = ci(np.concatenate([x["single"] for x in r]))
            print(f"   loc {loc:7.0f} km N {N:3d}: two-storm {m2:.4f} +- {h2:.4f}   single {m1:.4f} +- {h1:.4f}   ratio {m2 / m1:.3f}")
    (out / "t2.json").write_text(json.dumps(rep, indent=1))
    write_provenance(out / "t2.json", config=TwinConfig(model_noise=a.model_noise), seed=0, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"args": vars(a)})
    print(f"[T2] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
