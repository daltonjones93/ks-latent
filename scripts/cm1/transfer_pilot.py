"""T1: transfer pilot - two storms side by side in a domain twice as wide, no retraining.

Two test storms of the same environment are joined along x (2 x 400 km = 800 km). The domain edges are
quiescent, so the joined state is approximately a two-storm state until the storms' outflows meet. The
truth near the junction is not a CM1 solution, so every score leaves out `--junction-km` on each side of
it, and the single-storm runs leave out the same physical columns (A's right edge, B's left edge): with a
perfectly local model the combined scores then equal the mean of the two single-storm scores.

(a) Forecast: the model trained on single 400 km storms forecasts the 800 km two-storm state; nMSE per
    lead (kept columns, boundaries held at t0) of the joined forecast against the mean of the singles.
(b) DA: the latent LETKF on 1 storm and on 2, at ensemble sizes N, with 5 km localization and with
    effectively none (`--locs 5 10000`). Each two-storm member joins one member of each single-storm
    ensemble, so both halves get exactly the single-storm information. Paper I's claim, transferred:
    with localization the ratio (2 storms / mean of 1-storm) stays ~1 at every N; without it, the
    ensemble must cover twice the structure with the same N and the ratio grows, most at small N.
  python scripts/cm1/transfer_pilot.py --run ~/cm1_runs/full_v1_stagec --source cache:A+B+C --out ~/cm1_da/t1_pilot
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
from ks_latent.cm1.data import CM1Member, ColumnData, _natural_key, load_member  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, run_twin, summarize_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402

DX_KM = 0.25


def join(a: CM1Member, b: CM1Member) -> CM1Member:
    """`a` on the left, `b` on the right; frames cut to the shorter run."""
    if not np.allclose(a.t[:2], b.t[:2]):
        raise ValueError(f"{a.name}, {b.name}: different frame times")
    T = min(a.X.shape[0], b.X.shape[0])
    return CM1Member(name=f"{a.name}|{b.name}", X=np.concatenate([a.X[:T], b.X[:T]], axis=1),
                     cref=np.concatenate([a.cref[:T], b.cref[:T]], axis=1), t=a.t[:T].copy())


def keep_cols(NX: int, exclude_km) -> np.ndarray:
    x = (np.arange(NX) + 0.5) * DX_KM
    k = np.ones(NX, bool)
    for x0, x1 in exclude_km:
        k &= ~((x >= x0) & (x < x1))
    return k


@torch.no_grad()
def forecast_nmse(ae, prop, data: ColumnData, z, t0s, n_lead, halo, keep) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead nMSE (kept columns) of the forecast and of persistence, mean over start times."""
    dev = next(ae.parameters()).device
    kk = torch.as_tensor(keep, device=dev)
    fc, pe = [], []
    for t0 in t0s:
        lat = rollout_from(prop, z, t0, n_lead, halo, boundary="persistence")[:, 0]       # (L, P, c)
        dec = ae.decode(lat)                                                               # (L, C, NX)
        tr = data.full(0, slice(t0 + 1, t0 + n_lead + 1), device=dev)                      # (L, C, NX)
        u0 = data.full(0, t0, device=dev)[0]
        fc.append(((dec - tr) ** 2)[..., kk].mean(dim=(1, 2)).cpu().numpy())
        pe.append(((u0[None] - tr) ** 2)[..., kk].mean(dim=(1, 2)).cpu().numpy())
    return np.mean(fc, 0), np.mean(pe, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truths", nargs="*", default=[],
                    help="even number of names, joined in pairs; default: the run's same-environment test storms "
                         "(ens_A mem029-030 and ens_A2 test storms), paired in order")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--junction-km", type=float, default=20.0)
    ap.add_argument("--fc-t0-min", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    ap.add_argument("--fc-lead-min", type=float, default=60.0)
    ap.add_argument("--ens-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--locs", type=float, nargs="+", default=[5.0, 10000.0], help="localization radii (km)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--obs-set", default="th+refl", choices=["th", "th+refl"])
    ap.add_argument("--model-noise", type=float, default=0.05)
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
    truths = a.truths or [n for n in split["test"] if n.startswith("mem") or n.startswith("ens_A2_")]
    if len(truths) % 2:
        truths = truths[:-1]
    pairs = [(truths[i], truths[i + 1]) for i in range(0, len(truths), 2)][: a.n_pairs]
    seen = set(split["train"]) | set(split["val"])
    if any(t in seen for p in pairs for t in p) or not pairs:
        raise ValueError(f"truths must be unseen storms: {pairs}")
    refl_head = None
    if "refl" in a.obs_set:
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        refl_head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev).eval()
        refl_head.load_state_dict(ck["state_dict"])
    print(f"[T1] {run.name}: pairs {pairs}; junction +-{a.junction_km} km; boundaries held at t0", flush=True)
    J = a.junction_km
    report = {"pairs": [list(p) for p in pairs], "args": vars(a), "forecast": {}, "da": []}

    # ---- (a) forecasts ----
    for A, B in pairs:
        mA, mB = load_member(files[A]), load_member(files[B])
        mAB = join(mA, mB)
        W = mA.X.shape[1] * DX_KM                                      # 400 km
        dt = mA.dt
        t0s = [int(round(m * 60 / dt)) for m in a.fc_t0_min]
        nl = int(round(a.fc_lead_min * 60 / dt))
        res = {}
        for tag, m, excl in (("A", mA, [(W - J, W)]), ("B", mB, [(0.0, J)]), ("AB", mAB, [(W - J, W + J)])):
            d = ColumnData([m], norm, dev)
            z = encode_members(ae, d)[0]
            fc, pe = forecast_nmse(ae, prop, d, z, [t for t in t0s if t + nl < d.T[0]], nl, halo,
                                   keep_cols(d.NX, excl))
            res[tag] = {"nmse": fc.tolist(), "pers": pe.tolist()}
        report["forecast"][f"{A}|{B}"] = res
        L = sorted({i for i in (4, 29, nl - 1) if i < nl})
        single = (np.array(res["A"]["nmse"]) + np.array(res["B"]["nmse"])) / 2
        print(f"[T1 fc] {A} | {B}: nMSE at 5/30/60 min  joined " +
              " ".join(f"{res['AB']['nmse'][i]:.4f}" for i in L) + "  singles " +
              " ".join(f"{single[i]:.4f}" for i in L) + "  ratio " +
              " ".join(f"{res['AB']['nmse'][i] / single[i]:.3f}" for i in L), flush=True)
    (out / "t1.json").write_text(json.dumps(report, indent=1))
    if a.skip_da:
        return

    # ---- (b) DA: ensemble size, 1 storm vs 2, localized vs not ----
    pool = sorted(split["train"], key=_natural_key)
    if 2 * max(a.ens_sizes) > len(pool):
        raise ValueError(f"{len(pool)} training members cannot give two disjoint ensembles of {max(a.ens_sizes)}")
    cache = {}
    def member(n):
        if n not in cache:
            cache[n] = load_member(files[n])
        return cache[n]
    for A, B in pairs:
        mA, mB = load_member(files[A]), load_member(files[B])
        truths3 = {"A": mA, "B": mB, "AB": join(mA, mB)}
        W = mA.X.shape[1] * DX_KM
        dt = mA.dt
        t0, spc = int(round(a.t0_min * 60 / dt)), int(round(5 * 60 / dt))
        n_cycles = int(round(a.hours * 12))
        excl = {"A": ((W - J, W),), "B": ((0.0, J),), "AB": ((W - J, W + J),)}
        tdat = {k: ColumnData([m], norm, dev) for k, m in truths3.items()}
        ztr = {k: encode_members(ae, d)[0] for k, d in tdat.items()}
        for sd in a.seeds:
            rng = np.random.default_rng(1000 + sd)
            perm = rng.permutation(pool).tolist()
            for N in a.ens_sizes:
                s1, s2 = perm[:N], perm[N:2 * N]                 # disjoint: one ensemble per storm
                ens = {"A": [member(n) for n in s1], "B": [member(n) for n in s2]}
                ens["AB"] = [join(x, y) for x, y in zip(ens["A"], ens["B"])]
                Z0 = {k: torch.stack([z[t0] for z in encode_members(ae, ColumnData(v, norm, dev))]) for k, v in ens.items()}
                for loc in a.locs:
                    row = {"pair": f"{A}|{B}", "seed": sd, "N": N, "loc_km": loc}
                    for k in ("A", "B", "AB"):
                        tc = TwinConfig(t0=t0, n_cycles=n_cycles, steps_per_cycle=spc, halo=halo,
                                        obs_every_cells=8, sigma_obs=0.5, model_noise=a.model_noise,
                                        refl_every_cells=8 if refl_head is not None else 0, sigma_refl=5.0,
                                        letkf=LETKFConfig(loc_km=loc, rtps=a.rtps), boundary="persistence",
                                        exclude_km=excl[k], forecast_from=(), seed=sd)
                        s = summarize_twin(run_twin(ae, prop, tdat[k], ztr[k], Z0[k], norm, tc, refl_head=refl_head))
                        row[k] = {"prior": s["prior"]["nmse_mean_dec"], "analysis": s["analysis"]["nmse_mean_dec"],
                                  "th_sprd_err": s["prior"]["th_obs_spread"] / s["prior"]["th_obs_rmse"]}
                    single = (row["A"]["prior"] + row["B"]["prior"]) / 2
                    row["ratio_prior"] = row["AB"]["prior"] / single
                    report["da"].append(row)
                    print(f"[T1 da] {A}|{B} seed {sd} N {N:3d} loc {loc:7.0f} km: prior nMSE 1 storm {single:.4f}  "
                          f"2 storms {row['AB']['prior']:.4f}  ratio {row['ratio_prior']:.3f}  "
                          f"[{time.time() - t_start:.0f}s]", flush=True)
                    (out / "t1.json").write_text(json.dumps(report, indent=1))
    # summary: mean ratio by (N, loc) over pairs and seeds
    print("\n[T1] DA prior nMSE ratio, 2 storms / 1 storm (mean +- 95% CI over pairs x seeds):")
    summ = {}
    for loc in a.locs:
        for N in a.ens_sizes:
            r = np.array([d["ratio_prior"] for d in report["da"] if d["N"] == N and d["loc_km"] == loc])
            one = np.array([(d["A"]["prior"] + d["B"]["prior"]) / 2 for d in report["da"] if d["N"] == N and d["loc_km"] == loc])
            h = 1.96 * r.std(ddof=1) / np.sqrt(r.size) if r.size > 1 else float("nan")
            summ[f"loc{loc:g}_N{N}"] = {"ratio": float(r.mean()), "ci95": float(h), "one_storm_nmse": float(one.mean())}
            print(f"   loc {loc:7.0f} km  N {N:3d}: ratio {r.mean():.3f} +- {h:.3f}   (1-storm nMSE {one.mean():.4f})")
    report["summary"] = summ
    (out / "t1.json").write_text(json.dumps(report, indent=1))
    write_provenance(out / "t1.json", config=tc, seed=0, device=str(dev), wall_time_s=time.time() - t_start,
                     extra={"args": vars(a), "note": "config = the last twin run's TwinConfig; see args"})
    print(f"[T1] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
