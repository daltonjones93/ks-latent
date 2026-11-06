"""E3c / G13: shifted tilings in the CM1 latent DA (Paper II's fix for seams between patches).

The autoencoder works on 8-cell (2 km) patches, so every decoded field has seams on one fixed lattice. A
tiling offset by o cells sees the domain padded by o cells on the left (and 8 - o on the right) with the
edge columns (the far field is quiescent), so its patch boundaries fall elsewhere. Each tiling is a full
latent DA in its own lattice, all assimilating the SAME observations (same physical columns, same noise
draws: `TwinConfig.col_offset`), and the blended estimate is the average of their mean decoded fields on
the physical columns. Compared, per truth and observation seed:

  single     one tiling (offset 0), N members                   - the headline DA
  tiles_xN   len(offsets) tilings, N members each               - more compute
  tiles_eq   len(offsets) tilings, N / len(offsets) members each, disjoint, N members in all - equal cost

Score: nMSE of the blended mean field against the truth on the physical columns, second half of the cycles
(prior = forecast, and analysis), mean +- 95% CI over truths and paired differences against `single`.
  python scripts/cm1/da_tiling.py --run ~/cm1_runs/full_v1_stagec --source cache:A+B+C --include-test --out ~/cm1_da/e3c
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
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, run_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def pad(m: CM1Member, o: int, patch: int) -> CM1Member:
    """o cells of the left edge column on the left, (patch - o) % patch of the right one on the right."""
    r = (patch - o) % patch
    if o == 0 and r == 0:
        return m
    f = lambda x: np.concatenate([np.repeat(x[:, :1], o, 1), x, np.repeat(x[:, -1:], r, 1)], 1)
    return CM1Member(name=m.name, X=f(m.X), cref=f(m.cref), t=m.t)


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truths", nargs="*", default=[])
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--n-members", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--obs-set", default="th+refl+vr", choices=["th", "th+refl", "th+vr", "th+refl+vr"])
    ap.add_argument("--model-noise", type=float, default=0.05)
    ap.add_argument("--loc-km", type=float, default=5.0)
    ap.add_argument("--rtps", type=float, default=0.9)
    ap.add_argument("--t0-min", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--control", action="store_true",
                    help="also: K independent DA systems at offset 0 (same observations, own model noise) averaged, "
                         "against K tilings with the same distinct model noise: separates seam removal from averaging")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    patch, halo = ae.cfg.patch, cfg["stage_b"]["halo_sites"]
    if 0 not in a.offsets or any(not 0 <= o < patch for o in a.offsets):
        raise ValueError(f"offsets must include 0 and lie in [0, {patch})")
    K = len(a.offsets)
    if a.n_members % K:
        raise ValueError(f"--n-members {a.n_members} is not divisible by {K} tilings (equal-cost comparison)")
    files = member_files(a.source)
    truths = list(a.truths) + (list(split["test"]) if a.include_test else [])
    seen = set(split["train"]) | set(split["val"])
    if not truths or any(t in seen for t in truths):
        raise ValueError("truths must be storms the model never saw")
    ens_names = sorted(np.random.default_rng(0).choice(split["train"], a.n_members, replace=False).tolist(),
                       key=_natural_key)                       # as da_study --random-members
    ens = [load_member(files[n]) for n in ens_names]
    refl_head = None
    if "refl" in a.obs_set:
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        refl_head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev).eval()
        refl_head.load_state_dict(ck["state_dict"])
    dt = ens[0].dt
    t0, spc = int(round(a.t0_min * 60 / dt)), int(round(5 * 60 / dt))
    n_cycles = int(round(a.hours * 12))
    NXp = ens[0].X.shape[1]
    # latent initial ensembles per tiling (truth-independent)
    Z0 = {}
    for o in a.offsets:
        z = encode_members(ae, ColumnData([pad(m, o, patch) for m in ens], norm, dev))
        Z0[o] = torch.stack([zz[t0] for zz in z])
    print(f"[E3c] {run.name}: {len(truths)} truths x seeds {a.seeds}; tilings {a.offsets}; {a.n_members} members; "
          f"noise {a.model_noise}; obs {a.obs_set}", flush=True)
    rows = []
    for tn in truths:
        mt = load_member(files[tn])
        tdat = ColumnData([mt], norm, dev)
        U = torch.stack([tdat.full(0, t0 + (k + 1) * spc, device="cpu")[0] for k in range(n_cycles)])  # (cyc, C, NXp)
        pt = {o: ColumnData([pad(mt, o, patch)], norm, dev) for o in a.offsets}
        zt = {o: encode_members(ae, pt[o])[0] for o in a.offsets}
        for sd in a.seeds:
            def twin(o, members, noise_seed=None):
                tc = TwinConfig(t0=t0, n_cycles=n_cycles, steps_per_cycle=spc, halo=halo, obs_every_cells=8,
                                sigma_obs=0.5, model_noise=a.model_noise, assim_theta=True,
                                refl_every_cells=8 if refl_head is not None else 0, sigma_refl=5.0,
                                vr_tilts=(0.5, 2.0, 5.0) if "vr" in a.obs_set else (), sigma_vr=2.0,
                                letkf=LETKFConfig(loc_km=a.loc_km, rtps=a.rtps), boundary="persistence",
                                forecast_from=(), seed=sd, noise_seed=noise_seed, col_offset=o, n_cols_phys=NXp,
                                keep_fields=True)
                r = run_twin(ae, prop, pt[o], zt[o], Z0[o][members], norm, tc, refl_head=refl_head)
                return {st: torch.stack([c[st + "_field"] for c in r["cycles"]]).float() for st in ("prior", "analysis")}
            allm = torch.arange(a.n_members)
            fx = {o: twin(o, allm) for o in a.offsets}                                          # N each
            groups = allm.reshape(K, -1)
            feq = {o: twin(o, groups[i]) for i, o in enumerate(a.offsets)}                      # N / K each
            def score(fields_by_tile):
                d = {}
                for st in ("prior", "analysis"):
                    blend = torch.stack([f[st] for f in fields_by_tile]).mean(0)               # (cyc, C, NXp)
                    e = ((blend - U) ** 2).mean(dim=(1, 2))[n_cycles // 2:]
                    d[st] = float(e.mean())
                return d
            row = {"truth": tn, "seed": sd, "single": score([fx[0]]),
                   f"tiles_x{a.n_members}": score([fx[o] for o in a.offsets]),
                   "tiles_eq": score([feq[o] for o in a.offsets]),
                   "per_offset": {o: score([fx[o]]) for o in a.offsets}}
            if a.control:   # k = 0 is the single run itself (default noise seed)
                ns = lambda k: None if k == 0 else 7919 * (sd + 1) + k
                same = [fx[0]] + [twin(0, allm, ns(k)) for k in range(1, K)]
                tiles = [fx[0]] + [twin(o, allm, ns(k)) for k, o in enumerate(a.offsets) if k > 0]
                row["avg_same_offset"] = score(same)
                row["tiles_distinct_noise"] = score(tiles)
            rows.append(row)
            print(f"[E3c] {tn} seed {sd}: prior nMSE single {row['single']['prior']:.4f}  "
                  f"{K}x{a.n_members} {row[f'tiles_x{a.n_members}']['prior']:.4f}  "
                  f"{K}x{a.n_members // K} {row['tiles_eq']['prior']:.4f}  [{time.time() - t_start:.0f}s]", flush=True)
    # per truth (seed-averaged), then mean +- CI over truths and paired differences against `single`
    keys = ["single", f"tiles_x{a.n_members}", "tiles_eq"] + (["avg_same_offset", "tiles_distinct_noise"] if a.control else [])
    summ = {}
    print(f"\n[E3c] nMSE, mean +- 95% CI over {len(truths)} truths (second half of {n_cycles} cycles); paired vs single:")
    for st in ("prior", "analysis"):
        per = {k: np.array([np.mean([r[k][st] for r in rows if r["truth"] == t]) for t in truths]) for k in keys}
        for k in keys:
            m, h = ci(per[k])
            dm, dh = ci(per[k] - per["single"])
            wins = float(np.mean(per[k] < per["single"]))
            summ[f"{st}_{k}"] = {"mean": m, "ci95": h, "minus_single": dm, "minus_single_ci95": dh, "frac_lower": wins}
            print(f"   {st:8s} {k:12s} {m:.4f} +- {h:.4f}   vs single {dm:+.4f} +- {dh:.4f}   lower on {100 * wins:.0f}%")
    res = {"args": vars(a), "ensemble": ens_names, "rows": rows, "summary": summ}
    (out / "e3c.json").write_text(json.dumps(res, indent=1, default=str))
    write_provenance(out / "e3c.json", config=TwinConfig(model_noise=a.model_noise), seed=0, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"args": vars(a)})
    print(f"[E3c] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
