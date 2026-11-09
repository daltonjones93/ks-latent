"""FPf-D: distil a short channel-0 law from the trained propagator rather than from the encoded data.

FPe fitted dz0/dt from finite differences of the encoded latent trajectories. Here the target is the trained
LatentLaw's own channel-0 tendency F_0(z) (noise-free, exact in time), evaluated on the same encoded states:
  targets   data_c : 0.5 (z0[t+1] - z0[t-1])            (FPe-U's target)
            prop   : prop.law.tendency(z[t])[..., 0]     (the propagator's dz0/dt; depends on every channel)
            prop_s : the propagator's tendency at perturbed states z + sigma * sd_channel * N(0, 1) (off the data)
  libraries law4 : [1, d1z0, z0 d1z0, d2z0]              (FPe-U)
            law6 : [1, z0, d1z0, d2z0, z0^2, z0 d1z0]    (the FPf constrained row)
Fit by OLS on the training storms, scored on training and test storms (test: out of sample, training
coefficients). Also: the dense law's own z0-alone terms ("read-off") scored against its full tendency, and the
law4 coefficients from the propagator target converted to theta' units with the FPe-U gauge map, with a bootstrap
over training storms. Pre-registration: docs/cm1_notes.md, G0 addendum 5g.
  python scripts/cm1/law_distill.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/fpf_distill
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
from law_physical_units import convert  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.leadlaw import site_derivs  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402


def design(zc, d1, d2, lib):
    one = np.ones_like(zc)
    if lib == "law4":
        return np.stack([one, d1, zc * d1, d2], 1)
    return np.stack([one, zc, d1, d2, zc ** 2, zc * d1], 1)


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def storm_rows(z, prop, f0, f1, halo, sigma, sd, rng):
    """z (T, P, c) float64 -> dict of flat arrays over frames f0..f1-1, interior sites."""
    z0 = z[f0 - 1: f1 + 1, :, :1]
    zc3 = z0[1:-1]
    d1, d2 = (d[..., 0] for d in site_derivs(zc3, 2))
    zt = torch.as_tensor(z[f0:f1], dtype=torch.float32)
    with torch.no_grad():
        tp = prop.law.tendency(zt)[..., 0].double().numpy()
        eps = torch.as_tensor(rng.standard_normal(zt.shape) * sd * sigma, dtype=torch.float32)
        zs = zt + eps
        tps = prop.law.tendency(zs)[..., 0].double().numpy()
        co = {k: v.double().numpy() for k, v in prop.law.effective_coefficients().items()}
    zs0 = zs[..., :1].double().numpy()
    ds1, ds2 = (d[..., 0] for d in site_derivs(zs0, 2))
    # the dense law's z0-alone terms (no cap, no residual; cap tanh is ~identity at these magnitudes)
    A, b = co["A"], prop.law.b.detach().double().numpy()
    zz = zc3[..., 0]
    ro = b[0] + A[0, 0, 0] * zz + A[1, 0, 0] * d1 + A[2, 0, 0] * d2
    if "Q1" in co:
        ro = ro + co["Q1"][0, 0] * zz ** 2 + co["Q2"][0, 0] * zz * d1
    sl = slice(halo, -halo)
    out = {"data_c": 0.5 * (z0[2:] - z0[:-2])[..., 0], "prop": tp, "readoff": ro, "zc": zz, "d1": d1, "d2": d2,
           "prop_s": tps, "zs": zs0[..., 0], "ds1": ds1, "ds2": ds2}
    return {k: v[:, sl].ravel() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330")
    ap.add_argument("--sigma", type=float, default=0.1, help="off-data probe: noise in units of each channel's sd")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    if cfg["prop"]["body"] != "law":
        raise ValueError("law_distill needs a law propagator")
    prop.eval()
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))
    p = ae.cfg.patch
    rng = np.random.default_rng(0)

    def enc(names):
        zs, gauge = [], []
        for n in names:
            d = ColumnData([load_member(files[n])], norm, dev)
            with torch.no_grad():
                z = encode_members(ae, d)[0].cpu().numpy().astype(np.float64)
            sc = np.repeat(np.asarray(norm.scale, float), d.Z)[:, None]
            for t in range(f0, min(f1, z.shape[0] - 2), 5):
                u = d.full(0, t)[0].cpu().numpy() * sc
                gauge.append((z[t, :, 0], u[VARS.index("th") * d.Z].reshape(-1, p).mean(1)))
            zs.append(z)
        return zs, gauge
    ztr, gtr = enc(split["train"][: a.n_train])
    zte, _ = enc(split["test"][: a.n_test])
    sd = np.concatenate([z[f0:f1].reshape(-1, z.shape[-1]) for z in ztr]).std(0)
    R = {tag: [storm_rows(z, prop, f0, min(f1, z.shape[0] - 2), halo, a.sigma, sd, rng) for z in zs]
         for tag, zs in (("train", ztr), ("test", zte))}
    cat = lambda rows: {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    Ctr, Cte = cat(R["train"]), cat(R["test"])
    zf = np.concatenate([g[0] for g in gtr]); tf = np.concatenate([g[1] for g in gtr])
    r_map = float(np.corrcoef(zf, tf)[0, 1])
    s = float(np.sign(r_map) * tf.std(ddof=1) / zf.std(ddof=1)); b = float(tf.mean() - s * zf.mean())
    res = {"run": a.run, "sigma": a.sigma, "map": {"r": r_map, "s": s, "b": b}, "fits": {}}
    print(f"[distill] gauge map r {r_map:+.3f}, s {s:+.3f} K, b {b:+.3f} K", flush=True)
    # how much of the propagator's own dz0/dt does its z0-alone part carry, and how close is it to the data?
    res["prop_vs_data_R2"] = {t: r2(C["data_c"], C["prop"]) for t, C in (("train", Ctr), ("test", Cte))}
    res["readoff_vs_prop_R2"] = {t: r2(C["prop"], C["readoff"]) for t, C in (("train", Ctr), ("test", Cte))}
    print(f"[distill] propagator dz0/dt vs data centred difference: R2 train {res['prop_vs_data_R2']['train']:.3f}, "
          f"test {res['prop_vs_data_R2']['test']:.3f}; dense law's z0-alone terms vs its full tendency: R2 train "
          f"{res['readoff_vs_prop_R2']['train']:.3f}, test {res['readoff_vs_prop_R2']['test']:.3f}", flush=True)
    for target in ("data_c", "prop", "prop_s"):
        keys = ("zs", "ds1", "ds2") if target == "prop_s" else ("zc", "d1", "d2")
        for lib in ("law4", "law6"):
            X = design(*(Ctr[k] for k in keys), lib)
            coef, *_ = np.linalg.lstsq(X, Ctr[target], rcond=None)
            Xte = design(*(Cte[k] for k in keys), lib)
            f = {"coef": coef.tolist(), "R2_train": r2(Ctr[target], X @ coef), "R2_test": r2(Cte[target], Xte @ coef)}
            if lib == "law4" and target != "prop_s":
                f["phys"] = convert(coef, s, b, Ctr["zc"], Ctr["d1"])
            res["fits"][f"{target}/{lib}"] = f
            ph = f.get("phys")
            print(f"[distill] {target:6s} {lib}: R2 train {f['R2_train']:.3f} test {f['R2_test']:.3f}"
                  + (f"; c(theta') = {ph['c0_km_min']:+.3f} {ph['c1_km_min_per_K']:+.4f} theta' km/min, nu "
                     f"{ph['nu_m2_s']:.0f} m2/s, density-current c1 {ph['density_current_c1_km_min_per_K']}" if ph else ""),
                  flush=True)
    boots = []
    for _ in range(a.n_boot):
        idx = rng.integers(0, len(R["train"]), len(R["train"]))
        C = cat([R["train"][i] for i in idx])
        coef, *_ = np.linalg.lstsq(design(C["zc"], C["d1"], C["d2"], "law4"), C["prop"], rcond=None)
        boots.append(convert(coef, s, b, C["zc"], C["d1"]))
    res["bootstrap_prop_law4"] = {k: [float(np.percentile([x[k] for x in boots], q)) for q in (2.5, 50, 97.5)]
                                  for k in ("c0_km_min", "c1_km_min_per_K", "nu_m2_s", "c_eff_km_min")}
    for k, v in res["bootstrap_prop_law4"].items():
        print(f"[distill] bootstrap (prop target, law4) {k}: median {v[1]:+.4g}, 95% CI ({v[0]:+.4g}, {v[2]:+.4g})")
    json.dump(res, open(out / "distill.json", "w"), indent=1)
    print(f"[distill] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
