"""Track FP, FP0: does a compact "transport + sources" law exist in hand-built CM1 field variables?

Known-answer pilot for the field-latent PDE track (docs/cm1_plan.md, track FP). Reads raw CM1 GrADS
runs (total fields), builds theta, q_v (not conserved), theta_il, q_t (conserved under phase change),
the vorticity eta, precipitating and cloud condensate (`ks_latent.cm1.fieldpde`), and for frame pairs
(t, t + k dt) in a storm-centred mask reports, per variable:

  transport skill   1 - E|phi(t+dt) - SL phi(t)|^2 / E|phi(t+dt) - phi(t)|^2   (semi-Lagrangian step)
  source skill      the same with SL phi(t) + dt * (sparse source law), the law fitted on --fit members
  sparsity path     STLSQ thresholds -> terms kept, held-out R^2 of the Lagrangian residual
  selected terms    bootstrap selection frequencies and coefficients (physical units)

Fitting members and held-out members are disjoint (whole storms). Known answers: eta's law has
dB/dx with coefficient -1; q_t's has +V dq_p/dz (V a fall speed, a few m/s); q_v's has cond = w dq_sat/dz
in cloud with coefficient ~ +1 and theta's the same term with ~ -Lv/(cp pi) ~ -2500 to -3500 K.
  python scripts/cm1/fp_pilot.py --runs ~/cm1_bundle/ens_A/members/mem00{1,2,3,4} \\
      ~/cm1_bundle/ens_A2/members/mem00{1,2} ~/cm1_bundle/ens_B/members/mem0{03,16} --fit 4 --out ~/cm1_da/f0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ks_latent.cm1.fieldpde import (bootstrap_stlsq, ddx, ddz, departure_points, laplacian,  # noqa: E402
                                    moist_fields, r2, semi_lagrangian, stlsq)
from ks_latent.cm1.io import BASE_I, read_grads  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402

TARGETS = ("theta", "qv", "theta_il", "qt", "eta", "qp", "qcl")
COMMON = ("dBdx", "dz_qp", "cond", "qcl", "qp", "qcl_qp")   # no Lv/(cp pi)-scaled duplicates: near-collinear
OWN = ("lap", "self", "self_w_dlnrho")
TERMS = COMMON + OWN
RAW_VARS = ("th", "prs", "qv", "qc", "qr", "qi", "qs", "qg", "uinterp", "winterp")


def ctl_levels(ctl: Path) -> np.ndarray:
    """Model level heights (m) from a GrADS ctl's `zdef N levels` block."""
    tok = Path(ctl).read_text().split()
    i = [k for k, t in enumerate(tok) if t.lower() == "zdef"][0]
    n = int(tok[i + 1])
    if tok[i + 2].lower() == "linear":
        return (float(tok[i + 3]) + float(tok[i + 4]) * np.arange(n)) * 1000.0
    if tok[i + 2].lower() != "levels":
        raise ValueError(f"{ctl}: zdef is neither linear nor a levels list")
    return np.array([float(v) for v in tok[i + 3: i + 3 + n]]) * 1000.0


def member_rows(run: Path, a, rng) -> dict:
    ctl = run / "cm1out_s.ctl"
    z = ctl_levels(ctl)
    d = read_grads(ctl)
    t_s = read_grads(run / "cm1out_metadata.ctl")["mtime"].ravel()
    dt0 = float(np.median(np.diff(t_s)))
    base = BaseState.from_sounding(run / "input_sounding")
    rho0 = np.interp(z, base.z, base.rho0)
    bp = (d["th"][0, :, BASE_I].astype(np.float64), d["qv"][0, :, BASE_I].astype(np.float64))
    k = a.dt_stride
    dt = k * dt0
    i0, i1 = int(round(a.t_range[0] * 60 / dt0)), int(round(a.t_range[1] * 60 / dt0))
    starts = range(i0, min(i1, len(t_s) - k), max(1, int(round(a.t_step * 60 / dt0))))
    rows = {v: {"ep": [], "es": [], "terms": [], "pair": []} for v in TARGETS}
    for p, t in enumerate(starts):
        sub = {v: d[v][[t, t + k]] for v in RAW_VARS}
        f = moist_fields(sub, z, BASE_I, rho0, base_profiles=bp)
        act = d["cref"][t] > a.active_dbz
        if not act.any():
            continue
        cols = binary_dilation(act, iterations=a.mask_halo)
        kk, ii = np.nonzero(np.broadcast_to(cols[None], (len(z), cols.size)) & (z < a.z_max)[:, None])
        sel = rng.choice(len(kk), size=min(a.n_points, len(kk)), replace=False)
        kk, ii = kk[sel], ii[sel]
        dep = departure_points(f["u"][0], f["w"][0], f["u"][1], f["w"][1], dt, z)
        traj = lambda F: 0.5 * (semi_lagrangian(F[0], dep, order=1) + F[1])           # trajectory average
        com = {"dBdx": ddx(f["B"]), "dz_qp": ddz(f["qp"], z), "cond": f["cond"], "qcl": f["qcl"], "qp": f["qp"],
               "qcl_qp": f["qcl"] * f["qp"]}
        com = {n: traj(v)[kk, ii] for n, v in com.items()}
        for v in TARGETS:
            phi = f[v]
            sl = semi_lagrangian(phi[0], dep)
            own = {"lap": traj(laplacian(phi, z)), "self": traj(phi), "self_w_dlnrho": traj(phi * f["w_dlnrho"])}
            X = np.stack([com[n] for n in COMMON] + [own[n][kk, ii] for n in OWN], axis=1)
            r = rows[v]
            r["ep"].append((phi[1] - phi[0])[kk, ii]); r["es"].append((phi[1] - sl)[kk, ii])
            r["terms"].append(X); r["pair"].append(np.full(len(kk), p))
    del d
    out = {v: {"ep": np.concatenate(r["ep"]), "es": np.concatenate(r["es"]), "X": np.concatenate(r["terms"]),
               "pair": np.concatenate(r["pair"])} for v, r in rows.items()}
    return {"rows": out, "dt": dt, "n_pairs": len(starts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="raw CM1 run directories (cm1out_s.ctl)")
    ap.add_argument("--fit", type=int, required=True, help="the first FIT runs fit the laws; the rest are held out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-range", type=float, nargs=2, default=[60.0, 240.0], help="minutes")
    ap.add_argument("--t-step", type=float, default=3.0, help="minutes between sampled pairs")
    ap.add_argument("--dt-stride", type=int, default=1, help="frames per step (dt = stride x output interval)")
    ap.add_argument("--n-points", type=int, default=20000, help="sampled mask points per pair")
    ap.add_argument("--active-dbz", type=float, default=10.0)
    ap.add_argument("--mask-halo", type=int, default=40, help="cells added on each side of storm columns")
    ap.add_argument("--z-max", type=float, default=16000.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--threshold", type=float, default=0.05, help="the one used for the bootstrap table")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    runs = [Path(r).expanduser() for r in a.runs]
    if not 0 < a.fit < len(runs):
        raise ValueError("need at least one fitting and one held-out run")
    per = []
    for r in runs:
        per.append(member_rows(r, a, rng))
        print(f"[FP0] {r} done: {per[-1]['n_pairs']} pairs, dt {per[-1]['dt']:.0f} s [{time.time() - t0:.0f}s]", flush=True)
    dts = {p["dt"] for p in per}
    if len(dts) != 1:
        raise ValueError(f"runs have different steps {dts}")
    dt = dts.pop()
    res = {"runs": [str(r) for r in runs], "fit": a.fit, "dt_s": dt, "terms": list(TERMS), "targets": {}}
    hdr = "%-9s %9s %9s %9s  %s" % ("variable", "transport", "+sources", "resid R2", "sparsity path: threshold -> terms / held-out R2")
    print("\n[FP0] skill against persistence on held-out storms (1 = perfect, 0 = persistence)\n" + hdr)
    for v in TARGETS:
        cat = lambda key, idx: np.concatenate([per[i]["rows"][v][key] for i in idx])
        fi, ho = range(a.fit), range(a.fit, len(runs))
        Xf, rf = cat("X", fi), cat("es", fi) / dt
        gf = np.concatenate([per[i]["rows"][v]["pair"] + 10000 * i for i in fi])
        Xh, eph, esh = cat("X", ho), cat("ep", ho), cat("es", ho)
        path = []
        for th in a.thresholds:
            xi = stlsq(Xf, rf, th)
            path.append({"threshold": th, "n_terms": int((xi != 0).sum()), "r2_heldout": r2(esh / dt, Xh @ xi)})
        bs = bootstrap_stlsq(Xf, rf, gf, a.threshold, n_boot=20, seed=a.seed)
        xi = bs["full"]
        sp = float(np.sum(eph ** 2))
        rec = {"transport_skill": 1 - float(np.sum(esh ** 2)) / sp,
               "source_skill": 1 - float(np.sum((esh - dt * Xh @ xi) ** 2)) / sp,
               "transport_skill_fit": 1 - float(np.sum(cat("es", fi) ** 2)) / float(np.sum(cat("ep", fi) ** 2)),
               "resid_r2_heldout": r2(esh / dt, Xh @ xi), "path": path,
               "terms": {n: {"freq": float(bs["freq"][j]), "coef": float(xi[j]), "median": float(bs["median"][j])}
                         for j, n in enumerate(TERMS)}}
        res["targets"][v] = rec
        print("%-9s %9.3f %9.3f %9.3f  %s" % (v, rec["transport_skill"], rec["source_skill"], rec["resid_r2_heldout"],
                                              "  ".join(f"{p['threshold']:g}->{p['n_terms']}/{p['r2_heldout']:.2f}" for p in path)))
    print(f"\n[FP0] selected source terms at threshold {a.threshold} (bootstrap frequency >= 0.8; coefficient, physical units):")
    for v, rec in res["targets"].items():
        sel = [f"{n} {t['coef']:+.3g} ({t['freq']:.2f})" for n, t in rec["terms"].items() if t["freq"] >= 0.8]
        print(f"  {v:9s} " + ("; ".join(sel) if sel else "(none)"))
    res["wall_s"] = time.time() - t0
    (out / "f0.json").write_text(json.dumps(res, indent=1))
    print(f"[FP0] done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
