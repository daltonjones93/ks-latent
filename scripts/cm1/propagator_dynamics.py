"""Plan Q1-Q3: dynamical invariants of the CM1 latent propagator, against CM1.

Q1  Dispersion relation of the quiescent environment: the stencil symbol's eigenvalues about the
    homogeneous far-field latent state (growth rate and phase speed against wavelength, all 32 modes);
    the residual of that state as a fixed point. Data side: phase speeds of the (k, omega) power peak of w
    near 5 km in the storm-free parts of the domain, for CM1 (test storms) and for free surrogate runs, and
    the hydrostatic gravity-wave speed N H / pi from the base sounding.
Q2/Q3  For mature storm states: the best shift speed s and the residual r(T) of "the state repeats after T
    minutes up to a shift" for T = 1 ... 30, for the propagator (f^T) and for the encoded CM1 trajectory
    itself (Z(t + T) against Z(t)). A residual that stays small with a consistent s: a travelling wave
    (steady squall line); a dip at some T: a regeneration period. Per storm: speed (km/min) against the
    CM1 gust-front speed, the decoded cold-pool intensity C and the environment's low-level shear (RKW).
  python scripts/cm1/propagator_dynamics.py --run ~/cm1_runs/full_v1_stagec --source cache:A+A2+B --out ~/cm1_da/q_v1
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.dynamics import data_dispersion, dispersion, relative_equilibrium, shift_sites, stencil_blocks  # noqa: E402
from ks_latent.cm1.env import _profiles, read_manifest  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BASE_SOUNDING, BaseState, cold_pool_intensity  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.verify import main_pool_front_km  # noqa: E402

TS = (1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(Path.home() / "cm1_bundle/ens_B/manifest.csv"))
    ap.add_argument("--n-storms", type=int, default=10)
    ap.add_argument("--mature-min", type=float, nargs="+", default=[150.0, 210.0])
    ap.add_argument("--refine-iters", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo, patch = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    reach = prop.cfg.reach_sites
    site_km = patch * 0.25
    files = member_files(a.source)
    base = BaseState.from_sounding()
    z_m = base.z
    names = [n for n in split["test"] if n.startswith("ens_A2_")][:3] + [n for n in split["test"] if n.startswith("ens_B_")]
    names = names[:a.n_storms]
    log = lambda s: print(s, flush=True)
    data = {n: ColumnData([load_member(files[n])], norm, dev) for n in names}
    lat = {n: encode_members(ae, d)[0].double() for n, d in data.items()}
    dt_min = data[names[0]].dt / 60
    prop_d = copy.deepcopy(prop).double()        # a float64 copy for linearization; `prop` stays float32 for rollouts
    res = {"storms": names}

    # ---------------- Q1: quiescent dispersion relation ----------------
    t5 = int(round(5 / dt_min))
    P = lat[names[0]].shape[1]
    xs = (np.arange(P) + 0.5) * site_km
    far = np.abs(xs - 200.0) > 80.0
    z_bar = torch.stack([lat[n][t5][torch.as_tensor(far)].mean(0) for n in names]).mean(0)
    with torch.no_grad():
        Zh = z_bar[None, None].expand(1, 64, -1).clone()
        fp = float(((prop_d(Zh) - Zh)[0, 10:-10] ** 2).mean().sqrt() / (z_bar ** 2).mean().sqrt())
    A = stencil_blocks(prop_d, z_bar, reach)
    disp = dispersion(A, site_km, dt_min, nk=128)
    lam_k = disp["wavelength_km"]
    smax = disp["sigma"].max(axis=1)
    neutral = np.abs(disp["sigma"]) < 2e-3
    c_neutral = np.where(neutral, np.abs(disp["c_phase"]), np.nan)
    res["Q1"] = {"fixed_point_rel_residual": fp, "wavelength_km": lam_k.tolist(), "sigma_max": smax.tolist(),
                 "sigma": disp["sigma"].tolist(), "c_phase": disp["c_phase"].tolist()}
    i_g = int(np.argmax(smax))
    log(f"[Q1] quiescent state: relative fixed-point residual {fp:.2e}; max growth rate {smax.max():+.4f}/min at "
        f"{lam_k[i_g]:.1f} km; {int(neutral.any(axis=1).sum())} of {lam_k.size} wavenumbers have neutral modes")
    for lo, hi in ((4, 8), (8, 16), (16, 32), (32, 64), (64, 256)):
        sel = (lam_k >= lo) & (lam_k < hi)
        cs = c_neutral[sel]
        log(f"   wavelength {lo:3d}-{hi:3d} km: max growth {smax[sel].max():+.4f}/min; fastest neutral |c| "
            f"{np.nanmax(cs) if np.isfinite(cs).any() else float('nan'):.2f} km/min")
    # theory: N H / pi from the base sounding, 1-12 km
    prof = _profiles(BASE_SOUNDING)
    zz, th = prof[:, 0], prof[:, 1]
    sel = (zz > 1000) & (zz < 12000)
    N = float(np.sqrt(9.81 / th[sel].mean() * np.polyfit(zz[sel], th[sel], 1)[0]))
    c1 = N * 12000.0 / np.pi * 60 / 1000                                  # km/min
    res["Q1"]["theory_N"] = N; res["Q1"]["theory_c1_km_per_min"] = c1
    log(f"   theory: N = {N:.4f} /s, first hydrostatic gravity-wave mode N H / pi = {c1:.2f} km/min (H = 12 km; grid frame "
        f"Doppler-shifted by the environment wind)")
    # data: (k, omega) of w near 5 km in storm-free windows, CM1 against free surrogate runs
    k5 = int(np.argmin(np.abs(z_m - 5000)))
    iw = norm.vars.index("w") * data[names[0]].Z + k5
    t0, t1 = int(round(60 / dt_min)), int(round(240 / dt_min))
    spec = {"cm1": [], "surrogate": []}
    for n in names:
        d = data[n]
        tr = d.full(0, slice(t0, t1))[:, iw].numpy() * float(norm.scale[norm.vars.index("w")])     # (T, NX)
        with torch.no_grad():
            la = rollout_from(prop, lat[n].float(), t0, t1 - t0 - 1, halo, boundary="persistence")[:, 0]
            sw = np.stack([ae.decode(la[j:j + 1])[0, iw].numpy() for j in range(la.shape[0])]) * float(norm.scale[norm.vars.index("w")])
        NX = tr.shape[1]
        for key, F in (("cm1", tr), ("surrogate", sw)):
            for xa, xb in ((0, int(110 / 0.25)), (int(300 / 0.25), NX)):          # upstream and downstream far field
                dd = data_dispersion(F[:, xa:xb], 0.25, dt_min)
                spec[key].append((dd["k"], dd["c_peak"], dd["power"].sum(0)))
    def summarize(entries):
        out = {}
        for lo, hi in ((8, 16), (16, 32), (32, 64)):
            cs, ws = [], []
            for k, c, pw in entries:
                lk = 2 * np.pi / np.maximum(k, 1e-12)
                sel = (lk >= lo) & (lk < hi) & np.isfinite(c)
                cs.append(np.abs(c[sel])); ws.append(pw[sel])
            cs, ws = np.concatenate(cs), np.concatenate(ws)
            out[f"{lo}-{hi}km"] = float((cs * ws).sum() / ws.sum()) if ws.sum() > 0 else float("nan")
        return out
    res["Q1"]["data_c_km_per_min"] = {k: summarize(v) for k, v in spec.items()}
    log("   power-weighted |phase speed| of w at 5 km in the far field (km/min): " + "; ".join(
        f"{k}: " + ", ".join(f"{b} {v:.2f}" for b, v in s.items()) for k, s in res["Q1"]["data_c_km_per_min"].items()))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for j in range(disp["sigma"].shape[1]):
        axs[0].semilogx(lam_k, disp["sigma"][:, j], "k.", ms=1)
        axs[1].semilogx(lam_k, np.where(np.abs(disp["sigma"][:, j]) < 5e-3, disp["c_phase"][:, j], np.nan), "k.", ms=1)
    for key, col in (("cm1", "C3"), ("surrogate", "C0")):
        for lo, v in zip((12, 24, 48), res["Q1"]["data_c_km_per_min"][key].values()):
            axs[1].plot([lo], [v], "o", color=col, label=key if lo == 12 else None)
            axs[1].plot([lo], [-v], "o", color=col)
    axs[1].axhline(c1, color="0.5", ls=":", lw=1); axs[1].axhline(-c1, color="0.5", ls=":", lw=1)
    axs[0].set_xlabel("wavelength (km)"); axs[0].set_ylabel("growth rate (per min)"); axs[0].set_title("Q1: quiescent symbol, all modes")
    axs[1].set_xlabel("wavelength (km)"); axs[1].set_ylabel("phase speed, near-neutral modes (km/min)")
    axs[1].legend(fontsize=7, frameon=False); axs[1].set_title("dotted: N H / pi")
    fig.savefig(out / "fig_q1_dispersion.png", dpi=150)

    # ---------------- Q2/Q3: travelling waves and regeneration periods ----------------
    man = {}
    try:
        man = read_manifest(Path(a.manifest).expanduser(), "ens_B_")
    except FileNotFoundError:
        log("  (no ens_B manifest: shear not reported)")
    u_ref = prof[:, 3]
    du_base = float(np.interp(2500, zz, u_ref) - np.interp(0, zz, u_ref))
    rows = []
    margin = halo + reach + 2
    for n in names:
        z = lat[n]
        T = z.shape[0]
        d = data[n]
        for tm in a.mature_min:
            t = int(round(tm / dt_min))
            if t + max(TS) >= T:
                continue
            row = {"storm": n, "t_min": tm, "T": list(TS), "model": {}, "cm1": {}}
            for Tn in TS:
                r = relative_equilibrium(prop_d, z[t].clone(), Tn, halo, margin)
                row["model"][str(Tn)] = {"s_km_per_min": r["s_sites_per_step"] * site_km / dt_min, "residual": r["residual"]}
                # the CM1 trajectory itself: best shift s for Z(t + T) against Z(t)
                Zt, Zn = z[t], z[t + Tn]
                ref = Zt[margin:-margin] - Zt.mean(0, keepdim=True)
                sc = float((ref ** 2).mean())
                grid = np.arange(-2.0, 2.0001, 0.02)
                rr = [float(((shift_sites(Zn, torch.tensor(-g * Tn, dtype=Zn.dtype)) - Zt)[margin:-margin] ** 2).mean() / sc) for g in grid]
                i = int(np.argmin(rr))
                row["cm1"][str(Tn)] = {"s_km_per_min": float(grid[i]) * site_km / dt_min, "residual": rr[i]}
            # gust-front speed of CM1 over the next 30 min (surface theta')
            ith = norm.vars.index("th") * d.Z
            sc_th = float(norm.scale[norm.vars.index("th")])
            fronts = [main_pool_front_km(d.full(0, t + j)[0, ith].numpy() * sc_th) for j in range(0, 31, 5)]
            fr = np.array(fronts, float)
            row["cm1_front_speed_km_per_min"] = float(np.polyfit(np.arange(0, 31, 5)[np.isfinite(fr)], fr[np.isfinite(fr)], 1)[0]) \
                if np.isfinite(fr).sum() >= 3 else float("nan")
            # refine a travelling wave (T = 10) and decode its cold pool
            rz = relative_equilibrium(prop_d, z[t].clone(), 10, halo, margin, refine_z=True, iters=a.refine_iters, lam=1.0)
            with torch.no_grad():
                X = (ae.decode(rz["Z"].float()[None])[0] * torch.tensor(norm.channel_scale(d.Z), dtype=torch.float32)[:, None])
                Xp = X.reshape(len(norm.vars), d.Z, -1).numpy()
            row["refined_T10"] = {"residual_before": rz["residual_shift_fit"], "residual_after": rz["residual"],
                                  "s_km_per_min": rz["s_sites_per_step"] * site_km / dt_min,
                                  "cold_pool_C": float(np.nanmax(cold_pool_intensity(Xp, base)))}
            sh = man.get(n, {}).get("shear_scale", 1.0 if n.startswith("ens_A2_") else float("nan"))
            row["shear_scale"] = sh
            row["du_0_2p5km_m_s"] = du_base * sh if np.isfinite(sh) else float("nan")
            rows.append(row)
            m10, c10 = row["model"]["10"], row["cm1"]["10"]
            log(f"[Q2] {n} t {tm:.0f} min: T=10 speed model {m10['s_km_per_min']:+.3f} / CM1 latent {c10['s_km_per_min']:+.3f} / "
                f"CM1 front {row['cm1_front_speed_km_per_min']:+.3f} km/min; residual model {m10['residual']:.3f} CM1 {c10['residual']:.3f}; "
                f"refined {row['refined_T10']['residual_before']:.3f} -> {row['refined_T10']['residual_after']:.3f}, C "
                f"{row['refined_T10']['cold_pool_C']:.1f} m/s, du {row['du_0_2p5km_m_s']:.1f} m/s  [{time.time() - t_start:.0f}s]")
    res["Q2"] = rows
    # residual against T, mean over storms and times (a dip = a period)
    Rm = np.array([[r["model"][str(Tn)]["residual"] for Tn in TS] for r in rows])
    Rc = np.array([[r["cm1"][str(Tn)]["residual"] for Tn in TS] for r in rows])
    res["Q3"] = {"T": list(TS), "model_residual_mean": Rm.mean(0).tolist(), "cm1_residual_mean": Rc.mean(0).tolist()}
    log("[Q3] residual of 'repeats after T min up to a shift' (mean over states): T " + " ".join(f"{t:5d}" for t in TS))
    log("     model " + " ".join(f"{v:5.3f}" for v in Rm.mean(0)))
    log("     CM1   " + " ".join(f"{v:5.3f}" for v in Rc.mean(0)))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axs[0].plot(TS, Rm.mean(0), "o-", label="propagator"); axs[0].plot(TS, Rc.mean(0), "s-", label="CM1 (encoded)")
    axs[0].set_xlabel("T (min)"); axs[0].set_ylabel("relative residual"); axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_title("Q3: does the state repeat after T, up to a shift?")
    sm = [r["model"]["10"]["s_km_per_min"] for r in rows]; sc_ = [r["cm1_front_speed_km_per_min"] for r in rows]
    axs[1].plot(sc_, sm, "o"); lim = [np.nanmin(sc_ + sm) - 0.05, np.nanmax(sc_ + sm) + 0.05]
    axs[1].plot(lim, lim, ":", color="0.5"); axs[1].set_xlabel("CM1 gust-front speed (km/min)"); axs[1].set_ylabel("propagator travelling speed (km/min)")
    axs[1].set_title("Q2: propagation speed")
    fig.savefig(out / "fig_q2_q3.png", dpi=150)
    (out / "q.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[Q] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
