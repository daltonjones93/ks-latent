"""L2 / L2c / SM5: CM1 error growth from twin runs, against the surrogate.

Twin pairs (cluster/cm1 `make_ensemble.py --tiny-pert`): control and perturbed runs identical except for a
tiny theta field at t = 0. Per pair and time, the difference in the model's normalized units (all channels,
and per variable). Reported:
  - growth rates: the slope of ln(difference) over windows (per minute; doubling time = ln 2 / rate),
    only while the difference is below 10% of its saturation level (still small);
  - saturation: the difference between INDEPENDENT control runs (two equally likely storms), whose nMSE / 2
    approximates the error of the best possible deterministic forecast at long lead (L2c);
  - amplitude check: 1e-3 K against 1e-4 K pairs (same members) should give the same rates;
  - SM5: from the encoded truth at --t0-min, the spread growth of the stochastic surrogate's samples and the
    growth of a tiny latent perturbation in the deterministic surrogate, on the same axes.
  python scripts/cm1/twin_growth.py --ctrl ~/cm1_bundle/ens_L2_ctrl --pert ~/cm1_bundle/ens_L2_pert \
      --pert4 ~/cm1_bundle/ens_L2_pert4 --run ~/cm1_runs/full_v1_stagec --run-stoch ~/cm1_runs/full_v1_stagec_crps --out ~/cm1_da/l2
"""
import argparse
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
from da_twin import load_run, load_stochastic_run  # noqa: E402
from ks_latent.cm1.data import ColumnData, member_from_cm1io  # noqa: E402
from ks_latent.cm1.io import load_grads_run  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

WINDOWS = ((5, 15), (15, 30), (30, 60), (60, 120), (120, 180), (180, 240))


def load(ens: Path, mem: str):
    run = load_grads_run(ens / "members" / mem / "cm1out_s.ctl")
    return run["X"].astype(np.float64), run["t"], run


def diff_curves(Xa, Xb, scale) -> tuple[np.ndarray, np.ndarray]:
    """normalized RMS difference over time: all channels (T,) and per variable (T, V)."""
    T = min(Xa.shape[0], Xb.shape[0])
    d = ((Xa[:T] - Xb[:T]) / scale[None, :, None, None]) ** 2           # (T, V, Z, NX)
    return np.sqrt(d.mean(axis=(1, 2, 3))), np.sqrt(d.mean(axis=(2, 3)))


def rates(t_min: np.ndarray, d: np.ndarray, d_sat: float) -> dict:
    out = {}
    for w0, w1 in WINDOWS:
        sel = (t_min >= w0) & (t_min <= w1) & (d > 0) & (d < 0.1 * d_sat)
        out[f"{w0}-{w1}"] = float(np.polyfit(t_min[sel], np.log(d[sel]), 1)[0]) if sel.sum() >= 5 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--pert", required=True)
    ap.add_argument("--pert4", default="")
    ap.add_argument("--run", required=True, help="deterministic surrogate (its normalizer defines the units)")
    ap.add_argument("--run-stoch", default="")
    ap.add_argument("--t0-min", type=float, default=30.0, help="start of the surrogate comparison (SM5)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    ctrl, pert = Path(a.ctrl).expanduser(), Path(a.pert).expanduser()
    dev = torch.device("cpu")
    ae, prop, norm, cfg, _ = load_run(Path(a.run).expanduser(), dev)
    scale = np.asarray(norm.scale, float)
    mems = sorted(p.name for p in (ctrl / "members").glob("mem*") if (p / "cm1.done").exists()
                  and (pert / "members" / p.name / "cm1.done").exists())
    if not mems:
        raise ValueError("no finished control/perturbed pairs")
    res = {"members": mems, "pairs": {}, "pairs_1e-4": {}, "independent": []}
    curves, ctrl_X = {}, {}
    for mm in mems:
        Xc, t, _ = load(ctrl, mm)
        Xp, _, _ = load(pert, mm)
        d, dv = diff_curves(Xc, Xp, scale)
        curves[mm] = (t[:d.size] / 60.0, d, dv)
        ctrl_X[mm] = Xc
        print(f"[L2] {mm}: twin difference at 0/10/20/30/60/120 min: " +
              " ".join("%.1e" % d[int(k)] for k in (0, 10, 20, 30, 60, 120) if k < d.size), flush=True)
    # saturation: independent control runs (consecutive pairs)
    ind = []
    for m1, m2 in zip(mems[:-1], mems[1:]):
        d, _ = diff_curves(ctrl_X[m1], ctrl_X[m2], scale)
        ind.append(d)
    T = min(len(x) for x in ind)
    d_ind = np.mean([x[:T] for x in ind], axis=0)
    res["independent"] = d_ind.tolist()
    d_sat = float(np.max(d_ind))
    for mm in mems:
        tm, d, dv = curves[mm]
        res["pairs"][mm] = {"t_min": tm.tolist(), "d": d.tolist(), "d_by_var": dv.tolist(), "rates": rates(tm, d, d_sat)}
    if a.pert4:
        p4 = Path(a.pert4).expanduser()
        for mm in sorted(p.name for p in (p4 / "members").glob("mem*") if (p / "cm1.done").exists()):
            Xp, _, _ = load(p4, mm)
            d, dv = diff_curves(ctrl_X[mm], Xp, scale)
            tm = curves[mm][0][:d.size]
            res["pairs_1e-4"][mm] = {"d": d.tolist(), "rates": rates(tm, d, d_sat)}
    # summary of CM1 growth
    print(f"\n[L2] saturation (independent storms, max over time): normalized RMS {d_sat:.3f} (nMSE {d_sat ** 2:.3f}, "
          f"half of it {d_sat ** 2 / 2:.3f} = the best deterministic forecast's error at long lead, L2c)")
    print("[L2] growth rate (per min) by window, mean +- sd over pairs [doubling time]; only while < 10% of saturation:")
    summ = {}
    for w in res["pairs"][mems[0]]["rates"]:
        r = np.array([res["pairs"][m]["rates"][w] for m in mems])
        r4 = np.array([v["rates"][w] for v in res["pairs_1e-4"].values()]) if res["pairs_1e-4"] else np.array([])
        ok = np.isfinite(r)
        if ok.any():
            mu = float(r[ok].mean())
            summ[w] = {"rate": mu, "sd": float(r[ok].std()), "n": int(ok.sum()),
                       "rate_1e-4": float(np.nanmean(r4)) if r4.size and np.isfinite(r4).any() else float("nan")}
            print(f"   {w:>8s} min: {mu:.3f} +- {r[ok].std():.3f} ({ok.sum()} pairs) [{np.log(2) / mu if mu > 0 else np.inf:.1f} min]"
                  + (f"   1e-4 K pairs: {summ[w]['rate_1e-4']:.3f}" if r4.size else ""))
        else:
            print(f"   {w:>8s} min: saturated or too few points")
    res["summary"] = summ
    # SM5: surrogate error growth from the encoded truth at t0
    sur = {}
    m0 = mems[0]
    run0 = load_grads_run(ctrl / "members" / m0 / "cm1out_s.ctl")
    d0 = ColumnData([member_from_cm1io(run0, m0)], norm, dev)
    dt_min = d0.dt / 60
    t0 = int(round(a.t0_min / dt_min))
    nl = min(d0.T[0] - t0 - 1, int(round(180 / dt_min)))
    halo = cfg["stage_b"]["halo_sites"]
    with torch.no_grad():
        z = encode_members(ae, d0)[0]
        base = rollout_from(prop, z, t0, nl, halo, boundary="persistence")[:, 0]
        zp = z.clone()
        zp[t0] = zp[t0] + 1e-3 * torch.randn(zp[t0].shape, generator=torch.Generator().manual_seed(0))
        pertr = rollout_from(prop, zp, t0, nl, halo, boundary="persistence")[:, 0]
        dd = [float((((ae.decode(base[j:j + 1]) - ae.decode(pertr[j:j + 1])) ** 2).mean()).sqrt()) for j in range(nl)]
        sur["det_tiny_latent_perturbation"] = dd
        if a.run_stoch:
            ae_s, prop_s, norm_s, cfg_s, _ = load_stochastic_run(Path(a.run_stoch).expanduser(), dev)
            zs = encode_members(ae_s, d0)[0]
            la = rollout_from(prop_s, zs, t0, nl, halo, 8, gen=torch.Generator().manual_seed(0), boundary="persistence")
            sp = []
            for j in range(nl):
                dec = ae_s.decode(la[j])                                                 # (8, C, NX)
                sp.append(float((2 * dec.var(dim=0, unbiased=True).mean()).sqrt()))      # RMS pair difference
            sur["stoch_pair_difference"] = sp
    res["surrogate"] = sur
    res["surrogate_t0_min"] = a.t0_min
    (out / "l2.json").write_text(json.dumps(res, indent=1))
    # figure
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for i, mm in enumerate(mems):
        tm, d, _ = curves[mm]
        ax.semilogy(tm, d, color="C0", lw=0.8, alpha=0.7, label="CM1 twins, 1e-3 K" if i == 0 else None)
    for i, v in enumerate(res["pairs_1e-4"].values()):
        ax.semilogy(curves[mems[0]][0][:len(v["d"])], v["d"], color="C3", lw=0.8, alpha=0.8, label="CM1 twins, 1e-4 K" if i == 0 else None)
    ax.semilogy(np.arange(T) * dt_min, d_ind, "k-", lw=1.5, label="independent CM1 storms (saturation)")
    ts = a.t0_min + np.arange(1, nl + 1) * dt_min
    ax.semilogy(ts, sur["det_tiny_latent_perturbation"], color="C1", lw=1.5, label="deterministic surrogate, tiny latent perturbation")
    if "stoch_pair_difference" in sur:
        ax.semilogy(ts, sur["stoch_pair_difference"], color="C2", lw=1.5, label="stochastic surrogate, sample pairs")
    ax.set_xlabel("time (min)"); ax.set_ylabel("normalized RMS difference"); ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("Error growth: CM1 twins against the surrogate", fontsize=9)
    fig.savefig(out / "fig_l2_growth.png", dpi=130)
    print(f"[L2] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
