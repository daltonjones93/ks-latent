"""10d: what the nested-dropout latent channels of the pool's models mean physically (one job, all models on the same storms).

For each model (run directory) on the first --n-test storms of the first model's test split:
 1 ordering     reconstruction with only the FIRST m channels (the rest zeroed, as in nested-dropout training) against only the
                LAST m channels; R^2 per variable = 1 - err / var(data), interior sites.
 2 covariates   Pearson correlation of every channel z_k (per site) with per-site physical quantities: column means of u, w, th,
                qv, p, qliq, qice; surface th; column max and min of w; column-integrated condensate. Ridge R^2 of z_k on all of
                them (fit on half the storms, scored on the rest).
 3 scale        power-weighted wavelength (km) of each channel along x (anomaly about the domain mean).
 4 footprint    decode(z + std_k at one site) - decode(z) for channels 0..5 at the strongest-updraft site of mature frames,
                aligned on that site and averaged (signed), as x-z sections per variable in physical units.
 5 dynamics     30-min propagator rollouts with the learned noise set to zero (the central forecast), true halo, with channel set K
                FROZEN at its t0 value or FORCED to the encoded truth; all-variable decoded error ratio to the free rollout.
 6 consistency  |correlation| between channel i of model 0 and channel j of model r (i, j < 8), over all sites and frames.
 7 hovmoller    z_0..z_3 (x, t) of every model and surface th, column-max w, condensate path (x, t) for one storm.
  python scripts/cm1/latent_physics.py --runs R0 R1 R2 R3 --source cache:~/cm1_cache+~/cm1_cache_A2+~/cm1_cache_B --out ~/cm1_da/latent_physics
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _with_true_halo  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

MS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
FOOT_CH = [0, 1, 2, 3, 4, 5]
SETS = {**{f"c{k}": [k] for k in range(8)}, "c16": [16], "c31": [31], "g0-3": [0, 1, 2, 3], "g4-7": [4, 5, 6, 7],
        "g8-15": list(range(8, 16)), "g16-31": list(range(16, 32))}
COV = ["mean u", "mean w", "mean th", "mean qv", "mean p", "mean qliq", "mean qice", "surface th", "max w", "min w", "condensate path"]
HW = 15        # footprint half-width (sites)


def covariates(u, P, p, Z):
    """u (T, C, NX) normalized -> (T, P, len(COV)) per-site covariates."""
    T = u.shape[0]; V = len(VARS)
    x = u.reshape(T, V, Z, P, p)
    m = x.mean((2, 4))                                                     # (T, V, P)
    iw = VARS.index("w")
    out = [m[:, i] for i in range(V)]
    out += [x[:, VARS.index("th"), 0].mean(-1), x[:, iw].amax((1, 3)), x[:, iw].amin((1, 3)),
            (x[:, VARS.index("qliq")] + x[:, VARS.index("qice")]).mean((1, 3))]
    return torch.stack(out, -1)                                            # (T, P, K)


@torch.no_grad()
def roll(prop, z, t0, n, halo, K, mode):
    s = z[t0]; out = [s]
    eps = torch.zeros((1, prop.cfg.noise_channels, z.shape[1])) if prop.cfg.noise_channels else None
    for j in range(n):
        seen = _with_true_halo(s, z[t0 + j], halo)
        s = _with_true_halo(prop(seen[None], eps=eps)[0], z[t0 + j + 1], halo)
        if K and mode == "frozen":
            s = s.clone(); s[:, K] = z[t0][:, K]
        elif K and mode == "forced":
            s = s.clone(); s[:, K] = z[t0 + j + 1][:, K]
        out.append(s)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=10); ap.add_argument("--stride", type=int, default=3); ap.add_argument("--t-lo", type=int, default=40)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--lag", type=int, default=30)
    ap.add_argument("--hov-storm", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t_start = time.time(); torch.manual_seed(0)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cpu")
    log = lambda s: print(f"[latphys] {s} ({time.time() - t_start:.0f}s)", flush=True)
    models = [load_run(Path(r).expanduser(), dev) for r in a.runs]
    names = models[0][4]["test"][: a.n_test]
    files = member_files(a.source)
    norm0 = models[0][2]
    for m in models[1:]:
        if not np.allclose(np.asarray(m[2].scale), np.asarray(norm0.scale)):
            raise ValueError("models use different normalizers; this script assumes one")
    ae0 = models[0][0]; p = ae0.cfg.patch; c = ae0.cfg.latent_channels
    data = {nm: ColumnData([load_member(files[nm])], norm0, dev) for nm in names}
    Z = data[names[0]].Z; V = len(VARS)
    zk = BaseState.from_sounding().z / 1000.0
    scale = np.asarray(norm0.scale, float)
    res = {"runs": [Path(r).name for r in a.runs], "names": names, "vars": list(VARS), "cov": COV, "ms": MS, "sets": SETS,
           "foot_ch": FOOT_CH, "z_km": zk.tolist(), "scale": scale.tolist(), "per_model": []}
    lat_all = []
    for ri, (ae, prop, norm, cfg, split) in enumerate(models):
        halo, m_sites = cfg["stage_b"]["halo_sites"], ae.cfg.reach_sites
        mar = halo + m_sites
        Zs, Us, Cs, full = [], [], [], {}
        for nm in names:
            with torch.no_grad():
                z = encode_members(ae, data[nm])[0]
            full[nm] = z
            ts = list(range(a.t_lo, z.shape[0], a.stride))
            u = data[nm].full(0, slice(None))[ts]
            Zs.append(z[ts]); Us.append(u); Cs.append(covariates(u, z.shape[1], p, Z))
        Zall, Uall, Call = torch.cat(Zs), torch.cat(Us), torch.cat(Cs)
        N, P, _ = Zall.shape
        lat_all.append(Zall)
        log(f"{res['runs'][ri]}: encoded {N} frames")
        R = {}
        # 1 ordering
        sl = slice(mar * p, Uall.shape[-1] - mar * p)
        idx = np.linspace(0, N - 1, min(N, 80)).astype(int)
        zz, uu = Zall[idx], Uall[idx]
        uvar = uu[..., sl].reshape(len(idx), V, Z, -1).var((0, 2, 3), unbiased=False).numpy()
        def r2(zm):
            with torch.no_grad():
                e = torch.cat([((ae.decode(zm[i:i + 16]) - uu[i:i + 16]) ** 2)[..., sl] for i in range(0, len(idx), 16)])
            e = e.reshape(len(idx), V, Z, -1).mean((0, 2, 3)).numpy()
            return (1 - e / uvar).tolist(), float(1 - e.sum() / uvar.sum())
        first, last = {}, {}
        for m in MS:
            keep = torch.zeros(c); keep[:m] = 1; first[m] = r2(zz * keep)
            keep = torch.zeros(c); keep[c - m:] = 1; last[m] = r2(zz * keep)
        R["prefix_first"] = {str(m): {"by_var": v[0], "all": v[1]} for m, v in first.items()}
        R["prefix_last"] = {str(m): {"by_var": v[0], "all": v[1]} for m, v in last.items()}
        log(f"  ordering: first-1 R2 {first[1][1]:.2f}, first-4 {first[4][1]:.2f}, last-4 {last[4][1]:.2f}, all {first[32][1]:.3f}")
        # 2 covariates
        Pint = slice(mar, P - mar)
        Cf = Call[:, Pint].reshape(-1, len(COV)); Cs_ = (Cf - Cf.mean(0)) / (Cf.std(0) + 1e-9)
        Zf = Zall[:, Pint].reshape(-1, c); Zs_ = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
        R["corr"] = ((Zs_.T @ Cs_) / len(Zs_)).tolist()                     # (c, K)
        half = sum(len(x) for x in Zs[: len(Zs) // 2]) * (P - 2 * mar)
        if half == 0:
            half = len(Cs_) // 2                                            # one storm (smoke): split its frames
        Xtr, Xte = Cs_[:half], Cs_[half:]
        W = torch.linalg.solve(Xtr.T @ Xtr + 1e-3 * len(Xtr) * torch.eye(Xtr.shape[1]), Xtr.T @ Zs_[:half])
        R["ridge_r2"] = (1 - ((Zs_[half:] - Xte @ W) ** 2).mean(0) / Zs_[half:].var(0)).tolist()
        R["std"] = Zf.std(0).tolist()
        # 3 scale
        an = Zall - Zall.mean(1, keepdim=True)
        S = (torch.fft.rfft(an, dim=1).abs() ** 2).mean(0)[1:]                 # (P/2, c)
        wl = (P * p * 0.25 / torch.arange(1, S.shape[0] + 1)).double()
        R["wavelength_km"] = (1 / ((S.double() / wl[:, None]).sum(0) / S.double().sum(0))).tolist()
        R["frac_var_domain_mean"] = (Zall.mean(1).var(0) / Zall.reshape(-1, c).var(0)).tolist()
        # 4 footprint
        std = Zf.std(0)
        cases = []
        rng = np.random.default_rng(0)
        for mi in range(len(Zs)):
            wmax = Call[sum(len(x) for x in Zs[:mi]):sum(len(x) for x in Zs[:mi + 1]), :, COV.index("max w")]
            for t in rng.choice(np.arange(len(Zs[mi]) // 3, len(Zs[mi])), size=3, replace=False):
                s0 = int(wmax[t, mar + HW:P - mar - HW].argmax()) + mar + HW
                cases.append((mi, int(t), s0))
        foot = np.zeros((len(FOOT_CH), V, Z, (2 * HW + 1) * p))
        for ci, k in enumerate(FOOT_CH):
            for mi, t, s0 in cases:
                z0 = Zs[mi][t:t + 1]
                zp = z0.clone(); zp[0, s0, k] += std[k]
                with torch.no_grad():
                    dd = (ae.decode(zp) - ae.decode(z0))[0].reshape(V, Z, -1)
                foot[ci] += dd[:, :, (s0 - HW) * p:(s0 + HW + 1) * p].numpy()
            foot[ci] /= len(cases)
        foot *= scale[None, :, None, None]                                      # physical units
        np.save(out / f"foot_{ri}.npy", foot.astype(np.float32))
        R["foot_n_cases"] = len(cases)
        log(f"  footprints from {len(cases)} cases")
        # 5 dynamics
        dyn = {"free": [], "frozen": {k: [] for k in SETS}, "forced": {k: [] for k in SETS}, "persistence": []}
        for nm in names:
            z = full[nm]; d = data[nm]
            for t0 in [int(x) for x in a.t0.split(",")]:
                if t0 + a.lag + 1 > z.shape[0]:
                    continue
                ut = d.full(0, t0 + a.lag)[0]
                ev = lambda s: float(((ae.decode(s[None])[0] - ut)[:, mar * p:-mar * p] ** 2).mean())
                with torch.no_grad():
                    dyn["free"].append(ev(roll(prop, z, t0, a.lag, halo, [], "free")[-1]))
                    dyn["persistence"].append(ev(z[t0]))
                    for k, K in SETS.items():
                        dyn["frozen"][k].append(ev(roll(prop, z, t0, a.lag, halo, K, "frozen")[-1]))
                        dyn["forced"][k].append(ev(roll(prop, z, t0, a.lag, halo, K, "forced")[-1]))
        f = np.array(dyn["free"])
        R["dyn"] = {"free": float(f.mean()), "persistence": float(np.mean(dyn["persistence"])), "n": len(f),
                    "frozen_ratio": {k: float(np.mean(np.array(v) / f)) for k, v in dyn["frozen"].items()},
                    "forced_ratio": {k: float(np.mean(np.array(v) / f)) for k, v in dyn["forced"].items()}}
        log(f"  dynamics: frozen c0 {R['dyn']['frozen_ratio']['c0']:.2f}, forced 0-3 {R['dyn']['forced_ratio']['g0-3']:.2f}")
        # 7 hovmoller
        nm = names[a.hov_storm]
        np.save(out / f"hov_z_{ri}.npy", full[nm][:, :, :4].numpy().astype(np.float32))
        if ri == 0:
            u = data[nm].full(0, slice(None))
            cv = covariates(u, full[nm].shape[1], p, Z).numpy()
            x = u.reshape(u.shape[0], V, Z, -1)
            hov = {"surface th": (x[:, VARS.index("th"), 0] * scale[VARS.index("th")]).numpy(),
                   "max w": (x[:, VARS.index("w")].amax(1) * scale[VARS.index("w")]).numpy(),
                   "condensate": ((x[:, VARS.index("qliq")] * scale[VARS.index("qliq")]) ** 2
                                  + (x[:, VARS.index("qice")] * scale[VARS.index("qice")]) ** 2).mean(1).numpy()}
            np.savez(out / "hov_phys.npz", **hov, cov=cv)
            res["hov_storm"] = nm; res["dt_min"] = float(data[nm].dt) / 60.0
        res["per_model"].append(R)
    # 6 consistency
    cons = []
    for r in range(1, len(models)):
        A = lat_all[0].reshape(-1, c)[:, :8]; B = lat_all[r].reshape(-1, c)[:, :8]
        A = (A - A.mean(0)) / A.std(0); B = (B - B.mean(0)) / B.std(0)
        cons.append((A.T @ B / len(A)).tolist())
    res["consistency"] = cons
    json.dump(res, open(out / "latent_physics.json", "w"))
    log("done")


if __name__ == "__main__":
    main()
