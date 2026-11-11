"""Characterise the leading latent channels (pre-registration 5q): what is channel 0 (and 1-3), spatially, vertically, temporally?

Tests (test split of the run, normalised units, 8-column sites; one site = p columns):
 A  covariates   Pearson correlation of z_k at a site with per-site physical covariates (column mean / level-0 / column max / column min of each
                 variable over the site's columns) and the R^2 of a ridge fit of z_k on all covariates (fit on half the members, scored on the rest).
 B  scale        spatial spectrum of z_k along the sites (anomaly about the domain mean), power-weighted wavelength, fraction of variance at wavelengths
                 > 20 km; fraction of the total variance that is the domain mean (varies in time only); e-folding spatial lag and temporal lag.
 C  footprint    decode(z + d e_k at one site) - decode(z), d = 1 std of channel k: horizontal footprint (km, 50% and 10% of the peak) and vertical profile
                 per variable; the uniform (all-sites) perturbation response, as a fraction of the data's rms per variable.
 D  explained    variance of the decoded field explained by keeping only channel set K (others at their dataset mean) and increase of error when K
                 alone is set to its mean, per variable.
  python scripts/cm1/channel0_char.py --run ~/cm1_runs/fpc_nested_stagec --source cache:~/cm1_cache,~/cm1_cache_B --out ~/cm1_da/ch0_nested
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

CH = [0, 1, 2, 3, 4, 8, 16, 31]
SETS = {"0": [0], "0-3": [0, 1, 2, 3], "0-7": list(range(8)), "all-but-0": None}


def site_cov(u, P, p, V, Z):
    """u (T, C, NX) -> (T, P, 4V) covariates: column mean, level-0, max, min per variable."""
    T = u.shape[0]
    x = u.reshape(T, V, Z, P, p)
    m = x.mean((2, 4)); l0 = x[:, :, 0].mean(-1); mx = x.amax((2, 4)); mn = x.amin((2, 4))
    return torch.cat([m, l0, mx, mn], 1).permute(0, 2, 1)       # (T, P, 4V)


def acf_time(z, maxlag=40):
    """z (T, P) one member: pooled autocorrelation of the temporally demeaned series (weights sites by variance)."""
    a = z - z.mean(0, keepdim=True)
    v = (a * a).sum()
    return [float((a[:-k] * a[k:]).sum() / v) if k else 1.0 for k in range(maxlag + 1)]


def efold(r):
    for k, x in enumerate(r):
        if x < np.exp(-1):
            return k - 1 + (r[k - 1] - np.exp(-1)) / max(r[k - 1] - x, 1e-12)
    return float(len(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--names-prefix", default=""); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--t-lo", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    p, halo, m_sites = ae.cfg.patch, cfg["stage_b"]["halo_sites"], ae.cfg.reach_sites
    files = member_files(a.source)
    names = sorted(n for n in files if n.startswith(a.names_prefix))[: a.n_test] if a.names_prefix else split["test"][: a.n_test]
    c = ae.cfg.latent_channels; V = len(VARS)
    km_site = p * 0.25                                      # 250 m columns (2 km site for p = 8)
    Zs, Cov, U = [], [], []
    ac_t, ac_dm = {k: [] for k in CH}, {k: [] for k in CH}
    for name in names:
        d = ColumnData([load_member(files[name])], norm, dev)
        with torch.no_grad():
            z = encode_members(ae, d)[0]
        P = z.shape[1]; Z = d.Z
        ts = list(range(a.t_lo, z.shape[0], a.stride))
        u = d.full(0, slice(None))
        Zs.append(z[ts]); Cov.append(site_cov(u[ts], P, p, V, Z)); U.append(u[ts])
        for k in CH:
            ac_t[k].append(acf_time(z[a.t_lo:, :, k])); ac_dm[k].append(acf_time(z[a.t_lo:, :, k].mean(1, keepdim=True)))
        print(f"[ch0] {name} encoded ({time.time() - t_start:.0f}s)", flush=True)
    Zall = torch.cat(Zs); Uall = torch.cat(U); Call = torch.cat(Cov)           # (N,P,c) (N,C,NX) (N,P,4V)
    N, P, _ = Zall.shape; Z = Uall.shape[1] // V
    res = {"args": vars(a), "names": names, "vars": list(VARS), "channels": CH, "km_site": km_site, "N": N, "P": P}
    ztd = Zall.reshape(-1, c).std(0)                                           # per-channel std
    # ---- A covariates
    cov_names = [f"{kind}:{v}" for kind in ("mean", "lev0", "max", "min") for v in VARS]
    nm = len(names); half = nm // 2; rows = np.cumsum([0] + [len(x) for x in Zs])
    sel_tr = slice(0, rows[half]); sel_te = slice(rows[half], rows[-1])
    A = {"cov_names": cov_names}
    Cf = Call.reshape(-1, Call.shape[-1]); Cs = (Cf - Cf.mean(0)) / (Cf.std(0) + 1e-9)
    for k in CH:
        zk = Zall[:, :, k].reshape(-1); zs = (zk - zk.mean()) / (zk.std() + 1e-9)
        r = (Cs * zs[:, None]).mean(0)
        o = torch.argsort(r.abs(), descending=True)[:6]
        n_per = P
        Xtr = Cs[sel_tr.start * n_per:sel_tr.stop * n_per]; ytr = zs[sel_tr.start * n_per:sel_tr.stop * n_per]
        Xte = Cs[sel_te.start * n_per:sel_te.stop * n_per]; yte = zs[sel_te.start * n_per:sel_te.stop * n_per]
        W = torch.linalg.solve(Xtr.T @ Xtr + 1e-2 * len(Xtr) * torch.eye(Xtr.shape[1]), Xtr.T @ ytr)
        r2 = float(1 - ((yte - Xte @ W) ** 2).mean() / yte.var())
        A[f"c{k}"] = {"top": [(cov_names[i], float(r[i])) for i in o], "ridge_R2_heldout": r2}
    res["A_covariates"] = A
    # ---- B scale
    B = {}
    for k in CH:
        zk = Zall[:, :, k]
        dm = zk.mean(1, keepdim=True); an = zk - dm
        tot = float(zk.var()); frac_dm = float(dm.var() / tot)
        S = (torch.fft.rfft(an, dim=1).abs() ** 2).mean(0)[1:]
        wl = P * km_site / torch.arange(1, len(S) + 1)
        cum_long = float(S[wl > 20].sum() / S.sum()) if (wl > 20).any() else 0.0
        cen = float((S * (1 / wl)).sum() / S.sum())
        ac = torch.fft.irfft(torch.fft.rfft(an, dim=1).abs() ** 2, n=P, dim=1).mean(0); ac = (ac / ac[0]).numpy()
        lag = efold(ac[: P // 2].tolist())
        B[f"c{k}"] = {"frac_var_domain_mean": frac_dm, "frac_anom_power_wavelength_gt20km": cum_long,
                      "power_weighted_wavelength_km": 1 / cen, "spatial_efold_km": lag * km_site,
                      "temporal_efold_min_site": efold(np.mean(ac_t[k], 0).tolist()),
                      "temporal_efold_min_domain_mean": efold(np.mean(ac_dm[k], 0).tolist()),
                      "std": float(ztd[k])}
    res["B_scale"] = B
    # ---- C footprint
    rng = np.random.default_rng(0)
    cases = []
    off = 0
    for mi, z in enumerate(Zs):
        w = Cov[mi][:, :, VARS.index("w")]                                    # column-mean w
        wmax = U[mi].reshape(len(z), V, Z, -1)[:, VARS.index("w")].amax(1)   # (T, NX) column max
        pick = rng.choice(len(z), size=min(4, len(z)), replace=False)
        for t in pick:
            s0 = int(wmax[t].reshape(P, p).amax(1).argmax()) if True else 0
            if max(halo + m_sites + 12, 16) < s0 < P - max(halo + m_sites + 12, 16):
                cases.append((mi, int(t), s0))
    C = {}
    cs = max(1, len(cases))
    for k in CH:
        hor = np.zeros((V, P)); ver = np.zeros((V, Z)); unif = np.zeros((V, Z)); n = 0
        for mi, t, s0 in cases:
            z0 = Zs[mi][t:t + 1]
            with torch.no_grad():
                base = ae.decode(z0)
                zp = z0.clone(); zp[0, s0, k] += ztd[k]
                dp = (ae.decode(zp) - base)[0].reshape(V, Z, P, p)
                zu = z0.clone(); zu[0, :, k] += ztd[k]
                du = (ae.decode(zu) - base)[0].reshape(V, Z, P, p)
            hor += (dp ** 2).mean((1, 3)).sqrt().numpy()                       # (V, P)
            ver += (dp[:, :, max(s0 - 5, 0):s0 + 6] ** 2).mean((2, 3)).sqrt().numpy()
            sl = slice(halo + m_sites, P - halo - m_sites)
            unif += (du[..., sl, :] ** 2).mean((2, 3)).sqrt().numpy(); n += 1
            hs0 = s0
        hor /= max(n, 1); ver /= max(n, 1); unif /= max(n, 1)
        # horizontal footprint relative to the site (average over cases requires alignment: recompute aligned)
        al = np.zeros((V, 2 * 15 + 1)); n2 = 0
        for mi, t, s0 in cases:
            z0 = Zs[mi][t:t + 1]
            with torch.no_grad():
                zp = z0.clone(); zp[0, s0, k] += ztd[k]
                dp = (ae.decode(zp) - ae.decode(z0))[0].reshape(V, Z, P, p)
            al += (dp[:, :, s0 - 15:s0 + 16] ** 2).mean((1, 3)).sqrt().numpy(); n2 += 1
        al /= max(n2, 1)
        foot = {}
        for vi, v in enumerate(VARS):
            pk = al[vi].max()
            if pk < 1e-9:
                foot[v] = {"peak": float(pk)}; continue
            wid = lambda f: float((al[vi] >= f * pk).sum() * km_site)
            foot[v] = {"peak_rms": float(pk), "width50_km": wid(0.5), "width10_km": wid(0.1)}
        datarms = Uall.reshape(N, V, Z, -1).pow(2).mean((0, 3)).sqrt().numpy()
        C[f"c{k}"] = {"footprint": foot, "n_cases": n,
                      "uniform_resp_over_data_rms": {v: float(np.sqrt((unif[vi] ** 2).mean()) / np.sqrt((datarms[vi] ** 2).mean())) for vi, v in enumerate(VARS)},
                      "vertical_resp_profile_single_site": {v: ver[vi].tolist() for vi, v in enumerate(VARS)}}
        print(f"[ch0] footprint c{k} ({time.time() - t_start:.0f}s)", flush=True)
    res["C_footprint"] = C
    # ---- D variance explained
    zmean = Zall.reshape(-1, c).mean(0)
    sl = slice((halo + m_sites) * p, Uall.shape[-1] - (halo + m_sites) * p)
    idx = np.linspace(0, N - 1, min(N, 60)).astype(int)
    D = {}
    with torch.no_grad():
        zz = Zall[idx]; uu = Uall[idx]
        errf = lambda dec: ((dec - uu) ** 2)[..., sl].reshape(len(idx), V, Z, -1).mean((0, 2, 3)).numpy()
        e_full = errf(ae.decode(zz))
        zb = zmean.expand_as(zz).clone(); e_base = errf(ae.decode(zb))
        D["var_full_error"] = e_full.tolist(); D["error_mean_latent"] = e_base.tolist()
        for nmk, K in SETS.items():
            Kl = [k for k in range(c) if k != 0] if K is None else K
            keep = zb.clone(); keep[:, :, Kl] = zz[:, :, Kl]
            drop = zz.clone(); drop[:, :, Kl] = zmean[Kl]
            D[nmk] = {"R2_only_K_by_var": (1 - errf(ae.decode(keep)) / e_base).tolist(),
                      "error_ratio_drop_K_by_var": (errf(ae.decode(drop)) / e_full).tolist()}
    res["D_explained"] = D
    json.dump(res, open(out / "ch0.json", "w"))
    print("[ch0] done", f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
