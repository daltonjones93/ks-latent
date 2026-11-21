"""6a tests: the two-scale model against deterministic and one-model stochastic baselines, on held-out storms.

Models (--runs; any run dir load_run understands, two-scale included); stochastic models are sampled --samples times, deterministic
ones once. From encoded test states at --t0 (frames, 1 min), free rollouts with the true latent on the halo, decoded and scored on the
interior columns at lags --lags (min):
  T-spec   horizontal power spectrum (per variable, mean over levels and samples) / truth's, in wavelength bands (km)
  T-prob   per variable: RMSE of the sample mean, fair CRPS (MAE for one sample), spread / RMSE
  T-large  RMSE of the sample mean after a Gaussian low-pass in x (sigma --lowpass-km) against the low-passed truth
  T-front  |gust-front position error| (km) of each sample's surface theta' (main_pool_front_km), median over samples
  T-div    sample divergence D(k) = rms(sample_i - sample_j) / sqrt(2) (stochastic models only)
With --paired MANIFEST (+ --base / --branches caches) also T-chaos: from the CM1 restart state of each paired storm, the samples' D(k)
against CM1's own tiny-perturbation divergence rms(tiny - ctrl), k = 1..60 min, and the log-growth rate over 15-30 min.
  python scripts/cm1/twoscale_eval.py --runs ~/cm1_runs/ts_v1 ~/cm1_runs/fpc_nested_p8_dft ~/cm1_runs/fpc_nested_stagec \
      --source cache:A+A2+B --out ~/cm1_da/ts_eval
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
from ks_latent.cm1 import cell_losses as cl  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _with_true_halo  # noqa: E402
from ks_latent.cm1.train import crps_elements, encode_members  # noqa: E402
from ks_latent.cm1.verify import main_pool_front_km  # noqa: E402

BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
DX = 0.25


def is_stochastic(prop) -> bool:
    return getattr(prop.cfg, "noise_channels", 0) > 0


@torch.no_grad()
def rollout(prop, z, t0, n, halo, S, seed):
    """z (T, P, c) true latents -> (n + 1, S, P, c) from t0, halo true."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    s = z[t0][None].expand(S, -1, -1).clone(); out = [s]
    for j in range(n):
        seen = _with_true_halo(s, z[t0 + j][None].expand(S, -1, -1), halo)
        try:
            nxt = prop(seen, gen=gen)
        except TypeError:
            nxt = prop(seen)
        s = _with_true_halo(nxt, z[t0 + j + 1][None].expand(S, -1, -1), halo)
        out.append(s)
    return torch.stack(out)


def lowpass(x, sigma_km):
    """Gaussian low-pass along the last axis (open edges: reflect padding)."""
    s = sigma_km / DX
    r = int(np.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / s) ** 2); k = k / k.sum()
    sh = x.shape
    y = torch.nn.functional.pad(x.reshape(-1, 1, sh[-1]), (r, r), mode="reflect")
    return torch.nn.functional.conv1d(y, k.view(1, 1, -1)).reshape(sh)


def band_power(x, Zl):
    """x (..., C, W) -> (..., V, n_bands) mean power per band (levels averaged)."""
    V = len(VARS); W = x.shape[-1]
    f = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs() ** 2               # (..., C, W//2+1)
    f = f.reshape(*x.shape[:-2], V, Zl, -1).mean(-2)
    k = np.arange(f.shape[-1]); wl = np.where(k > 0, W * DX / np.maximum(k, 1), np.inf)
    return torch.stack([f[..., torch.as_tensor((wl >= lo) & (wl < hi))].sum(-1) for lo, hi in BANDS], -1)


def _to_ref(sc, norm, ref_scale):
    """this model's normalized units -> the reference model's"""
    if ref_scale is None:
        return sc
    f = np.array(norm.scale, dtype=float) / ref_scale
    for k in ("rmse", "crps", "spread", "rmse_lowpass"):
        sc[k] = (np.array(sc[k]) * f).tolist()
    for k in ("power", "power_truth"):
        sc[k] = (np.array(sc[k]) * (f ** 2)[:, None]).tolist()
    return sc


def pooled_cases(subs, names, files, a, lags, n, ref_scale, dev, t_start, rn):
    """10a: a multi-model ensemble. Each model (its own autoencoder and propagator) encodes the truth, rolls --samples / len(subs) members and
    decodes them; the members are converted to the first model's normalized units and scored together as one ensemble."""
    split0 = subs[0][4][a.split][: a.n_test]
    for m in subs[1:]:
        if m[4][a.split][: a.n_test] != split0:
            raise ValueError(f"pool {rn}: models have different {a.split} splits")
    ae0, _, norm0, cfg0, _ = subs[0]
    Sm = a.samples // len(subs)
    mar = max((m[3]["stage_b"]["halo_sites"] + m[0].cfg.reach_sites) * m[0].cfg.patch for m in subs)
    cases = []
    for nm in names:
        d0 = ColumnData([load_member(files[nm])], norm0, dev)
        per = []
        for ae, prop, norm, cfg, _ in subs:
            if not is_stochastic(prop):
                raise ValueError(f"pool {rn}: deterministic member model")
            d = ColumnData([load_member(files[nm])], norm, dev)
            z = encode_members(ae, d)[0]
            conv = torch.as_tensor(np.repeat(np.array(norm.scale, float) / np.array(norm0.scale, float), d.Z), dtype=torch.float32, device=dev)[:, None]
            per.append((ae, prop, cfg["stage_b"]["halo_sites"], z, conv))
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > per[0][3].shape[0]:
                continue
            trajs = []
            for j, (ae, prop, halo, z, conv) in enumerate(per):
                zb = z if a.boundary == "truth" else torch.cat([z[:t0 + 1], z[t0:t0 + 1].expand(z.shape[0] - t0 - 1, -1, -1)])
                trajs.append(rollout(prop, zb, t0, n, halo, Sm, seed=t0 + 1000 * j))
            case = {"member": nm, "t0": t0, "lags": {}}
            with torch.no_grad():
                for L in lags:
                    dec = torch.cat([ae.decode(tr[L]) * conv for (ae, _, _, _, conv), tr in zip(per, trajs)])
                    u = d0.full(0, t0 + L)[0]
                    case["lags"][str(L)] = _to_ref(score_case(ae0, dec, u, d0.Z, slice(mar, u.shape[-1] - mar), norm0, a.lowpass_km), norm0, ref_scale)
            cases.append(case)
        print(f"[ts_eval] {rn} {nm} ({time.time() - t_start:.0f}s)", flush=True)
    return cases


def score_case(ae, dec, u_true, Zl, sl, norm, lowpass_km):
    """dec (S, C, NX), u_true (C, NX) -> dict of per-variable metrics on the interior columns `sl`."""
    V = len(VARS)
    d, u = dec[..., sl].double(), u_true[..., sl].double()
    S = d.shape[0]
    mean = d.mean(0)
    per = lambda e: e.reshape(V, Zl, -1).mean((1, 2))
    out = {"rmse": per((mean - u) ** 2).sqrt().tolist(),
           "crps": per(crps_elements(d, u) if S > 1 else (d[0] - u).abs()).tolist(),
           "spread": per(d.var(0, unbiased=True) if S > 1 else torch.zeros_like(u)).sqrt().tolist()}
    # 9a: threshold-weighted CRPS (physical units: w in m/s, condensate as sqrt(g/kg)) and neighbourhood fractions skill score
    sc = torch.as_tensor(norm.channel_scale(Zl), dtype=torch.float64, device=d.device)[:, None]
    dp, up_ = d * sc, u * sc
    fd = cl.exceed_fields(dp, Zl, 2.0, float(np.sqrt(0.1))); ft = cl.exceed_fields(up_, Zl, 2.0, float(np.sqrt(0.1)))
    tw = []
    for a_, b_ in zip(fd, ft):                                   # up, down, liquid, ice exceedances: (S, Z, W) against (Z, W)
        e = crps_elements(a_, b_) if S > 1 else (a_[0] - b_).abs()
        tw.append(float(e.mean()))
    out["tw_crps"] = {"w_up": tw[0], "w_down": tw[1], "liquid": tw[2], "ice": tw[3], "mean": float(np.mean(tw))}
    fss = {}
    for name, fa, fb, thr in (("w", dp[:, cl.var_slice("w", Zl)].abs(), up_[cl.var_slice("w", Zl)].abs(), 3.0),
                              ("q", dp[:, cl.var_slice("qliq", Zl)] ** 2 + dp[:, cl.var_slice("qice", Zl)] ** 2,
                               up_[cl.var_slice("qliq", Zl)] ** 2 + up_[cl.var_slice("qice", Zl)] ** 2, 0.5)):
        bo = (fb > thr).double()
        for r_km in (4, 10, 20):
            K = int(round(r_km / 0.25))
            po = torch.nn.functional.avg_pool1d(bo.reshape(-1, 1, bo.shape[-1]), K, 1)
            vals = []
            for i in range(fa.shape[0]):
                pf = torch.nn.functional.avg_pool1d((fa[i] > thr).double().reshape(-1, 1, fa.shape[-1]), K, 1)
                den = float((pf ** 2).mean() + (po ** 2).mean())
                vals.append(1.0 - float(((pf - po) ** 2).mean()) / den if den > 0 else float("nan"))
            fss[f"{name}@{r_km}"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    out["fss"] = fss
    # 9d: updraft (w > 2 m/s) and downdraft (w < -1.5 m/s) presence: the ensemble fraction as the probability
    feat = {}
    # 9e: every feature of cell_losses.FEATURES (stored units: condensate thresholds as sqrt(q)); "up" / "down" kept as the w pair
    flist = [("up", "w", 1.0, 2.0), ("down", "w", -1.0, 1.5)] + [
        (k, v, sg, float(np.sqrt(th)) if v in ("qliq", "qice") else th) for k, (v, sg, th) in cl.FEATURES.items() if v != "w"]
    for name, fv, sign, thr in flist:
        fsl = cl.var_slice(fv, Zl)
        fm = (sign * dp[:, fsl] > thr).double()                       # (S, Z, W) member masks
        y = (sign * up_[fsl] > thr).double()
        pr = fm.mean(0)
        base = float(y.mean())
        fa_ = pr >= 0.5
        hits, miss, fal = float((fa_ & (y > 0)).sum()), float((~fa_ & (y > 0)).sum()), float((fa_ & (y == 0)).sum())
        kk = torch.round(pr * fm.shape[0]).long().flatten()
        rel_n = torch.bincount(kk, minlength=fm.shape[0] + 1).tolist()
        rel_h = torch.bincount(kk, weights=y.flatten(), minlength=fm.shape[0] + 1).tolist()
        fz = {}
        for r_km in (4, 10):
            K = int(round(r_km / 0.25))
            po = torch.nn.functional.avg_pool1d(y.reshape(-1, 1, y.shape[-1]), K, 1)
            vals = []
            for i in range(fm.shape[0]):
                pf = torch.nn.functional.avg_pool1d(fm[i].reshape(-1, 1, fm.shape[-1]), K, 1)
                den = float((pf ** 2).mean() + (po ** 2).mean())
                vals.append(1.0 - float(((pf - po) ** 2).mean()) / den if den > 0 else float("nan"))
            fz[f"fss@{r_km}"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
        # 9f: the soft probability a deterministic model's cross-entropy sees, sigmoid((sign*x - thr) / soft), binned in 20 bins
        # (counts and hits), to calibrate class weights from the model's own probabilities
        fsoft = cl.FEATURE_SOFT[fv]
        ps = torch.sigmoid((sign * dp[:, fsl] - thr) / fsoft).mean(0)
        kb = torch.clamp((ps * 20).long(), max=19).flatten()
        soft_n = torch.bincount(kb, minlength=20).tolist(); soft_h = torch.bincount(kb, weights=y.flatten(), minlength=20).tolist()
        feat[name] = {"soft_n": soft_n, "soft_hits": soft_h, "brier": float(((pr - y) ** 2).mean()), "base_rate": base, "csi": hits / max(hits + miss + fal, 1.0),
                      "bias": float(pr.sum() / max(float(y.sum()), 1.0)), "rel_n": rel_n, "rel_hits": rel_h, **fz}
    out["feat"] = feat
    out["power"] = band_power(d, Zl).mean(0).tolist(); out["power_truth"] = band_power(u, Zl).tolist()
    lm, lu = lowpass(mean, lowpass_km), lowpass(u, lowpass_km)
    out["rmse_lowpass"] = per((lm - lu) ** 2).sqrt().tolist()
    if S > 1:
        dd = [(d[i] - d[j]).pow(2).mean() for i in range(S) for j in range(i + 1, S)]
        out["divergence"] = float(torch.stack(dd).mean().sqrt() / np.sqrt(2))
    ith = VARS.index("th") * Zl
    sth = float(norm.scale[VARS.index("th")])
    ft = main_pool_front_km((u_true[ith] * sth).double().cpu().numpy()[None])[0]
    fs = main_pool_front_km((dec[:, ith] * sth).double().cpu().numpy())
    out["front_abs_err_km"] = float(np.nanmedian(np.abs(fs - ft))) if np.isfinite(ft) and np.isfinite(fs).any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--lags", default="5,15,30,60")
    ap.add_argument("--lowpass-km", type=float, default=10.0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", choices=["test", "val"], default="test", help="9g: score the validation storms (to choose cutoffs) or the test storms")
    ap.add_argument("--paired", default=None, help="manifest.json of the paired runs (T-chaos)")
    ap.add_argument("--base", default=None); ap.add_argument("--branches", default=None)
    ap.add_argument("--ref-norm-run", default=None,
                    help="8a: run dir whose normalizer.json defines the units: per-variable metrics of models with another normalizer "
                         "(trained on other storms) are converted to these units before averaging (power by the square)")
    ap.add_argument("--global-off", action="store_true", help="7a: also score each globally coupled run with its global features zeroed")
    ap.add_argument("--boundary", choices=["truth", "persistence"], default="truth",
                    help="halo after t0: the true state (default) or held at t0 (nothing after t0 known)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); t_start = time.time(); dev = torch.device(a.device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    models = {}
    for r in a.runs:
        if "=" in r:           # 10a: a pooled multi-model ensemble, name=runA+runB+...: each model gives --samples // n members
            pname, parts = r.split("=", 1)
            subs = [load_run(Path(q).expanduser(), dev) for q in parts.split("+")]
            if len(subs) < 2 or a.samples % len(subs):
                raise ValueError(f"pool {pname}: needs >= 2 models and --samples divisible by their number")
            models[pname] = ("pool", subs)
            continue
        r = Path(r).expanduser(); models[r.name] = load_run(r, dev)
        if a.global_off and getattr(models[r.name][1].cfg, "global_kind", ""):
            m2 = load_run(r, dev); m2[1].global_off = True; models[r.name + "@globaloff"] = m2
    res = {"vars": list(VARS), "bands_km": BANDS, "lags": lags, "boundary": a.boundary, "runs": {}}
    ref_scale = None
    if a.ref_norm_run:
        ref_scale = np.array(json.loads((Path(a.ref_norm_run).expanduser() / "normalizer.json").read_text())["scale"], dtype=float)
    if a.source:
        files = member_files(a.source)
        first = next(iter(models.values()))
        names = (first[1][0] if first[0] == "pool" else first)[4][a.split][: a.n_test]
        for rn, mdl in models.items():
            if mdl[0] == "pool":
                res["runs"][rn] = {"samples": a.samples, "pool": len(mdl[1]), "cases": pooled_cases(mdl[1], names, files, a, lags, n, ref_scale, dev, t_start, rn)}
                continue
            ae, prop, norm, cfg, split = mdl
            if split[a.split][: a.n_test] != names:
                print(f"[ts_eval] {rn}: different test split, skipped"); continue
            halo, p = cfg["stage_b"]["halo_sites"], ae.cfg.patch
            mar = (halo + ae.cfg.reach_sites) * p
            S = a.samples if is_stochastic(prop) else 1
            cases = []
            for nm in names:
                d = ColumnData([load_member(files[nm])], norm, dev)
                z = encode_members(ae, d)[0]
                for t0 in [int(x) for x in a.t0.split(",")]:
                    if t0 + n + 1 > z.shape[0]:
                        continue
                    zb = z if a.boundary == "truth" else torch.cat([z[:t0 + 1], z[t0:t0 + 1].expand(z.shape[0] - t0 - 1, -1, -1)])
                    traj = rollout(prop, zb, t0, n, halo, S, seed=t0)
                    case = {"member": nm, "t0": t0, "lags": {}}
                    with torch.no_grad():
                        for L in lags:
                            u = d.full(0, t0 + L)[0]
                            sc = score_case(ae, ae.decode(traj[L]), u, d.Z, slice(mar, u.shape[-1] - mar), norm, a.lowpass_km)
                            if ref_scale is not None:        # this model's normalized units -> the reference model's
                                f = np.array(norm.scale, dtype=float) / ref_scale
                                for k in ("rmse", "crps", "spread", "rmse_lowpass"):
                                    sc[k] = (np.array(sc[k]) * f).tolist()
                                for k in ("power", "power_truth"):
                                    sc[k] = (np.array(sc[k]) * (f ** 2)[:, None]).tolist()
                            case["lags"][str(L)] = sc
                    cases.append(case)
                print(f"[ts_eval] {rn} {nm} ({time.time() - t_start:.0f}s)", flush=True)
            res["runs"][rn] = {"samples": S, "cases": cases}
    if a.paired:
        sys.path.insert(0, str(Path(__file__).parent))
        from paired_compare import frame_at
        man = json.loads(Path(a.paired).expanduser().read_text())
        T0 = man["t0_s"]; ch = {}
        for rn, (ae, prop, norm, cfg, split) in models.items():
            halo = cfg["stage_b"]["halo_sites"]; mar = (halo + ae.cfg.reach_sites) * ae.cfg.patch
            S = a.samples if is_stochastic(prop) else 1
            per = {}
            for short, info in man["members"].items():
                base = load_member(next(Path(a.base).expanduser().glob(f"*__{short}.npz")))
                br = {nm: load_member(Path(a.branches).expanduser() / f"ens_Q__{b['dir']}.npz") for nm, b in info["branches"].items()}
                ctrl = br["ctrl"]; dt = base.dt
                dc = ColumnData([ctrl], norm, dev); db = ColumnData([base], norm, dev)
                u_c = torch.stack([db.full(0, frame_at(base, T0))[0]] + [dc.full(0, frame_at(ctrl, T0 + k * dt))[0] for k in range(1, n + 1)])
                rec = {}
                for amp in ("tiny1e-3", "tiny1e-4"):
                    if amp in br:
                        dm = ColumnData([br[amp]], norm, dev)
                        u_t = torch.stack([u_c[0]] + [dm.full(0, frame_at(br[amp], T0 + k * dt))[0] for k in range(1, n + 1)])
                        rec[f"cm1_{amp}"] = ((u_t - u_c)[..., mar:-mar] ** 2).mean((1, 2)).sqrt().tolist()
                with torch.no_grad():
                    z = torch.cat([ae.encode(u_c[i:i + 8]) for i in range(0, n + 1, 8)])
                    if S > 1:
                        traj = rollout(prop, z, 0, n, halo, S, seed=1)
                        Dm = []
                        for k in range(n + 1):
                            dk = ae.decode(traj[k])[..., mar:-mar]
                            Dm.append(float(torch.stack([(dk[i] - dk[j]).pow(2).mean() for i in range(S) for j in range(i + 1, S)]).mean().sqrt() / np.sqrt(2)))
                        rec["model_divergence"] = Dm
                # common-noise growth: ctrl and tiny-perturbation states, identical noise draws, from lags k0
                with torch.no_grad():
                    cn = {}
                    for amp in ("tiny1e-3",):
                        if amp not in br:
                            continue
                        dm = ColumnData([br[amp]], norm, dev)
                        u_t = torch.stack([u_c[0]] + [dm.full(0, frame_at(br[amp], T0 + k * dt))[0] for k in range(1, n + 1)])
                        zt = torch.cat([ae.encode(u_t[i:i + 8]) for i in range(0, n + 1, 8)])
                        D_cm1 = ((u_t - u_c)[..., mar:-mar] ** 2).mean((1, 2)).sqrt().numpy()
                        for k0 in (10, 20, 30):
                            a_ = rollout(prop, z, k0, 30, halo, 1, seed=100 + k0); b_ = rollout(prop, zt, k0, 30, halo, 1, seed=100 + k0)
                            Dj = [float((ae.decode(a_[j]) - ae.decode(b_[j]))[..., mar:-mar].pow(2).mean().sqrt()) for j in range(31)]
                            lg = lambda D: float((np.log(D[30] + 1e-12) - np.log(D[15] + 1e-12)) / 15)
                            cn[f"k0={k0}"] = {"D_model": Dj, "slope_model_15_30": lg(Dj), "slope_cm1_15_30": lg(D_cm1[k0:k0 + 31]),
                                              "D_model_over_cm1_at_30": Dj[30] / float(D_cm1[k0 + 30] + 1e-12)}
                    rec["common_noise"] = cn
                per[short] = rec
                print(f"[ts_eval] paired {rn} {short} ({time.time() - t_start:.0f}s)", flush=True)
            ch[rn] = per
        res["paired"] = ch
    json.dump(res, open(out / "ts_eval.json", "w")); print("[ts_eval] done")


if __name__ == "__main__":
    main()
