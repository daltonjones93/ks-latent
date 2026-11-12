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
    ap.add_argument("--paired", default=None, help="manifest.json of the paired runs (T-chaos)")
    ap.add_argument("--base", default=None); ap.add_argument("--branches", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); t_start = time.time(); dev = torch.device(a.device)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    models = {}
    for r in a.runs:
        r = Path(r).expanduser(); models[r.name] = load_run(r, dev)
    res = {"vars": list(VARS), "bands_km": BANDS, "lags": lags, "runs": {}}
    if a.source:
        files = member_files(a.source)
        names = next(iter(models.values()))[4]["test"][: a.n_test]
        for rn, (ae, prop, norm, cfg, split) in models.items():
            if split["test"][: a.n_test] != names:
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
                    traj = rollout(prop, z, t0, n, halo, S, seed=t0)
                    case = {"member": nm, "t0": t0, "lags": {}}
                    with torch.no_grad():
                        for L in lags:
                            u = d.full(0, t0 + L)[0]
                            case["lags"][str(L)] = score_case(ae, ae.decode(traj[L]), u, d.Z, slice(mar, u.shape[-1] - mar), norm, a.lowpass_km)
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
