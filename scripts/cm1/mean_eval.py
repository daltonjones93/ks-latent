"""10b: how good is the AVERAGE of the independently trained models? (user, 2026-11-23: "how well would averaging the output from the
independently trained models work? ... compute the statistics", no thresholding).

For each test case (storm x start time) and lag, forecasts in model 0's normalized units (condensate as sqrt(q), the training units):
  member     one member of model 0
  mean1_8    the mean of 8 members of model 0
  mean1_32   the mean of 32 members of model 0
  pool_mean  the mean of the pool: each of the --models contributes --members members (4 x 8 = 32)
  det        (optional --det RUN) a deterministic model's single forecast, for reference
Scores per variable, summed over cases: squared error (RMSE), bias, pattern correlation with the truth (centred over the scored domain),
power of the forecast field / power of the truth in horizontal wavelength bands (how much small-scale structure survives the averaging),
and RMSE in physical units (m/s, K, g/kg, hPa; condensate as q).

  python scripts/cm1/mean_eval.py --models R0 R1 R2 R3 --members 8 --det F7_RUN --split test --out FILE.json
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
from ensemble_design import roll, to_phys  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

DX_KM = 0.25
BANDS = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]
KINDS = ["member", "mean1_8", "mean1_32", "pool_mean", "det"]


def band_power(x):
    """x (V, Z, W) -> (V, n_bands): power per wavelength band, levels averaged"""
    W = x.shape[-1]
    f = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs() ** 2
    f = f.mean(-2)
    k = np.arange(f.shape[-1]); wl = np.where(k > 0, W * DX_KM / np.maximum(k, 1), np.inf)
    return torch.stack([f[..., torch.as_tensor((wl >= lo) & (wl < hi))].sum(-1) for lo, hi in BANDS], -1)


def to_norm(x, sc0):
    """physical (..., V, Z, W) -> model-0 normalized units with condensate as sqrt(q)"""
    y = x.clone()
    for v in ("qliq", "qice"):
        i = VARS.index(v); y[..., i, :, :] = y[..., i, :, :].clamp(min=0).sqrt()
    return y / sc0[:, None, None]


class Acc:
    def __init__(self):
        V, B = len(VARS), len(BANDS)
        self.se = np.zeros(V); self.bias = np.zeros(V); self.cor = np.zeros(V); self.pw = np.zeros((V, B)); self.pt = np.zeros((V, B))
        self.se_phys = np.zeros(V); self.n = 0

    def add(self, fn, tn, fp, tp):
        self.se += ((fn - tn) ** 2).double().mean((1, 2)).numpy()
        self.bias += (fn - tn).double().mean((1, 2)).numpy()
        a = fn - fn.mean((1, 2), keepdim=True); b = tn - tn.mean((1, 2), keepdim=True)
        self.cor += ((a * b).sum((1, 2)) / ((a ** 2).sum((1, 2)).sqrt() * (b ** 2).sum((1, 2)).sqrt() + 1e-12)).double().numpy()
        self.pw += band_power(fn).double().numpy(); self.pt += band_power(tn).double().numpy()
        self.se_phys += ((fp - tp) ** 2).double().mean((1, 2)).numpy()
        self.n += 1

    def to_json(self):
        n = max(self.n, 1)
        return {"rmse": np.sqrt(self.se / n).tolist(), "bias": (self.bias / n).tolist(), "corr": (self.cor / n).tolist(),
                "power_ratio": (self.pw / np.maximum(self.pt, 1e-30)).tolist(), "rmse_phys": np.sqrt(self.se_phys / n).tolist(), "cases": self.n}


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--det", default=None); ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n-storms", type=int, default=99); ap.add_argument("--t0", default="120,180,240")
    ap.add_argument("--lags", default="5,15,30,60,90")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False); t_start = time.time()
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    models = [load_run(Path(m).expanduser(), torch.device("cpu")) for m in a.models]
    det = load_run(Path(a.det).expanduser(), torch.device("cpu")) if a.det else None
    names = models[0][4][a.split][: a.n_storms]
    N, M = len(models), a.members
    files = member_files(a.source)
    sc0 = torch.as_tensor(np.array(models[0][2].scale, float), dtype=torch.float32)
    allm = models + ([det] if det else [])
    mar = max((m[3]["stage_b"]["halo_sites"] + m[0].cfg.reach_sites) * m[0].cfg.patch for m in allm)
    acc = {(k, L): Acc() for k in KINDS for L in lags}
    for nm in names:
        per = []
        for ae, prop, norm, cfg, _ in allm:
            d = ColumnData([load_member(files[nm])], norm, torch.device("cpu"))
            per.append((ae, prop, norm, cfg["stage_b"]["halo_sites"], encode_members(ae, d)[0], d))
        Z = per[0][5].Z
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > per[0][4].shape[0]:
                continue
            dec = lambda j, tr, L: to_phys(torch.cat([per[j][0].decode(tr[L][i:i + 16]) for i in range(0, tr.shape[1], 16)]), per[j][2], Z)[..., mar:-mar]
            pool = {L: [] for L in lags}
            for j in range(N):
                ae, prop, norm, halo, z, _ = per[j]
                tr = roll(prop, z, t0, n, halo, M, a.seed + 1000 * j + t0, 1.0, 0.0)
                for L in lags:
                    pool[L].append(dec(j, tr, L))
            ae, prop, norm, halo, z, _ = per[0]
            tr1 = roll(prop, z, t0, n, halo, 4 * M, a.seed + 77 + t0, 1.0, 0.0)
            if det:
                ae, prop, norm, halo, z, _ = per[N]
                trd = roll(prop, z, t0, n, halo, 1, a.seed + t0, 1.0, 0.0)
            for L in lags:
                tp = to_phys(per[0][5].full(0, t0 + L)[0], per[0][2], Z)[..., mar:-mar]
                s1 = dec(0, tr1, L)
                fc = {"member": s1[0], "mean1_8": s1[:M].mean(0), "mean1_32": s1.mean(0), "pool_mean": torch.cat(pool[L]).mean(0)}
                if det:
                    fc["det"] = dec(N, trd, L)[0]
                tn = to_norm(tp, sc0)
                for k, fp in fc.items():
                    acc[(k, L)].add(to_norm(fp, sc0), tn, fp, tp)
        print(f"[mean_eval] {nm} ({time.time() - t_start:.0f}s)", flush=True)
    out = {"config": vars(a), "vars": list(VARS), "bands_km": BANDS, "kinds": KINDS, "lags": lags, "storms": names,
           "scores": {f"{k}|{L}": A.to_json() for (k, L), A in acc.items() if A.n}, "wall_s": time.time() - t_start}
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).expanduser().write_text(json.dumps(out))
    print(f"[mean_eval] wrote {a.out} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
