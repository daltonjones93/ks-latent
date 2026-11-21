"""10b: design of the multi-model ensemble and its vote (user, 2026-11-21: "averaging independently trained models is probably the best idea,
and we should determine how many to use, how much noise to inject in them and how the voting scheme should work to maximize accuracy").

For one setting of the noise (forecast-time noise temperature `--temp` of every model's noise channels, and a Gaussian perturbation of the
initial latent state `--ic-noise`), on the validation or test storms:
  POOL    the first n of the --models (independently trained), M members each (n = 1 .. len(models))
  SINGLE  the first model alone with n x M members (the control: is a pool better than more members of one model?)
and for each, at each lag:
  fields  fair CRPS, RMSE of the ensemble mean and spread, per variable, in the first model's normalized units
  votes   for every feature of cell_losses.FEATURES (physical thresholds; condensate in g/kg):
          - K of S members (S = pool size): counts n[v], hits[v] for every vote count v, from which CSI / bias of "present if v >= K"
          - neighbourhood vote: the member fraction smoothed along x with a box of radius r km (Schwartz and Sobash 2017's neighbourhood
            ensemble probability), counts in 20 bins of the smoothed fraction, for thresholds in steps of 0.05
          - probability matching (Ebert 2001): the points with the largest (smoothed) fraction, as many as the members' mean area
Counts are summed over cases (aggregate CSI = sum hits / sum (hits + misses + false alarms)). Choosing the setting, the pool size and the
vote rule is done on the validation storms (ensemble_design_report.py); the test storms are scored with the same code for every setting so
the chosen one can be reported without a second round.

  python scripts/cm1/ensemble_design.py --models RUN0 RUN1 ... --members 8 --temp 1.0 --ic-noise 0.0 --split val --out FILE.json
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
from ks_latent.cm1.cell_losses import FEATURES  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _with_true_halo  # noqa: E402
from ks_latent.cm1.train import crps_elements, encode_members  # noqa: E402

DX_KM = 0.25
NBINS = 20


@torch.no_grad()
def roll(prop, z, t0, n, halo, S, seed, temp, ic):
    """(n + 1, S, P, c) from t0 with the true halo; noise temperature `temp`; initial latent + N(0, ic^2)."""
    prop.noise_temp = float(temp)
    g = torch.Generator().manual_seed(seed)
    s = z[t0][None].expand(S, -1, -1).clone()
    if ic > 0:
        s = s + ic * torch.randn(s.shape, generator=g)
    out = [s]
    for j in range(n):
        seen = _with_true_halo(s, z[t0 + j][None].expand(S, -1, -1), halo)
        s = _with_true_halo(prop(seen, gen=g), z[t0 + j + 1][None].expand(S, -1, -1), halo)
        out.append(s)
    prop.noise_temp = 1.0
    return torch.stack(out)


def to_phys(u, norm, Z):
    """(..., C, W) normalized -> (..., V, Z, W) physical, condensate as q (g/kg) = clip(sqrt-stored, 0)^2."""
    sc = torch.as_tensor(np.array(norm.scale, float), dtype=u.dtype)
    x = u.reshape(*u.shape[:-2], len(VARS), Z, u.shape[-1]) * sc[:, None, None]
    for v in ("qliq", "qice"):
        i = VARS.index(v)
        x[..., i, :, :] = x[..., i, :, :].clamp(min=0) ** 2
    return x


def box_x(f, r):
    """mean over a centred window of 2r + 1 columns along the last axis (edges: shrinking window)"""
    if r <= 0:
        return f
    c = torch.nn.functional.pad(torch.cumsum(f, -1), (1, 0))
    W = f.shape[-1]
    lo = torch.clamp(torch.arange(W) - r, min=0); hi = torch.clamp(torch.arange(W) + r + 1, max=W)
    return (c[..., hi] - c[..., lo]) / (hi - lo).to(f.dtype)


class Acc:
    """summed counts for one (pool kind, pool size, lag, feature)"""

    def __init__(self, S, radii):
        self.k_n = np.zeros(S + 1); self.k_h = np.zeros(S + 1)
        self.b_n = {r: np.zeros(NBINS) for r in radii}; self.b_h = {r: np.zeros(NBINS) for r in radii}
        self.pm = {r: np.zeros(3) for r in radii}               # hits, misses, false alarms

    def add(self, masks, y, radii_cols):
        S = masks.shape[0]
        v = masks.sum(0)
        yv = y.flatten().double()
        self.k_n += torch.bincount(v.flatten(), minlength=S + 1).numpy()
        self.k_h += torch.bincount(v.flatten(), weights=yv, minlength=S + 1).numpy()
        area = float(masks.flatten(1).sum(1).double().mean())
        for r, rc in radii_cols.items():
            f = box_x(v.double() / S, rc)
            b = torch.clamp((f * NBINS).long(), max=NBINS - 1).flatten()
            self.b_n[r] += torch.bincount(b, minlength=NBINS).numpy()
            self.b_h[r] += torch.bincount(b, weights=yv, minlength=NBINS).numpy()
            A = int(round(area))
            pred = torch.zeros(f.numel(), dtype=torch.bool)
            if A > 0:
                pred[torch.argsort(f.flatten(), descending=True)[:A]] = True
            yb = yv > 0
            self.pm[r] += [float((pred & yb).sum()), float((~pred & yb).sum()), float((pred & ~yb).sum())]

    def to_json(self):
        return {"k_n": self.k_n.tolist(), "k_h": self.k_h.tolist(),
                "bin_n": {str(r): x.tolist() for r, x in self.b_n.items()}, "bin_h": {str(r): x.tolist() for r, x in self.b_h.items()},
                "pm": {str(r): x.tolist() for r, x in self.pm.items()}}


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--ic-noise", type=float, default=0.0)
    ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--n-storms", type=int, default=99)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--lags", default="15,30,60")
    ap.add_argument("--radii-km", default="0,1,2,4"); ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False); t_start = time.time()
    lags = [int(x) for x in a.lags.split(",")]; n = max(lags)
    radii = [float(x) for x in a.radii_km.split(",")]; radii_cols = {r: int(round(r / DX_KM)) for r in radii}
    models = [load_run(Path(m).expanduser(), torch.device("cpu")) for m in a.models]
    names = models[0][4][a.split][: a.n_storms]
    for m in models[1:]:
        if m[4][a.split][: a.n_storms] != names:
            raise ValueError("models with different splits cannot be pooled")
    N, M = len(models), a.members
    files = member_files(a.source)
    norm0 = models[0][2]; sc0 = torch.as_tensor(np.array(norm0.scale, float))
    mar = max((m[3]["stage_b"]["halo_sites"] + m[0].cfg.reach_sites) * m[0].cfg.patch for m in models)
    kinds = ["pool"] + ([] if a.no_single else ["single"])
    acc = {(kd, nn, L, f): Acc(nn * M, radii) for kd in kinds for nn in range(1, N + 1) for L in lags for f in FEATURES}
    fld = {(kd, nn, L): {"crps": np.zeros(len(VARS)), "rmse2": np.zeros(len(VARS)), "spread2": np.zeros(len(VARS)), "cases": 0}
           for kd in kinds for nn in range(1, N + 1) for L in lags}
    n_cases = 0
    for nm in names:
        per = []
        for ae, prop, norm, cfg, _ in models:
            d = ColumnData([load_member(files[nm])], norm, torch.device("cpu"))
            per.append((ae, prop, norm, cfg["stage_b"]["halo_sites"], encode_members(ae, d)[0], d))
        Z = per[0][5].Z
        for t0 in [int(x) for x in a.t0.split(",")]:
            if t0 + n + 1 > per[0][4].shape[0]:
                continue
            members = {L: [] for L in lags}                      # physical (S_j, V, Z, W) per model, pooled order
            for j, (ae, prop, norm, halo, z, _) in enumerate(per):
                tr = roll(prop, z, t0, n, halo, M, a.seed + 1000 * j + t0, a.temp, a.ic_noise)
                for L in lags:
                    members[L].append(to_phys(ae.decode(tr[L]), norm, Z)[..., mar:-mar])
            single = {}
            if not a.no_single:
                ae, prop, norm, halo, z, _ = per[0]
                tr = roll(prop, z, t0, n, halo, N * M, a.seed + 77 + t0, a.temp, a.ic_noise)
                for L in lags:
                    single[L] = to_phys(torch.cat([ae.decode(tr[L][i:i + 16]) for i in range(0, N * M, 16)]), norm, Z)[..., mar:-mar]
            d0 = per[0][5]
            for L in lags:
                truth = to_phys(d0.full(0, t0 + L)[0], norm0, Z)[..., mar:-mar]
                pool_all = torch.cat(members[L])
                for kd in kinds:
                    for nn in range(1, N + 1):
                        ens = pool_all[: nn * M] if kd == "pool" else single[L][: nn * M]
                        # fields in model-0 normalized units (condensate back to sqrt for comparability with training units)
                        en, tn = ens.clone(), truth.clone()
                        for v in ("qliq", "qice"):
                            i = VARS.index(v); en[:, i] = en[:, i].sqrt(); tn[i] = tn[i].sqrt()
                        en = en / sc0[:, None, None]; tn = tn / sc0[:, None, None]
                        F_ = fld[(kd, nn, L)]
                        F_["crps"] += crps_elements(en.double(), tn.double()).mean((1, 2)).numpy()
                        F_["rmse2"] += ((en.mean(0) - tn) ** 2).double().mean((1, 2)).numpy()
                        F_["spread2"] += en.var(0, unbiased=True).double().mean((1, 2)).numpy()
                        F_["cases"] += 1
                        for f, (v, sign, thr) in FEATURES.items():
                            i = VARS.index(v)
                            acc[(kd, nn, L, f)].add(sign * ens[:, i] > thr, sign * truth[i] > thr, radii_cols)
            n_cases += 1
        print(f"[design] {nm} done ({time.time() - t_start:.0f}s, {n_cases} cases)", flush=True)
    out = {"config": vars(a), "vars": list(VARS), "features": list(FEATURES), "members_per_model": M, "n_models": N,
           "n_cases": n_cases, "storms": names, "radii_km": radii, "nbins": NBINS,
           "fields": {f"{kd}|{nn}|{L}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in F_.items()} for (kd, nn, L), F_ in fld.items()},
           "votes": {f"{kd}|{nn}|{L}|{f}": A.to_json() for (kd, nn, L, f), A in acc.items()}, "wall_s": time.time() - t_start}
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).expanduser().write_text(json.dumps(out))
    print(f"[design] wrote {a.out} ({n_cases} cases, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
