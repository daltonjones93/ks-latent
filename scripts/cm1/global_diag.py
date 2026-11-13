"""7a diagnostic: how far does a trained propagator's one-step dependence reach?
For test states z_n (encoded CM1, every --every min from --t0), the Jacobian of one step J[i, a, j, b] = d z_{n+1}[i, a] / d z_n[j, b]
(noise fixed, interior sites only as outputs), reduced to the Frobenius norm over channels per site pair, averaged over pairs
with the same offset j - i, normalized by the offset-0 value. A purely local stencil is exactly zero beyond its reach (3 sites);
the share of the total beyond the reach is the globally coupled model's use of distant information. Also, for "attn" runs, the
attention weight per layer and head averaged by |offset| (km).
  python scripts/cm1/global_diag.py --runs R1 R2 --source cache:... --out DIR"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=4); ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    files = member_files(a.source); res = {}
    for r in a.runs:
        r = Path(r).expanduser(); ae, prop, norm, cfg, split = load_run(r, torch.device("cpu"))
        reach = prop.cfg.reach_sites; site_km = ae.cfg.patch * 0.25
        prof, attn = [], {}
        blocks = list(getattr(getattr(prop, "glob", None), "blocks", []))
        for nm in split["test"][: a.n_test]:
            d = ColumnData([load_member(files[nm])], norm, torch.device("cpu"))
            z = encode_members(ae, d)[0]
            for t0 in map(int, a.t0.split(",")):
                zn = z[t0][None]; P, c = zn.shape[1:]
                eps = torch.zeros(1, prop.cfg.noise_channels, P) if prop.cfg.noise_channels else None
                f = lambda x: prop(x[None], eps=eps)[0]
                with torch.no_grad():
                    for b in blocks:
                        b.record = True
                    f(zn[0])
                    for li, b in enumerate(blocks):
                        A = b.last_attn[0]                                     # (heads, P, P)
                        i = torch.arange(P); off = (i[None, :] - i[:, None]).abs()
                        attn.setdefault(li, []).append([[float(A[h][off == o].sum() / P) for o in range(P)] for h in range(A.shape[0])])   # mass per |offset|
                        b.record = False
                J = jacrev(f)(zn[0]).detach()                                  # (P, c, P, c)
                Jn = J.pow(2).sum((1, 3)).sqrt()                               # (P, P)
                lo, hi = 8, P - 8                                              # outputs away from the halo / edges
                offs = np.arange(-(P - 1), P)
                pr = []
                for o in offs:
                    a0, a1 = max(lo, -o), min(hi, P - o)
                    if a1 <= a0:
                        pr.append(np.nan); continue
                    ii = torch.arange(a0, a1)
                    pr.append(float(Jn[ii, ii + o].mean()))
                prof.append(pr)
            print(f"[gdiag] {r.name} {nm}", flush=True)
        prof = np.nanmean(np.array(prof), 0); offs = np.arange(-(len(prof) // 2), len(prof) // 2 + 1)
        tot = np.nansum(prof); beyond = np.nansum(prof[np.abs(offs) > reach])
        res[r.name] = {"reach_sites": reach, "site_km": site_km, "offsets": offs.tolist(), "jac_profile": prof.tolist(),
                       "share_beyond_reach": float(beyond / tot),
                       "attn_by_dist": {str(k): np.mean(np.array(v), 0).tolist() for k, v in attn.items()}}
        print(f"[gdiag] {r.name}: share of one-step dependence beyond {reach} sites ({reach * site_km:g} km): {beyond / tot:.4f}")
        for k, v in res[r.name]["attn_by_dist"].items():
            v = np.array(v)                                                    # (heads, P)
            for h in range(v.shape[0]):
                w = v[h] / v[h].sum(); dmean = float((w * np.arange(len(w))).sum() * site_km)
                print(f"   layer {k} head {h}: mean attended distance {dmean:.1f} km, weight within 3 sites {w[:4].sum():.2f}")
    json.dump(res, open(out / "global_diag.json", "w")); print("[gdiag] done")


if __name__ == "__main__":
    main()
