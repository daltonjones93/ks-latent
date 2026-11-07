"""G0 claim 2: rank histograms of the theta' observations among the prior members, and the chi^2 test.

Reads a da_study output (rows.json, needs the per-cycle ranks recorded since 2026-11-07) and, for one mode:
  pre-registered   histogram pooled over truths, seeds and second-half cycles; chi^2 against flat, p > 0.01
                   and "not U-shaped". Reported for the members as they are ("plain") and with observation
                   error added to the members ("perturbed", Hamill 2001: the correct histogram for noisy
                   observations; chosen 2026-11-07, before any histogram was looked at).
  decorrelated     (exploratory, not pre-registered) the same test on a subsample whose ranks are closer to
                   independent: observations every --obs-step (default 10 = 20 km), every --cycle-step-th
                   cycle (default 3 = 15 min), seed 0 only. Pooled ranks are strongly correlated in space and
                   time, so the pooled test's p-value is far too small for any real deviation.
  shape            outer-bin ratio (mass in the two outermost bins / its flat value; > 1: U-shaped, too
                   little spread; < 1: dome-shaped, too much) and the mean rank bias (centre 0).
  python scripts/cm1/rank_test.py ~/cm1_da/g3_headline --mode latent
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def test(h: np.ndarray) -> dict:
    h = np.asarray(h, float)
    n, B = h.sum(), h.size
    e = n / B
    x2 = float(((h - e) ** 2 / e).sum())
    p = float(chi2.sf(x2, B - 1))
    outer = float((h[0] + h[-1]) / (2 * e))
    bias = float((h * (np.arange(B) - (B - 1) / 2)).sum() / n / ((B - 1) / 2))
    return {"n": int(n), "chi2": x2, "dof": B - 1, "p": p, "flat_at_p_0.01": p > 0.01, "outer_bin_ratio": outer,
            "mean_rank_bias": bias, "u_shaped": outer > 1.0, "hist": h.astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--mode", default="latent")
    ap.add_argument("--obs-step", type=int, default=10)
    ap.add_argument("--cycle-step", type=int, default=3)
    a = ap.parse_args()
    d = Path(a.study).expanduser()
    rows = json.loads((d / "rows.json").read_text())
    res = {"study": str(d), "mode": a.mode}
    for kind, hk, ck in (("plain", "rank_hist", "ranks_cycles"), ("perturbed", "rank_hist_pert", "ranks_pert_cycles")):
        runs = [s["prior"] for t in rows.values() for s in t[a.mode]]
        if not runs or hk not in runs[0]:
            raise KeyError(f"{d}: mode {a.mode} has no {hk}; rerun da_study with the current code")
        pooled = np.sum([r[hk] for r in runs], 0)
        B = len(pooled)
        sub = np.zeros(B)
        for t in rows.values():
            cyc = t[a.mode][0]["prior"][ck]                # seed 0
            for c in cyc[:: a.cycle_step]:
                sub += np.bincount(np.asarray(c)[:: a.obs_step], minlength=B)
        res[kind] = {"pooled": test(pooled), "decorrelated": test(sub)}
    (d / f"rank_test_{a.mode}.json").write_text(json.dumps(res, indent=1))
    for kind in ("plain", "perturbed"):
        for w in ("pooled", "decorrelated"):
            r = res[kind][w]
            print(f"[rank] {a.mode} {kind:9s} {w:12s} n {r['n']:7d}  chi2 {r['chi2']:10.1f} (dof {r['dof']})  p {r['p']:.2g}  "
                  f"outer-bin ratio {r['outer_bin_ratio']:.2f}  bias {r['mean_rank_bias']:+.3f}  "
                  f"{'flat' if r['flat_at_p_0.01'] else 'not flat'}, {'U-shaped' if r['u_shaped'] else 'not U-shaped'}")


if __name__ == "__main__":
    main()
