"""Aggregate task A1/A2/A5 outputs into one artifact for the 1D transfer table and figures.

Model: artifacts/experiments/e1_model/*.json (eval_1d_transfer.py), standardized ICs,
float32 unless stated. Truth: every 1D ground-truth realization on disk
(e1_groundtruth_lyapunov_L*.json incl. *_fullspec, *_repro*, and the Gate-1 value at L=100
from artifacts/replication_results.jsonl). Distinct D_KY values are distinct finite-time
realizations: thread count changes FFT/BLAS round-off, and chaos turns that into a
different trajectory; identical values (same trajectory) are counted once.

Writes artifacts/experiments/e1_model/summary_1d_transfer.json
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

EXP = Path("artifacts/experiments")
OUT = EXP / "e1_model" / "summary_1d_transfer.json"
LS = [25, 50, 100, 200, 400]


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return {"values": v.tolist(), "n": int(v.size), "mean": float(v.mean()),
            "half_range": float((v.max() - v.min()) / 2), "min": float(v.min()), "max": float(v.max())}


def main() -> None:
    model_files = sorted(glob.glob(str(EXP / "e1_model" / "*_L*_ic*_*.json")))
    recs = [json.loads(Path(f).read_text()) | {"file": f} for f in model_files if "summary" not in f]

    out = {"model": {}, "model_seeds": {}, "model_raw_ic": {}, "model_float64": {}, "truth": {},
           "record_values": {"model": {25: 0.0, 50: 11.06, 100: 22.14, 200: 44.61, 400: 89.13},
                             "source": "docs/ks_local_field_ltransfer.tex tab:e1_result (raw-IC encoding)"}}

    for L in LS:
        sel = [r for r in recs if r["model"] == "section224" and r["L"] == L and r["ic"] == "normalized"
               and r["tangent_dtype"] == "float32" and r["D_KY"] is not None]
        if sel:
            s = stats([r["D_KY"] for r in sel])
            s["ic_seeds"] = [r["ic_seed"] for r in sel]
            s["n_positive"] = [r["n_positive"] for r in sel]
            s["lambda1"] = [r["lambda1"] for r in sel]
            s["max_abs_u_model"] = [r["max_abs_u_model"] for r in sel]
            s["max_abs_u_true"] = [r["max_abs_u_true"] for r in sel]
            out["model"][L] = s
        raw = [r for r in recs if r["model"] == "section224" and r["L"] == L and r["ic"] == "raw"]
        if raw:
            out["model_raw_ic"][L] = [r["D_KY"] for r in raw]
        f64 = [r for r in recs if r["model"] == "section224" and r["L"] == L and r["tangent_dtype"] == "float64"]
        if f64:
            out["model_float64"][L] = stats([r["D_KY"] for r in f64]) | {"ic_seeds": [r["ic_seed"] for r in f64]}
        for tag in ("e4_seed1", "e4_seed2"):
            t = [r for r in recs if r["model"] == tag and r["L"] == L and r["ic"] == "normalized"]
            if t:
                out["model_seeds"].setdefault(L, {})[tag] = t[0]["D_KY"]

        truths = {}
        for f in glob.glob(str(EXP / f"e1_groundtruth_lyapunov_L{L}.json")) + glob.glob(
                str(EXP / f"e1_groundtruth_lyapunov_L{L}_*.json")):
            d = json.loads(Path(f).read_text())
            # same trajectory (differs only by tangent count / reproduced run) = one realization
            key = next((k for k in truths if abs(k - d["D_KY"]) < 0.01), d["D_KY"])
            truths.setdefault(key, {"D_KY": d["D_KY"], "lambda1": d["lambda1"], "n_positive": d["n_positive"],
                                    "n_directions": d["n_directions"], "total_time": d["total_time"],
                                    "files": []})["files"].append(f)
        if L == 100:
            for line in Path("artifacts/replication_results.jsonl").read_text().splitlines():
                d = json.loads(line)
                if d["name"] == "L100_D_KY":
                    key = next((k for k in truths if abs(k - d["value"]) < 0.01), d["value"])
                    truths.setdefault(key, {"D_KY": d["value"], "total_time": 2000.0,
                                                             "files": ["artifacts/replication_results.jsonl (Gate 1)"]})
                    break
        if truths:
            out["truth"][L] = stats([t["D_KY"] for t in truths.values()]) | {"realizations": list(truths.values())}

    for L in LS:
        if L in out["model"] and L in out["truth"]:
            m, t = out["model"][L]["mean"], out["truth"][L]["mean"]
            out["model"][L]["rel_err_vs_truth_mean_pct"] = 100 * (m - t) / t if t else None
            out["model"][L]["D_KY_over_L_mean"] = m / L
            out["truth"][L]["D_KY_over_L_mean"] = t / L
    rec_m = out["record_values"]["model"]
    out["a1_decision"] = {
        "rule": "accept the existing Table III if every model D_KY (IC seed 0, standardized) moves < 1% from the record",
        "moves_pct": {L: (100 * (r["D_KY"] - rec_m[L]) / rec_m[L] if rec_m[L] else None)
                      for L in LS for r in recs
                      if r["model"] == "section224" and r["L"] == L and r["ic"] == "normalized"
                      and r["ic_seed"] == 0 and r["tangent_dtype"] == "float32"},
    }
    moves = [abs(v) for v in out["a1_decision"]["moves_pct"].values() if v is not None]
    out["a1_decision"]["verdict"] = ("accept existing table" if moves and max(moves) < 1.0
                                     else "regenerate Table III and Fig. 3 from the new JSONs")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for L in LS:
        m = out["model"].get(L); t = out["truth"].get(L)
        print(L, "model", m and f"{m['mean']:.2f}±{m['half_range']:.2f} ({m['n']})",
              "truth", t and f"{t['mean']:.2f}±{t['half_range']:.2f} ({t['n']})",
              "err%", m and m.get("rel_err_vs_truth_mean_pct") and round(m["rel_err_vs_truth_mean_pct"], 2))
    print(out["a1_decision"])


if __name__ == "__main__":
    main()
