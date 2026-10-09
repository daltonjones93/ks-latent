"""Appendix D (task B12): every 2D model evaluation, scored with the paper's truth convention.

Outputs
  docs/paper1/tables/tabD_ks2d.tex, tabD_kolm01.tex, tabD_kolm005.tex, tabD_kolm0025.tex
      one row per model: mean D_KY over ICs (+- half-range, n) and the mean of per-IC
      relative errors, per domain size;
  artifacts/experiments/appendix_D_evaluations.csv
      one row per (model, size, IC): model D_KY, lambda_1, n_+, the truth used here, the
      error used here, and -- for transparency -- the `true_D_KY` / `rel_err_pct` fields stored
      in the evaluation JSON itself. Those stored fields are STALE (they use the truths of the
      day the evaluation ran: 23.26 / 56.84 at alpha=0.05, 30.34 at alpha=0.1 2x) and are
      never used in the paper.
Truth convention: mean over the runs in `_truths.py`; Kolmogorov alpha=0.05 at 1x is scored
per IC against the matched-IC run. Box-trained baselines come from the same files as Fig. 5.
"""
import csv
import glob
import json
import re

import numpy as np

from _style import EXP, MAIN_A01, ROOT
from _truths import runs

OUT = ROOT / "docs/paper1/tables"
TRUTH = {("ks2d", 1.0): None, ("ks2d", 1.5): None}
TSYS = {"ks2d": "ks2d", 0.1: "kolm0.1", 0.05: "kolm0.05", 0.025: "kolm0.025"}

DESC = {
    "p7_ks2d": "window, full reach, $c=3$, $\\kmax=12$",
    "p7b_ks2d": "canonical (main text)",
    "p1_a01": "window, full reach, $c=3$, $\\kmax=12$",
    "k_xs01": "canonical variant v1", "k_xs01_v2": "canonical variant v2", "k_xs01_v3": "canonical variant v3",
    "k_xs01_v4": "canonical variant v4", "k_xs01_v5": "canonical variant v5", "k_xs01_v7": "canonical variant v6",
    "p2_a005": "window, full reach, $c=3$, $\\kmax=12$",
    "c19_e6d0": "window, encoder reach 6, decoder 0, $c=3$",
    "c19_e6d6_c6": "window, full reach, $c=6$",
    "c19_e0d0_c6": "per-patch, $c=6$, $\\kmax=12$ (main text)",
    "c19_e0d0_c6_s1": "as above, second training seed",
    "c7b_a005": "window + learned consensus",
    "c9i_a005": "window + exact Biot--Savart velocity input",
    "p9_a005": "2D-patch crops, $\\kmax=12$",
    "p9b_a005": "2D-patch crops, $\\kmax=32$, end noise 0.05",
    "k_a005": "2D-patch crops, per-patch $c=6$, $\\kmax=32$",
    "k_xs005": "per-patch $c=6$, $\\kmax=32$, spectral regularizer",
    "k_xs005_noreg": "per-patch $c=6$, $\\kmax=32$",
    "k_xs005_v2": "per-patch $c=6$, $\\kmax=32$, 8 trajectories",
    "p6_i6": "windows of a $3\\times$ domain, interior 6 sites$^b$",
    "p6_i8": "windows of a $3\\times$ domain, interior 8 sites$^b$",
    "p6_i12": "windows of a $3\\times$ domain, interior 12 sites$^b$",
    "p6_full": "whole $3\\times$ periodic domain$^b$",
    "p8_a0025": "window, full reach, $c=3$, $\\kmax=12$",
    "k_xs0025": "canonical (main text)",
}
BOX = {  # box-trained baselines (same files as Fig. 5); (system, scale) -> list of files
    ("ks2d", 1.0): [f"icnorm/ks2d_x1_normalized_seed{s}.json" for s in (0, 1, 2)],
    ("ks2d", 1.5): ["icnorm/ks2d_x1.5_normalized.json", "icnorm/ks2d_x1.5_normalized_seed1.json",
                    "icnorm/ks2d_x1.5_normalized_seed2.json"],
    (0.1, 1): ["c4/s233_a01_x1_m0.json"], (0.1, 2): ["icnorm/s233_x2_normalized.json"],
    (0.1, 3): ["g3_kolmogorov_alpha_transfer_section233_3x.json"],
    (0.05, 1): ["c4/s231_a005_x1_m0.json"], (0.05, 2): ["icnorm/s231_x2_normalized.json"],
}
BOX_IC_SEED = {"c4/s231_a005_x1_m0.json": 0}


def truth_for(sys_key, scale):
    rs = runs(TSYS[sys_key], scale)
    return rs


def score(sys_key, scale, ic, dky):
    rs = truth_for(sys_key, scale)
    if sys_key == 0.05 and scale == 1:
        ref = {r["seed"]: r["D_KY"] for r in rs}[ic]
    else:
        ref = float(np.mean([r["D_KY"] for r in rs]))
    return ref, 100 * (dky - ref) / ref


rows_csv, by_sys = [], {}
for f in sorted(glob.glob(str(EXP / "crop/*_x*_seed*.json"))):
    name = f.split("/")[-1]
    if name.startswith("truth") or "summary" in name:
        continue
    m = re.match(r"(.+)_x([0-9.]+)_seed(\d+)\.json", name)
    tag, scale, ic = m[1], float(m[2]), int(m[3])
    d = json.loads(open(f).read())
    sys_key = "ks2d" if d.get("system") == "ks2d" else float(d["alpha"])
    if sys_key != "ks2d":
        scale = int(round(scale))
    have_truth = not (sys_key == "ks2d" and scale == 2.0)
    ref, err = score(sys_key, scale, ic, d["model_D_KY"]) if have_truth else (None, None)
    rows_csv.append({"model": tag, "system": sys_key, "scale": scale, "ic_seed": ic,
                     "model_D_KY": d["model_D_KY"], "model_lambda1": d.get("model_lambda1"),
                     "model_n_positive": d.get("model_n_positive"), "truth_used": ref, "rel_err_pct_used": err,
                     "stale_true_D_KY_in_json": d.get("true_D_KY"), "stale_rel_err_pct_in_json": d.get("rel_err_pct"),
                     "file": str((EXP / "crop" / name).relative_to(ROOT))})
    by_sys.setdefault(sys_key, {}).setdefault(tag, {}).setdefault(scale, []).append((ic, d["model_D_KY"], err))

for (sys_key, scale), files in BOX.items():
    for k, fn in enumerate(files):
        d = json.loads((EXP / fn).read_text())
        ic = BOX_IC_SEED.get(fn, k)
        ref, err = score(sys_key, scale, ic, d["model_D_KY"])
        rows_csv.append({"model": "box-trained", "system": sys_key, "scale": scale, "ic_seed": ic,
                         "model_D_KY": d["model_D_KY"], "model_lambda1": d.get("model_lambda1"),
                         "model_n_positive": d.get("model_n_positive"), "truth_used": ref, "rel_err_pct_used": err,
                         "stale_true_D_KY_in_json": d.get("true_D_KY"), "stale_rel_err_pct_in_json": d.get("rel_err_pct"),
                         "file": str((EXP / fn).relative_to(ROOT))})
        by_sys.setdefault(sys_key, {}).setdefault("box-trained", {}).setdefault(scale, []).append((ic, d["model_D_KY"], err))

with open(EXP / "appendix_D_evaluations.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
    w.writeheader(); w.writerows(rows_csv)

ORDER = {"ks2d": ["box-trained", "p7_ks2d", "p7b_ks2d"],
         0.1: ["box-trained", "p1_a01", "k_xs01", "k_xs01_v2", "k_xs01_v3", "k_xs01_v4", "k_xs01_v5", "k_xs01_v7"],
         0.05: ["box-trained", "p2_a005", "c19_e6d0", "c19_e6d6_c6", "c19_e0d0_c6", "c19_e0d0_c6_s1", "c7b_a005",
                "c9i_a005", "p9_a005", "p9b_a005", "k_a005", "k_xs005", "k_xs005_noreg", "k_xs005_v2",
                "p6_i6", "p6_i8", "p6_i12", "p6_full"],
         0.025: ["p8_a0025", "k_xs0025"]}
SCALES = {"ks2d": [1.0, 1.5, 2.0], 0.1: [1, 2, 3], 0.05: [1, 2, 3], 0.025: [1, 2, 3]}
FILE = {"ks2d": "tabD_ks2d", 0.1: "tabD_kolm01", 0.05: "tabD_kolm005", 0.025: "tabD_kolm0025"}
for sys_key, tags in ORDER.items():
    lines = ["% Generated by scripts/paper_figures/tabD_evaluations.py -- do not edit by hand."]
    for tag in tags:
        if tag not in by_sys.get(sys_key, {}):
            continue
        cells = []
        for sc in SCALES[sys_key]:
            v = by_sys[sys_key][tag].get(sc)
            if not v:
                cells.append("---"); continue
            vals = [x for _, x, _ in v]; errs = [e for _, _, e in v if e is not None]
            cell = f"{np.mean(vals):.1f}" + (f"$\\pm${(max(vals)-min(vals))/2:.1f}" if len(vals) > 1 else "") + f" ({len(vals)})"
            cell += f", {np.mean(np.abs(errs)):.1f}\\%" if errs else ", ---"
            cells.append(cell)
        if tag == "box-trained":
            DESC[tag] = "same architecture as \\texttt{p2}/\\texttt{p7}, $8\\times8$-site periodic box"
        label = "box-trained$^a$" if tag == "box-trained" else f"\\texttt{{{tag.replace('_', chr(92) + '_')}}}"
        if tag == MAIN_A01:
            label += "$^\\star$"
        lines.append(f"{label} & {DESC.get(tag, '')} & " + " & ".join(cells) + " \\\\")
    (OUT / f"{FILE[sys_key]}.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {len(rows_csv)} rows to appendix_D_evaluations.csv and {len(ORDER)} tables")
