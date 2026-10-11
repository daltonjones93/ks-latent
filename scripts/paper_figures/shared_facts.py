"""S5.1 (docs/split_paper_tasks.md): the facts both papers quote, recomputed from the result files,
and where each paper's text states them.

For every fact the script prints its value and, per paper (docs/paperI, docs/paperII: sections,
appendices, main.tex, generated tables), whether the formatted value appears. A fact that one paper
quotes and the other does not is fine; the check fails (exit 1) only if a paper quotes a
*different* value for a fact through one of the listed alternative spellings (`wrong`), so that a
stale or recomputed number cannot survive in one paper only.
Usage: python scripts/paper_figures/shared_facts.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from _truths import runs

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts/experiments"


def text(paper):
    fs = glob.glob(str(ROOT / f"docs/{paper}/sections/*.tex")) + glob.glob(str(ROOT / f"docs/{paper}/appendix/*.tex"))
    fs += [str(ROOT / f"docs/{paper}/main.tex")] + glob.glob(str(ROOT / f"docs/{paper}/tables/*.tex"))
    return "\n".join(Path(f).read_text() for f in fs)


def crop(tag):
    return json.loads((EXP / "crop" / f"{tag}.json").read_text())


facts = []  # (name, value string(s) that must be the ones used, wrong spellings that must not appear)
t44, t66 = runs("ks2d", 1.0), runs("ks2d", 1.5)
d44, d66 = [r["D_KY"] for r in t44], [r["D_KY"] for r in t66]
facts.append(("2D KS truth D_KY runs at 44^2", [f"${min(d44):.1f}$", f"${max(d44):.1f}$"], []))
facts.append(("2D KS truth D_KY mean at 44^2", [f"${np.mean(d44):.1f}$"], []))
facts.append(("2D KS truth D_KY mean at 66^2", [f"${np.mean(d66):.1f}$"], ["$189.1$"]))  # 189.05 rounds to 189.0
n44, n66 = [r["n_positive"] for r in t44], [r["n_positive"] for r in t66]
facts.append(("2D KS truth n+ at 44^2", [f"{min(n44)}--{max(n44)}"], []))
facts.append(("2D KS truth n+ at 66^2", [f"{min(n66)}--{max(n66)}"], []))
facts.append(("2D KS density 44^2, 66^2", [f"${np.mean(d44)/44**2:.4f}$", f"${np.mean(d66)/66**2:.4f}$"], []))
for tag, lab in (("p7c12d10", "noisy 12-channel"), ("p7c12", "clean 12-channel"), ("p7b", "6-channel")):
    e = [abs(crop(f"{tag}_ks2d_x{s}_seed0")["rel_err_pct"]) for s in ("1.0", "1.5")]
    facts.append((f"{lab} model D_KY error vs seed-0 truth (44^2, 66^2)", [f"${e[0]:.1f}\\%$", f"${e[1]:.1f}\\%$"], []))
for tag, lab in (("p7c12d10", "noisy 12-channel"),):
    v = [crop(f"{tag}_ks2d_x{s}_seed0")["model_D_KY"] for s in ("1.0", "1.5")]
    facts.append((f"{lab} model D_KY (44^2, 66^2)", [f"${v[0]:.1f}$", f"${v[1]:.1f}$"], []))
with __import__("h5py").File(ROOT / "artifacts/datasets/ks2d_wide6.h5", "r") as f:
    std = float(f["metadata"].attrs.get("normalization_std", f["metadata"].attrs.get("std", np.nan)))
facts.append(("2D KS observation error 0.1 std in field units", [f"${0.1 * std:.2f}$"], []))
facts.append(("latent dimension at 440^2, 12 channels", ["$76\\,800$"], ["$76{,}800$"]))
s1d = json.loads((EXP / "e1_model/summary_1d_transfer.json").read_text())
err1d = [abs(100 * (s1d["model"][str(L)]["mean"] - s1d["truth"][str(L)]["mean"]) / s1d["truth"][str(L)]["mean"])
         for L in (50, 100, 200, 400)]
facts.append(("1D model D_KY error range, L = 50-400", [f"${min(err1d):.1f}$--${max(err1d):.1f}\\%$"], []))
facts.append(("1D truth n+ at L = 200, 400", ["$26$--$27$", "$51$"], []))

T = {p: text(p).replace("$", "") for p in ("paperI", "paperII")}  # compare without math delimiters
bad = False
print(f"{'fact':58s} {'value':26s} {'paperI':8s} {'paperII':8s}")
for name, vals, wrong in facts:
    use = {p: all(v.replace("$", "") in T[p] for v in vals) for p in T}
    print(f"{name:58s} {' / '.join(vals):26s} {'yes' if use['paperI'] else '-':8s} {'yes' if use['paperII'] else '-':8s}")
    for p in T:
        for w in wrong:
            if w.replace("$", "") in T[p]:
                print(f"   !! {p} uses {w!r} for this fact")
                bad = True
sys.exit(1 if bad else 0)
