"""Registry of the 2D ground-truth runs used by the paper (Table II, Fig. 6, App. C).

Every value is read from its JSON on disk; nothing is hard-coded. Each entry lists the run
files for one (system, size); `runs()` returns the ones that exist, so runs that land later
(tasks A3, A4) enter the figures and tables automatically.

Reference convention (paper §4.4, decided 2026-10-08): the mean over all runs at a size;
at Kolmogorov alpha=0.05, 1x, models are scored per IC against the matched-IC run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments"

REGISTRY: dict[tuple[str, int | float], list[str]] = {
    ("ks2d", 1.0): ["g1_2dks_transfer_native44_n150.json",
                    "truth_ks2d/truth_ks2d_x1_seed1.json", "truth_ks2d/truth_ks2d_x1_seed2.json"],
    ("ks2d", 1.5): ["g1_2dks_transfer_full.json", "truth_ks2d/truth_ks2d_x1p5_seed1.json"],
    ("kolm0.1", 1): ["true_dky_kolmogorov_alpha0.1_trained_size.json"],
    ("kolm0.1", 2): ["true_dky_kolmogorov_alpha0.1_2x_transfer.json",
                     "crop/truth_a0p1_x2_seed1_T120.json", "crop/truth_a0p1_x2_seed2_T120.json"],
    ("kolm0.1", 3): ["true_dky_kolmogorov_alpha0.1_3x_transfer.json",
                     "crop/truth_a0p1_x3_seed1_T120.json", "crop/truth_a0p1_x3_seed2_T120.json"],
    ("kolm0.05", 1): ["crop/truth_a0p05_x1_seed0_T120.json", "crop/truth_a0p05_x1_seed1_T120.json",
                      "crop/truth_a0p05_x1_seed2_T120.json"],
    ("kolm0.05", 2): ["crop/truth_a0p05_x2_seed0_T120.json", "crop/truth_a0p05_x2_seed1_T120.json",
                      "crop/truth_a0p05_x2_seed2_T120.json"],
    ("kolm0.05", 3): ["crop/truth_a005_x3_seed0.json"],
    ("kolm0.025", 1): ["crop/truth_a0p025_x1_seed0.json"],
    ("kolm0.025", 2): ["crop/truth_a0p025_x2_seed0.json"],
    ("kolm0.025", 3): ["crop/truth_a0p025_x3_seed0.json"],
}


def _read(path: Path) -> dict:
    d = json.loads(path.read_text())
    dky = d.get("D_KY", d.get("true_D_KY"))
    return {
        "file": str(path.relative_to(ROOT)),
        "D_KY": float(dky),
        "lambda1": float(d.get("lambda1", d.get("true_lambda1"))),
        "n_positive": int(d.get("n_positive", d.get("true_n_positive"))),
        "n_directions": int(d["n_directions"]),
        "total_time": float(d.get("total_time", 40.0)),  # 40-unit runs predate the field
        "seed": int(d.get("seed", 0)),
    }


def runs(system: str, size) -> list[dict]:
    return [_read(EXP / f) for f in REGISTRY[(system, size)] if (EXP / f).exists()]


def values(system: str, size) -> list[float]:
    return [r["D_KY"] for r in runs(system, size)]
