"""Shared style and provenance for paper-1 figures.

Palette: the dataviz reference categorical palette in its documented order
(slots 1-4 pass the adjacent-pair checks; slots 1-3 also pass all-pairs, so
scatter/line figures use at most three hues). Truth is always drawn in
neutral ink, never in a series hue. Identity is never color-alone: every
figure carries a legend or direct labels.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
EXP = ART / "experiments"
OUT = ROOT / "docs/paper1/figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8c8b86", "#e6e5e1"

# Figure widths = text widths of the current layout, so that figures are included at natural size
# (scale 1) and every font prints at its designed size. AGU draft class (PJ2019.cls): text width
# 397.5 pt = 5.5 in. For REVTeX reprint (Chaos submission) use 3.37 / 7.0 instead.
COL1, COL2 = 3.37, 5.5
MIN_FONT = 7.0  # figure contract: no text smaller than 7 pt

# Fixed system hues across the paper (figure contract): 1D KS blue, 2D KS aqua, Kolmogorov
# orange. Kolmogorov drags are told apart by marker and lightness, never by hue alone.
HUE = {"ks1d": BLUE, "ks2d": AQUA, "kolm": ORANGE}
DRAG_MARKER = {0.1: "o", 0.05: "s", 0.025: "^"}


def shade(hex_color: str, t: float):
    """t in [-1, 1]: negative darkens toward ink, positive lightens toward white."""
    c = ([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)] if isinstance(hex_color, str)
         else list(hex_color)[:3])
    return tuple((1 - t) * x + t for x in c) if t >= 0 else tuple((1 + t) * x for x in c)


DRAG_COLOR = {0.1: shade(ORANGE, -0.35), 0.05: shade(ORANGE, 0.0), 0.025: shade(ORANGE, 0.45)}  # revtex single / double column width (inches)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


def save(fig, name: str, inputs: list[str]) -> Path:
    """Write `docs/paper1/figures/<name>.pdf` plus a provenance sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{name}.pdf"
    fig.savefig(pdf)
    prev = __import__("os").environ.get("FIG_PREVIEW_DIR")
    if prev:  # PNG preview for visual checks (not part of the paper)
        Path(prev).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(prev) / f"{name}.png", dpi=170)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never blocks a figure
        sha = "unknown"
    (OUT / f"{name}.meta.json").write_text(json.dumps({
        "figure": name, "git_sha": sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": sorted(inputs),
    }, indent=2))
    print(f"wrote {pdf}")
    return pdf


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# Main-text Kolmogorov alpha=0.1 model, chosen by the truth-free rule of task A10
# (docs/paper1/decisions.md; artifacts/experiments/a10_selection/decision.json).
def _main_a01() -> str:
    d = EXP / "a10_selection" / "decision.json"
    return json.loads(d.read_text())["selected"] if d.exists() else "k_xs01_v5"


MAIN_A01 = _main_a01()
