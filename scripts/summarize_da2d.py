"""Tabulate the 2D DA rerun (task E11): one row per run, with whole-run and
steady-state (cycles 50-100) skill and calibration, CPU time and peak RSS
parsed from the `/usr/bin/time -l` output saved next to each JSON."""

import json
import re
import statistics as st
import sys
from pathlib import Path

OUT = Path("artifacts/experiments/da2d")


def time_stats(tag: str) -> tuple[float, float, float]:
    p = OUT / f"da2d_{tag}.time"
    if not p.exists():
        return float("nan"), float("nan"), float("nan")
    t = p.read_text()
    real = re.search(r"([\d.]+) real", t)
    user = re.search(r"([\d.]+) user", t)
    sys_ = re.search(r"([\d.]+) sys", t)
    rss = re.search(r"(\d+)\s+maximum resident set size", t)
    cpu = float(user.group(1)) + float(sys_.group(1)) if user and sys_ else float("nan")
    return (float(real.group(1)) if real else float("nan"), cpu,
            int(rss.group(1)) / 2**30 if rss else float("nan"))


def main() -> None:
    rows = []
    for p in sorted(OUT.glob("da2d_*.json")):
        tag = p.stem[len("da2d_"):]
        d = json.loads(p.read_text())
        fr, da, sp = d["rmse_free_per_cycle"], d["rmse_da_per_cycle"], d["spread_per_cycle"]
        late = slice(len(da) // 2, None)
        skill_late = st.mean(f / a for f, a in zip(fr[late], da[late]))
        calib_late = st.mean(s / a for s, a in zip(sp[late], da[late]))
        real, cpu, rss = time_stats(tag)
        rows.append((tag, d["L"], d["d_latent"], d["n_ensemble"], d["method"], d["localizer"],
                     d.get("model_noise_std"), d["rmse_da"], d["rmse_free"], d["skill_free_over_da"],
                     skill_late, d["calibration_spread_over_rmse"], calib_late, cpu, rss))
    hdr = ("tag", "L", "d", "N", "method", "loc", "noise", "rmse_da", "rmse_free", "skill",
           "skill_ss", "calib", "calib_ss", "cpu_s", "rss_GB")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps([dict(zip(hdr, r)) for r in rows], indent=2))


if __name__ == "__main__":
    main()
