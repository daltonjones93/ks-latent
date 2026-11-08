"""Run a list of DA studies under the G0 protocol: tune on the validation storms, then score the test storms.

Each study in the JSON spec:
  {"name": "n4_c32_N8", "run": "full_v1_stagec", "args": "--modes latent --n-members 8 --random-members ...",
   "grid": {"--model-noise": [0.05, 0.1, 0.15]}, "test": "pick" | "all" | "none"}
For every point of the grid (the product of its values) a --tuning study runs on the 6 validation storms
(out <root>/<name>/val_<point>); the point with the lowest validation prior nMSE of the study's (single) mode is
chosen and written to <root>/<name>/choice.json; then the test storms run (--include-test) at the chosen point
("pick"), at every point ("all", for curves such as the localization sweep; the choice is still recorded), or not
at all. Finished studies (study.json present) are skipped, so the driver can be re-run after a time-out.

  python scripts/cm1/da_sweep.py --spec cluster/saidie/specs/n4.json --root ~/cm1_da/fable --source cache:A+A2+B
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def point_args(grid: dict, values: tuple) -> list:
    out = []
    for k, v in zip(grid, values):
        out += [k] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    return out


def tag(values: tuple) -> str:
    return "_".join("-".join(map(str, v)) if isinstance(v, list) else str(v) for v in values) or "base"


def study(out: Path, run: str, args: list, a, log) -> dict:
    if not (out / "study.json").exists():
        cmd = [sys.executable, str(HERE / "da_study.py"), "--run", str(Path(a.runs) / run), "--source", a.source,
               "--out", str(out), "--device", a.device] + args
        log(f"[sweep] {' '.join(shlex.quote(c) for c in cmd)}")
        t = time.time()
        subprocess.run(cmd, check=True)
        log(f"[sweep]   done in {(time.time() - t) / 60:.1f} min")
    return json.loads((out / "study.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", default=str(Path.home() / "cm1_runs"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--only", nargs="*", default=[], help="run only these study names")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    root = Path(a.root).expanduser(); root.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    specs = json.loads(Path(a.spec).read_text())
    for sp in specs:
        if a.only and sp["name"] not in a.only:
            continue
        base = shlex.split(sp["args"])
        modes = base[base.index("--modes") + 1: base.index("--modes") + 2] if "--modes" in base else ["latent"]
        grid = sp.get("grid", {})
        pts = list(itertools.product(*grid.values())) if grid else [()]
        d = root / sp["name"]; d.mkdir(exist_ok=True)
        rows = []
        for v in pts:
            s = study(d / f"val_{tag(v)}", sp["run"], base + point_args(grid, v) + ["--tuning"], a, log)
            m = s["modes"][modes[0]]["prior"]
            rows.append({"point": dict(zip(grid, v)), "tag": tag(v), "val_prior_nmse": m["nmse_mean_dec"]["mean"],
                         "val_spread_err": (m["th_obs_spread"]["mean"] / m["th_obs_rmse"]["mean"])
                         if "th_obs_spread" in m and "th_obs_rmse" in m else None})
        best = min(rows, key=lambda r: r["val_prior_nmse"])
        (d / "choice.json").write_text(json.dumps({"criterion": "lowest validation prior nMSE", "chosen": best,
                                                   "table": rows, "spec": sp}, indent=1))
        log(f"[sweep] {sp['name']}: " + "  ".join(f"{r['tag']} {r['val_prior_nmse']:.4f}" for r in rows)
            + f"  -> {best['tag']}")
        mode = sp.get("test", "pick")
        for v in pts if mode == "all" else ([tuple(best["point"].values())] if mode == "pick" else []):
            study(d / f"test_{tag(v)}", sp["run"], base + point_args(grid, v) + ["--include-test"], a, log)
    log("[sweep] all done")


if __name__ == "__main__":
    main()
