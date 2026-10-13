"""R7 (Paper II referee, 2026-10-12): online parameter estimation stalls. In the J16a / J17G estimation runs the
parameter spread reaches its floor (0.005; for 'abg' relative) by cycle 30-40 and the estimates stick
(nabla^2 1.25 -> 1.17; advection 2 -> 0.40; advection 1 -> 0.07). Rerun each estimation case with parameter
inflation: spread floor 0.02 and 0.05 (absolute for lap/adv, relative for abg; parameter RTPS 0.9 as before), and the
state RTPS retuned (the case's best value r and r - 0.2, since parameter spread adds state spread). Seed 0 grid first,
then seeds 1-2 at each case's best seed-0 setting. Every other argument is copied from the original run's JSON.
Three runs at a time, 2 threads each. Log: artifacts/logs/r7.log.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PY = "/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python"
E = Path("artifacts/experiments/da2d")
LOG = Path("artifacts/logs/r7.log")
BASES = ["J16a_lap1.25_rtps0.8_s0", "J16a_adv1_rtps0.9_s0", "J16a_adv2_rtps1.0_s0",
         "J17G_d0.1_pde_c8.25_rtps0.7_abg_s0", "J17G_d0.2_pde_c8.25_rtps0.8_abg_s0", "J17G_d0.3_pde_c8.25_rtps0.8_abg_s0"]
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}


def log(msg):
    with LOG.open("a") as f:
        f.write(f"{datetime.now():%F %T} [R7] {msg}\n")


def cli(args):
    out = []
    for k, v in args.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        out += [flag] if v is True else [flag, str(v)]
    return out


def run(base, floor, rtps, seed):
    a = json.load(open(E / f"i11_{base}.json"))["args"]
    tag = f"R7_{base.rsplit('_s0', 1)[0]}_pf{floor}_r{rtps}_s{seed}"
    if (E / f"i11_{tag}.json").exists():
        return tag
    a.update(param_min_std=floor, rtps=rtps, seed=seed, out=tag, workers=2)
    r = subprocess.run([PY, "scripts/exp_i11_pde_letkf.py", *cli(a)], env=ENV,
                       stdout=open(f"artifacts/logs/i11_{tag}.log", "w"), stderr=subprocess.STDOUT)
    res = E / f"i11_{tag}.json"
    s = json.load(open(res))["steady"] if res.exists() else {}
    log(f"{tag} rc={r.returncode} analysis {s.get('analysis', float('nan')):.3f} spread {s.get('spread', float('nan')):.3f}")
    return tag


def steady(tag):
    f = E / f"i11_{tag}.json"
    return json.load(open(f))["steady"]["analysis"] if f.exists() else float("inf")


def main():
    jobs = []
    for b in BASES:
        r0 = json.load(open(E / f"i11_{b}.json"))["args"]["rtps"]
        for fl in (0.02, 0.05):
            for r in (r0, round(r0 - 0.2, 2)):
                jobs.append((b, fl, r, 0))
    log(f"start: {len(jobs)} seed-0 runs")
    with ThreadPoolExecutor(3) as ex:
        tags = list(ex.map(lambda j: run(*j), jobs))
    best = {}
    for j, t in zip(jobs, tags):
        if j[0] not in best or steady(t) < steady(best[j[0]][1]):
            best[j[0]] = (j, t)
    for b, (j, t) in best.items():
        log(f"best seed 0 for {b}: {t} ({steady(t):.3f}; original {json.load(open(E / f'i11_{b}.json'))['steady']['analysis']:.3f})")
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: run(*j), [(j[0], j[1], j[2], s) for j, _ in best.values() for s in (1, 2)]))
    log("R7 DONE")


if __name__ == "__main__":
    sys.exit(main())
