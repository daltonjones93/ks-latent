"""Paper II referee M2 (2026-10-13): rerun the two 220^2 Sec. 6 cases (J5_L220_lap1.25_est, J5_L220_adv1_est;
param_min_std 0.005, stalled estimation) with the R7 settings: param_min_std 0.02 and the case's R7 best state RTPS
at 44^2 (lap1.25: 0.6; adv1: 0.7). Seed 0, all other args copied. Out: i11_R7_J5_L220_<case>_est.json."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "scripts")
from run_r7 import PY, cli  # noqa: E402

E = Path("artifacts/experiments/da2d")
ENV = {**os.environ, "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "VECLIB_MAXIMUM_THREADS": "3"}
CASES = {"lap1.25": 0.6, "adv1": 0.7}


def run(case):
    a = json.load(open(E / f"i11_J5_L220_{case}_est.json"))["args"]
    tag = f"R7_J5_L220_{case}_est"
    a.update(param_min_std=0.02, rtps=CASES[case], seed=0, out=tag)
    subprocess.run([PY, "scripts/exp_i11_pde_letkf.py", *cli(a)], env=ENV,
                   stdout=open(f"artifacts/logs/i11_{tag}.log", "w"), stderr=subprocess.STDOUT)


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, CASES))
