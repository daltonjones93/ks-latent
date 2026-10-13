"""Paper II referee M5 (2026-10-13): one-cycle model error of each estimation case AFTER the remedy, i.e. with the
model set to the parameters its selected filter configuration converged to (steady-window theta, mean over seeds
0-2), so the state-dependent error on the x-axis of Fig. model_error is defined the same way for every case.
Same estimator, independent seed-3 truth, 100 starts, 5 propagator steps as the originals. Writes
da2d/j16_model_error_<orig>_post.{npz,json}; summary in da2d/m5_post_summary.json."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PY = "/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python"
E = Path("artifacts/experiments/da2d")
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
# orig j16 name -> (selected config prefix (seeds _s0.._s2), kind, extra estimator args)
CASES = {
    "lap1.1": ("R7_J16a_lap1.1_rtps0.7_pf0.02_r0.5", "lap", []),
    "lap1.25": ("R7_J16a_lap1.25_rtps0.8_pf0.02_r0.6", "lap", []),
    "adv1": ("R7_J16a_adv1_rtps0.9_pf0.02_r0.7", "adv", []),
    "adv2": ("R7_J16a_adv2_rtps1.0_pf0.02_r0.8", "adv", []),
    "hyp5": ("R7_J17H_p5_c16_rtps0.7_abg_pf0.02_r0.5", "abg", ["--model-order", "5"]),
    "hyp6": ("R7_J17H_p6_c16_rtps0.7_abg_pf0.02_r0.7", "abg", ["--model-order", "6"]),
    "gani0.1": ("R7_J17G_d0.1_pde_c8.25_rtps0.7_abg_pf0.02_r0.7", "abg", ["--truth-ganiso", "0.1"]),
    "gani0.2": ("R7_J17G_d0.2_pde_c8.25_rtps0.8_abg_pf0.02_r0.8", "abg", ["--truth-ganiso", "0.2"]),
    "gani0.3": ("J17G_d0.3_pde_c8.25_rtps0.8_abg", "abg", ["--truth-ganiso", "0.3"]),
    "memB0.1": ("R7_J17B_e0.1_pde_c8.25_rtps0.7_lap_pf0.02_r0.5", "lap", ["--truth-memory", "0.1,2,-1"]),
    "memB0.2": ("J17B_e0.2_pde_c8.25_rtps0.7_lap", "lap", ["--truth-memory", "0.2,2,-1"]),
}


def theta(prefix, kind):
    v = []
    for s in range(3):
        d = json.load(open(E / f"i11_{prefix}_s{s}.json"))
        v.append(d["steady_vec"]["theta_mean"] if kind == "abg" else d["steady"]["theta_mean"])
    return np.mean(np.array(v, dtype=float), axis=0).tolist(), d["args"]


def run(name):
    prefix, kind, extra = CASES[name]
    th, args = theta(prefix, kind)
    if kind == "lap":
        model = ["--model-lap-coef", f"{th:.6f}"]
    elif kind == "adv":
        model = ["--model-advect-x", f"{th:.6f}"]
    else:
        model = ["--model-lap-coef", f"{th[0]:.6f}", "--model-hyper-coef", f"{th[1]:.6f}", "--model-nl-coef", f"{th[2]:.6f}"]
    out = f"{name}_post"
    if not (E / f"j16_model_error_{out}.json").exists():
        subprocess.run([PY, "scripts/exp_j16_estimate_pde_model_error.py", *model, *extra, "--n-starts", "100",
                        "--n-prop-steps", "5", "--seed", "3", "--out", out], env=ENV,
                       stdout=open(f"artifacts/logs/m5_{out}.log", "w"), stderr=subprocess.STDOUT, check=True)
    before = json.load(open(E / f"j16_model_error_{name}.json"))["sigma"]
    after = json.load(open(E / f"j16_model_error_{out}.json"))["sigma"]
    return {"case": name, "config": prefix, "theta": th, "norm_std": args.get("norm_std"),
            "sigma_before": before, "sigma_after": after}


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(run, CASES))
    (E / "m5_post_summary.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(r)
