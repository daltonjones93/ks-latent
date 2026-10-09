"""Crop-training pipeline (docs/ks_local_field_ltransfer.tex
Sec. crop_impl): P0(alpha=0.1) -> P1 -> P0(alpha=0.05) -> P2 -> truth
alpha=0.05 at 3x. Strictly one task at a time; waits for the C0/C4 chain and
the queued 2D-KS seed runs to finish first. Preemptable: every task is
skipped if its output exists; the truth run checkpoints its Benettin state.

Usage: python scripts/exp_crop_pipeline.py [--summarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "crop"
LOGS = A / "logs"
WAIT_FOR = ("run_c0_then_c4.sh", "run_icnorm_ks2d_seeds.sh")
RUNS = {
    "p1": {"alpha": 0.1, "data": A / "datasets/kolmogorov_wide16_alpha0.1.h5", "tag": "p1_a01"},
    "p2": {"alpha": 0.05, "data": A / "datasets/kolmogorov_wide16_alpha0.05.h5", "tag": "p2_a005"},
}
SCALES, SEEDS = (1, 2, 3), (0, 1, 2)
TRUTH3_A005 = OUT / "truth_a005_x3_seed0.json"


def log(msg):
    print(f"[pipe {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sh(cmd, logname):
    with open(LOGS / logname, "a") as fh:
        subprocess.run([sys.executable] + cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def ckpts(tag):
    return A / f"stage1_ae_kolmogorov_crop_{tag}.pt", A / f"stage2_prop_kolmogorov_crop_{tag}.pt"


def eval_path(tag, s, seed):
    return OUT / f"{tag}_x{s}_seed{seed}.json"


def run_truth_a005_x3():
    from ks_latent.analysis.lyapunov import kaplan_yorke_dimension, lyapunov_spectrum_kolmogorov
    from ks_latent.config import KolmogorovConfig
    cfg = KolmogorovConfig(Lx=6 * math.pi, Ly=2 * math.pi, NX=384, NY=128, nu=0.01, alpha=0.05, forcing_n=4,
                           dt=0.01, spinup_time=100.0, seed=0)
    t0 = time.time()
    res = lyapunov_spectrum_kolmogorov(
        cfg, n_directions=128, total_time=40.0, qr_interval=0.5, warmup_time=20.0, seed=0,
        max_abs_state=500.0, checkpoint_path=str(OUT / "truth_a005_x3_seed0.ckpt.npz"), checkpoint_every_time=4.0,
    )
    ex = res.exponents
    TRUTH3_A005.write_text(json.dumps({
        "alpha": 0.05, "scale": 3, "seed": 0, "n_directions": 128, "D_KY": kaplan_yorke_dimension(ex),
        "lambda1": float(ex[0]), "n_positive": int((ex > 0).sum()), "exponents": [float(x) for x in ex],
        "wall_s_this_session": time.time() - t0}, indent=2))


def tasks():
    t = []
    for key in ("p1", "p2"):
        r = RUNS[key]
        t.append((f"P0 alpha={r['alpha']}", lambda r=r: r["data"].exists(),
                  lambda r=r: sh(["scripts/gen_kolmogorov_wide_dataset.py", "--alpha", str(r["alpha"]),
                                  "--out", str(r["data"])], f"p0_wide_a{r['alpha']}.log")))
        t.append((f"{key.upper()} train", lambda r=r: ckpts(r["tag"])[1].exists(),
                  lambda r=r: sh(["scripts/train_kolmogorov_crop.py", "--dataset", str(r["data"]),
                                  "--tag", r["tag"]], f"crop_train_{r['tag']}.log")))
        for s in SCALES:
            for seed in SEEDS:
                truth_arg = (["--truth-json", str(TRUTH3_A005)] if (key == "p2" and s == 3) else [])
                t.append((f"{key.upper()} eval x{s} seed{seed}",
                          lambda r=r, s=s, seed=seed: eval_path(r["tag"], s, seed).exists(),
                          lambda r=r, s=s, seed=seed, ta=truth_arg: sh(
                              ["scripts/exp_crop_eval.py", "--ae-ckpt", str(ckpts(r["tag"])[0]),
                               "--prop-ckpt", str(ckpts(r["tag"])[1]), "--scale", str(s), "--seed", str(seed),
                               "--tag", f"{r['tag']}_x{s}_seed{seed}"] + ta, f"crop_eval_{r['tag']}.log")))
    t.append(("truth alpha=0.05 x3 seed0", lambda: TRUTH3_A005.exists(), run_truth_a005_x3))
    return t


def summarize():
    out = {}
    for key, r in RUNS.items():
        rows = {}
        for s in SCALES:
            vals = [json.loads(eval_path(r["tag"], s, sd).read_text()) for sd in SEEDS if eval_path(r["tag"], s, sd).exists()]
            if not vals:
                continue
            dky = [v["model_D_KY"] for v in vals]
            truth = vals[0]["true_D_KY"]
            if s == 3 and key == "p2" and TRUTH3_A005.exists():
                truth = json.loads(TRUTH3_A005.read_text())["D_KY"]
            errs = [abs(d - truth) / truth * 100 for d in dky] if truth else None
            rows[f"x{s}"] = {"D_KY": dky, "truth": truth, "err_pct": errs,
                             "err_pct_mean": sum(errs) / len(errs) if errs else None}
        out[key] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crop_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    if ap.parse_args().summarize:
        summarize()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    while any(subprocess.run(["pgrep", "-f", w], capture_output=True).returncode == 0 for w in WAIT_FOR):
        time.sleep(30)
    log("prerequisites finished; starting")
    for name, done, run in tasks():
        if done():
            log(f"skip (done): {name}")
            continue
        log(f"running: {name}")
        try:
            run()
            log(f"done: {name}")
        except subprocess.CalledProcessError as e:
            log(f"FAILED: {name} ({e}); continuing")
    summarize()
    log("ALL DONE")


if __name__ == "__main__":
    main()
