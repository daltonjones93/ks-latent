"""D3 (docs/oct_1_exp.md Sec. 3): does a SECOND, genuinely trainable
scale (`backbone="multiscale_local_mlp_2d"`) -- a regional mixer with its
own learned weights, exchanging with the fine scale via learned (not
fixed-formula) projections -- recover L-transfer at alpha values where
both a broad-window local model (D1) and an exact-physics but
unnormalized nonlocal term (D2, which collapsed under transfer) failed?

Primary test: alpha=0.05, same baseline as D2 (Section 231's own un-fixed
~50% x2 transfer error, artifacts/experiments/g2_kolmogorov_transfer_full.json).
Reuses Section 231's existing Stage-1 AE checkpoint as-is (--reuse-ae-ckpt).

Preemptable, single-threaded, one task at a time, lockfile -- same
discipline as D1/D2/T1.

Usage:
  python scripts/exp_d3_multiscale_transfer.py            # run / resume
  python scripts/exp_d3_multiscale_transfer.py --summarize
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "experiments" / "d3"
LOGS = REPO / "artifacts" / "logs"
DATASET = REPO / "artifacts" / "datasets" / "kolmogorov_trajectories_default.h5"  # alpha=0.05
REUSE_AE_CKPT = REPO / "artifacts" / "stage1_ae_kolmogorov_full_section231_kolmogorov_localmlp2d_window3.pt"

ALPHA = 0.05
SIZES = (1, 2, 3)
IC_SEEDS_BY_SIZE = {1: (0,), 2: (0, 1, 2), 3: (0, 1, 2)}
TAG = "section237_kolmogorov_alpha0p05_multiscale_local_mlp2d"
ATTN_WINDOW = 3  # fine scale's window, matches Section 231's own

BASELINE_MODEL_X2 = REPO / "artifacts/experiments/g2_kolmogorov_transfer_full.json"
D2_SUMMARY = REPO / "artifacts/experiments/d2/d2_summary.json"


def log(msg: str) -> None:
    print(f"[d3 {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ae_ckpt_path() -> Path:
    return REUSE_AE_CKPT


def prop_ckpt_path() -> Path:
    return REPO / f"artifacts/stage2_prop_kolmogorov_full_{TAG}_warmstart_k12_300ep.pt"


def model_result_path(size: int, ic_seed: int) -> Path:
    return REPO / "artifacts" / "experiments" / f"g3_kolmogorov_alpha_transfer_d3_multiscale_x{size}_ic{ic_seed}.json"


def run_train() -> None:
    logfile = LOGS / f"d3_train_{TAG}.log"
    log(f"train (backbone=multiscale_local_mlp_2d, reuse AE={REUSE_AE_CKPT.name}) -> {TAG} (log: {logfile})")
    cmd = [sys.executable, "scripts/train_kolmogorov_localfield_standalone.py",
           "--dataset", str(DATASET), "--tag", TAG, "--seed", "0",
           "--stage1-epochs", "200", "--stage2-epochs", "300", "--checkpoint-every", "20",
           "--backbone", "multiscale_local_mlp_2d", "--attn-window", str(ATTN_WINDOW),
           "--reuse-ae-ckpt", str(REUSE_AE_CKPT), "--resume"]
    with open(logfile, "a") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def run_model(size: int, ic_seed: int) -> None:
    tag = f"d3_multiscale_x{size}_ic{ic_seed}"
    logfile = LOGS / f"d3_model_{tag}.log"
    log(f"model D_KY: x{size}, IC seed {ic_seed}")
    cmd = [sys.executable, "scripts/exp_g3_kolmogorov_alpha_transfer.py",
           "--ae-ckpt", str(ae_ckpt_path()), "--prop-ckpt", str(prop_ckpt_path()),
           "--x-scale", str(float(size)), "--alpha", str(ALPHA), "--seed", str(ic_seed),
           "--tag", tag]
    with open(logfile, "a") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def task_list():
    tasks = [("train", lambda: prop_ckpt_path().exists(), run_train)]
    for size in SIZES:
        for ic in IC_SEEDS_BY_SIZE[size]:
            tasks.append((
                f"model x{size} ic{ic}",
                lambda size=size, ic=ic: model_result_path(size, ic).exists(),
                lambda size=size, ic=ic: run_model(size, ic),
            ))
    return tasks


def acquire_lock() -> Path:
    lock = OUT / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            sys.exit(f"Another D3 run is active (pid {pid}); not starting a second one.")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    lock.write_text(str(os.getpid()))
    return lock


def summarize() -> dict:
    summary = {"tag": TAG, "alpha": ALPHA, "by_size": {}}
    baseline = json.loads(BASELINE_MODEL_X2.read_text())
    baseline_true = baseline["true_D_KY"]
    baseline_x2_err = abs(baseline["model_D_KY"] - baseline_true) / baseline_true * 100.0
    summary["section231_window3_baseline_x2_err_pct"] = baseline_x2_err
    if D2_SUMMARY.exists():
        d2 = json.loads(D2_SUMMARY.read_text())
        summary["d2_biotsavart_x2_err_pct_mean"] = d2["by_size"]["x2"].get("rel_err_pct_mean")

    for size in SIZES:
        vals = []
        for ic in IC_SEEDS_BY_SIZE[size]:
            p = model_result_path(size, ic)
            if p.exists():
                vals.append(json.loads(p.read_text())["model_D_KY"])
        entry = {"model_D_KY_values": vals}
        if size == 2:
            rel_errs = [abs(v - baseline_true) / baseline_true * 100.0 for v in vals]
            entry["true_D_KY"] = baseline_true
            entry["rel_err_pct_values"] = rel_errs
            entry["rel_err_pct_mean"] = (sum(rel_errs) / len(rel_errs)) if rel_errs else None
        summary["by_size"][f"x{size}"] = entry

    x1 = summary["by_size"]["x1"]
    native_ok = bool(x1["model_D_KY_values"]) and x1["model_D_KY_values"][0] > 5.0
    summary["x1_collapse_check_passed"] = native_ok

    x2 = summary["by_size"]["x2"]
    if x2.get("rel_err_pct_mean") is not None:
        d3_err = x2["rel_err_pct_mean"]
        if d3_err <= 10.0:
            verdict = "CONFIRMED FIX (single digits, matches drag-at-0.1's own achievement)"
        elif d3_err < baseline_x2_err - 10:
            verdict = "PARTIAL FIX (better than baseline but not yet single digits)"
        else:
            verdict = "NOT CONFIRMED (no meaningful improvement over the un-fixed baseline)"
        summary["verdict_x2"] = verdict

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        s = summarize()
        print(json.dumps(s, indent=2))
        return

    try:
        os.nice(19)
    except OSError:
        pass
    lock = acquire_lock()
    try:
        for name, done_fn, run_fn in task_list():
            if done_fn():
                log(f"skip (already done): {name}")
                continue
            log(f"running: {name}")
            run_fn()
            log(f"done: {name}")
        s = summarize()
        log("all tasks complete. Verdict:")
        log(json.dumps({"x1_collapse_check_passed": s["x1_collapse_check_passed"],
                         "verdict_x2": s.get("verdict_x2")}, indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
