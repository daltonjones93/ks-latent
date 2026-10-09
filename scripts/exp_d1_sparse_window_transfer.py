"""D1 (docs/oct_1_exp.md Sec. 1): does Section 233's near-global propagator
receptive field (window=3, n_layers=2 -> 6/8 = 75% of the trained ring)
matter for alpha=0.1 Kolmogorov-flow L-transfer, or is linear drag alone
sufficient even with a genuinely sub-domain (window=1 -> 2/8 = 25%) window?

Single-variable retrain of Section 233's exact recipe with --attn-window 1
instead of 3. Everything else (n_sites_x=n_sites_y=8, local_channels=3,
alpha=0.1 dataset, seed 0, epoch counts) is identical, so d_latent is
unchanged (192) and this reuses the exact same Lyapunov/transfer cost
profile as every existing T1/G2/G3 run -- no new ground-truth computation
needed; the comparison is against Section 233's own already-measured
true_dky_kolmogorov_alpha0.1_{trained_size,2x_transfer,3x_transfer}.json
(the same ground truth Section 233's own headline numbers were measured
against).

Preemptable: every task is skipped if its output already exists. Light on
the machine: single-threaded BLAS/FFT/torch, os.nice(19), one task at a
time, lockfile.

Usage:
  python scripts/exp_d1_sparse_window_transfer.py            # run / resume
  python scripts/exp_d1_sparse_window_transfer.py --summarize
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
OUT = REPO / "artifacts" / "experiments" / "d1"
LOGS = REPO / "artifacts" / "logs"
DATASET = REPO / "artifacts" / "datasets" / "kolmogorov_trajectories_alpha0.1.h5"

ALPHA = 0.1
SIZES = (1, 2, 3)
IC_SEEDS_BY_SIZE = {1: (0,), 2: (0, 1, 2), 3: (0, 1, 2)}
TAG = "section235_kolmogorov_alpha0p1_localmlp2d_window1"
WINDOW = 1

# Section 233's own already-measured results, re-used as the comparison baseline
# (the exact same ground truth its own headline numbers were measured against).
SECTION233_TRUTH = {
    1: REPO / "artifacts/experiments/true_dky_kolmogorov_alpha0.1_trained_size.json",
    2: REPO / "artifacts/experiments/true_dky_kolmogorov_alpha0.1_2x_transfer.json",
    3: REPO / "artifacts/experiments/true_dky_kolmogorov_alpha0.1_3x_transfer.json",
}
SECTION233_MODEL = {
    1: REPO / "artifacts/experiments/g3_kolmogorov_alpha_transfer_section233_native.json",
    2: REPO / "artifacts/experiments/g3_kolmogorov_alpha_transfer_section233_2x.json",
    3: REPO / "artifacts/experiments/g3_kolmogorov_alpha_transfer_section233_3x.json",
}


def log(msg: str) -> None:
    print(f"[d1 {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ae_ckpt_path() -> Path:
    return REPO / f"artifacts/stage1_ae_kolmogorov_full_{TAG}.pt"


def prop_ckpt_path() -> Path:
    return REPO / f"artifacts/stage2_prop_kolmogorov_full_{TAG}_warmstart_k12_300ep.pt"


def model_result_path(size: int, ic_seed: int) -> Path:
    return REPO / "artifacts" / "experiments" / f"g3_kolmogorov_alpha_transfer_d1_window1_x{size}_ic{ic_seed}.json"


def run_train() -> None:
    logfile = LOGS / f"d1_train_{TAG}.log"
    log(f"train (window={WINDOW}) -> {TAG} (log: {logfile})")
    cmd = [sys.executable, "scripts/train_kolmogorov_localfield_standalone.py",
           "--dataset", str(DATASET), "--tag", TAG, "--seed", "0",
           "--stage1-epochs", "200", "--stage2-epochs", "300", "--checkpoint-every", "20",
           "--attn-window", str(WINDOW), "--resume"]
    with open(logfile, "a") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def run_model(size: int, ic_seed: int) -> None:
    tag = f"d1_window1_x{size}_ic{ic_seed}"
    logfile = LOGS / f"d1_model_{tag}.log"
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
            sys.exit(f"Another D1 run is active (pid {pid}); not starting a second one.")
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock from a killed run
    lock.write_text(str(os.getpid()))
    return lock


def summarize() -> dict:
    summary = {"tag": TAG, "window": WINDOW, "by_size": {}, "section233_window3": {}}
    for size in SIZES:
        truth = json.loads(SECTION233_TRUTH[size].read_text())["D_KY"]
        vals = []
        for ic in IC_SEEDS_BY_SIZE[size]:
            p = model_result_path(size, ic)
            if p.exists():
                vals.append(json.loads(p.read_text())["model_D_KY"])
        rel_errs = [abs(v - truth) / truth * 100.0 for v in vals]
        summary["by_size"][f"x{size}"] = {
            "true_D_KY": truth, "model_D_KY_values": vals,
            "rel_err_pct_values": rel_errs,
            "rel_err_pct_mean": (sum(rel_errs) / len(rel_errs)) if rel_errs else None,
        }
        if SECTION233_MODEL[size].exists():
            s233 = json.loads(SECTION233_MODEL[size].read_text())
            s233_err = abs(s233["model_D_KY"] - truth) / truth * 100.0
            summary["section233_window3"][f"x{size}"] = {
                "model_D_KY": s233["model_D_KY"], "rel_err_pct": s233_err,
            }

    x1 = summary["by_size"]["x1"]
    native_ok = bool(x1["model_D_KY_values"]) and x1["model_D_KY_values"][0] > 5.0
    summary["x1_collapse_check_passed"] = native_ok

    def verdict_at(size: int) -> str | None:
        d1 = summary["by_size"][f"x{size}"]["rel_err_pct_mean"]
        s233 = summary["section233_window3"].get(f"x{size}", {}).get("rel_err_pct")
        if d1 is None or s233 is None:
            return None
        margin = 3.0 if size == 2 else 5.0
        if d1 <= s233 + margin:
            return "PASS (coverage not load-bearing)"
        if d1 > 2 * s233:
            return "FAIL (coverage was load-bearing)"
        return "AMBIGUOUS (run window=2 before concluding)"

    summary["verdict"] = {
        "x1_collapse_check": "PASS" if native_ok else "FAIL (collapsed)",
        "x2": verdict_at(2), "x3": verdict_at(3),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d1_summary.json").write_text(json.dumps(summary, indent=2))
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
        log(json.dumps(s["verdict"], indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
