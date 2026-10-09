"""T1 (docs/ks_local_field_ltransfer.tex, Sec. transfer_diagnostic):
error bars on the alpha=0.1 Kolmogorov-flow extensivity/transfer numbers.

Three kinds of task, run strictly one at a time:
  truth  -- true-PDE D_KY at Lx in {1,2,3}x2pi, seeds {0,1,2}, total_time=120
            with running estimates recorded at t=40/80/120 (so the old
            t=40 numbers are reproduced as a consistency check and
            convergence in averaging time is visible).
  train  -- retrain Section 233's recipe with seeds 1 and 2.
  model  -- zero-retraining model D_KY for model seeds {0,1,2} x sizes
            {1,2,3}x x IC seeds {0,1,2} (via exp_g3_kolmogorov_alpha_transfer.py).

Preemptable: every task is skipped if its output already exists; truth
runs checkpoint their full Benettin state every few time units and resume
bitwise-identically; training resumes at the stage level. Kill it any time
and rerun the same command.

Light on the machine: single-threaded BLAS/FFT/torch, os.nice(19), one task
at a time, and a lockfile so two copies never run together.

Usage:
  python scripts/exp_t1_error_bars.py            # run / resume everything
  python scripts/exp_t1_error_bars.py --summarize
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

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "experiments" / "t1"
CKPT = OUT / "ckpt"
LOGS = REPO / "artifacts" / "logs"
DATASET = REPO / "artifacts" / "datasets" / "kolmogorov_trajectories_alpha0.1.h5"

ALPHA = 0.1
SIZES = (1, 2, 3)
SEEDS = (0, 1, 2)
N_DIRECTIONS = {1: 40, 2: 51, 3: 77}  # same as the original seed-0 runs
TOTAL_TIME = 120.0
RECORD_TIMES = (40.0, 80.0, 120.0)
WARMUP_TIME = 20.0
QR_INTERVAL = 0.5
CHECKPOINT_EVERY_TIME = 4.0

BASE_TAG = "section233_kolmogorov_alpha0p1_localmlp2d_window3"
# Model-seed-0 results that already exist from the G2-resolved experiments.
EXISTING_MODEL_TAGS = {
    (0, 1, 0): "section233_native",
    (0, 2, 0): "section233_2x",
    (0, 2, 1): "section233_2x_seed1",
    (0, 2, 2): "section233_2x_seed2",
    (0, 3, 0): "section233_3x",
}
AREA0 = (2 * math.pi) ** 2


def model_tag(model_seed: int) -> str:
    return BASE_TAG if model_seed == 0 else f"{BASE_TAG}_seed{model_seed}"


def model_ckpts(model_seed: int) -> tuple[Path, Path]:
    tag = model_tag(model_seed)
    return (REPO / "artifacts" / f"stage1_ae_kolmogorov_full_{tag}.pt",
            REPO / "artifacts" / f"stage2_prop_kolmogorov_full_{tag}_warmstart_k12_300ep.pt")


def model_result_path(model_seed: int, size: int, ic_seed: int) -> Path:
    tag = EXISTING_MODEL_TAGS.get((model_seed, size, ic_seed),
                                  f"t1_m{model_seed}_x{size}_ic{ic_seed}")
    return REPO / "artifacts" / "experiments" / f"g3_kolmogorov_alpha_transfer_{tag}.json"


def truth_result_path(size: int, seed: int) -> Path:
    return OUT / f"truth_x{size}_seed{seed}.json"


def log(msg: str) -> None:
    print(f"[t1 {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_truth(size: int, seed: int) -> None:
    from ks_latent.analysis.lyapunov import kaplan_yorke_dimension, lyapunov_spectrum_kolmogorov
    from ks_latent.config import KolmogorovConfig

    cfg = KolmogorovConfig(
        Lx=2 * math.pi * size, Ly=2 * math.pi, NX=128 * size, NY=128, nu=0.01, alpha=ALPHA,
        forcing_n=4, dt=0.01, spinup_time=100.0, seed=seed,
    )
    ckpt = CKPT / f"truth_x{size}_seed{seed}.npz"
    log(f"truth x{size} seed{seed}: n_dir={N_DIRECTIONS[size]} total_time={TOTAL_TIME} "
        f"({'resuming' if ckpt.exists() else 'fresh'})")
    t0 = time.time()
    res = lyapunov_spectrum_kolmogorov(
        cfg, n_directions=N_DIRECTIONS[size], total_time=TOTAL_TIME, qr_interval=QR_INTERVAL,
        warmup_time=WARMUP_TIME, seed=seed, max_abs_state=500.0,
        checkpoint_path=str(ckpt), checkpoint_every_time=CHECKPOINT_EVERY_TIME,
        record_times=list(RECORD_TIMES),
    )
    records = {}
    for t, ex in res.partial_exponents.items():
        try:
            dky = kaplan_yorke_dimension(ex)
        except ValueError:
            dky = None
        records[f"{t:g}"] = {"lambda1": float(ex[0]), "n_positive": int(np.sum(ex > 0)),
                             "D_KY": dky, "exponents": [float(x) for x in ex]}
    out = {"size": size, "seed": seed, "alpha": ALPHA, "Lx": cfg.Lx, "Ly": cfg.Ly,
           "NX": cfg.NX, "NY": cfg.NY, "n_directions": N_DIRECTIONS[size],
           "warmup_time": WARMUP_TIME, "qr_interval": QR_INTERVAL,
           "records": records, "wall_s_this_session": time.time() - t0}
    truth_result_path(size, seed).write_text(json.dumps(out, indent=2))
    log(f"truth x{size} seed{seed} done: D_KY by averaging time "
        + ", ".join(f"t={k}: {v['D_KY']}" for k, v in records.items()))


def run_train(model_seed: int) -> None:
    tag = model_tag(model_seed)
    logfile = LOGS / f"t1_train_{tag}.log"
    log(f"train model seed {model_seed} -> {tag} (log: {logfile})")
    cmd = [sys.executable, "scripts/train_kolmogorov_localfield_standalone.py",
           "--dataset", str(DATASET), "--tag", tag, "--seed", str(model_seed),
           "--stage1-epochs", "200", "--stage2-epochs", "300", "--checkpoint-every", "20",
           "--resume"]
    with open(logfile, "a") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def run_model(model_seed: int, size: int, ic_seed: int) -> None:
    ae, prop = model_ckpts(model_seed)
    tag = f"t1_m{model_seed}_x{size}_ic{ic_seed}"
    logfile = LOGS / f"t1_model_{tag}.log"
    log(f"model D_KY: model seed {model_seed}, x{size}, IC seed {ic_seed}")
    cmd = [sys.executable, "scripts/exp_g3_kolmogorov_alpha_transfer.py",
           "--ae-ckpt", str(ae), "--prop-ckpt", str(prop), "--x-scale", str(float(size)),
           "--alpha", str(ALPHA), "--seed", str(ic_seed), "--tag", tag]
    with open(logfile, "a") as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def task_list():
    """(name, done_fn, run_fn) in execution order: cheap truth first, then
    retraining, then the model evaluations, then the expensive truth runs."""
    tasks = []
    for s in SEEDS:
        tasks.append((f"truth x1 seed{s}", lambda s=s: truth_result_path(1, s).exists(),
                      lambda s=s: run_truth(1, s)))
    for m in (1, 2):
        tasks.append((f"train seed{m}", lambda m=m: model_ckpts(m)[1].exists(),
                      lambda m=m: run_train(m)))
    for m in SEEDS:
        for size in SIZES:
            for ic in SEEDS:
                tasks.append((f"model m{m} x{size} ic{ic}",
                              lambda m=m, size=size, ic=ic: model_result_path(m, size, ic).exists(),
                              lambda m=m, size=size, ic=ic: run_model(m, size, ic)))
    for size in (2, 3):
        for s in SEEDS:
            tasks.append((f"truth x{size} seed{s}", lambda size=size, s=s: truth_result_path(size, s).exists(),
                          lambda size=size, s=s: run_truth(size, s)))
    return tasks


def acquire_lock() -> Path:
    lock = OUT / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            sys.exit(f"Another T1 run is active (pid {pid}); not starting a second one.")
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock from a killed run
    lock.write_text(str(os.getpid()))
    return lock


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def summarize() -> dict:
    summary = {"truth": {}, "model": {}, "transfer_error": {}}
    for size in SIZES:
        area = AREA0 * size
        per_t = {}
        for t in RECORD_TIMES:
            vals = []
            for s in SEEDS:
                p = truth_result_path(size, s)
                if p.exists():
                    vals.append(json.loads(p.read_text())["records"][f"{t:g}"]["D_KY"])
            m, sd, n = _mean_std(vals)
            per_t[f"{t:g}"] = {"D_KY_mean": m, "D_KY_std": sd, "n": n,
                               "rho_mean": m / area if m is not None else None,
                               "rho_std": sd / area if sd is not None else None,
                               "values": vals}
        summary["truth"][f"x{size}"] = per_t

        mvals, by_model = [], {}
        for m in SEEDS:
            vs = []
            for ic in SEEDS:
                p = model_result_path(m, size, ic)
                if p.exists():
                    vs.append(json.loads(p.read_text())["model_D_KY"])
            by_model[f"m{m}"] = vs
            mvals += vs
        mm, msd, mn = _mean_std(mvals)
        summary["model"][f"x{size}"] = {"D_KY_mean": mm, "D_KY_std": msd, "n": mn, "by_model_seed": by_model}

        tr = per_t[f"{TOTAL_TIME:g}"]
        if tr["D_KY_mean"] and mm is not None:
            summary["transfer_error"][f"x{size}"] = {
                "rel_err_of_means": abs(mm - tr["D_KY_mean"]) / tr["D_KY_mean"],
                "rel_err_range": [abs(v - tr["D_KY_mean"]) / tr["D_KY_mean"] for v in mvals],
            }

    # Pre-registered T1 decision rule: is the 2x dip in rho outside the 1-sigma
    # bands of the 1x and 3x values (at the longest averaging time)?
    rows = [summary["truth"][f"x{s}"][f"{TOTAL_TIME:g}"] for s in SIZES]
    if all(r["n"] >= 2 for r in rows):
        (r1, r2, r3) = rows
        dip_real = (r2["rho_mean"] + r2["rho_std"] < r1["rho_mean"] - r1["rho_std"]) and \
                   (r2["rho_mean"] + r2["rho_std"] < r3["rho_mean"] - r3["rho_std"])
        summary["decision_rule_2x_dip_is_real"] = bool(dip_real)
    else:
        summary["decision_rule_2x_dip_is_real"] = "insufficient data"

    (OUT / "t1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--list", action="store_true", help="show task status and exit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        print(json.dumps(summarize(), indent=2))
        return
    tasks = task_list()
    if args.list:
        for name, done, _ in tasks:
            print(f"{'done   ' if done() else 'pending'}  {name}")
        return

    os.nice(19)
    lock = acquire_lock()
    try:
        pending = [t for t in tasks if not t[1]()]
        log(f"{len(tasks) - len(pending)}/{len(tasks)} tasks already done; {len(pending)} to go")
        failed = []
        for name, done, run in pending:
            if done():
                continue
            try:
                run()
            except Exception as exc:  # record loudly, keep going with independent tasks
                failed.append(name)
                log(f"FAILED: {name}: {type(exc).__name__}: {exc}")
                with open(OUT / "failed_tasks.log", "a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{type(exc).__name__}: {exc}\n")
                continue
            summarize()
        if failed:
            log(f"{len(failed)} task(s) FAILED, see {OUT / 'failed_tasks.log'}: {failed}")
        else:
            log("all T1 tasks complete")
        print(json.dumps(summarize(), indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
