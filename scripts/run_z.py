"""Block Z (2026-10-12): every remaining run Papers I/II need, beyond what is already running (Z2).
Thread-capped (machine load ~177 on 18 cores): a pool of 3 jobs at OMP_NUM_THREADS=3, the long 2D KS 88^2 truth on
its own at 4 threads, MPS trainings one at a time. Priority: alpha = 0.1 small-box (Paper I decision) first, then
the extra ICs (referee M3: >= 3 ICs per cell) for the small-box C and the window-trained p7b, then rollout statistics.
Log: artifacts/logs/z.log.
"""
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

PY = "/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python"
A, CR, L = Path("artifacts"), Path("artifacts/experiments/crop"), Path("artifacts/logs")
LOG = L / "z.log"
T01 = "r12_c6_a01_s1x_r3"


def env(n):
    return {**os.environ, **{k: str(n) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "VECLIB_MAXIMUM_THREADS")}}


def log(m):
    with LOG.open("a") as f:
        f.write(f"{datetime.now():%F %T} [Z] {m}\n")


def sh(name, cmd, out, threads=3):
    if out is not None and Path(out).exists():
        return
    log(f"start {name}")
    r = subprocess.run(cmd, env=env(threads), stdout=open(L / f"z_{name}.log", "w"), stderr=subprocess.STDOUT)
    tail = (L / f"z_{name}.log").read_text().strip().splitlines()[-1:] or [""]
    log(f"done {name} rc={r.returncode} {tail[0][:200]}")


def ckpts(tag):
    return ["--ae-ckpt", str(A / f"stage1_ae_kolmogorov_crop_{tag}.pt"),
            "--prop-ckpt", str(A / f"stage2_prop_kolmogorov_crop_{tag}.pt")]


def kev(tag, sc, s, truth=None):  # Kolmogorov D_KY eval
    t = f"{tag}_x{sc}_seed{s}"
    tj = ["--truth-json", str(truth)] if truth else []
    return (t, [PY, "scripts/exp_crop_eval.py", *ckpts(tag), "--scale", str(sc), "--seed", str(s), "--tag", t, *tj],
            CR / f"{t}.json")


def kev2d(tag, sc, s, truth=None):  # 2D KS D_KY eval
    t = f"{tag}_x{sc}_seed{s}"
    extra = (["--true-dky", str(truth)] if truth else []) + (["--n-directions", "512"] if sc == "2.0" else [])
    return (t, [PY, "scripts/exp_crop_eval_ks2d.py", *ckpts(tag), "--scale", sc, "--seed", str(s), "--tag", t, *extra],
            CR / f"{t}.json")


def train01(seed):
    tag = f"{T01}_seed{seed}"
    sh(f"train_{tag}", [PY, "scripts/train_kolmogorov_crop.py", "--tag", tag, "--seed", str(seed),
                        "--dataset", str(A / "datasets/kolmogorov_trajectories_alpha0.1.h5"), "--n-offsets", "16",
                        "--site-mix-radius", "0", "--n-site-mix-layers", "0", "--dec-site-mix-radius", "0",
                        "--dec-n-site-mix-layers", "0", "--k-max", "32", "--k-mid", "16", "--noise-end", "0.02",
                        "--prop-steps", "9000", "--crop-sites", "8", "--window", "3", "--prop-layers", "1",
                        "--local-channels", "6"], A / f"stage2_prop_kolmogorov_crop_{tag}.pt", threads=4)
    return tag


def main():
    log("start")
    # long truths, each on its own thread (not in the pool)
    side = [threading.Thread(target=sh, args=("truth_ks2d_x2_seed0", [PY, "scripts/run_truth_ks2d.py", "2.0", "0", "400"],
                                              Path("artifacts/experiments/truth_ks2d/truth_ks2d_x2_seed0.json"), 4)),
            threading.Thread(target=sh, args=("truth_ks2d_x1p5_seed2", [PY, "scripts/run_truth_ks2d.py", "1.5", "2", "300"],
                                              Path("artifacts/experiments/truth_ks2d/truth_ks2d_x1p5_seed2.json"), 3))]
    for t in side:
        t.start()
    # alpha = 0.1: 1x truths for ICs 1-2 (T120, as the 2x/3x reruns), then everything else
    jobs = []
    for s in (1, 2):
        jobs.append((f"truth_a0p1_x1_seed{s}", [PY, "scripts/run_truth_kolmogorov.py", "0.1", "1", str(s), "40", "100",
                                                 "120", "20"], CR / f"truth_a0p1_x1_seed{s}_T120.json"))
    jobs += [kev(T01, 2, 0), kev(T01, 2, 2, CR / "truth_a0p1_x2_seed2_T120.json"),
             kev(T01, 3, 0), kev(T01, 3, 2, CR / "truth_a0p1_x3_seed2_T120.json")]
    # extra ICs for the 2D KS small-box C (seed 0) and the window-trained p7b (truth 44^2 IC 2: 83.60)
    for tag in ("r4c_ks2d_s44_r3", "p7b_ks2d"):
        jobs += [kev2d(tag, "1.0", 2, 83.60), kev2d(tag, "1.5", 2), kev2d(tag, "2.0", 2)]
    jobs.append(("rollout_ks2d_r4c", [PY, "scripts/exp_rollout_stats_ks2d.py", "--tag", "r4c_ks2d_s44_r3",
                                      "--seeds", "0"], None))
    with ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(sh, *j) for j in jobs]
        # MPS: two more alpha = 0.1 training seeds, then their evals (2x IC 1; 1x IC 0)
        for seed in (1, 2):
            tag = train01(seed)
            futs += [ex.submit(sh, *kev(tag, 2, 1, CR / "truth_a0p1_x2_seed1_T120.json")), ex.submit(sh, *kev(tag, 1, 0))]
        [f.result() for f in futs]
        # 1x ICs 1-2 for the seed-0 model, once their truths exist
        futs = [ex.submit(sh, *kev(T01, 1, s, CR / f"truth_a0p1_x1_seed{s}_T120.json")) for s in (1, 2)]
        [f.result() for f in futs]
    for t in side:
        t.join()
    log("Z DONE")


if __name__ == "__main__":
    main()
