"""C7 (docs/ks_local_field_ltransfer.tex Sec. c7): learned consensus,
crop-trained, rank-protected, at alpha=0.05. Arm (a) is P2 itself. Arms:
  (b) c7b: P2's autoencoder + local_mlp_2d_consensus propagator
  (c) c7c: same + one-sided per-site log-det rank penalty (w_rank=0.02)
  (d) c7d: plain local_mlp_2d with c=12 channels (matched latent size), own AE
(b)/(c) use 64-site crops so the free interior stays 24 sites with the
consensus block's larger per-step reach (halo 14). Waits for the crop
pipeline and the queued 2D-KS seed runs; one task at a time; preemptable.
Usage: python scripts/exp_c7_pipeline.py [--summarize]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "crop"
LOGS = A / "logs"
DATA = A / "datasets/kolmogorov_wide16_alpha0.05.h5"
P2_AE = A / "stage1_ae_kolmogorov_crop_p2_a005.pt"
TRUTH3 = OUT / "truth_a005_x3_seed0.json"
WAIT_FOR = ("exp_crop_pipeline.py", "run_ks2d_seeds_after_pipeline.sh", "run_icnorm_ks2d_seeds.sh")
ARMS = {
    "c7b_a005": {"reuse_p2_ae": True, "args": ["--backbone", "local_mlp_2d_consensus", "--crop-sites", "64"]},
    "c7c_a005": {"reuse_p2_ae": True, "args": ["--backbone", "local_mlp_2d_consensus", "--crop-sites", "64",
                                               "--w-rank", "0.02"]},
    "c7d_a005": {"reuse_p2_ae": False, "args": ["--local-channels", "12"]},
}
SCALES, SEEDS = (1, 2, 3), (0, 1, 2)


def log(msg):
    print(f"[c7 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sh(cmd, logname):
    with open(LOGS / logname, "a") as fh:
        subprocess.run([sys.executable] + cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def ckpts(tag):
    return A / f"stage1_ae_kolmogorov_crop_{tag}.pt", A / f"stage2_prop_kolmogorov_crop_{tag}.pt"


def eval_path(tag, s, seed):
    return OUT / f"{tag}_x{s}_seed{seed}.json"


def train(tag, spec):
    ae, _ = ckpts(tag)
    if spec["reuse_p2_ae"] and not ae.exists():
        shutil.copy(P2_AE, ae)
    sh(["scripts/train_kolmogorov_crop.py", "--dataset", str(DATA), "--tag", tag] + spec["args"],
       f"crop_train_{tag}.log")


def tasks():
    t = []
    for tag, spec in ARMS.items():
        t.append((f"{tag} train", lambda tag=tag: ckpts(tag)[1].exists(), lambda tag=tag, spec=spec: train(tag, spec)))
        for s in SCALES:
            for seed in SEEDS:
                extra = ["--truth-json", str(TRUTH3)] if s == 3 else []
                t.append((f"{tag} eval x{s} seed{seed}", lambda tag=tag, s=s, seed=seed: eval_path(tag, s, seed).exists(),
                          lambda tag=tag, s=s, seed=seed, extra=extra: sh(
                              ["scripts/exp_crop_eval.py", "--ae-ckpt", str(ckpts(tag)[0]), "--prop-ckpt",
                               str(ckpts(tag)[1]), "--scale", str(s), "--seed", str(seed),
                               "--tag", f"{tag}_x{s}_seed{seed}"] + extra, f"crop_eval_{tag}.log")))
    return t


def summarize():
    out = {}
    truth3 = json.loads(TRUTH3.read_text())["D_KY"] if TRUTH3.exists() else None
    for tag in ["p2_a005"] + list(ARMS):
        rows = {}
        for s in SCALES:
            vals = [json.loads(eval_path(tag, s, sd).read_text()) for sd in SEEDS if eval_path(tag, s, sd).exists()]
            if not vals:
                continue
            truth = truth3 if s == 3 else vals[0]["true_D_KY"]
            dky = [v["model_D_KY"] for v in vals]
            errs = [abs(d - truth) / truth * 100 for d in dky] if truth else None
            rows[f"x{s}"] = {"D_KY": dky, "truth": truth, "err_pct": errs,
                             "err_pct_mean": sum(errs) / len(errs) if errs else None}
        out[tag] = rows
    (OUT / "c7_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    if ap.parse_args().summarize:
        summarize()
        return
    while any(subprocess.run(["pgrep", "-f", w], capture_output=True).returncode == 0 for w in WAIT_FOR):
        time.sleep(30)
    if not P2_AE.exists():
        log(f"P2 autoencoder missing ({P2_AE}); cannot run arms (b)/(c). Aborting.")
        return
    log("prerequisites finished; starting C7")
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
