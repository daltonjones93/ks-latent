"""P7 step 2 data: one wide square 2D KS domain, Lx = Ly = scale*44 at the native
grid spacing (NX = NY = 128*scale), same dt / dt_snap / spinup / trajectory length as
ks2d_trajectories_default.h5. Preemptable: one .npy per finished trajectory.
Stores the normalized float32 field (training-split stats) and metadata."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, time
from pathlib import Path
import h5py
import numpy as np
from ks_latent.config import KSConfig2D
from ks_latent.solver.dataset2d import _write_metadata_group_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d

p = argparse.ArgumentParser()
p.add_argument("--scale", type=int, default=6)
p.add_argument("--n-train", type=int, default=4)
p.add_argument("--n-val", type=int, default=1)
p.add_argument("--trajectory-time", type=float, default=25.0)
p.add_argument("--seed-base", type=int, default=900)
p.add_argument("--cubic", type=float, default=0.0, help="J17A: missing-process strength beta (u_t = KS(u) - beta P[u'^3])")
p.add_argument("--ganiso", type=float, default=0.0, help="J17G: truth u_t = KS(u) - (delta/2)(u_x^2 - u_y^2)")
p.add_argument("--memory", default=None,
               help="J17B: 'eps,tau,kappa[,coupling]' hidden-memory truth; only u is stored for training, v is saved "
                    "separately (traj<i>_v.npy) for diagnostics")
p.add_argument("--only", type=int, default=None, help="generate just this trajectory index and exit (parallel runs)")
p.add_argument("--out", required=True)
a = p.parse_args()
out = Path(a.out); tmp = out.parent / (out.stem + "_parts"); tmp.mkdir(parents=True, exist_ok=True)
cfg = KSConfig2D(Lx=44.0 * a.scale, Ly=44.0 * a.scale, NX=128 * a.scale, NY=128 * a.scale, seed=a.seed_base)
n_runs, n_steps = a.n_train + a.n_val, int(round(a.trajectory_time / cfg.dt))
for i in (range(n_runs) if a.only is None else [a.only]):
    part = tmp / f"traj{i}.npy"
    if part.exists():
        print(f"[ks2d-wide] skip traj {i}", flush=True); continue
    t0 = time.time()
    if a.memory:
        from ks_latent.solver.ks2d_memory import MemoryParams, integrate_memory, spinup_memory
        f_ = a.memory.split(",")
        mp = MemoryParams(float(f_[0]), float(f_[1]), float(f_[2]), f_[3] if len(f_) > 3 else "lap")
        us, vs = spinup_memory(cfg, np.random.default_rng(a.seed_base + i), mp)
        traj, vtraj = integrate_memory(us, vs, cfg, n_steps, mp)
        traj = traj.astype(np.float32)
        np.save(tmp / f"traj{i}_v.npy", vtraj[::10].astype(np.float32))  # every 10th snapshot: diagnostics only
    else:
        traj = integrate2d(spinup2d(cfg, np.random.default_rng(a.seed_base + i), cubic=a.cubic, ganiso=a.ganiso), cfg, n_steps=n_steps, cubic=a.cubic, ganiso=a.ganiso).astype(np.float32)
    np.save(part.with_suffix(".tmp.npy"), traj); os.replace(part.with_suffix(".tmp.npy"), part)
    print(f"[ks2d-wide] traj {i}/{n_runs} done in {time.time() - t0:.0f}s, shape {traj.shape}", flush=True)
if a.only is not None:
    raise SystemExit(0)
trajs = np.stack([np.load(tmp / f"traj{i}.npy") for i in range(n_runs)])
mean, std = float(trajs[: a.n_train].astype(np.float64).mean()), float(trajs[: a.n_train].astype(np.float64).std())
with h5py.File(out, "w") as f:
    f.create_dataset("trajectories", data=((trajs - mean) / std).astype(np.float32).reshape(n_runs, trajs.shape[1], -1),
                     compression="gzip")
    _write_metadata_group_2d(f, cfg, {"n_train": a.n_train, "n_val": a.n_val, "trajectory_time": a.trajectory_time,
                                      "normalization_mean": mean, "normalization_std": std, "scale": a.scale,
                                      "seed_base": a.seed_base,
                                      "system": ("ks2d" if a.cubic == 0.0 else f"ks2d_cubic{a.cubic:g}") + (f"_gani{a.ganiso:g}" if a.ganiso else "") + (f"_mem{a.memory}" if a.memory else ""),
                                      "cubic": a.cubic, "ganiso": a.ganiso, "memory": a.memory or ""})
print(f"[ks2d-wide] wrote {out} mean={mean:.4g} std={std:.4g}", flush=True)
