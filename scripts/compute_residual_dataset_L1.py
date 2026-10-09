"""R2/R3/R4 (`docs/ks_local_field_ltransfer.tex`, "A frozen, scale-incremental
hierarchy" -- revised plan, Level 0 = Section 230's existing checkpoint,
Level 1 = trained at G1's own transfer size): compute Level 0's own
zero-retraining rollout at L1 (Lx=Ly=66) for every trajectory in the L1
dataset, and save the RESIDUAL (true - Level-0-predicted) as a new
trajectory dataset in the same HDF5 format -- so Level 1 can be trained
with the EXACT SAME generic Stage1/Stage2 training scripts unchanged,
just pointed at residual data instead of raw KS data.

Simplification relative to the architecture note in
`docs/ks_local_field_ltransfer.tex` Section~\\ref{sec:residual_architecture}:
this first pass trains Level 1 as an INDEPENDENT autoregressive model on
the residual sequence (its own encode/propagate/decode, not conditioned
on Level 0's own latent state) rather than a conditioned propagator --
reuses 100% existing, already-tested code (zero new architecture
classes), at the cost of not giving Level 1 explicit access to what
Level 0 "sees." Flagged here as a real simplification, not silently
adopted as the final design.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d

AE_CKPT = "artifacts/stage1_ae_2dks_full_section230_ks2d_localfield_localmlp2d_window3.pt"
PROP_CKPT = "artifacts/stage2_prop_2dks_full_section230_ks2d_localfield_localmlp2d_window3_warmstart_k12_300ep.pt"
L1_DATASET = "artifacts/datasets/ks2d_residual_L1_transfer66.h5"
OUT_PATH = "artifacts/datasets/ks2d_residual_L1_level1_target.h5"

device = "cpu"

ae0, ae0_cfg, _ = load_autoencoder_checkpoint_resized_2d(
    AE_CKPT, new_n_sites_x=12, new_n_sites_y=12, new_NX=192, new_NY=192,
)
prop0, prop0_cfg, _ = load_propagator_checkpoint_resized_2d(PROP_CKPT, new_n_sites_x=12, new_n_sites_y=12)
ae0.eval(); prop0.eval()
for p in list(ae0.parameters()) + list(prop0.parameters()):
    p.requires_grad_(False)

with h5py.File(L1_DATASET, "r") as f:
    n_train = int(f["metadata"].attrs["n_train"])
    n_val = int(f["metadata"].attrs["n_val"])
    NX = int(f["metadata"].attrs["NX"])
    NY = int(f["metadata"].attrs["NY"])
    traj_norm = f["trajectories"][:]  # (n_runs, T, NX*NY), normalized
    mean = float(f["metadata"].attrs["normalization_mean"])
    std = float(f["metadata"].attrs["normalization_std"])

n_runs, T, D = traj_norm.shape
print(f"L1 dataset: n_runs={n_runs} T={T} NX=NY={NX} mean={mean:.4g} std={std:.4g}")

traj_raw = traj_norm * std + mean  # physical units, matches what Level 0 was trained on

residual_raw = np.empty_like(traj_raw)
with torch.no_grad():
    for i in range(n_runs):
        u0 = torch.tensor(traj_raw[i, 0], dtype=torch.float32).unsqueeze(0)  # (1, NX*NY)
        z0 = ae0.encode(u0)
        roll = prop0.rollout(z0, z0, T - 1)[0]  # (T-1, d_latent)
        z_all = torch.cat([z0, roll], dim=0)  # (T, d_latent)
        u0_pred = ae0.decode(z_all).numpy()  # (T, NX*NY)
        residual_raw[i] = traj_raw[i] - u0_pred
        if i % 5 == 0:
            print(f"  run {i}: max|residual|={np.abs(residual_raw[i]).max():.3f} "
                  f"max|true|={np.abs(traj_raw[i]).max():.3f}", flush=True)

train_res = residual_raw[:n_train]
res_mean = train_res.mean()
res_std = train_res.std()
residual_norm = (residual_raw - res_mean) / res_std
print(f"residual stats (train split): mean={res_mean:.4g} std={res_std:.4g}")

with h5py.File(OUT_PATH, "w") as f:
    f.create_dataset("trajectories", data=residual_norm, compression="gzip")
    f.create_dataset("trajectories_raw", data=residual_raw, compression="gzip")
    grp = f.create_group("metadata")
    grp.attrs["NX"] = NX
    grp.attrs["NY"] = NY
    grp.attrs["n_train"] = n_train
    grp.attrs["n_val"] = n_val
    grp.attrs["normalization_mean"] = res_mean
    grp.attrs["normalization_std"] = res_std
    grp.attrs["source"] = "residual = true(L1=66) - Level0(Section230, zero-shot resized)"

print(f"wrote {OUT_PATH}")
