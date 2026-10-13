"""Multi-field (omega, u, v) inputs for the Kolmogorov local-field autoencoder (R15 option 1, 2026-10-13).

Motivation (docs/split_paper_tasks.md Z9 and the 2026-10-13 discussion): in vorticity form the advecting velocity
at a point depends on vorticity everywhere (Biot-Savart), so a local model that reads only vorticity must infer the
velocity from its neighbourhood. Giving the encoder the velocity as well makes the advecting velocity local state
carried by the latent; the propagator stays local and is NOT given a global solve at each step (unlike the
`local_mlp_2d_biotsavart` backbone). The velocity is computed exactly, once, from the whole-domain vorticity when a
field is encoded (training data at the 1x box; ICs at 1x/2x/3x at evaluation).

Packing convention (`LocalFieldAutoencoderConfig2D.n_fields`): fields concatenated along the last (y) axis, so a
field of shape `(..., NX, NY)` becomes `(..., NX, n_fields*NY)` and the flat vector is `reshape(B, NX, nf, NY)`.
Normalization: omega keeps the dataset's (mean, std) standardization; u and v are each divided by their own
training-split std (`velocity_std`, stored in the checkpoint).
"""
from __future__ import annotations

import numpy as np
import torch

from ks_latent.models.propagator import _biot_savart_velocity


def velocity_from_vorticity(w: torch.Tensor, Lx: float, Ly: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`w`: physical vorticity `(..., NX, NY)` on a periodic `Lx x Ly` box -> `(u, v)`, each `(..., NX, NY)`,
    with `u = psi_y`, `v = -psi_x`, `lap psi = -w` (the solver's convention; k = 0 mode zero)."""
    lead, (NX, NY) = w.shape[:-2], w.shape[-2:]
    vel = _biot_savart_velocity(w.reshape(-1, NX, NY), Lx, Ly)
    return vel[..., 0].reshape(*lead, NX, NY), vel[..., 1].reshape(*lead, NX, NY)


def pack_wuv(w_norm: torch.Tensor, mean: float, std: float, Lx: float, Ly: float,
             velocity_std: tuple[float, float]) -> torch.Tensor:
    """`w_norm`: standardized vorticity `(..., NX, NY)` -> packed `(..., NX, 3*NY)` = (w_norm, u/su, v/sv)."""
    u, v = velocity_from_vorticity(w_norm * std + mean, Lx, Ly)
    su, sv = velocity_std
    return torch.cat([w_norm, u / su, v / sv], dim=-1)


@torch.no_grad()
def pack_wuv_chunked(traj: torch.Tensor, mean: float, std: float, Lx: float, Ly: float,
                     velocity_std: tuple[float, float] | None = None, n_std: int | None = None, chunk: int = 256):
    """`traj`: standardized vorticity `(N, T, NX, NY)` (CPU). Returns `(packed (N, T, NX, 3*NY) float32,
    velocity_std)`, filled in place chunk by chunk (memory: one output array). With `velocity_std=None` the stds
    are computed from the first `n_std` trajectories (pass n_train: the training split only)."""
    N, T, NX, NY = traj.shape
    out = torch.empty(N, T, NX, 3 * NY, dtype=torch.float32)
    of, tf = out.view(N * T, NX, 3 * NY), traj.reshape(N * T, NX, NY)
    for i in range(0, N * T, chunk):
        w = tf[i:i + chunk]
        u, v = velocity_from_vorticity(w.double() * std + mean, Lx, Ly)
        of[i:i + chunk, :, :NY] = w
        of[i:i + chunk, :, NY:2 * NY] = u.float()
        of[i:i + chunk, :, 2 * NY:] = v.float()
    if velocity_std is None:
        rows = out[: (N if n_std is None else n_std)]
        velocity_std = (float(rows[..., NY:2 * NY].double().std()), float(rows[..., 2 * NY:].double().std()))
    out[..., NY:2 * NY] /= velocity_std[0]
    out[..., 2 * NY:] /= velocity_std[1]
    return out, velocity_std


def encode_input(ae, w_norm: np.ndarray, ckpt: dict, Lx: float, Ly: float) -> torch.Tensor:
    """Flat encoder input `(1, NX*nf*NY)` float32 for one standardized vorticity IC `(NX, NY)`, for any
    checkpoint: single-field models get the field unchanged (the pre-R15 path, bit for bit); `wuv` models get
    the packed fields with the velocity computed on THIS domain (`Lx x Ly`)."""
    nf = getattr(ae.cfg, "n_fields", 1)
    w = torch.tensor(w_norm, dtype=torch.float64)
    if nf == 1:
        return torch.tensor(w_norm.reshape(1, -1), dtype=torch.float32)
    assert nf == 3 and ckpt.get("input_fields") == "wuv", f"unsupported n_fields={nf} / {ckpt.get('input_fields')}"
    packed = pack_wuv(w, ckpt["normalization_mean"], ckpt["normalization_std"], Lx, Ly, tuple(ckpt["velocity_std"]))
    return packed.reshape(1, -1).float()


def field0(decoded_flat: torch.Tensor, NX: int, NY: int, n_fields: int) -> torch.Tensor:
    """Decoded flat `(B, NX*nf*NY)` -> field 0 (vorticity, standardized) `(B, NX, NY)`."""
    return decoded_flat.reshape(-1, NX, n_fields, NY)[:, :, 0]
