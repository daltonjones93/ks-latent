#!/usr/bin/env python
"""Honest MPS-vs-CPU timing for one training epoch per model (brief §1.3.8).

"If MPS is not winning for a given model -- it often is not, for small
models dominated by launch overhead -- say so and default that model to
CPU. Do not assume MPS is faster."
"""

from __future__ import annotations

import time

import torch

from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.seeding import set_seed


def _synthetic_trajectories(n_runs: int, T: int, NX: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(NX, dtype=torch.float32) * 2 * torch.pi / NX
    t = torch.arange(T, dtype=torch.float32)
    traj = torch.empty(n_runs, T, NX)
    for r in range(n_runs):
        amp = torch.rand(1, generator=g).item() * 0.5 + 0.5
        phase0 = torch.rand(1, generator=g).item() * 2 * torch.pi
        omega = 0.05 + 0.02 * torch.rand(1, generator=g).item()
        traj[r] = amp * torch.cos(x.unsqueeze(0) - (phase0 + omega * t).unsqueeze(1))
    return traj


def bench_stage1(device: torch.device, epochs: int = 1) -> float:
    set_seed(0)
    ae_cfg = AutoencoderConfig()  # canonical NX=256 (changed 2026-08-29, see docs/RESULTS.md), d_latent=44
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent)
    train_cfg = Stage1TrainingConfig(epochs=epochs)
    train_traj = _synthetic_trajectories(50, 200, ae_cfg.NX, seed=1)
    val_traj = _synthetic_trajectories(4, 200, ae_cfg.NX, seed=2)
    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    t0 = time.time()
    train_stage1(ae, aux, train_traj, val_traj, train_cfg, device)
    if device.type == "mps":
        torch.mps.synchronize()
    return time.time() - t0


def bench_stage2(device: torch.device, epochs: int = 1) -> float:
    set_seed(0)
    d_latent = 44
    train_seq = torch.randn(200, 60, d_latent)
    val_seq = torch.randn(20, 60, d_latent)
    prop_cfg = PropagatorConfig(d_latent=d_latent)
    train_cfg = Stage2TrainingConfig(epochs=epochs)
    prop = LatentPropagator(prop_cfg)
    t0 = time.time()
    train_stage2(prop, train_seq, val_seq, train_cfg, device)
    if device.type == "mps":
        torch.mps.synchronize()
    return time.time() - t0


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    print(f"{'model':<12}{'device':<8}{'seconds/epoch':<15}")
    for name, fn in [("stage1_ae", bench_stage1), ("stage2_prop", bench_stage2)]:
        for device in devices:
            t = fn(device, epochs=1)
            print(f"{name:<12}{device.type:<8}{t:<15.3f}")


if __name__ == "__main__":
    main()
