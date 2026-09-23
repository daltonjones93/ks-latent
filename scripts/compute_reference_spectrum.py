#!/usr/bin/env python
"""Computes and saves a per-rank reference singular-value spectrum from a
trusted, already-validated AE+propagator checkpoint pair (added 2026-09-10,
Section 134, user-directed: "build the graded version" of
`propagator_spectrum_shape_loss` -- see that function's own docstring, and
`propagator_graded_spectrum_shape_loss`'s, for the full motivation).

Samples `--n-samples` real (encoded-trajectory) points, computes the FULL
singular-value spectrum of the propagator's own step Jacobian at each
(`ks_latent.analysis.diagnostics.propagator_step_jacobian_full_spectrum`),
and saves the per-rank MEDIAN across samples -- a robust empirical target,
not a single noisy point -- to a `.npy` file of shape `(d_latent,)`,
descending. This is the file `--spectrum-shape-graded-reference-path`
(train_stage1_patched.py/train_stage2_patched.py) expects.

    python scripts/compute_reference_spectrum.py \\
        --ae-checkpoint artifacts/stage1_ae_patched_full_section85_....pt \\
        --prop-checkpoint artifacts/stage2_prop_patched_full_section85_....pt \\
        --out artifacts/reference_spectra/section85_L100_d44.npy

Stated plainly (see `propagator_graded_spectrum_shape_loss`'s docstring):
this is a transfer from one trusted model's own measured spectrum to
another, not literal ground truth extracted from the raw PDE -- the
d-dim latent embedding is model-dependent, so there is no model-
independent "true" target spectrum expressible in these coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.analysis.diagnostics import propagator_step_jacobian_full_spectrum
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--n-runs", type=int, default=20, help="Trajectories to encode for sampling.")
    parser.add_argument("--n-samples", type=int, default=200, help="Real (state, step) points to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output .npy path, shape (d_latent,), descending.")
    args = parser.parse_args()

    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    prop, prop_cfg, _ = load_propagator_checkpoint(args.prop_checkpoint)
    ae.eval()
    prop.eval()

    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][: args.n_runs], dtype=torch.float32)
    n, T, NX = traj.shape
    with torch.no_grad():
        z_all = ae.encode(traj.reshape(n * T, NX)).reshape(n, T, -1)

    spectra = propagator_step_jacobian_full_spectrum(
        prop, z_all, n_samples=args.n_samples, seed=args.seed
    )  # (n_samples, d)
    reference = np.median(spectra, axis=0)  # (d,), descending (each column already sorted per-sample)
    assert np.all(np.diff(reference) <= 1e-9), "reference spectrum should be descending"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, reference.astype(np.float32))

    print(f"wrote {out_path} (shape {reference.shape})")
    print("full per-rank median reference spectrum:")
    for i, v in enumerate(reference):
        print(f"  [{i:2d}] {v:.5f}")


if __name__ == "__main__":
    main()
