#!/usr/bin/env python
"""Compare how PREDICTABLE two AEs' latent spaces are, independent of any
propagator's training (user-directed 2026-08-31, following the discovery
that an `mlp`/markovian propagator fits `history3_fullprop_wvar005_tw16_
wspatial005`'s latent space far more easily -- reliably across 3+ random
seeds -- than the AE fine-tuned from it, despite the fine-tuned AE having
BETTER structural diagnostics (D4/D6/D7)): "look at the latent space and
try to figure out what makes that run more predictable than the other."

Every measure here is TRAINING-FREE (no propagator fitting at all), so
none of them can be explained by "got a lucky/unlucky random init" the way
the propagator-fitting results could be -- differences found here are
properties of the latent space itself.

Reports four things per AE, encoding the SAME real trajectories with each:
  1. Latent increment scale: ||z(t+1)-z(t)|| relative to the attractor's
     own spread -- how big is one physical time step's jump, in units of
     how spread-out the attractor is. A bigger relative jump is a harder
     target for ANY smooth function to hit exactly.
  2. Best-possible LINEAR one-step map residual (closed-form ridge
     regression z(t+1) ~= A z(t) + b, train/test split) -- a completely
     training-free lower bound on one-step predictability. If even the
     BEST LINEAR map already does much better on one AE than the other,
     that is clean evidence the underlying one-step dynamics is smoother/
     simpler there, regardless of any specific nonlinear propagator's
     capacity or training.
  3. KNN local stretching factor: for many pairs of nearby z(t) states
     (true near-neighbours on the encoded attractor), how much farther
     apart do their z(t+1) images get? A local Lipschitz-like estimate of
     the one-step map -- higher means small differences in the current
     state blow up faster one step later, making the map inherently
     harder to fit smoothly with finite data/capacity.
  4. Decoder Jacobian norm at sampled attractor points: `||d decode(z)/dz||`
     (operator norm via power iteration). A larger norm means small latent
     errors turn into large physical-space errors -- decode() is more
     "sensitive," which can make apparent one-step unpredictability look
     worse than the true underlying physical sensitivity, purely as an
     artifact of the decoder's own conditioning.

    python scripts/analyze_latent_predictability.py --ae-a <path> --ae-b <path>
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ks_latent.models import load_autoencoder_checkpoint


def latent_increment_stats(z_traj: np.ndarray) -> dict:
    """`z_traj`: `(n_runs, T, d)`. Ratio of one-step latent jump size to the
    attractor's own overall spread (std across all points)."""
    diffs = z_traj[:, 1:] - z_traj[:, :-1]
    step_norms = np.linalg.norm(diffs, axis=-1).ravel()
    spread = float(z_traj.reshape(-1, z_traj.shape[-1]).std(axis=0).mean())
    return {
        "mean_step_norm": float(step_norms.mean()),
        "attractor_spread": spread,
        "relative_step_size": float(step_norms.mean() / (spread + 1e-12)),
    }


def covariance_spectrum_stats(z_traj: np.ndarray) -> dict:
    """Eigenspectrum of the latent covariance across all encoded states.
    Ridge regression and KNN distances are roughly scale/conditioning-
    invariant, but SGD-trained MLPs are NOT: a badly-conditioned or
    near-collapsed channel (tiny eigenvalue) can make gradient-based
    fitting far harder even when it doesn't show up in the other,
    scale-robust diagnostics above."""
    z_flat = z_traj.reshape(-1, z_traj.shape[-1])
    cov = np.cov(z_flat, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    participation_ratio = float(eigvals.sum() ** 2 / (eigvals ** 2).sum())
    return {
        "top_eigval": float(eigvals[0]),
        "min_eigval": float(eigvals[-1]),
        "condition_number": float(eigvals[0] / (eigvals[-1] + 1e-12)),
        "effective_rank": participation_ratio,
        "d_latent": len(eigvals),
    }


def linear_map_residual(z_traj: np.ndarray, k: int = 1, ridge: float = 1e-3, test_frac: float = 0.2, seed: int = 0) -> dict:
    """Closed-form ridge regression z(t+k) ~= A z(t) + b (a SEPARATE fit
    per horizon k, not an iterated one-step map -- this gives the actual
    achievable-with-a-static-linear-map error at horizon k, which is the
    fair comparison since Stage-2's curriculum trains directly on k-step
    rollouts, not on composing a fitted one-step map). Relative MSE on a
    held-out test split vs. a "predict no change" baseline. Training-free:
    no propagator, no gradient descent."""
    z_curr = z_traj[:, :-k].reshape(-1, z_traj.shape[-1])
    z_next = z_traj[:, k:].reshape(-1, z_traj.shape[-1])
    n = z_curr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    X = np.concatenate([z_curr[train_idx], np.ones((len(train_idx), 1))], axis=1)
    Y = z_next[train_idx]
    d = X.shape[1]
    W = np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ Y)  # (d+1, d_latent)

    X_test = np.concatenate([z_curr[test_idx], np.ones((len(test_idx), 1))], axis=1)
    pred = X_test @ W
    resid = pred - z_next[test_idx]
    baseline = z_next[test_idx] - z_curr[test_idx]  # "predict no change" baseline
    rel_mse = float((resid**2).sum() / (baseline**2).sum())
    return {"linear_relative_mse": rel_mse}


def knn_stretching_factor(z_traj: np.ndarray, n_pairs: int = 20000, k: int = 5, seed: int = 0) -> dict:
    """For many sampled z(t), find its k nearest OTHER same-time-index-
    pool neighbours and compare ||z(t+1)_i - z(t+1)_j|| / ||z(t)_i -
    z(t)_j|| -- median ratio across many close pairs. >1 means the map
    locally stretches distances (harder to fit smoothly); ~1 or below
    means it's locally well-behaved."""
    z_curr = z_traj[:, :-1].reshape(-1, z_traj.shape[-1])
    z_next = z_traj[:, 1:].reshape(-1, z_traj.shape[-1])
    n = z_curr.shape[0]
    rng = np.random.default_rng(seed)
    sample_size = min(4000, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    zc, zn = z_curr[sample_idx], z_next[sample_idx]

    # Pairwise distances within the sample (sample_size is small enough for a dense matrix).
    d2 = ((zc[:, None, :] - zc[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :k]

    ratios = []
    rng2 = np.random.default_rng(seed + 1)
    rows = rng2.choice(sample_size, size=min(n_pairs, sample_size), replace=True)
    for i in rows:
        j = nn_idx[i, rng2.integers(0, k)]
        dist_curr = np.linalg.norm(zc[i] - zc[j])
        dist_next = np.linalg.norm(zn[i] - zn[j])
        if dist_curr > 1e-8:
            ratios.append(dist_next / dist_curr)
    ratios = np.array(ratios)
    return {
        "knn_stretch_median": float(np.median(ratios)),
        "knn_stretch_p90": float(np.percentile(ratios, 90)),
    }


def decoder_jacobian_norm_jvp(ae, z_samples: torch.Tensor, n_power_iters: int = 15) -> float:
    """Mean spectral-norm estimate of d(decode)/dz over `z_samples`, via
    power iteration using forward-mode JVPs (`torch.autograd.functional.jvp`)
    composed with a VJP step -- standard power-iteration-on-J^T J."""
    from torch.autograd.functional import jvp, vjp

    norms = []
    for z0 in z_samples:
        z0 = z0.detach()
        v = torch.randn_like(z0)
        v = v / v.norm()
        # ViT decoders use scaled_dot_product_attention, whose CPU backward
        # isn't implemented for the flash/mem-efficient kernels double-
        # backward (JVP/VJP composition) needs -- force the math backend
        # (same fix already used in ks_latent/analysis/lyapunov.py).
        func = lambda z: ae.decode(z.unsqueeze(0)).squeeze(0)
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(n_power_iters):
                _, Jv = jvp(func, (z0,), (v,))
                _, JTJv = vjp(func, z0, v=Jv)
                norm = JTJv.norm()
                if norm < 1e-12:
                    break
                v = JTJv / norm
            _, Jv = jvp(func, (z0,), (v,))
        norms.append(float(Jv.norm().item()))
    return float(np.mean(norms))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-a", required=True, help="e.g. the 'predictable' AE")
    parser.add_argument("--ae-b", required=True, help="e.g. the 'harder' AE")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--n-jacobian-samples", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][: args.n_runs], dtype=torch.float32)
    n, T, NX = traj.shape

    for label, path in [(args.label_a, args.ae_a), (args.label_b, args.ae_b)]:
        ae, ae_cfg, _ = load_autoencoder_checkpoint(path)
        ae.eval()
        with torch.no_grad():
            z_traj = ae.encode(traj.reshape(n * T, NX)).reshape(n, T, -1)
        z_np = z_traj.numpy()

        inc = latent_increment_stats(z_np)
        cov = covariance_spectrum_stats(z_np)
        knn = knn_stretching_factor(z_np)
        z_flat = z_traj.reshape(-1, z_traj.shape[-1])
        sample_idx = torch.randperm(z_flat.shape[0])[: args.n_jacobian_samples]
        jac_norm = decoder_jacobian_norm_jvp(ae, z_flat[sample_idx])

        print(f"\n=== {label}: {path} ===")
        print(f"  relative one-step jump size (||dz||/attractor spread): {inc['relative_step_size']:.4f}")
        print(f"  KNN local stretch factor (median / p90):                {knn['knn_stretch_median']:.4f} / {knn['knn_stretch_p90']:.4f}")
        print(f"  latent covariance: top_eig={cov['top_eigval']:.4g} min_eig={cov['min_eigval']:.4g} "
              f"cond#={cov['condition_number']:.4g} eff_rank={cov['effective_rank']:.2f}/{cov['d_latent']}")
        print(f"  decoder Jacobian operator norm (mean over samples):     {jac_norm:.4f}")
        print(f"  best-linear-map relative MSE vs. horizon k (no-change baseline):")
        for k in (1, 2, 4, 8, 16):
            if k >= z_np.shape[1]:
                continue
            lin = linear_map_residual(z_np, k=k)
            print(f"    k={k:>2}: {lin['linear_relative_mse']:.4f}")


if __name__ == "__main__":
    main()
