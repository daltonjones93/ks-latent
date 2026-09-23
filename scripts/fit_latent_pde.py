#!/usr/bin/env python
"""Attempt to fit an interpretable, translation-equivariant local PDE/ODE
to a trained checkpoint's REAL latent dynamics, and test the fitted
equation's own rollout/Lyapunov fidelity against real data and the
trained propagator. User-directed 2026-09-06: "can you write code to do
this, we can test it on 98" -- following a discussion of whether the
smoothest/most D8-D9-coherent checkpoints (97, 98) are ready for literal
PDE-modeling. This is the one thing flagged repeatedly in this project's
own open questions (docs/model-research-summary-9-5-26.md item 5,
"Latent-smoothness metric still undefined... no direct smoothness metric
... has been computed yet. If the user wants to move toward literally
fitting a latent PDE, this is likely the next methodological gap to
fill") that had never actually been attempted before this script.

Method: SINDy-style sparse regression (Brunton, Proctor & Kutz 2016,
"Discovering governing equations from data") over a library of candidate
terms mirroring the TRUE KS PDE's own structure (u_t + u*u_x + u_xx +
u_xxxx = 0), applied PER LATENT INDEX under the hypothesis -- supported
by this project's own D4 (approximate translation equivariance) and D7/8
(genuine ring/periodic structure, Section 66) findings -- that the
latent index behaves like a periodic spatial coordinate the same way
KS's own physical grid does:

    1                       (constant)
    z_i                     (zeroth order)
    z_i^2                   (quadratic)
    (z_{i+1} - z_{i-1})/2   ~ u_x   (first central difference)
    z_i * (that)            ~ u*u_x (advection-like nonlinearity)
    z_{i+1} - 2 z_i + z_{i-1}           ~ u_xx (second difference)
    z_{i+2}-4z_{i+1}+6z_i-4z_{i-1}+z_{i-2}  ~ u_xxxx (fourth difference)
    z_{i+1} * z_{i-1}       (extra cross term, generality)

All terms use CIRCULAR shifts along the d_latent axis (the latent index
is treated as a ring, matching every other diagnostic in this codebase
that assumes this -- D3/D6/D7/D8's bandedness, spatial_coherence_loss).

A SINGLE shared coefficient vector is fit across every latent index at
once (pooling every (real time step, latent index) pair from REAL
encoded trajectories into one regression) -- this explicitly enforces
translation equivariance as a modeling CHOICE, rather than hoping the
fit discovers it. CAVEAT this script does not resolve: the latent
covariance's own eigenspectrum is typically far from flat across index
(this project's own conditioning-number diagnostics), so different
indices may not be equally well-explained by one shared law even where
D4 holds approximately for the ENCODER map itself -- the per-run R^2
reported below is the direct empirical check of how much this matters in
practice, not an assumption.

Fits against REAL data (encode(u_t) -> encode(u_{t+1}) transitions), NOT
against the trained propagator's own predictions -- a stricter test of
whether the underlying LATENT DYNAMICS itself (as revealed by real KS
trajectories) is well-described by this equation, independent of
whatever approximation the trained MLP propagator learned.

    python scripts/fit_latent_pde.py \\
        --ae-checkpoint artifacts/stage1_ae_patched_full_<TAG>.pt \\
        --prop-checkpoint artifacts/stage2_prop_patched_full_<TAG>_warmstart_k12_300ep.pt \\
        --tag <TAG>
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

FEATURE_NAMES = [
    "1", "z_i", "z_i^2", "dz1 (~u_x)", "z_i*dz1 (~u*u_x)",
    "dz2 (~u_xx)", "dz4 (~u_xxxx)", "z_{i+1}*z_{i-1}",
]


def build_library(Z: np.ndarray) -> np.ndarray:
    """`Z`: `(N, d)` real latent states. Returns `Theta`: `(N*d, 8)`, one
    row per (sample, latent index) pair -- see module docstring for the
    physical motivation of each column. All shifts are CIRCULAR along the
    `d` axis."""
    zm2 = np.roll(Z, 2, axis=1)
    zm1 = np.roll(Z, 1, axis=1)
    z0 = Z
    zp1 = np.roll(Z, -1, axis=1)
    zp2 = np.roll(Z, -2, axis=1)

    dz1 = (zp1 - zm1) / 2.0
    dz2 = zp1 - 2 * z0 + zm1
    dz4 = zp2 - 4 * zp1 + 6 * z0 - 4 * zm1 + zm2
    adv = z0 * dz1
    cross = zp1 * zm1

    cols = [np.ones_like(z0), z0, z0**2, dz1, adv, dz2, dz4, cross]
    stacked = np.stack(cols, axis=-1)  # (N, d, 8)
    return stacked.reshape(-1, stacked.shape[-1])


def stlsq(Theta: np.ndarray, y: np.ndarray, threshold: float = 0.05, n_iters: int = 15, ridge: float = 1e-8):
    """Sequential thresholded least squares -- the core SINDy algorithm
    (Brunton/Proctor/Kutz 2016). Each library column is normalized to
    unit std before thresholding, so `threshold` is one scale-free cutoff
    across physically very-different-magnitude terms (a raw z_i term vs.
    a fourth-difference term have wildly different natural scales);
    fitted coefficients are converted back to the RAW feature scale
    before returning. Returns `(xi_raw, active_mask)`."""
    col_scale = Theta.std(axis=0)
    col_scale[col_scale < 1e-12] = 1.0
    Theta_n = Theta / col_scale

    active = np.ones(Theta.shape[1], dtype=bool)
    xi_n = np.zeros(Theta.shape[1])
    for _ in range(n_iters):
        if active.sum() == 0:
            break
        A = Theta_n[:, active]
        xi_active, *_ = np.linalg.lstsq(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y, rcond=None)
        xi_n[:] = 0.0
        xi_n[active] = xi_active
        new_active = np.abs(xi_n) >= threshold
        if np.array_equal(new_active, active):
            break
        active = new_active
    xi_raw = xi_n / col_scale
    return xi_raw, active


class FittedLatentPDE:
    """Wraps a fitted sparse coefficient vector as a propagator-like
    object exposing the same `.step_one`/`.step`/`.rollout`/`.cfg.mode`
    interface as `LatentPropagator` in markovian mode -- lets this reuse
    the EXACT SAME Lyapunov machinery (`ks_latent.analysis.lyapunov`)
    every Gate 3 run in this project already uses, with zero new
    integration code. Integrates the fitted vector field via fixed-step
    RK4 over `ode_substeps` sub-steps per unit `dt_snap`, mirroring
    `ks_latent.models.propagator._NeuralODEDeltaBody`'s own convention
    (a single raw Euler step at h=1 is a poor numerical integrator for a
    genuinely chaotic system)."""

    class _Cfg:
        mode = "markovian"

        def __init__(self, d_latent: int):
            self.d_latent = d_latent

    def __init__(self, xi: np.ndarray, d_latent: int, ode_substeps: int = 4):
        self.xi = torch.tensor(xi, dtype=torch.float32)
        self.ode_substeps = ode_substeps
        self.cfg = FittedLatentPDE._Cfg(d_latent)
        self.mode = "markovian"

    def field(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d)` -> `dz/dt`: `(B, d)`, the fitted vector field,
        batched -- same circular-shift feature construction as
        `build_library`, in torch so it's differentiable via
        `torch.func.jvp` for the Lyapunov tangent-map machinery."""
        zm2 = torch.roll(z, 2, dims=-1)
        zm1 = torch.roll(z, 1, dims=-1)
        z0 = z
        zp1 = torch.roll(z, -1, dims=-1)
        zp2 = torch.roll(z, -2, dims=-1)
        dz1 = (zp1 - zm1) / 2.0
        dz2 = zp1 - 2 * z0 + zm1
        dz4 = zp2 - 4 * zp1 + 6 * z0 - 4 * zm1 + zm2
        adv = z0 * dz1
        cross = zp1 * zm1
        cols = torch.stack([torch.ones_like(z0), z0, z0**2, dz1, adv, dz2, dz4, cross], dim=-1)
        return cols @ self.xi.to(z.dtype)

    def step_one(self, z: torch.Tensor) -> torch.Tensor:
        h = 1.0 / self.ode_substeps
        z_t = z
        for _ in range(self.ode_substeps):
            k1 = self.field(z_t)
            k2 = self.field(z_t + 0.5 * h * k1)
            k3 = self.field(z_t + 0.5 * h * k2)
            k4 = self.field(z_t + h * k3)
            z_t = z_t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z_t

    def step(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return self.step_one(z_curr)  # markovian: z_prev accepted, ignored

    def rollout(self, z_prev: torch.Tensor, z_curr: torch.Tensor, k: int, step_noise: float = 0.0) -> torch.Tensor:
        outputs = []
        for _ in range(k):
            z_next = self.step(z_prev, z_curr)
            outputs.append(z_next)
            z_prev, z_curr = z_curr, z_next
        return torch.stack(outputs, dim=1)

    def to(self, device):
        return self

    def eval(self):
        return self


def r2_and_rmse(xi: np.ndarray, Zt: np.ndarray, Ztp1: np.ndarray, dt_snap: float) -> tuple[float, float]:
    Theta = build_library(Zt)
    pred = Theta @ xi
    y_true = ((Ztp1 - Zt) / dt_snap).reshape(-1)
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return float(r2), rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--n-train-runs", type=int, default=None,
                         help="Number of trajectories used to FIT the equation (default: the "
                         "dataset's own metadata n_train, matching Stage 1's own split).")
    parser.add_argument("--threshold", type=float, default=0.05,
                         help="STLSQ sparsity cutoff (on the normalized-column coefficient scale).")
    parser.add_argument("--stlsq-iters", type=int, default=15)
    parser.add_argument("--ode-substeps", type=int, default=4)
    parser.add_argument("--rollout-ks", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--skip-lyapunov", action="store_true",
                         help="Skip the (slow-ish) Lyapunov spectrum of the fitted equation.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    ae.eval()
    prop, prop_cfg, _ = load_propagator_checkpoint(args.prop_checkpoint, device="cpu")
    prop.eval()

    with h5py.File(args.dataset, "r") as f:
        traj = torch.tensor(f["trajectories"][:], dtype=torch.float32)  # (n_runs, T, NX)
        n_train_meta = int(f["metadata"].attrs["n_train"]) if "metadata" in f else None
    n_runs, T, NX = traj.shape
    n_train_runs = args.n_train_runs if args.n_train_runs is not None else (n_train_meta or int(0.8 * n_runs))

    with torch.no_grad():
        z_flat = ae.encode(traj.reshape(n_runs * T, NX))
    z_seq = z_flat.view(n_runs, T, -1).numpy()
    d = z_seq.shape[-1]
    print(f"Encoded {n_runs} trajectories (T={T} steps each) to d_latent={d}. "
          f"Fitting on the first {n_train_runs} runs, held out {n_runs - n_train_runs} for evaluation.")

    z_train, z_val = z_seq[:n_train_runs], z_seq[n_train_runs:]
    if z_val.shape[0] == 0:
        print("WARNING: no held-out runs available (n_train_runs >= n_runs) -- reusing train runs for eval.")
        z_val = z_train

    Z_t = z_train[:, :-1].reshape(-1, d)
    Z_tp1 = z_train[:, 1:].reshape(-1, d)
    Theta = build_library(Z_t)
    y = ((Z_tp1 - Z_t) / args.dt_snap).reshape(-1)

    xi, active = stlsq(Theta, y, threshold=args.threshold, n_iters=args.stlsq_iters)

    print("\n=== Fitted latent PDE (SINDy-style, ONE shared coefficient vector across all latent indices) ===")
    for name, coef, is_active in zip(FEATURE_NAMES, xi, active):
        marker = "*" if is_active else " "
        print(f"  {marker} {name:20s}  {coef: .6f}")

    r2_train, rmse_train = r2_and_rmse(xi, Z_t, Z_tp1, args.dt_snap)
    Zv_t, Zv_tp1 = z_val[:, :-1].reshape(-1, d), z_val[:, 1:].reshape(-1, d)
    r2_val, rmse_val = r2_and_rmse(xi, Zv_t, Zv_tp1, args.dt_snap)
    print(f"\nSingle-step derivative fit: R^2 train={r2_train:.4f} (rmse={rmse_train:.6f})  "
          f"R^2 held-out={r2_val:.4f} (rmse={rmse_val:.6f})")
    print("(R^2 well below the train value on held-out data would mean the shared-coefficient "
          "assumption is overfitting to the specific indices/trajectories seen during fitting.)")

    fitted = FittedLatentPDE(xi, d_latent=d, ode_substeps=args.ode_substeps)
    z_val_t = torch.tensor(z_val, dtype=torch.float32)

    print(f"\n=== Multi-step rollout comparison (held-out runs, MSE in latent space) ===")
    print(f"{'k':>4s}  {'fitted_vs_real':>16s}  {'propagator_vs_real':>20s}  {'fitted_vs_propagator':>22s}")
    is_markovian = getattr(prop, "mode", None) == "markovian"
    for k in args.rollout_ks:
        if T < k + 2:
            print(f"{k:4d}  (skipped -- trajectory too short, need T >= k+2)")
            continue
        z0, z1 = z_val_t[:, 0], z_val_t[:, 1]
        with torch.no_grad():
            fitted_roll = fitted.rollout(z0, z1, k)
            prop_roll = prop.rollout(z0, z1, k) if is_markovian else None
        real_roll = z_val_t[:, 2:2 + k]
        mse_fitted_real = torch.mean((fitted_roll - real_roll) ** 2).item()
        if prop_roll is not None:
            mse_prop_real = torch.mean((prop_roll - real_roll) ** 2).item()
            mse_fitted_prop = torch.mean((fitted_roll - prop_roll) ** 2).item()
        else:
            mse_prop_real, mse_fitted_prop = float("nan"), float("nan")
        print(f"{k:4d}  {mse_fitted_real:16.4f}  {mse_prop_real:20.4f}  {mse_fitted_prop:22.4f}")
    if not is_markovian:
        print(f"(propagator mode={getattr(prop, 'mode', None)!r} is not markovian -- "
              f"propagator_vs_real/fitted_vs_propagator columns are N/A, not directly comparable "
              f"to the fitted equation's single-current-state map without a history adapter.)")

    if not args.skip_lyapunov:
        from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
        print("\n=== Lyapunov spectrum of the FITTED equation (same machinery as Gate 3) ===")
        try:
            initial_pair = z_seq[0, 0:2].reshape(-1)
            result = lyapunov_spectrum_latent_propagator(
                fitted, initial_pair, mode="single_state", n_directions=d,
                n_steps=200, qr_every=1, dt_snap=args.dt_snap, warmup_steps=40, seed=args.seed,
            )
            print(f"D_KY={result.kaplan_yorke_dimension:.3f}  n_positive={result.n_positive}  "
                  f"lambda1={result.exponents[0]:.4f}  (true KS benchmark: D_KY~22)")
        except Exception as e:
            print(f"Lyapunov spectrum of the fitted equation failed: {e!r}")
            print("(A failure here doesn't necessarily mean the fit is bad -- it can mean the "
                  "fitted map diverges/decays too fast for Benettin's method's bracketing search, "
                  "same failure mode already seen for some trained propagators in this project.)")


if __name__ == "__main__":
    main()
