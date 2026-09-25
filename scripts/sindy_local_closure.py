"""User-directed 2026-09-25, follow-up to `scripts/fit_local_stencil_pde.py`:
"how much would it take to test the SINDy-style sparse selection for the
latent space of 224 to see if it collapses to a small human-readable set
of terms? ... please try this and document any findings."

Reuses `fit_local_stencil_pde.py`'s data pipeline (encode a real KS
trajectory dataset through a `local_field` checkpoint, build a shared,
translation-invariant local stencil regression problem) but replaces
dense Ridge with `pysindy.optimizers.STLSQ` (Sequential Thresholded
Least Squares, the standard SINDy sparse-selection algorithm) over a
NAMED degree-2 polynomial feature library, so surviving terms are
human-readable (`z[-1,c1]*z[+1,c0]`, not just an opaque coefficient
vector).

Per this project's own brief (Phase 8, `CLAUDE_CODE_BRIEF.md` -- "use
ensemble/bootstrap SINDy and report per-term selection probabilities,
not a single sparse fit; spurious terms are suppressed by more data,
not more regularization"): does BOOTSTRAP resampling at a chosen
threshold, reporting each term's selection frequency across resamples,
not a single point estimate.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
from pysindy.optimizers import STLSQ
from sklearn.linear_model import Ridge

from fit_local_stencil_pde import build_stencil_dataset, encode_to_field
from ks_latent.models import load_autoencoder_checkpoint


def build_named_quadratic_library(X_linear: np.ndarray, offsets: list[int], local_channels: int):
    """`X_linear`: `(N, (2w+1)*c)` raw window features (order: offset-major,
    channel-minor, matching `build_stencil_dataset`). Returns `(X_full,
    names)`: `X_full` = linear features followed by all degree<=2 cross/
    square terms (upper triangle, self-products included), `names` a
    matching list of human-readable strings, e.g. `z[-1,c1]`,
    `z[-1,c1]*z[+1,c0]`, `z[0,c0]^2`."""
    n_lin = X_linear.shape[1]
    lin_names = [f"z[{o:+d},c{c}]" for o in offsets for c in range(local_channels)]
    assert len(lin_names) == n_lin

    quad_cols, quad_names = [], []
    for i in range(n_lin):
        for j in range(i, n_lin):
            quad_cols.append(X_linear[:, i] * X_linear[:, j])
            if i == j:
                quad_names.append(f"{lin_names[i]}^2")
            else:
                quad_names.append(f"{lin_names[i]}*{lin_names[j]}")
    X_quad = np.stack(quad_cols, axis=1)
    X_full = np.concatenate([X_linear, X_quad], axis=1)
    names = lin_names + quad_names
    return X_full, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--n-train-traj", type=int, default=40)
    parser.add_argument("--n-val-traj", type=int, default=10)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--bootstrap-frac", type=float, default=0.3)
    parser.add_argument("--bootstrap-threshold", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=25, help="How many top-selected terms to print.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    ae, ae_cfg, ckpt = load_autoencoder_checkpoint(args.ae_checkpoint)
    ae.eval()
    n_sites, local_channels = ae_cfg.n_sites, ae_cfg.local_channels
    w = args.width
    offsets = list(range(-w, w + 1))
    print(f"[sindy] d_latent={n_sites*local_channels} n_sites={n_sites} local_channels={local_channels} width={w}")

    with h5py.File(args.dataset, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    n_traj = trajectories.shape[0]
    train_idx = list(range(args.n_train_traj))
    val_idx = list(range(n_traj - args.n_val_traj, n_traj))

    field_train = encode_to_field(ae, trajectories[train_idx], n_sites, local_channels)
    field_val = encode_to_field(ae, trajectories[val_idx], n_sites, local_channels)

    X_train_lin, y_train = build_stencil_dataset(field_train, w)
    X_val_lin, y_val = build_stencil_dataset(field_val, w)
    X_train, names = build_named_quadratic_library(X_train_lin, offsets, local_channels)
    X_val, _ = build_named_quadratic_library(X_val_lin, offsets, local_channels)
    print(f"[sindy] library size: {len(names)} terms, {X_train.shape[0]} pooled training rows "
          f"(target: channel-0 delta only, the anchor's own local-mean closure)")

    # Target channel 0 (the physically-anchored local mean) -- the single
    # cleanest, most interpretable target to look for a compact law in.
    y_train_c0 = y_train[:, 0]
    y_val_c0 = y_val[:, 0]

    # Standardize features (STLSQ's threshold is an absolute magnitude --
    # linear and quadratic terms live on very different raw scales).
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma

    print(f"\n[sindy] === sparsity path (threshold -> n_active_terms, held-out R^2) ===")
    dense_r2 = Ridge(alpha=1e-3).fit(X_train_n, y_train_c0).score(X_val_n, y_val_c0)
    print(f"[sindy] threshold=0.0 (dense Ridge reference)  n_active={X_train.shape[1]:>4}  val_R2={dense_r2:.4f}")
    for thr in args.thresholds:
        if thr == 0.0:
            continue
        opt = STLSQ(threshold=thr, alpha=1e-3, max_iter=50, normalize_columns=False)
        opt.fit(X_train_n, y_train_c0)
        coef = np.asarray(opt.coef_).flatten()
        active = np.flatnonzero(coef)
        y_val_pred = X_val_n @ coef
        ss_res = np.sum((y_val_c0 - y_val_pred) ** 2)
        ss_tot = np.sum((y_val_c0 - y_val_c0.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        print(f"[sindy] threshold={thr:<5}  n_active={len(active):>4}  val_R2={r2:.4f}")

    print(f"\n[sindy] === bootstrap selection stability at threshold={args.bootstrap_threshold} "
          f"({args.n_bootstrap} resamples, {args.bootstrap_frac:.0%} of pooled rows each) ===")
    n_rows = X_train_n.shape[0]
    n_sample = int(n_rows * args.bootstrap_frac)
    selection_count = np.zeros(X_train.shape[1])
    coef_sum = np.zeros(X_train.shape[1])
    for b in range(args.n_bootstrap):
        idx = rng.choice(n_rows, size=n_sample, replace=True)
        opt = STLSQ(threshold=args.bootstrap_threshold, alpha=1e-3, max_iter=50)
        opt.fit(X_train_n[idx], y_train_c0[idx])
        coef = np.asarray(opt.coef_).flatten()
        active = coef != 0
        selection_count += active
        coef_sum += coef

    selection_prob = selection_count / args.n_bootstrap
    mean_coef = np.divide(coef_sum, selection_count, out=np.zeros_like(coef_sum), where=selection_count > 0)
    order = np.argsort(-selection_prob)
    print(f"[sindy] top {args.top_k} terms by bootstrap selection probability:")
    print(f"{'term':<28} {'P(selected)':>12} {'mean coef (standardized)':>26}")
    for i in order[: args.top_k]:
        if selection_prob[i] == 0:
            break
        print(f"{names[i]:<28} {selection_prob[i]:>12.2f} {mean_coef[i]:>26.4f}")

    robust_terms = [names[i] for i in order if selection_prob[i] >= 0.8]
    print(f"\n[sindy] terms selected in >=80% of bootstrap resamples ({len(robust_terms)} of {len(names)}):")
    for t in robust_terms:
        print(f"  {t}")

    print("[sindy] done.")


if __name__ == "__main__":
    main()
