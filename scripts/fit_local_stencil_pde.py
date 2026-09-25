"""User-directed 2026-09-25: "do you think we would be able to fit a pde
more easily to the latent space of 224 since it's highly localized?
please try this and report on the results."

Fits a SHARED, translation-invariant local stencil regression -- the
literal Phase 11 "stencil propagator" idea (brief Section 13:
`z_j_dot = F(z_{j-w},...,z_j,...,z_{j+w})`, same weights at every site)
-- to a `local_field` checkpoint's own latent trajectory, using plain
Ridge regression (a genuinely PDE-like, small/interpretable model, not
a neural network) as the closure law. Sweeps stencil half-width `w` and
reports held-out R^2 -- if the latent's own dynamics are genuinely
local (as D3 bandedness claims), a SMALL `w` should already capture
most of the achievable fit, with wider `w` adding little.

Not a neural-network propagator comparison -- deliberately the opposite:
the question is whether a SMALL number of local, linear (or low-order
polynomial) terms already explains most of the one-step dynamics,
which is the actual "is this latent PDE-like" question, not "can SOME
model with enough capacity fit it."
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from ks_latent.models import load_autoencoder_checkpoint


def encode_to_field(ae, trajectories: torch.Tensor, n_sites: int, local_channels: int) -> np.ndarray:
    """`trajectories`: (n_traj, T, NX) -> `(n_traj, T, n_sites, local_channels)`, site-major."""
    n_traj, T, NX = trajectories.shape
    with torch.no_grad():
        z = ae.encode(trajectories.reshape(-1, NX))  # (n_traj*T, d_latent)
    return z.reshape(n_traj, T, n_sites, local_channels).numpy()


def build_stencil_dataset(field: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """`field`: `(n_traj, T, n_sites, c)`. For every `(traj, t, site)`,
    input = flattened window of `2w+1` circularly-neighboring sites'
    CURRENT `c`-dim state; target = that site's own NEXT-step delta.
    Pooled across all sites/times/trajectories -- one shared regression
    problem, matching the stencil's own translation-invariance."""
    n_traj, T, n_sites, c = field.shape
    delta = field[:, 1:] - field[:, :-1]  # (n_traj, T-1, n_sites, c)
    cur = field[:, :-1]  # (n_traj, T-1, n_sites, c), aligned with delta

    X_list, y_list = [], []
    for offset in range(-w, w + 1):
        X_list.append(np.roll(cur, shift=-offset, axis=2))  # neighbor at +offset from each site
    X = np.concatenate(X_list, axis=-1)  # (n_traj, T-1, n_sites, (2w+1)*c)
    X = X.reshape(-1, (2 * w + 1) * c)
    y = delta.reshape(-1, c)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--n-train-traj", type=int, default=40)
    parser.add_argument("--n-val-traj", type=int, default=10)
    parser.add_argument("--widths", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=1e-3, help="Ridge regularization strength.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial feature degree (1=linear).")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    ae, ae_cfg, ckpt = load_autoencoder_checkpoint(args.ae_checkpoint)
    ae.eval()
    n_sites, local_channels = ae_cfg.n_sites, ae_cfg.local_channels
    d_latent = n_sites * local_channels
    print(f"[{args.label}] d_latent={d_latent} n_sites={n_sites} local_channels={local_channels}")

    with h5py.File(args.dataset, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    n_traj = trajectories.shape[0]
    train_idx = list(range(args.n_train_traj))
    val_idx = list(range(n_traj - args.n_val_traj, n_traj))

    field_train = encode_to_field(ae, trajectories[train_idx], n_sites, local_channels)
    field_val = encode_to_field(ae, trajectories[val_idx], n_sites, local_channels)

    widths = args.widths if args.widths is not None else [0, 1, 2, 3, 4, 6, n_sites // 2]
    widths = sorted(set(w for w in widths if 0 <= w <= n_sites // 2))

    print(f"[{args.label}] {'width':>6} {'n_features':>11} {'train_R2':>10} {'val_R2':>10}")
    results = []
    for w in widths:
        X_train, y_train = build_stencil_dataset(field_train, w)
        X_val, y_val = build_stencil_dataset(field_val, w)

        if args.degree > 1:
            poly = PolynomialFeatures(degree=args.degree, include_bias=False)
            X_train = poly.fit_transform(X_train)
            X_val = poly.transform(X_val)

        model = Ridge(alpha=args.alpha)
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        results.append((w, X_train.shape[1], train_r2, val_r2))
        print(f"[{args.label}] {w:>6} {X_train.shape[1]:>11} {train_r2:>10.4f} {val_r2:>10.4f}")

    print(f"[{args.label}] done.")


if __name__ == "__main__":
    main()
