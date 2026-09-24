"""Lorenz-96 dataset generation -- exact schema mirror of
`ks_latent.solver.dataset` (KS's own dataset writer), so every downstream
consumer (`scripts/train_stage1_patched.py`'s `--dataset` loader,
`ks_latent.analysis.lyapunov`, the Gate 3/4 diagnostic scripts) reads a
Lorenz-96 HDF5 file with zero code changes: both write an `(n_traj, T, N)`
`trajectories` dataset (normalized) plus `trajectories_raw`, and a
`metadata` group carrying `n_train`, `dt_snap`, and the full producing
config. Added 2026-09-22, see `ks_latent.solver.lorenz96`'s module
docstring for the full motivation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np

from ks_latent.config import Lorenz96Config, config_hash
from ks_latent.solver.lorenz96 import integrate, l96_rhs, spinup
from ks_latent.utils.io import get_git_sha


def _write_metadata_group(f: h5py.File, cfg: Lorenz96Config, extra: dict) -> None:
    grp = f.create_group("metadata")
    for k, v in dataclasses.asdict(cfg).items():
        grp.attrs[k] = v
    grp.attrs["dt_snap"] = cfg.dt_snap
    grp.attrs["config_hash"] = config_hash(cfg)
    grp.attrs["git_sha"] = get_git_sha()
    for k, v in extra.items():
        grp.attrs[k] = v


def generate_trajectory_dataset(
    cfg: Lorenz96Config,
    path: str | Path,
    *,
    n_train: int = 50,
    n_val: int = 10,
    trajectory_time: float,
    include_derivative: bool = False,
    derivative_layout: str = "block",
) -> Path:
    """`n_train + n_val` independent trajectories, each spun up separately
    -- exact mirror of `ks_latent.solver.dataset.generate_trajectory_
    dataset`'s own contract and HDF5 layout. Normalization statistics
    (mean/std) computed once on the training split only, stored in
    metadata; downstream code must apply them, not recompute normalization
    elsewhere (same rule as KS's own dataset, brief §3.3).

    `include_derivative` (added 2026-09-23, Section 204, user-directed:
    "use not only x, but x' (the time derivative of x) as a state"):
    appends `dx/dt = l96_rhs(x, F)` to every stored snapshot -- EXACT and
    analytic (L96's right-hand side is a deterministic function of `x`
    alone, no hidden state, unlike a genuine Takens LAGGED delay
    embedding) -- doubling the stored per-snapshot state from `N` to `2N`
    (`x` then `x'`, concatenated along the last axis; downstream code
    that just reads `f['trajectories'].shape[-1]` as "the state
    dimension" needs no changes). Motivated by this project's own
    D_KY=n -> 2n+1 Takens/Sauer-Yorke-Casdagli literature note
    (`docs/LITERATURE_REVIEW_AND_FINDINGS.md`): rather than a lagged
    history window (this project's existing `n_history` propagator
    option), this augments the DATA itself with an independent smooth
    function of state, testing whether that alone gives the encoder
    enough information to make the dynamics closer to Markovian in a
    much lower-dimensional latent than raw `x` allows.

    `x` and `x'` are DIFFERENT physical quantities (a position and a
    rate) with different natural scales, so unlike the plain (`include_
    derivative=False`) case's single scalar mean/std over the WHOLE
    array (justified there because all `N` sites of `x` alone are
    statistically homogeneous by translation invariance -- brief's own
    "scalar not per-feature" rule), each block gets its OWN separate
    scalar mean/std (still a single scalar broadcast across all `N`
    sites of its own block, not per-site -- translation invariance still
    holds within each block). `normalization_mean`/`normalization_std`
    metadata become 2-element arrays `[x_stat, xprime_stat]` instead of
    a bare scalar in this case (nothing else in this codebase reads
    those attrs back, confirmed by grep, so this shape change is safe);
    `state_layout` and `n_phys=N` are additionally recorded so a
    downstream reader can tell this dataset apart from a plain `N`-dim
    one.

    `derivative_layout` (added 2026-09-23, Section 207, user-directed:
    "can you think of a way of augmenting 201 with x' that will work
    with the vit's assumptions"): `include_derivative=True` only.
    `"block"` (default, Sections 204-206's original layout): `[x_0..
    x_{N-1}, x'_0..x'_{N-1}]`, concatenated blocks -- found likely
    responsible for those sections' badly-converging reconstruction,
    since it hands a ViT's patch-tokenizer patches that are either
    "all x" or "all x'", breaking the positional encoding's single-ring
    assumption (an x'-patch is not "further along in space" than an
    x-patch; it's the SAME sites, a different quantity). `"interleaved"`:
    `[x_0, x'_0, x_1, x'_1, ..., x_{N-1}, x'_{N-1}]`, site-major/
    channel-minor -- pairs with `ViTAutoencoderConfig.n_channels=2`
    (see that field's docstring), which patch-tokenizes `patch_size`
    CONSECUTIVE SITES worth of both channels together, keeping each
    token a genuine, single physical location -- the fix this section
    actually implements and tests. `state_layout` is recorded as
    `"x_xprime_block"` or `"x_xprime_interleaved"` accordingly so a
    downstream reader (or a human skimming metadata) cannot mix the two
    up."""
    if derivative_layout not in ("block", "interleaved"):
        raise ValueError(f"derivative_layout must be 'block' or 'interleaved', got {derivative_layout!r}")
    path = Path(path)
    n_runs = n_train + n_val
    n_steps = int(round(trajectory_time / cfg.dt))

    trajectories_x = np.empty((n_runs, n_steps // cfg.snapshot_every + 1, cfg.N))
    for i in range(n_runs):
        rng = np.random.default_rng(cfg.seed + i)
        x0 = spinup(cfg, rng)
        trajectories_x[i] = integrate(x0, cfg, n_steps=n_steps)

    if include_derivative:
        trajectories_xprime = l96_rhs(trajectories_x, cfg.F)
        if derivative_layout == "block":
            trajectories = np.concatenate([trajectories_x, trajectories_xprime], axis=-1)
            x_slice, xp_slice = (Ellipsis, slice(None, cfg.N)), (Ellipsis, slice(cfg.N, None))
            state_layout = "x_xprime_block"
        else:
            trajectories = np.stack([trajectories_x, trajectories_xprime], axis=-1).reshape(
                *trajectories_x.shape[:-1], 2 * cfg.N
            )
            x_slice, xp_slice = (Ellipsis, slice(0, None, 2)), (Ellipsis, slice(1, None, 2))
            state_layout = "x_xprime_interleaved"
        train = trajectories[:n_train]
        mean_x, std_x = train[x_slice].mean(), train[x_slice].std()
        mean_xp, std_xp = train[xp_slice].mean(), train[xp_slice].std()
        normalized = trajectories.copy()
        normalized[x_slice] = (trajectories[x_slice] - mean_x) / std_x
        normalized[xp_slice] = (trajectories[xp_slice] - mean_xp) / std_xp
        mean, std = np.array([mean_x, mean_xp]), np.array([std_x, std_xp])
        extra_meta = {"state_layout": state_layout, "n_phys": cfg.N}
    else:
        trajectories = trajectories_x
        train = trajectories[:n_train]
        mean = train.mean()
        std = train.std()
        normalized = (trajectories - mean) / std
        extra_meta = {"state_layout": "x", "n_phys": cfg.N}

    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=normalized, compression="gzip")
        f.create_dataset("trajectories_raw", data=trajectories, compression="gzip")
        _write_metadata_group(
            f,
            cfg,
            {
                "n_train": n_train,
                "n_val": n_val,
                "trajectory_time": trajectory_time,
                "normalization_mean": mean,
                "normalization_std": std,
                **extra_meta,
            },
        )
    return path
