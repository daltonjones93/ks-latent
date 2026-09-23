"""On-device windowing and shift augmentation (brief §1.3.4: preload the
whole training tensor once, index-shuffle over it -- no `DataLoader`, no
per-batch host<->device copies)."""

from __future__ import annotations

import torch


def roll_batch(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Per-row circular shift along the last axis. `x`: `(B, NX)` or
    `(B, W, NX)`; `shifts`: `(B,)`. One common shift per row, applied
    identically across the window axis if present (brief §5.1: "one common
    cyclic shift per window")."""
    NX = x.shape[-1]
    ar = torch.arange(NX, device=x.device)
    if x.dim() == 2:
        idx = (ar.unsqueeze(0) - shifts.unsqueeze(1)) % NX
        return torch.gather(x, 1, idx)
    if x.dim() == 3:
        idx = (ar.view(1, 1, NX) - shifts.view(-1, 1, 1)) % NX
        idx = idx.expand(-1, x.shape[1], -1)
        return torch.gather(x, 2, idx)
    raise ValueError(f"roll_batch supports 2D or 3D tensors, got {x.dim()}D")


def make_window_index(n_runs: int, T: int, window: int, device=None) -> torch.Tensor:
    """All valid `(run_idx, start_idx)` pairs for windows of length `window`
    drawn from `T`-long sequences, as an `(N, 2)` tensor."""
    if T < window:
        raise ValueError(f"sequence length T={T} shorter than window={window}")
    starts = torch.arange(T - window + 1, device=device)
    runs = torch.arange(n_runs, device=device)
    grid_r, grid_s = torch.meshgrid(runs, starts, indexing="ij")
    return torch.stack([grid_r.reshape(-1), grid_s.reshape(-1)], dim=1)


def gather_windows(sequences: torch.Tensor, index: torch.Tensor, window: int) -> torch.Tensor:
    """`sequences`: `(n_runs, T, C)`; `index`: `(B, 2)` of `(run_idx,
    start_idx)`. Returns `(B, window, C)`."""
    run_idx = index[:, 0]
    start_idx = index[:, 1]
    offsets = torch.arange(window, device=sequences.device)
    idx = start_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B, window)
    return sequences[run_idx.unsqueeze(1), idx]


def index_shuffle_batches(
    n: int, batch_size: int, generator: torch.Generator | None = None
) -> list[torch.Tensor]:
    """One epoch's worth of index batches, from a single shuffled permutation
    (not per-batch random sampling with replacement). Uses the global torch
    RNG (seeded via `ks_latent.utils.seeding.set_seed`) when `generator` is
    not given -- ground rule 3 determinism relies on that seed call, not on
    passing generators through every call site."""
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]
