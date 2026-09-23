"""Latent-coordinate permutation wrapper (added 2026-08-31, user-directed):
"train an autoencoder with good spatial coherence, then after stage 1, we
apply D3, to determine the best permutation for local spatial coherence.
We could then apply this permutation to all latent states, and train a
stage 2 propagator with these permuted, hopefully spatially coherent
states."

Permuting the latent axis is a free symmetry of any trained autoencoder:
`decode(z) == decode(P^-1 (P z))` for any permutation `P`, so wrapping an
already-trained (and otherwise UNCHANGED) `ae`'s `encode`/`decode` with a
fixed permutation and its inverse produces an autoencoder that is
functionally identical for reconstruction (bit-for-bit, no retraining) but
whose latent axis is relabeled according to `permutation`. The point: `D3`
(`ks_latent/analysis/diagnostics.py`'s `jacobian_coupling` +
`bandedness_p_value`) finds a Fiedler-vector reordering of a propagator's
Jacobian coupling matrix that is significantly more banded than a random
permutation would be -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md
Section 20 for the full motivation and results. Training a NEW propagator
(deliberately a narrow-receptive-field one, e.g. `backbone="masked_mlp"`
with a small `attn_window`) IN THIS PERMUTED COORDINATE SYSTEM tests
whether a local propagator was previously failing only because
`local_mlp`/`masked_mlp` (Sections 10/11) imposed locality in the raw,
arbitrary latent index order -- the saved Fiedler permutations found in
this document are essentially scrambled relative to that raw order (not
close to the identity), so this is a genuinely different hypothesis than
what was tested there.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PermutedAutoencoder(nn.Module):
    """Wraps any autoencoder exposing `encode`/`decode` (kept as an
    unmodified submodule -- no weights are touched) so that `encode`
    returns `z` reindexed by `permutation`, and `decode` expects `z` in
    that same reindexed order (undoing it internally before calling the
    wrapped `ae.decode`). `self.cfg` proxies the wrapped `ae.cfg` so code
    that reads e.g. `ae.cfg.d_latent` directly (`run_diagnostics.py`,
    `ks_latent/training/loops.py`) keeps working unmodified."""

    def __init__(self, ae: nn.Module, permutation: np.ndarray | torch.Tensor):
        super().__init__()
        self.ae = ae
        self.cfg = ae.cfg
        perm = torch.as_tensor(permutation, dtype=torch.long)
        self.register_buffer("permutation", perm)
        self.register_buffer("inverse_permutation", torch.argsort(perm))

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(u)[..., self.permutation]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z[..., self.inverse_permutation])

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z


def load_latent_permutation(path: str, key: str = "d3_permutation") -> np.ndarray:
    """Load a saved permutation array. Accepts a bare `.npy` file, or an
    `.npz` archive (the `diagnostics_arrays{tag}.npz` format
    `scripts/run_diagnostics.py` writes), reading `key` from it -- either
    `"d3_permutation"` (default: the propagator-Jacobian-coupling-derived
    ordering, Section 20), `"d6_permutation"` (the raw-encoded-data
    TEMPORAL-coherence-derived ordering -- lagged correlation across time,
    Section 21), or `"d7_permutation"` (the raw-encoded-data SAME-TIME
    correlation-derived ordering -- no lag, no propagator at all, the most
    literal "do neighboring latent variables vary together," Section 22)."""
    if path.endswith(".npz"):
        return np.load(path)[key]
    return np.load(path)
