"""Plain MLP encoder/decoder ("Track A"), added 2026-08-29.

Ported from a reference implementation
(`/Users/daltonjones/Documents/experiments/ks_latent/models.py`,
`MLPEncoder`/`MLPDecoder`) as an alternative to
`ks_latent/models/autoencoder_patched.py`'s patched-transformer -- see
`MLPAutoencoderConfig`'s docstring and CLAUDE_CODE_BRIEF.md §5.1/5.2 "Ported
improvements" addendum for the motivating comparison. Exposes the same
`encode`/`decode`/`forward` interface as `KSAutoencoderPatched` so it is a
drop-in replacement everywhere an autoencoder is passed (`train_stage1`,
`eval_stage1_reconstruction`, `encode_dataset_with_shifts`, Stage-2
scripts).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import MLPAutoencoderConfig


def _mlp_trunk(sizes: list[int]) -> nn.Sequential:
    """Hidden stack: `Linear -> LayerNorm -> GELU`, repeated. No final layer
    (the caller adds its own linear head)."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(nn.LayerNorm(b))
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class KSAutoencoderMLP(nn.Module):
    """`NX -> hidden... -> d_latent` encoder, exact mirror decoder, both with
    a linear (unconstrained) output -- see `MLPAutoencoderConfig`'s
    docstring for the full rationale."""

    def __init__(self, cfg: MLPAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        enc_sizes = [cfg.NX, *cfg.hidden]
        self.enc_trunk = _mlp_trunk(enc_sizes)
        self.enc_out = nn.Linear(enc_sizes[-1], cfg.d_latent)

        dec_sizes = [cfg.d_latent, *reversed(cfg.hidden)]
        self.dec_trunk = _mlp_trunk(dec_sizes)
        self.dec_out = nn.Linear(dec_sizes[-1], cfg.NX)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        return self.enc_out(self.enc_trunk(u))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        return self.dec_out(self.dec_trunk(z))

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
