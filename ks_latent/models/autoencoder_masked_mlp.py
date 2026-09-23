"""Locally-receptive-field MLP encoder/decoder (`encoder_kind="masked_mlp"`,
added 2026-08-31, user-directed): "try using a masked mlp for the encoder
and decoder" -- see `MaskedMLPAutoencoderConfig`'s docstring for the full
motivation. Mirrors `ks_latent/models/autoencoder_mlp.py`'s
`KSAutoencoderMLP` structurally, but every `Linear` is replaced by
`MaskedLinearRect`, a fixed circular-band mask on a RECTANGULAR weight
matrix (generalizing `ks_latent/models/propagator.py`'s `MaskedLinear`,
which only handles the square, dimension-preserving case used by the
`masked_mlp` PROPAGATOR backbone).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import MaskedMLPAutoencoderConfig
from ks_latent.models.spectral_norm import _sn


def _circular_band_mask_rect(
    dim_out: int, dim_in: int, window: int | None, ref_dim: int
) -> torch.Tensor | None:
    """`(dim_out, dim_in)` 0/1 mask. Both axes are treated as points evenly
    spaced on the SAME normalized circular ring (`idx / dim` in `[0, 1)`),
    regardless of `dim_out`/`dim_in` themselves possibly differing (e.g.
    `NX=256` physical points vs. `d_latent=44` latent channels) -- this is
    what makes a single `window` meaningful across layers of different
    width. `window` is expressed in `ref_dim`-sized-ring units (i.e. the
    same units as `attn_window` elsewhere in this codebase, always in
    `d_latent` units) and converted to a fractional threshold via
    `window / ref_dim`. `None`: fully dense (no mask)."""
    if window is None:
        return None
    out_pos = torch.arange(dim_out).float() / dim_out
    in_pos = torch.arange(dim_in).float() / dim_in
    diff = (out_pos.unsqueeze(1) - in_pos.unsqueeze(0)).abs()
    diff = torch.minimum(diff, 1.0 - diff)
    frac_window = window / ref_dim
    return (diff <= frac_window).float()


class MaskedLinearRect(nn.Module):
    """`Linear(dim_in, dim_out)` with an optional fixed circular-band mask
    (see `_circular_band_mask_rect`). `window=None`: ordinary dense
    `Linear`. As with `ks_latent/models/propagator.py`'s `MaskedLinear`,
    a masked-out entry receives exactly zero gradient for as long as this
    module is trained with a finite `window`, so it stays exactly zero
    (not merely small) -- the mask is a non-persistent buffer excluded
    from `state_dict()`, so a trained local module's weights can still be
    copied directly into a fresh dense module of the same shape if a
    warm-start experiment is wanted later.

    `nonexpansive` (added 2026-09-04, see `ks_latent.models.propagator.
    ResidualMLPBlock`'s docstring for the full mechanism): spectral-
    normalizes `self.linear` (`_sn`/`_SpectralNormLinear` -- a manual
    power-iteration implementation, NOT PyTorch's built-in `spectral_norm`
    parametrization, which was found to crash under `--amp` on MPS; see
    `_SpectralNormLinear`'s docstring) BEFORE the mask is applied --
    `forward` calls `effective_weight()` instead of reading `.weight`
    directly when `nonexpansive` is set. Same practical heuristic and
    caveat as `MaskedLinear`'s own `nonexpansive` option (masking a
    spectral-norm-1 matrix isn't a formally guaranteed <=1 operator norm
    in full generality, but is the same effective, empirically-verified
    approach spectral normalization already is elsewhere)."""

    def __init__(self, dim_in: int, dim_out: int, window: int | None, ref_dim: int, nonexpansive: bool = False):
        super().__init__()
        self.nonexpansive = nonexpansive
        self.linear = _sn(dim_in, dim_out) if nonexpansive else nn.Linear(dim_in, dim_out)
        mask = _circular_band_mask_rect(dim_out, dim_in, window, ref_dim)
        if mask is not None and not nonexpansive:
            with torch.no_grad():
                self.linear.weight.mul_(mask)
        self.register_buffer(
            "mask", mask if mask is not None else torch.ones(dim_out, dim_in), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.effective_weight() if self.nonexpansive else self.linear.weight
        return F.linear(x, weight * self.mask, self.linear.bias)


class _MaskedMLPTrunk(nn.Module):
    """Chain of `MaskedLinearRect -> GELU` (no `LayerNorm` -- see this
    module's file docstring and `_MaskedMLPResidualBlock` in
    `propagator.py` for why: `nn.LayerNorm` normalizes across the entire
    feature axis, which here is the physical/latent-index axis a circular
    mask is trying to respect, silently reintroducing full global coupling
    regardless of masking)."""

    def __init__(self, sizes: list[int], window: int | None, ref_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [MaskedLinearRect(a, b, window, ref_dim) for a, b in zip(sizes[:-1], sizes[1:])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


class KSAutoencoderMaskedMLP(nn.Module):
    """`NX -> hidden... -> d_latent` encoder, exact mirror decoder, every
    `Linear` replaced by `MaskedLinearRect` at a shared `cfg.mask_window`
    (in `d_latent`-ring units) -- see `MaskedMLPAutoencoderConfig`'s
    docstring."""

    def __init__(self, cfg: MaskedMLPAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        w, ref = cfg.mask_window, cfg.d_latent
        enc_sizes = [cfg.NX, *cfg.hidden]
        self.enc_trunk = _MaskedMLPTrunk(enc_sizes, w, ref)
        self.enc_out = MaskedLinearRect(enc_sizes[-1], cfg.d_latent, w, ref)

        dec_sizes = [cfg.d_latent, *reversed(cfg.hidden)]
        self.dec_trunk = _MaskedMLPTrunk(dec_sizes, w, ref)
        self.dec_out = MaskedLinearRect(dec_sizes[-1], cfg.NX, w, ref)

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
