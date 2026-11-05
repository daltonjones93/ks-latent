"""Column local-field autoencoder and local stencil propagator for CM1.

Autoencoder. A column of `C = V*Z` channels at each of `NX` grid cells is
mapped to a latent field `(P, c)` with `P = NX / patch` sites:
  encoder  Conv1d(C -> H, kernel = stride = patch)   (exact patchify)
           n_mix x [Conv1d(H -> H, 2r+1) + GELU], residual
           Conv1d(H -> c, 1, bias=False)
  decoder  the mirror, ConvTranspose1d(H -> C, kernel = stride = patch)
Locality is horizontal only: each site sees its own patch plus `n_mix * r`
sites either side (encoder reach), and the decoder adds the same reach.
Shifting the input by a multiple of `patch` shifts the latent by whole
sites, exactly, at sites farther than the reach from the domain edges
(tested). Padding is never circular: the lateral boundaries are open.

Encoder biases. The final 1x1 layer has no bias, for the reason found in
the 1D KS local field (`autoencoder_local_field.py`): a free offset per
channel drifts during training and nothing in the loss resists it.

Non-negative outputs. The condensate channels (sqrt of mixing ratio) pass
through a sharp softplus, `log(1 + exp(beta x)) / beta`, so decoded
condensate is >= 0 and reaches ~0 (5e-6 at x = -1 for beta = 10) without
killing the gradient as a ReLU would.

Propagator. `z_{n+1} = z_n + f(z_n)` on the latent sites, `f` a stack of
`layers` convolutions of radius `r` (reach `layers * r` sites per step),
with the last layer zero-initialized so the map is exactly the identity at
initialization. Markovian: the PDE is first order in time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ColumnAEConfig:
    n_channels: int = 7 * 62  # C = V*Z
    n_levels: int = 62
    patch: int = 8  # cells per site (8 x 250 m = 2 km)
    hidden: int = 256
    latent_channels: int = 16
    n_mix: int = 2
    mix_radius: int = 1
    padding_mode: str = "replicate"
    positive_beta: float = 10.0

    @property
    def reach_sites(self) -> int:
        return self.n_mix * self.mix_radius


class _Mixer(nn.Module):
    def __init__(self, H: int, r: int, n: int, padding_mode: str):
        super().__init__()
        if padding_mode == "circular":
            raise ValueError("circular padding is wrong for CM1 (open lateral boundaries)")
        self.convs = nn.ModuleList(nn.Conv1d(H, H, 2 * r + 1, padding=r, padding_mode=padding_mode)
                                   for _ in range(n))

    def forward(self, h):
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return h


class ColumnFieldAE(nn.Module):
    """`u`: `(B, C, NX)` normalized columns -> `z`: `(B, P, c)` -> `u_hat`: `(B, C, NX)`."""

    def __init__(self, cfg: ColumnAEConfig, positive_mask: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        C, H, c, p = cfg.n_channels, cfg.hidden, cfg.latent_channels, cfg.patch
        self.enc_in = nn.Conv1d(C, H, p, stride=p)
        self.enc_mix = _Mixer(H, cfg.mix_radius, cfg.n_mix, cfg.padding_mode)
        self.enc_out = nn.Conv1d(H, c, 1, bias=False)
        self.dec_in = nn.Conv1d(c, H, 1)
        self.dec_mix = _Mixer(H, cfg.mix_radius, cfg.n_mix, cfg.padding_mode)
        self.dec_out = nn.ConvTranspose1d(H, C, p, stride=p)
        pm = torch.zeros(C, dtype=torch.bool) if positive_mask is None else positive_mask.bool()
        if pm.shape != (C,):
            raise ValueError(f"positive_mask shape {tuple(pm.shape)} != ({C},)")
        self.register_buffer("positive_mask", pm)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        if u.shape[-1] % self.cfg.patch:
            raise ValueError(f"width {u.shape[-1]} is not a multiple of patch {self.cfg.patch}")
        h = F.gelu(self.enc_in(u))
        h = self.enc_mix(h)
        return self.enc_out(h).transpose(1, 2)  # (B, P, c)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec_in(z.transpose(1, 2)))
        h = self.dec_mix(h)
        out = self.dec_out(h)
        if self.positive_mask.any():
            pos = F.softplus(out, beta=self.cfg.positive_beta)
            out = torch.where(self.positive_mask[None, :, None], pos, out)
        return out

    def forward(self, u):
        z = self.encode(u)
        return self.decode(z), z


@dataclass
class StencilPropConfig:
    latent_channels: int = 16
    hidden: int = 256
    layers: int = 3
    radius: int = 1
    padding_mode: str = "replicate"

    @property
    def reach_sites(self) -> int:
        return self.layers * self.radius


class StencilPropagator(nn.Module):
    """`z`: `(B, P, c)` -> `z + f(z)`, `f` local with reach `layers * radius` sites."""

    def __init__(self, cfg: StencilPropConfig):
        super().__init__()
        if cfg.padding_mode == "circular":
            raise ValueError("circular padding is wrong for CM1 (open lateral boundaries)")
        self.cfg = cfg
        c, H, r = cfg.latent_channels, cfg.hidden, cfg.radius
        dims = [c] + [H] * (cfg.layers - 1)
        self.convs = nn.ModuleList(nn.Conv1d(a, H, 2 * r + 1, padding=r, padding_mode=cfg.padding_mode)
                                   for a in dims)
        self.head = nn.Conv1d(H, c, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z.transpose(1, 2)
        for conv in self.convs:
            h = F.gelu(conv(h))
        return z + self.head(h).transpose(1, 2)


def forced_halo_rollout(prop: StencilPropagator, z_true: torch.Tensor, k: int, halo: int,
                        noise: float = 0.0) -> torch.Tensor:
    """`z_true`: `(B, k+1, P, c)`. Rolls `k` steps from `z_true[:, 0]` (+ input
    noise), setting the `halo` sites at each end to the true state before
    every step (the boundary condition a limited-area model is given).
    Returns all `k` predicted states, `(B, k, P, c)`, halo included (true)."""
    state = z_true[:, 0]
    if noise > 0:
        state = state + noise * torch.randn_like(state)
    P = z_true.shape[2]
    out = []
    for j in range(k):
        if halo > 0:
            state = torch.cat([z_true[:, j, :halo], state[:, halo:P - halo], z_true[:, j, P - halo:]], dim=1)
        state = prop(state)
        if halo > 0:
            state = torch.cat([z_true[:, j + 1, :halo], state[:, halo:P - halo], z_true[:, j + 1, P - halo:]], dim=1)
        out.append(state)
    return torch.stack(out, dim=1)
