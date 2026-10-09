"""Anisotropic 2D local-field encoder (`encoder_kind="local_field_2d_aniso"`,
added 2026-09-29, user-directed after G2's own reconciliation check
confirmed Kolmogorov flow's zero-retraining transfer gap is genuine, not
an IC artifact -- see `docs/ks_local_field_ltransfer.tex`'s "orientation
hypothesis" section).

**Why this exists, and why it is NOT rotation-equivariant.** The
2D-KS-specific idea this followed from (steerable/rotation-equivariant
convolutions) assumes the underlying physics is isotropic. Kolmogorov
flow is not: its forcing `f(y)=-nu*n^3*cos(n*y)` singles out `y`
specifically, giving exact translation invariance in `x` but only
DISCRETE period-`2*pi/n` invariance in `y` (confirmed directly by
`ks_latent.solver.kolmogorov`'s own equivariance tests -- see that
module's docstring). Building in full rotation equivariance here would
impose a symmetry the true dynamics does not have.

**What this class does instead**: keeps `KSAutoencoderLocalField2D`'s
exact patchify/site-mixing/gauge-anchor structure (still fully
weight-shared -- hence equivariant -- in BOTH spatial axes at the LAYER
level), but concatenates two extra, FIXED (non-learned, non-trainable)
input channels -- `cos(forcing_n*y)`, `sin(forcing_n*y)`, evaluated at
each pixel (encoder input) or site (decoder input) -- alongside the
physical field / latent channels. These two channels are, by
construction, EXACTLY invariant under `y -> y + 2*pi/forcing_n` (the
system's own real residual symmetry: `cos(forcing_n*(y+2*pi/forcing_n))
= cos(forcing_n*y + 2*pi) = cos(forcing_n*y)`) while varying
meaningfully within one period -- giving the learned weights explicit,
exact access to "where am I in the forcing's own phase," rather than
`KSAutoencoderLocalField2D`'s implicit (and for THIS system, physically
wrong) assumption that `y` is a free translation symmetry just like `x`.
Translation equivariance in `x` is completely unaffected (the phase
channels depend only on `y`), so `x`-direction L-transfer (this
project's own G2 experiment) still works via the same zero-retraining
resize mechanism, unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import LocalFieldAutoencoderAniso2DConfig


class KSAutoencoderLocalFieldAniso2D(nn.Module):
    """`u: (B, NX*NY) -> z: (B, n_sites_x*n_sites_y*local_channels)`,
    identical public interface to `KSAutoencoderLocalField2D` (flat
    physical vector in/out, reshaped to `(NX,NY)` only internally --
    same reasoning: `train_stage1`/`train_stage2` hard-code a flat
    `NX` convention). The only structural difference: `enc_patchify`
    takes 3 input channels (`[u, cos(forcing_n*Y), sin(forcing_n*Y)]`
    instead of 1), and `dec_in` takes `local_channels+2` input channels
    (the site-level phase pair concatenated to the latent field)."""

    def __init__(self, cfg: LocalFieldAutoencoderAniso2DConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        px, py = cfg.patch_x, cfg.patch_y
        r, layers = cfg.site_mix_radius, cfg.n_site_mix_layers

        self.enc_patchify = nn.Conv2d(3, H, kernel_size=(px, py), stride=(px, py))
        self.enc_mix = nn.ModuleList(
            [
                nn.Conv2d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular")
                for _ in range(layers)
            ]
        )
        self.enc_out = (
            nn.Conv2d(H, cfg.local_channels - 1, kernel_size=1, bias=False)
            if cfg.local_channels > 1 else None
        )

        self.dec_in = nn.Conv2d(cfg.local_channels + 2, H, kernel_size=1)
        self.dec_mix = nn.ModuleList(
            [
                nn.Conv2d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular")
                for _ in range(layers)
            ]
        )
        self.dec_unpatchify = nn.ConvTranspose2d(H, 1, kernel_size=(px, py), stride=(px, py))

        # Fixed (non-learned) phase channels -- see module docstring.
        # Pixel-level (encoder input): phase(j) = 2*pi*forcing_n*(j+0.5)/NY.
        y_pixel_idx = torch.arange(cfg.NY).float() + 0.5
        phase_pixel = 2.0 * math.pi * cfg.forcing_n * y_pixel_idx / cfg.NY
        cos_pixel = torch.cos(phase_pixel).view(1, 1, 1, cfg.NY)  # broadcast over (B, C, NX, NY)
        sin_pixel = torch.sin(phase_pixel).view(1, 1, 1, cfg.NY)
        self.register_buffer("cos_y_pixel", cos_pixel)
        self.register_buffer("sin_y_pixel", sin_pixel)

        # Site-level (decoder input): phase(j) = 2*pi*forcing_n*(j+0.5)/n_sites_y.
        y_site_idx = torch.arange(cfg.n_sites_y).float() + 0.5
        phase_site = 2.0 * math.pi * cfg.forcing_n * y_site_idx / cfg.n_sites_y
        cos_site = torch.cos(phase_site).view(1, 1, 1, cfg.n_sites_y)  # broadcast over (B, C, n_sites_x, n_sites_y)
        sin_site = torch.sin(phase_site).view(1, 1, 1, cfg.n_sites_y)
        self.register_buffer("cos_y_site", cos_site)
        self.register_buffer("sin_y_site", sin_site)

    def encode(self, u_flat: torch.Tensor) -> torch.Tensor:
        """`u_flat`: `(B, NX*NY)` -> `z`: `(B, n_sites_x*n_sites_y*local_channels)`."""
        cfg = self.cfg
        B = u_flat.shape[0]
        Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels
        u = u_flat.reshape(B, cfg.NX, cfg.NY)

        u_in = torch.cat(
            [u.unsqueeze(1), self.cos_y_pixel.expand(B, 1, cfg.NX, cfg.NY),
             self.sin_y_pixel.expand(B, 1, cfg.NX, cfg.NY)],
            dim=1,
        )  # (B, 3, NX, NY)
        h = F.gelu(self.enc_patchify(u_in))  # (B, H, Px, Py)
        for conv in self.enc_mix:
            h = F.gelu(conv(h))

        px, py = cfg.patch_x, cfg.patch_y
        anchor = u.reshape(B, Px, px, Py, py).mean(dim=(2, 4))  # (B, Px, Py)
        if self.enc_out is not None:
            residual = self.enc_out(h)  # (B, c-1, Px, Py)
            z_field = torch.cat([anchor.unsqueeze(1), residual], dim=1)  # (B, c, Px, Py)
        else:
            z_field = anchor.unsqueeze(1)

        return z_field.permute(0, 2, 3, 1).reshape(B, Px * Py * c)  # site-major flatten

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, n_sites_x*n_sites_y*local_channels)` -> `u_hat_flat`: `(B, NX*NY)`."""
        cfg = self.cfg
        B = z.shape[0]
        Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels

        z_field = z.reshape(B, Px, Py, c).permute(0, 3, 1, 2)  # (B, c, Px, Py)
        z_in = torch.cat(
            [z_field, self.cos_y_site.expand(B, 1, Px, Py), self.sin_y_site.expand(B, 1, Px, Py)],
            dim=1,
        )  # (B, c+2, Px, Py)
        h = F.gelu(self.dec_in(z_in))
        for conv in self.dec_mix:
            h = F.gelu(conv(h))
        u_hat = self.dec_unpatchify(h).squeeze(1)  # (B, NX, NY)
        return u_hat.reshape(B, cfg.NX * cfg.NY)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
