"""2D generalization of `KSAutoencoderLocalField`
(`encoder_kind="local_field_2d"`, added 2026-09-28,
`docs/ks_local_field_ltransfer.tex` "Staged next steps" Step 1) -- see
`LocalFieldAutoencoderConfig2D`'s own docstring for the sizing/latent-
layout convention. Every layer is the direct `Conv2d`/`ConvTranspose2d`
analogue of the 1D class's own `Conv1d`/`ConvTranspose1d` layers, same
names, same gauge-anchor design (channel 0 = non-learned physical
average over each 2D patch), same no-bias output projection (Section
153's fix for the constant-offset-drift artifact, carried over
unchanged since the same risk -- an unconstrained additive offset with
nothing in the loss constraining MEAN -- applies identically in 2D).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import LocalFieldAutoencoderConfig2D


class KSAutoencoderLocalField2D(nn.Module):
    """`u: (B, NX*NY) -> z: (B, n_sites_x*n_sites_y*local_channels)`, a
    genuine 2D local field, flattened site-major (site_x outermost,
    site_y next, channel innermost). Channel 0 at every site is the
    exact, non-learned physical average of `u` over that site's 2D
    patch.

    **Public `encode`/`decode` take/return a FLAT `(B, NX*NY)` physical
    field, not `(B, NX, NY)`** -- reshaped to/from 2D only internally.
    This is deliberate, not an oversight: `ks_latent.training.loops.
    train_stage1`/`train_stage2` (reused unmodified for 2D training, the
    same "reuse the validated loop, don't fork it" choice the
    multiscale-KS experiment already made) hard-code `n_runs, T, NX =
    trajectories.shape` and `.reshape(-1, NX)` -- a flat physical-state
    convention baked into the shared training loop. Keeping the public
    interface flat, exactly like every other encoder in this codebase,
    means zero changes are needed there; only this class needs to know
    the flat `NX*NY` vector is secretly a `(NX,NY)` grid."""

    def __init__(self, cfg: LocalFieldAutoencoderConfig2D):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        px, py = cfg.patch_x, cfg.patch_y
        r, layers = cfg.site_mix_radius, cfg.n_site_mix_layers

        self.enc_patchify = nn.Conv2d(1, H, kernel_size=(px, py), stride=(px, py))
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

        self.dec_in = nn.Conv2d(cfg.local_channels, H, kernel_size=1)
        dr = r if getattr(cfg, "dec_site_mix_radius", None) is None else cfg.dec_site_mix_radius
        dl = layers if getattr(cfg, "dec_n_site_mix_layers", None) is None else cfg.dec_n_site_mix_layers
        self.dec_mix = nn.ModuleList(
            [
                nn.Conv2d(H, H, kernel_size=2 * dr + 1, padding=dr, padding_mode="circular")
                for _ in range(dl)
            ]
        )
        self.dec_unpatchify = nn.ConvTranspose2d(H, 1, kernel_size=(px, py), stride=(px, py))

    def encode(self, u_flat: torch.Tensor) -> torch.Tensor:
        """`u_flat`: `(B, NX*NY)` -> `z`: `(B, n_sites_x*n_sites_y*local_channels)`."""
        cfg = self.cfg
        B = u_flat.shape[0]
        Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels
        u = u_flat.reshape(B, cfg.NX, cfg.NY)

        h = F.gelu(self.enc_patchify(u.unsqueeze(1)))  # (B, H, Px, Py)
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
        h = F.gelu(self.dec_in(z_field))
        for conv in self.dec_mix:
            h = F.gelu(conv(h))
        u_hat = self.dec_unpatchify(h).squeeze(1)  # (B, NX, NY)
        return u_hat.reshape(B, cfg.NX * cfg.NY)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
