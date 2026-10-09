"""Multiscale extension of `KSAutoencoderLocalField`
(`encoder_kind="multiscale_local_field"`, added 2026-09-28) -- see
`MultiscaleLocalFieldAutoencoderConfig`'s own docstring for the full
motivation and design rationale (combining Section 224's validated
local_field/local_mlp recipe with the multiscale-latent idea in
`docs/spatially_structured_latent_DA_FMM_research_idea.md`).

Encoder: `u: (B, NX) -> z: (B, d_latent)`, `z = concat([z_fine, z_regional,
z_global])`, each block site-major internally. Fine scale is a literal,
unmodified instantiation of Section 224's own encoder sub-modules
(`enc_patchify`/`enc_mix`/gauge-anchor/`enc_out`). Regional and global
scales pool the FINE scale's own hidden feature map (not `u` directly)
through one more circular-conv site-mixing stage each, with their own
independent gauge anchor computed directly from `u` (the physical
average over that scale's own, larger support).

Decoder: mirror. Each scale's own `z` block is "lifted" to the shared
hidden width via its own `1x1` conv/linear, regional and global are
upsampled (nearest-neighbor, circular-safe by construction since
`fine_n_sites` is an exact multiple of `regional_n_sites`/
`global_n_sites`) back to the fine resolution, and all three lifted
maps are SUMMED before running Section 224's own `dec_mix`/
`dec_unpatchify` (also reused verbatim) on the combined signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import MultiscaleLocalFieldAutoencoderConfig


class KSAutoencoderMultiscaleLocalField(nn.Module):
    """`u: (B, NX) -> z: (B, d_latent)`, a genuine multiscale local field.
    Channel 0 of EVERY scale is the exact, non-learned physical average
    of `u` over that scale's own support -- never produced by the
    learned conv stack."""

    def __init__(self, cfg: MultiscaleLocalFieldAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden
        patch = cfg.NX // cfg.fine_n_sites
        r, layers = cfg.fine_site_mix_radius, cfg.fine_n_site_mix_layers

        # --- Fine scale: Section 224's own encoder, verbatim. ---
        self.enc_patchify = nn.Conv1d(1, H, kernel_size=patch, stride=patch)
        self.enc_mix = nn.ModuleList(
            [nn.Conv1d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular") for _ in range(layers)]
        )
        self.enc_out_fine = (
            nn.Conv1d(H, cfg.fine_local_channels - 1, kernel_size=1, bias=False)
            if cfg.fine_local_channels > 1 else None
        )

        # --- Regional scale: pool fine hidden features, one more mixing stage. ---
        self.regional_pool = nn.AvgPool1d(cfg.regional_pool_factor, stride=cfg.regional_pool_factor)
        self.regional_mix = nn.Conv1d(H, H, kernel_size=3, padding=1, padding_mode="circular")
        self.enc_out_regional = (
            nn.Conv1d(H, cfg.regional_local_channels - 1, kernel_size=1, bias=False)
            if cfg.regional_local_channels > 1 else None
        )

        # --- Global scale: pool regional hidden features down to 1 site. ---
        self.global_pool = nn.AvgPool1d(cfg.global_pool_factor, stride=cfg.global_pool_factor)
        self.enc_out_global = (
            nn.Conv1d(H, cfg.global_local_channels - 1, kernel_size=1, bias=False)
            if cfg.global_local_channels > 1 else None
        )

        # --- Decoder: per-scale lift, then Section 224's own dec_mix/dec_unpatchify. ---
        self.dec_lift_fine = nn.Conv1d(cfg.fine_local_channels, H, kernel_size=1)
        self.dec_lift_regional = nn.Conv1d(cfg.regional_local_channels, H, kernel_size=1)
        self.dec_lift_global = nn.Linear(cfg.global_local_channels, H)
        self.dec_mix = nn.ModuleList(
            [nn.Conv1d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular") for _ in range(layers)]
        )
        self.dec_unpatchify = nn.ConvTranspose1d(H, 1, kernel_size=patch, stride=patch)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        cfg = self.cfg
        B = u.shape[0]

        # Fine.
        h_fine = F.gelu(self.enc_patchify(u.unsqueeze(1)))  # (B, H, P_fine)
        for conv in self.enc_mix:
            h_fine = F.gelu(conv(h_fine))
        patch = cfg.NX // cfg.fine_n_sites
        anchor_fine = u.reshape(B, cfg.fine_n_sites, patch).mean(dim=-1)  # (B, P_fine)
        if self.enc_out_fine is not None:
            residual_fine = self.enc_out_fine(h_fine)  # (B, c_fine-1, P_fine)
            z_fine = torch.cat([anchor_fine.unsqueeze(1), residual_fine], dim=1)
        else:
            z_fine = anchor_fine.unsqueeze(1)
        z_fine_flat = z_fine.transpose(1, 2).reshape(B, cfg.fine_n_sites * cfg.fine_local_channels)

        # Regional.
        h_regional = F.gelu(self.regional_mix(self.regional_pool(h_fine)))  # (B, H, P_regional)
        regional_patch = patch * cfg.regional_pool_factor
        anchor_regional = u.reshape(B, cfg.regional_n_sites, regional_patch).mean(dim=-1)
        if self.enc_out_regional is not None:
            residual_regional = self.enc_out_regional(h_regional)
            z_regional = torch.cat([anchor_regional.unsqueeze(1), residual_regional], dim=1)
        else:
            z_regional = anchor_regional.unsqueeze(1)
        z_regional_flat = z_regional.transpose(1, 2).reshape(B, cfg.regional_n_sites * cfg.regional_local_channels)

        # Global.
        h_global = self.global_pool(h_regional)  # (B, H, 1)
        global_patch = regional_patch * cfg.global_pool_factor
        anchor_global = u.reshape(B, cfg.global_n_sites, global_patch).mean(dim=-1)  # (B, 1)
        if self.enc_out_global is not None:
            residual_global = self.enc_out_global(h_global)  # (B, c_global-1, 1)
            z_global = torch.cat([anchor_global.unsqueeze(1), residual_global], dim=1)
        else:
            z_global = anchor_global.unsqueeze(1)
        z_global_flat = z_global.transpose(1, 2).reshape(B, cfg.global_n_sites * cfg.global_local_channels)

        return torch.cat([z_fine_flat, z_regional_flat, z_global_flat], dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        cfg = self.cfg
        B = z.shape[0]
        d_fine = cfg.fine_n_sites * cfg.fine_local_channels
        d_regional = cfg.regional_n_sites * cfg.regional_local_channels

        z_fine = z[:, :d_fine].reshape(B, cfg.fine_n_sites, cfg.fine_local_channels).transpose(1, 2)
        z_regional = z[:, d_fine:d_fine + d_regional].reshape(
            B, cfg.regional_n_sites, cfg.regional_local_channels
        ).transpose(1, 2)
        z_global = z[:, d_fine + d_regional:].reshape(B, cfg.global_n_sites, cfg.global_local_channels)

        h_fine = self.dec_lift_fine(z_fine)  # (B, H, P_fine)
        h_regional = self.dec_lift_regional(z_regional)  # (B, H, P_regional)
        h_regional_up = F.interpolate(h_regional, scale_factor=cfg.regional_pool_factor, mode="nearest")
        h_global = self.dec_lift_global(z_global.squeeze(1))  # (B, H)
        h_global_up = h_global.unsqueeze(-1).expand(-1, -1, cfg.fine_n_sites)  # (B, H, P_fine)

        h = F.gelu(h_fine + h_regional_up + h_global_up)
        for conv in self.dec_mix:
            h = F.gelu(conv(h))
        return self.dec_unpatchify(h).squeeze(1)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
