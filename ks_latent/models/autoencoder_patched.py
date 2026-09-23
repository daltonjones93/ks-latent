"""Stage-1 patched-transformer autoencoder (brief §5.1, PROJECT_HANDOFF.md).

Encoder: PatchEmbed MLP -> local transformer shared across groups -> mean-pool
-> prepend learned query tokens -> global transformer -> keep query outputs
-> Linear to latent.

Decoder: learned query bank + broadcast-add of the latent -> transformer ->
PatchUnembed Linear back to the field.

Only `PatchEmbed` is an MLP (with a GELU nonlinearity); `enc_to_latent`,
`dec_from_latent`, and `PatchUnembed` are single `nn.Linear` layers (brief
§5.1: "Only the patch embedding is an MLP").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import AutoencoderConfig


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.mlp(patches)


def _transformer_encoder(cfg: AutoencoderConfig, num_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_ff,
        dropout=cfg.dropout,
        activation="gelu",
        norm_first=True,
        batch_first=True,
    )
    # enable_nested_tensor's fast path is unavailable with norm_first=True
    # anyway; disabling it explicitly avoids a spurious warning on every
    # forward pass rather than silently falling back each time.
    return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)


class KSAutoencoderPatched(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.patch_embed_hidden, cfg.d_model)

        self.local_transformer = _transformer_encoder(cfg, cfg.n_local_layers)
        self.global_transformer = _transformer_encoder(cfg, cfg.n_global_layers)
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.n_query_tokens, cfg.d_model) * 0.02)
        self.enc_to_latent = nn.Linear(cfg.n_query_tokens * cfg.d_model, cfg.d_latent)

        self.dec_query_bank = nn.Parameter(torch.randn(1, cfg.n_patches, cfg.d_model) * 0.02)
        self.dec_from_latent = nn.Linear(cfg.d_latent, cfg.d_model)
        self.decoder_transformer = _transformer_encoder(cfg, cfg.n_global_layers)
        self.patch_unembed = nn.Linear(cfg.d_model, cfg.patch_size)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        cfg = self.cfg
        B = u.shape[0]
        patches = u.view(B, cfg.n_patches, cfg.patch_size)
        tokens = self.patch_embed(patches)  # (B, n_patches, d_model)

        # Local transformer within each group of `group_size` consecutive
        # patches, weight-shared across all n_groups groups by folding the
        # group axis into the batch axis rather than instantiating a
        # separate module per group.
        grouped = tokens.view(B, cfg.n_groups, cfg.group_size, cfg.d_model)
        grouped = grouped.reshape(B * cfg.n_groups, cfg.group_size, cfg.d_model)
        grouped = self.local_transformer(grouped)
        grouped = grouped.view(B, cfg.n_groups, cfg.group_size, cfg.d_model)
        group_tokens = grouped.mean(dim=2)  # (B, n_groups, d_model)

        query = self.query_tokens.expand(B, -1, -1)
        seq = torch.cat([query, group_tokens], dim=1)  # (B, n_query + n_groups, d_model)
        out = self.global_transformer(seq)
        query_out = out[:, : cfg.n_query_tokens, :]  # (B, n_query, d_model)
        z = self.enc_to_latent(query_out.reshape(B, -1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        cfg = self.cfg
        B = z.shape[0]
        bank = self.dec_query_bank.expand(B, -1, -1)  # (B, n_patches, d_model)
        z_proj = self.dec_from_latent(z).unsqueeze(1)  # (B, 1, d_model)
        tokens = bank + z_proj
        out = self.decoder_transformer(tokens)
        patches = self.patch_unembed(out)  # (B, n_patches, patch_size)
        return patches.reshape(B, cfg.NX)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
