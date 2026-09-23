"""Phase 10's original architecturally-local latent FIELD (`encoder_kind=
"local_field"`, added 2026-09-10, Section 135) -- see
`LocalFieldAutoencoderConfig`'s docstring for the full motivation: after
`masked_mlp_expand` (Sections 128-134) kept collapsing regardless of how
its Jacobian spectrum was regularized, this builds locality in from the
start via a genuine `(n_sites, local_channels)` field, with channel 0
gauge-anchored to a real physical average, rather than hoping a flat,
arbitrarily-ordered latent index happens to correspond to physical
position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import LocalFieldAutoencoderConfig


class KSAutoencoderLocalField(nn.Module):
    """`u: (B, NX) -> z: (B, n_sites*local_channels)`, `z` a genuine local
    field, flattened SITE-MAJOR (see `LocalFieldAutoencoderConfig`'s
    docstring for why). Channel 0 of the field is NEVER produced by the
    learned conv stack -- it is the exact local physical average of `u`."""

    def __init__(self, cfg: LocalFieldAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        P, c, r, layers, H = (
            cfg.n_sites, cfg.local_channels, cfg.site_mix_radius, cfg.n_site_mix_layers, cfg.hidden
        )
        patch = cfg.patch_size

        # Encoder: exact non-overlapping patchify (no padding needed --
        # patch_size divides NX exactly), then circular site-mixing convs,
        # then a 1x1 conv to the LEARNED residual channels only (c-1 of
        # them -- channel 0 is the gauge anchor, never routed through here).
        self.enc_patchify = nn.Conv1d(1, H, kernel_size=patch, stride=patch)
        self.enc_mix = nn.ModuleList(
            [nn.Conv1d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular") for _ in range(layers)]
        )
        # `bias=False` (added 2026-09-11, user-directed: "the root fix" --
        # see this module's own docstring for the finding this addresses).
        # A per-channel Conv1d bias is a completely free, unconstrained
        # additive offset -- nothing in the existing loss (w_var/w_logdet/
        # w_decorr all care about SCALE/covariance, never about MEAN)
        # stops it from drifting to an arbitrary constant during training.
        # Measured directly on a real trained checkpoint (Section 153):
        # channel 0 (anchor, never routed through this layer) has mean
        # 0.000, but the two free residual channels had drifted to means
        # of 7.494 and 8.607 -- an offset an order of magnitude larger
        # than the real per-site physical variation (std ~0.8-1.3).
        # Because z is flattened SITE-MAJOR (period `local_channels`=3 in
        # the raw index), that offset pattern is a near-pure period-3
        # signal, which aliases onto EXACTLY one self-FFT mode
        # (`d_latent/local_channels`, here 96/3=32) -- confirmed
        # empirically: pde_head's own standalone rollout collapsed onto
        # >98% of its energy at just that one mode plus DC, for EVERY one
        # of 10 different real initial conditions tested, regardless of
        # encoder/pde_head training recipe. This is a pure flattening
        # ARTIFACT, not physical KS structure, and no amount of pde-side
        # regularization (L1-sparsity, forcing the pde's own constant term
        # to zero, etc.) can fix it, since it lives in the ENCODER's own
        # representation, upstream of anything the pde_head sees.
        #
        # A first attempt at this fix subtracted each residual channel's
        # own per-sample mean ACROSS SITES -- a real bug, caught
        # immediately by test_receptive_field_bounded_and_matches_
        # analytic_formula: a global (all-`P`-sites) reduction makes
        # EVERY output site depend on EVERY input site, destroying the
        # encoder's core architectural locality guarantee (this whole
        # design's reason for existing over `masked_mlp_expand`). Removing
        # the LEARNABLE BIAS PARAMETER instead is purely local -- it never
        # changes which inputs affect which outputs, only removes an
        # input-INDEPENDENT additive constant the network was otherwise
        # free to learn.
        self.enc_out = nn.Conv1d(H, c - 1, kernel_size=1, bias=False) if c > 1 else None

        # Decoder: exact mirror. dec_in always takes all c channels
        # (including the anchor) -- the anchor is real information the
        # decoder is meant to use, not discarded.
        self.dec_in = nn.Conv1d(c, H, kernel_size=1)
        self.dec_mix = nn.ModuleList(
            [nn.Conv1d(H, H, kernel_size=2 * r + 1, padding=r, padding_mode="circular") for _ in range(layers)]
        )
        self.dec_unpatchify = nn.ConvTranspose1d(H, 1, kernel_size=patch, stride=patch)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, n_sites*local_channels)`, site-major."""
        cfg = self.cfg
        B = u.shape[0]
        P, c, patch = cfg.n_sites, cfg.local_channels, cfg.patch_size

        h = F.gelu(self.enc_patchify(u.unsqueeze(1)))  # (B, H, P)
        for conv in self.enc_mix:
            h = F.gelu(conv(h))

        anchor = u.reshape(B, P, patch).mean(dim=-1)  # (B, P) -- the gauge anchor, NOT learned
        if self.enc_out is not None:
            residual = self.enc_out(h)  # (B, c-1, P)
            z_field = torch.cat([anchor.unsqueeze(1), residual], dim=1)  # (B, c, P)
        else:
            z_field = anchor.unsqueeze(1)  # (B, 1, P) -- local_channels=1, pure coarse-graining

        return z_field.transpose(1, 2).reshape(B, P * c)  # site-major flatten

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, n_sites*local_channels)`, site-major -> `u_hat`: `(B, NX)`."""
        cfg = self.cfg
        B = z.shape[0]
        P, c = cfg.n_sites, cfg.local_channels

        z_field = z.reshape(B, P, c).transpose(1, 2)  # (B, c, P)
        h = F.gelu(self.dec_in(z_field))
        for conv in self.dec_mix:
            h = F.gelu(conv(h))
        return self.dec_unpatchify(h).squeeze(1)  # (B, NX)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
