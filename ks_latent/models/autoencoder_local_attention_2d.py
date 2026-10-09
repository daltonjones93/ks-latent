"""Local-attention variant of `KSAutoencoderLocalField2D`
(`encoder_kind="local_attention_2d"`, added 2026-09-29, user-directed:
"please implement the local attention variant you described above" --
`CLAUDE_CODE_BRIEF.md` Sec. 12.2.2's "local attention variant" of the
local field architecture, built for the first time here). ISOTROPIC --
returns to `LocalFieldAutoencoderConfig2D`'s own plain setup (the
propagator that collapsed under G2b was the phase-conditioned
anisotropic one; this is an orthogonal question -- does a more
expressive LOCAL site-mixing operator narrow Kolmogorov flow's transfer
gap, independent of the drag/extensivity question the `alpha` increase
addresses).

**Same overall shape as `KSAutoencoderLocalField2D`** (patchify ->
`n_site_mix_layers` local-mixing layers -> project to channels; embed
channels -> mixing layers -> unpatchify), same gauge anchor (channel 0 =
non-learned physical patch average), same flat `(B, NX*NY)` public
interface. The ONLY structural change: site-mixing is windowed
multi-head self-attention with a RELATIVE position bias
(`_WindowedAttnBlock2D`), not a circular `Conv2d`.

**Why this transfers with zero retraining, exactly like the conv
version.** `CLAUDE_CODE_BRIEF.md` Sec. 12.2.2's own "Translation
equivariance is not automatic here" requirements, satisfied here:

1. Every learned weight (`Q`/`K`/`V`/output projections, the FFN, the
   relative-bias table) has a shape depending only on `d_model`/
   `n_heads`/`window`/`mlp_ratio` -- NEVER on `n_sites_x`/`n_sites_y`.
   The bias table in particular is `(n_heads, window+1, window+1)`,
   indexed by the (per-axis, circular) OFFSET between two sites, not by
   either site's absolute index, so it never grows with the site grid.
2. Attention is masked to a fixed SQUARE window (Chebyshev: per-axis
   circular distance `<= window` on BOTH axes -- the same
   `(2*window+1, 2*window+1)` neighborhood a `Conv2d` kernel of that
   radius covers), never global. This is what keeps the receptive field
   bounded in PHYSICAL units as the domain grows, instead of
   automatically spanning the whole (now larger) domain the way
   un-windowed/global attention would (see `KSAutoencoderViT`'s own
   global attention, which does NOT have this property).
3. The SAME block (identical weights) is applied at every site -- no
   per-site parameters anywhere.

The structural buffers that encode the fixed window (`mask`/`dx_idx`/
`dy_idx` in `_WindowedAttnBlock2D`) are registered with
`persistent=False` specifically so they are excluded from the
checkpoint's `state_dict` and always recomputed fresh from whatever
`n_sites_x`/`n_sites_y` the CURRENT model instance was constructed
with -- this is what lets `load_state_dict(strict=True)` succeed when
loading trained weights into a differently-sized model
(`load_autoencoder_checkpoint_resized_2d_attn`,
`ks_latent/models/__init__.py`): the buffers would otherwise carry a
stale, wrongly-shaped `(N_old, N_old)` mask/index tensor that could
never match a resized model's `(N_new, N_new)` slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import LocalAttentionAutoencoderConfig2D


class _WindowedAttnBlock2D(nn.Module):
    """One windowed multi-head self-attention + FFN block over a
    `(n_sites_x, n_sites_y)` token grid, flattened site-major
    (`t = ix * n_sites_y + iy`, matching `KSAutoencoderLocalField2D`'s
    own site-major flatten convention).

    `bias_table[h, dx, dy]` is a learned scalar added to every
    query-key attention logit whose two sites are `dx` sites apart
    (circular, x-axis) and `dy` sites apart (circular, y-axis) --
    the direct attention analogue of a `Conv2d` kernel's own per-offset
    weight, applied identically at every query position. Entries with
    `dx > window` or `dy > window` are masked to `-inf` regardless of
    the (clamped, otherwise-unused) bias table value there."""

    def __init__(
        self, d_model: int, n_heads: int, window: int, mlp_ratio: int, dropout: float,
        n_sites_x: int, n_sites_y: int,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.bias_table = nn.Parameter(torch.zeros(n_heads, window + 1, window + 1))

        ix_grid, iy_grid = torch.meshgrid(
            torch.arange(n_sites_x), torch.arange(n_sites_y), indexing="ij"
        )
        ix_flat = ix_grid.reshape(-1)
        iy_flat = iy_grid.reshape(-1)
        dx = (ix_flat.unsqueeze(1) - ix_flat.unsqueeze(0)).abs()
        dx = torch.minimum(dx, n_sites_x - dx)
        dy = (iy_flat.unsqueeze(1) - iy_flat.unsqueeze(0)).abs()
        dy = torch.minimum(dy, n_sites_y - dy)
        mask = (dx <= window) & (dy <= window)
        dx_idx = torch.clamp(dx, max=window).long()
        dy_idx = torch.clamp(dy, max=window).long()
        # persistent=False: these depend only on n_sites_x/n_sites_y (fixed
        # at construction), NEVER on any learned weight -- see module
        # docstring for why they must be excluded from state_dict.
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("dx_idx", dx_idx, persistent=False)
        self.register_buffer("dy_idx", dy_idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, n_sites_x*n_sites_y, d_model)`."""
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, N, head_dim)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.bias_table[:, self.dx_idx, self.dy_idx]  # (n_heads, N, N)
        logits = logits + bias.unsqueeze(0)
        logits = logits.masked_fill(~self.mask.view(1, 1, N, N), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        x = x + self.dropout(out)
        return x + self.dropout(self.mlp(self.norm2(x)))


class KSAutoencoderLocalAttention2D(nn.Module):
    """`u: (B, NX*NY) -> z: (B, n_sites_x*n_sites_y*local_channels)`,
    identical public interface / site-major flatten convention to
    `KSAutoencoderLocalField2D`."""

    def __init__(self, cfg: LocalAttentionAutoencoderConfig2D):
        super().__init__()
        self.cfg = cfg
        px, py = cfg.patch_x, cfg.patch_y
        Px, Py = cfg.n_sites_x, cfg.n_sites_y
        d_model = cfg.d_model

        self.enc_patch_embed = nn.Linear(px * py, d_model)
        self.enc_blocks = nn.ModuleList(
            [
                _WindowedAttnBlock2D(d_model, cfg.n_heads, cfg.window, cfg.mlp_ratio, cfg.dropout, Px, Py)
                for _ in range(cfg.n_site_mix_layers)
            ]
        )
        self.enc_out = (
            nn.Linear(d_model, cfg.local_channels - 1, bias=False) if cfg.local_channels > 1 else None
        )

        self.dec_in = nn.Linear(cfg.local_channels, d_model)
        self.dec_blocks = nn.ModuleList(
            [
                _WindowedAttnBlock2D(d_model, cfg.n_heads, cfg.window, cfg.mlp_ratio, cfg.dropout, Px, Py)
                for _ in range(cfg.n_site_mix_layers)
            ]
        )
        self.dec_out = nn.Linear(d_model, px * py)

    def encode(self, u_flat: torch.Tensor) -> torch.Tensor:
        """`u_flat`: `(B, NX*NY)` -> `z`: `(B, n_sites_x*n_sites_y*local_channels)`."""
        cfg = self.cfg
        B = u_flat.shape[0]
        Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels
        px, py = cfg.patch_x, cfg.patch_y
        u = u_flat.reshape(B, cfg.NX, cfg.NY)

        patches = u.reshape(B, Px, px, Py, py).permute(0, 1, 3, 2, 4).reshape(B, Px * Py, px * py)
        h = F.gelu(self.enc_patch_embed(patches))  # (B, N, d_model)
        for block in self.enc_blocks:
            h = block(h)

        anchor = u.reshape(B, Px, px, Py, py).mean(dim=(2, 4)).reshape(B, Px * Py)  # (B, N)
        if self.enc_out is not None:
            residual = self.enc_out(h)  # (B, N, c-1)
            z_field = torch.cat([anchor.unsqueeze(-1), residual], dim=-1)  # (B, N, c)
        else:
            z_field = anchor.unsqueeze(-1)

        return z_field.reshape(B, Px * Py * c)  # site-major flatten

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, n_sites_x*n_sites_y*local_channels)` -> `u_hat_flat`: `(B, NX*NY)`."""
        cfg = self.cfg
        B = z.shape[0]
        Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels
        px, py = cfg.patch_x, cfg.patch_y

        z_field = z.reshape(B, Px * Py, c)
        h = F.gelu(self.dec_in(z_field))
        for block in self.dec_blocks:
            h = block(h)
        patches = self.dec_out(h)  # (B, N, px*py)
        u_hat = patches.reshape(B, Px, Py, px, py).permute(0, 1, 3, 2, 4).reshape(B, cfg.NX, cfg.NY)
        return u_hat.reshape(B, cfg.NX * cfg.NY)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
