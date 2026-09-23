"""ViT (function-space) + Fourier-MLP (frequency-space) hybrid encoder/
decoder (`encoder_kind="vit_fourier_hybrid"`), added 2026-09-04,
user-directed: "I don't think the spectral norm is worth pursuing ...
I want a Vit for function space, the same structure as section 52 plus
I want a Fourier mlp that only acts on frequency space ... let's make the
absolute value of each output of each layer sum to the same value which
is the original sum of the absolute values of the frequency coefficients
... This way we can impose a conservation law, but we don't need to use
something as complicated as the spectral norm. Then just add the output
of the ViT and the Fourier mlp."

Two parallel sub-networks, summed:
  - `vit`: an ordinary `KSAutoencoderViT` (see `ViTAutoencoderConfig`),
    operating on the raw physical field -- exactly Section 52's structure
    (`patch_size=8, n_heads=4, n_blocks=3, mlp_ratio=4, pool="mean",
    pos_encoding="linear", attn_window=4, token_window=16,
    readout="linear"`), `d_model` re-tuned to hit the size target below.
  - `fourier_encoder`/`fourier_decoder`: `ConservedFourierMLP` (see that
    class's docstring), operating PURELY on Fourier coefficients -- no
    raw-value path at all, unlike `KSAutoencoderFourierMLP`'s masked+
    Fourier split.

`encode(u) = vit.encode(u) + fourier_encoder(fourier_features(u))`,
mirrored for `decode`. Sized so the whole class's total parameter count
roughly matches Section 75's `KSAutoencoderFourierMLP` (~1M), with the
`vit`/Fourier-MLP portions commensurate (roughly equal shares) -- see
`scripts/section80_vit_fourier_hybrid.sh` for the exact sizing sweep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import ViTAutoencoderConfig, ViTFourierHybridAutoencoderConfig
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.propagator import FourierIFFTBody


def _fourier_features(x: torch.Tensor, n_modes: int) -> torch.Tensor:
    """`x`: `(B, n)`. Returns `(B, 2*n_modes)`: real then imaginary parts
    of the first `n_modes` `rfft` frequencies. Explicit `float()` upcast
    (same bfloat16-under-autocast bug class as `SpectralConv1d`/
    `logdet_barrier_loss`/etc elsewhere in this codebase -- `torch.fft.
    rfft` doesn't support bfloat16)."""
    orig_dtype = x.dtype
    x_ft = torch.fft.rfft(x.float(), dim=-1)[:, :n_modes]
    return torch.cat([x_ft.real, x_ft.imag], dim=-1).to(orig_dtype)


class ConservedFourierMLP(nn.Module):
    """Fourier-features-only MLP with an L1 "conservation law" at every
    layer instead of a spectral-norm/Lipschitz bound -- added 2026-09-04,
    user-directed: "let's make the absolute value of each output of each
    layer sum to the same value which is the original sum of the absolute
    values of the frequency coefficients (ie |c1| +|c2| + ... etc). This
    way we can impose a conservation law, but we don't need to use
    something as complicated as the spectral norm." (Spectral
    normalization was tried first for a different, `fourier_mlp`-based
    architecture -- see `FourierMLPAutoencoderConfig.nonexpansive`'s
    docstring -- and abandoned after it crippled reconstruction capacity;
    this is a deliberately much simpler alternative.)

    Every layer's raw output is rescaled by a single per-sample SCALAR
    (`target_l1 / current_l1`) so its L1 norm exactly equals the L1 norm
    of the ORIGINAL input features (computed once, before any layer
    processes them) -- unlike spectral normalization, this places no
    constraint on the WEIGHT MATRIX itself (no operator-norm bound, no
    lost capacity from restricting what each layer's linear map can do);
    the network is still completely free to reshape/redistribute the
    total mass across dimensions and directions however it likes, it just
    can't grow or shrink the TOTAL absolute magnitude from one layer's
    output to the next -- a conservation law on the L1 "mass", not a
    Lipschitz/contraction bound on the map itself.

    Purely a Fourier-coefficient network: no raw-value/masked path at
    all, unlike `KSAutoencoderFourierMLP`'s two-network split -- "a
    Fourier mlp that only acts on frequency space." No `irfft` either --
    output is used directly (summed with the ViT branch elsewhere), not
    inverse-transformed back to state space."""

    def __init__(self, in_modes: int, out_dim: int, hidden: int, n_blocks: int, eps: float = 1e-8):
        super().__init__()
        in_dim = 2 * in_modes
        self.input_proj = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.eps = eps

    def _conserve(self, h: torch.Tensor, target_l1: torch.Tensor) -> torch.Tensor:
        current_l1 = h.abs().sum(dim=-1, keepdim=True)
        return h * (target_l1 / (current_l1 + self.eps))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """`feats`: `(B, 2*in_modes)` -- rfft features of some input (see
        `_fourier_features`). Returns `(B, out_dim)`."""
        target_l1 = feats.abs().sum(dim=-1, keepdim=True)
        h = self._conserve(self.act(self.input_proj(feats)), target_l1)
        for block in self.blocks:
            h = self._conserve(self.act(block(h)), target_l1)
        return self._conserve(self.output_proj(h), target_l1)


class KSAutoencoderViTFourierHybrid(nn.Module):
    """`encode(u) = vit.encode(u) + fourier_encoder(fourier_features(u))`;
    mirror for `decode`. See module docstring for the full architecture
    and motivation.

    `cfg.fourier_kind` (see `ViTFourierHybridAutoencoderConfig`'s
    docstring) picks which class `fourier_encoder`/`fourier_decoder` are:
    `"conserved"` (default) = `ConservedFourierMLP`; `"ifft"` (added
    2026-09-04, Section 81) = `FourierIFFTBody` -- Section 75's own
    Fourier-path mechanism, used here with NO raw-value/masked path at
    all ("only using the frequency components"). Both share the same
    `forward(feats) -> output` interface, so `encode`/`decode` below
    don't need to know or care which one is active."""

    def __init__(self, cfg: ViTFourierHybridAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = KSAutoencoderViT(cfg.vit)
        self.enc_modes = min(
            cfg.enc_fno_modes if cfg.enc_fno_modes is not None else cfg.vit.NX // 2 + 1,
            cfg.vit.NX // 2 + 1,
        )
        self.dec_modes = min(
            cfg.dec_fno_modes if cfg.dec_fno_modes is not None else cfg.vit.d_latent // 2 + 1,
            cfg.vit.d_latent // 2 + 1,
        )
        if cfg.fourier_kind == "conserved":
            self.fourier_encoder = ConservedFourierMLP(
                self.enc_modes, cfg.vit.d_latent, cfg.fourier_hidden, cfg.fourier_blocks,
            )
            self.fourier_decoder = ConservedFourierMLP(
                self.dec_modes, cfg.vit.NX, cfg.fourier_hidden, cfg.fourier_blocks,
            )
        else:
            self.fourier_encoder = FourierIFFTBody(
                self.enc_modes, cfg.vit.d_latent, cfg.fourier_hidden, cfg.fourier_blocks,
                dropout=0.0, zero_init=False, out_modes=cfg.enc_out_modes,
            )
            self.fourier_decoder = FourierIFFTBody(
                self.dec_modes, cfg.vit.NX, cfg.fourier_hidden, cfg.fourier_blocks,
                dropout=0.0, zero_init=False, out_modes=cfg.dec_out_modes,
            )

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        feats = _fourier_features(u, self.enc_modes)
        return self.vit.encode(u) + self.fourier_encoder(feats)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        feats = _fourier_features(z, self.dec_modes)
        return self.vit.decode(z) + self.fourier_decoder(feats)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
