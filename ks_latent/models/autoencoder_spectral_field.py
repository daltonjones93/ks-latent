"""`encoder_kind="spectral_field"` (added 2026-09-06, see
docs/sine_transform_pde_plan.md and `SpectralFieldAutoencoderConfig`'s
docstring for the full design/motivation).

`KSAutoencoderSpectralField` composes a field-producing inner model --
either a `KSAutoencoderViT` (`pool` in `("none", "local")`, so its own
`encode(x)` already returns a genuine spatially-indexed scalar field `w`
of length `N_w`) or, added 2026-09-09, a plain `KSAutoencoderMLP` (no
attention/tokenization/windowing at all -- every output position can
depend on every input position with no architectural locality bias),
selected via `cfg.vit`/`cfg.mlp` (exactly one is set) -- with a FIXED,
non-learned rFFT transform (`ks_latent.models.spectral_field`): `z =
encode_to_spectrum(w, K)`, `w_hat = decode_from_spectrum(z, K, N_w)`. The
transform carries no parameters of its own -- every learned parameter in
this class lives inside the wrapped inner model. Both `KSAutoencoderViT`
and `KSAutoencoderMLP` expose the identical `encode`/`decode` interface,
so `self.inner` is used generically regardless of which one was selected.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import SpectralFieldAutoencoderConfig
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.spectral_field import decode_from_spectrum, encode_to_spectrum


class KSAutoencoderSpectralField(nn.Module):
    def __init__(self, cfg: SpectralFieldAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.inner = KSAutoencoderViT(cfg.vit) if cfg.vit is not None else KSAutoencoderMLP(cfg.mlp)
        self.K = cfg.K
        self.N_w = cfg.N_w

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, 2*K)`."""
        w = self.inner.encode(u)  # (B, N_w), a genuine spatial field
        return encode_to_spectrum(w, self.K)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, 2*K)` -> `u_hat`: `(B, NX)`."""
        w_hat = decode_from_spectrum(z, self.K, self.N_w)
        return self.inner.decode(w_hat)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
