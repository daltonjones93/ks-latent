"""Hybrid Fourier+MLP encoder/decoder ("fourier_mlp" encoder kind), added
2026-09-03. See `FourierMLPAutoencoderConfig`'s docstring for the full
architecture and motivation (Section 66's discovered latent-index
periodicity, and `PropagatorConfig`'s `backbone="fourier_mlp"` winning
this session's propagator-architecture comparison). Exposes the same
`encode`/`decode`/`forward` interface as every other autoencoder class in
this package (`KSAutoencoderPatched`/`KSAutoencoderMLP`/etc) so it is a
drop-in replacement everywhere an autoencoder is passed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ks_latent.config import FourierMLPAutoencoderConfig
from ks_latent.models.autoencoder_masked_mlp import MaskedLinearRect
from ks_latent.models.propagator import FourierIFFTBody, MaskedMLPResidualBlock, MLPDeltaBody


def _fourier_features(x: torch.Tensor, n_modes: int, nonexpansive: bool = False) -> torch.Tensor:
    """`x`: `(B, n)`. Returns `(B, 2*n_modes)`: real then imaginary parts
    of the first `n_modes` `rfft` frequencies. Explicit `float()` upcast
    (same bfloat16-under-autocast bug class as `SpectralConv1d`/
    `logdet_barrier_loss`/`_FourierMLPHistoryDeltaBody`'s own feature
    computation -- `torch.fft.rfft` doesn't support bfloat16).
    `nonexpansive` (added 2026-09-04): uses `norm="ortho"` (an isometry by
    Parseval's theorem) instead of the default "backward" normalization,
    which is NOT norm-preserving -- see `FourierIFFTBody`'s matching
    `irfft` docstring."""
    orig_dtype = x.dtype
    norm = "ortho" if nonexpansive else None
    x_ft = torch.fft.rfft(x.float(), dim=-1, norm=norm)[:, :n_modes]
    return torch.cat([x_ft.real, x_ft.imag], dim=-1).to(orig_dtype)


def _inverse_fourier_features(z: torch.Tensor, out_len: int) -> torch.Tensor:
    """Mirror of `_fourier_features`, inverting instead of transforming --
    see `FourierMLPAutoencoderConfig.dec_use_ifft`'s docstring. `z`:
    `(B, d_latent)`, treated as the non-negative-frequency half of a
    Hermitian spectrum (imaginary part zero). Returns `(B, out_len)`: the
    `irfft`-reconstructed physical-domain signal. Explicit complex-cast +
    `float()` upcast (same bfloat16-under-autocast bug class as
    `_fourier_features` -- `torch.fft.irfft` doesn't support bfloat16 and
    requires a complex input)."""
    orig_dtype = z.dtype
    z32 = z.float()
    z_complex = torch.complex(z32, torch.zeros_like(z32))
    out = torch.fft.irfft(z_complex, n=out_len, dim=-1)
    return out.to(orig_dtype)


class _MaskedRectPath(nn.Module):
    """Masked raw-value path shared by the encoder (`NX -> d_latent`) and
    decoder (`d_latent -> NX`) -- see `FourierMLPAutoencoderConfig.
    attn_window`'s docstring. One rectangular circular-band-masked layer
    (`MaskedLinearRect`, `ref_dim=d_latent` -- same convention as
    `ViTAutoencoderConfig.pool="banded"`) to cross between the two
    differently-sized rings, then `n_blocks` DIMENSION-PRESERVING square
    masked residual blocks (`MaskedMLPResidualBlock`/`MaskedLinear`, the
    same machinery `backbone="masked_mlp"` uses) once both sides of the
    computation live in `d_latent`-sized space, exactly mirroring
    `_MaskedMLPDeltaBody`'s own input_proj/blocks/output_proj shape.
    `reverse=True` puts the rectangular crossing LAST instead of FIRST
    (for the decoder, which starts already in `d_latent` space and needs
    to expand out to `NX`).

    `nonexpansive` (added 2026-09-04, see `ks_latent.models.propagator.
    ResidualMLPBlock`'s docstring): threaded into `cross`/`blocks`
    (spectral normalization + damped residual). Composition of non-
    expansive maps is itself non-expansive, so no further change is
    needed here beyond passing the flag through."""

    def __init__(
        self, in_dim: int, out_dim: int, d_latent: int, window: int, n_blocks: int,
        dropout: float, reverse: bool, nonexpansive: bool = False,
    ):
        super().__init__()
        self.reverse = reverse
        if reverse:
            self.blocks = nn.ModuleList(
                [MaskedMLPResidualBlock(in_dim, window, dropout, nonexpansive=nonexpansive) for _ in range(n_blocks)]
            )
            self.cross = MaskedLinearRect(in_dim, out_dim, window, ref_dim=d_latent, nonexpansive=nonexpansive)
        else:
            self.cross = MaskedLinearRect(in_dim, out_dim, window, ref_dim=d_latent, nonexpansive=nonexpansive)
            self.blocks = nn.ModuleList(
                [MaskedMLPResidualBlock(out_dim, window, dropout, nonexpansive=nonexpansive) for _ in range(n_blocks)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            for block in self.blocks:
                x = block(x)
            return self.cross(x)
        x = self.cross(x)
        for block in self.blocks:
            x = block(x)
        return x


class KSAutoencoderFourierMLP(nn.Module):
    """`u` (`NX` raw values) + Fourier features -> `MLPDeltaBody` -> `z`;
    mirror for `decode`. See module/`FourierMLPAutoencoderConfig`
    docstrings. `cfg.attn_window` set: splits into a masked raw-value path
    (`_MaskedRectPath`) SUMMED with a dense Fourier-coefficient path,
    instead of one dense body over the concatenation of both.

    `cfg.nonexpansive` (added 2026-09-04, see `ks_latent.models.
    propagator.ResidualMLPBlock`'s docstring for the full mechanism):
    threaded into every sub-network (`_MaskedRectPath`/`FourierIFFTBody`)
    AND changes the masked+Fourier SUM in `encode`/`decode` to an AVERAGE
    -- summing two <=1-Lipschitz branches only guarantees <=2-Lipschitz;
    averaging keeps the combined map itself <=1-Lipschitz."""

    def __init__(self, cfg: FourierMLPAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.enc_modes = (
            cfg.enc_fno_modes if cfg.enc_fno_modes is not None else cfg.NX // 2 + 1
        )
        self.dec_modes = (
            cfg.dec_fno_modes if cfg.dec_fno_modes is not None else cfg.d_latent // 2 + 1
        )
        self.enc_modes = min(self.enc_modes, cfg.NX // 2 + 1)
        self.dec_modes = min(self.dec_modes, cfg.d_latent // 2 + 1)

        # Encoder: masked iff attn_window is set (unaffected by dec_use_ifft).
        # fourier_ifft_readout (highest priority) swaps the Fourier-only
        # sub-network for a FourierIFFTBody -- see FourierMLPAutoencoderConfig's
        # docstring -- keeping the masked raw path (if attn_window is set)
        # exactly as before, untouched and non-interacting.
        if cfg.fourier_ifft_readout:
            self.encoder = None
            if cfg.attn_window is None:
                self.encoder_masked = None
            else:
                self.encoder_masked = _MaskedRectPath(
                    cfg.NX, cfg.d_latent, cfg.d_latent, cfg.attn_window, cfg.n_blocks, cfg.dropout,
                    reverse=False, nonexpansive=cfg.nonexpansive,
                )
            self.encoder_fourier = FourierIFFTBody(
                self.enc_modes, cfg.d_latent, cfg.hidden, cfg.n_blocks, cfg.dropout, zero_init=False,
                nonexpansive=cfg.nonexpansive,
            )
        elif cfg.attn_window is None:
            self.encoder = MLPDeltaBody(
                cfg.NX + 2 * self.enc_modes, cfg.d_latent, cfg.hidden, cfg.n_blocks,
                cfg.dropout, zero_init=False, nonexpansive=cfg.nonexpansive,
            )
            self.encoder_masked = None
            self.encoder_fourier = None
        else:
            self.encoder = None
            self.encoder_masked = _MaskedRectPath(
                cfg.NX, cfg.d_latent, cfg.d_latent, cfg.attn_window, cfg.n_blocks, cfg.dropout,
                reverse=False, nonexpansive=cfg.nonexpansive,
            )
            self.encoder_fourier = MLPDeltaBody(
                2 * self.enc_modes, cfg.d_latent, cfg.hidden, cfg.n_blocks, cfg.dropout, zero_init=False,
                nonexpansive=cfg.nonexpansive,
            )

        # Decoder: fourier_ifft_readout (highest priority) gives the decoder
        # the SAME two-network structure as the encoder -- a masked
        # raw-value path (_MaskedRectPath, at dec_attn_window if given, else
        # falling back to attn_window) SUMMED with a FourierIFFTBody,
        # ignoring dec_use_ifft entirely for the decoder. `dec_attn_window`
        # is independent of the encoder's `attn_window` (added 2026-09-03,
        # user-directed: "I want the propagator and the decoder to have the
        # same structure as the encoder... but I want the attn_window to be
        # much larger" -- corrects this class's initial fourier_ifft_readout
        # implementation, which wrongly dropped the decoder's/propagator's
        # raw-value path entirely instead of just widening its window).
        # Otherwise: dec_use_ifft forces the fully-dense ifft-featurized
        # body regardless of attn_window -- otherwise masked iff attn_window
        # is set, same as the encoder.
        if cfg.fourier_ifft_readout:
            self.decoder = None
            dec_window = cfg.dec_attn_window if cfg.dec_attn_window is not None else cfg.attn_window
            if dec_window is None:
                self.decoder_masked = None
            else:
                self.decoder_masked = _MaskedRectPath(
                    cfg.d_latent, cfg.NX, cfg.d_latent, dec_window, cfg.n_blocks, cfg.dropout,
                    reverse=True, nonexpansive=cfg.nonexpansive,
                )
            self.decoder_fourier = FourierIFFTBody(
                self.dec_modes, cfg.NX, cfg.hidden, cfg.n_blocks, cfg.dropout, zero_init=False,
                nonexpansive=cfg.nonexpansive,
            )
        elif cfg.dec_use_ifft:
            self.decoder = MLPDeltaBody(
                cfg.d_latent + cfg.NX, cfg.NX, cfg.hidden, cfg.n_blocks, cfg.dropout, zero_init=False,
            )
            self.decoder_masked = None
            self.decoder_fourier = None
        elif cfg.attn_window is None:
            self.decoder = MLPDeltaBody(
                cfg.d_latent + 2 * self.dec_modes, cfg.NX, cfg.hidden, cfg.n_blocks,
                cfg.dropout, zero_init=False, nonexpansive=cfg.nonexpansive,
            )
            self.decoder_masked = None
            self.decoder_fourier = None
        else:
            self.decoder = None
            self.decoder_masked = _MaskedRectPath(
                cfg.d_latent, cfg.NX, cfg.d_latent, cfg.attn_window, cfg.n_blocks, cfg.dropout,
                reverse=True, nonexpansive=cfg.nonexpansive,
            )
            self.decoder_fourier = MLPDeltaBody(
                2 * self.dec_modes, cfg.NX, cfg.hidden, cfg.n_blocks, cfg.dropout, zero_init=False,
                nonexpansive=cfg.nonexpansive,
            )

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        if self.cfg.fourier_ifft_readout:
            feats = _fourier_features(u, self.enc_modes, nonexpansive=self.cfg.nonexpansive)
            out = self.encoder_fourier(feats)
            if self.encoder_masked is not None:
                masked_out = self.encoder_masked(u)
                out = 0.5 * (out + masked_out) if self.cfg.nonexpansive else out + masked_out
            return out
        feats = _fourier_features(u, self.enc_modes)
        if self.cfg.attn_window is None:
            return self.encoder(torch.cat([u, feats], dim=-1))
        return self.encoder_masked(u) + self.encoder_fourier(feats)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        if self.cfg.fourier_ifft_readout:
            feats = _fourier_features(z, self.dec_modes, nonexpansive=self.cfg.nonexpansive)
            out = self.decoder_fourier(feats)
            if self.decoder_masked is not None:
                masked_out = self.decoder_masked(z)
                out = 0.5 * (out + masked_out) if self.cfg.nonexpansive else out + masked_out
            return out
        if self.cfg.dec_use_ifft:
            ifft_feat = _inverse_fourier_features(z, self.cfg.NX)
            return self.decoder(torch.cat([z, ifft_feat], dim=-1))
        feats = _fourier_features(z, self.dec_modes)
        if self.cfg.attn_window is None:
            return self.decoder(torch.cat([z, feats], dim=-1))
        return self.decoder_masked(z) + self.decoder_fourier(feats)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
