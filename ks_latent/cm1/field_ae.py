"""Track FP, FP1: a 2D latent field autoencoder (no patches, no per-column channels).

The latent is m fields phi_k(x, z) on the x-z plane, at the data's resolution or on a grid coarsened by
a strided convolution (overlapping kernels: no patch boundaries). Height stays a spatial coordinate, so a
law for phi can contain vertical derivatives explicitly instead of hiding them in channel mixing.

- Encoder: dilated 2D convolutions over (z, x) (receptive field ~16 km x ~34 levels: nonlocal enough for
  a streamfunction-like inversion), replicate padding (open boundaries), height features as extra inputs,
  then an optional strided convolution to the latent grid and a 1x1 head to m channels.
- Decoder: "jet" -- a pointwise MLP of phi and its first and second finite differences (phi, phi_x, phi_z,
  phi_xx, phi_xz, phi_zz) and height features; "point" -- of phi and height only. The jet decoder is
  local in phi's 2-jet, which lets few fields carry more variables (u = -psi_z, w = psi_x from one
  streamfunction); comparing the two at equal m measures that. On a coarse latent grid the jet is
  computed there and bilinearly interpolated to the data grid before the pointwise MLP (a continuous,
  neural-field-like decoder). Condensate outputs are clamped >= 0.

Data layout: normalized columns `(B, C, NX)` with C = V*Z variable-major, as `ColumnData` serves them;
`forward` returns `(reconstruction (B, C, NX), latent (B, m, Zc, Xc))`. Translation-equivariant in x for
shifts that are multiples of `stride_x`, away from the lateral boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FieldAEConfig:
    n_vars: int = 7
    n_levels: int = 62
    m: int = 4
    stride_x: int = 1
    stride_z: int = 1
    decoder: str = "jet"            # "jet" | "point"
    width: int = 64
    dilations: tuple = ((1, 1), (2, 2), (4, 4), (8, 8), (4, 16), (1, 1))   # (dz, dx) per encoder layer
    dec_hidden: int = 128
    dec_layers: int = 3
    n_zfeat: int = 4                # Fourier height features: 1 + 2 * n_zfeat channels

    def __post_init__(self):
        if self.decoder not in ("jet", "point"):
            raise ValueError(f"decoder must be 'jet' or 'point', not {self.decoder!r}")
        if self.m < 1 or self.stride_x < 1 or self.stride_z < 1:
            raise ValueError("m and strides must be >= 1")
        self.dilations = tuple(tuple(d) for d in self.dilations)

    @property
    def numbers_per_point(self) -> float:
        """Latent numbers per data grid point (x, z): m / (stride_x * stride_z)."""
        return self.m / (self.stride_x * self.stride_z)


def height_features(Z: int, n: int) -> torch.Tensor:
    zeta = torch.linspace(0, 1, Z)
    f = [zeta] + [g(torch.pi * k * zeta) for k in range(1, n + 1) for g in (torch.sin, torch.cos)]
    return torch.stack(f)                                     # (1 + 2n, Z)


def jet(phi: torch.Tensor) -> torch.Tensor:
    """(B, m, Z, X) -> (B, 6m, Z, X): phi, d_x, d_z, d_xx, d_xz, d_zz (central differences in index units,
    replicate padding)."""
    p = F.pad(phi, (1, 1, 1, 1), mode="replicate")
    c = p[..., 1:-1, 1:-1]
    dx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    dz = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    dxx = p[..., 1:-1, 2:] - 2 * c + p[..., 1:-1, :-2]
    dzz = p[..., 2:, 1:-1] - 2 * c + p[..., :-2, 1:-1]
    dxz = 0.25 * (p[..., 2:, 2:] - p[..., 2:, :-2] - p[..., :-2, 2:] + p[..., :-2, :-2])
    return torch.cat([c, dx, dz, dxx, dxz, dzz], dim=1)


class FieldAE(nn.Module):
    def __init__(self, cfg: FieldAEConfig, positive_mask: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        V, Z, W = cfg.n_vars, cfg.n_levels, cfg.width
        self.register_buffer("zfeat", height_features(Z, cfg.n_zfeat))
        nf = self.zfeat.shape[0]
        layers, cin = [], V + nf
        for dz, dx in cfg.dilations:
            layers += [nn.Conv2d(cin, W, 3, padding=(dz, dx), dilation=(dz, dx), padding_mode="replicate"), nn.GELU()]
            cin = W
        self.enc = nn.Sequential(*layers)
        sz, sx = cfg.stride_z, cfg.stride_x
        self.down = (nn.Conv2d(W, W, (2 * sz + 1, 2 * sx + 1), stride=(sz, sx), padding=(sz, sx),
                               padding_mode="replicate") if (sz, sx) != (1, 1) else nn.Identity())
        self.head = nn.Conv2d(W, cfg.m, 1)
        nin = (6 * cfg.m if cfg.decoder == "jet" else cfg.m) + nf
        dec, c = [], nin
        for _ in range(cfg.dec_layers - 1):
            dec += [nn.Conv2d(c, cfg.dec_hidden, 1), nn.GELU()]
            c = cfg.dec_hidden
        dec.append(nn.Conv2d(c, V, 1))
        self.dec = nn.Sequential(*dec)
        pm = positive_mask.reshape(V, Z)[:, 0].clone()
        if not torch.equal(positive_mask.reshape(V, Z), pm[:, None].expand(V, Z)):
            raise ValueError("positive mask must be per variable (all levels alike)")
        self.register_buffer("pos", pm.view(1, V, 1, 1))

    def _z(self, B: int, X: int, Z: int | None = None) -> torch.Tensor:
        zf = self.zfeat if Z is None else F.interpolate(self.zfeat[None], size=Z, mode="linear", align_corners=True)[0]
        return zf[None, :, :, None].expand(B, -1, -1, X)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        B, C, X = u.shape
        V, Z = self.cfg.n_vars, self.cfg.n_levels
        x = u.reshape(B, V, Z, X)
        h = self.enc(torch.cat([x, self._z(B, X)], dim=1))
        return self.head(self.down(h))

    def decode(self, phi: torch.Tensor, X: int) -> torch.Tensor:
        B = phi.shape[0]
        V, Z = self.cfg.n_vars, self.cfg.n_levels
        feat = jet(phi) if self.cfg.decoder == "jet" else phi
        if feat.shape[-2:] != (Z, X):
            sz, sx = self.cfg.stride_z, self.cfg.stride_x
            feat = F.interpolate(feat, size=(feat.shape[-2] * sz, feat.shape[-1] * sx), mode="bilinear",
                                 align_corners=False)[..., :Z, :X]
        y = self.dec(torch.cat([feat, self._z(B, X)], dim=1))
        y = torch.where(self.pos, F.relu(y), y)
        return y.reshape(B, V * Z, X)

    def forward(self, u: torch.Tensor, env=None):
        phi = self.encode(u)
        return self.decode(phi, u.shape[-1]), phi
