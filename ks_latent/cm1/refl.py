"""Reflectivity observation operator (E1b): composite reflectivity from a column.

CM1's reflectivity comes from its two-moment microphysics (it needs the number concentrations, which
our state does not carry), so the operator is learned: a small per-column network maps the
normalized column (all V x Z channels at one grid cell, plus its two neighbours: kernel 3) to
composite reflectivity, the column maximum (dBZ), the quantity a radar composite gives. Output is
softplus, so >= 0 dBZ ("no echo" is 0; CM1 composite reflectivity is >= 0 in these runs).

It is trained on TRUE columns of the training members, and scored on test members both on true
columns (the operator's own error) and on decoded columns (what DA feeds it: operator + autoencoder
error). In twin experiments the observations are CM1's own composite reflectivity, so this operator's
error is a real representation error the filter must live with.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReflHeadConfig:
    n_channels: int = 7 * 62
    hidden: int = 256
    layers: int = 3
    kernel: int = 3


class ReflectivityHead(nn.Module):
    """`u` (B, C, NX) normalized columns -> composite reflectivity (B, NX), dBZ >= 0."""

    def __init__(self, cfg: ReflHeadConfig):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.n_channels] + [cfg.hidden] * (cfg.layers - 1)
        self.convs = nn.ModuleList(nn.Conv1d(a, cfg.hidden, cfg.kernel if i == 0 else 1,
                                             padding=cfg.kernel // 2 if i == 0 else 0, padding_mode="replicate")
                                   for i, a in enumerate(dims))
        self.out = nn.Conv1d(cfg.hidden, 1, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = u
        for conv in self.convs:
            h = F.gelu(conv(h))
        return 10.0 * F.softplus(self.out(h)[:, 0] / 10.0)  # soft floor at 0 dBZ, ~linear above ~10


class CompositeReflectivity:
    """Observation operator for the LETKF: composite reflectivity at chosen columns of decoded members."""

    def __init__(self, head: ReflectivityHead, columns, dx_km: float = 0.25):
        import numpy as np
        self.head = head.eval()
        self.columns = np.asarray(columns, dtype=int)
        self.x_km = (self.columns + 0.5) * dx_km

    @torch.no_grad()
    def __call__(self, ae, Z: torch.Tensor):
        return self.head(ae.decode(Z))[:, self.columns].double().cpu().numpy()
