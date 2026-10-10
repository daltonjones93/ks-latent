"""Manifold-gated propagators (task I31, user idea, 2026-10-09).

Two experts and a per-site soft gate:

    z' = g * M_on(z) + (1 - g) * M_off(z),  per site (all c channels of a site share its gate)

where g is near 1 for sites close to the autoencoder's data manifold and near 0 far from it.
"Off" is measured by the per-site round-trip distance

    d_s(z) = rms over the site's c channels of [E(D(z)) - z]_s,

which is small for encodings of true states (it is not exactly 0: E(D(E(u))) != E(u) in general)
and grows as z leaves the manifold. The gate is g_s = sigmoid((d0 - d_s) / w), with d0 and w set
from the distribution of d_s over clean encodings of an independent trajectory (never the DA
truth): d0 = a high quantile (default 0.95), w = a fraction of the clean spread.

M_off is either a second propagator (the I30(d) noise-fine-tuned one) or "project then M_on":
M_on(E(D(z))), which needs no second model. The gate is local (per site, from the site's own
round-trip, which with a per-patch decoder and per-patch encoder depends only on that site's
latent), so the gated model stays local and transfers to any lattice size.

Wraps propagators with the markovian `.step(z_prev, z_curr)` interface used by
`ks_latent.da.forecast.forecast_ensemble`.
"""
from __future__ import annotations

from typing import Callable

import torch


def site_roundtrip_distance(z: torch.Tensor, encode: Callable, decode: Callable, n_sites: int) -> torch.Tensor:
    """(B, n_sites * c) -> (B, n_sites): rms over each site's channels of E(D(z)) - z."""
    with torch.no_grad():
        r = encode(decode(z)) - z
    return r.reshape(z.shape[0], n_sites, -1).pow(2).mean(-1).sqrt()


class GatedPropagator:
    """`on`, `off`: objects with `.step(z_prev, z_curr)`; `off=None` means "project then `on`".
    `d0`, `width`: gate threshold and width (latent units). Records the mean gate of the last
    call in `last_mean_gate` (a diagnostic: the fraction of sites routed to `on`)."""

    mode = "markovian"

    def __init__(self, on, off, encode: Callable, decode: Callable, n_sites: int, d0: float, width: float):
        if width <= 0:
            raise ValueError(f"gate width must be > 0, got {width}")
        self.on, self.off, self.encode, self.decode = on, off, encode, decode
        self.n_sites, self.d0, self.width = n_sites, d0, width
        self.last_mean_gate = float("nan")

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        d = site_roundtrip_distance(z, self.encode, self.decode, self.n_sites)
        return torch.sigmoid((self.d0 - d) / self.width)  # (B, n_sites)

    def step(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = self.gate(z_curr)
            self.last_mean_gate = float(g.mean())
            z_on = self.on.step(z_prev, z_curr)
            if self.off is None:
                zp = self.encode(self.decode(z_curr))
                z_off = self.on.step(zp, zp)
            else:
                z_off = self.off.step(z_prev, z_curr)
            B = z_curr.shape[0]
            g3 = g.unsqueeze(-1)  # (B, n_sites, 1)
            out = g3 * z_on.reshape(B, self.n_sites, -1) + (1 - g3) * z_off.reshape(B, self.n_sites, -1)
            return out.reshape(B, -1)


def calibrate_gate(z_clean: torch.Tensor, encode: Callable, decode: Callable, n_sites: int,
                   quantile: float = 0.95, width_frac: float = 0.25) -> tuple[float, float]:
    """Threshold d0 = `quantile` of the clean per-site round-trip distance; width = `width_frac`
    x its standard deviation."""
    d = site_roundtrip_distance(z_clean, encode, decode, n_sites).reshape(-1)
    d0 = float(torch.quantile(d, quantile))
    w = float(d.std()) * width_frac
    return d0, max(w, 1e-8)
