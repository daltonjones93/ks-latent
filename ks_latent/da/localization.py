"""Gaspari-Cohn latent-field localization (brief §15, Phase 13; Part 4.3
of `docs/LITERATURE_REVIEW_AND_FINDINGS.md`).

Distance-BASED localization (contrast with `ks_latent/da/sec.py`'s
distance-FREE methods): valid on a local latent FIELD
(`ks_latent.models.autoencoder_local_field.KSAutoencoderLocalField`,
`z` shaped `(n_sites, local_channels)`, flattened SITE-MAJOR) because
that architecture, unlike a plain flat latent vector, gives every latent
coordinate a genuine physical site on the domain. Applying this to a
flat, non-spatially-organized latent (e.g. `KSAutoencoderViT`'s
`pool="mean"` output) would be meaningless -- there is no physical
distance between coordinate 5 and coordinate 30 of a densely-pooled
vector.

**Site-major flattening, confirmed by reading `autoencoder_local_field.py`
directly** (not assumed): `z_field.transpose(1, 2).reshape(B, n_sites *
local_channels)`, i.e. for flat latent index `i`, `site(i) = i //
local_channels`, `channel(i) = i % local_channels`. `build_latent_taper_
matrix` below respects this exactly -- the taper depends only on
`site(i)`/`site(j)`, identical for every channel pair at a given
site-pair distance.
"""

from __future__ import annotations

import torch


def gaspari_cohn_taper(distance: torch.Tensor, c: float) -> torch.Tensor:
    r"""Standard piecewise-quintic Gaspari & Cohn (1999) taper function,
    "Construction of correlation functions in two and three dimensions,"
    *Q. J. R. Meteorol. Soc.* 125:723 -- the compactly-supported function
    used throughout operational ensemble DA (LETKF etc.) to build a valid
    (positive-semidefinite) correlation taper from physical distance
    alone. Support radius `2c` (`taper(distance) == 0` for `distance >=
    2c`); `c` itself is the "half-width" -- NOT the support radius.

    For `r = distance / c`:

        0 <= r <= 1:  -1/4 r^5 + 1/2 r^4 + 5/8 r^3 - 5/3 r^2 + 1
        1 <= r <= 2:   1/12 r^5 - 1/2 r^4 + 5/8 r^3 + 5/3 r^2 - 5 r + 4 - 2/(3r)
        r > 2:         0

    Vectorized over `distance` (any shape); `c` a scalar `> 0`. The
    `2/(3r)` term in the middle branch is finite everywhere it's
    evaluated (`r >= 1 > 0` on that branch, masked before division), so
    no singularity handling is needed beyond the branch mask itself.
    """
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c!r}")
    r = distance / c
    r_safe = torch.where(r > 0, r, torch.ones_like(r))  # avoid 0/0 in the unused branch below

    near = -0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - (5.0 / 3.0) * r**2 + 1.0
    far = (
        (1.0 / 12.0) * r**5 - 0.5 * r**4 + 0.625 * r**3 + (5.0 / 3.0) * r**2 - 5.0 * r + 4.0
        - (2.0 / 3.0) / r_safe
    )
    out = torch.where(r <= 1.0, near, torch.where(r <= 2.0, far, torch.zeros_like(r)))
    return out


def _circular_site_distance(n_sites: int) -> torch.Tensor:
    """`(n_sites, n_sites)` matrix of circular index distance (`min(|i-j|,
    n_sites-|i-j|)`), in SITE units -- both KS and L96 are periodic
    domains, so circular (not linear) distance is the physically correct
    metric here."""
    idx = torch.arange(n_sites)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return torch.minimum(diff, n_sites - diff).to(torch.float64)


def build_latent_taper_matrix(n_sites: int, channels: int, c: float) -> torch.Tensor:
    """`(n_sites*channels, n_sites*channels)` Gaspari-Cohn taper for a
    local latent field flattened SITE-MAJOR (`site(i) = i // channels`),
    using CIRCULAR site distance measured in SITE units (`c`/the support
    radius `2c` are also in site units -- e.g. `c=2` means the taper
    reaches zero 4 sites away). Every channel pair at a given site-pair
    gets the identical taper weight (channel identity does not affect
    physical distance); the diagonal is exactly `1.0` (self-distance
    zero, `gaspari_cohn_taper(0, c) == 1.0` by construction).

    Apply via `ks_latent.da.sec.apply_fixed_taper`/`make_fixed_taper_
    localizer` (the SAME Schur-product mechanism the SEC/fixed-empirical-
    taper localizers already use -- `ParticleFlowFilter`'s `localize_fn`
    hook does not care whether the taper came from data or from a
    distance formula, brief §15's own framing: "a Schur product on the
    ensemble covariance before it enters F")."""
    if n_sites < 1 or channels < 1:
        raise ValueError(f"n_sites and channels must be >= 1, got n_sites={n_sites!r}, channels={channels!r}")
    site_dist = _circular_site_distance(n_sites)
    site_taper = gaspari_cohn_taper(site_dist, c)
    return site_taper.repeat_interleave(channels, dim=0).repeat_interleave(channels, dim=1)
