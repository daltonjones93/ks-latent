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


def _circular_lattice_distance_2d(n_sites_x: int, n_sites_y: int) -> torch.Tensor:
    """`(P, P)` matrix of 2D periodic (toroidal) Euclidean site distance,
    `P = n_sites_x * n_sites_y`, for a SITE-MAJOR flattening with `x`
    outer and `y` inner (`site_x(i) = i // n_sites_y`, `site_y(i) = i %
    n_sites_y` -- the ordering `KSAutoencoderLocalField2D.encode` actually
    produces before the channel dimension is folded in, confirmed by
    reading its `z_field.permute(0, 2, 3, 1).reshape(B, Px * Py * c)`
    directly: `Px` is the outer reshape dimension). Distance in SITE
    units, circular (not linear) in each axis independently, then
    combined via the ordinary Euclidean norm -- isotropic localization,
    appropriate for 2D KS/Kolmogorov's own isotropic (or, for Kolmogorov,
    the locally-isotropic small-scale) spatial structure."""
    P = n_sites_x * n_sites_y
    idx = torch.arange(P)
    site_x = idx // n_sites_y
    site_y = idx % n_sites_y
    dx = (site_x.unsqueeze(0) - site_x.unsqueeze(1)).abs()
    dx = torch.minimum(dx, n_sites_x - dx)
    dy = (site_y.unsqueeze(0) - site_y.unsqueeze(1)).abs()
    dy = torch.minimum(dy, n_sites_y - dy)
    return torch.sqrt((dx**2 + dy**2).to(torch.float64))


def build_latent_taper_matrix_2d(n_sites_x: int, n_sites_y: int, channels: int, c: float) -> torch.Tensor:
    """2D analogue of `build_latent_taper_matrix`, for a local latent
    FIELD on a 2D site grid (`KSAutoencoderLocalField2D`,
    `(P, channels)`, `P = n_sites_x * n_sites_y`, flattened site-major
    with `x` outer, `y` inner, channel innermost -- see
    `_circular_lattice_distance_2d`'s docstring for the exact convention,
    confirmed against the encoder's own flatten line). `c` (the
    Gaspari-Cohn half-width, support radius `2c`) is in SITE units,
    Euclidean 2D distance -- e.g. `c=2` reaches zero at 2D (toroidal)
    distance `4`. Returns `(P*channels, P*channels)`; every channel pair
    at a given site-pair shares the identical taper weight (channel
    identity does not affect physical distance), diagonal exactly `1.0`.

    Why this is the needed piece for 2D latent DA (brief §15/Phase 13,
    `docs/ks_local_field_ltransfer.tex` sec:consolidation Phase 4): the
    1D `build_latent_taper_matrix` assumes a 1D ring of sites and cannot
    express 2D (toroidal) distance; this function is what
    `ks_latent.da.cycling.run_da_experiment`'s `localize_fn` needs for
    any 2D local-field model (2D KS, Kolmogorov) deployed -- at native
    size or transferred to a larger domain via
    `ks_latent.models.load_autoencoder_checkpoint_resized_2d` -- since
    `n_sites_x`/`n_sites_y` change with transfer and the taper must be
    rebuilt at the deployed size, exactly as the 1D version already is
    in `scripts/run_da_pff_ltransfer.py`."""
    if n_sites_x < 1 or n_sites_y < 1 or channels < 1:
        raise ValueError(
            f"n_sites_x, n_sites_y and channels must be >= 1, got "
            f"n_sites_x={n_sites_x!r}, n_sites_y={n_sites_y!r}, channels={channels!r}"
        )
    site_dist = _circular_lattice_distance_2d(n_sites_x, n_sites_y)
    site_taper = gaspari_cohn_taper(site_dist, c)
    return site_taper.repeat_interleave(channels, dim=0).repeat_interleave(channels, dim=1)


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
