"""Jacobian colouring for zero-decoder-reach 2D local-field models
(`docs/ks_local_field_ltransfer.tex` sec:c19_da_hypotheses, test T1;
Curtis-Powell-Reid graph colouring, as applied to sparse Jacobians in
general -- see e.g. Gebremedhin, Manne & Pothen 2005 for the general
method this specializes).

**The exploited structure.** `KSAutoencoderLocalField2D.decode` with
`dec_site_mix_radius == 0` or `dec_n_site_mix_layers == 0` (zero decoder
reach, C19/Phase 0, `docs/ks_local_field_ltransfer.tex`
sec:phase0_results) has a decoder Jacobian `J = d(decode)/dz` that is
EXACTLY block-diagonal: physical pixel `p` depends only on the `channels`
latent coordinates of the one site `s(p)` owning its patch (confirmed by
`tests/unit/test_autoencoder_local_field_2d.py
::test_decode_jacobian_is_exactly_block_diagonal_at_zero_reach`). Every
site's block occupies disjoint rows of `J` (disjoint physical pixels),
so columns that share a "colour" -- here, latent channel index `k`,
shared across every site -- never have a nonzero entry in the same row.
That means a single directional derivative (`jvp`) with the tangent
"one-hot in channel `k` at every site simultaneously" recovers, in one
pass, the *exact* value of every nonzero entry at colour `k` across the
*entire* domain: for output row `p` owned by site `s(p)`, the only
nonzero contribution to that directional derivative is
`J[p, s(p)*channels + k]` itself (every other site's channel-`k`
tangent component multiplies an exactly-zero Jacobian entry at row `p`,
by block-diagonality). `channels` such passes (6 for this project's
canonical autoencoder) therefore recover the *complete* Jacobian of the
observation operator `h = obs_op o decode`, however large the domain --
replacing `ks_latent.da.letkf._jacobian_at`'s `jacrev(h)`, which costs
one reverse-mode pass **per observation** (`m`, which grows with domain
area) via `torch.vmap` internally.

This module is only valid under the zero-decoder-reach precondition
above; `assert_decoder_zero_reach` checks it the same way
`ks_latent.training.crop._resize_decoder_for_spectral_reg` already does.
Applying it to a nonzero-reach decoder silently produces a WRONG
Jacobian (columns outside each row's "true" site would alias into the
wrong colour's contribution, rather than raising), so every public
entry point here requires the precondition to have been checked by the
caller.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.func import jvp

ObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (d,) -> (m,), float64


def assert_decoder_zero_reach(ae_cfg) -> None:
    """Same precondition `ks_latent.training.crop._resize_decoder_for_
    spectral_reg` already checks before trusting decoder locality: zero
    reach means either radius 0 (every mixing layer is a 1x1 conv, no
    cross-site mixing regardless of layer count) or zero layers (no
    mixing layers at all, regardless of radius). Either alone is
    sufficient; `ae_cfg` is duck-typed (works for
    `LocalFieldAutoencoderConfig2D` or any config exposing the same
    field names, falling back to the encoder's own `site_mix_radius`/
    `n_site_mix_layers` if the decoder-specific fields are absent --
    mirroring the resize helper's own fallback)."""
    dr = getattr(ae_cfg, "site_mix_radius", None) if getattr(ae_cfg, "dec_site_mix_radius", None) is None \
        else ae_cfg.dec_site_mix_radius
    dl = getattr(ae_cfg, "n_site_mix_layers", None) if getattr(ae_cfg, "dec_n_site_mix_layers", None) is None \
        else ae_cfg.dec_n_site_mix_layers
    if not (dr == 0 or dl == 0):
        raise ValueError(
            f"Jacobian colouring needs decoder reach 0 (dec_site_mix_radius={dr}, "
            f"dec_n_site_mix_layers={dl}); a nonzero-reach decoder would make the "
            "coloured Jacobian silently WRONG (channel-colour contributions from "
            "different sites would alias into the same output row), not merely slow."
        )


def obs_site_map_2d(
    NX: int, NY: int, n_sites_x: int, n_sites_y: int, stride_x: int, stride_y: int,
) -> torch.Tensor:
    """`(m,)` int64 site index (site-major, `x` outer / `y` inner --
    `KSAutoencoderLocalField2D`'s own convention) owning each observed
    pixel of a regular `[::stride_x, ::stride_y]` grid over an
    `(NX, NY)` field, in the exact row-major flatten order
    `scripts/run_da_pff_2d_ltransfer.py::_sparse_obs_operator_2d`
    produces (`u_phys.reshape(NX, NY)[::stride_x, ::stride_y].reshape(-1)`).
    Site ownership of pixel `(ix, iy)`: `(ix // patch_x, iy // patch_y)`,
    matching `test_decode_jacobian_is_exactly_block_diagonal_at_zero_reach`'s
    own convention exactly."""
    if NX % n_sites_x != 0 or NY % n_sites_y != 0:
        raise ValueError(f"NX={NX}/NY={NY} must be divisible by n_sites_x={n_sites_x}/n_sites_y={n_sites_y}")
    px, py = NX // n_sites_x, NY // n_sites_y
    ix = torch.arange(0, NX, stride_x)
    iy = torch.arange(0, NY, stride_y)
    sx = (ix // px).unsqueeze(1)  # (mx, 1)
    sy = (iy // py).unsqueeze(0)  # (1, my)
    site = (sx * n_sites_y + sy).reshape(-1)
    return site.to(torch.int64)


def coloured_obs_jacobian_2d_compact(
    h: ObsOperator, z_bar: torch.Tensor, site_of_obs: torch.Tensor, channels: int,
) -> torch.Tensor:
    """The COMPACT form of the coloured Jacobian: `(m, channels)`, where
    row `i` holds the (only) nonzero entries of `d h/dz` at observation
    `i`, i.e. `J_full[i, site_of_obs[i]*channels : site_of_obs[i]*channels+channels]`
    -- every other entry of that row is exactly zero by construction
    (block-diagonality), so this is a lossless, dense-free
    representation. `channels` forward-mode `jvp` calls total, each
    costing one forward pass through `h`, regardless of `m` or the
    number of sites -- the whole point of the colouring (see module
    docstring). `z_bar`: `(d,)`, float64 (CPU, per the brief's device
    policy for `jacrev`/autodiff-adjacent ops -- `jvp` is forward-mode
    but kept on the same device/dtype for consistency with
    `ks_latent.da.letkf._jacobian_at`)."""
    d = z_bar.shape[0]
    if d % channels != 0:
        raise ValueError(f"z_bar has {d} coords, not divisible by channels={channels}")
    m = site_of_obs.shape[0]
    J_compact = torch.zeros(m, channels, dtype=z_bar.dtype)
    idx = torch.arange(m)
    for k in range(channels):
        v = torch.zeros(d, dtype=z_bar.dtype)
        v[k::channels] = 1.0  # channel k at every site (site-major: flat index = site*channels + channel)
        _, jk = jvp(h, (z_bar,), (v,))  # jk: (m,)
        J_compact[idx, k] = jk
    return J_compact


def coloured_obs_jacobian_2d(
    h: ObsOperator, z_bar: torch.Tensor, site_of_obs: torch.Tensor, channels: int,
) -> torch.Tensor:
    """Dense `(m, d)` reconstruction of the coloured Jacobian -- a thin
    wrapper around `coloured_obs_jacobian_2d_compact` that scatters the
    compact form into a full (mostly-zero) matrix. Provided for
    equality-testing against `ks_latent.da.letkf._jacobian_at`'s
    `jacrev` (brief's ground rule 1: no untested numerics) and for
    drop-in use anywhere a dense `(m, d)` Jacobian is expected (e.g.
    `ks_latent.da.letkf.LETKF`, by constructing an `h` wrapper that
    calls this instead of `jacrev`). For the scalable per-site filter
    (`ks_latent.da.letkf_local`), use the compact form directly --
    materializing this dense matrix costs `O(m*d)` memory, exactly the
    cost this colouring is meant to let DA avoid at the covariance/gain
    level."""
    d = z_bar.shape[0]
    J_compact = coloured_obs_jacobian_2d_compact(h, z_bar, site_of_obs, channels)
    m = site_of_obs.shape[0]
    J = torch.zeros(m, d, dtype=z_bar.dtype)
    idx = torch.arange(m)
    cols = site_of_obs.unsqueeze(1) * channels + torch.arange(channels).unsqueeze(0)  # (m, channels)
    J[idx.unsqueeze(1), cols] = J_compact
    return J
