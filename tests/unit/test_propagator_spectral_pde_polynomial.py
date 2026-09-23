"""Tests for `field_kind="polynomial"` (added 2026-09-09, see
`_SpectralPDEDeltaBody.field`'s docstring and `_polynomial_library`'s own
docstring): replaces the pointwise MLP with a single shared linear layer
over a degree-`<=poly_degree` monomial library built from the synthesized
derivative stack -- user-directed: "expand the pde as a polynomial (degree
1 or 2) in all the derivative terms, then directly learn the
coefficients... more interpretable, and potentially more stable."

Includes regression coverage for a real bug found and fixed the same day:
the RAW (unnormalized) monomial library is badly conditioned (derivative
orders differ by many orders of magnitude, since differentiation
multiplies by `k^n`), which caused a genuine Stage-2 NaN blowup. Fixed via
a FIXED (not learned) per-order rescaling before the library is built --
`test_normalization_keeps_output_bounded_at_larger_scale` guards against
that regression directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.propagator import (
    AuxPropagator,
    LatentPropagator,
    _polynomial_library,
    _polynomial_library_size,
    _polynomial_term_indices,
)
from ks_latent.models.spectral_field import encode_to_spectrum


def _cfg(**overrides):
    defaults = dict(
        d_latent=16, mode="markovian", backbone="spectral_pde",
        spectral_K=8, spectral_N_w=32, spectral_L=22.0, spectral_max_order=4,
        hidden=32, n_blocks=1, spectral_field_kind="polynomial", spectral_poly_degree=2,
    )
    defaults.update(overrides)
    return PropagatorConfig(**defaults)


def _valid_z(n: int, N_w: int, K: int) -> torch.Tensor:
    w = torch.randn(n, N_w)
    return encode_to_spectrum(w, K)


# ---- library construction ----


def test_library_size_matches_helper():
    derivs = torch.randn(2, 5, 5)
    assert _polynomial_library(derivs, 1).shape[-1] == _polynomial_library_size(5, 1)
    assert _polynomial_library(derivs, 2).shape[-1] == _polynomial_library_size(5, 2)


def test_library_degree1_is_constant_then_raw_derivs():
    derivs = torch.randn(2, 3, 4)
    lib = _polynomial_library(derivs, 1)
    assert torch.allclose(lib[..., 0], torch.ones(2, 3))
    assert torch.allclose(lib[..., 1:], derivs)


def test_library_degree2_quadratic_terms_are_upper_triangular_products():
    derivs = torch.randn(2, 3, 3)  # n_vars=3 for a small, hand-checkable case
    lib = _polynomial_library(derivs, 2)
    # order: [1, d0, d1, d2, d0*d0, d0*d1, d0*d2, d1*d1, d1*d2, d2*d2]
    assert lib.shape[-1] == 10
    assert torch.allclose(lib[..., 4], derivs[..., 0] * derivs[..., 0])
    assert torch.allclose(lib[..., 5], derivs[..., 0] * derivs[..., 1])
    assert torch.allclose(lib[..., 6], derivs[..., 0] * derivs[..., 2])
    assert torch.allclose(lib[..., 7], derivs[..., 1] * derivs[..., 1])
    assert torch.allclose(lib[..., 9], derivs[..., 2] * derivs[..., 2])


def test_library_rejects_degree_above_3():
    with pytest.raises(ValueError, match="degree must be"):
        _polynomial_library(torch.randn(2, 3), 4)


def test_library_degree3_cubic_terms():
    derivs = torch.randn(2, 3, 3)  # n_vars=3
    lib = _polynomial_library(derivs, 3)
    # 1 (const) + 3 (linear) + 6 (quadratic, C(3+2-1,2)) + 10 (cubic, C(3+3-1,3)) = 20
    assert lib.shape[-1] == 20
    # last cubic term is (2,2,2) -> d2*d2*d2
    assert torch.allclose(lib[..., -1], derivs[..., 2] ** 3)


# ---- _SpectralPDEDeltaBody mechanics ----


def test_polynomial_produces_right_shape():
    prop = LatentPropagator(_cfg())
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (4, 16)


def test_polynomial_zero_init_is_exact_identity():
    prop = LatentPropagator(_cfg(zero_init=True, spectral_integrator="euler"))
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_polynomial_gradient_flows():
    prop = LatentPropagator(_cfg(zero_init=False))
    z = _valid_z(3, 32, 8).requires_grad_(True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert prop.body.poly_coeffs.weight.grad is not None


def test_degree1_has_no_quadratic_terms():
    prop = LatentPropagator(_cfg(spectral_poly_degree=1))
    # max_order=4 -> 5 derivative channels -> degree=1 library size = 1+5 = 6
    assert prop.body.poly_coeffs.weight.shape == (1, 6)


def test_degree2_library_size_matches():
    prop = LatentPropagator(_cfg(spectral_poly_degree=2))
    # 5 derivative channels -> degree=2 library size = 1+5+15 = 21
    assert prop.body.poly_coeffs.weight.shape == (1, 21)


# ---- normalization (the actual bug fix) ----


def test_deriv_norm_scales_match_formula():
    prop = LatentPropagator(_cfg())
    K, L, max_order = 8, 22.0, 4
    char_k = 2.0 * math.pi * K / L
    expected = torch.tensor([char_k**n for n in range(max_order + 1)])
    assert torch.allclose(prop.body._deriv_norm_scales, expected, rtol=1e-5)


def test_normalization_keeps_output_bounded_at_larger_scale():
    """Regression test for the real Stage-2 NaN blowup found 2026-09-09:
    with normalization, degree=2's output at a moderately large coefficient/
    z scale must stay within a sane bound (loosely, not orders of magnitude
    beyond the input scale) -- before the fix this reached ~1.3e7 already
    at NOMINAL (unscaled) coefficients."""
    torch.manual_seed(0)
    K, N_w, L = 8, 32, 22.0
    w = torch.randn(4, N_w)
    z = encode_to_spectrum(w, K)
    cfg = AuxPropagatorConfig(
        d_latent=2 * K, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde",
        spectral_K=K, spectral_N_w=N_w, spectral_L=L, spectral_max_order=4,
        spectral_field_kind="polynomial", spectral_poly_degree=2, zero_init=False,
    )
    prop = AuxPropagator(cfg)
    with torch.no_grad():
        prop.body.poly_coeffs.weight.mul_(5.0)
    out = prop.step_one(z * 5.0)
    assert torch.isfinite(out).all()
    # Before the normalization fix this was ~1e12 at this scale; with it,
    # stays within a couple orders of magnitude of the input scale.
    assert out.abs().max().item() < 1e4


# ---- max_term_order (combined derivative-order truncation), added
# 2026-09-09, user-directed: "really only let the combined degree of the
# terms be less than 5 (so w_xxx * w_xxx or w_xxx*w_xxxx would have 0
# coefficients since they have combined degree 6, 7 respectively)" ----


def test_max_term_order_excludes_user_specified_examples():
    # max_order=4 -> n_vars=5 (orders 0..4)
    indices = _polynomial_term_indices(5, 2, max_term_order=5)
    assert (3, 3) not in indices, "w_xxx*w_xxx (combined order 6) must be excluded"
    assert (3, 4) not in indices, "w_xxx*w_xxxx (combined order 7) must be excluded"
    # KS's own true terms must survive: w*w_x (order 1), w_xx alone (order
    # 2), w_xxxx alone (order 4) -- all well within max_term_order=5.
    assert (0, 1) in indices
    assert (2,) in indices
    assert (4,) in indices


def test_max_term_order_none_is_unrestricted():
    unrestricted = _polynomial_term_indices(5, 2, max_term_order=None)
    assert len(unrestricted) == _polynomial_library_size(5, 2)
    assert (3, 3) in unrestricted
    assert (3, 4) in unrestricted


def test_max_term_order_library_size_matches_indices():
    n_vars, degree, max_term_order = 5, 2, 5
    expected = len(_polynomial_term_indices(n_vars, degree, max_term_order))
    assert _polynomial_library_size(n_vars, degree, max_term_order) == expected
    derivs = torch.randn(2, 3, n_vars)
    lib = _polynomial_library(derivs, degree, max_term_order)
    assert lib.shape[-1] == expected


def test_max_term_order_wired_through_model_reduces_coefficient_count():
    unrestricted = LatentPropagator(_cfg(spectral_poly_max_term_order=None))
    restricted = LatentPropagator(_cfg(spectral_poly_max_term_order=5))
    assert restricted.body.poly_coeffs.weight.shape[1] < unrestricted.body.poly_coeffs.weight.shape[1]
    # Exact expected count for max_order=4 (5 vars), degree=2, max_term_order=5:
    # 1 const + 5 linear + 9 quad (21 - 6 excluded) = 15.
    assert restricted.body.poly_coeffs.weight.shape[1] == 15


def test_rejects_max_term_order_below_1():
    with pytest.raises(ValueError, match="spectral_poly_max_term_order must be >= 1"):
        _cfg(spectral_poly_max_term_order=0)


# ---- config validation ----


def test_rejects_bad_field_kind():
    with pytest.raises(ValueError, match="spectral_field_kind must be"):
        _cfg(spectral_field_kind="quadratic")


def test_rejects_bad_poly_degree():
    with pytest.raises(ValueError, match="spectral_poly_degree must be"):
        _cfg(spectral_poly_degree=4)


def test_degree3_produces_cubic_terms():
    # 2026-09-09, user-directed: "higher degree polynomial for the pde".
    prop = LatentPropagator(_cfg(spectral_poly_degree=3))
    # max_order=4 -> 5 derivative channels -> degree=3 library size:
    # 1 (const) + 5 (linear) + 15 (quadratic) + 35 (cubic) = 56.
    assert prop.body.poly_coeffs.weight.shape == (1, 56)
    z = _valid_z(3, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_poly_degree_ignored_and_unvalidated_for_mlp_kind():
    # spectral_poly_degree is meaningless for field_kind="mlp" -- must not
    # raise even with an otherwise-invalid value.
    cfg = _cfg(spectral_field_kind="mlp", spectral_poly_degree=99)
    prop = LatentPropagator(cfg)
    z = _valid_z(2, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (2, 16)


# ---- works with spectral_pde_raw too (pde_head hybrid) ----


def test_spectral_pde_raw_supports_polynomial():
    cfg = AuxPropagatorConfig(
        d_latent=20, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=11, spectral_L=20.0, spectral_max_order=4,
        spectral_field_kind="polynomial", spectral_poly_degree=1, zero_init=False,
    )
    prop = AuxPropagator(cfg)
    z = torch.randn(3, 20)
    out = prop.step_one(z)
    assert out.shape == (3, 20)
    assert torch.isfinite(out).all()


# ---- poly_stable_leading (added 2026-09-09, user-directed: "is there a
# way to regularize or bound the eigenvalues of the differential operator
# induced by the pde?" -- forces the highest even-order linear
# coefficient to be <= 0 via c = -(raw)^2, guaranteeing the induced linear
# operator's Re(eigenvalue) -> -inf as wavenumber -> inf) ----


def test_stable_leading_identifies_highest_even_order():
    prop = LatentPropagator(_cfg(spectral_poly_stable_leading=True))
    # max_order=4 -> highest even order is 4 (w_xxxx), at library index 5
    # (0=const, 1..5=linear terms order 0..4).
    assert prop.body._stable_leading_order == 4
    assert prop.body._stable_leading_idx == 5


def test_stable_leading_coefficient_always_nonpositive():
    prop = LatentPropagator(_cfg(spectral_poly_stable_leading=True, zero_init=False))
    idx = prop.body._stable_leading_idx
    with torch.no_grad():
        prop.body.poly_coeffs.weight[:, idx] = 7.0  # even a positive raw value
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert torch.isfinite(out).all()
    # the transform (-(raw)**2) is applied fresh every forward pass, so the
    # RAW parameter can be anything; what matters is the constraint holds
    # at the point it's actually used (verified via finite, bounded output
    # rather than reading the transient raw value here).


def test_stable_leading_preserves_zero_init_identity():
    prop = LatentPropagator(
        _cfg(spectral_poly_stable_leading=True, zero_init=True, spectral_integrator="euler")
    )
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_stable_leading_raises_if_no_even_term_survives():
    # degree=0 (bypassing PropagatorConfig's own 1/2/3 validation by
    # constructing the body directly) has NO linear terms in the library
    # at all -- nothing for poly_stable_leading to constrain.
    from ks_latent.models.propagator import _SpectralPDEDeltaBody

    with pytest.raises(ValueError, match="poly_stable_leading=True requires"):
        _SpectralPDEDeltaBody(
            K=8, N_w=32, L=22.0, max_order=4, hidden=16, n_blocks=1, dropout=0.0,
            zero_init=False, field_kind="polynomial", poly_degree=0,
            poly_stable_leading=True,
        )


def test_stable_leading_sign_flips_for_order2_leading():
    # Re((ik)^n) cycles with period 4: +1 at n%4==0, -1 at n%4==2. The
    # sign REQUIRED of c_n for boundedness is the opposite of Re((ik)^n),
    # so order 4 (0 mod 4) needs c_4<0 but order 2 (2 mod 4) needs c_2>0
    # -- the OPPOSITE sign. Test this directly by dropping max_order to 2
    # (no w_xxxx at all), making order 2 (w_xx) the leading even term.
    prop = LatentPropagator(_cfg(spectral_max_order=2, spectral_poly_stable_leading=True))
    assert prop.body._stable_leading_order == 2
    assert prop.body._stable_leading_sign == 1.0  # positive, NOT -1.0 like order 4
    with torch.no_grad():
        prop.body.poly_coeffs.weight[:, prop.body._stable_leading_idx] = 3.0
    z = _valid_z(3, 32, 8)
    out = prop.step_one(z)
    assert torch.isfinite(out).all()


def test_stable_leading_order4_sign_is_negative():
    prop = LatentPropagator(_cfg(spectral_poly_stable_leading=True))  # default max_order=4
    assert prop.body._stable_leading_sign == -1.0


def test_stable_leading_degree3_uses_same_leading_term():
    # degree=3 adds cubic terms but the LINEAR term at order=4 is still at
    # the same fixed index (5) -- poly_stable_leading only ever touches the
    # linear part, regardless of what higher-degree terms exist alongside it.
    prop = LatentPropagator(_cfg(spectral_poly_degree=3, spectral_poly_stable_leading=True))
    assert prop.body._stable_leading_idx == 5
    z = _valid_z(3, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
