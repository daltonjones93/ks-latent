"""Tests for `field_kind="rational"` (added 2026-09-27, Section 226, user-
directed: "let's try the rational function idea. keep the functional form
pretty simple. at most degree 3 in denominator and numerator and try to
explicitly avoid poles"). See `_SpectralPDEDeltaBody`'s "rational" branch
docstring (`__init__` and `field()`) for the full derivation:
`correction = N(library) / (1 + softplus(D(library)))`, both `N`/`D` a
plain linear layer over the same degree-<=poly_degree monomial library
`field_kind="polynomial"` already uses. `1 + softplus(...) >= 1` for any
real input -- the actual pole-avoidance mechanism under test here, not
just an empirical claim.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import (
    AuxPropagator,
    LatentPropagator,
    _polynomial_library_size,
)
from ks_latent.models.spectral_field import encode_to_spectrum


def _cfg(**overrides):
    defaults = dict(
        d_latent=16, mode="markovian", backbone="spectral_pde",
        spectral_K=8, spectral_N_w=32, spectral_L=22.0, spectral_max_order=4,
        hidden=32, n_blocks=1, spectral_field_kind="rational", spectral_poly_degree=2,
    )
    defaults.update(overrides)
    return PropagatorConfig(**defaults)


def _valid_z(n: int, N_w: int, K: int) -> torch.Tensor:
    w = torch.randn(n, N_w)
    return encode_to_spectrum(w, K)


def test_constructs_and_produces_right_shape():
    cfg = _cfg()
    prop = LatentPropagator(cfg)
    z = _valid_z(4, cfg.spectral_N_w, cfg.spectral_K)
    out = prop.step_one(z)
    assert out.shape == z.shape


def test_identity_at_init():
    cfg = _cfg(zero_init=True)
    prop = LatentPropagator(cfg)
    z = _valid_z(4, cfg.spectral_N_w, cfg.spectral_K)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = _valid_z(4, cfg.spectral_N_w, cfg.spectral_K)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-5)


def test_denominator_is_always_at_least_one():
    """The actual pole-avoidance guarantee is `1 + softplus(raw) >= 1` for
    *any* real `raw` -- this holds independent of what the (unconstrained)
    denominator linear layer learns, since softplus is non-negative
    everywhere. Sweep a wide range of raw pre-activation values, including
    extreme ones, and confirm the floor holds and stays finite (softplus
    itself is numerically safe at large |x|, unlike a bare exp-based
    formula would be)."""
    raw = torch.tensor([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
    denom = 1.0 + torch.nn.functional.softplus(raw)
    assert torch.isfinite(denom).all()
    assert (denom >= 1.0).all()


def test_finite_output_under_large_but_realistic_weights():
    """A more realistic stress test than driving the monomial library
    itself into float32 overflow (which produces inf/inf = nan regardless
    of the rational-vs-polynomial choice, and isn't specific to the
    pole-avoidance mechanism this class adds): weights an order of
    magnitude above typical trained scale, moderate input magnitude."""
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    body = prop.body
    with torch.no_grad():
        body.rational_denom_coeffs.weight.copy_(torch.randn_like(body.rational_denom_coeffs.weight) * 10.0)
        body.rational_denom_coeffs.bias.copy_(torch.randn_like(body.rational_denom_coeffs.bias) * 10.0)
        body.rational_num_coeffs.weight.copy_(torch.randn_like(body.rational_num_coeffs.weight) * 10.0)
    z = _valid_z(8, cfg.spectral_N_w, cfg.spectral_K) * 2.0
    out = prop.step_one(z)
    assert torch.isfinite(out).all()


def test_gradient_flows_to_both_numerator_and_denominator():
    cfg = _cfg(zero_init=False)
    prop = LatentPropagator(cfg)
    z = _valid_z(4, cfg.spectral_N_w, cfg.spectral_K)
    out = prop.step_one(z)
    out.sum().backward()
    body = prop.body
    assert body.rational_num_coeffs.weight.grad is not None
    assert body.rational_num_coeffs.weight.grad.abs().sum().item() > 0.0
    assert body.rational_denom_coeffs.weight.grad is not None
    assert body.rational_denom_coeffs.weight.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_degree_1_2_3_all_construct_with_expected_library_size(degree):
    cfg = _cfg(spectral_poly_degree=degree)
    prop = LatentPropagator(cfg)
    n_vars = cfg.spectral_max_order + 1
    expected = _polynomial_library_size(n_vars, degree, None)
    assert prop.body.rational_num_coeffs.weight.shape == (1, expected)
    assert prop.body.rational_denom_coeffs.weight.shape == (1, expected)


def test_invalid_poly_degree_rejected_for_rational():
    with pytest.raises(ValueError, match="spectral_poly_degree"):
        _cfg(spectral_poly_degree=4)


def test_works_with_aux_propagator_too():
    from ks_latent.config import AuxPropagatorConfig
    cfg = AuxPropagatorConfig(
        d_latent=16, mode="markovian", backbone="spectral_pde",
        spectral_K=8, spectral_N_w=32, spectral_L=22.0, spectral_max_order=4,
        hidden=32, n_blocks=1, spectral_field_kind="rational", spectral_poly_degree=2,
    )
    prop = AuxPropagator(cfg)
    z = _valid_z(4, cfg.spectral_N_w, cfg.spectral_K)
    out = prop.step_one(z)
    assert out.shape == z.shape
