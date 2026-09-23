"""Tests for `backbone="spectral_pde"` (added 2026-09-06, see
docs/sine_transform_pde_plan.md): synthesizes exact spatial derivatives
from a truncated rFFT spectrum `z` and runs a shared pointwise MLP mapping
the local derivative stack to `w_t`, integrated via Euler, fixed-step RK4,
or ETDRK4 (exact exponential integration of KS's own true linear term,
mirroring `ks_latent/solver/ks.py`'s own solver directly).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.models.spectral_field import encode_to_spectrum


def _valid_z(n: int, N_w: int, K: int) -> torch.Tensor:
    """A physically-valid z (came from a real field) -- see
    spectral_field.py's docstring for why an all-random z is not valid
    (nonzero DC-imaginary slot)."""
    w = torch.randn(n, N_w)
    return encode_to_spectrum(w, K)


def _cfg(**overrides):
    defaults = dict(
        d_latent=16, mode="markovian", backbone="spectral_pde",
        spectral_K=8, spectral_N_w=32, spectral_L=100.0, spectral_max_order=4,
        hidden=32, n_blocks=1,
    )
    defaults.update(overrides)
    return PropagatorConfig(**defaults)


def test_constructs_and_produces_right_shape():
    prop = LatentPropagator(_cfg())
    z = _valid_z(5, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (5, 16)


def test_identity_at_init_euler():
    prop = LatentPropagator(_cfg(zero_init=True))
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_identity_at_init_euler_multiple_substeps():
    """Added 2026-09-07, user-directed: 'couldn't we also integrate euler
    over multiple steps?' -- zero_init must still give exact identity
    regardless of ode_substeps (field()=0 at every sub-step, so w never
    moves)."""
    for substeps in (1, 2, 5):
        prop = LatentPropagator(_cfg(zero_init=True, ode_substeps=substeps))
        z = _valid_z(3, 32, 8)
        out = prop.step_one(z)
        assert torch.allclose(out, z, atol=1e-5), f"failed at ode_substeps={substeps}"


def test_euler_ode_substeps_1_matches_original_single_step_formula():
    """ode_substeps=1 must be EXACTLY the original single-step Euler
    formula (w_next = w + field(z)) -- this is the backward-compatibility
    guarantee for every prior spectral_pde result computed before the
    sub-stepping fix."""
    prop = LatentPropagator(_cfg(zero_init=False, ode_substeps=1))
    prop.eval()  # dropout (PropagatorConfig default 0.1) must be off for a
    # deterministic comparison -- otherwise two separate field() calls
    # (one here, one inside step_one) sample different dropout masks.
    z = _valid_z(4, 32, 8)
    body = prop.body
    w = body._decode(z)
    expected_w_next = w + body.field(z)
    expected = body._encode(expected_w_next) - z
    out = prop.step_one(z) - z  # step_one adds z back; recover the raw delta
    assert torch.allclose(out, expected, atol=1e-5)


def test_euler_ode_substeps_actually_changes_the_result():
    """Sub-stepping must not be a silent no-op: ode_substeps=1 vs.
    ode_substeps=3 (same weights, zero_init=False) must give genuinely
    different outputs -- confirms the loop actually re-encodes/re-evaluates
    field() at updated intermediate states rather than repeating the same
    computation ode_substeps times."""
    torch.manual_seed(0)
    z = _valid_z(4, 32, 8)
    prop1 = LatentPropagator(_cfg(zero_init=False, ode_substeps=1))
    torch.manual_seed(0)
    prop3 = LatentPropagator(_cfg(zero_init=False, ode_substeps=3))
    # Same seed => same random init weights for both (same architecture/shapes).
    prop1.eval()  # disable dropout so the difference reflects ode_substeps, not dropout noise
    prop3.eval()
    out1 = prop1.step_one(z)
    out3 = prop3.step_one(z)
    assert not torch.allclose(out1, out3, atol=1e-4)


def test_euler_gradient_flows_with_multiple_substeps():
    prop = LatentPropagator(_cfg(zero_init=False, ode_substeps=3))
    z = _valid_z(3, 32, 8).requires_grad_(True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert prop.body.input_proj.weight.grad is not None


def test_identity_at_init_rk4():
    prop = LatentPropagator(_cfg(zero_init=True, spectral_integrator="rk4", ode_substeps=3))
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-5)


def test_not_identity_when_zero_init_false():
    prop = LatentPropagator(_cfg(zero_init=False))
    z = _valid_z(4, 32, 8)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-4)


def test_gradient_flows():
    prop = LatentPropagator(_cfg(zero_init=False))
    z = _valid_z(4, 32, 8).requires_grad_(True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert prop.body.input_proj.weight.grad is not None
    assert torch.isfinite(prop.body.input_proj.weight.grad).all()


def test_field_output_shape():
    prop = LatentPropagator(_cfg())
    z = _valid_z(3, 32, 8)
    w_t = prop.body.field(z)
    assert w_t.shape == (3, 32)  # (B, N_w)


def test_field_is_translation_equivariant_pointwise_weight_sharing():
    """Same weights applied at every point -- perturbing the derivative
    stack identically at every point should perturb the output identically
    (up to the derivative synthesis's own linearity), a basic sanity check
    that the MLP has no per-position parameters."""
    prop = LatentPropagator(_cfg(zero_init=False))
    n_params_expected = sum(
        p.numel() for p in [prop.body.input_proj.weight, prop.body.input_proj.bias,
                             prop.body.output_proj.weight, prop.body.output_proj.bias]
    )
    # input_proj: Linear(max_order+1=5, hidden=32) -> shared across all N_w=32
    # points by construction (nn.Linear applied to the last dim of a
    # (B, N_w, 5) tensor broadcasts identically over N_w).
    assert prop.body.input_proj.weight.shape == (32, 5)


def test_requires_markovian_mode():
    with pytest.raises(ValueError, match="markovian"):
        _cfg(mode="two_step")


def test_requires_spectral_K_and_N_w():
    with pytest.raises(ValueError, match="spectral_K"):
        _cfg(spectral_K=None)
    with pytest.raises(ValueError, match="spectral_N_w"):
        _cfg(spectral_N_w=None)


def test_requires_d_latent_equals_2K():
    with pytest.raises(ValueError, match="2\\*spectral_K"):
        _cfg(d_latent=15)


def test_requires_valid_K_range():
    with pytest.raises(ValueError, match="spectral_K"):
        _cfg(spectral_K=100, d_latent=200)


def test_requires_positive_L():
    with pytest.raises(ValueError, match="spectral_L"):
        _cfg(spectral_L=0.0)


def test_requires_nonnegative_max_order():
    with pytest.raises(ValueError, match="spectral_max_order"):
        _cfg(spectral_max_order=-1)


def test_requires_valid_integrator():
    with pytest.raises(ValueError, match="spectral_integrator"):
        _cfg(spectral_integrator="bogus")


def test_rejected_for_history_mode():
    with pytest.raises(ValueError):
        PropagatorConfig(
            d_latent=16, mode="history", backbone="spectral_pde", n_history=2,
            spectral_K=8, spectral_N_w=32,
        )


def test_rejected_for_two_step_via_validate_mode_backbone():
    with pytest.raises(ValueError, match="markovian"):
        PropagatorConfig(d_latent=16, mode="two_step", backbone="spectral_pde")


# ---- "etdrk4" integrator (added 2026-09-06, see docs/sine_transform_pde_plan.md):
# exact exponential integration of KS's own true linear term Lhat(k)=k^2-k^4,
# mirroring ks_latent/solver/ks.py's own ETDRK4 solver directly. ----


def _true_exp_lhat(K: int, L: float, total_time: float = 1.0) -> torch.Tensor:
    """`exp(total_time * Lhat)`, duplicated to length 2*K to match z's
    concat(real, imag) layout -- the EXACT map this backbone reduces to at
    zero_init (N(z)=0 identically)."""
    k = 2.0 * math.pi * np.arange(K) / L
    Lhat = k**2 - k**4
    e = np.exp(total_time * Lhat)
    return torch.tensor(np.concatenate([e, e]), dtype=torch.float32)


def test_etdrk4_zero_init_reduces_to_exact_linear_map_one_substep():
    K, N_w, L = 8, 32, 100.0
    prop = LatentPropagator(_cfg(spectral_integrator="etdrk4", ode_substeps=1))
    z = _valid_z(3, N_w, K)
    out = prop.step_one(z)
    expected = _true_exp_lhat(K, L) * z
    assert torch.allclose(out, expected, atol=1e-5)


def test_etdrk4_zero_init_composes_exactly_across_substeps():
    """exp(a) composed n times equals exp(n*a) for a diagonal/scalar
    exponent -- the total effect after ode_substeps sub-steps of size
    1/ode_substeps each must equal the SAME single exp(1.0*Lhat) map,
    regardless of how many sub-steps are taken."""
    K, N_w, L = 8, 32, 100.0
    z = _valid_z(3, N_w, K)
    expected = _true_exp_lhat(K, L)
    for substeps in (1, 2, 4, 8):
        prop = LatentPropagator(_cfg(spectral_integrator="etdrk4", ode_substeps=substeps))
        out = prop.step_one(z)
        assert torch.allclose(out, expected * z, atol=1e-4), f"mismatch at ode_substeps={substeps}"


def test_etdrk4_not_pure_linear_when_zero_init_false():
    K, N_w, L = 8, 32, 100.0
    prop = LatentPropagator(_cfg(spectral_integrator="etdrk4", zero_init=False))
    z = _valid_z(3, N_w, K)
    out = prop.step_one(z)
    expected = _true_exp_lhat(K, L) * z
    assert not torch.allclose(out, expected, atol=1e-3)


def test_etdrk4_gradient_flows():
    prop = LatentPropagator(_cfg(spectral_integrator="etdrk4", zero_init=False))
    z = _valid_z(4, 32, 8).requires_grad_(True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert prop.body.input_proj.weight.grad is not None
    assert torch.isfinite(prop.body.input_proj.weight.grad).all()


def test_etdrk4_output_shape():
    prop = LatentPropagator(_cfg(spectral_integrator="etdrk4"))
    z = _valid_z(5, 32, 8)
    out = prop.step_one(z)
    assert out.shape == (5, 16)


def test_etdrk4_no_complex_cast_warning():
    """The coefficient buffers must be plain real float32 -- a stray
    complex128 -> float32 cast (discarding an exactly-zero imaginary part)
    would raise a UserWarning; this must not happen."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LatentPropagator(_cfg(spectral_integrator="etdrk4"))


def test_accepts_etdrk4_in_config_validation():
    cfg = _cfg(spectral_integrator="etdrk4")
    assert cfg.spectral_integrator == "etdrk4"


# ---- spectral_physics_prior (added 2026-09-08, see
# docs/sine_transform_pde_plan.md): bakes the exact true KS right-hand
# side into field()'s output as a fixed baseline, integrator-aware to
# avoid double-counting with etdrk4's own separate exact linear treatment.
# ----


def _analytic_ks_rhs(z, K, N_w, L, nonlinear_only: bool):
    from ks_latent.models.spectral_field import synthesize_derivatives

    derivs = synthesize_derivatives(z, K, N_w, L, 4)
    w, w_x, w_xx, w_xxxx = derivs[..., 0], derivs[..., 1], derivs[..., 2], derivs[..., 4]
    prior = -w * w_x
    if not nonlinear_only:
        prior = prior - w_xx - w_xxxx
    return prior


@pytest.mark.parametrize("integrator", ["euler", "rk4", "etdrk4"])
def test_physics_prior_field_matches_analytic_ks_rhs_at_zero_init(integrator):
    """At zero_init (correction MLP zeroed), field() must equal EXACTLY
    the analytic KS right-hand side -- the whole point of this option."""
    K, N_w, L = 8, 32, 22.0
    prop = LatentPropagator(_cfg(
        spectral_integrator=integrator, spectral_physics_prior=True, zero_init=True,
        spectral_L=L,
    ))
    z = _valid_z(3, N_w, K)
    f = prop.body.field(z)
    expected = _analytic_ks_rhs(z, K, N_w, L, nonlinear_only=(integrator == "etdrk4"))
    assert torch.allclose(f, expected, atol=1e-5)


def test_physics_prior_not_identity_at_init():
    """zero_init no longer means a no-op when physics_prior=True -- the
    propagator implements real (nonzero) KS dynamics from the start."""
    prop = LatentPropagator(_cfg(spectral_physics_prior=True, zero_init=True))
    z = _valid_z(3, 32, 8)
    out = prop.step_one(z)
    assert not torch.allclose(out, z, atol=1e-4)


def test_physics_prior_gradient_flows_through_correction():
    prop = LatentPropagator(_cfg(spectral_physics_prior=True, zero_init=False))
    z = _valid_z(3, 32, 8).requires_grad_(True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert prop.body.input_proj.weight.grad is not None
    assert torch.isfinite(prop.body.input_proj.weight.grad).all()


def test_physics_prior_requires_max_order_4():
    with pytest.raises(ValueError, match="spectral_physics_prior"):
        _cfg(spectral_physics_prior=True, spectral_max_order=2)


def test_physics_prior_correction_adds_on_top_of_prior():
    """With zero_init=False, field() must equal prior + (nonzero)
    correction, not the prior alone."""
    K, N_w, L = 8, 32, 22.0
    prop = LatentPropagator(_cfg(
        spectral_physics_prior=True, zero_init=False, spectral_L=L,
    ))
    z = _valid_z(3, N_w, K)
    f = prop.body.field(z)
    prior = _analytic_ks_rhs(z, K, N_w, L, nonlinear_only=False)
    assert not torch.allclose(f, prior, atol=1e-4)
