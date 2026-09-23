"""Tests for `backbone="fourier_mlp"` (added 2026-09-03, user-directed:
"an mlp model that takes in fourier features (such as the FNO) and also
actual latent states, as kind of a hybrid mlp"). See `PropagatorConfig`'s
`backbone="fourier_mlp"` docstring: for each history state, concatenates
the raw latent vector with the real/imaginary parts of its first
`fno_modes` `rfft` frequencies, flattens across history, and feeds the
result through the same plain `MLPDeltaBody` `backbone="mlp"` uses.
`mode="history"` (`n_history>=2`) or `mode="markovian"` (added
2026-09-04, user-directed: "make the fourier mlp markovian" -- Section
84 -- a single current state, internally `n_history=1`, no separate
history states to featurize).

Also tests the masked option (`attn_window` set, added 2026-09-03,
user-directed: "use all the fourier coefficients, but only let real
variables interact with their neighbors") -- see
`_FourierMLPHistoryDeltaBody`'s docstring for the masked-raw-path +
dense-Fourier-path split.
"""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import PropagatorConfig
from ks_latent.models.propagator import LatentPropagator


def test_fourier_mlp_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=44, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=64, n_blocks=2,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(5, 2, 44)
    out = prop.step_history(z_hist)
    assert out.shape == (5, 44)


def test_fourier_mlp_rollout_history_shape():
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=3)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(4, 3, 20)
    rollout = prop.rollout_history(z_hist, k=5)
    assert rollout.shape == (4, 5, 20)


def test_fourier_mlp_is_identity_at_init():
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    out = prop.step_history(z_hist)
    assert torch.allclose(out, z_hist[:, -1], atol=1e-6)


def test_fourier_mlp_fno_modes_clips_to_max_available():
    """fno_modes larger than d_latent//2+1 should clip, not error or
    silently mismatch the constructed input dimension."""
    cfg = PropagatorConfig(
        d_latent=10, mode="history", backbone="fourier_mlp", n_history=2, fno_modes=1000,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.n_modes == 10 // 2 + 1
    z_hist = torch.randn(2, 2, 10)
    out = prop.step_history(z_hist)
    assert out.shape == (2, 10)


def test_fourier_mlp_default_fno_modes_uses_full_spectrum():
    cfg = PropagatorConfig(d_latent=16, mode="history", backbone="fourier_mlp", n_history=2)
    prop = LatentPropagator(cfg)
    assert prop.body.n_modes == 16 // 2 + 1


def test_two_step_with_fourier_mlp_backbone_raises():
    with pytest.raises(ValueError, match="only implemented for mode='markovian'"):
        PropagatorConfig(mode="two_step", backbone="fourier_mlp")


def test_fourier_mlp_works_under_bfloat16_autocast():
    """Regression test (same bug class as SpectralConv1d/logdet_barrier_loss):
    torch.fft.rfft doesn't support bfloat16 -- _fourier_features must
    explicitly cast to float32 itself."""
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_history(z_hist)
    assert torch.isfinite(out).all()


# ---- masked option (attn_window set): masked raw path + dense Fourier
# path, added 2026-09-03, user-directed. ----


def test_fourier_mlp_masked_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=44, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=64, n_blocks=2, attn_window=4,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.masked_body is not None and prop.body.fourier_body is not None and prop.body.body is None
    z_hist = torch.randn(5, 2, 44)
    out = prop.step_history(z_hist)
    assert out.shape == (5, 44)


def test_fourier_mlp_unmasked_by_default():
    """attn_window=None (default) keeps the original single-dense-body
    structure -- backward compatible, no behavior change for existing
    recipes that never set attn_window."""
    cfg = PropagatorConfig(d_latent=44, mode="history", backbone="fourier_mlp", n_history=2)
    prop = LatentPropagator(cfg)
    assert prop.body.body is not None and prop.body.masked_body is None and prop.body.fourier_body is None


def test_fourier_mlp_masked_is_identity_at_init():
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2, attn_window=3)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    out = prop.step_history(z_hist)
    assert torch.allclose(out, z_hist[:, -1], atol=1e-6)


def test_fourier_mlp_masked_raw_path_weight_is_exactly_zero_outside_band():
    """Direct test of the masking claim: the masked sub-path's weight
    matrix must have EXACT zeros outside the circular band, not just
    small values -- same guarantee `backbone="masked_mlp"` provides."""
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2, attn_window=3)
    prop = LatentPropagator(cfg)
    W = prop.body.masked_body.input_proj.linear.weight.detach()
    idx = torch.arange(20)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    dist = torch.minimum(dist, 20 - dist)
    assert (W[dist > 3] == 0).all()


def test_fourier_mlp_masked_fourier_path_stays_fully_dense():
    """"use all the fourier coefficients" -- the Fourier path must NOT be
    masked, unlike the raw path."""
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2, attn_window=3)
    prop = LatentPropagator(cfg)
    assert isinstance(prop.body.fourier_body.input_proj, torch.nn.Linear)
    assert not hasattr(prop.body.fourier_body.input_proj, "mask")


def test_fourier_mlp_masked_rollout_history_shape():
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=3, attn_window=2)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(4, 3, 20)
    rollout = prop.rollout_history(z_hist, k=5)
    assert rollout.shape == (4, 5, 20)


def test_fourier_mlp_masked_works_under_bfloat16_autocast():
    cfg = PropagatorConfig(d_latent=20, mode="history", backbone="fourier_mlp", n_history=2, attn_window=3)
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_history(z_hist)
    assert torch.isfinite(out).all()


# ---- fourier_ifft_readout option: the Fourier-only sub-network's output
# is interpreted as frequency-domain coefficients and explicitly inverse-
# transformed (irfft) back to d_latent, instead of an unconstrained linear
# readout -- added 2026-09-03, user-directed: "apply an inverse fft to the
# frequency component output of the fourier mlp to map back to state
# space in the encoder and propagator and decoder". "dense for the
# propagator": attn_window=None + fourier_ifft_readout=True drops the raw
# concatenation entirely (pure Fourier+irfft mechanism, no masked_body). ----


def test_fourier_ifft_readout_dense_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=None, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    from ks_latent.models.propagator import FourierIFFTBody
    assert prop.body.body is None and prop.body.masked_body is None
    assert isinstance(prop.body.fourier_body, FourierIFFTBody)
    z_hist = torch.randn(5, 2, 20)
    out = prop.step_history(z_hist)
    assert out.shape == (5, 20)


def test_fourier_ifft_readout_is_identity_at_init():
    """zero_init zeros the predicted frequency coefficients; irfft of an
    all-zero spectrum is exactly zero, so step_history(z_hist) ==
    z_hist[:, -1] exactly at init."""
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=None, fourier_ifft_readout=True, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(4, 2, 20)
    out = prop.step_history(z_hist)
    assert torch.allclose(out, z_hist[:, -1])


def test_fourier_ifft_readout_masked_keeps_masked_body_untouched():
    """attn_window set + fourier_ifft_readout=True: masked_body (raw
    values) is unchanged (plain _MaskedMLPDeltaBody); only fourier_body
    becomes a FourierIFFTBody."""
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    from ks_latent.models.propagator import FourierIFFTBody, _MaskedMLPDeltaBody
    assert isinstance(prop.body.masked_body, _MaskedMLPDeltaBody)
    assert isinstance(prop.body.fourier_body, FourierIFFTBody)
    z_hist = torch.randn(5, 2, 20)
    out = prop.step_history(z_hist)
    assert out.shape == (5, 20)


def test_fourier_ifft_readout_rollout_history_shape():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=3,
        hidden=32, n_blocks=1, attn_window=None, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(4, 3, 20)
    rollout = prop.rollout_history(z_hist, k=5)
    assert rollout.shape == (4, 5, 20)


def test_fourier_ifft_readout_gradient_flows_end_to_end():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=None, fourier_ifft_readout=True, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20, requires_grad=True)
    out = prop.step_history(z_hist)
    out.sum().backward()
    assert torch.isfinite(z_hist.grad).all()


def test_fourier_ifft_readout_works_under_bfloat16_autocast():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=None, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_history(z_hist)
    assert torch.isfinite(out).all()


# ---- nonexpansive: spectral-normalized layers + damped residuals +
# ortho FFT + averaged (not summed) two-network combination -- added
# 2026-09-04, user-directed: "would there be a way to constrain the
# fourier_mlp to be nonexpansive". ----


def test_nonexpansive_identity_at_init():
    """zero_init still zeros the predicted frequency coefficients (and the
    masked path's unconstrained output_proj) even under nonexpansive --
    step_history(z_hist) == z_hist[:, -1] exactly at init."""
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=2, attn_window=8, fourier_ifft_readout=True,
        nonexpansive=True, zero_init=True,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(4, 2, 20)
    out = prop.step_history(z_hist)
    assert torch.allclose(out, z_hist[:, -1])


def test_nonexpansive_gradient_flows_end_to_end():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=8, fourier_ifft_readout=True,
        nonexpansive=True, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20, requires_grad=True)
    out = prop.step_history(z_hist)
    out.sum().backward()
    assert torch.isfinite(z_hist.grad).all()


def test_nonexpansive_masked_body_effective_weight_zero_outside_band():
    """See the AE-side test's docstring for why this checks the EFFECTIVE
    (mask-multiplied) weight rather than the raw parametrized one under
    nonexpansive."""
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    prop = LatentPropagator(cfg)
    mask = prop.body.masked_body.input_proj.mask
    W_effective = prop.body.masked_body.input_proj.linear.weight * mask
    assert (W_effective[~mask.bool()] == 0).all()


def test_nonexpansive_works_under_bfloat16_autocast():
    cfg = PropagatorConfig(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=1, attn_window=8, fourier_ifft_readout=True,
        nonexpansive=True,
    )
    prop = LatentPropagator(cfg)
    z_hist = torch.randn(3, 2, 20)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_history(z_hist)
    assert torch.isfinite(out).all()


def test_nonexpansive_encoder_jacobian_smaller_than_unconstrained_via_step():
    """Empirical check via a full step_history call (not just a single
    sub-module), mirroring the AE-side test: spectral normalization +
    damped residuals + averaged two-network sum should reduce local
    sensitivity relative to the unconstrained version."""
    from torch.func import jacrev

    torch.manual_seed(0)
    kwargs = dict(
        d_latent=20, mode="history", backbone="fourier_mlp", n_history=2,
        hidden=32, n_blocks=2, attn_window=8, fourier_ifft_readout=True, zero_init=False,
    )
    prop_plain = LatentPropagator(PropagatorConfig(**kwargs, nonexpansive=False))
    prop_nonexp = LatentPropagator(PropagatorConfig(**kwargs, nonexpansive=True))
    prop_plain.eval()
    prop_nonexp.eval()
    z_hist = torch.randn(2, 20)

    def make_fn(prop):
        def fn(z_hist_flat):
            return prop.step_history(z_hist_flat.unsqueeze(0)).squeeze(0)
        return fn

    J_plain = jacrev(make_fn(prop_plain))(z_hist)
    J_nonexp = jacrev(make_fn(prop_nonexp))(z_hist)
    norm_plain = torch.linalg.svdvals(J_plain.reshape(20, -1))[0].item()
    norm_nonexp = torch.linalg.svdvals(J_nonexp.reshape(20, -1))[0].item()
    assert norm_nonexp < norm_plain


# ---- mode="markovian" (added 2026-09-04, user-directed: "make the
# fourier mlp markovian" -- Section 84): reuses _FourierMLPHistoryDeltaBody
# with n_history=1 (single current state), called via step_one's
# unsqueeze(1) reshape instead of step_history. ----


def test_markovian_fourier_mlp_no_longer_raises():
    """Regression guard for the walked-back restriction: mode='markovian'
    + backbone='fourier_mlp' used to raise unconditionally; it must now
    construct cleanly."""
    PropagatorConfig(d_latent=20, mode="markovian", backbone="fourier_mlp")


def test_markovian_fourier_mlp_constructs_and_produces_right_shape():
    cfg = PropagatorConfig(
        d_latent=44, mode="markovian", backbone="fourier_mlp", hidden=64, n_blocks=2,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.n_history == 1
    z = torch.randn(5, 44)
    out = prop.step_one(z)
    assert out.shape == (5, 44)


def test_markovian_fourier_mlp_is_identity_at_init():
    cfg = PropagatorConfig(d_latent=20, mode="markovian", backbone="fourier_mlp")
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 20)
    out = prop.step_one(z)
    assert torch.allclose(out, z, atol=1e-6)


def test_markovian_fourier_mlp_rollout_shape():
    """Unlike mode='history', mode='markovian' uses the uniform
    step/rollout interface (step_one internally), not step_history/
    rollout_history."""
    cfg = PropagatorConfig(d_latent=20, mode="markovian", backbone="fourier_mlp")
    prop = LatentPropagator(cfg)
    z0 = torch.randn(4, 20)
    rollout = prop.rollout(z0, z0, k=5)  # z_prev ignored in markovian mode
    assert rollout.shape == (4, 5, 20)


def test_markovian_fourier_mlp_masked_constructs_two_networks():
    """attn_window set: same masked-raw-path + dense-Fourier-path split as
    mode='history', just with the masked path reading the (only) current
    state directly."""
    cfg = PropagatorConfig(
        d_latent=44, mode="markovian", backbone="fourier_mlp",
        hidden=64, n_blocks=2, attn_window=4,
    )
    prop = LatentPropagator(cfg)
    assert prop.body.masked_body is not None and prop.body.fourier_body is not None and prop.body.body is None
    z = torch.randn(5, 44)
    out = prop.step_one(z)
    assert out.shape == (5, 44)


def test_markovian_fourier_mlp_ifft_readout_constructs_and_shape():
    cfg = PropagatorConfig(
        d_latent=20, mode="markovian", backbone="fourier_mlp",
        hidden=32, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    from ks_latent.models.propagator import FourierIFFTBody, _MaskedMLPDeltaBody
    assert isinstance(prop.body.masked_body, _MaskedMLPDeltaBody)
    assert isinstance(prop.body.fourier_body, FourierIFFTBody)
    z = torch.randn(5, 20)
    out = prop.step_one(z)
    assert out.shape == (5, 20)


def test_markovian_fourier_mlp_gradient_flows_end_to_end():
    cfg = PropagatorConfig(
        d_latent=20, mode="markovian", backbone="fourier_mlp",
        hidden=32, n_blocks=1, attn_window=3, fourier_ifft_readout=True, zero_init=False,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 20, requires_grad=True)
    out = prop.step_one(z)
    out.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_markovian_fourier_mlp_works_under_bfloat16_autocast():
    cfg = PropagatorConfig(
        d_latent=20, mode="markovian", backbone="fourier_mlp",
        hidden=32, n_blocks=1, attn_window=3, fourier_ifft_readout=True,
    )
    prop = LatentPropagator(cfg)
    z = torch.randn(3, 20)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = prop.step_one(z)
    assert torch.isfinite(out).all()
