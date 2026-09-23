"""Stage-1 AE / Stage-2 propagator architecture and training-mechanics tests
(brief §5.3). Small configs throughout -- these test the mechanism, not
scientific reproduction (that's the replication-tier Gate 3 run).
"""

from __future__ import annotations

import torch
import pytest

from ks_latent.config import AutoencoderConfig, AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.training.losses import decorr_var_loss, reconstruction_loss


def _small_ae_cfg(**overrides) -> AutoencoderConfig:
    base = dict(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    base.update(overrides)
    return AutoencoderConfig(**base)


def _smooth_periodic_batch(n: int, NX: int, n_modes: int = 3) -> torch.Tensor:
    """Smooth low-mode periodic signals -- unlike white noise, these have a
    consistent *shape* under circular shift for an equivariant encoder to
    latch onto (used by the shift-equivariance test)."""
    x = torch.arange(NX, dtype=torch.float32) * 2 * torch.pi / NX
    u = torch.zeros(n, NX)
    for j in range(1, n_modes + 1):
        amp = torch.randn(n, 1)
        phase = torch.rand(n, 1) * 2 * torch.pi
        u += amp * torch.cos(j * x.unsqueeze(0) + phase)
    return u


def test_autoencoder_shapes():
    cfg = _small_ae_cfg()
    ae = KSAutoencoderPatched(cfg)
    u = torch.randn(5, cfg.NX)
    u_hat, z = ae(u)
    assert z.shape == (5, cfg.d_latent)
    assert u_hat.shape == (5, cfg.NX)
    assert ae.encode(u).shape == (5, cfg.d_latent)
    assert ae.decode(z).shape == (5, cfg.NX)


def test_propagator_shapes():
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=2, dropout=0.0)
    prop = LatentPropagator(cfg)
    z_prev = torch.randn(5, 6)
    z_curr = torch.randn(5, 6)
    z_next = prop.step(z_prev, z_curr)
    assert z_next.shape == (5, 6)
    rollout = prop.rollout(z_prev, z_curr, k=4)
    assert rollout.shape == (5, 4, 6)


def test_propagator_is_identity_at_init():
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=2, dropout=0.0, zero_init=True)
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(8, 6)
    z_curr = torch.randn(8, 6)
    z_next = prop.step(z_prev, z_curr)
    assert torch.allclose(z_next, z_curr, atol=1e-6)

    rollout = prop.rollout(z_prev, z_curr, k=16)
    expected = z_curr.unsqueeze(1).expand(-1, 16, -1)
    assert torch.allclose(rollout, expected, atol=1e-6)


def test_propagator_without_zero_init_is_not_identity():
    torch.manual_seed(0)
    cfg = PropagatorConfig(d_latent=6, hidden=16, n_blocks=2, dropout=0.0, zero_init=False)
    prop = LatentPropagator(cfg)
    prop.eval()
    z_prev = torch.randn(8, 6)
    z_curr = torch.randn(8, 6)
    z_next = prop.step(z_prev, z_curr)
    assert not torch.allclose(z_next, z_curr, atol=1e-3)


def test_overfit_single_batch_autoencoder():
    torch.manual_seed(0)
    cfg = _small_ae_cfg(d_model=24, dim_ff=24, d_latent=8)
    ae = KSAutoencoderPatched(cfg)
    u = torch.randn(8, cfg.NX)
    opt = torch.optim.AdamW(ae.parameters(), lr=8e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    for _ in range(500):
        u_hat, z = ae(u)
        loss = reconstruction_loss(u_hat, u)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    assert loss.item() < 1e-4


def test_overfit_single_batch_propagator():
    torch.manual_seed(0)
    cfg = PropagatorConfig(d_latent=6, hidden=32, n_blocks=2, dropout=0.0)
    prop = LatentPropagator(cfg)
    z_prev = torch.randn(8, 6)
    z_curr = torch.randn(8, 6)
    z_target = torch.randn(8, 6)
    opt = torch.optim.AdamW(prop.parameters(), lr=3e-3)
    for _ in range(500):
        z_next = prop.step(z_prev, z_curr)
        loss = ((z_next - z_target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-4


def test_aux_propagator_gradient_reaches_encoder():
    """L_pred must flow gradients back into the encoder (brief §5.1) -- easy
    to break silently (e.g. by accidentally decoding a detached z)."""
    torch.manual_seed(0)
    ae_cfg = _small_ae_cfg()
    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=2))

    window = torch.randn(4, 4, ae_cfg.NX)  # (B=4, window=4, NX)
    u_flat = window.reshape(16, ae_cfg.NX)
    u_hat, z = ae(u_flat)
    z_win = z.view(4, 4, -1)
    z0, z1 = z_win[:, 0], z_win[:, 1]
    u2, u3 = window[:, 2], window[:, 3]

    z2_hat = aux(z0, z1)
    z3_hat = aux(z1, z2_hat)
    u2_hat = ae.decode(z2_hat)
    u3_hat = ae.decode(z3_hat)
    l_pred = 0.5 * (reconstruction_loss(u2_hat, u2) + reconstruction_loss(u3_hat, u3))

    ae.zero_grad()
    l_pred.backward()
    encoder_grad_norm = ae.patch_embed.mlp[0].weight.grad.norm().item()
    assert encoder_grad_norm > 0.0, "L_pred did not flow gradients into the encoder"


def test_decorr_var_loss_zero_for_identity_covariance():
    torch.manual_seed(0)
    d = 8
    z = torch.randn(2000, d)  # approx unit-variance, uncorrelated
    l_decorr, l_var = decorr_var_loss(z)
    assert l_decorr.item() < 0.01
    assert l_var.item() < 0.02


def test_shift_equivariance_improves_with_training():
    """Measures (does not just assert pass/fail) how approximately
    shift-equivariant the encoder is -- needed by diagnostic D4 (Phase 6).
    A briefly-trained encoder (with translation-augmented reconstruction
    data) should be measurably *more* shift-equivariant than a random-init
    one, since reconstructing shifted copies of the same underlying signal
    pressures the encoder toward it.
    """
    torch.manual_seed(0)
    cfg = _small_ae_cfg()

    def equivariance_error(ae, u, shifts):
        errs = []
        z = ae.encode(u)
        for c in shifts:
            u_shift = torch.roll(u, shifts=c, dims=-1)
            z_shift = ae.encode(u_shift)
            errs.append((z_shift - z).pow(2).mean().item())
        return sum(errs) / len(errs)

    ae = KSAutoencoderPatched(cfg)
    shifts = [1, 2, 4, 8]
    u = _smooth_periodic_batch(16, cfg.NX)
    error_before = equivariance_error(ae, u, shifts)

    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
    for _ in range(300):
        c = int(torch.randint(0, cfg.NX, (1,)))
        u_aug = torch.roll(u, shifts=c, dims=-1)
        u_hat, _ = ae(u_aug)
        loss = reconstruction_loss(u_hat, u_aug)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    error_after = equivariance_error(ae, u, shifts)
    assert error_after < error_before
