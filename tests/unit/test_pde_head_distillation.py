"""Tests for `pde_head` distillation (added 2026-09-08, see
docs/sine_transform_pde_plan.md, user-directed: "joint training seems
pretty smart. can you implement this idea", then refined: "I would like to
be able to use the encoder and decoder and propagator from 95 ... within
the propagator, I want to take the fourier transform of the latent states,
and train the spectral pde with this information"): a SECOND, always
`backbone="spectral_pde_raw"` propagator trained ALONGSIDE a free/
unconstrained `aux` propagator via a single-step distillation loss against
`aux`'s own (detached) realized one-step prediction, rather than as the
primary rollout-trained propagator. `spectral_pde_raw` works on the RAW
latent `z` of ANY encoder (no dedicated `spectral_field` encoder required --
`z` need not already be a truncated rFFT spectrum). See `train_stage1`'s
own docstring for the full mechanism.

These tests directly replicate `train_stage1`'s exact `pde_head` code path
(same call sequence: `ae.encode` -> `aux.rollout` -> `z_pred[:, 0].detach()`
-> `pde_head.step_one`) to verify the gradient-isolation property the whole
design rests on: `aux` (the free model) must receive ZERO gradient from
this term (so it stays free to fit the real dynamics, uncorrupted by
pressure to look local), while `pde_head` and the ENCODER (via `z`) both
receive real gradient from it. Uses a PLAIN (non-spectral) encoder
deliberately, matching what `train_stage1_patched.py --pde-distill` now
actually builds for any `--encoder`.
"""

from __future__ import annotations

import torch

from ks_latent.config import AutoencoderConfig, AuxPropagatorConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.losses import reconstruction_loss
from ks_latent.utils.seeding import set_seed


def _build_ae_and_heads(NX: int = 32, d_latent: int = 20):
    ae_cfg = AutoencoderConfig(
        NX=NX, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=d_latent, dropout=0.0,
    )
    ae = KSAutoencoderPatched(ae_cfg)

    aux_cfg = AuxPropagatorConfig(d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="mlp")
    aux = AuxPropagator(aux_cfg)

    pde_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=16, n_blocks=1, mode="markovian", backbone="spectral_pde_raw",
        spectral_K=d_latent // 2 + 1, spectral_L=float(d_latent), spectral_max_order=4,
        spectral_integrator="etdrk4", zero_init=False,
    )
    pde_head = AuxPropagator(pde_cfg)
    return ae, aux, pde_head, d_latent


def _distill_step(ae, aux, pde_head, u: torch.Tensor, detach_target: bool = True):
    """Exact replica of train_stage1's pde_head code path for a single
    markovian-mode batch (window=1, k_now=1): encode -> aux.rollout(1 step)
    -> z_pred[:, 0] (detached iff detach_target) as target -> pde_head.
    step_one as prediction. `detach_target` mirrors
    Stage1TrainingConfig.pde_distill_detach_target."""
    z = ae.encode(u)
    z_curr_in = z
    z_pred = aux.rollout(z_curr_in, z_curr_in, 1)  # (b, 1, d)
    g_target = z_pred[:, 0]
    if detach_target:
        g_target = g_target.detach()
    p_pred = pde_head.step_one(z_curr_in)
    return reconstruction_loss(p_pred, g_target)


def test_aux_receives_zero_gradient_from_distillation_loss():
    set_seed(0)
    ae, aux, pde_head, d_latent = _build_ae_and_heads()
    u = torch.randn(6, 32)

    loss = _distill_step(ae, aux, pde_head, u, detach_target=True)
    loss.backward()

    for name, p in aux.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"aux param {name!r} received nonzero gradient from the distillation loss -- "
            "the target must be detached so aux stays free to fit real dynamics."
        )


def test_aux_receives_nonzero_gradient_when_mutual():
    """pde_distill_detach_target=False (added 2026-09-08, user-directed:
    'I think the loss should be mutual for stage 1 training too'): aux
    must now receive real gradient from this term."""
    set_seed(0)
    ae, aux, pde_head, d_latent = _build_ae_and_heads()
    u = torch.randn(6, 32)

    loss = _distill_step(ae, aux, pde_head, u, detach_target=False)
    loss.backward()

    grads = [p.grad for p in aux.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_pde_head_receives_nonzero_gradient_from_distillation_loss():
    set_seed(0)
    ae, aux, pde_head, d_latent = _build_ae_and_heads()
    u = torch.randn(6, 32)

    loss = _distill_step(ae, aux, pde_head, u)
    loss.backward()

    grads = [p.grad for p in pde_head.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_encoder_receives_nonzero_gradient_from_distillation_loss():
    """The whole point: the encoder should be pressured toward a
    representation pde_head can fit, not just pde_head itself."""
    set_seed(0)
    ae, aux, pde_head, d_latent = _build_ae_and_heads()
    u = torch.randn(6, 32)

    loss = _distill_step(ae, aux, pde_head, u)
    loss.backward()

    grads = [p.grad for p in ae.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_loss_is_zero_at_matching_outputs():
    """Sanity: if pde_head's prediction exactly matches aux's, the loss is 0."""
    set_seed(0)
    ae, aux, pde_head, d_latent = _build_ae_and_heads()
    z = torch.randn(4, d_latent)
    g_target = aux.rollout(z, z, 1)[:, 0].detach()
    loss = reconstruction_loss(g_target, g_target)
    assert loss.item() == 0.0
