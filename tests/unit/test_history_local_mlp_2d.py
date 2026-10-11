"""mode="history" + backbone="local_mlp_2d" (J17B memory test, 2026-10-10)."""
import dataclasses

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig
from ks_latent.models import load_propagator_checkpoint_resized_2d
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.crop import forced_halo_rollout, forced_halo_rollout_history

Px, Py, C, H = 6, 5, 3, 4


def _cfg(mode="history", px=Px, py=Py):
    return AuxPropagatorConfig(d_latent=px * py * C, mode=mode, n_history=H if mode == "history" else 3,
                               backbone="local_mlp_2d", attn_window=1, local_mlp2d_n_sites_x=px,
                               local_mlp2d_n_sites_y=py, local_mlp2d_local_channels=C,
                               local_mlp2d_token_d_model=16, local_mlp2d_n_layers=2)


def _randomize(prop, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in prop.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return prop.eval()


def test_identity_at_init():
    prop = AuxPropagator(_cfg()).eval()
    z = torch.randn(2, H, Px * Py * C)
    assert torch.equal(prop.step_history(z), z[:, -1])


def test_rejects_wrong_history_length():
    prop = AuxPropagator(_cfg())
    with pytest.raises(ValueError, match="n_history"):
        prop.step_history(torch.randn(2, H - 1, Px * Py * C))


def test_translation_equivariance():
    prop = _randomize(AuxPropagator(_cfg()))
    z = torch.randn(2, H, Px, Py, C)
    out = prop.step_history(z.reshape(2, H, -1)).reshape(2, Px, Py, C)
    zr = torch.roll(z, (2, 1), dims=(2, 3))
    out_r = prop.step_history(zr.reshape(2, H, -1)).reshape(2, Px, Py, C)
    assert torch.allclose(out_r, torch.roll(out, (2, 1), dims=(1, 2)), atol=1e-5)


def test_uses_the_past():
    prop = _randomize(AuxPropagator(_cfg()))
    z = torch.randn(1, H, Px * Py * C)
    z2 = z.clone(); z2[:, 0] += 1.0
    assert not torch.allclose(prop.step_history(z), prop.step_history(z2))


def test_resize_roundtrip(tmp_path):
    prop = _randomize(AuxPropagator(_cfg()))
    path = tmp_path / "p.pt"
    torch.save({"prop_state_dict": prop.state_dict(), "prop_config": prop.cfg}, path)
    big, cfg, _ = load_propagator_checkpoint_resized_2d(path, new_n_sites_x=12, new_n_sites_y=10)
    assert cfg.mode == "history" and cfg.n_history == H
    # a field periodic with the small grid's period gives the tiled small-grid answer
    z = torch.randn(1, H, Px, Py, C)
    zb = z.repeat(1, 1, 2, 2, 1)
    out = prop.step_history(z.reshape(1, H, -1)).reshape(1, Px, Py, C)
    outb = big.eval().step_history(zb.reshape(1, H, -1)).reshape(1, 2 * Px, 2 * Py, C)
    assert torch.allclose(outb, out.repeat(1, 2, 2, 1), atol=1e-5)


def test_history_rollout_reduces_to_markovian_when_past_is_ignored():
    """Copy a markovian model's weights into a history model whose embedding reads only the newest
    state: the forced-halo history rollout must equal the markovian one on the same targets."""
    mk = _randomize(AuxPropagator(_cfg("markovian")), seed=1)
    hi = AuxPropagator(_cfg()).eval()
    sd = {k: v for k, v in mk.state_dict().items() if not k.startswith("body.token_embed.weight")}
    hi.load_state_dict(sd, strict=False)
    with torch.no_grad():
        w = torch.zeros_like(hi.body.token_embed.weight)
        w[:, -C:] = mk.body.token_embed.weight
        hi.body.token_embed.weight.copy_(w)
    k, halo = 3, 1
    win = torch.randn(2, H + k, Px, Py, C)
    ph = forced_halo_rollout_history(hi, win, k, halo, halo_y=halo)
    pm = forced_halo_rollout(mk, win[:, H - 1:], k, halo, halo_y=halo)
    assert torch.allclose(ph, pm, atol=1e-5)
