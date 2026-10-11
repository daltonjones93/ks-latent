"""1D window training (ks_latent/training/crop1d.py): forced-halo exactness, bank layout, window
gathering across the periodic seam, and size independence of the saved checkpoints."""
import pytest
import torch

from ks_latent.config import LocalFieldAutoencoderConfig, PropagatorConfig
from ks_latent.models import build_autoencoder, build_propagator_from_config
from ks_latent.training.crop import forced_halo_rollout
from ks_latent.training.crop1d import (encode_crop_bank_1d, gather_crops_1d, resized_ae_config,
                                       resized_prop_config)

pytestmark = pytest.mark.unit
C, W, LAYERS = 3, 3, 2  # channels, mixer radius, blocks: reach 6 sites


def _prop(P):
    cfg = PropagatorConfig(d_latent=P * C, mode="markovian", backbone="local_mlp", n_tokens=P, token_d_model=8,
                           token_n_layers=LAYERS, token_mlp_ratio=2, attn_window=W, zero_init=False)
    return build_propagator_from_config(cfg).double().eval(), cfg


def _ae(P, mix_layers=0):
    cfg = LocalFieldAutoencoderConfig(NX=16 * P, n_sites=P, local_channels=C, site_mix_radius=1,
                                      n_site_mix_layers=mix_layers, hidden=8)
    return build_autoencoder("local_field", cfg).double().eval(), cfg


def _big_domain_forced(prop_big, z_true, x0, Pu, halo, k):
    """Propagator on the large periodic domain, every site outside the window interior set to
    its true value before every step; returns the window interior."""
    state = z_true[0].clone()
    lo, hi = x0 + halo, x0 + Pu - halo
    out = []
    for j in range(k):
        inp = z_true[j].clone()
        inp[:, lo:hi] = state[:, lo:hi]
        state = prop_big.step_one(inp.reshape(1, -1)).reshape(1, -1, C)
        out.append(state[:, lo:hi])
    return torch.stack(out, 1)


@pytest.mark.parametrize("halo,exact", [(6, True), (5, False)])
def test_forced_halo_exact_in_1d(halo, exact):
    torch.manual_seed(0)
    P_big, Pu, k, x0 = 40, 20, 4, 7
    prop_big, cfg = _prop(P_big)
    prop_win = build_propagator_from_config(resized_prop_config(cfg, Pu, C)).double().eval()
    prop_win.load_state_dict(prop_big.state_dict())
    z_true = torch.randn(k + 1, 1, P_big, C, dtype=torch.float64)
    ref = _big_domain_forced(prop_big, z_true, x0, Pu, halo, k)
    win = z_true[:, :, x0:x0 + Pu].permute(1, 0, 2, 3).unsqueeze(3)  # (1, k+1, Pu, 1, C)
    with torch.no_grad():
        got = forced_halo_rollout(prop_win, win, k, halo).squeeze(3)
    err = (got - ref).abs().max().item()
    assert (err < 1e-10) == exact, err


def test_gather_wraps_periodic_seam():
    u = torch.arange(2 * 3 * 10, dtype=torch.float64).reshape(2, 3, 10)
    w = gather_crops_1d(u, torch.tensor([1]), torch.tensor([2]), torch.tensor([8]), 4)
    assert w.tolist() == [[u[1, 2, 8].item(), u[1, 2, 9].item(), u[1, 2, 0].item(), u[1, 2, 1].item()]]


def test_bank_layout_matches_direct_encoding():
    torch.manual_seed(1)
    ae, _ = _ae(8)
    u = torch.randn(2, 5, 16 * 30, dtype=torch.float64)
    bank = encode_crop_bank_1d(ae, u, n_offsets=3, crop_sites=8, enc_sites=1, seed=4, batch=2)
    assert bank.shape == (6, 5, 6, 1, C)
    x0 = torch.randint(0, u.shape[2], (3,), generator=torch.Generator().manual_seed(4)).tolist()[0]
    win = gather_crops_1d(u, torch.zeros(5, dtype=torch.long), torch.arange(5), torch.full((5,), x0), 16 * 8)
    direct = ae.encode(win).reshape(5, 8, C)[:, 1:7]
    assert torch.allclose(bank[0, :, :, 0], direct)


def test_saved_reference_size_is_the_same_per_patch_model():
    """Per-patch AE trained at the window size and saved at 16 sites: identical encodings of the
    same patches (the weights do not depend on the number of sites)."""
    torch.manual_seed(2)
    ae, cfg = _ae(40)
    ref = build_autoencoder("local_field", resized_ae_config(cfg, 16)).double().eval()
    ref.load_state_dict(ae.state_dict())
    u = torch.randn(3, 16 * 40, dtype=torch.float64)
    z_big = ae.encode(u).reshape(3, 40, C)[:, :16]
    z_ref = ref.encode(u[:, :256]).reshape(3, 16, C)
    assert torch.allclose(z_big, z_ref, atol=1e-12)
    assert torch.allclose(ae.decode(ae.encode(u))[:, :256], ref.decode(ref.encode(u[:, :256])), atol=1e-12)
