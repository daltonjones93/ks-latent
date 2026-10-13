"""R15 (2026-10-13) Kolmogorov small-box options: (1) velocity input fields for the local-field AE,
(2) the larger-domain statistics regularizer. Both are off by default; the defaults must reproduce the
earlier code exactly."""
import math

import pytest
import torch
import torch.nn.functional as F

from ks_latent.config import AuxPropagatorConfig, LocalFieldAutoencoderConfig2D
from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
from ks_latent.models.kolmogorov_fields import (encode_input, field0, pack_wuv, pack_wuv_chunked,
                                                velocity_from_vorticity)
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.crop import CropPropConfig, LargeDomainStats, train_crop_prop

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- option 1: velocity input fields
def test_velocity_matches_analytic_single_mode():
    """w = sin(2x) cos(1.5y) on [0,2pi) x [0,4pi): psi = w / 6.25, u = psi_y, v = -psi_x."""
    Lx, Ly, NX, NY = 2 * math.pi, 4 * math.pi, 32, 64
    x = torch.arange(NX, dtype=torch.float64)[:, None] * Lx / NX
    y = torch.arange(NY, dtype=torch.float64)[None, :] * Ly / NY
    w = torch.sin(2 * x) * torch.cos(1.5 * y)
    u, v = velocity_from_vorticity(w.unsqueeze(0), Lx, Ly)
    assert torch.allclose(u[0], -1.5 * torch.sin(2 * x) * torch.sin(1.5 * y) / 6.25, atol=1e-12)
    assert torch.allclose(v[0], -2.0 * torch.cos(2 * x) * torch.cos(1.5 * y) / 6.25, atol=1e-12)


def _old_encode_decode(ae, u_flat):
    """The pre-R15 single-field encode/decode, verbatim, as the reference."""
    cfg = ae.cfg
    B = u_flat.shape[0]
    Px, Py, c = cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels
    u = u_flat.reshape(B, cfg.NX, cfg.NY)
    h = F.gelu(ae.enc_patchify(u.unsqueeze(1)))
    for conv in ae.enc_mix:
        h = F.gelu(conv(h))
    anchor = u.reshape(B, Px, cfg.patch_x, Py, cfg.patch_y).mean(dim=(2, 4))
    z = torch.cat([anchor.unsqueeze(1), ae.enc_out(h)], dim=1).permute(0, 2, 3, 1).reshape(B, -1)
    hd = F.gelu(ae.dec_in(z.reshape(B, Px, Py, c).permute(0, 3, 1, 2)))
    for conv in ae.dec_mix:
        hd = F.gelu(conv(hd))
    return z, ae.dec_unpatchify(hd).squeeze(1).reshape(B, cfg.NX * cfg.NY)


def test_single_field_path_unchanged_including_old_pickled_config():
    torch.manual_seed(0)
    cfg = LocalFieldAutoencoderConfig2D(NX=64, NY=64, n_sites_x=4, n_sites_y=4, local_channels=3,
                                        site_mix_radius=1, n_site_mix_layers=1, hidden=8)
    del cfg.__dict__["n_fields"]  # a config pickled before R15 has no such attribute
    ae = KSAutoencoderLocalField2D(cfg).double()
    assert ae.enc_patchify.weight.shape[1] == 1 and ae.dec_unpatchify.weight.shape[1] == 1
    u = torch.randn(3, 64 * 64, dtype=torch.float64)
    z_ref, uh_ref = _old_encode_decode(ae, u)
    assert torch.equal(ae.encode(u), z_ref)
    assert torch.equal(ae.decode(z_ref), uh_ref)


def _ae3():
    cfg = LocalFieldAutoencoderConfig2D(NX=64, NY=64, n_sites_x=4, n_sites_y=4, local_channels=3,
                                        site_mix_radius=0, n_site_mix_layers=0, hidden=8, n_fields=3)
    return KSAutoencoderLocalField2D(cfg).double()


def test_multifield_shapes_anchor_and_x_equivariance():
    torch.manual_seed(1)
    ae = _ae3()
    packed = torch.randn(2, 64, 3 * 64, dtype=torch.float64)  # (B, NX, 3*NY): fields along y
    z = ae.encode(packed.reshape(2, -1))
    assert z.shape == (2, 4 * 4 * 3)
    w = packed[:, :, :64]
    anchor = w.reshape(2, 4, 16, 4, 16).mean(dim=(2, 4))
    assert torch.allclose(z.reshape(2, 4, 4, 3)[..., 0], anchor, atol=1e-12)  # channel 0 = patch mean of omega
    assert ae.decode(z).shape == (2, 64 * 3 * 64)
    z_roll = ae.encode(torch.roll(packed, 16, dims=1).reshape(2, -1)).reshape(2, 4, 4, 3)
    assert torch.allclose(z_roll, torch.roll(z.reshape(2, 4, 4, 3), 1, dims=1), atol=1e-12)
    assert field0(ae.decode(z), 64, 64, 3).shape == (2, 64, 64)


def test_encode_input_packing():
    torch.manual_seed(2)
    w = torch.randn(64, 64, dtype=torch.float64).numpy()
    ae1 = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(NX=64, NY=64, n_sites_x=4, n_sites_y=4))
    assert torch.equal(encode_input(ae1, w, {}, 2 * math.pi, 2 * math.pi),
                       torch.tensor(w.reshape(1, -1), dtype=torch.float32))
    ck = {"input_fields": "wuv", "normalization_mean": 0.1, "normalization_std": 2.0, "velocity_std": [0.5, 0.25]}
    x = encode_input(_ae3(), w, ck, 2 * math.pi, 2 * math.pi).reshape(64, 3 * 64).double()
    u, v = velocity_from_vorticity(torch.tensor(w) * 2.0 + 0.1, 2 * math.pi, 2 * math.pi)
    assert torch.allclose(x[:, :64], torch.tensor(w), atol=1e-6)
    assert torch.allclose(x[:, 64:128], u / 0.5, atol=1e-5)
    assert torch.allclose(x[:, 128:], v / 0.25, atol=1e-5)


def test_pack_chunked_matches_direct_and_uses_training_rows_for_std():
    torch.manual_seed(3)
    traj = torch.randn(3, 5, 32, 32)
    out, (su, sv) = pack_wuv_chunked(traj, 0.0, 1.5, 2 * math.pi, 2 * math.pi, n_std=2, chunk=4)
    assert out.shape == (3, 5, 32, 96)
    u, v = velocity_from_vorticity(traj[:2].double() * 1.5, 2 * math.pi, 2 * math.pi)
    assert su == pytest.approx(float(u.std()), rel=1e-5) and sv == pytest.approx(float(v.std()), rel=1e-5)
    direct = pack_wuv(traj.double(), 0.0, 1.5, 2 * math.pi, 2 * math.pi, (su, sv))
    assert torch.allclose(out.double(), direct, atol=1e-5)


# ---------------------------------------------------------------- option 2: larger-domain stats regularizer
def _prop_and_bank(seed=0):
    torch.manual_seed(seed)
    c = 3
    ae = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(
        NX=64, NY=64, n_sites_x=4, n_sites_y=4, local_channels=c, site_mix_radius=0, n_site_mix_layers=0,
        hidden=8))
    pcfg = AuxPropagatorConfig(d_latent=4 * 4 * c, mode="markovian", n_history=3, backbone="local_mlp_2d",
                               attn_window=1, local_mlp2d_n_sites_x=4, local_mlp2d_n_sites_y=4,
                               local_mlp2d_local_channels=c, local_mlp2d_token_d_model=8, local_mlp2d_n_layers=1,
                               zero_init=False)
    prop = AuxPropagator(pcfg)
    bank = 0.5 * torch.randn(3, 12, 4, 4, c)
    return ae, prop, bank


def test_stats_regularizer_gradient_reaches_propagator():
    ae, prop, bank = _prop_and_bank()
    for q in ae.parameters():
        q.requires_grad_(False)
    cfg = CropPropConfig(halo_sites=1, w_stats=1.0, stats_pool=3, stats_burn=2, stats_k=2)
    st = LargeDomainStats(prop, ae, bank, cfg, site_len=2 * math.pi / 4)
    assert st.big.cfg.local_mlp2d_n_sites_x == 8 and st.pool.shape == (3, 8, 4, 3)
    loss = st.loss(prop)
    assert torch.isfinite(loss)
    loss.backward()
    g = [q.grad for q in prop.parameters()]
    assert all(x is not None for x in g) and sum(float(x.abs().sum()) for x in g) > 0
    assert all(q.grad is None for q in st.big.parameters())  # only the trained propagator's params get grads


def _train(**kw):
    ae, prop, bank = _prop_and_bank(seed=5)
    for q in ae.parameters():
        q.requires_grad_(False)
    cfg = CropPropConfig(halo_sites=1, batch=4, steps=6, k_max=3, k_mid=2, val_every=3, warmup=2, **kw)
    torch.manual_seed(11)
    res = train_crop_prop(prop, bank, bank, cfg, log=lambda s: None, ae=ae, site_len=2 * math.pi / 4)
    return prop.state_dict(), res


def test_stats_weight_zero_reproduces_old_training_exactly():
    sd0, r0 = _train()
    sd1, r1 = _train(w_stats=0.0, stats_scale=3, stats_every=1, stats_pool=5)
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    assert r0["best_val_kmax_mse"] == r1["best_val_kmax_mse"]
    sd2, r2 = _train(w_stats=1.0, stats_every=1, stats_pool=3, stats_burn=1, stats_k=1)
    assert "stats_var" in r2["history"][-1]
    assert not all(torch.equal(sd0[k], sd2[k]) for k in sd0)  # the term is live when on
