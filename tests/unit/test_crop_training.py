"""Tests for ks_latent/training/crop.py: crop gathering wraps the periodic
wide domain correctly, and the forced-halo rollout's free interior is EXACTLY
what the same weights produce on a larger periodic domain (the property that
makes crop training equivalent to training at large L)."""

from __future__ import annotations

import pytest
import torch

from ks_latent.config import AuxPropagatorConfig
from ks_latent.models.propagator import AuxPropagator
from ks_latent.training.crop import _k_now, CropPropConfig, forced_halo_rollout, gather_x_crops


def test_gather_x_crops_wraps_periodically():
    NX, NY = 10, 3
    u = torch.arange(NX, dtype=torch.float32)[None, None, :, None].expand(2, 4, NX, NY).clone()
    out = gather_x_crops(u, torch.tensor([1]), torch.tensor([2]), torch.tensor([8]), 4)
    assert out.shape == (1, 4, NY)
    assert out[0, :, 0].tolist() == [8.0, 9.0, 0.0, 1.0]


def test_gather_x_crops_time_windows():
    u = torch.randn(2, 6, 12, 2)
    out = gather_x_crops(u, torch.tensor([0, 1]), torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([0, 5]), 4)
    assert out.shape == (2, 3, 4, 2)
    assert torch.equal(out[1, 2], u[1, 5, 5:9])


def _prop(n_sites_x, window=1):
    cfg = AuxPropagatorConfig(
        d_latent=n_sites_x * 4 * 3, mode="markovian", backbone="local_mlp_2d", attn_window=window,
        local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=4, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=8, local_mlp2d_n_layers=2, zero_init=False, dropout=0.0,
    )
    return AuxPropagator(cfg).eval()


def test_forced_halo_interior_matches_large_periodic_domain():
    torch.manual_seed(0)
    window, halo = 1, 2  # receptive field n_layers*window = 2 sites
    big = _prop(20, window)
    small = _prop(10, window)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 20, 4, 3)
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 20, 4, 3)
        strip = z_full[:, 5:15]
        win = torch.stack([strip, torch.zeros_like(strip)], dim=1)  # truth at step 1 unused for k=1
        pred = forced_halo_rollout(small, win, k=1, halo=halo)
    assert pred.shape == (1, 1, 10 - 2 * halo, 4, 3)
    assert torch.allclose(pred[0, 0], out_full[0, 5 + halo:15 - halo], atol=1e-6)


@pytest.mark.parametrize("halo", [2, 3])
def test_forced_halo_too_small_breaks_exactness(halo):
    """With halo < receptive field, the crop's circular padding leaks into the
    interior -- the reason halo must be >= n_layers*window. halo = 3 is the
    reach minus one (the case Appendix B of the paper describes), halo = 2 the
    reach minus two."""
    torch.manual_seed(1)
    big = _prop(20, window=2)  # receptive field 4 sites
    small = _prop(10, window=2)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 20, 4, 3)
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 20, 4, 3)
        strip = z_full[:, 5:15]
        pred = forced_halo_rollout(small, torch.stack([strip, strip], 1), k=1, halo=halo)
    assert not torch.allclose(pred[0, 0], out_full[0, 5 + halo:15 - halo], atol=1e-4)


def test_k_curriculum_endpoints():
    cfg = CropPropConfig()
    assert _k_now(0.0, cfg) == 2
    assert _k_now(cfg.k_mid_frac, cfg) == cfg.k_mid
    assert _k_now(0.99, cfg) == cfg.k_max


# --- C7: learned consensus backbone and rank penalty ---

def _cons(n_sites_x, zero_init=True):
    cfg = AuxPropagatorConfig(
        d_latent=n_sites_x * 4 * 3, mode="markovian", backbone="local_mlp_2d_consensus", attn_window=1,
        local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=4, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=8, local_mlp2d_n_layers=2, zero_init=zero_init, dropout=0.0,
        consensus_n_iter=3, consensus_radius=1,
    )
    return AuxPropagator(cfg).eval()


def test_consensus_identity_at_init():
    p = _cons(8)
    z = torch.randn(2, 8 * 4 * 3)
    assert torch.allclose(p.step_one(z), z, atol=1e-6)


def test_consensus_block_is_noop_at_init():
    """The consensus output projection is zero-initialized, so a freshly built
    consensus body must match plain local_mlp_2d with the same shared weights."""
    torch.manual_seed(0)
    cons = _cons(8, zero_init=False)
    plain = _prop(8, window=1)
    plain.load_state_dict({k: v for k, v in cons.state_dict().items() if "consensus" not in k})
    z = torch.randn(2, 8 * 4 * 3)
    with torch.no_grad():
        assert torch.allclose(cons.step_one(z), plain.step_one(z), atol=1e-6)


def test_consensus_exact_site_shift_equivariance():
    torch.manual_seed(1)
    p = _cons(8, zero_init=False)
    for blk in p.body.consensus:
        torch.nn.init.normal_(blk.out.weight, std=0.1)
    z = torch.randn(2, 8, 4, 3)
    zs = torch.roll(z, shifts=(3, 1), dims=(1, 2))
    with torch.no_grad():
        a = p.step_one(z.reshape(2, -1)).reshape(2, 8, 4, 3)
        b = p.step_one(zs.reshape(2, -1)).reshape(2, 8, 4, 3)
    assert torch.allclose(torch.roll(a, shifts=(3, 1), dims=(1, 2)), b, atol=1e-5)


def test_consensus_resizes_via_standard_loader(tmp_path):
    from ks_latent.models import load_propagator_checkpoint_resized_2d
    p = _cons(8, zero_init=False)
    path = tmp_path / "c.pt"
    torch.save({"prop_state_dict": p.state_dict(), "prop_config": p.cfg}, path)
    q, cfg, _ = load_propagator_checkpoint_resized_2d(str(path), new_n_sites_x=16, new_n_sites_y=4)
    assert cfg.d_latent == 16 * 4 * 3
    assert torch.isfinite(q.step_one(torch.randn(2, cfg.d_latent))).all()


def test_per_site_logdet_detects_rank_collapse():
    from ks_latent.training.crop import per_site_logdet
    torch.manual_seed(2)
    full = torch.randn(4, 10, 6, 3)
    collapsed = full[..., :1].expand(-1, -1, -1, 3).clone()  # all channels identical -> rank 1
    assert per_site_logdet(full) > per_site_logdet(collapsed) + 5.0


def test_consensus_forced_halo_exact_when_halo_covers_relaxation_reach():
    """Per-step reach = n_layers*(window + n_iter*radius) = 2*(1+3) = 8 sites."""
    torch.manual_seed(3)
    big, small = _cons(30, zero_init=False), _cons(20, zero_init=False)
    for blk in big.body.consensus:
        torch.nn.init.normal_(blk.out.weight, std=0.1)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 30, 4, 3)
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 30, 4, 3)
        strip = z_full[:, 5:25]
        pred = forced_halo_rollout(small, torch.stack([strip, strip], 1), k=1, halo=8)
    assert torch.allclose(pred[0, 0], out_full[0, 13:17], atol=1e-5)


# --- C9: global Biot-Savart velocity input, crop-consistent ---

def _bs_prop(n_sites_x, site_len=0.5, zero_init=False, zero_init_embed=False, vel_scale=1.0):
    cfg = AuxPropagatorConfig(
        d_latent=n_sites_x * 4 * 3, mode="markovian", backbone="local_mlp_2d_biotsavart", attn_window=1,
        local_mlp2d_n_sites_x=n_sites_x, local_mlp2d_n_sites_y=4, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=8, local_mlp2d_n_layers=2, zero_init=zero_init, dropout=0.0,
        biot_savart_Lx=n_sites_x * site_len, biot_savart_Ly=4 * site_len,
        biot_savart_vel_scale=vel_scale, biot_savart_zero_init_embed=zero_init_embed,
    )
    return AuxPropagator(cfg).eval()


def test_c9_strip_rollout_matches_full_domain_deployment():
    """Training path (strip + forced halo + velocity solved on the WIDE domain
    from truth anchors outside / predicted anchors inside) must equal the
    deployment path (the same weights on the full periodic domain, solving
    velocity from its own channel 0) at the free interior."""
    torch.manual_seed(4)
    big, small = _bs_prop(20, vel_scale=0.7), _bs_prop(10, vel_scale=0.7)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 20, 4, 3)
    anchors = z_full[..., 0]  # consistent: channel 0 IS the anchor field
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 20, 4, 3)
        strip = z_full[:, 5:15]
        win = torch.stack([strip, strip], dim=1)
        bs = (anchors[:, None].expand(1, 2, 20, 4), torch.tensor([5]), 20 * 0.5, 4 * 0.5)
        pred = forced_halo_rollout(small, win, k=1, halo=2, bs=bs)
    assert small.body.velocity_override is None  # cleared after rollout
    assert torch.allclose(pred[0, 0], out_full[0, 7:13], atol=1e-5)


def test_c9_global_velocity_differs_from_crop_only_solve():
    """Sanity: solving Biot-Savart on the crop alone (what D2 would do on a crop)
    gives a different velocity -- the reason C9 solves on the wide domain."""
    from ks_latent.models.propagator import _biot_savart_velocity
    from ks_latent.training.crop import global_velocity_for_strip
    torch.manual_seed(5)
    anchors = torch.randn(1, 20, 4, dtype=torch.float64)
    state = torch.zeros(1, 10, 4, 3, dtype=torch.float64)
    state[..., 0] = anchors[:, 5:15]
    v_global = global_velocity_for_strip(anchors, state, torch.tensor([5]), 2, 10.0, 2.0)
    v_crop = _biot_savart_velocity(anchors[:, 5:15], 5.0, 2.0)
    assert torch.allclose(v_global, _biot_savart_velocity(anchors, 10.0, 2.0)[:, 5:15])
    assert not torch.allclose(v_global, v_crop, atol=1e-3)


def test_c9_zero_init_embed_is_noop_and_d2_defaults_unchanged():
    torch.manual_seed(6)
    p = _bs_prop(8, zero_init=False, zero_init_embed=True)
    plain = _prop(8, window=1)
    plain.load_state_dict({k: v for k, v in p.state_dict().items() if "velocity_embed" not in k})
    z = torch.randn(2, 8 * 4 * 3)
    with torch.no_grad():
        assert torch.allclose(p.step_one(z), plain.step_one(z), atol=1e-6)
    d2 = _bs_prop(8)
    assert d2.body.vel_scale == 1.0 and d2.body.velocity_embed.weight.abs().sum() > 0


def test_aux_config_conversion_copies_every_shared_field():
    """Regression (2026-10-03): the AuxPropagatorConfig -> PropagatorConfig
    conversion used a hand-maintained list and dropped later fields (D2's
    resized biot_savart_Lx among them)."""
    import dataclasses
    from ks_latent.config import PropagatorConfig
    from ks_latent.models.propagator import aux_cfg_to_propagator_cfg
    cfg = AuxPropagatorConfig(
        d_latent=8 * 4 * 3, mode="markovian", backbone="local_mlp_2d_biotsavart", attn_window=1,
        local_mlp2d_n_sites_x=8, local_mlp2d_n_sites_y=4, local_mlp2d_local_channels=3,
        biot_savart_Lx=12.5, biot_savart_Ly=3.5, biot_savart_vel_scale=0.3, biot_savart_zero_init_embed=True,
        consensus_n_iter=7,
    )
    out = aux_cfg_to_propagator_cfg(cfg)
    shared = {f.name for f in dataclasses.fields(PropagatorConfig)} & {f.name for f in dataclasses.fields(AuxPropagatorConfig)}
    for name in shared:
        assert getattr(out, name) == getattr(cfg, name), name
    assert AuxPropagator(cfg).body.Lx == 12.5


def test_resized_biotsavart_aux_checkpoint_gets_new_Lx(tmp_path):
    """The D2 transfer bug: resizing an AuxPropagatorConfig checkpoint must
    change the body's Lx, not just the config's."""
    from ks_latent.models import load_propagator_checkpoint_resized_2d_biotsavart
    p = _bs_prop(8, site_len=0.5)
    path = tmp_path / "bs.pt"
    torch.save({"prop_state_dict": p.state_dict(), "prop_config": p.cfg}, path)
    q, cfg, _ = load_propagator_checkpoint_resized_2d_biotsavart(str(path), new_n_sites_x=16, new_n_sites_y=4)
    assert abs(q.body.Lx - 16 * 0.5) < 1e-9 and abs(cfg.biot_savart_Lx - 8.0) < 1e-9


# --- P7: 2D crops (isotropic systems) ---

def test_gather_xy_crops_wraps_both_axes():
    from ks_latent.training.crop import gather_xy_crops
    NX, NY = 10, 6
    u = (torch.arange(NX)[:, None] * 100 + torch.arange(NY)[None, :]).float()[None, None].expand(2, 3, NX, NY).clone()
    out = gather_xy_crops(u, torch.tensor([1]), torch.tensor([2]), torch.tensor([8]), torch.tensor([5]), 3, 3)
    assert out.shape == (1, 3, 3)
    assert out[0, :, 0].tolist() == [805.0, 905.0, 5.0] and out[0, 0, :].tolist() == [805.0, 800.0, 801.0]
    win = gather_xy_crops(u, torch.tensor([0]), torch.tensor([[0, 2]]), torch.tensor([9]), torch.tensor([0]), 2, 2)
    assert win.shape == (1, 2, 2, 2) and win[0, 1, 1, 1].item() == 1.0


def _prop2d(nx, ny, window=1):
    cfg = AuxPropagatorConfig(
        d_latent=nx * ny * 3, mode="markovian", backbone="local_mlp_2d", attn_window=window,
        local_mlp2d_n_sites_x=nx, local_mlp2d_n_sites_y=ny, local_mlp2d_local_channels=3,
        local_mlp2d_token_d_model=8, local_mlp2d_n_layers=2, zero_init=False, dropout=0.0,
    )
    return AuxPropagator(cfg).eval()


def test_forced_halo_2d_interior_matches_large_periodic_domain():
    torch.manual_seed(7)
    big, small = _prop2d(20, 12), _prop2d(10, 8)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 20, 12, 3)
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 20, 12, 3)
        strip = z_full[:, 5:15, 2:10]
        pred = forced_halo_rollout(small, torch.stack([strip, strip], 1), k=1, halo=2, halo_y=2)
    assert pred.shape == (1, 1, 6, 4, 3)
    assert torch.allclose(pred[0, 0], out_full[0, 7:13, 4:8], atol=1e-5)


def test_forced_halo_2d_needs_y_halo():
    """With an x-halo only, the crop's y wraparound leaks into the interior."""
    torch.manual_seed(8)
    big, small = _prop2d(20, 12), _prop2d(10, 8)
    small.load_state_dict(big.state_dict())
    z_full = torch.randn(1, 20, 12, 3)
    with torch.no_grad():
        out_full = big.step_one(z_full.reshape(1, -1)).reshape(1, 20, 12, 3)
        strip = z_full[:, 5:15, 2:10]
        pred = forced_halo_rollout(small, torch.stack([strip, strip], 1), k=1, halo=2, halo_y=0)
    # interior y-sites >= reach from the crop's y-edges are still exact; the edge sites are not
    assert torch.allclose(pred[0, 0, :, 2:6], out_full[0, 7:13, 4:8], atol=1e-5)
    assert not torch.allclose(pred[0, 0, :, [0, 1, 6, 7]], out_full[0, 7:13][:, [2, 3, 8, 9]], atol=1e-4)


def test_crop_ae_and_bank_2d_smoke():
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
    from ks_latent.training.crop import CropAEConfig, encode_crop_bank, train_crop_ae
    torch.manual_seed(9)
    u = torch.randn(2, 6, 16 * 12, 16 * 12)  # wide 12x12-site periodic domain
    ae = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(
        NX=16 * 8, NY=16 * 8, n_sites_x=8, n_sites_y=8, local_channels=3, site_mix_radius=1,
        n_site_mix_layers=1, hidden=8))
    cfg = CropAEConfig(crop_sites=8, crop_sites_y=8, mask_sites=2, enc_sites=1, batch=2, steps=2, val_every=1)
    res = train_crop_ae(ae, u[:1], u[1:], cfg, log=lambda s: None)
    assert res["best_val_recon_interior"] > 0
    bank = encode_crop_bank(ae, u, n_offsets=2, cfg_crop_sites=8, enc_sites=1, seed=0, crop_sites_y=8)
    assert bank.shape == (4, 6, 6, 6, 3)


def test_c19_decoder_reach_independent_of_encoder():
    """C19: dec_site_mix_radius/dec_n_site_mix_layers set the decoder's reach
    separately; reach 0 means a site's latent only affects its own patch."""
    import torch
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
    for dr, dl in ((0, 0), (1, 2)):
        cfg = LocalFieldAutoencoderConfig2D(NX=16 * 12, NY=16 * 4, n_sites_x=12, n_sites_y=4, site_mix_radius=2,
                                            n_site_mix_layers=3, dec_site_mix_radius=dr, dec_n_site_mix_layers=dl)
        torch.manual_seed(0)
        ae = KSAutoencoderLocalField2D(cfg).double()
        z = torch.randn(1, 12 * 4 * 3, dtype=torch.float64)
        dz = torch.zeros_like(z).reshape(1, 12, 4, 3)
        dz[0, 6, :, 1] = 1.0
        diff = (ae.decode(z + dz.reshape(1, -1)) - ae.decode(z)).reshape(12, 16, 64).abs().amax(dim=(1, 2))
        touched = set(torch.nonzero(diff > 1e-12).flatten().tolist())
        assert touched <= set(range(6 - dr * dl, 6 + dr * dl + 1)), (dr, dl, touched)
        assert 6 in touched


def test_p9_y_align_offsets_are_multiples():
    """P9: with y_align, every 2D crop-bank y-offset is a multiple of y_align
    (so each crop sees the forcing cos(4y) at the same phase)."""
    import torch
    from ks_latent.training import crop as C
    seen = []
    orig = C.gather_xy_crops

    def spy(u, n, t, x0, y0, Wc, Wcy):
        seen.extend(y0.tolist())
        return orig(u, n, t, x0, y0, Wc, Wcy)

    C.gather_xy_crops = spy
    try:
        from ks_latent.config import LocalFieldAutoencoderConfig2D
        from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
        ae = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(NX=64, NY=64, n_sites_x=4, n_sites_y=4,
                                                                      site_mix_radius=1, n_site_mix_layers=1))
        u = torch.randn(1, 3, 128, 128)
        C.encode_crop_bank(ae, u, 6, 4, 1, seed=0, crop_sites_y=4, y_align=32)
    finally:
        C.gather_xy_crops = orig
    assert seen and all(y % 32 == 0 for y in seen)


def test_spectral_band_shares_recovers_known_mode():
    """A single-mode field concentrated at |k|~k0 should land ~entirely in the
    band containing k0 and ~0 in a disjoint band, and shares must sum to 1."""
    import math
    import torch
    from ks_latent.training.crop import spectral_band_shares
    NX = NY = 64
    Lx = Ly = 2 * math.pi * 8  # k0 below in band units
    x = torch.arange(NX, dtype=torch.float64) * Lx / NX
    y = torch.arange(NY, dtype=torch.float64) * Ly / NY
    X, Y = torch.meshgrid(x, y, indexing="ij")
    k0 = 2 * math.pi / Lx * 4  # a mode with streamwise wavenumber index 4
    field = torch.cos(k0 * X)[None]  # (1, NX, NY), single x-wavenumber
    shares = spectral_band_shares(field, Lx, Ly, ((0.0, k0 - 0.05), (k0 - 0.05, k0 + 0.05), (k0 + 0.05, 999.0)))
    assert shares[1] > 0.95
    assert shares[0] < 1e-6 and shares[2] < 1e-6
    assert abs(shares.sum().item() - 1.0) < 1e-6


def test_spectral_band_shares_differentiable():
    import torch
    from ks_latent.training.crop import spectral_band_shares
    field = torch.randn(2, 3, 16, 16, requires_grad=True, dtype=torch.float64)
    shares = spectral_band_shares(field, 10.0, 10.0, ((0.0, 1.0), (1.0, 5.0)))
    shares.sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()


def test_spectral_regularizer_asserts_on_nonzero_decoder_reach():
    """_resize_decoder_for_spectral_reg must refuse a nonzero-reach decoder
    (would silently introduce circular-wraparound artifacts at the grid edge)."""
    import pytest
    import torch
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
    from ks_latent.training.crop import _resize_decoder_for_spectral_reg
    ae = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(
        NX=64, NY=64, n_sites_x=4, n_sites_y=4, site_mix_radius=2, n_site_mix_layers=3))
    with pytest.raises(AssertionError):
        _resize_decoder_for_spectral_reg(ae, 6, 6)
    # zero reach: must not raise
    ae0 = KSAutoencoderLocalField2D(LocalFieldAutoencoderConfig2D(
        NX=64, NY=64, n_sites_x=4, n_sites_y=4, site_mix_radius=0, n_site_mix_layers=0, local_channels=6))
    resized = _resize_decoder_for_spectral_reg(ae0, 6, 6)
    assert resized.cfg.n_sites_x == 6 and resized.cfg.n_sites_y == 6


def test_train_crop_prop_spectral_regularizer_runs_and_changes_loss():
    """Smoke test: w_spectral>0 with a zero-reach AE runs end-to-end (decode
    inside the training loop, gradient flows, no crash) and the spectral
    target/loss machinery actually engages (reported in the training log)."""
    import torch
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    from ks_latent.models import AuxPropagatorConfig, AuxPropagator
    from ks_latent.models.autoencoder_local_field_2d import KSAutoencoderLocalField2D
    from ks_latent.training.crop import CropPropConfig, encode_crop_bank, train_crop_prop
    torch.manual_seed(0)
    ae_cfg = LocalFieldAutoencoderConfig2D(NX=16 * 10, NY=16 * 10, n_sites_x=10, n_sites_y=10,
                                          site_mix_radius=0, n_site_mix_layers=0, local_channels=4)
    ae = KSAutoencoderLocalField2D(ae_cfg).double().eval()
    for q in ae.parameters():
        q.requires_grad_(False)
    u = torch.randn(2, 6, 160, 160, dtype=torch.float64)
    bank_tr = encode_crop_bank(ae, u, n_offsets=2, cfg_crop_sites=10, enc_sites=0, seed=0)
    bank_va = encode_crop_bank(ae, u, n_offsets=2, cfg_crop_sites=10, enc_sites=0, seed=1)
    Pu, Pv, c = bank_tr.shape[2], bank_tr.shape[3], ae_cfg.local_channels
    prop_cfg = AuxPropagatorConfig(d_latent=Pu * Pv * c, mode="markovian", backbone="local_mlp_2d", attn_window=1,
                                   local_mlp2d_n_sites_x=Pu, local_mlp2d_n_sites_y=Pv, local_mlp2d_local_channels=c,
                                   local_mlp2d_token_d_model=8, local_mlp2d_n_layers=1)
    prop = AuxPropagator(prop_cfg).double()
    pcfg = CropPropConfig(halo_sites=2, steps=4, batch=4, k_max=2, k_mid=2, val_every=2,
                          w_spectral=1.0, spectral_bands=((0.0, 1.0), (1.0, 3.0)))
    logs = []
    res = train_crop_prop(prop, bank_tr, bank_va, pcfg, log=logs.append, ae=ae, site_len=1.0)
    assert "best_val_kmax_mse" in res
    assert any("spectral target shares" in m for m in logs)
