"""8a: ViT column autoencoder (site tokens, SwiGLU, relative-position attention)."""
import pytest
import torch

from ks_latent.cm1.models import ColumnAEConfig, SwiGLU, ViTColumnAE, build_autoencoder

C, Z, P_ = 7 * 4, 4, 4


def _ae(radius=1, **kw):
    torch.manual_seed(0)
    cfg = ColumnAEConfig(n_channels=C, n_levels=Z, patch=P_, latent_channels=6, kind="vit", vit_dim=32, vit_layers=2,
                         vit_heads=4, vit_mlp_ratio=4.0, vit_radius=radius, **kw)
    pm = torch.zeros(C, dtype=torch.bool); pm[-Z:] = True
    return build_autoencoder(cfg, pm).eval()


def test_shapes_and_positivity():
    ae = _ae()
    u = torch.randn(3, C, 40 * P_)
    uh, z = ae(u)
    assert z.shape == (3, 40, 6) and uh.shape == u.shape
    assert (uh[:, -Z:] >= 0).all()


def test_default_size_is_200M():
    cfg = ColumnAEConfig(n_channels=434, n_levels=62, latent_channels=32, kind="vit")
    n = sum(p.numel() for p in ViTColumnAE(cfg).parameters())
    assert 190e6 < n < 210e6, n
    assert cfg.reach_sites == 8


def test_swiglu_up_projection():
    m = SwiGLU(16, 64)
    assert m.w12.weight.shape == (128, 16) and m.w3.weight.shape == (16, 64)


@pytest.mark.parametrize("radius", [1, 2])
def test_windowed_encoder_is_local(radius):
    """A site's latent depends only on sites within layers x radius."""
    ae = _ae(radius)
    reach = ae.cfg.reach_sites
    u = torch.randn(1, C, 40 * P_)
    u2 = u.clone(); u2[..., (20 + reach + 1) * P_:] += 5.0
    with torch.no_grad():
        z, z2 = ae.encode(u), ae.encode(u2)
    assert torch.allclose(z[:, :21], z2[:, :21], atol=1e-5)
    assert not torch.allclose(z[:, 21:], z2[:, 21:])


def test_global_encoder_is_not_local():
    ae = _ae(0)
    u = torch.randn(1, C, 40 * P_)
    u2 = u.clone(); u2[..., 38 * P_:] += 5.0
    with torch.no_grad():
        assert not torch.allclose(ae.encode(u)[:, 0], ae.encode(u2)[:, 0])


@pytest.mark.parametrize("radius", [1, 2])
def test_shift_equivariance_interior(radius):
    """Shifting by whole sites shifts the latent, exactly at sites farther than the reach from the edges."""
    ae = _ae(radius)
    u = torch.randn(1, C, 60 * P_)
    s = 3
    with torch.no_grad():
        z, zs = ae.encode(u), ae.encode(torch.roll(u, s * P_, dims=-1))
    r = ae.cfg.reach_sites
    assert torch.allclose(torch.roll(z, s, dims=1)[:, s + r:60 - r], zs[:, s + r:60 - r], atol=1e-5)


def test_prefix_truncation():
    ae = _ae()
    ae.set_prefix(2, torch.full((6,), 0.5))
    z = ae.encode(torch.randn(1, C, 20 * P_))
    assert torch.all(z[..., 2:] == 0.5)


def test_overfit_one_batch():
    ae = _ae().train()
    u = torch.randn(4, C, 24 * P_)
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    l0 = None
    for _ in range(200):
        uh, _ = ae(u)
        loss = torch.nn.functional.mse_loss(uh[:, :-Z], u[:, :-Z])
        l0 = l0 or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3 * l0


def test_mlp_propagator_identity_local_stochastic():
    from ks_latent.cm1.models import StencilPropagator, StencilPropConfig
    torch.manual_seed(0)
    p = StencilPropagator(StencilPropConfig(latent_channels=5, hidden=16, layers=3, radius=1, body="mlp", noise_channels=2))
    z = torch.randn(2, 30, 5)
    assert torch.equal(p(z), z)                                    # zero-init head: identity at init
    with torch.no_grad():
        p.head.weight.normal_()
    eps = torch.randn(2, 2, 30)
    z2 = z.clone(); z2[:, 20:] += 4.0                               # beyond the reach (3 sites) of site <= 16
    assert torch.allclose(p(z, eps=eps)[:, :17], p(z2, eps=eps)[:, :17], atol=1e-6)
    assert not torch.allclose(p(z, eps=eps)[:, 17:], p(z2, eps=eps)[:, 17:])
    assert not torch.allclose(p(z), p(z))                           # fresh noise each call


def test_memmapped_members_match_loaded(tmp_path):
    import numpy as np
    from ks_latent.cm1.data import ColumnData, Normalizer, load_member, save_member, synthetic_members
    ms = synthetic_members(2, T=6, NX=64, Z=4)
    for m in ms:
        save_member(m, tmp_path / f"{m.name}.npz")
    eager = [load_member(tmp_path / f"{m.name}.npz") for m in ms]
    lazy = [load_member(tmp_path / f"{m.name}.npz", mmap=True) for m in ms]
    assert isinstance(lazy[0].X, np.memmap) and np.array_equal(np.asarray(lazy[0].X), eager[0].X)
    norm = Normalizer.fit(eager)
    de, dl = ColumnData(eager, norm, torch.device("cpu")), ColumnData(lazy, norm, torch.device("cpu"))
    assert dl.lazy and not de.lazy
    ge, gl = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    we, *_ = de.windows(3, 16, 2, ge)
    wl, *_ = dl.windows(3, 16, 2, gl)
    assert we.shape == wl.shape and torch.allclose(we, wl, rtol=2e-3, atol=2e-3)   # eager stores normalized float16
    assert torch.allclose(de.full(1, slice(1, 4)), dl.full(1, slice(1, 4)), rtol=2e-3, atol=2e-3)
    assert dl.full(0, 2).shape == (1, de.C, 64)


def test_attention_propagator_swiglu():
    from ks_latent.cm1.models import StencilPropagator, StencilPropConfig
    p = StencilPropagator(StencilPropConfig(latent_channels=5, hidden=16, layers=2, radius=1, body="attention",
                                            heads=4, attn_swiglu=True, noise_channels=1))
    assert isinstance(p.blocks[0].mlp, SwiGLU) and p.blocks[0].mlp.w3.weight.shape == (16, 64)
    z = torch.randn(2, 12, 5)
    assert torch.equal(p(z), z)


def test_stage_a_prediction_term():
    """8a: Stage A's prediction term trains the auxiliary propagator, sends gradient to the autoencoder, needs aux, and leaves
    w_pred = 0 runs unchanged."""
    import copy
    from ks_latent.cm1.data import ColumnData, Normalizer, synthetic_members
    from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder
    from ks_latent.cm1.train import StageAConfig, train_stage_a
    ms = synthetic_members(3, T=12, NX=64 * 4, Z=4)
    norm = Normalizer.fit(ms)
    d = ColumnData(ms, norm, torch.device("cpu"))
    def build():
        torch.manual_seed(0)
        cfg = ColumnAEConfig(n_channels=d.C, n_levels=4, patch=4, hidden=16, latent_channels=6, n_mix=1)
        return build_autoencoder(cfg, None)
    def aux_():
        torch.manual_seed(1)
        return StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, layers=2, radius=1))
    kw = dict(window_sites=24, batch=4, steps=6, val_every=3, val_batches=1, warmup=1)
    # w_pred = 0: identical to a run without aux
    a1, a2 = build(), build()
    train_stage_a(a1, d, d, StageAConfig(**kw), log=lambda *_: None)
    train_stage_a(a2, d, d, StageAConfig(**kw), log=lambda *_: None)
    assert all(torch.equal(p, q) for p, q in zip(a1.parameters(), a2.parameters()))
    with pytest.raises(ValueError):
        train_stage_a(build(), d, d, StageAConfig(w_pred=1.0, **kw), log=lambda *_: None)
    with pytest.raises(ValueError):
        train_stage_a(build(), d, d, StageAConfig(**kw), log=lambda *_: None, aux=aux_())
    # with the term: the auxiliary propagator learns (its zero-init head moves), the autoencoder differs from the no-term run
    ae, ax = build(), aux_()
    h0 = ax.head.weight.detach().clone()
    out = train_stage_a(ae, d, d, StageAConfig(w_pred=1.0, pred_k=2, pred_batch=2, **kw), log=lambda *_: None, aux=ax)
    assert not torch.equal(ax.head.weight, h0)
    assert any(not torch.equal(p, q) for p, q in zip(ae.parameters(), a1.parameters()))
    assert "train_pred" in out["history"][-1] and out["history"][-1]["train_pred"] > 0


def test_75M_configuration():
    from ks_latent.cm1.models import ColumnAEConfig, ViTColumnAE
    cfg = ColumnAEConfig(n_channels=434, n_levels=62, latent_channels=32, kind="vit", vit_dim=736, vit_heads=23)
    n = sum(p.numel() for p in ViTColumnAE(cfg).parameters())
    assert 73e6 < n < 76e6, n


def test_mixer_propagator():
    """8a: the Paper I/II local_mlp block as the CM1 propagator: identity at init, reach layers x radius, stochastic, ~0.75 M."""
    from ks_latent.cm1.models import StencilPropagator, StencilPropConfig
    torch.manual_seed(0)
    p = StencilPropagator(StencilPropConfig(latent_channels=5, hidden=16, layers=3, radius=1, body="mixer", noise_channels=2))
    z = torch.randn(2, 30, 5)
    assert torch.equal(p(z), z)
    with torch.no_grad():
        p.head.weight.normal_()
    eps = torch.randn(2, 2, 30)
    z2 = z.clone(); z2[:, 20:] += 4.0                              # beyond the reach (3 sites) of sites <= 16
    assert torch.allclose(p(z, eps=eps)[:, :17], p(z2, eps=eps)[:, :17], atol=1e-5)
    assert not torch.allclose(p(z, eps=eps)[:, 17:], p(z2, eps=eps)[:, 17:])
    assert not torch.allclose(p(z), p(z))
    big = StencilPropagator(StencilPropConfig(latent_channels=32, hidden=128, layers=3, radius=1, body="mixer", noise_channels=4))
    n = sum(q.numel() for q in big.parameters())
    base = sum(q.numel() for q in StencilPropagator(StencilPropConfig(latent_channels=32, hidden=256, layers=3, radius=1,
                                                                       noise_channels=4)).parameters())
    assert 1.5 * base < n < 2.2 * base, (n, base)
