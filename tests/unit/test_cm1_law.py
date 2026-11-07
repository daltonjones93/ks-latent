"""FPc: the structured latent PDE propagator (body 'law') and nested-dropout channel ordering."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.data import VARS, CM1Member, ColumnData, Normalizer, positive_channel_mask
from ks_latent.cm1.models import (ColumnAEConfig, ColumnFieldAE, LatentLaw, StencilPropagator, StencilPropConfig,
                                  forced_halo_rollout)
from ks_latent.cm1.train import nested_truncate

pytestmark = pytest.mark.unit
V, Z, PATCH = len(VARS), 4, 4


def _law_prop(c=3, **kw):
    return StencilPropagator(StencilPropConfig(latent_channels=c, body="law", **kw))


def test_identity_at_init_and_reach_by_integrator():
    z = torch.randn(2, 20, 3, dtype=torch.float64)
    for kw, reach in (({"law_integrator": "heun"}, 2), ({"law_integrator": "euler"}, 1),
                      ({"law_integrator": "euler", "law_substeps": 3}, 3)):
        p = _law_prop(**kw).double()
        torch.testing.assert_close(p(z), z)
        assert p.cfg.reach_sites == reach
        for t in p.law.coefficient_tensors().values():
            torch.nn.init.normal_(t, std=0.1)
        torch.nn.init.normal_(p.law.R[-1].weight, std=0.1)
        zz = z.clone().requires_grad_(True)
        p(zz)[0, 10].sum().backward()
        nz = zz.grad[0].abs().sum(-1).nonzero()[:, 0]
        assert nz.min() == 10 - reach and nz.max() == 10 + reach


def test_recovers_a_known_linear_latent_pde():
    """Data from dz/dt = A1 d1z + A2 d2z (advection and diffusion with channel coupling), stepped with the same
    integrator: fitting the law with no quadratic or residual part recovers A1 and A2."""
    torch.manual_seed(0)
    c, P = 2, 40
    true = LatentLaw(c, quadratic=False, residual_hidden=0, integrator="heun").double()
    with torch.no_grad():
        true.A[1] = torch.tensor([[-0.4, 0.1], [0.0, 0.3]])
        true.A[2] = torch.tensor([[0.2, 0.0], [0.05, 0.1]])
    x = torch.linspace(0, 6.28, P, dtype=torch.float64)
    z0 = torch.stack([torch.stack([torch.sin(k * x + ph), torch.cos(0.5 * k * x - ph)], -1)
                      for k, ph in ((1.0, 0.3), (2.0, 1.0), (3.0, -0.5), (1.5, 2.0))])
    with torch.no_grad():
        z1 = true(z0)
    fit = LatentLaw(c, quadratic=False, residual_hidden=0, integrator="heun").double()
    opt = torch.optim.Adam(fit.parameters(), lr=0.02)
    for _ in range(1500):
        loss = ((fit(z0) - z1)[:, 2:-2] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.testing.assert_close(fit.A[1].detach(), true.A[1], atol=2e-3, rtol=0)
    torch.testing.assert_close(fit.A[2].detach(), true.A[2], atol=2e-3, rtol=0)


def test_penalty_l1_exempts_the_diagonal_and_residual_share_accumulates_only_in_training():
    p = _law_prop(c=2, law_l1=0.5, law_res_weight=1.0)
    with torch.no_grad():
        p.law.A[0] = torch.tensor([[3.0, -1.0], [2.0, 4.0]])
        p.law.A[1, 0, 0] = 0.5
    expected = 0.5 * (1.0 + 2.0 + 0.5)
    assert abs(p.penalty().item() - expected) < 1e-6
    torch.nn.init.normal_(p.law.R[-1].weight, std=0.5)
    p.eval(); p(torch.randn(1, 8, 2))
    assert not p.law._res
    p.train(); p(torch.randn(1, 8, 2))
    assert p.law._res
    assert p.penalty().item() > expected
    assert not p.law._res


def test_prune_keeps_zeros_through_training_and_checkpoints(tmp_path):
    torch.manual_seed(1)
    p = _law_prop(c=3)
    for t in p.law.coefficient_tensors().values():
        torch.nn.init.normal_(t, std=0.1)
    kept = p.law.prune(0.1)
    assert kept == p.law.n_terms()
    zero_before = {k: (v == 0) for k, v in p.law.coefficient_tensors().items()}
    opt = torch.optim.Adam(p.parameters(), lr=0.05)
    for _ in range(5):
        loss = ((p(torch.randn(2, 10, 3)) - 1) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    eff = {k: v * p.law._mask(k) for k, v in p.law.coefficient_tensors().items()}
    for k in eff:
        assert (eff[k][zero_before[k]] == 0).all()
    torch.save(p.state_dict(), tmp_path / "p.pt")
    q = _law_prop(c=3)
    q.load_state_dict(torch.load(tmp_path / "p.pt"))
    z = torch.randn(1, 10, 3)
    torch.testing.assert_close(q(z), p(z))


def test_nested_truncate_keeps_a_prefix_of_channels():
    g = torch.Generator().manual_seed(0)
    z = torch.ones(64, 5, 6)
    assert torch.equal(nested_truncate(z, 0.0, g), z)
    t = nested_truncate(z, 1.0, g)
    k = t[:, 0].sum(-1)
    assert ((k >= 1) & (k <= 6)).all() and len(k.unique()) > 3
    for b in range(64):
        n = int(k[b])
        assert t[b, :, :n].eq(1).all() and t[b, :, n:].eq(0).all()


def _member(name, T=10, seed=0, NXm=96):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, NXm, V, Z)).astype(np.float32)
    X[:, :, [VARS.index(v) for v in ("qliq", "qice")]] = np.abs(X[:, :, [VARS.index(v) for v in ("qliq", "qice")]])
    cref = np.zeros((T, NXm), np.float32)
    cref[:, NXm // 2 - 3:NXm // 2 + 3] = 30.0
    return CM1Member(name=name, X=X.astype(np.float16), cref=cref, t=np.arange(T) * 60.0)


def test_law_trains_in_stages_b_and_c_with_nested_dropout():
    from ks_latent.cm1.train import StageAConfig, StageBConfig, StageCConfig, train_stage_a, train_stage_b, train_stage_c
    torch.manual_seed(0)
    ms = [_member(f"m{i}", seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms)
    tr, va = ColumnData(ms[:2], nrm, torch.device("cpu")), ColumnData(ms[2:], nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    ra = train_stage_a(ae, tr, va, StageAConfig(window_sites=14, batch=2, steps=4, val_every=2, warmup=1, val_batches=1,
                                                nested_frac=0.5), log=lambda s: None)
    assert np.isfinite(ra["best_val_recon"])
    prop = _law_prop(c=3, law_l1=1e-3, law_res_weight=0.1)
    z = [torch.randn(10, 24, 3) for _ in range(2)]
    act = [torch.zeros((0, 2), dtype=torch.long)] * 2
    rb = train_stage_b(prop, z, z, act, act, PATCH, StageBConfig(window_sites=12, halo_sites=2, batch=4, steps=4,
                                                                 k_max=3, val_every=2, warmup=1), log=lambda s: None)
    assert np.isfinite(rb["best_val_mse"])
    p0 = {k: v.clone() for k, v in prop.state_dict().items()}
    rc = train_stage_c(ae, prop, tr, va, StageCConfig(window_sites=14, halo_sites=2, batch=2, steps=4, k_max=3,
                                                      val_every=2, warmup=1, w_sfc_th=0.0, nested_frac=0.5),
                       log=lambda s: None)
    assert np.isfinite(rc["best_val_decoded_rollout_mse"])
    assert any((prop.state_dict()[k] != p0[k]).any() for k in p0)
    out = forced_halo_rollout(prop, torch.randn(2, 4, 24, 3), 3, 2)
    assert out.shape == (2, 3, 24, 3) and torch.isfinite(out).all()


def test_law_rejects_history_and_noise():
    with pytest.raises(ValueError):
        _law_prop(n_hist=2)
    with pytest.raises(ValueError):
        _law_prop(noise_channels=2)
