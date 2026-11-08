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


def test_global_mean_term_couples_every_site_and_is_penalized_and_pruned():
    p = _law_prop(c=2, law_global=True, law_l1=1.0, law_residual_hidden=0, law_quadratic=False,
                  law_integrator="euler").double()    # one Euler stage: the far-site effect is exactly G d<z>
    z = torch.randn(1, 30, 2, dtype=torch.float64)
    torch.testing.assert_close(p(z), z)                        # identity at init
    with torch.no_grad():
        p.law.G[:] = torch.tensor([[0.5, 0.0], [0.0, -0.2]])
    assert abs(p.penalty().item() - 0.7) < 1e-6
    z2 = z.clone(); z2[0, 29, 0] += 3.0                         # change a far site
    d = (p(z2) - p(z))[0, 0, 0].item()                          # site 0 feels it through the mean only
    assert abs(d - 0.5 * 3.0 / 30) < 1e-3
    kept = p.law.prune(0.3)
    assert p.law._mask("G").sum() == 1 and kept == 3        # + the always-kept diagonal of A0 (2 entries)


def test_repeated_pruning_only_removes_terms_and_reports_read_masked_coefficients():
    """FPd: after pruning, Adam momentum and weight decay move masked raw parameters off zero; a later pruning
    must not bring those terms back, and term counts / reports must read the masked (effective) coefficients."""
    torch.manual_seed(2)
    p = _law_prop(c=3, law_residual_hidden=0)
    for t in p.law.coefficient_tensors().values():
        torch.nn.init.normal_(t, std=0.1)
    opt = torch.optim.AdamW(p.parameters(), lr=0.05, weight_decay=0.1)
    for _ in range(3):                         # build up momentum on every coefficient
        loss = ((p(torch.randn(2, 10, 3)) - 1) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    kept1 = p.law.prune(0.1)
    mask1 = p.law.masks.clone()
    for _ in range(3):                         # stale momentum moves the masked raw parameters
        loss = ((p(torch.randn(2, 10, 3)) - 1) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    raw_masked = torch.cat([v.detach().flatten() for v in p.law.coefficient_tensors().values()])[mask1 == 0]
    assert raw_masked.abs().max() > 0          # the drift this guards against is real
    assert p.law.n_terms() <= kept1
    kept2 = p.law.prune(0.0)                   # threshold 0 would keep everything without the mask
    assert kept2 <= kept1 and ((p.law.masks > 0) <= (mask1 > 0)).all()
    assert p.law.max_coupling() >= 0.1


def test_site_smoothness_is_scale_free_and_zero_for_linear_fields():
    from ks_latent.cm1.train import site_smoothness
    x = torch.linspace(-1, 1, 30)
    lin = torch.stack([x, 2 * x - 0.3], -1)[None].repeat(2, 1, 1)
    assert site_smoothness(lin).item() < 1e-10
    z = torch.randn(2, 30, 4)
    torch.testing.assert_close(site_smoothness(3.0 * z), site_smoothness(z))
    assert site_smoothness(z).item() > 1.0     # independent sites: E d2^2 = 6 var


def test_stage_c_fpd_ramps_residual_weight_and_prunes():
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    torch.manual_seed(0)
    ms = [_member(f"m{i}", seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms)
    tr, va = ColumnData(ms[:2], nrm, torch.device("cpu")), ColumnData(ms[2:], nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    prop = _law_prop(c=3, law_l1=1e-5, law_res_weight=0.1)
    for t in prop.law.coefficient_tensors().values():
        torch.nn.init.normal_(t, std=0.05)
    n0 = prop.law.n_terms()
    msgs = []
    rc = train_stage_c(ae, prop, tr, va, StageCConfig(window_sites=14, halo_sites=2, batch=2, steps=8, k_max=2,
                                                      val_every=2, warmup=1, law_l1=1e-3, law_res_weight_end=2.0,
                                                      law_res_ramp_frac=0.5, law_prune_every=2,
                                                      law_prune_start_frac=0.5, law_prune_rel=0.3,
                                                      w_site_smooth=0.1), log=msgs.append)
    assert prop.cfg.law_l1 == 1e-3 and abs(prop.cfg.law_res_weight - 2.0) < 1e-9
    assert any("pruned below" in m for m in msgs)
    assert prop.law.n_terms() < n0 and rc["history"][-1]["law_terms"] == prop.law.n_terms()
    assert np.isfinite(rc["best_val_decoded_rollout_mse"])
    conv = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1))
    with pytest.raises(ValueError):
        train_stage_c(ae, conv, tr, va, StageCConfig(window_sites=14, halo_sites=2, batch=2, steps=2, k_max=2,
                                                     law_prune_every=1), log=lambda s: None)


def test_tendency_cap_bounds_rollouts_and_is_transparent_for_small_tendencies():
    """FPc fix: an explicit quadratic law blew up in a 45-step rollout (Saidie 8999). With a cap, every step is
    bounded by cap per stage; for small tendencies the capped law equals the uncapped one."""
    torch.manual_seed(0)
    free, capped = _law_prop(c=3, law_residual_hidden=0), _law_prop(c=3, law_residual_hidden=0, law_cap=1.0)
    with torch.no_grad():
        for p in (free, capped):
            p.law.Q1.fill_(0.5)                  # dz/dt ~ z^2: finite-time blow-up without a bound
    z = torch.full((1, 12, 3), 2.0)
    zf, zc = z.clone(), z.clone()
    with torch.no_grad():
        for _ in range(45):
            zf, zc = free(zf), capped(zc)
    assert not torch.isfinite(zf).all()
    assert torch.isfinite(zc).all() and (zc - z).abs().max() <= 45 * 1.0 + 1e-4
    small = 1e-3 * torch.randn(2, 12, 3)
    torch.testing.assert_close(capped(small), free(small), atol=1e-8, rtol=1e-6)
    with pytest.raises(ValueError):
        _law_prop(law_cap=-1.0)


def test_stage_c_freeze_ae_trains_only_the_propagator():
    """M3: with freeze_ae the autoencoder (hence the latent) is bit-identical after Stage C, and a stochastic
    propagator still learns (its parameters move)."""
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    torch.manual_seed(0)
    ms = [_member(f"m{i}", seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms)
    tr, va = ColumnData(ms[:2], nrm, torch.device("cpu")), ColumnData(ms[2:], nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    sp = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, noise_channels=2))
    ae0 = {k: v.clone() for k, v in ae.state_dict().items()}
    p0 = [p.detach().clone() for p in sp.parameters()]
    train_stage_c(ae, sp, tr, va, StageCConfig(window_sites=14, halo_sites=2, batch=2, steps=4, k_max=2, val_every=2,
                                               warmup=1, crps_samples=2, freeze_ae=True), log=lambda s: None)
    assert all(torch.equal(ae0[k], v) for k, v in ae.state_dict().items())
    assert any(not torch.equal(a, b.detach()) for a, b in zip(p0, sp.parameters()))
