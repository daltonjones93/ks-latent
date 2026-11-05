"""CM1 column local field (ks_latent/cm1): normalization, window sampling, exact locality and
equivariance of the autoencoder and propagator, the forced-halo rollout, and a smoke run of both
training stages."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.data import VARS, CM1Member, ColumnData, Normalizer, positive_channel_mask, split_members
from ks_latent.cm1.evaluate import forecast_report, front_x_km, reconstruction_report
from ks_latent.cm1.models import (ColumnAEConfig, ColumnFieldAE, StencilPropagator, StencilPropConfig,
                                  forced_halo_rollout)

pytestmark = pytest.mark.unit
V, Z, NX, PATCH = len(VARS), 4, 96, 4


def _member(name, T=6, seed=0, NXm=NX):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, NXm, V, Z)).astype(np.float32)
    X[:, :, [VARS.index(v) for v in ("qliq", "qice")]] = np.abs(X[:, :, [VARS.index(v) for v in ("qliq", "qice")]])
    cref = np.zeros((T, NXm), np.float32)
    cref[:, NXm // 2 - 3:NXm // 2 + 3] = 30.0  # a storm in the middle
    return CM1Member(name=name, X=X.astype(np.float16), cref=cref, t=np.arange(T) * 60.0)


def _ae(c=3, n_mix=2, r=1):
    cfg = ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=c, n_mix=n_mix,
                         mix_radius=r)
    return ColumnFieldAE(cfg, positive_channel_mask(Z)).double().eval()


def _prop(c=3, layers=2, r=1, zero=True):
    p = StencilPropagator(StencilPropConfig(latent_channels=c, hidden=8, layers=layers, radius=r)).double().eval()
    if not zero:
        torch.nn.init.normal_(p.head.weight, std=0.3)
    return p


def test_normalizer_is_rms_per_variable_and_keeps_zero():
    m = _member("a")
    nrm = Normalizer.fit([m], stride=1)
    x = m.X.astype(np.float64)
    np.testing.assert_allclose(nrm.scale, np.sqrt((x**2).mean(axis=(0, 1, 3))), rtol=1e-9)
    assert nrm.channel_scale(Z).shape == (V * Z,)
    d = ColumnData([m], nrm, torch.device("cpu"))
    zero = np.zeros_like(m.X); zero_m = CM1Member("z", zero, m.cref, m.t)
    assert torch.count_nonzero(ColumnData([zero_m], nrm, torch.device("cpu")).u[0]) == 0
    assert d.C == V * Z


def test_windows_never_wrap_and_storm_windows_contain_storm():
    m = _member("a")
    d = ColumnData([m], Normalizer.fit([m]), torch.device("cpu"))
    g = torch.Generator().manual_seed(0)
    w, mem, t0, x0 = d.windows(64, 32, 3, g, active_frac=0.0)
    assert w.shape == (64, 3, V * Z, 32)
    assert int(x0.min()) >= 0 and int((x0 + 32).max()) <= NX and int((t0 + 3).max()) <= 6
    full = d.full(0)  # (T, C, NX)
    for b in range(5):
        torch.testing.assert_close(w[b], full[t0[b]:t0[b] + 3, :, x0[b]:x0[b] + 32])
    _, _, _, xa = d.windows(64, 32, 1, g, active_frac=1.0)
    assert ((xa <= NX // 2) & (xa + 32 > NX // 2)).all()
    with pytest.raises(ValueError):
        d.windows(1, NX + 1, 1, g)


def test_split_by_member():
    tr, va, te = split_members([f"m{i}" for i in range(10)], 1, 2)
    assert (len(tr), len(va), len(te)) == (7, 1, 2) and not set(tr) & set(te)
    with pytest.raises(ValueError):
        split_members(["a", "b"], 1, 1)


def test_ae_shapes_and_nonnegative_condensate():
    ae = _ae()
    u = torch.randn(2, V * Z, NX, dtype=torch.float64)
    uh, z = ae(u)
    assert z.shape == (2, NX // PATCH, 3) and uh.shape == u.shape
    pm = positive_channel_mask(Z)
    assert pm.sum() == 2 * Z and (uh[:, pm] >= 0).all()
    with pytest.raises(ValueError):
        ae.encode(torch.randn(1, V * Z, NX + 1, dtype=torch.float64))
    with pytest.raises(ValueError):
        ColumnFieldAE(ColumnAEConfig(padding_mode="circular"))


def test_ae_shift_equivariance_exact_in_interior():
    ae = _ae()
    u = torch.randn(1, V * Z, NX, dtype=torch.float64)
    s = 2  # sites
    z, zs = ae.encode(u), ae.encode(torch.roll(u, s * PATCH, dims=-1))
    P, reach = NX // PATCH, ae.cfg.reach_sites
    j = slice(reach + s, P - reach)  # sites whose neighbourhood is real data in both
    torch.testing.assert_close(zs[:, j], z[:, slice(reach, P - reach - s)], atol=1e-12, rtol=0)


@pytest.mark.parametrize("n_mix,r", [(1, 1), (2, 1), (2, 2)])
def test_encoder_and_decoder_receptive_fields_match_reach(n_mix, r):
    ae = _ae(n_mix=n_mix, r=r)
    P, reach, j = NX // PATCH, n_mix * r, NX // PATCH // 2
    u = torch.randn(1, V * Z, NX, dtype=torch.float64, requires_grad=True)
    ae.encode(u)[0, j].sum().backward()
    cells = u.grad.abs().sum(dim=(0, 1)) > 0
    sites = cells.reshape(P, PATCH).any(dim=1).nonzero().ravel()
    assert sites.min() == j - reach and sites.max() == j + reach
    z = torch.randn(1, P, 3, dtype=torch.float64, requires_grad=True)
    ae.decode(z)[0, :, j * PATCH].sum().backward()
    dsites = (z.grad.abs().sum(dim=(0, 2)) > 0).nonzero().ravel()
    assert dsites.min() == j - reach and dsites.max() == j + reach


def test_propagator_identity_at_init_and_local():
    z = torch.randn(2, 20, 3, dtype=torch.float64)
    torch.testing.assert_close(_prop()(z), z, atol=0, rtol=0)
    p = _prop(layers=2, r=1, zero=False)
    z2 = z.clone(); z2[:, 10] += 1.0
    changed = ((p(z2) - p(z)).abs().sum(dim=(0, 2)) > 0).nonzero().ravel()
    assert changed.min() == 10 - 2 and changed.max() == 10 + 2


def test_forced_halo_window_rollout_equals_full_domain_rollout():
    p = _prop(layers=2, r=1, zero=False)
    reach, P, k = 2, 40, 4
    traj = [torch.randn(1, P, 3, dtype=torch.float64)]
    for _ in range(k):
        traj.append(p(traj[-1]))
    traj = torch.stack(traj, dim=1)  # (1, k+1, P, c): the "truth"
    x0, W = 10, 16
    win = traj[:, :, x0:x0 + W]
    pred = forced_halo_rollout(p, win, k, halo=reach)
    torch.testing.assert_close(pred, win[:, 1:], atol=1e-12, rtol=0)
    bad = forced_halo_rollout(p, win, k, halo=reach - 1)  # halo too thin: the window edge leaks in
    assert (bad - win[:, 1:]).abs().max() > 1e-6


def test_front_position():
    th = torch.zeros(2, 100)
    th[0, :37] = -3.0
    f = front_x_km(th)
    assert abs(f[0].item() - 36.5 * 0.25) < 1e-6 and torch.isnan(f[1])


@pytest.mark.integration
def test_smoke_both_stages_and_reports():
    from ks_latent.cm1.train import StageAConfig, StageBConfig, encode_members, train_stage_a, train_stage_b
    torch.manual_seed(0)
    ms = [_member(f"m{i}", T=8, seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms[:2])
    dev = torch.device("cpu")
    tr, va = ColumnData(ms[:2], nrm, dev), ColumnData(ms[2:], nrm, dev)
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    ra = train_stage_a(ae, tr, va, StageAConfig(window_sites=12, batch=4, steps=6, val_every=3, warmup=1), log=lambda s: None)
    assert np.isfinite(ra["best_val_recon"])
    z_tr, z_va = encode_members(ae, tr), encode_members(ae, va)
    prop = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=2, radius=1))
    pa = lambda d: [a // 1 for a in d.active]
    rb = train_stage_b(prop, z_tr, z_va, pa(tr), pa(va), PATCH,
                       StageBConfig(window_sites=12, halo_sites=2, batch=4, steps=6, k_max=3, val_every=3, warmup=1),
                       log=lambda s: None)
    assert np.isfinite(rb["best_val_mse"])
    rr = reconstruction_report(ae, va, nrm)
    assert set(rr["rmse_phys_all"]) == set(VARS)
    fc = forecast_report(ae, prop, va, z_va, nrm, t0s=[0, 2], n_lead=3, halo=2)
    assert len(fc["nmse"]) == 3 and len(fc["nmse_pers"]) == 3
