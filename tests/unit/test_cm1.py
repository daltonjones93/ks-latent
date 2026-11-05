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


@pytest.mark.integration
def test_data_on_another_device_than_model_gives_same_reports():
    """Data in host memory, model elsewhere: every consumer must move what it needs (here both are CPU,
    but `full(device=)` and the reports must accept a device argument and give identical numbers)."""
    from ks_latent.cm1.train import encode_members
    ms = [_member(f"m{i}", T=8, seed=i) for i in range(2)]
    nrm = Normalizer.fit(ms)
    d = ColumnData(ms, nrm, torch.device("cpu"))
    ae = _ae().float()
    z = encode_members(ae, d)
    assert z[0].shape == (8, NX // PATCH, 3)
    torch.testing.assert_close(d.full(0, 3, device=torch.device("cpu")), d.full(0, 3))
    prop = _prop().float()
    a = forecast_report(ae, prop, d, z, nrm, t0s=[1], n_lead=2, halo=2)
    assert np.isfinite(a["nmse"]).all()


def test_cached_members_round_trip_in_natural_order(tmp_path):
    from ks_latent.cm1.data import load_cached_members, save_member
    names = ["mem10", "mem2", "mem1"]
    for i, n in enumerate(names):
        m = _member(n, seed=i)
        save_member(m, tmp_path / f"src__{n}.npz")
    got = load_cached_members(tmp_path)
    assert [m.name for m in got] == ["mem1", "mem2", "mem10"]
    ref = _member("mem2", seed=1)
    np.testing.assert_array_equal(got[1].X, ref.X)
    np.testing.assert_array_equal(got[1].cref, ref.cref)
    assert got[1].dt == 60.0
    save_member(_member("mem1"), tmp_path / "other__mem1.npz")
    with pytest.raises(ValueError):
        load_cached_members(tmp_path)  # two sources mixed: refuse
    assert len(load_cached_members(tmp_path, "src__*.npz")) == 3
    with pytest.raises(FileNotFoundError):
        load_cached_members(tmp_path / "empty")


def test_synthetic_members_layout():
    from ks_latent.cm1.data import synthetic_members
    ms = synthetic_members(3, T=10, NX=128, Z=4)
    assert len(ms) == 3 and ms[0].X.shape == (10, 128, V, 4) and ms[0].dt == 60.0
    pos = [VARS.index(v) for v in ("qliq", "qice")]
    assert (ms[0].X[:, :, pos] >= 0).all() and (ms[0].cref > 10).any()


def test_rollout_length_sampling():
    from ks_latent.cm1.train import StageBConfig, _k_now
    cfg = StageBConfig(k_max=20, k_ramp_frac=0.5)
    assert _k_now(0.0, cfg) == 2 and _k_now(0.25, cfg) == 11 and _k_now(0.9, cfg) == 20
    u = StageBConfig(k_max=20, k_ramp_frac=0.5, k_sampling="uniform")
    g = torch.Generator().manual_seed(0)
    ks = [_k_now(0.9, u, g) for _ in range(2000)]
    assert min(ks) == 2 and max(ks) == 20 and 9 < np.mean(ks) < 13   # uniform on 2..20, mean 11
    assert all(_k_now(0.0, u, g) == 2 for _ in range(20))            # curriculum still caps it
    with pytest.raises(ValueError):
        _k_now(0.5, u)


def test_grouped_encoder_keeps_variables_separate():
    """groups = V: each variable's latent channels depend only on that variable's input levels, and its
    decoded levels only on its own latent channels (exact, by gradient)."""
    cfg = ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=V * 3, latent_channels=V * 2,
                         n_mix=1, mix_radius=1, groups=V)
    ae = ColumnFieldAE(cfg, positive_channel_mask(Z)).double().eval()
    cpv = 2  # latent channels per variable
    for v in (0, 3, 6):
        u = torch.randn(1, V * Z, NX, dtype=torch.float64, requires_grad=True)
        ae.encode(u)[0, :, v * cpv:(v + 1) * cpv].sum().backward()
        used = (u.grad.abs().sum(dim=(0, 2)) > 0).reshape(V, Z).any(dim=1)
        assert used.nonzero().ravel().tolist() == [v]
        z = torch.randn(1, NX // PATCH, V * cpv, dtype=torch.float64, requires_grad=True)
        ae.decode(z)[0, v * Z:(v + 1) * Z].sum().backward()
        lat = (z.grad.abs().sum(dim=(0, 1)) > 0).reshape(V, cpv).any(dim=1)
        assert lat.nonzero().ravel().tolist() == [v]
    with pytest.raises(ValueError):
        ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, hidden=16, latent_channels=V, groups=V))


def test_history_propagator_forced_halo_equals_full_domain():
    cfg = StencilPropConfig(latent_channels=3, hidden=8, layers=2, radius=1, n_hist=2)
    p = StencilPropagator(cfg).double().eval()
    torch.nn.init.normal_(p.head.weight, std=0.3)
    P, k = 40, 4
    traj = [torch.randn(1, P, 3, dtype=torch.float64) for _ in range(2)]
    for _ in range(k):
        traj.append(p(torch.stack(traj[-2:], dim=1)))
    traj = torch.stack(traj, dim=1)                       # (1, 2 + k, P, c): 2 history frames + k steps
    win = traj[:, :, 10:26]
    pred = forced_halo_rollout(p, win, k, halo=2)
    torch.testing.assert_close(pred, win[:, 2:], atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        p(torch.randn(1, P, 3, dtype=torch.float64))      # a single frame is not a 2-frame history
    with pytest.raises(ValueError):
        forced_halo_rollout(p, win[:, :3], k, halo=2)     # too few frames for history + k


def test_stochastic_propagator_samples_and_identity_at_init():
    cfg = StencilPropConfig(latent_channels=3, hidden=8, layers=2, radius=1, noise_channels=2)
    p0 = StencilPropagator(cfg).double().eval()
    z = torch.randn(2, 20, 3, dtype=torch.float64)
    torch.testing.assert_close(p0(z), z, atol=0, rtol=0)              # zero head: identity whatever the noise
    p = StencilPropagator(cfg).double().eval()
    torch.nn.init.normal_(p.head.weight, std=0.3)
    a = p(z, gen=torch.Generator().manual_seed(0))
    b = p(z, gen=torch.Generator().manual_seed(0))
    c = p(z, gen=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert (a - c).abs().max() > 1e-6
    legacy = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=2, radius=1))
    assert legacy.convs[0].in_channels == 3                           # defaults keep the original shapes


def test_crps_loss_known_values():
    from ks_latent.cm1.train import crps_loss
    y = torch.zeros(5)
    same = torch.zeros(4, 5)
    assert crps_loss(same, y).item() == 0.0                              # perfect, zero-spread ensemble
    two = torch.tensor([[1.0], [-1.0]])                                  # samples +-1 around y = 0
    # E|X-y| = 1 and E|X-X'| = 2 over the 2 off-diagonal pairs, so the fair CRPS is 1 - 0.5*2 = 0
    assert abs(crps_loss(two, torch.zeros(1)).item() - 0.0) < 1e-7
    off = torch.tensor([[1.0], [1.0]])                                   # both samples off by 1: CRPS = 1
    assert abs(crps_loss(off, torch.zeros(1)).item() - 1.0) < 1e-7
    with pytest.raises(ValueError):
        crps_loss(torch.zeros(1, 3), torch.zeros(3))


@pytest.mark.integration
def test_stage_b_trains_history_and_stochastic_propagators():
    from ks_latent.cm1.train import StageBConfig, train_stage_b
    torch.manual_seed(0)
    z = [torch.randn(12, 16, 3) for _ in range(2)]
    act = [torch.zeros((0, 2), dtype=torch.long)] * 2
    for kw, bk in (({"n_hist": 2}, {}), ({"noise_channels": 2}, {"crps_samples": 3})):
        prop = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, radius=1, **kw))
        r = train_stage_b(prop, z, z, act, act, PATCH, StageBConfig(window_sites=12, halo_sites=1, batch=4, steps=4,
                                                                    k_max=3, val_every=2, warmup=1, **bk), log=lambda s: None)
        assert np.isfinite(r["best_val_mse"])


def test_vertical_front_end_keeps_x_locality_and_equivariance():
    cfg = ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3, vertical_channels=4,
                         vertical_kernel=3)
    ae = ColumnFieldAE(cfg, positive_channel_mask(Z)).double().eval()
    u = torch.randn(1, V * Z, NX, dtype=torch.float64)
    z, zs = ae.encode(u), ae.encode(torch.roll(u, 2 * PATCH, dims=-1))
    P, r = NX // PATCH, cfg.reach_sites
    torch.testing.assert_close(zs[:, r + 2:P - r], z[:, r:P - r - 2], atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, hidden=V * 2, latent_channels=V, groups=V,
                                     vertical_channels=4))


@pytest.mark.integration
def test_stage_c_runs_and_updates_both_models():
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    torch.manual_seed(0)
    ms = [_member(f"m{i}", T=10, seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms)
    tr, va = ColumnData(ms[:2], nrm, torch.device("cpu")), ColumnData(ms[2:], nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    prop = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, radius=1))
    a0 = {k: v.clone() for k, v in ae.state_dict().items()}
    p0 = {k: v.clone() for k, v in prop.state_dict().items()}
    r = train_stage_c(ae, prop, tr, va, StageCConfig(window_sites=14, halo_sites=1, batch=2, steps=4, k_max=3,
                                                     val_every=2, warmup=1, w_sfc_th=2.0, sfc_levels=2),
                      log=lambda s: None)
    assert np.isfinite(r["best_val_decoded_rollout_mse"])
    assert any((ae.state_dict()[k] != a0[k]).any() for k in a0)
    assert any((prop.state_dict()[k] != p0[k]).any() for k in p0)


def test_environment_features_and_conditioned_encoder(tmp_path):
    from ks_latent.cm1.env import load_env_files, sounding_env
    from ks_latent.cm1.physics import BASE_SOUNDING
    f0 = sounding_env(BASE_SOUNDING)
    assert f0.shape == (3 * 62,) and np.abs(f0).max() == 0                    # the reference itself: all zero
    lines = BASE_SOUNDING.read_text().splitlines()
    mod = [lines[0]] + [" ".join([l.split()[0], l.split()[1], str(float(l.split()[2]) + 1.0)] + l.split()[3:])
                        for l in lines[1:] if l.strip()]
    (tmp_path / "s").write_text("\n".join(mod))
    f1 = sounding_env(tmp_path / "s")
    assert np.allclose(f1[62:124], 1.0) and np.allclose(f1[:62], 0) and np.allclose(f1[124:], 0)   # qv + 1 g/kg
    cfg = ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3, env_raw_dim=6)
    ae = ColumnFieldAE(cfg, positive_channel_mask(Z)).double().eval()
    u = torch.randn(1, V * Z, NX, dtype=torch.float64)
    e0, e1 = torch.zeros(1, 6, dtype=torch.float64), torch.ones(1, 6, dtype=torch.float64)
    assert (ae.encode(u, e0) - ae.encode(u, e1)).abs().max() > 1e-6
    with pytest.raises(ValueError):
        ae.encode(u)                                                           # conditioned: env required
    zs, z = ae.encode(torch.roll(u, 2 * PATCH, dims=-1), e1), ae.encode(u, e1)
    r, P = cfg.reach_sites, NX // PATCH
    torch.testing.assert_close(zs[:, r + 2:P - r], z[:, r:P - r - 2], atol=1e-12, rtol=0)   # still equivariant in x


def test_split_species_pipeline_and_physics():
    """D15: rain/snow/graupel as separate variables. The normalizer, positivity mask, reports and
    condensate physics follow the data's variable set; total condensate equals the 7-variable one."""
    from ks_latent.cm1.data import VARS_SPLIT, synthetic_members, vars_for
    from ks_latent.cm1.physics import BaseState, condensate_mass, low_level_liquid
    from ks_latent.cm1.train import encode_members
    ms = synthetic_members(3, T=6, NX=96, Z=Z, split=True)
    nrm = Normalizer.fit(ms[:2])
    assert nrm.vars == VARS_SPLIT and vars_for(10) == VARS_SPLIT and vars_for(7) == VARS
    with pytest.raises(ValueError):
        vars_for(8)
    mask = positive_channel_mask(Z, nrm.vars).reshape(10, Z)
    assert mask[5:].all() and not mask[:5].any()
    dev = torch.device("cpu")
    va = ColumnData(ms[2:], nrm, dev)
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=10 * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z, nrm.vars))
    rr = reconstruction_report(ae, va, nrm)
    assert list(rr["rmse_phys_all"]) == list(VARS_SPLIT)
    prop = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=2, radius=1))
    fc = forecast_report(ae, prop, va, encode_members(ae, va), nrm, t0s=[0], n_lead=2, halo=2)
    assert np.isfinite(fc["nmse"]).all()
    # physics: species sums reproduce the combined liquid/ice condensate exactly
    rng = np.random.default_rng(0)
    base = BaseState.from_sounding()
    Zr = base.z.size
    S = np.abs(rng.normal(size=(10, Zr, 40)))                       # sqrt(g/kg) per species
    C = S[:7].copy()
    C[5] = np.sqrt(S[5] ** 2 + S[6] ** 2)                           # qliq = sqrt(qc + qr)
    C[6] = np.sqrt(S[7] ** 2 + S[8] ** 2 + S[9] ** 2)               # qice = sqrt(qi + qs + qg)
    np.testing.assert_allclose(condensate_mass(S, base), condensate_mass(C, base), rtol=1e-12)
    np.testing.assert_allclose(low_level_liquid(S, base), low_level_liquid(C, base), rtol=1e-12)


def test_synthetic_member_names_are_unique():
    from ks_latent.cm1.data import synthetic_members
    assert [m.name for m in synthetic_members(4, T=4, NX=32, Z=2)] == ["syn000", "syn001", "syn002", "syn003"]


def test_mirror_member_and_env_are_involutions():
    """D9: mirroring reverses x, negates only u', and twice is the identity (data, name, env)."""
    from ks_latent.cm1.data import mirror_member, synthetic_members
    from ks_latent.cm1.env import UMOVE, mirror_env
    from ks_latent.cm1.physics import BaseState
    m = synthetic_members(1, T=5, NX=64, Z=4)[0]
    mm = mirror_member(m)
    assert mm.name == m.name + "_mirror"
    np.testing.assert_array_equal(mm.X[:, :, 0], -m.X[:, ::-1, 0])
    np.testing.assert_array_equal(mm.X[:, :, 1:], m.X[:, ::-1, 1:])
    np.testing.assert_array_equal(mm.cref, m.cref[:, ::-1])
    back = mirror_member(mm)
    assert back.name == m.name
    np.testing.assert_array_equal(back.X, m.X)
    Zr = BaseState.from_sounding().z.size
    f = np.random.default_rng(0).normal(size=3 * Zr).astype(np.float32)
    np.testing.assert_allclose(mirror_env(mirror_env(f)), f, atol=1e-5)
    np.testing.assert_array_equal(mirror_env(f)[:2 * Zr], f[:2 * Zr])
    # a sounding moving with the grid (u = umove everywhere) is its own mirror image
    from ks_latent.cm1.env import SCALES, _profiles
    from ks_latent.cm1.physics import BASE_SOUNDING
    u_ref = _profiles(BASE_SOUNDING)[:, 3]
    g = np.zeros(3 * Zr, np.float32); g[2 * Zr:] = (UMOVE - u_ref) / SCALES[2]
    np.testing.assert_allclose(mirror_env(g), g, atol=1e-5)


def test_crps_elements_matches_pairwise_crps():
    from ks_latent.cm1.train import crps_elements, crps_loss
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(5, 3, 7, generator=g, dtype=torch.float64), torch.randn(3, 7, generator=g, dtype=torch.float64)
    torch.testing.assert_close(crps_elements(x, y).mean(), crps_loss(x, y), rtol=1e-12, atol=1e-12)
    # identical samples: CRPS is the MAE
    torch.testing.assert_close(crps_elements(y[None].repeat(4, 1, 1), y + 1.0), torch.ones_like(y))


def test_stage_c_crps_trains_a_stochastic_propagator_and_reports_probabilistic_scores():
    from ks_latent.cm1.train import StageCConfig, encode_members, train_stage_c
    torch.manual_seed(0)
    ms = [_member(f"m{i}", T=10, seed=i) for i in range(3)]
    nrm = Normalizer.fit(ms)
    tr, va = ColumnData(ms[:2], nrm, torch.device("cpu")), ColumnData(ms[2:], nrm, torch.device("cpu"))
    ae = ColumnFieldAE(ColumnAEConfig(n_channels=V * Z, n_levels=Z, patch=PATCH, hidden=16, latent_channels=3),
                       positive_channel_mask(Z))
    prop = StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, radius=1, noise_channels=2))
    with pytest.raises(ValueError):
        train_stage_c(ae, StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, radius=1)),
                      tr, va, StageCConfig(window_sites=14, halo_sites=1, steps=1, crps_samples=3), log=lambda s: None)
    p0 = {k: v.clone() for k, v in prop.state_dict().items()}
    r = train_stage_c(ae, prop, tr, va, StageCConfig(window_sites=14, halo_sites=1, batch=2, steps=4, k_max=3,
                                                     val_every=2, warmup=1, crps_samples=3), log=lambda s: None)
    assert np.isfinite(r["best_val_decoded_rollout_mse"])
    assert any((prop.state_dict()[k] != p0[k]).any() for k in p0)
    z = encode_members(ae, va)
    fc = forecast_report(ae, prop, va, z, nrm, t0s=[0], n_lead=3, halo=1, n_samples=4)
    assert len(fc["crps"]) == 3 and len(fc["spread_err"]) == 3 and all(np.isfinite(fc["crps"]))
    det = forecast_report(ae, StencilPropagator(StencilPropConfig(latent_channels=3, hidden=8, layers=1, radius=1)),
                          va, z, nrm, t0s=[0], n_lead=3, halo=1)
    assert "spread_err" not in det and len(det["crps"]) == 3        # deterministic: CRPS = MAE, no spread
