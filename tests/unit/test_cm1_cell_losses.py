"""9a: threshold-weighted, pooled and low-pass terms for convective cells (ks_latent/cm1/cell_losses.py) and their use in Stage C."""
import numpy as np
import pytest
import torch

from ks_latent.cm1 import cell_losses as cl
from ks_latent.cm1.data import VARS

Z, W = 4, 64
C = len(VARS) * Z
SCALE = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])        # u, w, th, qv, prs, qliq, qice


def _x(): return torch.zeros(1, 3, C, W)


def test_thresholds_in_physical_units():
    tw, tq = cl.tau_norm(SCALE, 2.0, 0.09)
    assert abs(tw - 2.0) < 1e-12 and abs(tq - 0.3 / 0.5) < 1e-12      # sqrt(0.09) / scale


def test_tw_mae_ignores_the_quiet_region_and_scores_cells():
    tw, tq = cl.tau_norm(SCALE, 2.0, 0.1)
    d, t = _x(), _x()
    d[..., cl.var_slice("w", Z), :] = 1.0; t[..., cl.var_slice("w", Z), :] = -1.5          # both below the 2 m/s threshold: almost no weight
    quiet = float(cl.tw_mae(d, t, Z, tw, tq))
    t2 = t.clone(); t2[..., cl.var_slice("w", Z), 10] = 6.0                                  # a strong updraft in the truth only
    cell = float(cl.tw_mae(d, t2, Z, tw, tq))
    assert cell > 0.0 and quiet < 0.05 * float(cl.tw_mae(d, t2 * 0 + 6.0, Z, tw, tq))      # sub-threshold differences weigh far less than a full-field cell
    d2 = d.clone(); d2[..., cl.var_slice("w", Z), 10] = -6.0                                 # a strong downdraft in the forecast only
    assert float(cl.tw_mae(d2, t, Z, tw, tq)) > 0.0
    d.requires_grad_(True)
    cl.tw_mae(d, t2, Z, tw, tq).backward()
    assert float(d.grad[..., cl.var_slice('w', Z), 10].sum()) < 0     # below the threshold, a missed updraft still pulls the forecast up (negative gradient of the loss on w there)


def test_pooled_term_tolerates_displacement_more_than_the_pointwise_one():
    tw, tq = cl.tau_norm(SCALE, 2.0, 0.1)
    t, d_near, d_far = _x(), _x(), _x()
    t[..., cl.var_slice("w", Z), 30] = 8.0
    d_near[..., cl.var_slice("w", Z), 33] = 8.0                                              # 0.75 km away
    d_far[..., cl.var_slice("w", Z), 55] = 8.0                                               # 6 km away
    pt = lambda a: float(cl.tw_mae(a, t, Z, tw, tq))
    pl = lambda a: float(cl.pooled_exceed_mae(a, t, Z, tw, tq, (4.0, 8.0)))
    assert abs(pt(d_near) - pt(d_far)) < 1e-6                    # pointwise: a miss is a miss whatever the distance
    assert pl(d_near) < pl(d_far)                                # pooled: near is better than far
    assert pl(t) == 0.0
    assert (pl(d_far) - pl(d_near)) / pl(d_far) > 0.05


def test_lowpass_ignores_small_scales_and_keeps_large_ones():
    x = torch.randn(2, 5, C, W)
    hi = torch.zeros(2, 5, C, W); hi[..., ::2] = 0.3; hi[..., 1::2] = -0.3                    # alternating columns
    assert float(cl.lowpass_mse(hi, torch.zeros_like(hi), 2.0)) < 1e-3 * float((hi ** 2).mean())
    lo = torch.ones(2, 5, C, W)
    assert abs(float(cl.lowpass_mse(lo, torch.zeros_like(lo), 2.0)) - 1.0) < 1e-4            # a uniform offset passes
    assert float(cl.lowpass_mse(x, x, 2.0)) == 0.0


def test_stage_c_with_the_terms_runs_and_logs_them():
    from ks_latent.cm1.data import ColumnData, Normalizer, synthetic_members
    from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    ms = synthetic_members(3, T=60, NX=96 * 4, Z=Z)
    norm = Normalizer.fit(ms); d = ColumnData(ms, norm, torch.device("cpu"))
    assert d.chan_scale.shape == (d.C,)
    torch.manual_seed(0)
    ae = build_autoencoder(ColumnAEConfig(n_channels=d.C, n_levels=Z, patch=4, hidden=16, latent_channels=6, n_mix=1), None)
    prop = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, layers=2, radius=1))
    logs = []
    cfg = StageCConfig(window_sites=40, halo_sites=2, batch=2, steps=4, k_max=4, val_every=2, val_batches=1, warmup=1,
                       tw_weight=0.3, pool_weight=0.3, lp_weight=0.3, lp_sigma_km=1.0, select_last=True)
    out = train_stage_c(ae, prop, d, d, cfg, log=logs.append)
    assert all(np.isfinite(h["l_roll"]) for h in out["history"])
    assert {"term_tw", "term_pool", "term_lp"} <= set(out["history"][-1])


def test_lowpass_chunks_give_the_same_result():
    x = torch.randn(3, 11, C, W)
    a, b = cl.lowpass_x(x, 2.0, chunk=7), cl.lowpass_x(x, 2.0, chunk=10 ** 6)
    assert torch.allclose(a, b, atol=1e-6)


def test_pooled_crps_scores_the_ensemble_not_just_the_mean():
    """9c: an ensemble that brackets the truth's cell positions scores better than a sharp ensemble at a wrong position, and the term is zero-free
    of the quiet region; gradients reach the members."""
    torch.manual_seed(0)
    tw, tq = 2.0, 0.6
    t = torch.zeros(2, 3, C, W); t[..., cl.var_slice("w", Z), 20:24] = 6.0
    S = 4
    shift = lambda k: torch.roll(t, k, dims=-1)
    sharp_wrong = torch.stack([shift(8)] * S)                              # all members put the cell 8 columns away
    spread_ok = torch.stack([shift(k) for k in (-3, -1, 1, 3)])           # members scattered around the truth
    a = float(cl.pooled_exceed_crps(sharp_wrong, t, Z, tw, tq, (4.0,)))
    b = float(cl.pooled_exceed_crps(spread_ok, t, Z, tw, tq, (4.0,)))
    assert b < a
    assert float(cl.pooled_exceed_crps(torch.stack([t] * S), t, Z, tw, tq)) < 1e-6
    d = spread_ok.clone().requires_grad_(True)
    cl.pooled_exceed_crps(d, t, Z, tw, tq).backward()
    assert d.grad.abs().sum() > 0


def test_stage_c_stochastic_with_pooled_crps():
    from ks_latent.cm1.data import ColumnData, Normalizer, synthetic_members
    from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    ms = synthetic_members(3, T=60, NX=96 * 4, Z=Z)
    norm = Normalizer.fit(ms); d = ColumnData(ms, norm, torch.device("cpu"))
    torch.manual_seed(0)
    ae = build_autoencoder(ColumnAEConfig(n_channels=d.C, n_levels=Z, patch=4, hidden=16, latent_channels=6, n_mix=1), None)
    prop = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, layers=2, radius=1, noise_channels=2))
    cfg = StageCConfig(window_sites=40, halo_sites=2, batch=2, steps=4, k_max=4, val_every=2, val_batches=1, warmup=1,
                       crps_samples=3, pool_crps_weight=1.0)
    out = train_stage_c(ae, prop, d, d, cfg, log=lambda *_: None)
    assert all(np.isfinite(h["l_roll"]) for h in out["history"])
    assert "term_pool_crps" in out["history"][-1] and out["history"][-1]["term_pool_crps"] > 0
    with pytest.raises(ValueError):
        train_stage_c(ae, prop, d, d, StageCConfig(window_sites=40, halo_sites=2, batch=2, steps=2, k_max=4, pool_crps_weight=1.0), log=lambda *_: None)


def _w_field(vals):
    """A (C, W) column field with w = vals at every level, other variables 0."""
    x = torch.zeros(C, W); x[cl.var_slice("w", Z)] = torch.as_tensor(vals, dtype=torch.float32)
    return x


def test_feature_masks_separate_updrafts_and_downdrafts():
    w = torch.zeros(W); w[5:9] = 3.0; w[20:24] = -2.0; w[30] = 1.9; w[31] = -1.4
    yu, yd = cl.feature_masks(_w_field(w), Z, 2.0, 1.5)
    assert yu[0].nonzero().flatten().tolist() == [5, 6, 7, 8]
    assert yd[0].nonzero().flatten().tolist() == [20, 21, 22, 23]


def test_feature_log_score_deterministic_rewards_sharp_correct_cells():
    """9d: the cross-entropy prefers a sharp cell in the right place to the blurred (spread-out, sub-threshold) version an MSE favours under
    position uncertainty, and has the right gradient sign at a missed cell and in the quiet region."""
    truth = torch.zeros(W); truth[20:24] = 4.0
    t = _w_field(truth)
    sharp = _w_field(truth)
    blur = _w_field(torch.nn.functional.conv1d(truth.view(1, 1, -1), torch.full((1, 1, 9), 1 / 9), padding=4).flatten())
    bs = sum(cl.feature_log_score(sharp[None], t, Z, 2.0, 1.5, 0.5))
    bb = sum(cl.feature_log_score(blur[None], t, Z, 2.0, 1.5, 0.5))
    assert bs < bb
    d = torch.zeros(1, C, W, requires_grad=True)                     # a forecast that misses the cell
    sum(cl.feature_log_score(d, t, Z, 2.0, 1.5, 0.5)).backward()
    g = d.grad[0, cl.var_slice("w", Z)]
    assert (g[:, 20:24] < 0).all()                                   # descent raises w at the missed cell
    assert g[:, 40:].abs().max() < g[:, 20:24].abs().min()           # the quiet region at w = 0 is already near-certain


def test_feature_log_score_ensemble_is_proper_not_per_member():
    """An ensemble whose member fraction matches the event frequency scores better than a collapsed ensemble at the wrong answer; members that
    disagree get the ensemble probability, not individual penalties."""
    t1 = _w_field(torch.full((W,), 4.0))                              # updraft everywhere
    S = 4
    half = torch.stack([_w_field(torch.full((W,), 4.0)) if i < 2 else _w_field(torch.zeros(W)) for i in range(S)])
    none = torch.stack([_w_field(torch.zeros(W))] * S)
    assert sum(cl.feature_log_score(half, t1, Z, 2.0, 1.5, 0.5)) < sum(cl.feature_log_score(none, t1, Z, 2.0, 1.5, 0.5))
    u, _ = cl.feature_log_score(half, t1, Z, 2.0, 1.5, 0.5)
    assert abs(float(u) - np.log(1 / 0.5)) < 0.05                    # p = 0.5 -> log 2


def test_stage_c_with_feature_cross_entropy_both_kinds():
    from ks_latent.cm1.data import ColumnData, Normalizer, synthetic_members
    from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder
    from ks_latent.cm1.train import StageCConfig, train_stage_c
    ms = synthetic_members(3, T=60, NX=96 * 4, Z=Z)
    norm = Normalizer.fit(ms); d = ColumnData(ms, norm, torch.device("cpu"))
    kw = dict(window_sites=40, halo_sites=2, batch=2, steps=4, k_max=4, val_every=2, val_batches=1, warmup=1, feat_weight=1.0, feat_up=0.5, feat_down=0.5)
    for noise, extra in ((0, {"w_mse_scale": 0.0}), (2, {"crps_samples": 3})):
        torch.manual_seed(0)
        ae = build_autoencoder(ColumnAEConfig(n_channels=d.C, n_levels=Z, patch=4, hidden=16, latent_channels=6, n_mix=1), None)
        prop = StencilPropagator(StencilPropConfig(latent_channels=6, hidden=8, layers=2, radius=1, noise_channels=noise))
        out = train_stage_c(ae, prop, d, d, StageCConfig(**kw, **extra), log=lambda *_: None)
        h = out["history"][-1]
        assert np.isfinite(h["l_roll"]) and h["term_feat_up"] > 0 and h["term_feat_down"] > 0
    with pytest.raises(ValueError):
        train_stage_c(ae, prop, d, d, StageCConfig(window_sites=40, halo_sites=2, batch=2, steps=2, k_max=4, w_mse_scale=0.0), log=lambda *_: None)
