"""9a: threshold-weighted, pooled and low-pass terms for convective cells (ks_latent/cm1/cell_losses.py) and their use in Stage C."""
import numpy as np
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
