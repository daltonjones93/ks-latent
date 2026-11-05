"""Verification metrics for the CM1 gust front (ks_latent/cm1/verify.py): known answers."""
import numpy as np
import pytest

from ks_latent.cm1.verify import displacement_km, fss, main_pool_front_km

pytestmark = pytest.mark.unit
DX, NX = 0.25, 1600
x = (np.arange(NX) + 0.5) * DX


def pool(west, east, depth=5.0, edge=1.0):
    """A cold pool from `west` to `east` km with smooth (tanh) edges."""
    return -depth * 0.25 * (1 + np.tanh((x - west) / edge)) * (1 - np.tanh((x - east) / edge))


def test_main_pool_ignores_isolated_cold_patch_ahead_and_bridges_gaps():
    th = pool(100, 180)
    assert abs(main_pool_front_km(th) - 180) < 1.0
    th_patch = th.copy()
    th_patch[(x > 200) & (x < 201)] = -3.0          # a new cell's 1 km cold patch, 20 km ahead
    assert abs(main_pool_front_km(th_patch) - main_pool_front_km(th)) < 1e-9
    th_wide = th + pool(200, 206, depth=2.0)        # a 6 km pool ahead, but much less cold mass
    assert abs(main_pool_front_km(th_wide) - main_pool_front_km(th)) < 1e-9
    th_gap = th.copy()
    th_gap[(x > 150) & (x < 151)] = 0.5             # a 1 km warm hole inside the pool
    assert abs(main_pool_front_km(th_gap) - main_pool_front_km(th)) < 1e-9
    # once the cold air ahead is close enough to merge, the front really advances
    th_merge = th + pool(180.5, 190, depth=3.0)
    assert main_pool_front_km(th_merge) > 188
    # two comparable pools (a young storm): always the eastern one, never a flip by size
    two = pool(150, 170) + pool(175, 190, depth=4.0)
    assert abs(main_pool_front_km(two) - 190) < 1.0
    two_b = pool(150, 170) + pool(175, 190, depth=6.0)
    assert abs(main_pool_front_km(two_b) - 190) < 1.0
    assert np.isnan(main_pool_front_km(np.zeros(NX)))
    assert main_pool_front_km(np.stack([th, np.zeros(NX)])).shape == (2,)


@pytest.mark.parametrize("delta", [-3.0, -0.4, 0.0, 1.3, 5.0])
def test_displacement_recovers_known_shift(delta):
    ob = pool(100, 180)
    fc = pool(100 + delta, 180 + delta)
    est = displacement_km(fc, ob, center_km=180)
    assert abs(est - delta) < 0.03, est


def test_displacement_nan_outside_domain_and_without_front():
    ob = pool(100, 180)
    assert np.isnan(displacement_km(ob, ob, center_km=float("nan")))
    assert np.isnan(displacement_km(ob, ob, center_km=0.1, window_km=1.0))


def test_fss_bounds_and_scale_dependence():
    ob = np.zeros(NX); ob[400:420] = 1.0            # a 5 km object
    assert all(v == 1.0 for v in fss(ob, ob, 0.5, [1, 8, 64]).values())
    fc = np.zeros(NX); fc[440:460] = 1.0            # same object displaced 10 km
    s = fss(fc, ob, 0.5, [1, 4, 16, 64, 256])
    assert s[1] == 0.0
    vals = [s[n] for n in (1, 4, 16, 64, 256)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])) and vals[-1] > 0.8
    assert np.isnan(fss(np.zeros(NX), np.zeros(NX), 0.5, [4])[4])
