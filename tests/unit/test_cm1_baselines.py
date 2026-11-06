"""H1: advection-extrapolation baseline."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.baselines import _shift, advection_forecast, best_shift


def _field(V, Z, NX, t, speeds):
    """Gaussian bumps per level, level k moving speeds[k] cells per frame."""
    x = torch.arange(NX, dtype=torch.float64)
    u = torch.zeros(V, Z, NX, dtype=torch.float64)
    for k in range(Z):
        for v in range(V):
            u[v, k] = torch.exp(-((x - 60 - 10 * v - speeds[k] * t) / 6.0) ** 2)
    return u.reshape(V * Z, NX)


def test_shift_fills_edges():
    u = torch.arange(6.0)
    assert _shift(u, 2).tolist() == [0, 0, 0, 1, 2, 3]
    assert _shift(u, -2).tolist() == [2, 3, 4, 5, 5, 5]
    assert _shift(u, 0).tolist() == u.tolist()


def test_best_shift_recovers_motion_and_keeps_zero_when_featureless():
    a = _field(2, 1, 200, 0, [1.0])
    b = _field(2, 1, 200, 7, [1.0])
    assert best_shift(a, b, 20) == 7
    assert best_shift(a, _field(2, 1, 200, 0, [-1.0]) * 0 + _field(2, 1, 200, -5, [1.0]), 20) == -5
    z = torch.zeros(3, 50)
    assert best_shift(z, z, 10) == 0


@pytest.mark.parametrize("mode", ["global", "per_level"])
def test_advection_forecast_exact_for_translation(mode):
    V, Z, NX, lag = 2, 3, 240, 5
    speeds = [1.0, 1.0, 1.0] if mode == "global" else [1.0, -0.4, 0.0]
    hist = torch.stack([_field(V, Z, NX, t, speeds) for t in range(lag + 1)])
    fc, sh = advection_forecast(hist, 10, Z, lag=lag, mode=mode)
    assert sh == [round(s * lag) for s in speeds]
    truth = torch.stack([_field(V, Z, NX, lag + n, speeds) for n in range(1, 11)])
    torch.testing.assert_close(fc[[4, 9]], truth[[4, 9]], atol=1e-9, rtol=0)   # whole-cell leads
    with pytest.raises(ValueError):
        advection_forecast(hist[:3], 4, Z, lag=lag)
