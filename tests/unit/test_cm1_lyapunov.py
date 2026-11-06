"""L1/L5: CM1 latent Lyapunov adapter and spreading, on maps with known answers."""
import numpy as np
import pytest
import torch

from ks_latent.cm1.lyapunov import latent_spectrum, spreading


class _Diag:
    """z_j,c -> a_c z_j,c: every interior site contributes log a_c to the spectrum."""
    def __init__(self, a):
        self.a = torch.as_tensor(a, dtype=torch.float64)
        self.cfg = None

    def eval(self):
        return self

    def __call__(self, z):
        return z * self.a


class _Shift:
    """z_j -> z_{j-1} (information moves one site per step to the right)."""
    cfg = None

    def eval(self):
        return self

    def __call__(self, z):
        return torch.cat([z[:, :1], z[:, :-1]], dim=1)


@pytest.mark.parametrize("mode", ["own", "truth"])
def test_spectrum_of_a_diagonal_map(mode):
    a = [1.5, 0.9, 0.5]
    P, c, halo = 8, 3, 1
    T = 260
    z = torch.randn(T, P, c, dtype=torch.float64)
    r = latent_spectrum(_Diag(a), z, t_start=0, n_steps=20, halo=halo, mode=mode, warmup=200, qr_every=2)   # align with the invariant directions first
    expect = np.sort(np.repeat(np.log(a), P - 2 * halo))[::-1]
    np.testing.assert_allclose(r.exponents, expect, atol=1e-10)
    assert r.n_positive == P - 2 * halo


def test_spectrum_rejects_a_window_past_the_end():
    with pytest.raises(ValueError):
        latent_spectrum(_Diag([1.0]), torch.zeros(10, 4, 1, dtype=torch.float64), t_start=5, n_steps=10, halo=1)


def test_spreading_speed_of_a_shift_map():
    P, c, halo = 60, 2, 1
    z = torch.randn(40, P, c, dtype=torch.float64)
    s = spreading(_Shift(), z, t_start=0, n_steps=20, halo=halo, site=20)
    assert abs(s["speed_sites_per_step"]["0.01"] - 1.0) < 1e-9
