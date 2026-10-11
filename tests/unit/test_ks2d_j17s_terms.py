"""J17S candidate processes (2026-10-10): anisotropic gradient term and Burgers-type x term in 2D KS."""
import numpy as np
import pytest

from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d import (
    _nonlinear2d, get_operator2d, grad_aniso_term2d, step2d, step_with_tangent2d,
)

CFG = KSConfig2D(Lx=22.0, Ly=22.0, NX=32, NY=32, dt=0.02, snapshot_every=5, spinup_time=0.0, seed=0)


def _grid():
    x = np.arange(CFG.NX) * CFG.Lx / CFG.NX
    return np.meshgrid(x, x, indexing="ij")


def test_grad_aniso_single_mode_analytic():
    op = get_operator2d(CFG)
    X, _ = _grid()
    k = 2 * np.pi * 3 / CFG.Lx
    u = np.cos(k * X)
    got = np.fft.ifft2(grad_aniso_term2d(np.fft.fft2(u), 0.4, op.KX, op.KY)).real
    want = -0.2 * (k**2 * np.sin(k * X) ** 2 - 0.5 * k**2)  # mean removed
    assert np.allclose(got, want, atol=1e-10)


def test_delta_one_switches_off_y_nonlinearity():
    op = get_operator2d(CFG)
    _, Y = _grid()
    u = np.cos(2 * np.pi * 2 * Y / CFG.Ly)
    v = np.fft.fft2(u)
    tot = _nonlinear2d(v, op.KX, op.KY) + grad_aniso_term2d(v, 1.0, op.KX, op.KY)
    tot[0, 0] = 0.0
    assert np.abs(np.fft.ifft2(tot).real).max() < 1e-10


@pytest.mark.parametrize("kw", [{"ganiso": 0.5}, {"aniso": 0.5}, {"ganiso": -0.3, "aniso": 0.2}])
def test_tangent_matches_finite_difference(kw):
    op = get_operator2d(CFG)
    rng = np.random.default_rng(0)
    X, Y = _grid()
    u = np.cos(2 * np.pi * X / CFG.Lx) + 0.5 * np.sin(2 * np.pi * 2 * Y / CFG.Ly) + 0.1 * rng.standard_normal(X.shape)
    u -= u.mean()
    du = rng.standard_normal(u.shape); du -= du.mean()
    v, dv = np.fft.fft2(u), np.fft.fft2(du)
    _, dV = step_with_tangent2d(v, dv[None], op, **kw)
    eps = 1e-6
    fd = (step2d(v + eps * dv, op, **kw) - step2d(v - eps * dv, op, **kw)) / (2 * eps)
    assert np.allclose(np.fft.ifft2(dV[0]).real, np.fft.ifft2(fd).real, atol=1e-7)


def test_ganiso_zero_is_standard():
    op = get_operator2d(CFG)
    v = np.fft.fft2(np.random.default_rng(1).standard_normal((CFG.NX, CFG.NY)))
    v[0, 0] = 0
    assert np.array_equal(step2d(v, op), step2d(v, op, ganiso=0.0))
