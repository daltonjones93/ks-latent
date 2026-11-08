"""FPe: the leading-channel law recovers known lattice laws, first and second order in time."""
import numpy as np
import pytest

from ks_latent.cm1.leadlaw import backward_velocity, fit, library, rollout, smooth_t, training_rows


def _lap(z):
    return np.roll(z, 1, 0) - 2 * z + np.roll(z, -1, 0)


def _d1(z):
    return 0.5 * (np.roll(z, -1, 0) - np.roll(z, 1, 0))


def _fields(P, n, rng):
    x = np.arange(P)
    return [sum(rng.normal() * np.sin(2 * np.pi * m * x / P + rng.uniform(0, 6)) for m in (1, 2, 3)) for _ in range(n)]


def test_library_terms_and_no_derivative_products():
    z = np.random.default_rng(0).normal(size=(4, 10, 2))
    F, names = library(z, "pde")
    assert names[0] == "1" and F.shape[-1] == len(names)
    assert not any(n.count("d") == 2 and "*" in n for n in names)       # no d*z * d*z products
    F2, n2 = library(z, "ode", v=z)
    assert "v1" in n2 and "z0*v1" in n2 and not any(n.startswith("d") for n in n2)


def test_recovers_first_order_advection_diffusion():
    rng = np.random.default_rng(1)
    P, T, c, D, sub = 40, 120, 0.3, 0.2, 50
    Xs, Ys, G = [], [], []
    for g, z in enumerate(_fields(P, 4, rng)):
        traj = []
        for _ in range(T):
            traj.append(z.copy())
            for _ in range(sub):
                z = z + (-c * _d1(z) + D * _lap(z)) / sub
        X, Y, names = training_rows(np.array(traj)[:, :, None], "pde", 1, halo=2)
        Xs.append(X); Ys.append(Y); G.append(np.full(len(X), g))
    xi, _ = fit(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G), threshold=0.05, n_boot=3)
    coef = dict(zip(names, xi[:, 0]))
    assert abs(coef["d1z0"] + c) < 0.03 and abs(coef["d2z0"] - D) < 0.04
    assert sum(abs(v) > 1e-9 for k, v in coef.items() if k not in ("1", "d1z0", "d2z0")) <= 1


def test_recovers_second_order_wave_equation_and_rolls_it_out():
    """z_tt = c^2 d2z: the time_order-2 pde fit finds c^2 on d2z0, nothing large on v; its rollout tracks the data."""
    rng = np.random.default_rng(2)
    P, T, c2, sub = 48, 150, 0.25, 40
    trajs = []
    for z in _fields(P, 3, rng):
        v = np.zeros_like(z); traj = []
        dt = 1.0 / sub
        for _ in range(T):
            traj.append(z.copy())
            for _ in range(sub):                       # leapfrog-like (symplectic Euler)
                v = v + dt * c2 * _lap(z)
                z = z + dt * v
        trajs.append(np.array(traj)[:, :, None])
    Xs, Ys, G = [], [], []
    for g, tr in enumerate(trajs):
        X, Y, names = training_rows(tr, "pde", 2, halo=2)
        Xs.append(X); Ys.append(Y); G.append(np.full(len(X), g))
    xi, _ = fit(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G), threshold=0.05, n_boot=3)
    coef = dict(zip(names, xi[:, 0]))
    assert abs(coef["d2z0"] - c2) < 0.03
    assert max(abs(v) for k, v in coef.items() if k not in ("1", "d2z0")) < 0.05
    tr = trajs[0]
    pred = rollout(xi, tr[10], 20, "pde", cap=0.0, halo=0, time_order=2, v0=backward_velocity(tr, 10))
    i = slice(12, 36)        # the data are periodic, the law's edges are not: score sites the edges cannot reach
    err = np.abs(pred[-1, i] - tr[30, i]).mean() / np.abs(tr[30, i]).mean()
    assert np.isfinite(pred).all() and err < 0.1
    with pytest.raises(ValueError):
        rollout(xi, tr[10], 2, "pde", cap=0.0, halo=0, time_order=2)


def test_smooth_t_preserves_constants_and_length():
    z = np.ones((20, 5, 2))
    assert np.allclose(smooth_t(z, 2.0), 1.0) and smooth_t(z, 0) is z
