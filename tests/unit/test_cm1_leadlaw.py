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


def test_torch_library_matches_numpy_and_refit_recovers_a_perturbed_law():
    """The differentiable library equals the numpy one; refitting the support on rollouts repairs coefficients
    that were perturbed away from the true advection-diffusion law."""
    import torch
    from ks_latent.cm1.leadlaw import refit_rollout, torch_library
    rng = np.random.default_rng(3)
    z = rng.normal(size=(5, 12, 2)); v = rng.normal(size=(5, 12, 2))
    for kind in ("ode", "pde"):
        for vv in (None, v):
            F, _ = library(z, kind, vv)
            Ft = torch_library(torch.as_tensor(z), kind, None if vv is None else torch.as_tensor(vv)).numpy()
            assert np.allclose(F, Ft)
    P, T, c, D, sub = 40, 80, 0.3, 0.2, 50
    trajs = []
    for z in _fields(P, 3, rng):
        traj = []
        for _ in range(T):
            traj.append(z.copy())
            for _ in range(sub):
                z = z + (-c * _d1(z) + D * _lap(z)) / sub
        trajs.append(np.array(traj)[:, :, None])
    _, names = library(trajs[0][:2], "pde")
    xi = np.zeros((len(names), 1)); xi[names.index("d1z0")] = -0.2; xi[names.index("d2z0")] = 0.1
    xi0 = xi.copy()
    new = refit_rollout(xi, trajs, "pde", halo=3, k_max=10, iters=300, batch=4, lr=1e-2, cap=0.0)
    assert np.array_equal(xi, xi0)                                       # the input is not modified
    assert abs(new[names.index("d1z0"), 0] + c) < 0.05 and abs(new[names.index("d2z0"), 0] - D) < 0.05
    assert np.count_nonzero(new) == 2                                    # the support is kept


def test_fourth_order_stencils_and_recovery_of_hyperdiffusion():
    """d3/d4 stencils are exact on cubic/quartic polynomials (interior); the pde4 fit recovers
    z_t = -c d1z + D d2z - H d4z (a damped Kuramoto-Sivashinsky-type linear operator)."""
    from ks_latent.cm1.leadlaw import site_derivs
    x = np.arange(20, dtype=float)[:, None]
    d1, d2, d3, d4 = site_derivs(x ** 4, 4)
    i = slice(2, -2)
    assert np.allclose(d4[i], 24.0) and np.allclose(d3[i], 24 * x[i])          # d3(x^4) = 24x, d4 = 24 exactly
    rng = np.random.default_rng(4)
    P, T, c, D, H, sub = 48, 100, 0.2, 0.1, 0.05, 50
    d4f = lambda z: np.roll(z, 2) - 4 * np.roll(z, 1) + 6 * z - 4 * np.roll(z, -1) + np.roll(z, -2)
    Xs, Ys, G = [], [], []
    xs = np.arange(P)
    rich = [sum(rng.normal() * np.sin(2 * np.pi * m * xs / P + rng.uniform(0, 6)) for m in range(1, 11)) for _ in range(4)]
    for g, z in enumerate(rich):           # wavenumbers up to 10, so d4 is not negligible against d2
        traj = []
        for _ in range(T):
            traj.append(z.copy())
            for _ in range(sub):
                z = z + (-c * _d1(z) + D * _lap(z) - H * d4f(z)) / sub
        X, Y, names = training_rows(np.array(traj)[:12, :, None], "pde4", 1, halo=3)   # early frames, before damping
        Xs.append(X); Ys.append(Y); G.append(np.full(len(X), g))
    xi, _ = fit(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G), threshold=0.05, n_boot=3)
    coef = dict(zip(names, xi[:, 0]))
    assert abs(coef["d1z0"] + c) < 0.03 and abs(coef["d2z0"] - D) < 0.04 and abs(coef["d4z0"] + H) < 0.02
