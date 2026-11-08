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


def test_dd_and_cubic_options():
    """+dd adds exactly the derivative x derivative products; +cubic the cubic terms with at most one derivative;
    numpy and torch libraries agree with both; a KPZ-type law z_t = D d2z + lam (d1z)^2 is recovered with +dd;
    a cubic saturation z_t = a z - b z^3 + D d2z with +cubic."""
    import torch
    from ks_latent.cm1.leadlaw import parse_kind, torch_library
    z = np.random.default_rng(5).normal(size=(3, 10, 2))
    _, base = library(z, "pde")
    _, wdd = library(z, "pde+dd")
    extra = set(wdd) - set(base)
    assert extra and all(t.count("*") == 1 and t.count("d") == 2 for t in extra)
    assert "d1z0*d1z0" in extra and "d1z0*d2z1" in extra
    _, wc = library(z, "pde4+cubic")
    cub = [t for t in wc if t.count("*") == 2]
    assert "z0*z0*z0" in cub and "z0*z1*d4z1" in cub and not any(t.count("d") > 1 for t in cub)
    for kind in ("pde+dd", "pde4+dd+cubic", "ode+cubic"):
        for vv in (None, z):
            F, _ = library(z, kind, vv)
            assert np.allclose(F, torch_library(torch.as_tensor(z), kind, None if vv is None else torch.as_tensor(vv)).numpy())
    with pytest.raises(ValueError):
        parse_kind("ode+dd")
    with pytest.raises(ValueError):
        parse_kind("pde+quartic")
    rng = np.random.default_rng(6)
    P, T, sub = 40, 60, 50
    for kind, law, check in (
            ("pde+dd", lambda z: 0.2 * _lap(z) + 0.15 * _d1(z) ** 2,
             lambda c: abs(c["d2z0"] - 0.2) < 0.03 and abs(c["d1z0*d1z0"] - 0.15) < 0.03),
            ("pde+cubic", lambda z: 0.05 * z - 0.05 * z ** 3 + 0.2 * _lap(z),
             lambda c: abs(c["z0"] - 0.05) < 0.02 and abs(c["z0*z0*z0"] + 0.05) < 0.02 and abs(c["d2z0"] - 0.2) < 0.03)):
        Xs, Ys, G = [], [], []
        for g, z in enumerate(_fields(P, 4, rng)):
            traj = []
            for _ in range(T):
                traj.append(z.copy())
                for _ in range(sub):
                    z = z + law(z) / sub
            X, Y, names = training_rows(np.array(traj)[:, :, None], kind, 1, halo=3)
            Xs.append(X); Ys.append(Y); G.append(np.full(len(X), g))
        xi, _ = fit(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(G), threshold=0.05, n_boot=3)
        assert check(dict(zip(names, xi[:, 0]))), kind


def test_mixed_derivative_bbm_type_law_is_recovered_rolled_out_and_refit():
    """+mixed: (1 - l2 d2x) dz/dt = -c d1z + D d2z. With l2 FIXED and scanned, the residual is smallest at the
    true l2, where the fit recovers c and D; the implicit rollout tracks the data; the operator matches its
    definition; the refit keeps the support and the mixed operator non-negative."""
    from ks_latent.cm1.leadlaw import d2_matrix, mixed_operator, refit_rollout, site_derivs, with_mixed
    P = 40
    D2 = d2_matrix(P)
    zz = np.random.default_rng(7).normal(size=(P, 1))
    assert np.allclose(D2 @ zz[:, 0], site_derivs(zz, 2)[1][:, 0])
    A = np.array([[0.3, 0.1], [0.0, 0.2]])
    w = np.random.default_rng(8).normal(size=(P, 2))
    lhs = w - np.stack([0.3 * (D2 @ w[:, 0]), 0.1 * (D2 @ w[:, 0]) + 0.2 * (D2 @ w[:, 1])], 1)
    assert np.allclose(mixed_operator(A, P) @ w.reshape(-1), lhs.reshape(-1))
    rng = np.random.default_rng(9)
    T, c, D, l2, sub = 80, 0.3, 0.1, 0.5, 40
    Lp = lambda z: np.roll(z, 1) - 2 * z + np.roll(z, -1)
    Minv = np.linalg.inv(np.eye(P) - l2 * np.stack([Lp(e) for e in np.eye(P)], 1))   # periodic generator
    trajs = []
    for z in _fields(P, 4, rng):
        traj = []
        for _ in range(T):
            traj.append(z.copy())
            for _ in range(sub):
                z = z + Minv @ (-c * _d1(z) + D * Lp(z)) / sub
        trajs.append(np.array(traj)[:, :, None])
    res = {}
    for l2s in (0.0, 0.25, 0.5, 1.0):
        rows = [training_rows(tr, "pde+mixed", 1, halo=3, l2=l2s) for tr in trajs]
        X, Y = np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])
        G = np.concatenate([np.full(len(r[0]), g) for g, r in enumerate(rows)])
        xi, freq = fit(X, Y, G, threshold=0.05, n_boot=3)
        res[l2s] = (np.mean((Y - X @ xi) ** 2) / np.var(Y), xi, freq, rows[0][2])
    assert min(res, key=lambda q: res[q][0]) == 0.5
    _, xi, freq, names = res[0.5]
    xi, freq, names = with_mixed(xi, freq, names, 0.5)
    coef = dict(zip(names, xi[:, 0]))
    assert names[-1] == "d2tz0" and coef["d2tz0"] == 0.5
    assert abs(coef["d1z0"] + c) < 0.03 and abs(coef["d2z0"] - D) < 0.03
    tr = trajs[0]
    pred = rollout(xi, tr[10], 15, "pde+mixed", cap=0.0, halo=0)
    i = slice(12, 28)
    assert np.abs(pred[-1, i] - tr[25, i]).mean() / np.abs(tr[25, i]).mean() < 0.1
    bad = xi.copy(); bad[-1, 0] = 0.01                                   # start the refit from a too-small l2
    new = refit_rollout(bad, trajs, "pde+mixed", halo=3, k_max=5, iters=30, batch=2, cap=0.0)
    assert np.isfinite(new).all() and np.array_equal(new != 0, bad != 0) and new[-1, 0] >= 0
    with pytest.raises(ValueError):
        training_rows(tr, "pde+mixed", 2, halo=3)
    with pytest.raises(ValueError):
        with_mixed(xi[:-1], freq[:-1], names[:-1], -0.1)
