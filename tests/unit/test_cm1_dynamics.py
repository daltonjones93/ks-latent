"""Plan Q: dispersion relations and relative equilibria on stencil maps with known answers."""
import numpy as np
import torch

from ks_latent.cm1.dynamics import data_dispersion, dispersion, relative_equilibrium, shift_sites, stencil_blocks


class _Linear(torch.nn.Module):
    """z_j^{n+1} = sum_m a_m z_{j+m}^n per channel (a translation-invariant linear stencil)."""
    def __init__(self, a):
        super().__init__()
        self.a = torch.as_tensor(a, dtype=torch.float64)
        self.cfg = None

    def forward(self, z):
        R = (len(self.a) - 1) // 2
        zp = torch.cat([z[:, :1].expand(-1, R, -1), z, z[:, -1:].expand(-1, R, -1)], dim=1)
        P = z.shape[1]
        return sum(self.a[m] * zp[:, m:m + P] for m in range(len(self.a)))


def test_stencil_blocks_and_advection_dispersion():
    prop = _Linear([1.0, 0.0, 0.0])                       # z_j <- z_{j-1}: one site per step to the right
    A = stencil_blocks(prop, torch.zeros(2), reach=1)
    np.testing.assert_allclose(A[:, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(A[:, 0, 1], 0.0, atol=1e-12)
    d = dispersion(A, dx_km=2.0, dt_min=1.0, nk=16)
    np.testing.assert_allclose(d["sigma"], 0.0, atol=1e-12)               # neutral
    ok = d["k"] * 2.0 < np.pi * 0.99                                       # below the Nyquist aliasing
    np.testing.assert_allclose(d["c_phase"][ok], 2.0, atol=1e-9)          # 1 site (2 km) per min


def test_diffusion_dispersion():
    a = 0.2
    prop = _Linear([a, 1 - 2 * a, a])
    d = dispersion(stencil_blocks(prop, torch.zeros(1), reach=1), dx_km=1.0, dt_min=1.0, nk=32)
    expect = np.log(np.abs(1 - 2 * a * (1 - np.cos(d["k"]))))
    np.testing.assert_allclose(d["sigma"][:, 0], expect, atol=1e-10)


def test_data_dispersion_finds_a_wave_speed():
    T, NX, dx, dt, c = 128, 256, 0.25, 1.0, 1.5
    x, t = np.arange(NX) * dx, np.arange(T) * dt
    f = np.cos(2 * np.pi / 16.0 * (x[None] - c * t[:, None]))             # wavelength 16 km at 1.5 km/min
    d = data_dispersion(f, dx, dt)
    i = np.argmin(np.abs(d["k"] - 2 * np.pi / 16.0))
    assert abs(d["c_peak"][i] - c) < 0.15


def test_shift_and_relative_equilibrium_of_advection():
    Z = torch.arange(10, dtype=torch.float64)[:, None].repeat(1, 2)
    torch.testing.assert_close(shift_sites(Z, torch.tensor(1.0, dtype=torch.float64))[2:], Z[1:-1])
    prop = _Linear([1.0, 0.0, 0.0])                       # every field is a travelling wave at 1 site/step
    g = torch.Generator().manual_seed(0)
    Z0 = torch.randn(40, 2, generator=g, dtype=torch.float64)
    r = relative_equilibrium(prop, Z0, T=2, halo=2, margin=6)
    assert abs(r["s_sites_per_step"] - 1.0) < 0.02 and r["residual"] < 1e-3
    i0 = int(np.argmin(np.abs(np.array(r["scan_s"]))))
    assert r["scan_residual"][i0] > 0.5                                    # not a fixed point (s = 0)
    rz = relative_equilibrium(prop, Z0, T=2, halo=2, margin=6, refine_z=True, iters=50, lam=1.0)
    assert rz["residual"] < 1e-2 and float(((rz["Z"] - Z0) ** 2).mean()) < 0.05   # stays near the start
