"""Plan Q: dynamical invariants of the CM1 latent propagator (a translation-invariant stencil map).

Q1  `stencil_blocks` + `dispersion`: linearize one step about a homogeneous latent state z_bar; away from the
    domain edges the Jacobian is a block-Toeplitz stencil A_m = d f_j / d z_{j+m}, m = -R..R, so a Fourier
    mode z_j = v e^{i k j Delta} is multiplied per step by the eigenvalues mu of J(k) = sum_m A_m e^{i k m Delta}.
    Growth rate sigma = ln|mu| / dt, phase speed c = -arg(mu) / (k dt) (positive: towards larger x), group
    speed d omega / d k by finite differences along a tracked branch.
    `data_dispersion`: the (k, omega) power spectrum of a field in x-t (the data side of the check).
Q2/Q3  `relative_equilibrium`: minimize || shift(f^T(Z), -s T) - Z || over the interior of Z and a real shift
    speed s (sites per step; linear interpolation for fractional shifts), from an encoded storm state. T = 1
    with small residual: a travelling wave (steady squall line in the moving frame); larger T: a relative
    periodic orbit (cell-regeneration cycle).
"""
from __future__ import annotations

import numpy as np
import torch


def stencil_blocks(prop, z_bar: torch.Tensor, reach: int, P: int = 0) -> np.ndarray:
    """(2 reach + 1, c, c) Jacobian blocks of one step at the centre site of a homogeneous field z_bar (c,),
    A[m + reach] = d f_centre / d z_{centre + m}. The field is wide enough that edges do not reach the centre."""
    c = z_bar.numel()
    P = P or 4 * reach + 9
    j = P // 2
    base = z_bar.reshape(1, 1, c).expand(1, P, c).clone().double()
    prop = prop.double()

    def centre_out(win):                               # win (2R+1, c) -> f(...)_centre (c,)
        z = base.clone()
        z[0, j - reach:j + reach + 1] = win
        return prop(z)[0, j]
    win0 = base[0, j - reach:j + reach + 1].clone()
    Jac = torch.autograd.functional.jacobian(centre_out, win0)          # (c, 2R+1, c)
    return Jac.permute(1, 0, 2).detach().numpy()                         # (2R+1, c_out, c_in)


def dispersion(A: np.ndarray, dx_km: float, dt_min: float, nk: int = 129) -> dict:
    """Eigenvalues of the stencil symbol on k in (0, pi / dx]. Returns k (rad/km), wavelength (km), mu (nk, c)
    sorted by growth rate, sigma (per min) and c_phase (km/min)."""
    R = (A.shape[0] - 1) // 2
    m = np.arange(-R, R + 1)
    k = np.linspace(np.pi / dx_km / nk, np.pi / dx_km, nk)
    mus = []
    for kk in k:
        J = np.tensordot(np.exp(1j * kk * m * dx_km), A, axes=(0, 0))
        ev = np.linalg.eigvals(J)
        mus.append(ev[np.argsort(-np.abs(ev))])
    mu = np.array(mus)
    sigma = np.log(np.abs(mu)) / dt_min
    c_phase = -np.angle(mu) / (k[:, None] * dt_min)
    return {"k": k, "wavelength_km": 2 * np.pi / k, "mu": mu, "sigma": sigma, "c_phase": c_phase}


def data_dispersion(field: np.ndarray, dx_km: float, dt_min: float) -> dict:
    """(T, NX) x-t field -> power P(omega, k) (rfft in x, fft in t, means removed, Hann windows) and, for each
    k, the phase speed of the power peak (km/min; positive towards larger x)."""
    T, NX = field.shape
    f = field - field.mean()
    f = f * np.hanning(T)[:, None] * np.hanning(NX)[None, :]
    F = np.fft.fft(np.fft.rfft(f, axis=1), axis=0)                      # (T, NX//2+1)
    Pw = np.abs(np.fft.fftshift(F, axes=0)) ** 2
    om = np.fft.fftshift(np.fft.fftfreq(T, d=dt_min)) * 2 * np.pi          # rad/min
    k = np.fft.rfftfreq(NX, d=dx_km) * 2 * np.pi                           # rad/km
    # a mode e^{i(kx - omega t)} appears in numpy's forward FFT (kernel e^{-i 2 pi f t}) at frequency -omega
    c_peak = np.full(k.size, np.nan)
    for i in range(1, k.size):
        c_peak[i] = -om[np.argmax(Pw[:, i])] / k[i]
    return {"k": k, "omega": om, "power": Pw, "c_peak": c_peak}


def shift_sites(Z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Z (P, c) shifted by s sites towards larger index (fractional: linear interpolation; edges repeated)."""
    P = Z.shape[0]
    x = torch.arange(P, dtype=Z.dtype, device=Z.device) - s
    x = x.clamp(0, P - 1)
    i0 = x.floor().long().clamp(0, P - 2)
    w = (x - i0.to(Z.dtype))[:, None]
    return (1 - w) * Z[i0] + w * Z[i0 + 1]


def relative_equilibrium(prop, Z0: torch.Tensor, T: int, halo: int, margin: int, iters: int = 400,
                         lr: float = 0.01, s_range=(-2.0, 2.0), refine_z: bool = False, lam: float = 1.0) -> dict:
    """Residual r(Z, s) = ||shift(f^T(Z), -s T) - Z||^2 / ||Z0 - mean||^2 on the interior (margin sites left out).
    Step 1: the best shift speed s (sites per step) for the given state Z0: a coarse scan over s_range, then
    Adam on s. Step 2 (refine_z): also adjust the interior of Z (halo held), with a penalty lam ||Z - Z0||^2 /
    ||Z0 - mean||^2 that keeps it near the storm (without it the search collapses to a homogeneous field,
    which trivially repeats itself under any shift). Returns Z, s, residuals before and after, history."""
    prop = prop.eval()
    for p_ in prop.parameters():
        p_.requires_grad_(False)
    P, c = Z0.shape
    edge = Z0.detach().clone()
    ref = Z0[margin:P - margin] - Z0.mean(0, keepdim=True)
    scale = float((ref ** 2).mean()) + 1e-12

    def roll(Z):
        Y = Z[None]
        for _ in range(T):
            Y = prop(Y)
            Y = torch.cat([edge[None, :halo], Y[:, halo:P - halo], edge[None, P - halo:]], dim=1)
        return Y[0]

    def res(Z, Y, s):
        return ((shift_sites(Y, -s * T) - Z)[margin:P - margin] ** 2).mean() / scale
    with torch.no_grad():
        Y0 = roll(Z0)
        grid = np.arange(s_range[0], s_range[1] + 1e-9, 0.05)
        scan = [float(res(Z0, Y0, torch.tensor(g, dtype=Z0.dtype))) for g in grid]
    s = torch.tensor(float(grid[int(np.argmin(scan))]), dtype=Z0.dtype, requires_grad=True)
    r_start = min(scan)
    opt = torch.optim.Adam([s], lr=lr)
    for _ in range(100):
        opt.zero_grad()
        r = res(Z0, Y0, s)
        r.backward()
        opt.step()
    with torch.no_grad():
        r_shift = float(res(Z0, Y0, s))
    hist, Z = [], Z0.detach().clone()
    if refine_z:
        inner = Z0[halo:P - halo].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([inner, s], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            Z = torch.cat([edge[:halo], inner, edge[P - halo:]])
            r = res(Z, roll(Z), s)
            loss = r + lam * ((Z - Z0) ** 2).mean() / scale
            loss.backward()
            opt.step()
            hist.append(float(r))
        with torch.no_grad():
            Z = torch.cat([edge[:halo], inner, edge[P - halo:]])
            r_final = float(res(Z, roll(Z), s))
    else:
        r_final = r_shift
    return {"Z": Z.detach(), "s_sites_per_step": float(s.detach()), "residual_scan_best": r_start,
            "residual_shift_fit": r_shift, "residual": r_final, "scan_s": grid.tolist(), "scan_residual": scan,
            "history": hist}
