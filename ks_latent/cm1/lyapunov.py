"""Finite-time Lyapunov spectra of the CM1 latent propagator (plan L1) and perturbation spreading (L5).

The propagator maps the latent field z (P sites x c channels) one frame ahead. The `halo` sites at each end
are the boundary condition; here they are HELD at their state at the window start, so the step is a map
on the interior, D = (P - 2 halo) * c dimensions, and Paper I's `ks_latent.analysis.lyapunov.benettin`
(QR reorthonormalization, float64 on CPU) applies unchanged. Two linearizations:

  mode="own"    the base trajectory is the propagator's own free run from the encoded truth at the start;
                the exponents are those of the surrogate's dynamics.
  mode="truth"  the tangent map is evaluated on the ENCODED CM1 TRAJECTORY (the base "state" handed to
                benettin is ignored and replaced by the next true latent state). With no tangent-linear
                CM1 available, this is the surrogate's estimate of CM1's own tangent dynamics on the
                latent manifold; a free surrogate run drifts away from the true states.

A squall line is not on a stationary attractor (it evolves over 6 h, open boundaries), so these are
finite-time exponents over a window, per frame, converted to per minute with `dt_min`.
Tangents are pushed through the network with `torch.func.jvp` in the model's precision (float32), in
chunks; the QR is float64 on CPU (brief section 1.2).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp, vmap

from ks_latent.analysis.lyapunov import LyapunovResult, benettin


def _interior_map(prop, edge: torch.Tensor, halo: int):
    """f(interior (D,)) -> interior (D,) of one propagator step with the boundary sites fixed to `edge`."""
    P, c = edge.shape
    Pi = P - 2 * halo

    def f(x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([edge[:halo], x.reshape(Pi, c), edge[P - halo:]], dim=0) if halo else x.reshape(Pi, c)
        return prop(z[None])[0, halo:P - halo].reshape(-1)
    return f


def make_step_fn(prop, z_traj: torch.Tensor, t_start: int, halo: int, mode: str = "truth", chunk: int = 512):
    """A `benettin` step for the latent trajectory `z_traj` (T, P, c) from frame `t_start`."""
    if mode not in ("own", "truth"):
        raise ValueError(f"unknown mode {mode!r}")
    dev = z_traj.device
    edge = z_traj[t_start].clone()
    f = _interior_map(prop, edge, halo)
    P = z_traj.shape[1]
    interior = lambda t: z_traj[t, halo:P - halo].reshape(-1)
    counter = {"n": 0}

    def step(state: np.ndarray, tangent: np.ndarray):
        if mode == "truth":
            x = interior(t_start + counter["n"])
        else:
            x = torch.as_tensor(state, dtype=z_traj.dtype, device=dev)
        V = torch.as_tensor(tangent, dtype=z_traj.dtype, device=dev)
        outs, x_next = [], None
        with torch.no_grad():
            for i in range(0, V.shape[0], chunk):
                xn, jv = vmap(lambda v: jvp(f, (x,), (v,)))(V[i:i + chunk])
                outs.append(jv)
                x_next = xn[0]
        counter["n"] += 1
        if mode == "truth":
            x_next = interior(t_start + counter["n"])
        return x_next.double().cpu().numpy(), torch.cat(outs).double().cpu().numpy()
    return step


def latent_spectrum(prop, z_traj: torch.Tensor, t_start: int, n_steps: int, halo: int, *, mode: str = "truth",
                    warmup: int = 0, n_directions: int | None = None, qr_every: int = 2, dt_min: float = 1.0,
                    record_steps: list[int] | None = None, seed: int = 0) -> LyapunovResult:
    """Finite-time spectrum over frames [t_start + warmup, t_start + warmup + n_steps], per minute.
    The boundary sites are held at frame t_start throughout."""
    if t_start + warmup + n_steps >= z_traj.shape[0]:
        raise ValueError("window runs past the end of the trajectory")
    P, c = z_traj.shape[1:]
    D = (P - 2 * halo) * c
    prop.eval()
    step = make_step_fn(prop, z_traj, t_start, halo, mode)
    x0 = z_traj[t_start, halo:P - halo].reshape(-1).double().cpu().numpy()
    return benettin(step, x0, n_directions=n_directions or D, n_steps=n_steps, qr_every=qr_every,
                    dt_per_step=dt_min, warmup_steps=warmup, rng=np.random.default_rng(seed),
                    record_steps=record_steps)


@torch.no_grad()
def spreading(prop, z_traj: torch.Tensor, t_start: int, n_steps: int, halo: int, site: int,
              thresholds=(1e-2, 1e-3, 1e-4), seed: int = 0) -> dict:
    """L5: a random perturbation at one site, pushed through the tangent map along the encoded truth; per
    step and threshold, the distance (sites) from `site` to the farthest site whose perturbation norm
    exceeds threshold x the current maximum. The front speed (sites per step) is the slope."""
    P, c = z_traj.shape[1:]
    Pi = P - 2 * halo
    if not halo <= site < P - halo:
        raise ValueError("perturbed site must be in the interior")
    g = torch.Generator().manual_seed(seed)
    v = torch.zeros(Pi, c, dtype=z_traj.dtype)
    v[site - halo] = torch.randn(c, generator=g, dtype=z_traj.dtype)
    v = (v / v.norm()).reshape(-1).to(z_traj.device)
    f = _interior_map(prop, z_traj[t_start].clone(), halo)
    radii = {th: [] for th in thresholds}
    for n in range(n_steps):
        x = z_traj[t_start + n, halo:P - halo].reshape(-1)
        _, v = jvp(f, (x,), (v,))
        amp = v.reshape(Pi, c).norm(dim=1).cpu().numpy()
        v = v / v.norm()                                     # keep it finite; the shape is what matters
        for th in thresholds:
            far = np.nonzero(amp >= th * amp.max())[0] + halo
            radii[th].append(int(np.abs(far - site).max()))
    out = {"site": site, "radius_sites": {str(th): r for th, r in radii.items()}, "speed_sites_per_step": {}}
    t = np.arange(1, n_steps + 1)
    for th, r in radii.items():
        r = np.asarray(r, float)
        ok = (r > 0) & (r < min(site - halo, P - halo - 1 - site))       # before the front reaches an edge
        out["speed_sites_per_step"][str(th)] = float(np.polyfit(t[ok], r[ok], 1)[0]) if ok.sum() >= 3 else float("nan")
    return out
