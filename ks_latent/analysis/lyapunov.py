"""Lyapunov spectra via Benettin's method with QR reorthonormalization.

Brief §6.2: "Write this with a clean interface accepting any lattice model,
not just the 44-dim propagator." The core `benettin` function is agnostic to
what the state/tangent vectors represent -- it only needs a callable that
advances a base state and a batch of tangent vectors together by one step.
That makes it reusable for the raw KS PDE (this phase, `lyapunov_spectrum_ks`
below), the trained latent propagator (Phase 4, `single_state`/`two_step`
modes), and the gauge-invariant lattice-model comparator of Phase 12.

QR reorthonormalization must be done on a fixed (time-independent) inner
product for the exponents to be well defined; which fixed inner product is
used does not affect the resulting exponents. For the KS PDE this is done in
*physical* space (plain Euclidean norm on the real (NX,) field), not in the
Fourier coefficients, to avoid the conjugate-symmetry bookkeeping a complex
QR would otherwise require.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.fft import fft, ifft
from torch.func import jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from ks_latent.config import KSConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.solver.ks import get_operator, spinup, step_with_tangent


def _maybe_noise_disabled(propagator) -> contextlib.AbstractContextManager:
    """`EnsemblePropagator.noise_disabled()` if `propagator` has one
    (`torch.randn` cannot run inside the `vmap`/`jvp` tangent-map
    linearization below -- see its docstring and
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4), else a no-op --
    every other propagator in this codebase is already deterministic."""
    hook = getattr(propagator, "noise_disabled", None)
    return hook() if hook is not None else contextlib.nullcontext()

StepWithTangentFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class LyapunovResult:
    exponents: np.ndarray  # sorted descending, physical-time units (1/time)
    n_directions: int
    total_time: float
    kaplan_yorke_dimension: float
    n_positive: int


def kaplan_yorke_dimension(exponents_desc: np.ndarray) -> float:
    """D_KY = j + (sum_{i<=j} lambda_i) / |lambda_{j+1}|, j = largest run with
    non-negative partial sum (Kaplan & Yorke 1979)."""
    exponents_desc = np.asarray(exponents_desc)
    cumsum = 0.0
    j = 0
    for lam in exponents_desc:
        if cumsum + lam < 0:
            break
        cumsum += lam
        j += 1
    if j == 0:
        return 0.0
    if j >= len(exponents_desc):
        raise ValueError(
            "Not enough Lyapunov directions computed to bracket D_KY "
            f"(cumulative sum of all {len(exponents_desc)} exponents is still "
            "non-negative). Compute more directions."
        )
    lam_next = exponents_desc[j]
    return j + cumsum / abs(lam_next)


def benettin(
    step_with_tangent_fn: StepWithTangentFn,
    state0: np.ndarray,
    *,
    n_directions: int,
    n_steps: int,
    qr_every: int,
    dt_per_step: float,
    warmup_steps: int = 0,
    rng: np.random.Generator | None = None,
    max_abs_state: float | None = None,
) -> LyapunovResult:
    """Benettin's algorithm for the Lyapunov spectrum of any discrete-time map.

    `step_with_tangent_fn(state, tangent_batch) -> (state_next, tangent_batch_next)`
    advances the base trajectory and a `(n_directions, state_dim)` batch of
    tangent vectors by one step, in whatever real vector representation the
    caller chooses (must support `np.linalg.qr` sensibly, i.e. be a real
    Euclidean space).

    `n_steps` and `warmup_steps` must each be a multiple of `qr_every` --
    every interval is renormalized, none is partial, so the exponent average
    is over uniform-length intervals.
    """
    if n_steps % qr_every != 0:
        raise ValueError(f"n_steps={n_steps} must be a multiple of qr_every={qr_every}")
    if warmup_steps % qr_every != 0:
        raise ValueError(f"warmup_steps={warmup_steps} must be a multiple of qr_every={qr_every}")
    if rng is None:
        rng = np.random.default_rng(0)

    state_dim = np.asarray(state0).shape[-1]
    q0, _ = np.linalg.qr(rng.normal(size=(state_dim, n_directions)))
    tangent = np.ascontiguousarray(q0.T)  # (n_directions, state_dim), rows orthonormal
    state = np.asarray(state0).copy()

    def check_bounded(s):
        if not np.all(np.isfinite(s)):
            raise RuntimeError("Reference trajectory produced non-finite values (NaN/Inf).")
        if max_abs_state is not None and np.max(np.abs(s)) > max_abs_state:
            raise RuntimeError(
                f"Reference trajectory escaped bound: max|state|={np.max(np.abs(s)):.6g} "
                f"> max_abs_state={max_abs_state:.6g}. The reference orbit is not on a "
                "bounded attractor; check the model/parameters before trusting the spectrum."
            )

    def run(n: int, accumulate: bool) -> np.ndarray:
        nonlocal state, tangent
        log_sums = np.zeros(n_directions)
        for interval_start in range(0, n, qr_every):
            for _ in range(qr_every):
                state, tangent = step_with_tangent_fn(state, tangent)
                check_bounded(state)
            q, r = np.linalg.qr(tangent.T)
            diag_r = np.diag(r)
            signs = np.sign(diag_r)
            signs[signs == 0.0] = 1.0
            q = q * signs[None, :]
            diag_r = diag_r * signs
            tangent = np.ascontiguousarray(q.T)
            if accumulate:
                log_sums += np.log(np.abs(diag_r))
        return log_sums

    if warmup_steps > 0:
        run(warmup_steps, accumulate=False)

    log_sums = run(n_steps, accumulate=True)
    total_time = n_steps * dt_per_step
    exponents = log_sums / total_time
    order = np.argsort(exponents)[::-1]
    exponents_sorted = exponents[order]
    n_positive = int(np.sum(exponents_sorted > 0))
    d_ky = kaplan_yorke_dimension(exponents_sorted)
    return LyapunovResult(
        exponents=exponents_sorted,
        n_directions=n_directions,
        total_time=total_time,
        kaplan_yorke_dimension=d_ky,
        n_positive=n_positive,
    )


def _ks_step_with_tangent_physical_space(cfg: KSConfig, M: int) -> StepWithTangentFn:
    op = get_operator(cfg, M=M)

    def step_fn(u_phys: np.ndarray, tangent_phys: np.ndarray):
        v = fft(u_phys)
        dV = fft(tangent_phys, axis=-1)
        v_next, dV_next = step_with_tangent(v, dV, op)
        u_next = ifft(v_next).real
        tangent_next = ifft(dV_next, axis=-1).real
        return u_next, tangent_next

    return step_fn


def lyapunov_spectrum_ks(
    cfg: KSConfig,
    *,
    n_directions: int,
    total_time: float,
    qr_interval: float,
    warmup_time: float = 0.0,
    seed: int = 0,
    initial_state: np.ndarray | None = None,
    max_abs_state: float = 50.0,
    M: int = 32,
) -> LyapunovResult:
    """Full Lyapunov spectrum of the raw KS PDE via the variational ETDRK4
    integrator (`ks_latent.solver.ks.step_with_tangent`).

    `total_time`, `qr_interval`, `warmup_time` are physical time (not steps);
    `cfg.dt_snap` is irrelevant here -- this operates at the solver's raw
    `cfg.dt`, not the snapshot stride, since the tangent map must be
    consistent with every ETDRK4 sub-step.
    """
    step_fn = _ks_step_with_tangent_physical_space(cfg, M)
    if initial_state is None:
        initial_state = spinup(cfg, np.random.default_rng(seed), M=M)

    qr_every = max(1, int(round(qr_interval / cfg.dt)))
    n_steps = (int(round(total_time / cfg.dt)) // qr_every) * qr_every
    warmup_steps = (int(round(warmup_time / cfg.dt)) // qr_every) * qr_every

    return benettin(
        step_fn,
        initial_state,
        n_directions=n_directions,
        n_steps=n_steps,
        qr_every=qr_every,
        dt_per_step=cfg.dt,
        warmup_steps=warmup_steps,
        rng=np.random.default_rng(seed + 1),
        max_abs_state=max_abs_state,
    )


def _make_single_state_step_fn(propagator: LatentPropagator) -> StepWithTangentFn:
    """"single_state" mode (brief §6.2): the "z -> z approximation used
    originally". The trained propagator is really a map on the 2d-dim
    history `(z_{n-1}, z_n)`; this mode approximates it as a self-contained
    map `g: R^d -> R^d`, `g(z) = step(z, z)` (feeding the same state into
    both history slots), which is the only way to get a well-defined
    single-state map at all out of a two-argument step function. Its
    accuracy relative to the true dynamics is exactly the open question
    the handoff flags -- that's what `two_step` mode below is for comparing
    against, not a claim that this mode is correct.
    """
    propagator = propagator.to("cpu").eval()

    def g(z: torch.Tensor) -> torch.Tensor:
        return propagator.step(z.unsqueeze(0), z.unsqueeze(0)).squeeze(0)

    def step_fn(state: np.ndarray, tangent: np.ndarray):
        z = torch.tensor(state, dtype=torch.float32)
        tangent_t = torch.tensor(tangent, dtype=torch.float32)
        z_next = g(z)

        def jvp_one(v):
            _, jv = jvp(g, (z,), (v,))
            return jv

        # attn_window/vit backbones route through nn.MultiheadAttention's
        # scaled_dot_product_attention, whose default CPU "flash" kernel has
        # no forward-AD (dual-number) support -- caught 2026-08-29 running
        # Gate 3 against the vit propagator ("Trying to use forward AD with
        # _scaled_dot_product_flash_attention_for_cpu..."). The MATH backend
        # is a plain, fully-differentiable-in-all-modes fallback; forcing it
        # here is a no-op for the mlp backbone (no attention involved).
        with sdpa_kernel(SDPBackend.MATH), _maybe_noise_disabled(propagator):
            tangent_next = vmap(jvp_one)(tangent_t)
        return z_next.detach().numpy(), tangent_next.detach().numpy()

    return step_fn


def _make_two_step_step_fn(propagator: LatentPropagator) -> StepWithTangentFn:
    """"two_step" mode (brief §6.2): the *exact* tangent map on the true
    2d-dimensional phase space `(z_{n-1}, z_n)`. The map
    `f(z_{n-1}, z_n) = (z_n, step(z_{n-1}, z_n))` is a genuine, well-defined
    dynamical system (unlike `single_state`'s self-referential proxy), so
    its Jacobian via `torch.func.jvp` is the exact tangent map, no
    approximation involved.
    """
    propagator = propagator.to("cpu").eval()
    d = propagator.cfg.d_latent

    def f(pair: torch.Tensor) -> torch.Tensor:
        z_prev, z_curr = pair[:d].unsqueeze(0), pair[d:].unsqueeze(0)
        z_next = propagator.step(z_prev, z_curr).squeeze(0)
        return torch.cat([pair[d:], z_next])

    def step_fn(state: np.ndarray, tangent: np.ndarray):
        pair = torch.tensor(state, dtype=torch.float32)
        tangent_t = torch.tensor(tangent, dtype=torch.float32)
        pair_next = f(pair)

        def jvp_one(v):
            _, jv = jvp(f, (pair,), (v,))
            return jv

        # See _make_single_state_step_fn's comment above the matching guard.
        with sdpa_kernel(SDPBackend.MATH), _maybe_noise_disabled(propagator):
            tangent_next = vmap(jvp_one)(tangent_t)
        return pair_next.detach().numpy(), tangent_next.detach().numpy()

    return step_fn


def _make_history_step_fn(propagator: LatentPropagator) -> StepWithTangentFn:
    """`mode="history"` counterpart of `_make_two_step_step_fn` (added
    2026-08-29, user-directed): the *exact* tangent map on the true
    `n_history * d_latent`-dimensional phase space `(z_{n-H+1}, ...,
    z_n)`, generalizing the two-state case to an arbitrary history length
    the same way `_ViTHistoryDeltaBody` generalizes `"two_step"`."""
    propagator = propagator.to("cpu").eval()
    d = propagator.cfg.d_latent
    n_hist = propagator.cfg.n_history

    def f(flat: torch.Tensor) -> torch.Tensor:
        z_hist = flat.view(n_hist, d).unsqueeze(0)  # (1, n_history, d)
        z_next = propagator.step_history(z_hist).squeeze(0)  # (d,)
        return torch.cat([flat[d:], z_next])

    def step_fn(state: np.ndarray, tangent: np.ndarray):
        flat = torch.tensor(state, dtype=torch.float32)
        tangent_t = torch.tensor(tangent, dtype=torch.float32)
        flat_next = f(flat)

        def jvp_one(v):
            _, jv = jvp(f, (flat,), (v,))
            return jv

        # See _make_single_state_step_fn's comment above the matching guard.
        with sdpa_kernel(SDPBackend.MATH), _maybe_noise_disabled(propagator):
            tangent_next = vmap(jvp_one)(tangent_t)
        return flat_next.detach().numpy(), tangent_next.detach().numpy()

    return step_fn


def lyapunov_spectrum_latent_propagator(
    propagator: LatentPropagator,
    initial_pair: np.ndarray,
    *,
    mode: str,
    n_directions: int,
    n_steps: int,
    qr_every: int,
    dt_snap: float,
    warmup_steps: int = 0,
    seed: int = 0,
    max_abs_state: float = 1e3,
) -> LyapunovResult:
    """Lyapunov spectrum of a trained `LatentPropagator`, in physical time
    units via `dt_snap` (brief §6.2: report lambda_1 = lambda_1_step / Delta_t,
    checked against the literature bound lambda_1 <~ 0.1 for KS -- a
    mismatch almost always means dt_snap was lost somewhere upstream).

    `initial_pair`: `(2, d_latent)` or `(2*d_latent,)`, a genuine
    `(z_{n-1}, z_n)` pair from an on-attractor encoded trajectory --
    except for `mode="history"`, where it must instead be `(n_history,
    d_latent)` or `(n_history*d_latent,)`, a genuine length-`n_history`
    history window (oldest to newest). `mode`: `"single_state"` (state is
    `z_n` alone, `R^d`), `"two_step"` (state is the full `(z_{n-1}, z_n)`
    pair, `R^{2d}`), or `"history"` (added 2026-08-29, user-directed;
    state is the full history window, `R^{n_history*d}`, only valid for a
    `propagator.mode == "history"` propagator) -- see the adapters above
    for what each approximates/requires.
    """
    d = propagator.cfg.d_latent
    if mode == "history":
        n_hist = propagator.cfg.n_history
        state0 = np.asarray(initial_pair, dtype=np.float64).reshape(-1)
        if state0.shape[0] != n_hist * d:
            raise ValueError(
                f"initial_pair must have n_history*d_latent={n_hist * d} elements for mode='history'"
            )
        step_fn = _make_history_step_fn(propagator)
    else:
        initial_pair = np.asarray(initial_pair, dtype=np.float64).reshape(-1)
        if initial_pair.shape[0] != 2 * d:
            raise ValueError(f"initial_pair must have 2*d_latent={2 * d} elements")
        if mode == "single_state":
            step_fn = _make_single_state_step_fn(propagator)
            state0 = initial_pair[d:]
        elif mode == "two_step":
            step_fn = _make_two_step_step_fn(propagator)
            state0 = initial_pair
        else:
            raise ValueError(f"mode must be 'single_state', 'two_step', or 'history', got {mode!r}")

    return benettin(
        step_fn,
        state0,
        n_directions=n_directions,
        n_steps=n_steps,
        qr_every=qr_every,
        dt_per_step=dt_snap,
        warmup_steps=warmup_steps,
        rng=np.random.default_rng(seed),
        max_abs_state=max_abs_state,
    )
