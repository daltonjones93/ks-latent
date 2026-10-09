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
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.fft import fft, ifft
from torch.func import jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from ks_latent.config import KolmogorovConfig, KSConfig, KSConfig2D, SWEConfig
from ks_latent.models.propagator import LatentPropagator
from ks_latent.solver.ks import get_operator, spinup, step_with_tangent
from ks_latent.solver.ks2d import get_operator2d, spinup2d, step_with_tangent2d
from ks_latent.solver.kolmogorov import (
    get_operator_kolmogorov,
    spinup_kolmogorov,
    step_with_tangent_kolmogorov,
)
from ks_latent.solver.swe import get_operator_swe, spinup_swe, step_with_tangent_swe


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
    # Running estimates at intermediate averaging times (physical time ->
    # exponents sorted descending), only when `record_steps` was requested.
    partial_exponents: dict[float, np.ndarray] | None = None


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
    checkpoint_path: str | None = None,
    checkpoint_every: int = 0,
    record_steps: list[int] | None = None,
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

    Preemption: with `checkpoint_path` and `checkpoint_every > 0` (in QR
    intervals), the full algorithm state is written atomically every
    `checkpoint_every` intervals and on completion; if the file already
    exists and matches this run's shape/length, the run resumes from it and
    reproduces an uninterrupted run: bitwise for a deterministic step
    function (see tests/unit/test_lyapunov_checkpoint.py), and to the
    step function's own run-to-run floating-point noise otherwise (the
    Kolmogorov FFT path differs by ~1e-15 between two uninterrupted runs;
    a kill -9/resume drill matched at the same level).
    `record_steps` (multiples of `qr_every`, counted from the end of warmup)
    stores running exponent estimates, so one long run also yields the
    shorter-window estimates for a convergence check.
    """
    if n_steps % qr_every != 0:
        raise ValueError(f"n_steps={n_steps} must be a multiple of qr_every={qr_every}")
    if warmup_steps % qr_every != 0:
        raise ValueError(f"warmup_steps={warmup_steps} must be a multiple of qr_every={qr_every}")
    record_steps = sorted(record_steps or [])
    for rs in record_steps:
        if rs % qr_every != 0 or not 0 < rs <= n_steps:
            raise ValueError(f"record_steps entry {rs} must be a multiple of qr_every in (0, n_steps]")
    if rng is None:
        rng = np.random.default_rng(0)

    state_dim = np.asarray(state0).shape[-1]
    q0, _ = np.linalg.qr(rng.normal(size=(state_dim, n_directions)))
    tangent = np.ascontiguousarray(q0.T)  # (n_directions, state_dim), rows orthonormal
    state = np.asarray(state0).copy()

    n_warm_intervals = warmup_steps // qr_every
    n_main_intervals = n_steps // qr_every
    fingerprint = np.array([state_dim, n_directions, qr_every, n_warm_intervals, n_main_intervals])
    intervals_done = 0  # counts warmup intervals first, then main intervals
    log_sums = np.zeros(n_directions)
    partial: dict[int, np.ndarray] = {}

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        with np.load(checkpoint_path) as ck:
            if not np.array_equal(ck["fingerprint"], fingerprint):
                raise ValueError(
                    f"Checkpoint {checkpoint_path} was written by a different run "
                    f"(fingerprint {ck['fingerprint'].tolist()} vs {fingerprint.tolist()}); "
                    "delete it or use a different path."
                )
            state = ck["state"].copy()
            tangent = np.ascontiguousarray(ck["tangent"])
            log_sums = ck["log_sums"].copy()
            intervals_done = int(ck["intervals_done"])
            for rs, ex in zip(ck["partial_steps"], ck["partial_exponents"]):
                partial[int(rs)] = ex.copy()

    def save_checkpoint():
        if checkpoint_path is None:
            return
        tmp = checkpoint_path + ".tmp.npz"
        steps = sorted(partial)
        np.savez(
            tmp,
            fingerprint=fingerprint,
            state=state,
            tangent=tangent,
            log_sums=log_sums,
            intervals_done=intervals_done,
            partial_steps=np.array(steps, dtype=np.int64),
            partial_exponents=(np.stack([partial[s] for s in steps]) if steps
                               else np.zeros((0, n_directions))),
        )
        os.replace(tmp, checkpoint_path)

    def check_bounded(s):
        if not np.all(np.isfinite(s)):
            raise RuntimeError("Reference trajectory produced non-finite values (NaN/Inf).")
        if max_abs_state is not None and np.max(np.abs(s)) > max_abs_state:
            raise RuntimeError(
                f"Reference trajectory escaped bound: max|state|={np.max(np.abs(s)):.6g} "
                f"> max_abs_state={max_abs_state:.6g}. The reference orbit is not on a "
                "bounded attractor; check the model/parameters before trusting the spectrum."
            )

    record_set = set(record_steps)
    total_intervals = n_warm_intervals + n_main_intervals
    while intervals_done < total_intervals:
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
        intervals_done += 1
        if intervals_done > n_warm_intervals:
            log_sums += np.log(np.abs(diag_r))
            main_steps = (intervals_done - n_warm_intervals) * qr_every
            if main_steps in record_set:
                partial[main_steps] = np.sort(log_sums / (main_steps * dt_per_step))[::-1]
        if checkpoint_every > 0 and intervals_done % checkpoint_every == 0:
            save_checkpoint()
    save_checkpoint()

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
        partial_exponents=(
            {s * dt_per_step: ex for s, ex in sorted(partial.items())} if record_steps else None
        ),
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


def _ks2d_step_with_tangent_physical_space(cfg: KSConfig2D, M: int) -> StepWithTangentFn:
    """2D analogue of `_ks_step_with_tangent_physical_space`. `benettin`
    is written for a flat `(state_dim,)` vector space (QR needs a plain
    2D matrix, `(n_directions, state_dim)`) -- this wrapper flattens the
    `(NX,NY)` physical field to `(NX*NY,)` and each tangent direction the
    same way at the boundary, reshaping back to `(NX,NY)` only internally
    to call `step_with_tangent2d`."""
    op = get_operator2d(cfg, M=M)
    NX, NY = cfg.NX, cfg.NY

    def step_fn(u_flat: np.ndarray, tangent_flat: np.ndarray):
        u_phys = u_flat.reshape(NX, NY)
        v = np.fft.fft2(u_phys)
        n_directions = tangent_flat.shape[0]
        tangent_phys = tangent_flat.reshape(n_directions, NX, NY)
        dV = np.fft.fft2(tangent_phys, axes=(-2, -1))
        v_next, dV_next = step_with_tangent2d(v, dV, op)
        u_next = np.fft.ifft2(v_next).real
        tangent_next = np.fft.ifft2(dV_next, axes=(-2, -1)).real
        return u_next.reshape(NX * NY), tangent_next.reshape(n_directions, NX * NY)

    return step_fn


def lyapunov_spectrum_ks2d(
    cfg: KSConfig2D,
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
    """2D analogue of `lyapunov_spectrum_ks` -- full Lyapunov spectrum of
    the 2D KS PDE via the variational ETDRK4 integrator
    (`ks_latent.solver.ks2d.step_with_tangent2d`). `initial_state`, if
    given, must be `(NX,NY)` (flattened internally); the result's own
    `exponents`/`D_KY` are in the same flat-vector convention every other
    `LyapunovResult` in this module uses, so downstream code (Kaplan-Yorke,
    `n_positive`, etc.) needs no 2D-specific handling at all."""
    step_fn = _ks2d_step_with_tangent_physical_space(cfg, M)
    if initial_state is None:
        initial_state = spinup2d(cfg, np.random.default_rng(seed), M=M)
    initial_state_flat = initial_state.reshape(cfg.NX * cfg.NY)

    qr_every = max(1, int(round(qr_interval / cfg.dt)))
    n_steps = (int(round(total_time / cfg.dt)) // qr_every) * qr_every
    warmup_steps = (int(round(warmup_time / cfg.dt)) // qr_every) * qr_every

    return benettin(
        step_fn,
        initial_state_flat,
        n_directions=n_directions,
        n_steps=n_steps,
        qr_every=qr_every,
        dt_per_step=cfg.dt,
        warmup_steps=warmup_steps,
        rng=np.random.default_rng(seed + 1),
        max_abs_state=max_abs_state,
    )


def _kolmogorov_step_with_tangent_physical_space(cfg: KolmogorovConfig, M: int) -> StepWithTangentFn:
    """Kolmogorov-flow analogue of `_ks2d_step_with_tangent_physical_space`
    -- same flatten-at-the-boundary wrapper (`benettin` needs a flat
    `(state_dim,)` vector space), calling `step_with_tangent_kolmogorov`
    internally."""
    op = get_operator_kolmogorov(cfg, M=M)
    NX, NY = cfg.NX, cfg.NY

    def step_fn(u_flat: np.ndarray, tangent_flat: np.ndarray):
        u_phys = u_flat.reshape(NX, NY)
        v = np.fft.fft2(u_phys)
        n_directions = tangent_flat.shape[0]
        tangent_phys = tangent_flat.reshape(n_directions, NX, NY)
        dV = np.fft.fft2(tangent_phys, axes=(-2, -1))
        v_next, dV_next = step_with_tangent_kolmogorov(v, dV, op)
        u_next = np.fft.ifft2(v_next).real
        tangent_next = np.fft.ifft2(dV_next, axes=(-2, -1)).real
        return u_next.reshape(NX * NY), tangent_next.reshape(n_directions, NX * NY)

    return step_fn


def lyapunov_spectrum_kolmogorov(
    cfg: KolmogorovConfig,
    *,
    n_directions: int,
    total_time: float,
    qr_interval: float,
    warmup_time: float = 0.0,
    seed: int = 0,
    initial_state: np.ndarray | None = None,
    max_abs_state: float = 200.0,
    M: int = 32,
    checkpoint_path: str | None = None,
    checkpoint_every_time: float = 0.0,
    record_times: list[float] | None = None,
) -> LyapunovResult:
    """Full Lyapunov spectrum of forced 2D Navier-Stokes (Kolmogorov flow)
    via the variational ETDRK4 integrator (`ks_latent.solver.kolmogorov.
    step_with_tangent_kolmogorov`) -- same interface and flat-vector
    convention as `lyapunov_spectrum_ks2d`, so downstream code (Kaplan-Yorke,
    `n_positive`, etc.) needs no Kolmogorov-specific handling at all.
    `max_abs_state=200.0` (vs. 2D KS's `50.0`) since vorticity magnitudes
    at moderate Re run somewhat larger than KS's own `u` scale.

    `checkpoint_path`/`checkpoint_every_time`/`record_times` (physical time,
    measured after warmup) are passed through to `benettin` for
    preemptable runs and running-average convergence checks."""
    step_fn = _kolmogorov_step_with_tangent_physical_space(cfg, M)
    if initial_state is None:
        initial_state = spinup_kolmogorov(cfg, np.random.default_rng(seed), M=M)
    initial_state_flat = initial_state.reshape(cfg.NX * cfg.NY)

    qr_every = max(1, int(round(qr_interval / cfg.dt)))
    n_steps = (int(round(total_time / cfg.dt)) // qr_every) * qr_every
    warmup_steps = (int(round(warmup_time / cfg.dt)) // qr_every) * qr_every
    record_steps = [
        (int(round(t / cfg.dt)) // qr_every) * qr_every for t in (record_times or [])
    ]
    checkpoint_every = (
        max(1, int(round(checkpoint_every_time / (qr_every * cfg.dt)))) if checkpoint_every_time > 0 else 0
    )

    return benettin(
        step_fn,
        initial_state_flat,
        n_directions=n_directions,
        n_steps=n_steps,
        qr_every=qr_every,
        dt_per_step=cfg.dt,
        warmup_steps=warmup_steps,
        rng=np.random.default_rng(seed + 1),
        max_abs_state=max_abs_state,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        record_steps=record_steps or None,
    )


def _swe_step_with_tangent_physical_space(cfg: SWEConfig, M: int) -> StepWithTangentFn:
    """SWE analogue of `_kolmogorov_step_with_tangent_physical_space` --
    same flatten-at-the-boundary wrapper (`benettin` needs a flat
    `(state_dim,)` vector space), calling `step_with_tangent_swe`
    internally. The only real difference from the single-field solvers:
    the physical state is `(3,NX,NY)` (three prognostic fields, not
    one), so the flat dimension is `3*NX*NY`, not `NX*NY`."""
    op = get_operator_swe(cfg, M=M)
    NX, NY = cfg.NX, cfg.NY

    def step_fn(q_flat: np.ndarray, tangent_flat: np.ndarray):
        q_phys = q_flat.reshape(3, NX, NY)
        q_hat = np.fft.fft2(q_phys, axes=(-2, -1))
        n_directions = tangent_flat.shape[0]
        tangent_phys = tangent_flat.reshape(n_directions, 3, NX, NY)
        dQ_hat = np.fft.fft2(tangent_phys, axes=(-2, -1))
        q_hat_next, dQ_hat_next = step_with_tangent_swe(q_hat, dQ_hat, op)
        q_next = np.fft.ifft2(q_hat_next, axes=(-2, -1)).real
        tangent_next = np.fft.ifft2(dQ_hat_next, axes=(-2, -1)).real
        return q_next.reshape(3 * NX * NY), tangent_next.reshape(n_directions, 3 * NX * NY)

    return step_fn


def lyapunov_spectrum_swe(
    cfg: SWEConfig,
    *,
    n_directions: int,
    total_time: float,
    qr_interval: float,
    warmup_time: float = 0.0,
    seed: int = 0,
    initial_state: np.ndarray | None = None,
    max_abs_state: float = 200.0,
    M: int = 32,
    checkpoint_path: str | None = None,
    checkpoint_every_time: float = 0.0,
    record_times: list[float] | None = None,
) -> LyapunovResult:
    """Full Lyapunov spectrum of the rotating shallow water equations via
    the variational ETDRK4 ("integrating factor") integrator
    (`ks_latent.solver.swe.step_with_tangent_swe`) -- same interface and
    flat-vector convention as `lyapunov_spectrum_kolmogorov`/
    `lyapunov_spectrum_ks2d`, so downstream code (Kaplan-Yorke,
    `n_positive`, etc.) needs no SWE-specific handling at all.
    `max_abs_state=200.0` is an UNTESTED default carried over from
    Kolmogorov flow's own choice, not a measured bound for this system --
    no real spectrum has been computed for SWE yet (ground rule 6:
    `SWEConfig.tau0`'s own docstring already flags that `tau0=0.1` is an
    unvalidated starting guess); revisit this default once a first real
    run's state magnitudes are known.

    `checkpoint_path`/`checkpoint_every_time`/`record_times` (physical time,
    measured after warmup) are passed through to `benettin` for
    preemptable runs and running-average convergence checks."""
    step_fn = _swe_step_with_tangent_physical_space(cfg, M)
    if initial_state is None:
        initial_state = spinup_swe(cfg, np.random.default_rng(seed), M=M)
    initial_state_flat = initial_state.reshape(3 * cfg.NX * cfg.NY)

    qr_every = max(1, int(round(qr_interval / cfg.dt)))
    n_steps = (int(round(total_time / cfg.dt)) // qr_every) * qr_every
    warmup_steps = (int(round(warmup_time / cfg.dt)) // qr_every) * qr_every
    record_steps = [
        (int(round(t / cfg.dt)) // qr_every) * qr_every for t in (record_times or [])
    ]
    checkpoint_every = (
        max(1, int(round(checkpoint_every_time / (qr_every * cfg.dt)))) if checkpoint_every_time > 0 else 0
    )

    return benettin(
        step_fn,
        initial_state_flat,
        n_directions=n_directions,
        n_steps=n_steps,
        qr_every=qr_every,
        dt_per_step=cfg.dt,
        warmup_steps=warmup_steps,
        rng=np.random.default_rng(seed + 1),
        max_abs_state=max_abs_state,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        record_steps=record_steps or None,
    )


def _make_single_state_step_fn(
    propagator: LatentPropagator, tangent_dtype: torch.dtype = torch.float32
) -> StepWithTangentFn:
    """"single_state" mode (brief §6.2): the "z -> z approximation used
    originally". The trained propagator is really a map on the 2d-dim
    history `(z_{n-1}, z_n)`; this mode approximates it as a self-contained
    map `g: R^d -> R^d`, `g(z) = step(z, z)` (feeding the same state into
    both history slots), which is the only way to get a well-defined
    single-state map at all out of a two-argument step function. Its
    accuracy relative to the true dynamics is exactly the open question
    the handoff flags -- that's what `two_step` mode below is for comparing
    against, not a claim that this mode is correct.

    `tangent_dtype` (task A2, docs/full_paper_tasks.md): the precision of the
    base step and the tangent JVPs. The default float32 matches every model
    spectrum reported so far; `torch.float64` runs a float64 deep copy of the
    propagator (the caller's module is not modified) as a precision control.
    The QR in `benettin` is float64 either way.
    """
    if tangent_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"tangent_dtype must be torch.float32 or torch.float64, got {tangent_dtype}")
    propagator = propagator.to("cpu").eval()
    if tangent_dtype == torch.float64:
        import copy

        propagator = copy.deepcopy(propagator).double().eval()

    def g(z: torch.Tensor) -> torch.Tensor:
        return propagator.step(z.unsqueeze(0), z.unsqueeze(0)).squeeze(0)

    def step_fn(state: np.ndarray, tangent: np.ndarray):
        z = torch.tensor(state, dtype=tangent_dtype)
        tangent_t = torch.tensor(tangent, dtype=tangent_dtype)
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
    tangent_dtype: torch.dtype = torch.float32,
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
        if tangent_dtype != torch.float32 and mode != "single_state":
            raise ValueError("tangent_dtype other than float32 is only implemented for mode='single_state'")
        initial_pair = np.asarray(initial_pair, dtype=np.float64).reshape(-1)
        if initial_pair.shape[0] != 2 * d:
            raise ValueError(f"initial_pair must have 2*d_latent={2 * d} elements")
        if mode == "single_state":
            step_fn = _make_single_state_step_fn(propagator, tangent_dtype)
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
