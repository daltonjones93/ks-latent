"""Task J16 (2026-10-09): offline model-error statistics for the physics filter's wrong model.

On an INDEPENDENT truth (seed 3, never assimilated), start the filter's model from the true state
(resampled to the model grid), run it for one DA cycle, and compare with the truth one cycle later:
    r = R(u(t + T)) - M_wrong(R(u(t))),     R = spectral resampling to the model grid.
Saves, on the model grid: `bias` = mean r (for --bias-correct), `spec_shape` = mean |FFT(r - bias)|^2
normalized to unit maximum (spatial correlation shape for --q-scale), `sigma` = std of r - bias.
Model errors as exp_i11_pde_letkf.py: --model-lap-coef, --model-advect-x, --model-forcing-amp,
--model-nx. Writes artifacts/experiments/da2d/j16_model_error_<out>.npz (+ .json summary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).parent))
from exp_i11_pde_letkf import spectral_resample, step_batch  # noqa: E402

from ks_latent.config import KSConfig2D  # noqa: E402
from ks_latent.solver.ks2d import get_operator2d, integrate2d, modified_operator2d  # noqa: E402
from ks_latent.solver.twin import terrain_forcing  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--NX", type=int, default=128)
    ap.add_argument("--model-nx", type=int, default=None)
    ap.add_argument("--model-lap-coef", type=float, default=1.0)
    ap.add_argument("--model-advect-x", type=float, default=0.0)
    ap.add_argument("--model-forcing-amp", type=float, default=0.0)
    ap.add_argument("--truth-cubic", type=float, default=0.0, help="J17A: truth u_t = KS(u) - beta P[u'^3]")
    ap.add_argument("--n-starts", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed in (0, 1, 2):
        raise ValueError("seeds 0-2 are the DA truths; estimate model error on an independent one")
    dt, NX = 0.02, a.NX
    M = a.model_nx or NX
    cfg = KSConfig2D(Lx=a.L, Ly=a.L, NX=NX, NY=NX, dt=dt, snapshot_every=5, spinup_time=200.0, seed=a.seed)
    mcfg = KSConfig2D(Lx=a.L, Ly=a.L, NX=M, NY=M, dt=dt, snapshot_every=5, spinup_time=200.0, seed=a.seed)
    if a.truth_cubic == 0.0:
        u0, tkw = np.load(f"artifacts/cache/ks2d_spinup_L{a.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy"), {}
    else:  # J17A: start on the new system's attractor (cached 100-unit adjustment, as the DA truths)
        from ks_latent.solver.twin import twin_truth
        u0, tkw = twin_truth(cfg, a.seed, cubic=a.truth_cubic)
    n_cyc = a.n_prop_steps * cfg.snapshot_every  # solver steps per cycle
    U = integrate2d(u0, cfg, n_steps=(a.n_starts + 1) * n_cyc, snapshot_every=n_cyc, **tkw)  # (n_starts + 2, NX, NX)
    op = modified_operator2d(mcfg, a.model_lap_coef) if a.model_lap_coef != 1.0 else get_operator2d(mcfg)
    shift = None if a.model_advect_x == 0.0 else np.exp(-1j * a.model_advect_x * op.KX * dt)
    forc = None if a.model_forcing_amp == 0.0 else np.fft.fft2(terrain_forcing(mcfg, a.model_forcing_amp))
    starts = spectral_resample(U[:a.n_starts], M)
    targets = spectral_resample(U[1:a.n_starts + 1], M)
    V = sfft.fft2(starts - starts.mean(axis=(1, 2), keepdims=True))
    for _ in range(n_cyc):
        V = step_batch(V, op, forc, shift)
    R = targets - sfft.ifft2(V).real  # (n_starts, M, M)
    bias = R.mean(0)
    Rc = R - bias
    spec = np.mean(np.abs(np.fft.fft2(Rc)) ** 2, axis=0)
    out = f"artifacts/experiments/da2d/j16_model_error_{a.out}"  # not Path.with_suffix: tags contain dots
    np.savez(out + ".npz", bias=bias, spec_shape=spec / spec.max(), sigma=float(Rc.std()))
    summ = {"args": vars(a), "bias_rms": float(np.sqrt(np.mean(bias ** 2))), "sigma": float(Rc.std()),
            "residual_rms": float(np.sqrt(np.mean(R ** 2)))}
    Path(out + ".json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
