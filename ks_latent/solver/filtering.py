"""Gaussian low-pass filtering in Fourier space (brief §3.3, used by Phases 8, 10)."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft

from ks_latent.solver.ks import wavenumbers


def gaussian_lowpass(u: np.ndarray, L: float, ell: float) -> np.ndarray:
    """Apply a Gaussian low-pass filter G_ell * u with width `ell` (physical units).

    `u` has shape `(..., NX)`. The filter is `exp(-0.5 * (k*ell)**2)` in
    Fourier space, applied along the last axis.
    """
    NX = u.shape[-1]
    k = wavenumbers(L, NX)
    kernel = np.exp(-0.5 * (k * ell) ** 2)
    return ifft(kernel * fft(u, axis=-1), axis=-1).real
