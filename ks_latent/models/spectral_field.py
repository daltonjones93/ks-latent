"""Shared rFFT transform + analytic spatial-derivative synthesis helpers for
the "spectral-latent PDE discovery" design (added 2026-09-06, see
docs/sine_transform_pde_plan.md for the full analysis and motivation).

Used by both `ks_latent.models.autoencoder_spectral_field.KSAutoencoderSpectralField`
(the `x <-> w <-> z` encoder/decoder half) and
`ks_latent.models.propagator._SpectralPDEDeltaBody` (`backbone="spectral_pde"`,
the propagator half) -- both need the exact same fixed, non-learned
`w <-> z` transform, so it lives here once rather than being duplicated.

Convention (matching `ks_latent.models.propagator.FourierIFFTBody`'s
existing real/imaginary-concat convention): a length-`N_w` real physical
field `w` has an rFFT spectrum of `N_w//2+1` complex modes; keeping only the
lowest `K` of them and representing each as (real, imaginary) gives a flat
`2*K`-dim real vector `z = concat(coeffs[:K].real, coeffs[:K].imag)`.

**Known, harmless artifact**: mode 0 (DC)'s imaginary part is mathematically
always exactly zero for the rFFT of any real signal (`X[0] = sum_n x[n]`,
real by construction) -- so index `K` of `z` (`Im(mode 0)`) is a
structurally wasted, always-zero degree of freedom for any `z` that
actually came from `encode_to_spectrum` applied to a real field (which is
the only way `z` is ever produced during real training/inference). A
`decode_from_spectrum` -> `encode_to_spectrum` round trip is an EXACT
identity on every other index, but will silently zero out whatever was in
index `K` if fed a synthetic `z` that assigned it a nonzero value (as a
naive `torch.randn` test vector would) -- not a bug, just a reminder that
"any `2*K`-vector" is a slightly larger representation than the actual
degrees of freedom a real field's truncated spectrum has. Tests/callers
constructing a synthetic `z` for round-trip checks should derive it from a
real field via `encode_to_spectrum`, not sample it directly, to avoid
tripping over this non-issue. (For even `N_w`, the same is true of the
Nyquist mode if `K` ever reaches it -- irrelevant in every intended use of
this design, where `K` is well below `N_w//2+1`.)

Originally proposed as a sine transform; changed to rFFT 2026-09-06
(user-confirmed) because a pure sine series does not close under odd-order
spatial derivatives (needed for KS's `u*u_x` term) -- see the plan doc's
§2.1. rFFT closes under every derivative order via one diagonal multiplier
`(i*k)^n`, matches KS's actual periodic boundary condition, and matches
`ks_latent/solver/ks.py`'s own ETDRK4 solver representation exactly.
"""

from __future__ import annotations

import math

import torch


def rfft_wavenumbers(n_freq: int, L: float, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Physical angular wavenumbers `k_m = 2*pi*m/L` for `m = 0..n_freq-1`,
    matching `torch.fft.rfft`'s mode ordering (mode `m` is the coefficient
    of `e^{i*k_m*x}`, same convention `ks_latent.solver.ks.wavenumbers` uses
    for the full-spectrum case, restricted to the non-negative half rFFT
    keeps)."""
    m = torch.arange(n_freq, device=device, dtype=dtype)
    return 2.0 * math.pi * m / L


def encode_to_spectrum(w: torch.Tensor, K: int) -> torch.Tensor:
    """`w`: `(..., N_w)` real physical field -> `z`: `(..., 2*K)` real,
    the lowest `K` rFFT modes' (real, imaginary) parts concatenated. Fixed,
    non-learned, exact (no truncation error beyond dropping modes `>= K`).

    Explicit `float()` upcast before `rfft` and cast back afterward (same
    bfloat16-under-autocast issue class as `FourierIFFTBody`/`SpectralConv1d`
    elsewhere in this codebase -- `torch.fft.rfft` does not support
    bfloat16)."""
    orig_dtype = w.dtype
    coeffs = torch.fft.rfft(w.float(), dim=-1)[..., :K]
    z = torch.cat([coeffs.real, coeffs.imag], dim=-1)
    return z.to(orig_dtype)


def decode_from_spectrum(z: torch.Tensor, K: int, N_w: int) -> torch.Tensor:
    """Inverse of `encode_to_spectrum`: `z`: `(..., 2*K)` -> `w_hat`:
    `(..., N_w)`, zero-padding modes `K..N_w//2` before `irfft`. Exact
    round-trip with `encode_to_spectrum` up to float precision when `K`
    covers the field's full spectrum (`K == N_w//2+1`); a genuine (exact,
    not approximate) ideal low-pass reconstruction otherwise."""
    orig_dtype = z.dtype
    z = z.float()
    real, imag = z[..., :K], z[..., K : 2 * K]
    coeffs_k = torch.complex(real, imag)
    n_freq = N_w // 2 + 1
    full = coeffs_k.new_zeros(*coeffs_k.shape[:-1], n_freq)
    full[..., :K] = coeffs_k
    w_hat = torch.fft.irfft(full, n=N_w, dim=-1)
    return w_hat.to(orig_dtype)


def synthesize_derivatives(z: torch.Tensor, K: int, N_w: int, L: float, max_order: int) -> torch.Tensor:
    """`z`: `(B, 2*K)` -> `(B, N_w, max_order+1)`: the physical-space
    fields `w, w_x, w_xx, ..., w^(max_order)`, each computed EXACTLY (to
    float precision) by multiplying `z`'s (zero-padded) rFFT coefficients
    by `(i*k)^n` before `irfft` -- `d^n/dx^n e^{ikx} = (ik)^n e^{ikx}`, so
    this is diagonal (closed-form, no discretization error) at every order,
    unlike a finite-difference or learned-convolution derivative estimate.

    Order 0 (`n=0`, multiplier `1`) is exactly `decode_from_spectrum(z, K,
    N_w)` -- i.e. `w` itself is always the first feature channel.

    Known edge case (only relevant if `K` is close to the FULL spectrum,
    not the intended small-`K`-truncation use of this design -- see
    Trefethen, "Spectral Methods in MATLAB," on differentiating at the
    Nyquist mode): for even `N_w`, the Nyquist mode's contribution to
    ODD-order derivatives is ambiguous for a real signal. Not handled
    specially here since `K` is expected to be well below `N_w//2+1` in
    every intended use of this design."""
    orig_dtype = z.dtype
    z = z.float()
    B = z.shape[0]
    real, imag = z[:, :K], z[:, K : 2 * K]
    coeffs_k = torch.complex(real, imag)
    n_freq = N_w // 2 + 1
    full = coeffs_k.new_zeros(B, n_freq)
    full[:, :K] = coeffs_k
    k = rfft_wavenumbers(n_freq, L, device=z.device, dtype=torch.float32)
    fields = []
    for n in range(max_order + 1):
        mult = (1j * k) ** n  # (n_freq,) complex
        field = torch.fft.irfft(full * mult, n=N_w, dim=-1)  # (B, N_w) real
        fields.append(field)
    out = torch.stack(fields, dim=-1)  # (B, N_w, max_order+1)
    return out.to(orig_dtype)
