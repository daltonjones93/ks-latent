"""CPU float64 linear algebra with loud failure (brief §1.2, ground rule 2).

MPS lacks `eigh` and its `cholesky`/`svd` support is float32-only and not
trustworthy for the covariance work in the DA and dimension-estimation code.
Every function here moves its input to CPU float64, computes, and casts the
result back to the caller's original device/dtype. None of them silently
swallow a numerical failure: `robust_cholesky`'s one permitted "fallback" is
re-deriving a Cholesky factor from a clipped eigendecomposition when the
input is PSD up to roundoff, which is a documented numerical technique, not
error-swallowing -- anything worse than roundoff-scale indefiniteness raises.
"""

from __future__ import annotations

import torch


def _to_cpu_f64(A: torch.Tensor) -> torch.Tensor:
    return A.to(device="cpu", dtype=torch.float64)


def _cast_back(result: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return result.to(device=like.device, dtype=like.dtype)


def condition_number(A: torch.Tensor) -> float:
    """2-norm condition number of a symmetric matrix, via CPU float64 eigh."""
    eigvals = torch.linalg.eigvalsh(_to_cpu_f64(A))
    lo = eigvals.abs().min().item()
    hi = eigvals.abs().max().item()
    if lo == 0.0:
        return float("inf")
    return hi / lo


def robust_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition, always computed on CPU in float64.

    Returns (eigvals, eigvecs) cast back to `A`'s original device and dtype.
    Raises `torch.linalg.LinAlgError` (unmodified) on genuine failure -- there
    is nothing to fall back to for `eigh` itself.
    """
    A64 = _to_cpu_f64(A)
    eigvals, eigvecs = torch.linalg.eigh(A64)
    return _cast_back(eigvals, A), _cast_back(eigvecs, A)


def robust_cholesky(
    A: torch.Tensor,
    *,
    roundoff_rtol: float = 1e-8,
) -> torch.Tensor:
    """Cholesky factor of a symmetric PSD matrix, computed on CPU in float64.

    If direct Cholesky fails because the matrix is indefinite only at the
    scale of floating-point roundoff (smallest eigenvalue more negative than
    `-roundoff_rtol * largest_eigenvalue`), reconstructs a PSD matrix by
    clipping negative eigenvalues to zero and retries. Anything worse raises
    `RuntimeError` with the condition number in the message (ground rule 2).
    """
    A64 = _to_cpu_f64(A)
    try:
        L = torch.linalg.cholesky(A64)
        return _cast_back(L, A)
    except torch._C._LinAlgError:
        pass

    eigvals, eigvecs = torch.linalg.eigh(A64)
    max_eig = eigvals.max().item()
    min_eig = eigvals.min().item()

    if min_eig < -roundoff_rtol * max(max_eig, 1e-300):
        cond = max_eig / max(abs(min_eig), 1e-300)
        raise RuntimeError(
            "robust_cholesky: matrix is indefinite beyond roundoff "
            f"(min eigenvalue {min_eig:.6g}, max eigenvalue {max_eig:.6g}, "
            f"condition number {cond:.6g}). This is not a numerical-noise "
            "case; refusing to silently clip it. Check the caller for a sign "
            "error or a genuinely singular covariance."
        )

    clipped = eigvals.clamp_min(0.0)
    A64_psd = (eigvecs * clipped) @ eigvecs.T
    # Nudge onto the PD boundary so Cholesky of an exactly-PSD (rank-deficient
    # after clipping) matrix doesn't fail a second time.
    jitter = roundoff_rtol * max(max_eig, 1e-300)
    A64_psd = A64_psd + jitter * torch.eye(A64_psd.shape[-1], dtype=torch.float64)
    try:
        L = torch.linalg.cholesky(A64_psd)
    except torch._C._LinAlgError as e:
        cond = max_eig / max(abs(min_eig), 1e-300)
        raise RuntimeError(
            "robust_cholesky: eigh-clipped fallback still failed Cholesky "
            f"(min eigenvalue {min_eig:.6g}, max eigenvalue {max_eig:.6g}, "
            f"condition number {cond:.6g})."
        ) from e
    return _cast_back(L, A)
