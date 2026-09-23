"""Device routing policy (brief §1.2-1.3).

MPS is float32-only and has real operator gaps (no `eigh`, partial `torch.fft`
support, immature `torch.compile`). The rule of thumb: anything precision- or
correctness-critical (the solver, Lyapunov QR, covariance/eigendecompositions)
runs on CPU in float64; training and inference, where speed matters more than
the last few bits of precision, run on MPS in float32.

Callers should almost never call `get_device()` directly for a specific
numerical task -- use `get_compute_device(task)` so the routing table lives
in exactly one place.
"""

from __future__ import annotations

import ks_latent  # noqa: F401  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch import)
import torch

# task -> (device_str, dtype)
_ROUTING_TABLE: dict[str, tuple[str, torch.dtype]] = {
    "solver": ("cpu", torch.float64),
    "training": ("auto", torch.float32),
    "inference": ("auto", torch.float32),
    "lyapunov_qr": ("cpu", torch.float64),
    "lyapunov_jvp": ("auto", torch.float32),
    "linalg": ("cpu", torch.float64),
    "jacobian": ("cpu", torch.float32),
    "dimension": ("cpu", torch.float64),
    "topology": ("cpu", torch.float64),
    "sindy": ("cpu", torch.float64),
}


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a device preference to a concrete `torch.device`.

    `prefer="auto"` resolves mps -> cuda -> cpu. Any other string is passed
    through after checking availability, raising if unavailable rather than
    silently falling back (ground rule 2: fail loudly).
    """
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device("mps")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {prefer!r}")


def get_compute_device(task: str) -> tuple[torch.device, torch.dtype]:
    """Return the (device, dtype) the routing table assigns to `task`.

    Example: `device, dtype = get_compute_device("lyapunov_qr")`.
    """
    if task not in _ROUTING_TABLE:
        raise ValueError(
            f"Unknown task {task!r}. Known tasks: {sorted(_ROUTING_TABLE)}. "
            "Add new tasks explicitly to _ROUTING_TABLE rather than guessing a device."
        )
    device_str, dtype = _ROUTING_TABLE[task]
    device = get_device("auto") if device_str == "auto" else get_device(device_str)
    return device, dtype


def routing_table() -> dict[str, tuple[str, torch.dtype]]:
    """Read-only view of the routing table, for tests and reporting."""
    return dict(_ROUTING_TABLE)


def mps_memory_report() -> dict[str, int] | None:
    """Current/driver MPS allocated-memory snapshot, or None off-MPS."""
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }


def empty_mps_cache() -> None:
    """`torch.mps.empty_cache()`, guarded for non-MPS machines (brief §1.3.9)."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
