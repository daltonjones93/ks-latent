"""Device routing and linalg-fallback tests (brief §1.2 end, "asserts the
routing table is respected and that robust_eigh/robust_cholesky agree with a
direct CPU float64 computation to 1e-10")."""

from __future__ import annotations

import torch

from ks_latent.utils.device import (
    empty_mps_cache,
    get_compute_device,
    get_device,
    mps_memory_report,
    routing_table,
)
from ks_latent.utils.linalg import robust_cholesky, robust_eigh


def test_routing_table_solver_and_linalg_are_cpu_float64():
    for task in ["solver", "lyapunov_qr", "linalg", "dimension", "topology", "sindy"]:
        device, dtype = get_compute_device(task)
        assert device.type == "cpu", f"{task} must route to CPU"
        assert dtype == torch.float64, f"{task} must use float64"


def test_routing_table_training_resolves_to_available_device():
    device, dtype = get_compute_device("training")
    assert dtype == torch.float32
    assert device.type in ("mps", "cuda", "cpu")


def test_get_device_explicit_cpu():
    assert get_device("cpu").type == "cpu"


def test_get_device_unknown_preference_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown device preference"):
        get_device("tpu")


def test_get_device_mps_matches_availability():
    import pytest

    if torch.backends.mps.is_available():
        assert get_device("mps").type == "mps"
    else:
        with pytest.raises(RuntimeError, match="MPS requested but not available"):
            get_device("mps")


def test_mps_memory_report_and_empty_cache_are_safe_on_any_machine():
    report = mps_memory_report()
    if torch.backends.mps.is_available():
        assert report is not None
        assert "current_allocated_bytes" in report
        assert "driver_allocated_bytes" in report
    else:
        assert report is None
    empty_mps_cache()  # must not raise either way


def test_routing_table_unknown_task_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown task"):
        get_compute_device("not_a_real_task")


def test_routing_table_is_read_only_copy():
    table = routing_table()
    table["solver"] = ("mps", torch.float32)  # mutate the returned copy
    device, dtype = get_compute_device("solver")
    assert device.type == "cpu" and dtype == torch.float64  # module table unaffected


def test_robust_eigh_matches_direct_cpu_float64():
    # float64 input throughout: isolates the "agrees with a direct CPU
    # float64 computation to 1e-10" claim from the (separate, intentional)
    # precision loss of round-tripping through a float32 caller dtype.
    torch.manual_seed(0)
    A = torch.randn(20, 20, dtype=torch.float64)
    A = A @ A.T  # symmetric

    eigvals, eigvecs = robust_eigh(A)
    ref_eigvals, ref_eigvecs = torch.linalg.eigh(A)

    assert torch.allclose(eigvals, ref_eigvals, atol=1e-10)
    # Eigenvectors can differ by sign; compare via reconstruction instead.
    recon = (eigvecs * eigvals) @ eigvecs.T
    assert torch.allclose(recon, A, atol=1e-10)


def test_robust_eigh_float32_input_is_upgraded_then_downcast():
    """float32 callers get a float64-quality intermediate computation, but the
    return dtype matches what they passed in (silent-precision-change would
    be a nasty caller-facing surprise)."""
    torch.manual_seed(0)
    A = torch.randn(20, 20, dtype=torch.float32)
    A = A @ A.T
    eigvals, eigvecs = robust_eigh(A)
    assert eigvals.dtype == torch.float32
    assert eigvecs.dtype == torch.float32
    recon = (eigvecs.double() * eigvals.double()) @ eigvecs.double().T
    assert torch.allclose(recon, A.double(), atol=1e-4)  # float32 round-trip precision


def test_robust_cholesky_matches_direct_cpu_float64():
    torch.manual_seed(1)
    L_true = torch.randn(15, 15, dtype=torch.float64).tril()
    A = (L_true @ L_true.T).float()

    L = robust_cholesky(A)
    ref_L = torch.linalg.cholesky(A.double())
    assert torch.allclose(L.double(), ref_L, atol=1e-10)
    assert torch.allclose(L.double() @ L.double().T, A.double(), atol=1e-8)


def test_robust_cholesky_fallback_on_roundoff_indefinite_matrix():
    torch.manual_seed(2)
    d = 10
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    # Smallest eigenvalue is itself tiny (1e-3), so perturbing it to a small
    # negative number is a genuine roundoff-scale event, not a structural
    # change to the matrix (unlike corrupting one of the O(1-10) eigenvalues).
    eigvals = torch.linspace(1e-3, 10.0, d, dtype=torch.float64)
    A = (Q * eigvals) @ Q.T
    eigvals_bad = eigvals.clone()
    eigvals_bad[0] = -1e-10
    A_bad = (Q * eigvals_bad) @ Q.T

    L = robust_cholesky(A_bad, roundoff_rtol=1e-6)
    # The fallback clips the tiny negative eigenvalue to 0 and adds a
    # roundoff-scale jitter before re-factoring, so the reconstruction is
    # close to the original PSD matrix up to that jitter (~1e-6 * max_eig)
    # plus the ~1e-3 eigenvalue itself being zeroed out -- the test that
    # matters is that it succeeds at all instead of raising, landing near
    # the intended matrix rather than somewhere arbitrary.
    assert torch.allclose(L @ L.T, A, atol=2e-3)


def test_robust_cholesky_raises_on_genuinely_indefinite_matrix():
    import pytest

    A = torch.tensor([[1.0, 0.0], [0.0, -5.0]])
    with pytest.raises(RuntimeError, match="condition number"):
        robust_cholesky(A)
