"""Gate 1 (brief §3.2, §18, §21): the full-PDE Kaplan-Yorke dimension.

"test_L100_kaplan_yorke ... This is the gate on Phase 1." Do not proceed to
later phases if this fails -- write it up in docs/REPLICATION_LOG.md and
stop (ground rule 6).
"""

from __future__ import annotations

import pytest

from ks_latent.analysis.lyapunov import lyapunov_spectrum_ks
from ks_latent.config import KSConfig
from ks_latent.utils.replication import record_result


@pytest.mark.slow
def test_L22_lyapunov():
    """L=22: lambda_1 in [0.043, 0.05], D_KY in [5.2, 5.6] (brief §3.2)."""
    cfg = KSConfig(L=22.0, NX=128, dt=0.05, snapshot_every=5, spinup_time=200.0)
    result = lyapunov_spectrum_ks(
        cfg,
        n_directions=15,
        total_time=4000.0,
        qr_interval=1.0,
        warmup_time=400.0,
        seed=0,
        max_abs_state=50.0,
    )
    lambda1 = result.exponents[0]
    d_ky = result.kaplan_yorke_dimension

    lambda1_ok = 0.043 <= lambda1 <= 0.05
    d_ky_ok = 5.2 <= d_ky <= 5.6

    record_result(
        "L22_lambda1", lambda1, "[0.043, 0.05]", "must fall in range", lambda1_ok, "1/time"
    )
    record_result("L22_D_KY", d_ky, "[5.2, 5.6]", "must fall in range", d_ky_ok)

    assert lambda1_ok, f"lambda_1={lambda1:.4f} not in [0.043, 0.05]"
    assert d_ky_ok, f"D_KY={d_ky:.4f} not in [5.2, 5.6]"


@pytest.mark.slow
def test_L100_kaplan_yorke():
    """L=100: D_KY in [21, 24] (brief table; Edson et al. / Koopman-paper bound).

    **Gate 1: do not proceed to Phase 2 if this fails.**
    """
    cfg = KSConfig(L=100.0, NX=1024, dt=0.05, snapshot_every=5, spinup_time=500.0)
    result = lyapunov_spectrum_ks(
        cfg,
        n_directions=35,
        total_time=2000.0,
        qr_interval=1.0,
        warmup_time=300.0,
        seed=0,
        max_abs_state=50.0,
    )
    d_ky = result.kaplan_yorke_dimension
    d_ky_ok = 21.0 <= d_ky <= 24.0

    record_result("L100_D_KY", d_ky, "[21, 24]", "must fall in range", d_ky_ok)
    record_result(
        "L100_lambda1_physical",
        result.exponents[0],
        "<= 0.1",
        "literature bound",
        result.exponents[0] <= 0.1,
        "1/time",
    )
    record_result(
        "L100_n_positive_exponents", result.n_positive, "~11", "qualitative", True
    )

    assert d_ky_ok, f"GATE 1 FAILED: D_KY={d_ky:.4f} not in [21, 24]. Do not proceed to Phase 2."
