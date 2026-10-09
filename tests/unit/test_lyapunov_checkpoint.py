"""Preemption support in `benettin` (T1, docs/ks_local_field_ltransfer.tex
Sec. transfer_diagnostic): a run killed mid-way and resumed from its
checkpoint must give bitwise the same spectrum as an uninterrupted run, and
`record_steps` running estimates must equal shorter standalone runs."""

from __future__ import annotations

import numpy as np
import pytest

from ks_latent.analysis.lyapunov import benettin


class _Killed(Exception):
    pass


def _map_fns(dim=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) / np.sqrt(dim)

    def step(state, tangent):
        # mildly nonlinear so state and tangent both matter
        new_state = A @ state + 0.1 * np.tanh(state)
        J = A + 0.1 * np.diag(1.0 / np.cosh(state) ** 2)
        return new_state, tangent @ J.T

    return step, rng.normal(size=dim)


def _run(step, s0, **kw):
    return benettin(step, s0, n_directions=4, n_steps=60, qr_every=3, dt_per_step=0.1,
                    warmup_steps=12, rng=np.random.default_rng(7), **kw)


def test_resume_after_kill_is_bitwise_identical(tmp_path):
    step, s0 = _map_fns()
    reference = _run(step, s0)

    ckpt = str(tmp_path / "run.npz")
    calls = {"n": 0}

    def dying_step(state, tangent):
        calls["n"] += 1
        if calls["n"] == 41:  # mid main phase, not on a checkpoint boundary
            raise _Killed
        return step(state, tangent)

    with pytest.raises(_Killed):
        _run(dying_step, s0, checkpoint_path=ckpt, checkpoint_every=2)

    resumed = _run(step, s0, checkpoint_path=ckpt, checkpoint_every=2)
    np.testing.assert_array_equal(resumed.exponents, reference.exponents)
    assert resumed.kaplan_yorke_dimension == reference.kaplan_yorke_dimension


def test_completed_checkpoint_returns_same_result_without_stepping(tmp_path):
    step, s0 = _map_fns()
    ckpt = str(tmp_path / "run.npz")
    first = _run(step, s0, checkpoint_path=ckpt, checkpoint_every=5)

    def must_not_step(state, tangent):
        raise AssertionError("resumed completed run should not step")

    again = _run(must_not_step, s0, checkpoint_path=ckpt, checkpoint_every=5)
    np.testing.assert_array_equal(again.exponents, first.exponents)


def test_mismatched_checkpoint_rejected(tmp_path):
    step, s0 = _map_fns()
    ckpt = str(tmp_path / "run.npz")
    _run(step, s0, checkpoint_path=ckpt, checkpoint_every=5)
    with pytest.raises(ValueError, match="different run"):
        benettin(step, s0, n_directions=3, n_steps=60, qr_every=3, dt_per_step=0.1,
                 checkpoint_path=ckpt, checkpoint_every=5)


def test_record_steps_match_shorter_runs():
    step, s0 = _map_fns()
    long_run = _run(step, s0, record_steps=[30, 60])
    short = benettin(step, s0, n_directions=4, n_steps=30, qr_every=3, dt_per_step=0.1,
                     warmup_steps=12, rng=np.random.default_rng(7))
    np.testing.assert_allclose(long_run.partial_exponents[3.0], short.exponents, rtol=0, atol=1e-12)
    np.testing.assert_allclose(long_run.partial_exponents[6.0], long_run.exponents, rtol=0, atol=1e-12)


def test_resume_preserves_partial_records(tmp_path):
    step, s0 = _map_fns()
    reference = _run(step, s0, record_steps=[15, 30, 60])
    ckpt = str(tmp_path / "run.npz")
    calls = {"n": 0}

    def dying_step(state, tangent):
        calls["n"] += 1
        if calls["n"] == 55:  # after the t=1.5 record, before t=3.0
            raise _Killed
        return step(state, tangent)

    with pytest.raises(_Killed):
        _run(dying_step, s0, checkpoint_path=ckpt, checkpoint_every=1, record_steps=[15, 30, 60])
    resumed = _run(step, s0, checkpoint_path=ckpt, checkpoint_every=1, record_steps=[15, 30, 60])
    for t, ex in reference.partial_exponents.items():
        np.testing.assert_array_equal(resumed.partial_exponents[t], ex)
