"""Plan P: event detection, features, probes and scores on synthetic data with known answers."""
import numpy as np
import pytest

from ks_latent.cm1.probe import (near_active, auc, brier_skill, event_candidates, fit_logistic, fit_ridge, kmeans, onsets,
                                 predict_logistic, predict_ridge, r2, site_reduce, window_features)


def test_site_reduce():
    a = np.arange(16, dtype=float).reshape(1, 16)
    np.testing.assert_array_equal(site_reduce(a, 4), [[3, 7, 11, 15]])
    np.testing.assert_array_equal(site_reduce(a, 4, "min"), [[0, 4, 8, 12]])


def test_event_candidates_and_onsets():
    s = np.zeros((20, 2))
    s[12:, 0] = 1.0                                    # site 0: onset at t = 12; site 1: never
    s[0:3, 1] = 1.0                                    # site 1 active early, then quiet
    ts, ps, ys = event_candidates(s, 0.5, tau=5, quiet=4)
    lab = {(t, p): y for t, p, y in zip(ts, ps, ys)}
    assert lab[(7, 0)] and lab[(11, 0)]                # onset within (t, t + 5]
    assert not lab[(6, 0)]                             # 12 is outside (6, 11]
    assert (12, 0) not in lab                          # active now: never a candidate
    assert (5, 1) not in lab and lab[(6, 1)] == 0      # site 1 needs 4 quiet frames (3..6) first
    assert onsets(s, 0.5, quiet=4) == [(12, 0)]
    d = np.full((10, 1), 1.0); d[6:] = -3.0            # a drop below the threshold (cold-pool arrival)
    ts, ps, ys = event_candidates(d, -2.0, tau=3, quiet=2, above=False)
    assert ys[list(ts).index(4)] and not ys[list(ts).index(2)]


def test_window_features_edges():
    F = np.arange(5 * 4 * 1, dtype=float).reshape(5, 4, 1)
    w = window_features(F, np.array([0, 2]), np.array([0, 3]), half=1)
    np.testing.assert_array_equal(w[0], [F[0, 0, 0], F[0, 0, 0], F[0, 1, 0]])   # left edge repeated
    np.testing.assert_array_equal(w[1], [F[2, 2, 0], F[2, 3, 0], F[2, 3, 0]])   # right edge repeated


def test_auc_brier_and_probes():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5           # ties
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=2000)) > 0
    m = fit_logistic(X, y)
    p = predict_logistic(m, X)
    assert auc(y, p) > 0.95 and brier_skill(y, p, y.mean()) > 0.5
    assert abs(m["w"][0] / m["w"][1] - 2.0) < 0.5                       # recovers the direction
    t = 3 * X[:, 2] - X[:, 4] + 0.1 * rng.normal(size=2000)
    assert r2(t, predict_ridge(fit_ridge(X, t, l2=1e-6), X)) > 0.99


def test_kmeans_recovers_clusters():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(c, 0.1, size=(100, 2)) for c in ((0, 0), (3, 0), (0, 3))])
    C, lab = kmeans(X, 3)
    assert len(set(lab[:100])) == 1 and len(set(lab[100:200])) == 1 and len(set(lab)) == 3


def test_near_active():
    s = np.zeros((2, 10)); s[1, 5] = 1.0
    n = near_active(s, 0.5, half=2)
    assert not n[0].any() and n[1].tolist() == [False, False, False, True, True, True, True, True, False, False]


def test_mlp_probe_reads_what_a_linear_probe_cannot():
    from ks_latent.cm1.probe import fit_mlp, predict_mlp
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 4)); Xt = rng.normal(size=(2000, 4))
    y, yt = X[:, 0] * X[:, 1], Xt[:, 0] * Xt[:, 1]                        # no linear signal at all
    assert r2(yt, predict_ridge(fit_ridge(X, y), Xt)) < 0.05
    assert r2(yt, predict_mlp(fit_mlp(X, y, "reg"), Xt)) > 0.8
    c, ct = y > 0, yt > 0
    assert auc(ct, predict_logistic(fit_logistic(X, c), Xt)) < 0.6
    p = predict_mlp(fit_mlp(X, c, "clf"), Xt)
    assert auc(ct, p) > 0.9 and p.min() >= 0 and p.max() <= 1
    m1, m2 = fit_mlp(X, y, "reg", epochs=3), fit_mlp(X, y, "reg", epochs=3)            # seeded: reproducible
    np.testing.assert_array_equal(predict_mlp(m1, Xt), predict_mlp(m2, Xt))
