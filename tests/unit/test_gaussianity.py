import numpy as np

from ks_latent.analysis.gaussianity import mardia, mc_pvalues


def test_mardia_gaussian_expectations():
    rng = np.random.default_rng(0)
    p, N = 4, 20000
    b1, b2 = mardia(rng.standard_normal((N, p)) @ rng.standard_normal((p, p)))
    assert abs(b2 - p * (p + 2)) < 0.5 and b1 < 0.05


def test_mc_pvalues_reject_skewed_and_accept_gaussian():
    rng = np.random.default_rng(1)
    N, p = 128, 6
    X = rng.standard_normal((N, p))
    pg = mc_pvalues(mardia(X), N, p, 300, rng)
    Y = rng.exponential(size=(N, p))
    pe = mc_pvalues(mardia(Y), N, p, 300, rng)
    assert min(pg) > 0.01
    assert pe[0] < 0.01 and pe[1] < 0.05


def test_mc_pvalues_uniform_under_null():
    rng = np.random.default_rng(2)
    ps = [mc_pvalues(mardia(rng.standard_normal((64, 4))), 64, 4, 99, rng)[0] for _ in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10
