import numpy as np
import torch

from ks_latent.da.pff_kernel import KernelPFF


def test_kernel_pff_linear_gaussian_matches_kalman():
    g = torch.Generator().manual_seed(0)
    d, m, N = 2, 2, 400
    Bt = torch.tensor([[1.0, 0.5], [0.5, 0.8]], dtype=torch.float64)
    Z0 = torch.randn(N, d, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(Bt).T
    Hm = torch.tensor([[1.0, 0.0], [0.3, 1.0]], dtype=torch.float64)
    R = torch.full((m,), 0.25, dtype=torch.float64)
    y = torch.tensor([1.0, -0.5], dtype=torch.float64)
    B = torch.cov(Z0.T)
    K = B @ Hm.T @ torch.linalg.inv(Hm @ B @ Hm.T + torch.diag(R))
    mean_k = Z0.mean(0) + K @ (y - Hm @ Z0.mean(0))
    cov_k = B - K @ Hm @ B
    f = KernelPFF(lambda Z: Z @ Hm.T, R, alpha=0.1, n_steps=2000, max_move=0.02, tol=1e-7)
    Za = f.analyze(Z0, y).Z_analysis
    assert torch.allclose(Za.mean(0), mean_k, atol=0.05)
    rel = np.abs(np.diag(torch.cov(Za.T).numpy()) / np.diag(cov_k.numpy()) - 1)
    assert rel.max() < 0.25, rel


def test_kernel_pff_no_information_keeps_ensemble():
    g = torch.Generator().manual_seed(1)
    Z0 = torch.randn(50, 3, generator=g, dtype=torch.float64)
    R = torch.full((1,), 1e12, dtype=torch.float64)
    f = KernelPFF(lambda Z: Z[:, :1], R, n_steps=50)
    Za = f.analyze(Z0, torch.zeros(1, dtype=torch.float64)).Z_analysis
    # prior term alone: the flow is stationary for the prior up to kernel/sampling effects
    assert float((Za.mean(0) - Z0.mean(0)).norm()) < 0.1
    assert abs(float(Za.std(0).mean() / Z0.std(0).mean()) - 1) < 0.25
