"""G2b: the local analog forecast model (ks_latent/cm1/analog.py)."""
import pytest
import torch

from ks_latent.cm1.analog import AnalogConfig, AnalogPropagator, _patches

pytestmark = pytest.mark.unit


def test_patches_replicate_edges():
    z = torch.arange(5.0).reshape(1, 5, 1)
    p = _patches(z, 1)
    assert p.shape == (1, 5, 3)
    assert p[0, 0].tolist() == [0.0, 0.0, 1.0] and p[0, 4].tolist() == [3.0, 4.0, 4.0] and p[0, 2].tolist() == [1, 2, 3]


def test_recovers_the_archived_step_exactly():
    """A state that is in the archive: with k = 1 its own patch is the nearest, so one analog step reproduces
    the next archived frame exactly (every site, edges included). Holds with and without the PCA."""
    g = torch.Generator().manual_seed(0)
    traj = [torch.randn(12, 9, 3, generator=g), torch.randn(10, 9, 3, generator=g)]
    for pca in (0, 6):
        an = AnalogPropagator(traj, AnalogConfig(radius=1, k=1, frame_stride=1, pca_dim=pca, stochastic=False))
        for z, t in ((traj[0], 4), (traj[1], 7)):
            torch.testing.assert_close(an(z[t][None]), z[t + 1][None], atol=1e-5, rtol=0)


def test_mean_and_drawn_increments():
    g = torch.Generator().manual_seed(1)
    traj = [torch.randn(30, 6, 2, generator=g)]
    z = traj[0][5][None].repeat(4, 1, 1)
    mean = AnalogPropagator(traj, AnalogConfig(radius=0, k=5, frame_stride=1, pca_dim=0, stochastic=False))
    drawn = AnalogPropagator(traj, AnalogConfig(radius=0, k=5, frame_stride=1, pca_dim=0, stochastic=True))
    m = mean(z)
    assert torch.allclose(m[0], m[3])                                    # deterministic: members stay equal
    a = drawn(z, gen=torch.Generator().manual_seed(3))
    b = drawn(z, gen=torch.Generator().manual_seed(3))
    torch.testing.assert_close(a, b)                                     # reproducible with the generator
    assert not torch.allclose(a[0], a[1]) or not torch.allclose(a[2], a[3])   # members differ
    # the mean step equals the average of the k drawn candidates' increments
    idx = mean.neighbours((z[0] - mean.mu) @ mean.proj)                  # (P, k)
    torch.testing.assert_close(m[0] - z[0], mean.vals[idx].mean(1))
    with pytest.raises(ValueError):
        AnalogPropagator([torch.randn(3, 2, 2)], AnalogConfig(k=50, frame_stride=1, pca_dim=0))
