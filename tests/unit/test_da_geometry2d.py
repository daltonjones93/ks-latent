import h5py
import numpy as np
import pytest

from ks_latent.da.geometry2d import deployment_grid_2d, training_dx


def _write_meta(path, Lx, Ly, NX, NY):
    with h5py.File(path, "w") as f:
        g = f.create_group("metadata")
        g.attrs["Lx"], g.attrs["Ly"], g.attrs["NX"], g.attrs["NY"] = Lx, Ly, NX, NY


def test_training_dx_reads_metadata(tmp_path):
    p = tmp_path / "d.h5"
    _write_meta(p, 264.0, 264.0, 768, 768)
    assert training_dx(str(p)) == pytest.approx(264.0 / 768)


def test_training_dx_rejects_anisotropic_grid(tmp_path):
    p = tmp_path / "d.h5"
    _write_meta(p, 264.0, 132.0, 768, 768)
    with pytest.raises(ValueError):
        training_dx(str(p))


@pytest.mark.parametrize("L,n_sites", [(44.0, 8), (220.0, 40), (330.0, 60), (440.0, 80)])
def test_deployment_grid_keeps_training_spacing(L, n_sites):
    dx = 264.0 / 768
    g = deployment_grid_2d(L, dx, patch=16)
    assert g.n_sites == n_sites
    assert g.NX == 16 * n_sites
    assert g.L / g.NX == pytest.approx(dx)
    assert g.site_length == pytest.approx(5.5)


def test_deployment_grid_rejects_fractional_sites():
    with pytest.raises(ValueError):
        deployment_grid_2d(50.0, 264.0 / 768, patch=16)


def test_old_bug_geometry_is_not_the_training_spacing():
    # The 2026-10-06 DA runs used L=44 with the crop checkpoint's 40 sites:
    # site length 1.1, i.e. dx 5x finer than training. The helper refuses
    # to produce that grid for L=44.
    g = deployment_grid_2d(44.0, 264.0 / 768, patch=16)
    assert g.n_sites != 40
    assert np.isclose(44.0 / 40, 1.1) and not np.isclose(g.site_length, 1.1)
