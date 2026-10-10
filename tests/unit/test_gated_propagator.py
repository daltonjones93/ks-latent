"""Tests for ks_latent/da/gated.py (task I31)."""
import pytest
import torch

from ks_latent.da.gated import GatedPropagator, calibrate_gate, site_roundtrip_distance

N_SITES, C = 4, 3


class _Shift:
    def __init__(self, s):
        self.s = s

    def step(self, z_prev, z_curr):
        return z_curr + self.s


def _proj():
    """A toy autoencoder whose manifold is {channel 2 of every site = 0}: decode drops channel 2."""
    def encode(u):
        return u

    def decode(z):
        z = z.clone().reshape(z.shape[0], N_SITES, C)
        z[..., 2] = 0
        return z.reshape(z.shape[0], -1)
    return encode, decode


def test_roundtrip_distance_is_per_site_offmanifold_size():
    enc, dec = _proj()
    z = torch.zeros(2, N_SITES * C)
    z.reshape(2, N_SITES, C)[1, 3, 2] = 0.6  # member 1, site 3 off the manifold
    d = site_roundtrip_distance(z, enc, dec, N_SITES)
    expected = torch.zeros(2, N_SITES); expected[1, 3] = 0.6 / C ** 0.5
    torch.testing.assert_close(d, expected)


def test_gate_routes_each_site_separately():
    enc, dec = _proj()
    gp = GatedPropagator(_Shift(1.0), _Shift(-1.0), enc, dec, N_SITES, d0=0.1, width=1e-3)
    z = torch.zeros(1, N_SITES * C)
    z.reshape(1, N_SITES, C)[0, 2, 2] = 1.0  # only site 2 off-manifold
    out = gp.step(z, z).reshape(1, N_SITES, C)
    on = z.reshape(1, N_SITES, C) + 1.0; off = z.reshape(1, N_SITES, C) - 1.0
    for s in range(N_SITES):
        torch.testing.assert_close(out[0, s], (off if s == 2 else on)[0, s])
    assert abs(gp.last_mean_gate - 0.75) < 1e-6


def test_projection_expert_when_off_is_none():
    enc, dec = _proj()
    gp = GatedPropagator(_Shift(0.0), None, enc, dec, N_SITES, d0=0.1, width=1e-3)
    z = torch.ones(1, N_SITES * C)  # every site off-manifold (channel 2 = 1)
    out = gp.step(z, z)
    torch.testing.assert_close(out, dec(z))  # projected, then identity propagator


def test_calibrate_gate_and_validation():
    enc, dec = _proj()
    z = torch.randn(500, N_SITES * C) * 0.1
    d0, w = calibrate_gate(z, enc, dec, N_SITES, quantile=0.9)
    d = site_roundtrip_distance(z, enc, dec, N_SITES).reshape(-1)
    assert abs(float((d <= d0).float().mean()) - 0.9) < 0.02 and w > 0
    with pytest.raises(ValueError):
        GatedPropagator(_Shift(0.0), None, enc, dec, N_SITES, d0=0.1, width=0.0)
