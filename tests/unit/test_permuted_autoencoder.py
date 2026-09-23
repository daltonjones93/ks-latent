"""Tests for `ks_latent/models/permuted_autoencoder.py` (added 2026-08-31,
user-directed) -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 20."""

from __future__ import annotations

import numpy as np
import torch

from ks_latent.config import MLPAutoencoderConfig
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation


def _make_ae(d_latent=8):
    cfg = MLPAutoencoderConfig(NX=32, hidden=(16,), d_latent=d_latent)
    return KSAutoencoderMLP(cfg), cfg


def test_encode_reindexes_by_permutation():
    ae, _ = _make_ae()
    ae.eval()
    perm = np.random.default_rng(0).permutation(8)
    wrapped = PermutedAutoencoder(ae, perm)
    u = torch.randn(4, 32)
    with torch.no_grad():
        z_orig = ae.encode(u)
        z_wrapped = wrapped.encode(u)
    assert torch.allclose(z_wrapped, z_orig[..., perm])


def test_decode_round_trip_is_exact():
    """Permuting is a free symmetry: reconstruction must be bit-for-bit
    identical to the unwrapped autoencoder's, since decode(encode(u)) is
    unchanged in effect, only relabeled."""
    ae, _ = _make_ae()
    ae.eval()
    perm = np.random.default_rng(1).permutation(8)
    wrapped = PermutedAutoencoder(ae, perm)
    u = torch.randn(4, 32)
    with torch.no_grad():
        u_hat_orig = ae.decode(ae.encode(u))
        u_hat_wrapped = wrapped.decode(wrapped.encode(u))
    assert torch.allclose(u_hat_wrapped, u_hat_orig, atol=1e-6)


def test_identity_permutation_is_a_no_op():
    ae, _ = _make_ae()
    ae.eval()
    identity = np.arange(8)
    wrapped = PermutedAutoencoder(ae, identity)
    u = torch.randn(3, 32)
    with torch.no_grad():
        z_orig = ae.encode(u)
        z_wrapped = wrapped.encode(u)
    assert torch.equal(z_wrapped, z_orig)


def test_cfg_proxy():
    ae, cfg = _make_ae()
    wrapped = PermutedAutoencoder(ae, np.arange(8))
    assert wrapped.cfg is cfg
    assert wrapped.cfg.d_latent == 8
    assert wrapped.cfg.NX == 32


def test_parameters_are_shared_not_copied():
    """No weights are touched -- this must be a pure wrapper."""
    ae, _ = _make_ae()
    wrapped = PermutedAutoencoder(ae, np.arange(8))
    assert sum(p.numel() for p in wrapped.parameters()) == sum(p.numel() for p in ae.parameters())
    for p_ae, p_wrapped in zip(ae.parameters(), wrapped.parameters()):
        assert p_ae is p_wrapped


def test_load_latent_permutation_from_npz(tmp_path):
    perm = np.array([3, 1, 0, 2])
    path = tmp_path / "diagnostics_arrays_test.npz"
    np.savez(path, d3_permutation=perm, other_key=np.zeros(2))
    loaded = load_latent_permutation(str(path))
    assert np.array_equal(loaded, perm)


def test_load_latent_permutation_from_npz_d6_key(tmp_path):
    """User-directed 2026-08-31: the raw-encoded-data (no propagator
    needed) temporal-coherence permutation, `d6_permutation`, must be
    selectable via the `key` argument -- see Section 21."""
    d3_perm = np.array([3, 1, 0, 2])
    d6_perm = np.array([1, 3, 2, 0])
    path = tmp_path / "diagnostics_arrays_test.npz"
    np.savez(path, d3_permutation=d3_perm, d6_permutation=d6_perm)
    assert np.array_equal(load_latent_permutation(str(path), key="d3_permutation"), d3_perm)
    assert np.array_equal(load_latent_permutation(str(path), key="d6_permutation"), d6_perm)


def test_load_latent_permutation_from_npy(tmp_path):
    perm = np.array([2, 0, 1])
    path = tmp_path / "perm.npy"
    np.save(path, perm)
    loaded = load_latent_permutation(str(path))
    assert np.array_equal(loaded, perm)


def test_double_permutation_inverts():
    """Applying inverse_permutation to a permuted vector recovers the
    original order -- sanity-checks argsort-as-inverse is correct."""
    ae, _ = _make_ae()
    perm = np.random.default_rng(2).permutation(8)
    wrapped = PermutedAutoencoder(ae, perm)
    z = torch.arange(8).float()
    z_permuted = z[..., wrapped.permutation]
    z_recovered = z_permuted[..., wrapped.inverse_permutation]
    assert torch.equal(z_recovered, z)
