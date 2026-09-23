"""Tests for on-device windowing/shift-augmentation utilities (brief §1.3.4)."""

from __future__ import annotations

import torch

from ks_latent.training.data import gather_windows, index_shuffle_batches, make_window_index, roll_batch


def test_roll_batch_2d_matches_manual_roll():
    x = torch.arange(20).reshape(4, 5).float()
    shifts = torch.tensor([0, 1, 2, 3])
    out = roll_batch(x, shifts)
    for i in range(4):
        expected = torch.roll(x[i], shifts=int(shifts[i]))
        assert torch.equal(out[i], expected)


def test_roll_batch_3d_applies_same_shift_across_window():
    x = torch.arange(2 * 3 * 5).reshape(2, 3, 5).float()
    shifts = torch.tensor([1, 2])
    out = roll_batch(x, shifts)
    for b in range(2):
        for w in range(3):
            expected = torch.roll(x[b, w], shifts=int(shifts[b]))
            assert torch.equal(out[b, w], expected)


def test_make_window_index_covers_all_valid_windows():
    n_runs, T, window = 3, 10, 4
    index = make_window_index(n_runs, T, window)
    assert index.shape == (n_runs * (T - window + 1), 2)
    assert index[:, 1].max().item() == T - window


def test_gather_windows_shapes_and_values():
    sequences = torch.arange(2 * 6 * 3).reshape(2, 6, 3).float()
    index = torch.tensor([[0, 0], [1, 2]])
    out = gather_windows(sequences, index, window=3)
    assert out.shape == (2, 3, 3)
    assert torch.equal(out[0], sequences[0, 0:3])
    assert torch.equal(out[1], sequences[1, 2:5])


def test_index_shuffle_batches_partitions_all_indices():
    n = 23
    batches = index_shuffle_batches(n, batch_size=7)
    all_idx = torch.cat(batches)
    assert sorted(all_idx.tolist()) == list(range(n))
