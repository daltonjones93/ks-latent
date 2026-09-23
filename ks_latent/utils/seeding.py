"""Deterministic seeding (brief ground rule 3, §1.3.7).

CPU runs with the same seed and config must be bitwise identical; MPS only
gets a loose-tolerance guarantee (MPS kernels are not bitwise-deterministic
across runs even with a fixed seed).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_cpu: bool = True) -> None:
    """Seed python's `random`, numpy, and every torch RNG (CPU, MPS, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cpu:
        torch.use_deterministic_algorithms(True, warn_only=True)
