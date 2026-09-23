#!/usr/bin/env python
"""Verify every import ks_latent needs and print found/version/required (brief §1.1).

Run first, before anything else: `mamba run -n da_env python scripts/check_env.py`.
"""

from __future__ import annotations

import importlib
import sys

# (import name, distribution name, minimum version or None)
REQUIRED = [
    ("torch", "torch", None),
    ("scipy", "scipy", None),
    ("numpy", "numpy", None),
    ("matplotlib", "matplotlib", None),
    ("gudhi", "gudhi", None),
    ("gtda", "giotto-tda", None),
    ("ripser", "ripser", None),
    ("spyder_kernels", "spyder-kernels", None),
    ("pytest", "pytest", None),
    ("pytest_cov", "pytest-cov", None),
    ("hypothesis", "hypothesis", None),
    ("yaml", "pyyaml", None),
    ("h5py", "h5py", None),
    ("pandas", "pandas", None),
    ("sklearn", "scikit-learn", None),
    ("pysindy", "pysindy", None),
    ("pywt", "PyWavelets", None),
    ("persim", "persim", None),
    ("mpmath", "mpmath", None),
]


def main() -> int:
    rows = []
    all_ok = True
    for import_name, dist_name, required_version in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            rows.append((dist_name, "OK", version, required_version or "-"))
        except Exception as e:  # noqa: BLE001 -- this script's whole job is to report failures
            all_ok = False
            rows.append((dist_name, "MISSING", str(e), required_version or "-"))

    name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'package':<{name_w}}{'found':<10}{'version':<20}{'required':<10}")
    for dist_name, status, version, required_version in rows:
        print(f"{dist_name:<{name_w}}{status:<10}{version:<20}{required_version:<10}")

    try:
        import torch

        print(f"\ntorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        pass

    if not all_ok:
        print(
            "\nSome packages are missing. Install with:\n"
            "  mamba install -n da_env -c conda-forge <pkg>   # prefer conda-forge\n"
            "  pip install <pkg>                              # only if unavailable there"
        )
        return 1
    print("\nAll required packages found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
