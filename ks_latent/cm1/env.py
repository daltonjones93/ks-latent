"""Environment features for conditioning the CM1 model on the storm's environment (D11).

Columns are stored as perturbations from each member's own far-field t=0 column, which removes the base
state, i.e. exactly what distinguishes the environments of ens_B (low-level moisture, shear). The
environment is defined by a member's input sounding, so the features come from it directly:

  env = [ (theta - theta_ref) / 1 K, (qv - qv_ref) / 1 g/kg, (u - u_ref) / 5 m/s ]  on the Z model levels

with the reference the unperturbed control sounding (`base_sounding.txt`); the sounding heights are the
model level heights. Near zero for the original ensemble (theta perturbations of ~0.1 K low down), and
different across ens_B. `build_env_file` writes {member name: features} for one ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ks_latent.cm1.physics import BASE_SOUNDING

SCALES = np.array([1.0, 1.0, 5.0])   # K, g/kg, m/s


def _profiles(path) -> np.ndarray:
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    rows = np.array([[float(v) for v in l[:4]] for l in lines[1:]])   # z, theta, qv, u
    return rows


def sounding_env(path, ref=BASE_SOUNDING) -> np.ndarray:
    """(3 * Z,) float32 environment features of one member's input sounding."""
    a, r = _profiles(path), _profiles(ref)
    if a.shape != r.shape or not np.allclose(a[:, 0], r[:, 0]):
        raise ValueError(f"{path}: sounding levels differ from the reference sounding's")
    f = (a[:, 1:4] - r[:, 1:4]) / SCALES[None, :]                     # (Z, 3)
    return f.T.reshape(-1).astype(np.float32)                        # variable-major


def build_env_file(ens_dir, out_json, name_prefix: str = "") -> dict:
    """{prefix + member: features} for every member directory with a sounding."""
    out = {}
    for d in sorted(Path(ens_dir).glob("members/mem*")):
        s = d / "input_sounding"
        if s.exists():
            out[name_prefix + d.name] = sounding_env(s).tolist()
    Path(out_json).write_text(json.dumps(out))
    return out


def load_env_files(paths) -> dict:
    env = {}
    for p in paths:
        for k, v in json.loads(Path(p).read_text()).items():
            if k in env:
                raise ValueError(f"member {k} in two environment files")
            env[k] = np.asarray(v, np.float32)
    return env
