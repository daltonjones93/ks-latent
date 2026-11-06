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


UMOVE = 12.5   # m/s, the moving grid's speed in every ensemble (namelist umove)


def mirror_env(f: np.ndarray, umove: float = UMOVE, ref=BASE_SOUNDING) -> np.ndarray:
    """Environment features of the x-mirrored member (D9). Mirroring negates the grid-relative wind,
    u - umove -> -(u - umove), so the ground-relative sounding becomes u_m = 2 umove - u; theta and qv
    are unchanged. With u = u_ref + 5 f_u: f_u,m = (2 umove - 2 u_ref) / 5 - f_u. An involution."""
    f = np.asarray(f, np.float32).copy()
    u_ref = _profiles(ref)[:, 3]
    Z = u_ref.size
    if f.size != 3 * Z:
        raise ValueError(f"{f.size} features, expected 3 x {Z} levels")
    f[2 * Z:] = (2 * umove - 2 * u_ref) / SCALES[2] - f[2 * Z:]
    return f


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


def read_manifest(path, prefix: str = "") -> dict:
    """{prefix + member: {column: float}} from an ensemble's manifest.csv (cluster/cm1/make_ensemble.py),
    e.g. ens_B's dtheta_sfc, qv_offset_gkg, shear_scale, max_rh."""
    lines = [l.strip().split(",") for l in Path(path).read_text().splitlines() if l.strip()]
    head = lines[0]
    if head[0] != "member":
        raise ValueError(f"{path}: expected a 'member' column first, got {head}")
    return {prefix + r[0]: {k: float(v) for k, v in zip(head[1:], r[1:])} for r in lines[1:]}


def parse_box(spec: str) -> list[tuple[str, float, float]]:
    """'qv_offset_gkg:-inf:-1.2+shear_scale:1.1:inf' -> [(column, lo, hi), ...]: a member is inside when
    lo <= value < hi for every column (D10)."""
    box = []
    for part in spec.split("+"):
        k, lo, hi = part.split(":")
        box.append((k, float(lo), float(hi)))
    if not box:
        raise ValueError("empty environment box")
    return box


def members_in_box(manifest: dict, box) -> list[str]:
    for k, _, _ in box:
        if any(k not in row for row in manifest.values()):
            raise ValueError(f"column {k!r} not in the manifest")
    return sorted(n for n, row in manifest.items() if all(lo <= row[k] < hi for k, lo, hi in box))
