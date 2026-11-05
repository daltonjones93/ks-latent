"""CM1 column data: storage, normalization and window sampling.

Layout. One member is an array `(T, NX, V, Z)` of the transformed fields of
`scripts/cm1/cm1_io.py` (u', w, theta', qv', p' as perturbations from the
far-field t=0 column; liquid and ice condensate as sqrt(g/kg)). Models see a
column as `C = V*Z` channels, variable-major (`c = v*Z + k`).

Normalization. One scale per variable, no shift: every variable is already
an anomaly or a non-negative amount, so zero stays zero (needed for the
non-negative condensate channels, whose decoder output is constrained >= 0).

Storage. float16 (relative precision ~5e-4, far below model errors), so 31
members x 361 one-minute frames (~15 GB) fit on one large GPU; batches are
cast to float32.

Device. The data may live on a different device from the model (e.g. CPU
memory with the model on a small GPU): `windows` returns tensors on the data
device and callers move them; `full(..., device=)` moves on request.

Sampling. Random x-windows of `width` cells, never wrapping (open lateral
boundaries). A fraction `active_frac` of windows is centred on a storm column
(composite reflectivity > `active_dbz`), because most of the domain is quiet
environment. Splits are by whole member, never by frame or window.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

VARS = ("u", "w", "th", "qv", "prs", "qliq", "qice")
POSITIVE_VARS = ("qliq", "qice")


@dataclass
class CM1Member:
    """One member: `X` (T, NX, V, Z) float16, `cref` (T, NX) float32, frame times `t` (s)."""
    name: str
    X: np.ndarray
    cref: np.ndarray
    t: np.ndarray

    @property
    def dt(self) -> float:
        d = np.diff(self.t)
        if not np.allclose(d, d[0]):
            raise ValueError(f"{self.name}: non-uniform frame spacing {np.unique(d)}")
        return float(d[0])


def member_from_cm1io(run: dict, name: str) -> CM1Member:
    """From `cm1_io.load_run` / `load_grads_run` output (`X` is (T, V, Z, NX))."""
    X = np.ascontiguousarray(run["X"].transpose(0, 3, 1, 2)).astype(np.float16)
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite values in the fields")
    return CM1Member(name=name, X=X, cref=run["cref"].astype(np.float32), t=np.asarray(run["t"], float).ravel())


def save_member(m: CM1Member, path: Path) -> None:
    np.savez(path, X=m.X, cref=m.cref, t=m.t, name=m.name)


def load_member(path: Path) -> CM1Member:
    d = np.load(path)
    return CM1Member(name=str(d["name"]), X=d["X"], cref=d["cref"], t=d["t"])


@dataclass
class Normalizer:
    """Per-variable scale; `apply` maps (..., V, Z) physical -> normalized."""
    scale: np.ndarray  # (V,)
    vars: tuple = VARS

    @classmethod
    def fit(cls, members: list[CM1Member], stride: int = 4) -> "Normalizer":
        V = members[0].X.shape[2]
        ss = np.zeros(V)
        n = 0
        for m in members:
            x = m.X[::stride, ::stride].astype(np.float64)  # (t, x, V, Z)
            ss += (x**2).sum(axis=(0, 1, 3))
            n += x.shape[0] * x.shape[1] * x.shape[3]
        scale = np.sqrt(ss / n)
        if (scale <= 0).any():
            raise ValueError(f"zero scale for a variable: {dict(zip(VARS, scale))}")
        return cls(scale=scale)

    def to_dict(self) -> dict:
        return {"vars": list(self.vars), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(scale=np.asarray(d["scale"]), vars=tuple(d["vars"]))

    def channel_scale(self, n_levels: int) -> np.ndarray:
        return np.repeat(self.scale, n_levels)  # (V*Z,), variable-major


def positive_channel_mask(n_levels: int, vars_: tuple = VARS) -> torch.Tensor:
    """Boolean (V*Z,) mask of the non-negative (condensate) channels."""
    m = np.zeros((len(vars_), n_levels), bool)
    for i, v in enumerate(vars_):
        if v in POSITIVE_VARS:
            m[i] = True
    return torch.from_numpy(m.ravel())


class ColumnData:
    """Members on one device as normalized float16 tensors `(T, NX, C)`.

    `windows(B, width, n_frames)` returns `(B, n_frames, C, width)` float32
    windows (consecutive frames, same x-position), plus the members, start
    frames and x-offsets used.
    """

    def __init__(self, members: list[CM1Member], norm: Normalizer, device: torch.device,
                 active_dbz: float = 10.0):
        if not members:
            raise ValueError("no members")
        self.names = [m.name for m in members]
        dts = {m.dt for m in members}
        if len(dts) != 1:
            raise ValueError(f"members have different frame spacings {dts}")
        self.dt = dts.pop()
        T, NX, V, Z = members[0].X.shape
        self.NX, self.V, self.Z, self.C = NX, V, Z, V * Z
        self.device = device
        s = torch.tensor(norm.channel_scale(Z), dtype=torch.float32)
        self.u = []
        self.active = []  # per member: (n_active, 2) int64 (t, x) of storm columns
        for m in members:
            if m.X.shape[1:] != (NX, V, Z):
                raise ValueError(f"{m.name}: shape {m.X.shape} differs from {members[0].X.shape}")
            x = torch.from_numpy(m.X.astype(np.float32)).reshape(m.X.shape[0], NX, V * Z) / s
            self.u.append(x.to(torch.float16).to(device))
            self.active.append(torch.from_numpy(np.argwhere(m.cref > active_dbz)))
        self.T = [x.shape[0] for x in self.u]
        self.cref = [torch.from_numpy(m.cref) for m in members]

    def windows(self, B: int, width: int, n_frames: int, gen: torch.Generator, active_frac: float = 0.5):
        if width > self.NX:
            raise ValueError(f"window {width} wider than the domain {self.NX}")
        mem = torch.randint(0, len(self.u), (B,), generator=gen)
        t0 = torch.empty(B, dtype=torch.long)
        x0 = torch.empty(B, dtype=torch.long)
        use_act = torch.rand(B, generator=gen) < active_frac
        for b in range(B):
            m = int(mem[b])
            Tm = self.T[m] - n_frames + 1
            act = self.active[m]
            act = act[act[:, 0] < Tm] if use_act[b] and len(act) else act[:0]
            if len(act):
                t, xc = act[torch.randint(0, len(act), (1,), generator=gen)][0].tolist()
                x0[b] = min(max(xc - width // 2, 0), self.NX - width)
                t0[b] = t
            else:
                t0[b] = torch.randint(0, Tm, (1,), generator=gen)
                x0[b] = torch.randint(0, self.NX - width + 1, (1,), generator=gen)
        out = torch.stack([self.u[int(m)][int(t):int(t) + n_frames, int(x):int(x) + width]
                           for m, t, x in zip(mem, t0, x0)])  # (B, F, width, C)
        return out.float().permute(0, 1, 3, 2).contiguous(), mem, t0, x0

    def full(self, m: int, t: slice | int = slice(None), device: torch.device | None = None) -> torch.Tensor:
        """Whole domain of member `m`: `(T', C, NX)` float32, on `device` (default: the data's device)."""
        x = self.u[m][t]
        if x.dim() == 2:
            x = x[None]
        return x.to(device or self.device).float().permute(0, 2, 1).contiguous()


def split_members(names: list[str], n_val: int, n_test: int) -> tuple[list[str], list[str], list[str]]:
    """Last `n_test` members test, the `n_val` before them validation, the rest training."""
    if n_val + n_test >= len(names):
        raise ValueError(f"{len(names)} members cannot give {n_val} val + {n_test} test + >=1 train")
    n_tr = len(names) - n_val - n_test
    return names[:n_tr], names[n_tr:n_tr + n_val], names[n_tr + n_val:]
