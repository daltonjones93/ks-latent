"""Readers for CM1 output (NetCDF and GrADS) and the column transform.

`transform` builds the `(T, V, Z, NX)` state of `data.VARS` from raw CM1
fields: u, theta, qv and p become perturbations from the far-field t=0
column (x index `BASE_I`); w is kept; liquid (qc+qr) and ice (qi+qs+qg)
condensate become sqrt(g/kg). GrADS output (`output_format=1`) is
direct-access little-endian float32: for each time, each variable in `.ctl`
order, `nlev` levels (0 = a 2D field) of `nx*ny` values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BASE_I = 1500  # far-field column at t=0 (the bubble is at x = 200 km, radius 10 km)


def transform(g) -> np.ndarray:
    """`g(name) -> (T, Z, NX)` float64 raw field; returns `(T, V, Z, NX)` float32."""
    th, qv, prs = g("th"), g("qv"), g("prs")
    base = lambda a: a[0, :, BASE_I][None, :, None]
    qliq = g("qc") + g("qr")
    qice = g("qi") + g("qs") + g("qg")
    u = g("uinterp")
    X = np.stack(
        [
            u - base(u),
            g("winterp"),
            th - base(th),
            (qv - base(qv)) * 1e3,          # g/kg
            (prs - base(prs)) / 100.0,      # hPa
            np.sqrt(np.clip(qliq, 0, None) * 1e3),
            np.sqrt(np.clip(qice, 0, None) * 1e3),
        ],
        axis=1,
    ).astype(np.float32)
    return X


def parse_ctl(ctl: Path) -> dict:
    lines = Path(ctl).read_text().splitlines()
    info, vars_, i = {}, [], 0
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1
            continue
        key = tok[0].lower()
        if key == "dset":
            info["dset"] = tok[1].replace("^", "")
        elif key in ("xdef", "ydef"):
            info[key] = int(tok[1])
        elif key == "zdef":
            info["zdef"] = int(tok[1])
            if tok[2] == "levels":
                i += info["zdef"]
        elif key == "tdef":
            info["tdef"] = int(tok[1])
        elif key == "vars":
            n = int(tok[1])
            for k in range(n):
                v = lines[i + 1 + k].split()
                vars_.append((v[0], int(v[1])))
            i += n
        i += 1
    info["vars"] = vars_
    return info


def read_grads(ctl) -> dict:
    """`{var: (T, nz, nx)}` (2D vars `(T, nx)`), y squeezed. Truncated files are read up to the last
    complete time (a run still in progress); a file with no complete time raises."""
    ctl = Path(ctl)
    info = parse_ctl(ctl)
    nx, ny, nt = info["xdef"], info["ydef"], info["tdef"]
    per_t = sum(max(n, 1) for _, n in info["vars"]) * nx * ny
    raw = np.fromfile(ctl.parent / info["dset"], dtype="<f4")
    nt = min(nt, raw.size // per_t)
    if nt == 0:
        raise ValueError(f"{ctl}: no complete output time in {info['dset']}")
    raw = raw[: nt * per_t].reshape(nt, per_t)
    out, off = {}, 0
    for name, nlev in info["vars"]:
        L = max(nlev, 1)
        a = raw[:, off: off + L * nx * ny].reshape(nt, L, ny, nx)[:, :, 0, :]
        out[name] = a[:, 0] if nlev == 0 else a
        off += L * nx * ny
    return out


def load_netcdf_run(path) -> dict:
    """`{'X': (T, V, Z, NX), 't', 'cref', 'x', 'z'}` from a CM1 `cm1out.nc`."""
    import netCDF4 as nc

    ds = nc.Dataset(path)
    ds.set_auto_mask(False)
    g = lambda k: ds[k][:, :, 0, :].astype(np.float64)
    out = dict(X=transform(g), t=ds["time"][:].astype(np.float64), x=ds["xh"][:].astype(np.float64),
               z=ds["zh"][:].astype(np.float64), cref=ds["cref"][:, 0, :].astype(np.float32),
               dbz=ds["dbz"][:, :, 0, :].astype(np.float32))
    ds.close()
    return out


def load_grads_run(ctl) -> dict:
    """Same keys as `load_netcdf_run` (no coordinates), from a GrADS run directory's `cm1out_s.ctl`."""
    ctl = Path(ctl)
    d = read_grads(ctl)
    meta = read_grads(ctl.parent / "cm1out_metadata.ctl")
    nt = min(d["th"].shape[0], meta["mtime"].shape[0])
    g = lambda k: d[k][:nt].astype(np.float64)
    return dict(X=transform(g), t=meta["mtime"].ravel()[:nt].astype(np.float64),
                cref=d["cref"][:nt].astype(np.float32), dbz=d["dbz"][:nt].astype(np.float32))
