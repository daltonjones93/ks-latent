"""Read CM1 GrADS (output_format=1) output: <prefix>_s.ctl/.dat etc.

Layout: direct-access float32, native (little) endian on this machine; for
each time, each variable in .ctl order, nlev levels (0 = one 2D level) of
nx*ny values. 2D x-z runs have ny=1.
"""
from pathlib import Path

import numpy as np


def parse_ctl(ctl: Path):
    lines = ctl.read_text().splitlines()
    info, vars_, i = {}, [], 0
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1; continue
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
    """Return {var: array (T, nz, nx)} (2D surface vars: (T, nx)), squeezing y."""
    ctl = Path(ctl)
    info = parse_ctl(ctl)
    nx, ny, nt = info["xdef"], info["ydef"], info["tdef"]
    per_t = sum(max(n, 1) for _, n in info["vars"]) * nx * ny
    raw = np.fromfile(ctl.parent / info["dset"], dtype="<f4")
    nt = min(nt, raw.size // per_t)
    raw = raw[: nt * per_t].reshape(nt, per_t)
    out, off = {}, 0
    for name, nlev in info["vars"]:
        L = max(nlev, 1)
        a = raw[:, off: off + L * nx * ny].reshape(nt, L, ny, nx)[:, :, 0, :]
        out[name] = a[:, 0] if nlev == 0 else a
        off += L * nx * ny
    return out
