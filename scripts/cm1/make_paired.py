"""Paired CM1 runs, stage 2 (run on Cashew): branch directories from the stage-1 restarts.

Per base member: the gust front xf at 7200 s (frame 120 of the base output, detector given by --detector), then branches
that all restart from the same restart file and run to 10800 s with 1-min output:
  ctrl, p{amp}_{offset} (Gaussian theta' bump, sigma 2 km, lowest 6 levels, linear taper), m0.5_0 (negative), tiny1e-3, tiny1e-4.
Offsets are in km from xf (negative = ahead, i.e. toward -x). Writes <out>/members/<base>_<branch> and manifest.json
(bump centre cell var6 for each branch) so the comparison reproduces the bump exactly.
  python make_paired.py --base ~/cm1_bundle/ens_P --cache ~/cm1_cache_P --out ~/cm1_bundle/ens_Q --detector edge1
"""
from __future__ import annotations
import argparse, json, re, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from front_detect import detect, site_theta  # noqa: E402
from ks_latent.cm1.data import load_member  # noqa: E402

SIGMA_M, NLEV, T0_S = 2000.0, 6, 7200


def set_nl(text, key, val):
    pat = re.compile(rf"(?m)^ *{key} *=.*$")
    if not pat.search(text):
        raise ValueError(f"{key} not in namelist")
    return pat.sub(f" {key} = {val},", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="edge1"); ap.add_argument("--offsets-km", default="-20,-10,0,10")
    ap.add_argument("--amp", type=float, default=0.5); ap.add_argument("--small-amp", type=float, default=0.05)
    a = ap.parse_args()
    base, out = Path(a.base).expanduser(), Path(a.out).expanduser()
    offs = [float(x) for x in a.offsets_km.split(",")]
    man = {"detector": a.detector, "sigma_m": SIGMA_M, "nlev": NLEV, "t0_s": T0_S, "members": {}}
    for f in sorted(Path(a.cache).expanduser().glob("*.npz")):
        m = load_member(f)
        short = f.stem.split("__")[-1]                       # memA029
        th = site_theta(m, int(round(T0_S / m.dt)))
        xf = detect(th, a.detector)
        if th.min() > -2.0:
            print(f"{short}: no cold pool at t0 (min {th.min():.2f} K), skipped"); continue
        bd = base / "members" / short
        branches = {"ctrl": {}}
        for o in offs:
            branches[f"p{a.amp:g}_{o:+g}"] = {"var5": a.amp, "var6": (xf + o / 2.0) * 8 + 3.5, "off": o}
        branches[f"m{a.amp:g}_+0"] = {"var5": -a.amp, "var6": xf * 8 + 3.5, "off": 0.0}
        branches[f"p{a.small_amp:g}_+0"] = {"var5": a.small_amp, "var6": xf * 8 + 3.5, "off": 0.0}
        branches["tiny1e-3"] = {"var3": 1e-3}
        branches["tiny1e-4"] = {"var3": 1e-4}
        man["members"][short] = {"front_site": int(xf), "theta_min": float(th.min()), "branches": {}}
        for name, par in branches.items():
            D = out / "members" / f"{short}_{name}".replace("+", "p").replace("-", "n")
            if D.exists(): shutil.rmtree(D)
            D.mkdir(parents=True)
            for g in bd.iterdir():
                if g.name.startswith("cm1out") or g.name in ("cm1.done", "cm1.log", "cm1_config.txt", "cm1.queued", "cm1.failed"):
                    continue
                shutil.copy2(g, D / g.name)
            s = (D / "namelist.input").read_text()
            for k, v in (("irst", 1), ("rstnum", 1), ("timax", 10800.0), ("rstfrq", -900.0), ("tapfrq", 60.0)):
                s = set_nl(s, k, v)
            for k, v in par.items():
                if k.startswith("var"): s = set_nl(s, k, v)
            if "var5" in par:
                s = set_nl(s, "var10", SIGMA_M); s = set_nl(s, "var11", float(NLEV))
            (D / "namelist.input").write_text(s)
            man["members"][short]["branches"][name] = {**par, "dir": D.name}
        print(f"{short}: front site {xf} (theta' min {th.min():.1f} K), {len(branches)} branches")
    (out).mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
