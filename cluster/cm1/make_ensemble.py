#!/usr/bin/env python3
"""Generate a CM1 2D squall-line ensemble: one directory per member.

Standard library only (runs with any python3 >= 3.6 on a cluster).

Each member directory gets:
  namelist.input  - the template namelist with run length, output interval and
                    output variables set (see --timax, --tapfrq, --trim-output)
  input_sounding  - the base sounding, perturbed (see below)
  links to cm1.exe and the lookup tables in the template directory
and the ensemble directory gets manifest.json / manifest.csv recording every
member's settings and random draws, so the ensemble is reproducible.

Perturbations (all optional, recorded in the manifest):
  --theta-sigma S   independent N(0, S^2) theta perturbations (K) at the surface
                    and at every sounding level below --theta-zmax (m). This is
                    how the existing 10 members were made (S ~ 0.1 K, z < 500 m).
  --qv-offset A B   boundary-layer moisture offset (g/kg) drawn uniformly in [A, B],
                    applied fully below --qv-zmax and tapered to zero over 500 m.
  --shear-scale A B multiply the wind profile by a factor drawn in [A, B].
  --noise-seed member  give each member its own seed for CM1's random initial noise (namelist var1 =
                    member number; needs the seed_var1 source patch, see patches/apply_patches.py).
                    mem000 keeps var1 = 0, i.e. the original fixed seed. --noise-offset N adds N, so
                    different ensembles never share noise (e.g. 1000 for one, 2000 for the next).
  --design corners  instead of random draws, cycle through the 4 corners of the --qv-offset x
                    --shear-scale box (a pilot of the extremes).
Every perturbed sounding is checked for supersaturation; the script stops with
an error rather than write one.

Example (30 members in the original environment, 1-min output, 6 h):
  python3 make_ensemble.py --template template --exe build/run/cm1.exe \
      --out ens_A --n 30 --seed 1
"""
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys

TABLES = [
    "LANDUSE.TBL", "RRTMG_LW_DATA", "RRTMG_SW_DATA", "freezeH2O.dat",
    "ishmael-gamma-tab.bin", "ishmael-qi-qc.bin", "ishmael-qi-qr.bin",
    "lsnudge_0001.dat", "p3_lookupTable_1.dat-v5.4_2momI",
    "p3_lookupTable_1.dat-v5.4_3momI", "p3_lookupTable_2.dat-v5.3",
    "qr_acr_qgV2.dat", "qr_acr_qsV2.dat",
]
# fields we do not use (subgrid TKE, eddy viscosities, staggered winds, v=0 in 2D)
TRIM = {"output_tke": 0, "output_km": 0, "output_kh": 0, "output_u": 0,
        "output_v": 0, "output_w": 0, "output_vinterp": 0}

G, CP, RD = 9.81, 1004.0, 287.04


def read_sounding(path):
    with open(path) as f:
        lines = [l.split() for l in f if l.strip()]
    sfc = [float(v) for v in lines[0]]                 # p_sfc (hPa), theta_sfc (K), qv_sfc (g/kg)
    rows = [[float(v) for v in l] for l in lines[1:]]  # z (m), theta, qv (g/kg), u, v
    return sfc, rows


def write_sounding(path, sfc, rows):
    with open(path, "w") as f:
        f.write("%10.2f  %10.4f  %10.4f\n" % tuple(sfc))
        for r in rows:
            f.write("%10.2f  %10.4f  %10.5f  %10.4f  %10.4f\n" % tuple(r))


def max_rh(sfc, rows):
    """Max relative humidity (fraction) over the sounding, hydrostatic pressure."""
    p0, th0, q0 = sfc[0], sfc[1], sfc[2] / 1000.0
    pi = (p0 / 1000.0) ** (RD / CP)
    z_prev, thv_prev = 0.0, th0 * (1 + 0.61 * q0)
    worst = 0.0
    for z, th, qv, _, _ in rows:
        q = qv / 1000.0
        thv = th * (1 + 0.61 * q)
        pi -= G / CP * (z - z_prev) * 0.5 * (1 / thv + 1 / thv_prev)
        z_prev, thv_prev = z, thv
        T = th * pi
        p = 1000.0 * pi ** (CP / RD)
        es = 6.112 * math.exp(17.67 * (T - 273.15) / (T - 29.65))
        qs = 0.622 * es / (p - es)
        worst = max(worst, q / qs)
    return worst


def set_namelist(text, key, value):
    pat = re.compile(r"^(\s*%s\s*=\s*)([^,\n]*)(,?)" % re.escape(key), re.M)
    if not pat.search(text):
        sys.exit("namelist key not found: %s" % key)
    return pat.sub(lambda m: m.group(1) + str(value) + m.group(3), text, count=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="dir with namelist.input, input_sounding, lookup tables")
    ap.add_argument("--exe", required=True, help="path to the built cm1.exe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=1, help="number of the first member")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timax", type=float, default=21600.0, help="run length (s)")
    ap.add_argument("--tapfrq", type=float, default=60.0, help="output interval (s)")
    ap.add_argument("--output-format", type=int, default=2, choices=[1, 2], help="1 GrADS, 2 NetCDF")
    ap.add_argument("--no-trim-output", action="store_true", help="keep TKE, eddy viscosities, staggered winds")
    ap.add_argument("--theta-sigma", type=float, default=0.1)
    ap.add_argument("--theta-zmax", type=float, default=500.0)
    ap.add_argument("--qv-offset", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--qv-zmax", type=float, default=1000.0)
    ap.add_argument("--shear-scale", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--exe-may-not-exist-yet", action="store_true",
                    help="link cm1.exe even if it is not built yet (go.sh: the build is a queued job)")
    ap.add_argument("--noise-seed", choices=["fixed", "member"], default="fixed",
                    help="fixed: CM1's one seed for all members; member: var1 = member number")
    ap.add_argument("--noise-offset", type=int, default=0, help="added to var1 with --noise-seed member")
    ap.add_argument("--design", choices=["random", "corners"], default="random")
    ap.add_argument("--reference-sounding", help="also make mem000 with this exact sounding (validation run)")
    a = ap.parse_args()

    tmpl = os.path.abspath(a.template)
    exe = os.path.abspath(a.exe)
    for f in ["namelist.input", "input_sounding"] + TABLES:
        if not os.path.exists(os.path.join(tmpl, f)):
            sys.exit("missing in template: %s" % f)
    if not os.path.exists(exe) and not a.exe_may_not_exist_yet:
        sys.exit("cm1.exe not found: %s" % exe)
    if a.noise_seed == "member" and os.path.exists(exe):  # a queued build (go.sh) is always patched
        patches = os.path.join(os.path.dirname(os.path.dirname(exe)), "PATCHES")  # build/run/cm1.exe -> build/PATCHES
        if not (os.path.exists(patches) and "seed_var1" in open(patches).read()):
            sys.exit("--noise-seed member needs a CM1 built with the seed_var1 patch (cluster/cm1/build_cm1.sh "
                     "applies it; %s not found or lacks it). Rebuild: delete the build directory and re-run." % patches)
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit("output directory exists and is not empty: %s (refusing to overwrite)" % a.out)

    with open(os.path.join(tmpl, "namelist.input")) as f:
        nml = f.read()
    nml = set_namelist(nml, "timax", "%.1f" % a.timax)
    nml = set_namelist(nml, "tapfrq", "%.1f" % a.tapfrq)
    nml = set_namelist(nml, "output_format", a.output_format)
    nml = set_namelist(nml, "output_filetype", 1)
    if not a.no_trim_output:
        for k, v in TRIM.items():
            nml = set_namelist(nml, k, v)

    sfc0, rows0 = read_sounding(os.path.join(tmpl, "input_sounding"))
    base_rh = max_rh(sfc0, rows0)
    rng = random.Random(a.seed)
    members = []
    if a.reference_sounding:
        members.append(("mem000", read_sounding(a.reference_sounding), {"reference": os.path.abspath(a.reference_sounding)}))
    for i in range(a.start, a.start + a.n):
        sfc, rows = list(sfc0), [list(r) for r in rows0]
        if a.design == "corners":
            c = (i - a.start) % 4
            dq, ss = a.qv_offset[c % 2], a.shear_scale[c // 2]
        else:
            dq = rng.uniform(*a.qv_offset)
            ss = rng.uniform(*a.shear_scale)
        dth_sfc = rng.gauss(0.0, a.theta_sigma)
        sfc[1] += dth_sfc
        dths = []
        for r in rows:
            if r[0] < a.theta_zmax:
                d = rng.gauss(0.0, a.theta_sigma)
                r[1] += d
                dths.append(round(d, 5))
            w = 1.0 if r[0] <= a.qv_zmax else max(0.0, 1.0 - (r[0] - a.qv_zmax) / 500.0)
            r[2] = max(0.0, r[2] + dq * w)
            r[3] *= ss
            r[4] *= ss
        sfc[2] = max(0.0, sfc[2] + dq)
        members.append(("mem%03d" % i, (sfc, rows),
                        {"dtheta_sfc": round(dth_sfc, 5), "dtheta_levels_below_zmax": dths,
                         "qv_offset_gkg": round(dq, 4), "shear_scale": round(ss, 4)}))

    os.makedirs(os.path.join(a.out, "members"))
    manifest = []
    for name, (sfc, rows), info in members:
        rh = max_rh(sfc, rows)
        if rh > max(1.0, base_rh + 1e-6):
            sys.exit("%s: perturbed sounding is supersaturated (max RH %.3f); narrow --qv-offset" % (name, rh))
        d = os.path.join(a.out, "members", name)
        os.makedirs(d)
        if "reference" in info:   # copy byte for byte, so mem000 reproduces the original run exactly
            shutil.copyfile(info["reference"], os.path.join(d, "input_sounding"))
        else:
            write_sounding(os.path.join(d, "input_sounding"), sfc, rows)
        seed = (a.noise_offset + int(name[3:])) if (a.noise_seed == "member" and name != "mem000") else 0
        with open(os.path.join(d, "namelist.input"), "w") as f:
            f.write(set_namelist(nml, "var1", "%.1f" % seed))
        info["noise_seed_var1"] = seed
        os.symlink(exe, os.path.join(d, "cm1.exe"))
        for t in TABLES:
            os.symlink(os.path.join(tmpl, t), os.path.join(d, t))
        manifest.append(dict(member=name, max_rh=round(rh, 4), **info))

    meta = {"args": vars(a), "template": tmpl, "exe": exe, "base_max_rh": round(base_rh, 4),
            "members": manifest}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["member", "dtheta_sfc", "qv_offset_gkg", "shear_scale", "max_rh"])
        for m in manifest:
            w.writerow([m["member"], m.get("dtheta_sfc", ""), m.get("qv_offset_gkg", ""),
                        m.get("shear_scale", ""), m["max_rh"]])
    print("made %d members in %s (base sounding max RH %.3f)" % (len(manifest), os.path.join(a.out, "members"), base_rh))


if __name__ == "__main__":
    main()
