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
  --nx N            domain width in cells (namelist nx; default: the template's).
  --two-storms SEP  two identical initiating thermals SEP metres apart, symmetric about the domain centre
                    (namelist var2; needs the two_storms_var2 source patch). E.g. --nx 3200 --two-storms
                    400000: storms at 200 and 600 km, each owning a 400 km half like a single 400 km run.
  --tiny-pert AMP   an extra tiny theta field, uniform in [-AMP, AMP] K, from a hash (not CM1's random stream;
                    namelist var3; needs the tiny_pert_var3 patch). With the same --seed, --n and noise settings,
                    an ensemble made with --tiny-pert is the twin of one made without (plan L2). Its seed (var4)
                    is the member number, so every pair gets its own tiny field.
  --design lhs      (10c) a Latin-hypercube design over the environment, with rejection: in addition to --qv-offset and
                    --shear-scale, the free-tropospheric temperature (--dtheta-aloft, K: 0 below 1.5 km, full from 4 to 11 km,
                    back to 0 at 13 km; negative = steeper lapse rate = more CAPE), mid-level humidity (--rh-aloft: qv above
                    2.5 km times a factor, tapered in from 1.5 km, capped at 95% RH), the depth of the low-level shear layer
                    (--shear-depth, km: the wind profile is stretched vertically; the base reaches its 10 m/s at 2.6 km) and
                    deep-layer shear (--upper-shear, m/s: added linearly from the shear-layer top to 9 km) and the
                    boundary-layer temperature at constant relative humidity (--dtheta-bl, K). Each member's
                    mixed-layer CAPE / CIN, 0-3 and 0-6 km bulk shear, 3-6 km RH and precipitable water are computed and
                    members outside --cape-range or above --shear06-max are redrawn (recorded: draws tried).
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


# ---- 10c: environment diagnostics (stdlib only) ----
EPS, LV = 0.622, 2.5e6
RH_CAP = 0.90   # 10c pilot 1: a 95% cap with cooling aloft let CM1's random initial noise start clouds all over the domain


def min_dthdz(rows, zmin=200.0):
    """smallest d(theta)/dz (K/km) above zmin: < 0 is a superadiabatic layer that overturns everywhere at t = 0"""
    return min(1000.0 * (b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(rows, rows[1:]) if a[0] >= zmin)


def _qs(T, p):
    es = 6.112 * math.exp(17.67 * (T - 273.15) / (T - 29.65))
    return EPS * es / (p - es)


def profile(sfc, rows, dz=50.0):
    """the sounding interpolated to a dz grid from the surface: lists z (m), p (hPa), T (K), qv (kg/kg), u (m/s), hydrostatic"""
    zs = [0.0] + [r[0] for r in rows]
    th = [sfc[1]] + [r[1] for r in rows]
    qv = [sfc[2]] + [r[2] for r in rows]
    uu = [rows[0][3]] + [r[3] for r in rows]
    def interp(xs, ys, x):
        for i in range(1, len(xs)):
            if x <= xs[i]:
                w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + w * (ys[i] - ys[i - 1])
        return ys[-1]
    Z, P, T, Q, U = [], [], [], [], []
    pi = (sfc[0] / 1000.0) ** (RD / CP)
    z, thv_prev = 0.0, sfc[1] * (1 + 0.61 * sfc[2] / 1000.0)
    while z <= zs[-1]:
        t_, q_ = interp(zs, th, z), interp(zs, qv, z) / 1000.0
        thv = t_ * (1 + 0.61 * q_)
        if z > 0:
            pi -= G / CP * dz * 0.5 * (1 / thv + 1 / thv_prev)
        thv_prev = thv
        Z.append(z); P.append(1000.0 * pi ** (CP / RD)); T.append(t_ * pi); Q.append(q_); U.append(interp(zs, uu, z))
        z += dz
    return Z, P, T, Q, U


def _theta_e(T, p, q):
    """Bolton (1980) equivalent potential temperature; q in kg/kg"""
    r = 1000.0 * q
    e = max(1e-6, q * p / (EPS + q))
    Td = 243.5 / (17.67 / math.log(e / 6.112) - 1) + 273.15
    TL = 1.0 / (1.0 / (Td - 56.0) + math.log(T / Td) / 800.0) + 56.0
    return T * (1000.0 / p) ** (0.2854 * (1 - 0.28e-3 * r)) * math.exp((3.376 / TL - 0.00254) * r * (1 + 0.81e-3 * r))


def env_stats(sfc, rows, ml_depth=500.0):
    """mixed-layer (lowest ml_depth) pseudo-adiabatic parcel: CAPE, CIN (J/kg); bulk shear 0-3 / 0-6 km (m/s, u only: 2D);
    mean RH 3-6 km; precipitable water (mm); LCL height (m)"""
    Z, P, T, Q, U = profile(sfc, rows)
    n_ml = max(1, int(ml_depth / (Z[1] - Z[0])))
    th_ml = sum(T[i] * (1000.0 / P[i]) ** (RD / CP) for i in range(n_ml)) / n_ml
    q_ml = sum(Q[:n_ml]) / n_ml
    the = _theta_e(th_ml * (P[0] / 1000.0) ** (RD / CP), P[0], q_ml)
    cape = cin = 0.0; lcl = None; lfc_found = False; cin_acc = 0.0
    for i in range(1, len(Z)):
        p = P[i]; Td_ = th_ml * (p / 1000.0) ** (RD / CP)
        if _qs(Td_, p) > q_ml:                     # unsaturated: dry adiabat, constant q
            Tp, qp = Td_, q_ml
        else:                                      # saturated: solve theta_e(T, p, qs(T)) = the by bisection
            if lcl is None:
                lcl = Z[i]
            lo, hi = 150.0, Td_ + 80.0                 # the moist adiabat runs far warmer than the dry one aloft
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _theta_e(mid, p, _qs(mid, p)) > the:
                    hi = mid
                else:
                    lo = mid
            Tp = 0.5 * (lo + hi); qp = _qs(Tp, p)
        b = G * (Tp * (1 + 0.61 * qp) - T[i] * (1 + 0.61 * Q[i])) / (T[i] * (1 + 0.61 * Q[i])) * (Z[i] - Z[i - 1])
        if b > 0 and lcl is not None:
            lfc_found = True; cape += b
        elif b < 0 and not lfc_found:
            cin_acc += b
    cin = cin_acc
    def at(arr, z):
        return arr[min(len(arr) - 1, int(round(z / (Z[1] - Z[0]))))]
    rh36 = [Q[i] / _qs(T[i], P[i]) for i in range(len(Z)) if 3000.0 <= Z[i] <= 6000.0]
    pw = sum(Q[i] * (P[i - 1] - P[i]) * 100.0 / G for i in range(1, len(Z)))      # kg/m^2 = mm
    return {"mlcape": round(cape, 1), "mlcin": round(cin, 1), "lcl_m": lcl, "min_dthdz": round(min_dthdz(rows), 2),
            "shear03": round(at(U, 3000.0) - U[0], 2),
            "shear06": round(at(U, 6000.0) - U[0], 2), "rh36": round(sum(rh36) / max(1, len(rh36)), 3), "pw_mm": round(pw, 1)}


def perturb_env(sfc0, rows0, dth_aloft, rh_aloft, depth_km, upper, ss, dq, qv_zmax, dth_bl=0.0, cap=0.0):
    """10c: the environment perturbations of --design lhs, in this order: wind (depth stretch, magnitude, deep-layer increment),
    boundary-layer temperature at constant relative humidity (dth_bl, K, full below 1 km, tapered to 0 at 3 km: a warmer
    boundary layer is also moister, the realistic route to high CAPE), free-tropospheric theta, mid-level humidity (capped at 95% RH),
    boundary-layer moisture offset."""
    sfc, rows = list(sfc0), [list(r) for r in rows0]
    if dth_bl:
        pi = (sfc[0] / 1000.0) ** (RD / CP)
        T0 = sfc[1] * pi
        sfc[2] = sfc[2] * _qs(T0 + dth_bl * pi, sfc[0]) / _qs(T0, sfc[0])
        sfc[1] += dth_bl
        z_prev, thv_prev = 0.0, sfc0[1] * (1 + 0.61 * sfc0[2] / 1000.0)
        for r, r0 in zip(rows, rows0):
            thv = r0[1] * (1 + 0.61 * r0[2] / 1000.0)
            pi -= G / CP * (r0[0] - z_prev) * 0.5 * (1 / thv + 1 / thv_prev)
            z_prev, thv_prev = r0[0], thv
            w = 1.0 if r0[0] <= 1000.0 else max(0.0, 1.0 - (r0[0] - 1000.0) / 2000.0)   # taper 1-3 km: keeps d(theta)/dz > 0
            if w > 0:
                T = r0[1] * pi; p = 1000.0 * pi ** (CP / RD); dT = dth_bl * w * pi
                r[2] = r0[2] * _qs(T + dT, p) / _qs(T, p)
                r[1] = r0[1] + dth_bl * w
    zb = [r[0] for r in rows0]; ub = [r[3] for r in rows0]
    def u_base(z):
        if z <= zb[0]:
            return ub[0]
        for i in range(1, len(zb)):
            if z <= zb[i]:
                return ub[i - 1] + (z - zb[i - 1]) / (zb[i] - zb[i - 1]) * (ub[i] - ub[i - 1])
        return ub[-1]
    D = 1000.0 * depth_km
    for r in rows:
        z = r[0]
        u = ss * u_base(z * 2600.0 / D)
        u += upper * min(1.0, max(0.0, (z - D) / (9000.0 - D))) if D < 9000.0 else 0.0
        r[3] = u
        ramp = 0.0 if z < 1500 else (z - 1500) / 2500.0 if z < 4000 else 1.0 if z < 11000 else max(0.0, 1 - (z - 11000) / 2000.0)
        r[1] += dth_aloft * ramp
        # capping inversion (10c pilot 2): warming that rises over 1.0-1.4 km and fades by 5 km, so d(theta)/dz stays > 0
        capw = 0.0 if z < 1000 else (z - 1000) / 400.0 if z < 1400 else max(0.0, 1.0 - (z - 1400) / 3600.0)
        r[1] += cap * capw
    # hydrostatic T, p of the perturbed temperature profile for the RH cap
    pi = (sfc[0] / 1000.0) ** (RD / CP); z_prev = 0.0; thv_prev = sfc[1] * (1 + 0.61 * sfc[2] / 1000.0)
    for r in rows:
        w_m = 0.0 if r[0] < 1500 else min(1.0, (r[0] - 1500) / 1000.0)
        q = r[2] * (1.0 + (rh_aloft - 1.0) * w_m)
        w = 1.0 if r[0] <= qv_zmax else max(0.0, 1.0 - (r[0] - qv_zmax) / 500.0)
        q = max(0.0, q + dq * w)
        thv = r[1] * (1 + 0.61 * q / 1000.0)
        pi -= G / CP * (r[0] - z_prev) * 0.5 * (1 / thv + 1 / thv_prev)
        z_prev, thv_prev = r[0], thv
        T = r[1] * pi; p = 1000.0 * pi ** (CP / RD)
        r[2] = min(q, max(RH_CAP * 1000.0 * _qs(T, p), r[2])) if r[0] > 1000.0 else q     # never below the unperturbed value
    sfc[2] = max(0.0, sfc[2] + dq)
    return sfc, rows


def lhs(rng, n, k):
    """n x k Latin hypercube on [0, 1)"""
    cols = []
    for _ in range(k):
        perm = list(range(n)); rng.shuffle(perm)
        cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    return [[cols[j][i] for j in range(k)] for i in range(n)]


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
    ap.add_argument("--design", choices=["random", "corners", "lhs"], default="random")
    ap.add_argument("--dtheta-aloft", type=float, nargs=2, default=[0.0, 0.0], help="10c, lhs: free-tropospheric theta change (K)")
    ap.add_argument("--rh-aloft", type=float, nargs=2, default=[1.0, 1.0], help="10c, lhs: factor on qv above 2.5 km")
    ap.add_argument("--shear-depth", type=float, nargs=2, default=[2.6, 2.6], help="10c, lhs: low-level shear depth (km)")
    ap.add_argument("--upper-shear", type=float, nargs=2, default=[0.0, 0.0], help="10c, lhs: deep-layer wind increment (m/s)")
    ap.add_argument("--dtheta-bl", type=float, nargs=2, default=[0.0, 0.0], help="10c, lhs: boundary-layer theta change at constant RH (K)")
    ap.add_argument("--cape-range", type=float, nargs=2, default=[0.0, 1e9], help="10c, lhs: accepted mixed-layer CAPE (J/kg)")
    ap.add_argument("--shear06-max", type=float, default=1e9, help="10c, lhs: accepted max 0-6 km bulk shear (m/s)")
    ap.add_argument("--cap-theta", type=float, nargs=2, default=[0.0, 0.0], help="10c, lhs: capping-inversion warming (K) over 1.0-1.4 km, fading by 5 km")
    ap.add_argument("--cin-per-cape", type=float, default=0.0, help="10c, lhs: reject unless |CIN| >= this x (CAPE - 600) (J/kg)")
    ap.add_argument("--min-dthdz", type=float, default=-1e9, help="10c, lhs: reject soundings with d(theta)/dz below this (K/km) above 200 m")
    ap.add_argument("--rh36-max", type=float, default=1e9, help="10c, lhs: reject soundings with mean 3-6 km RH above this")
    ap.add_argument("--reference-sounding", help="also make mem000 with this exact sounding (validation run)")
    ap.add_argument("--nx", type=int, default=0, help="domain width in cells (0: the template's)")
    ap.add_argument("--two-storms", type=float, default=0.0, help="separation (m) of two initiating thermals (var2)")
    ap.add_argument("--tiny-pert", type=float, default=0.0, help="amplitude (K) of an extra tiny theta field (var3)")
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
    if a.two_storms:   # an unpatched cm1.exe ignores var2 and silently makes ONE storm at the centre
        patches = os.path.join(os.path.dirname(os.path.dirname(exe)), "PATCHES")
        if not (os.path.exists(patches) and "two_storms_var2" in open(patches).read()):
            sys.exit("--two-storms needs a CM1 built with the two_storms_var2 patch (%s not found or lacks it)" % patches)
        if not a.nx:
            sys.exit("--two-storms: give --nx too (two storms need a wider domain)")
    if a.tiny_pert:   # an unpatched cm1.exe ignores var3: the "twin" would be identical to its control
        patches = os.path.join(os.path.dirname(os.path.dirname(exe)), "PATCHES")
        if not (os.path.exists(patches) and "tiny_pert_var3" in open(patches).read()):
            sys.exit("--tiny-pert needs a CM1 built with the tiny_pert_var3 patch (%s not found or lacks it)" % patches)
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
    if a.nx:
        nml = set_namelist(nml, "nx", a.nx)
    if a.two_storms:
        nml = set_namelist(nml, "var2", "%.1f" % a.two_storms)

    sfc0, rows0 = read_sounding(os.path.join(tmpl, "input_sounding"))
    base_rh = max_rh(sfc0, rows0)
    rng = random.Random(a.seed)
    members = []
    if a.reference_sounding:
        members.append(("mem000", read_sounding(a.reference_sounding), {"reference": os.path.abspath(a.reference_sounding)}))
    if a.design == "lhs":
        box = [a.qv_offset, a.shear_scale, a.dtheta_aloft, a.rh_aloft, a.shear_depth, a.upper_shear, a.dtheta_bl, a.cap_theta]
        tried, accepted = 0, []
        while len(accepted) < a.n:
            for x in lhs(rng, a.n, len(box)):
                if len(accepted) >= a.n:
                    break
                tried += 1
                dq, ss, dta, rha, dep, ups, dtb, cap = [lo + u * (hi - lo) for u, (lo, hi) in zip(x, box)]
                sfc, rows = perturb_env(sfc0, rows0, dta, rha, dep, ups, ss, dq, a.qv_zmax, dtb, cap)
                if max_rh(sfc, rows) > max(1.0, base_rh + 1e-6):
                    continue
                st = env_stats(sfc, rows)
                if not (a.cape_range[0] <= st["mlcape"] <= a.cape_range[1]) or abs(st["shear06"]) > a.shear06_max:
                    continue
                if st["min_dthdz"] < a.min_dthdz or st["rh36"] > a.rh36_max:
                    continue
                if -st["mlcin"] < a.cin_per_cape * (st["mlcape"] - 600.0):   # 10c pilot 2: CM1's domain-wide 0.25 K noise sets off cells
                    continue                                                  # everywhere when CAPE is high and CIN weak

                accepted.append((dq, ss, dta, rha, dep, ups, dtb, cap, sfc, rows, st))
            if tried > 200 * a.n:
                sys.exit("lhs: fewer than %d of %d draws accepted; widen --cape-range / --shear06-max" % (len(accepted), tried))
        for i, (dq, ss, dta, rha, dep, ups, dtb, cap, sfc, rows, st) in zip(range(a.start, a.start + a.n), accepted):
            dth_sfc = rng.gauss(0.0, a.theta_sigma)
            sfc[1] += dth_sfc
            dths = []
            for r in rows:
                if r[0] < a.theta_zmax:
                    d = rng.gauss(0.0, a.theta_sigma); r[1] += d; dths.append(round(d, 5))
            members.append(("mem%03d" % i, (sfc, rows),
                            dict(dtheta_sfc=round(dth_sfc, 5), dtheta_levels_below_zmax=dths, qv_offset_gkg=round(dq, 4),
                                 shear_scale=round(ss, 4), dtheta_aloft=round(dta, 3), rh_aloft=round(rha, 3),
                                 shear_depth_km=round(dep, 3), upper_shear=round(ups, 3), dtheta_bl=round(dtb, 3), cap_theta=round(cap, 3),
                                 draws_tried=tried, **st)))
    for i in (range(a.start, a.start + a.n) if a.design != "lhs" else []):
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
        text = set_namelist(nml, "var1", "%.1f" % seed)
        if a.tiny_pert:   # %g: %.1f would write 0.001 as 0.0
            text = set_namelist(set_namelist(text, "var3", "%g" % a.tiny_pert), "var4", "%.1f" % int(name[3:]))
        with open(os.path.join(d, "namelist.input"), "w") as f:
            f.write(text)
        info["tiny_pert_var3"] = a.tiny_pert
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
        extra = ["dtheta_bl", "cap_theta", "dtheta_aloft", "rh_aloft", "shear_depth_km", "upper_shear", "mlcape", "mlcin", "min_dthdz", "shear03", "shear06", "rh36", "pw_mm"]
        extra = [k for k in extra if any(k in m for m in manifest)]
        w.writerow(["member", "dtheta_sfc", "qv_offset_gkg", "shear_scale", "max_rh"] + extra)
        for m in manifest:
            w.writerow([m["member"], m.get("dtheta_sfc", ""), m.get("qv_offset_gkg", ""),
                        m.get("shear_scale", ""), m["max_rh"]] + [m.get(k, "") for k in extra])
    print("made %d members in %s (base sounding max RH %.3f)" % (len(manifest), os.path.join(a.out, "members"), base_rh))


if __name__ == "__main__":
    main()
