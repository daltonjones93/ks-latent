"""C16 (docs/ks_local_field_ltransfer.tex Sec. self_similar): is Kolmogorov flow
approximately self-similar across scales? No training.

On the wide 16x2pi datasets (alpha = 0.05 and 0.1), using the training
trajectories (normalized vorticity; all quantities below are ratios or slopes,
so the global normalization cancels):

  (i)  streamwise velocity-energy spectrum E(k_x) and its log-log slope;
       velocity structure functions S2(r), S3(r) along x (longitudinal u).
       Inverse-cascade predictions: E ~ k^-5/3, S2 ~ r^2/3, S3 > 0 and ~ r.
  (ii) band-pass scale decomposition along x at block widths l in
       {1, 2, 4, 8, 16} sites (1 site = pi/4): omega_l = B_l omega - B_2l omega
       (B = streamwise block average). Self-similarity predicts
       std(omega_l) ~ l^-2/3 and correlation time tau(l) ~ l^2/3, and that the
       PDFs of omega_l/std collapse and the autocorrelations collapse after
       t -> t / tau(l).
  (iii) (if C8's truth basis exists) unstable-subspace variance per octave of
       |k| at 3x, alpha = 0.05.

Output: artifacts/experiments/c16/c16_result.json, with the scale range over
which the rescaled PDFs and autocorrelations collapse within 15% and the
pre-registered verdict (proceed to C17/C18 iff that range spans >= 4x at
alpha = 0.05).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import math
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "c16"
DATA = {0.05: A / "datasets/kolmogorov_wide16_alpha0.05.h5", 0.1: A / "datasets/kolmogorov_wide16_alpha0.1.h5"}
TRUTH_CKPT = A / "experiments/crop/truth_a005_x3_seed0.ckpt.npz"
TRUTH_JSON = A / "experiments/crop/truth_a005_x3_seed0.json"
LEVELS = (1, 2, 4, 8, 16)  # block widths in sites
SITE = 16  # grid points per site
TOL = 0.15


def load(alpha, t_stride=2):
    with h5py.File(DATA[alpha], "r") as f:
        md = dict(f["metadata"].attrs)
        n_tr = int(md["n_train"])
        x = f["trajectories"][:n_tr, ::t_stride].astype(np.float64)
    NX, NY = int(md["NX"]), int(md["NY"])
    return x.reshape(x.shape[0], x.shape[1], NX, NY), md, float(md["dt_snap"]) * t_stride


def velocity(om, Lx, Ly):
    NX, NY = om.shape[-2:]
    kx = np.fft.fftfreq(NX, d=Lx / NX) * 2 * np.pi
    ky = np.fft.fftfreq(NY, d=Ly / NY) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    psi = np.fft.fft2(om) / K2
    psi[..., 0, 0] = 0
    u = np.fft.ifft2(1j * KY * psi).real
    v = np.fft.ifft2(-1j * KX * psi).real
    return u, v, KX, KY


def part_i(om, Lx, Ly):
    NX = om.shape[-2]
    u, v, KX, KY = velocity(om, Lx, Ly)
    U = np.abs(np.fft.fft(u, axis=-2)) ** 2 + np.abs(np.fft.fft(v, axis=-2)) ** 2
    E = U.sum(axis=-1).mean(axis=tuple(range(U.ndim - 2)))  # (NX,)
    m = np.arange(1, NX // 2)
    E = E[1:NX // 2] + E[-1:-(NX // 2):-1][: len(m)]
    k = 2 * np.pi * m / Lx

    def slope(lo, hi):
        sel = (k >= lo) & (k <= hi)
        return float(np.polyfit(np.log(k[sel]), np.log(E[sel]), 1)[0]) if sel.sum() >= 3 else None

    spec = {"slope_k_0.25_1": slope(0.25, 1.0), "slope_k_1_3": slope(1.0, 3.0), "slope_k_5_15": slope(5.0, 15.0)}
    dx = Lx / NX
    rs = [2 ** j for j in range(0, 10) if 2 ** j < NX // 2]
    S2, S3 = [], []
    for r in rs:
        du = np.roll(u, -r, axis=-2) - u
        S2.append(float((du ** 2).mean()))
        S3.append(float((du ** 3).mean()))
    r_phys = np.array(rs) * dx
    sel = (r_phys >= 0.5) & (r_phys <= 6.0)
    s2_slope = float(np.polyfit(np.log(r_phys[sel]), np.log(np.array(S2)[sel]), 1)[0]) if sel.sum() >= 3 else None
    return {"spectrum_slopes": spec, "S2_slope_r_0.5_6": s2_slope,
            "S3_sign_r_0.5_6": [int(np.sign(x)) for x, rr in zip(S3, r_phys) if 0.5 <= rr <= 6.0],
            "r_phys": r_phys.tolist(), "S2": S2, "S3": S3}


def block_avg_x(om, w):
    """Streamwise block average over w grid points, broadcast back to full grid."""
    NX = om.shape[-2]
    b = om.reshape(*om.shape[:-2], NX // w, w, om.shape[-1]).mean(axis=-2)
    return np.repeat(b, w, axis=-2)


def acf(x, max_lag):
    """Mean autocorrelation along time axis 1 of x: (N, T, ...)."""
    x = x - x.mean(axis=1, keepdims=True)
    var = (x ** 2).mean()
    return np.array([(x[:, : x.shape[1] - l] * x[:, l:]).mean() / var for l in range(max_lag)])


def part_ii(om, dt):
    stds, taus, pdfs, acfs = {}, {}, {}, {}
    bins = np.linspace(-5, 5, 41)
    for l in LEVELS:
        band = block_avg_x(om, l * SITE) - block_avg_x(om, 2 * l * SITE)
        sd = float(band.std())
        stds[l] = sd
        h, _ = np.histogram(band.ravel() / sd, bins=bins, density=True)
        pdfs[l] = h
        a = acf(band[:, :, ::max(1, l * SITE // 4), ::4], max_lag=min(60, band.shape[1] - 1))
        acfs[l] = a
        below = np.where(a < 1 / math.e)[0]
        taus[l] = float(below[0] * dt) if len(below) else None
    L = np.array(LEVELS, dtype=float)
    sd_slope = float(np.polyfit(np.log(L), np.log([stds[l] for l in LEVELS]), 1)[0])
    tv = [(l, taus[l]) for l in LEVELS if taus[l]]
    tau_slope = float(np.polyfit(np.log([a for a, _ in tv]), np.log([b for _, b in tv]), 1)[0]) if len(tv) >= 3 else None

    def pdf_dist(a, b):
        return float(0.5 * np.abs(pdfs[a] - pdfs[b]).sum() * (bins[1] - bins[0]))

    def acf_dist(a, b):
        ta, tb = taus[a], taus[b]
        if not ta or not tb:
            return None
        s = np.linspace(0, 2.0, 21)  # rescaled time t/tau
        fa = np.interp(s * ta / dt, np.arange(len(acfs[a])), acfs[a])
        fb = np.interp(s * tb / dt, np.arange(len(acfs[b])), acfs[b])
        return float(np.max(np.abs(fa - fb)))

    pairs = {f"{a}-{b}": {"pdf_tv_distance": pdf_dist(a, b), "acf_max_diff_rescaled": acf_dist(a, b)}
             for a, b in zip(LEVELS[:-1], LEVELS[1:])}
    ok = [all(v is not None and v <= TOL for v in p.values()) for p in pairs.values()]
    best, cur, start, best_span = 0, 0, 0, (None, None)
    for i, good in enumerate(ok):
        cur = cur + 1 if good else 0
        if cur > best:
            best, best_span = cur, (LEVELS[i + 1 - cur], LEVELS[i + 1])
    factor = (best_span[1] / best_span[0]) if best else 1.0
    return {"std_by_level": stds, "std_slope (pred -2/3)": sd_slope, "tau_by_level": taus,
            "tau_slope (pred +2/3)": tau_slope, "pairwise": pairs,
            "collapse_range_sites": best_span, "collapse_factor": factor}


def part_iii():
    if not (TRUTH_CKPT.exists() and TRUTH_JSON.exists()):
        return None
    n = int(json.loads(TRUTH_JSON.read_text())["n_positive"])
    with np.load(TRUTH_CKPT) as c:
        Q = c["tangent"][:n]
    NX, NY = 384, 128
    q, _ = np.linalg.qr(Q.reshape(n, -1).T)
    V = q.T.reshape(n, NX, NY)
    P = (np.abs(np.fft.fft2(V)) ** 2).sum(axis=0)
    kx = np.fft.fftfreq(NX, d=6 * np.pi / NX) * 2 * np.pi
    ky = np.fft.fftfreq(NY, d=2 * np.pi / NY) * 2 * np.pi
    K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = [0, 0.5, 1, 2, 4, 8, 16, 32, 1e9]
    tot = P.sum()
    return {f"|k| in [{edges[i]},{edges[i+1]})": float(P[(K >= edges[i]) & (K < edges[i + 1])].sum() / tot * n)
            for i in range(len(edges) - 1)} | {"n_positive": n, "units": "dimensions-equivalent per band"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for alpha in (0.05, 0.1):
        om, md, dt = load(alpha)
        Lx, Ly = float(md["Lx"]), float(md["Ly"])
        res[f"alpha={alpha}"] = {"i": part_i(om, Lx, Ly), "ii": part_ii(om, dt)}
        print(f"[c16] alpha={alpha}: spectrum {res[f'alpha={alpha}']['i']['spectrum_slopes']}, "
              f"collapse {res[f'alpha={alpha}']['ii']['collapse_range_sites']}", flush=True)
    res["iii_alpha0.05_x3"] = part_iii()
    f05 = res["alpha=0.05"]["ii"]["collapse_factor"]
    res["verdict"] = ("PROCEED to C17/C18 (collapse over >= 4x at alpha=0.05)" if f05 >= 4 else
                      f"DO NOT PROCEED (collapse range only {f05:g}x at alpha=0.05); fall back to C15")
    (OUT / "c16_result.json").write_text(json.dumps(res, indent=2, default=str))
    print(res["verdict"], flush=True)


if __name__ == "__main__":
    main()
