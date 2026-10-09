"""C13 (docs/ks_local_field_ltransfer.tex Sec. two_scale): how closed are the
large scales? No training of neural networks.

For a sharp spectral low-pass at |k| < k_c (k_c in {3, 2, 1}, i.e. cutoff
wavelengths 2pi/3, pi, 2pi; all below the forcing wavenumber 4, so the filtered
forcing vanishes), on the wide 16x2pi datasets:

  T  = d(omega_bar)/dt                    centred difference of consecutive snapshots
  R  = -u_bar.grad(omega_bar) + nu lap(omega_bar) - alpha omega_bar     (exact coarse terms)
  E  = -[ filter(u.grad omega) - u_bar.grad(omega_bar) ]                  (exact eddy forcing)

Reported fractions of var(T) (on held-out validation trajectories):
  consistency   1 - var(T - R - E)/var(T)   time-discretization check (should be ~1)
  (i) resolved  1 - var(T - R)/var(T)
  (ii) closure  1 - var(T - R - E_hat)/var(T), E_hat = linear least-squares closure in
                local coarse features (omega_bar and derivatives, u_bar, v_bar) and
                small-scale enstrophy features (z' = filter((omega - omega_bar)^2) and
                derivatives), fitted on training trajectories
  (iii) remainder = 1 - (ii)
Pre-registered: a large-scale model is viable if (ii) >= 0.8 at k_c = 1 (cutoff 2pi);
a remainder > 0.3 means the closure must be stochastic.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
OUT = A / "experiments" / "c13"
DATA = {0.05: A / "datasets/kolmogorov_wide16_alpha0.05.h5", 0.1: A / "datasets/kolmogorov_wide16_alpha0.1.h5"}
KCS = (3.0, 2.0, 1.0)
T_STEP = 8  # centre-snapshot stride
PT_STRIDE = (16, 4)  # spatial subsampling of grid points used in the regression


class Spec:
    def __init__(self, NX, NY, Lx, Ly):
        kx = np.fft.fftfreq(NX, d=Lx / NX) * 2 * np.pi
        ky = np.fft.fftfreq(NY, d=Ly / NY) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        self.Kmag = np.sqrt(self.K2)

    def dx(self, fh):
        return np.fft.ifft2(1j * self.KX * fh).real

    def dy(self, fh):
        return np.fft.ifft2(1j * self.KY * fh).real

    def vel(self, wh):
        ph = wh * self.K2inv
        return np.fft.ifft2(1j * self.KY * ph).real, np.fft.ifft2(-1j * self.KX * ph).real

    def lowpass(self, fh, kc):
        return fh * (self.Kmag < kc)


def fields(w, S, kc, nu, alpha):
    """w: (NX, NY) raw vorticity. Returns omega_bar_hat, R, E, features (F, NX, NY)."""
    wh = np.fft.fft2(w)
    u, v = S.vel(wh)
    adv_h = np.fft.fft2(u * S.dx(wh) + v * S.dy(wh))
    wbh = S.lowpass(wh, kc)
    wb = np.fft.ifft2(wbh).real
    ub, vb = S.vel(wbh)
    radv = ub * S.dx(wbh) + vb * S.dy(wbh)
    E = -(np.fft.ifft2(S.lowpass(adv_h, kc)).real - radv)
    lap = np.fft.ifft2(-S.K2 * wbh).real
    R = -radv + nu * lap - alpha * wb
    small = w - wb
    zh = S.lowpass(np.fft.fft2(small ** 2), kc)
    feats = [wb, S.dx(wbh), S.dy(wbh), lap, np.fft.ifft2(S.K2 ** 2 * wbh).real, ub, vb,
             np.fft.ifft2(zh).real, S.dx(zh), S.dy(zh), np.fft.ifft2(-S.K2 * zh).real]
    return wbh, R, E, np.stack(feats)


def collect(traj, S, kc, nu, alpha, dt, mean, std):
    rows = {"T": [], "R": [], "E": [], "X": []}
    T_len = traj.shape[0]
    for t in range(1, T_len - 1, T_STEP):
        w_m, w0, w_p = (traj[t + d] * std + mean for d in (-1, 0, 1))
        wbh_m = S.lowpass(np.fft.fft2(w_m), kc)
        wbh_p = S.lowpass(np.fft.fft2(w_p), kc)
        Tt = np.fft.ifft2(wbh_p - wbh_m).real / (2 * dt)
        _, R, E, X = fields(w0, S, kc, nu, alpha)
        sl = (slice(None, None, PT_STRIDE[0]), slice(None, None, PT_STRIDE[1]))
        rows["T"].append(Tt[sl].ravel())
        rows["R"].append(R[sl].ravel())
        rows["E"].append(E[sl].ravel())
        rows["X"].append(X[(slice(None),) + sl].reshape(X.shape[0], -1).T)
    return {k: np.concatenate(v) for k, v in rows.items()}


def frac(T, pred):
    return float(1 - np.var(T - pred) / np.var(T))


def run_alpha(alpha):
    with h5py.File(DATA[alpha], "r") as f:
        md = dict(f["metadata"].attrs)
        n_tr = int(md["n_train"])
        trajs = f["trajectories"][:]
    NX, NY = int(md["NX"]), int(md["NY"])
    trajs = trajs.reshape(trajs.shape[0], trajs.shape[1], NX, NY).astype(np.float64)
    S = Spec(NX, NY, float(md["Lx"]), float(md["Ly"]))
    nu, dt = float(md["nu"]), float(md["dt_snap"])
    mean, std = float(md["normalization_mean"]), float(md["normalization_std"])
    out = {}
    for kc in KCS:
        tr = [collect(trajs[i], S, kc, nu, alpha, dt, mean, std) for i in range(n_tr)]
        va = [collect(trajs[i], S, kc, nu, alpha, dt, mean, std) for i in range(n_tr, trajs.shape[0])]
        cat = lambda d, k: np.concatenate([x[k] for x in d])
        Xtr = np.column_stack([cat(tr, "X"), np.ones(len(cat(tr, "E")))])
        coef, *_ = np.linalg.lstsq(Xtr, cat(tr, "E"), rcond=None)
        Tv, Rv, Ev = cat(va, "T"), cat(va, "R"), cat(va, "E")
        Ehat = np.column_stack([cat(va, "X"), np.ones(len(Ev))]) @ coef
        out[f"k_c={kc:g} (cutoff wavelength {2*np.pi/kc:.2f})"] = {
            "consistency_T_vs_R_plus_E": frac(Tv, Rv + Ev),
            "i_resolved_only": frac(Tv, Rv),
            "ii_resolved_plus_linear_closure": frac(Tv, Rv + Ehat),
            "iii_remainder": 1 - frac(Tv, Rv + Ehat),
            "eddy_forcing_var_over_T_var": float(np.var(Ev) / np.var(Tv)),
            "linear_closure_R2_on_E": frac(Ev, Ehat),
            "n_points_val": int(len(Tv)),
        }
        print(f"[c13] alpha={alpha} k_c={kc}: " + json.dumps({k: round(v, 3) for k, v in
              out[f'k_c={kc:g} (cutoff wavelength {2*np.pi/kc:.2f})'].items() if isinstance(v, float)}), flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {f"alpha={a}": run_alpha(a) for a in (0.05, 0.1)}
    key = [k for k in res["alpha=0.05"] if k.startswith("k_c=1 ")][0]
    ii = res["alpha=0.05"][key]["ii_resolved_plus_linear_closure"]
    rem = res["alpha=0.05"][key]["iii_remainder"]
    res["verdict"] = (("VIABLE" if ii >= 0.8 else "NOT VIABLE with a local linear closure")
                      + f" at cutoff 2pi, alpha=0.05 (ii={ii:.2f})"
                      + ("; closure must be STOCHASTIC (remainder > 0.3)" if rem > 0.3 else ""))
    (OUT / "c13_result.json").write_text(json.dumps(res, indent=2))
    print(res["verdict"], flush=True)


if __name__ == "__main__":
    main()
