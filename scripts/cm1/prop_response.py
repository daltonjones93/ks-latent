"""Propagator interpretation (pre-registration 5j): (1) physical response function, (2) linearisation vs DMD.

1. Perturb a real state by a small surface theta' bump, encode, roll the propagator (halo held at truth), decode, and
   subtract the unperturbed rollout: R(x, k).  Offsets are measured from the gust front (steepest |d theta'_sfc/dx|);
   negative = ahead (the storm moves toward -x).
2. Jacobian of the one-step latent map over the interior sites at z(t0): eigenvalues in physical units, decoded
   leading modes, and an exact-DMD comparison of the data in the same POD subspace.
  python scripts/cm1/prop_response.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/prop_resp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import forced_halo_rollout  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0


def site_mean(x, p):
    return x.reshape(*x.shape[:-1], -1, p).mean(-1)


def front_site(th, lo, hi):
    g = np.abs(np.gradient(th))
    g[:lo] = 0
    g[hi:] = 0
    return int(np.argmax(g))


@torch.no_grad()
def decode_theta(ae, z, ci, sc_th, p, chunk=32):
    out = []
    for i in range(0, z.shape[0], chunk):
        u = ae.decode(z[i:i + chunk])[:, ci].cpu().numpy() * sc_th
        out.append(site_mean(u, p))
    return np.concatenate(out)


def slope(y, t):
    return float(np.polyfit(t, y, 1)[0])


def exp1(ae, prop, z, u0_fn, tcase, ci, sc_th, p, halo, a, front_t):
    """One (storm, t0). Returns list of per-(offset, amp) records and the mean response arrays."""
    P = z.shape[1]
    u0 = u0_fn(tcase)                                         # (C, NX) scaled
    th0 = site_mean(u0[ci].cpu().numpy() * sc_th, p)
    if th0.min() > -a.min_cold:
        return None, "no cold pool"
    xf = front_site(th0, halo, P - halo)
    n = a.lags
    win = z[tcase:tcase + n + 1]
    rec, resp, keys = [], [], []
    ub, meta = [], []
    cell = np.arange(u0.shape[-1])
    for off in a.offsets_km:
        s0 = xf + int(round(off / SITE_KM))
        if s0 - 3 < halo or s0 + 3 >= P - halo:
            continue
        xc = s0 * p + p / 2
        bump = np.exp(-((cell - xc) * (SITE_KM / p)) ** 2 / (2 * a.sigma_km ** 2))
        for amp in a.amps:
            for sg in (+1, -1):
                u = u0.clone()
                for l in range(a.n_lev):
                    u[ci + l] += torch.as_tensor(sg * amp * (1 - l / a.n_lev) * bump / SC[ci + l], dtype=u.dtype)
                ub.append(u); meta.append((off, amp, sg, s0))
    if not ub:
        return None, "edge"
    with torch.no_grad():
        zp0 = ae.encode(torch.stack(ub))
        z00 = ae.encode(u0[None])
        B = zp0.shape[0]
        wb = win[None].expand(B, -1, -1, -1).clone()
        wb[:, 0] = zp0
        w0 = win[None].clone()
        w0[:, 0] = z00[0]
        outp = forced_halo_rollout(prop, wb, n, halo)          # (B, n, P, c)
        out0 = forced_halo_rollout(prop, w0, n, halo)          # (1, n, P, c)
    th_p = np.concatenate([decode_theta(ae, zp0, ci, sc_th, p)[:, None],
                           decode_theta(ae, outp.reshape(-1, *outp.shape[2:]), ci, sc_th, p).reshape(B, n, P)], 1)
    th_0 = np.concatenate([decode_theta(ae, z00, ci, sc_th, p)[:, None],
                           decode_theta(ae, out0.reshape(-1, *out0.shape[2:]), ci, sc_th, p).reshape(1, n, P)], 1)
    R = th_p - th_0                                            # (B, n+1, P)
    kk = np.arange(n + 1) * front_t["dt_min"]
    sites = np.arange(P)
    amp_k = np.sqrt((R ** 2).sum(-1))                          # (B, n+1)
    for b, (off, amp, sg, s0) in enumerate(meta):
        w = np.abs(R[b]) + 1e-12
        cen = (w * sites).sum(-1) / w.sum(-1)
        m = (kk >= 5) & (kk <= 30)
        v = slope(cen[m], kk[m]) * SITE_KM                     # km/min
        rec.append({"off": off, "amp": amp, "sign": sg, "centroid_v": v, "gain15": float(amp_k[b, min(15, n)] / max(amp_k[b, 0], 1e-12)),
                    "gain30": float(amp_k[b, min(30, n)] / max(amp_k[b, 0], 1e-12)),
                    "gain60": float(amp_k[b, n] / max(amp_k[b, 0], 1e-12)), "k0_floor_ratio": float(amp_k[b, 0] / amp)})
    # linearity and amplitude scaling per offset
    lin = {}
    for off in a.offsets_km:
        ix = {(m_[1], m_[2]): i for i, m_ in enumerate(meta) if m_[0] == off}
        if not ix:
            continue
        k15 = min(15, n)
        big, small = max(a.amps), min(a.amps)
        Rp, Rm = R[ix[(big, 1)], k15], R[ix[(big, -1)], k15]
        L = float(np.linalg.norm(Rp + Rm) / max(np.linalg.norm(Rp - Rm), 1e-12))
        S = float((np.linalg.norm(Rp - Rm) / (2 * big)) / max(np.linalg.norm(R[ix[(small, 1)], k15] - R[ix[(small, -1)], k15]) / (2 * small), 1e-12))
        lin[off] = {"L": L, "S": S}
    return {"records": rec, "lin": lin, "xf": xf, "R_mean": {off: np.mean([R[b] * sg for b, (o, am, sg, s0) in enumerate(meta)
                                                                          if o == off and am == max(a.amps)], 0) for off in a.offsets_km if any(m_[0] == off for m_ in meta)},
            "s0": {off: next(m_[3] for m_ in meta if m_[0] == off) for off in a.offsets_km if any(m_[0] == off for m_ in meta)}}, "ok"


def exp2(ae, prop, z, tcase, ci, sc_th, p, halo, a, dt_min):
    P, c = z.shape[1], z.shape[2]
    sl = slice(halo, P - halo)
    z0 = z[tcase].clone()
    v0 = z0[sl].reshape(-1)

    def step(v):
        zz = z0.clone()
        zz = torch.cat([zz[:halo], v.view(P - 2 * halo, c), zz[P - halo:]], 0)
        return prop(zz[None])[0][sl].reshape(-1)
    with torch.no_grad():
        J = torch.func.jacrev(step)(v0).double().numpy()
    lam, V = np.linalg.eig(J)
    sig = np.log(np.abs(lam) + 1e-300) / dt_min
    om = np.angle(lam) / dt_min
    order = np.argsort(-sig)
    out = {"n_dof": int(J.shape[0]), "n_unstable": int((np.abs(lam) > 1).sum()), "max_sigma_per_min": float(sig.max()),
           "efold_min": float(1 / sig.max()) if sig.max() > 0 else None, "modes": []}
    for i in order[:a.n_modes]:
        v = V[:, i]
        eps = 0.1 * np.linalg.norm(v0.numpy()) / np.linalg.norm(v)
        zp = z0.clone()
        zp[sl] += torch.as_tensor(np.real(v * eps), dtype=z0.dtype).view(P - 2 * halo, c)
        d = decode_theta(ae, torch.stack([zp, z0]), ci, sc_th, p)
        f = (d[0] - d[1])[halo:P - halo]
        spec = np.abs(np.fft.rfft(f * np.hanning(len(f))))
        k_idx = int(np.argmax(spec[1:]) + 1)
        kcyc = k_idx / (len(f) * SITE_KM)                      # cycles per km
        w2 = np.abs(v).reshape(P - 2 * halo, c)
        w_site = (w2 ** 2).sum(1)
        pr = float(w_site.sum() ** 2 / (w_site ** 2).sum())
        out["modes"].append({"sigma": float(sig[i]), "omega": float(om[i]), "k_cyc_per_km": float(kcyc),
                             "phase_speed_km_min": float(abs(om[i]) / (2 * np.pi * kcyc)) if kcyc > 0 else None,
                             "part_ratio_sites": pr, "decoded_amp_K": float(np.abs(f).max())})
    # DMD of the data in the same window and POD subspace
    W = a.dmd_window
    X = z[tcase:tcase + W + 1][:, sl].reshape(W + 1, -1).double().numpy().T          # (n, W+1)
    X0, X1 = X[:, :-1], X[:, 1:]
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    r = a.pod_r
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    Ad = Ur.T @ X1 @ Vr @ np.diag(1 / Sr)
    Jh = Ur.T @ J @ Ur
    ld, Wd = np.linalg.eig(Ad)
    Phi = Ur @ Wd
    b = np.linalg.lstsq(Phi, X[:, 0].astype(complex), rcond=None)[0]
    lead = np.argsort(-np.abs(b) * np.linalg.norm(Phi, axis=0))[:4]
    lj = np.linalg.eigvals(Jh)
    rows = []
    for i in lead:
        j = int(np.argmin(np.abs(lj - ld[i])))
        sd, od = np.log(abs(ld[i])) / dt_min, np.angle(ld[i]) / dt_min
        sj, oj = np.log(abs(lj[j])) / dt_min, np.angle(lj[j]) / dt_min
        # an eigenvalue and its conjugate are the same mode: compare |omega|
        rows.append({"dmd_sigma": float(sd), "dmd_omega": float(od), "J_sigma": float(sj), "J_omega": float(oj),
                     "d_sigma": float(sj - sd), "d_omega": float(abs(abs(oj) - abs(od))),
                     "dist": float(abs(lj[j] - ld[i])), "null_dist": float(abs(1 - ld[i]))})
    out["match"] = rows
    out["subspace_energy"] = float((Sr ** 2).sum() / (S ** 2).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--t0", default="120,180,240")
    ap.add_argument("--lags", type=int, default=60)
    ap.add_argument("--amps", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 0.1])
    ap.add_argument("--offsets-km", type=lambda s: [float(x) for x in s.split(",")], default=[-40, -20, -10, -4, 0, 4, 10, 20, 40])
    ap.add_argument("--sigma-km", type=float, default=2.0)
    ap.add_argument("--n-lev", type=int, default=6)
    ap.add_argument("--n-modes", type=int, default=10)
    ap.add_argument("--dmd-window", type=int, default=30)
    ap.add_argument("--pod-r", type=int, default=8)
    ap.add_argument("--min-cold", type=float, default=2.0, help="skip cases whose surface theta' minimum is above -MIN_COLD K")
    ap.add_argument("--skip-exp2", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]; p = ae.cfg.patch
    files = member_files(a.source)
    global SC
    res = {"args": vars(a), "cases": []}
    names = split["test"][: a.n_test]
    for name in names:
        d = ColumnData([load_member(files[name])], norm, dev)
        SC = np.asarray(norm.channel_scale(d.Z), float)
        ci = VARS.index("th") * d.Z
        sc_th = SC[ci]
        dt_min = d.dt / 60.0
        with torch.no_grad():
            z = encode_members(ae, d)[0]
        T = z.shape[0]
        for t_case in [int(x) for x in a.t0.split(",")]:
            if t_case + a.lags + 1 > T or t_case + a.dmd_window + 1 > T:
                continue
            # data front speed over t0..t0+30
            nf = min(30, T - 1 - t_case)
            ths = np.stack([site_mean(d.full(0, t)[0, ci].numpy() * sc_th, p) for t in range(t_case, t_case + nf + 1)])
            fx = np.array([front_site(th, halo, th.shape[0] - halo) for th in ths])
            tt = np.arange(nf + 1) * dt_min
            v_f = slope(fx.astype(float), tt) * SITE_KM
            case = {"member": name, "t0": t_case, "front_speed_km_min": v_f, "front_site0": int(fx[0])}
            r1, why = exp1(ae, prop, z, lambda t: d.full(0, t)[0], t_case, ci, sc_th, p, halo, a, {"dt_min": dt_min})
            if r1 is None:
                case["exp1_skipped"] = why
            else:
                case["exp1"] = {"records": r1["records"], "lin": {str(k): v for k, v in r1["lin"].items()}}
                np.savez_compressed(out / f"R_{name}_{t_case}.npz", **{f"off{int(k)}": v for k, v in r1["R_mean"].items()},
                                    s0=json.dumps(r1["s0"]))
            if not a.skip_exp2:
                try:
                    case["exp2"] = exp2(ae, prop, z, t_case, ci, sc_th, p, halo, a, dt_min)
                except Exception as e:                      # reported, not hidden
                    case["exp2_error"] = repr(e)
            res["cases"].append(case)
            print(f"[resp] {name} t0={t_case}: v_f {v_f:+.3f} km/min, exp1 {'skipped: ' + why if r1 is None else 'ok'}"
                  + (f", exp2 n_unstable {case['exp2']['n_unstable']}, max sigma {case['exp2']['max_sigma_per_min']:.3f}/min" if 'exp2' in case else ""),
                  flush=True)
    json.dump(res, open(out / "prop_response.json", "w"), indent=1)
    print(f"[resp] done in {time.time() - t0:.0f}s, {len(res['cases'])} cases -> {out}")


if __name__ == "__main__":
    main()
