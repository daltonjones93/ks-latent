"""Pre-registration 5k: the propagator's response and growth against paired CM1 runs.

Inputs: the stage-1 base members (cache, frames to 7260 s), the branch members (cache of ens_Q, restarts at 7200 s run
to 10800 s) and ens_Q/manifest.json (bump centres). For each storm:
  response  R_CM1(k) = surface theta' (site means) of a bump branch minus the control at lag k min.
            R_model(k): the SAME bump added to the control state at t0 (theta' channels, lowest nlev levels, taper
            1-l/nlev, centre var6 in 0-based cells, sigma var10 m), encoded, rolled with the control's latents as
            the halo, decoded, minus the model's own unperturbed rollout. Metrics at k = 15/30/60: spatial correlation,
            norm ratio, centroid velocity (k = 5-30).
  growth    D(k) = rms difference over ALL normalised channels of tiny-field branch minus control (CM1), growth rates
            over 5-15 / 15-30 / 30-60 min from the slope of ln D; the model: (i) the same for a uniform [-eps, eps] theta'
            field encoded and rolled; (ii) the leading finite-time Lyapunov exponent of the latent one-step map along
            the control rollout (tangent propagation with renormalisation, halo tangent zero).
  python scripts/cm1/paired_compare.py --run ~/cm1_runs/fpc_nested_stagec --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
      --manifest ~/cm1_bundle/ens_Q/manifest.json --out ~/cm1_da/paired
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import forced_halo_rollout  # noqa: E402

SITE_KM = 2.0


def site_mean(x, p):
    return x.reshape(*x.shape[:-1], -1, p).mean(-1)


def frame_at(m, t_s):
    i = int(np.argmin(np.abs(m.t - t_s)))
    if abs(m.t[i] - t_s) > 1e-6:
        raise ValueError(f"{m.name}: no frame at {t_s} s (nearest {m.t[i]})")
    return i


def centroid_speed(R, kk, lo=5, hi=30):
    w = np.abs(R) + 1e-12
    cen = (w * np.arange(R.shape[-1])).sum(-1) / w.sum(-1)
    m = (kk >= lo) & (kk <= hi)
    return float(np.polyfit(kk[m], cen[m], 1)[0] * SITE_KM)


def growth_rates(D, kk, windows=((5, 15), (15, 30), (30, 60))):
    out = {}
    for lo, hi in windows:
        m = (kk >= lo) & (kk <= hi) & (D > 0)
        out[f"{lo}-{hi}"] = float(np.polyfit(kk[m], np.log(D[m]), 1)[0]) if m.sum() >= 3 else None
    return out


@torch.no_grad()
def dec_all(ae, z, chunk=8):
    return torch.cat([ae.decode(z[i:i + chunk]) for i in range(0, z.shape[0], chunk)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--base", required=True); ap.add_argument("--branches", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lags", type=int, default=60); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-lyap-vec", type=int, default=4)
    a = ap.parse_args()
    t0w = time.time(); torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo, p = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    man = json.loads(Path(a.manifest).expanduser().read_text())
    T0, n = man["t0_s"], a.lags
    sigma_m, nlev = man["sigma_m"], int(man["nlev"])
    res = {"manifest": man, "storms": {}}
    for short, info in man["members"].items():
        base = load_member(next(Path(a.base).expanduser().glob(f"*__{short}.npz")))
        br = {nm: load_member(Path(a.branches).expanduser() / f"ens_Q__{b['dir']}.npz") for nm, b in info["branches"].items()}
        dt = base.dt
        d_base = ColumnData([base], norm, dev)
        Z = d_base.Z; ci = VARS.index("th") * Z
        sc = np.asarray(norm.channel_scale(Z), float); sc_th = sc[ci]
        i0 = frame_at(base, T0)

        def col(mem, idx_s):
            return ColumnData([mem], norm, dev).full(0, frame_at(mem, idx_s))[0]
        # control trajectory frames (scaled, (C,NX)) for lags 0..n
        ctrl = br["ctrl"]
        dctrl = ColumnData([ctrl], norm, dev)
        u_ctrl = [d_base.full(0, i0)[0]] + [dctrl.full(0, frame_at(ctrl, T0 + k * dt))[0] for k in range(1, n + 1)]
        with torch.no_grad():
            z_ctrl = ae.encode(torch.stack(u_ctrl))                                  # (n+1, P, c)
        P = z_ctrl.shape[1]
        th_ctrl = np.stack([site_mean(u[ci].numpy() * sc_th, p) for u in u_ctrl])    # (n+1, P)
        kk = np.arange(n + 1) * dt / 60.0
        S = {"front_site": info["front_site"], "response": {}, "growth": {}}
        cell = np.arange(u_ctrl[0].shape[-1])
        for nm, b in info["branches"].items():
            if "var5" not in b:
                continue
            mem = br[nm]; dmem = ColumnData([mem], norm, dev)
            th_b = np.stack([site_mean(u_ctrl[0][ci].numpy() * sc_th, p)] + [
                site_mean(dmem.full(0, frame_at(mem, T0 + k * dt))[0, ci].numpy() * sc_th, p) for k in range(1, n + 1)])
            R_cm1 = th_b - th_ctrl; R_cm1[0] = 0.0
            # model: same bump on the control state at t0
            u = u_ctrl[0].clone()
            bump = np.exp(-0.5 * ((cell - b["var6"]) * (SITE_KM / p * 1000.0) / sigma_m) ** 2)
            for l in range(nlev):
                u[ci + l] += torch.as_tensor(b["var5"] * (1 - l / nlev) * bump / sc[ci + l], dtype=u.dtype)
            with torch.no_grad():
                zp0 = ae.encode(u[None]); z00 = z_ctrl[:1]
                wb = z_ctrl[None].clone(); wb[:, 0] = zp0
                w0 = z_ctrl[None].clone()
                rp = forced_halo_rollout(prop, wb, n, halo)[0]; r0 = forced_halo_rollout(prop, w0, n, halo)[0]
                zp = torch.cat([zp0, rp]); z0 = torch.cat([z00, r0])
                thm = lambda z: np.stack([site_mean(ae.decode(z[i:i + 1])[0, ci].numpy() * sc_th, p) for i in range(z.shape[0])])
                R_mod = thm(zp) - thm(z0)
            rec = {"off": b["off"], "amp": b["var5"], "floor_ratio": float(np.linalg.norm(R_mod[0]) / max(abs(b["var5"]) * np.sqrt(((bump * 1.0).reshape(-1, p).mean(1) ** 2).sum()), 1e-12))}
            for k in (15, 30, 60):
                k = min(k, n)
                a_, b_ = R_mod[k], R_cm1[k]
                rec[f"corr{k}"] = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else None
                rec[f"ratio{k}"] = float(np.linalg.norm(a_) / max(np.linalg.norm(b_), 1e-12))
            rec["v_model"], rec["v_cm1"] = centroid_speed(R_mod, kk), centroid_speed(R_cm1, kk)
            rec["norm_cm1"] = {str(k): float(np.linalg.norm(R_cm1[min(k, n)])) for k in (5, 15, 30, 60)}
            S["response"][nm] = rec
            np.savez_compressed(out / f"R_{short}_{nm}.npz", R_cm1=R_cm1, R_mod=R_mod)
        # linearity of CM1's own response
        r = S["response"]
        def cm1R(nm): return np.load(out / f"R_{short}_{nm}.npz")["R_cm1"][min(15, n)]
        keys = {k: v for k, v in info["branches"].items() if "var5" in v}
        pos = next((k for k, v in keys.items() if v["off"] == 0 and v["var5"] > 0.2), None)
        neg = next((k for k, v in keys.items() if v["off"] == 0 and v["var5"] < 0), None)
        small = next((k for k, v in keys.items() if v["off"] == 0 and 0 < v["var5"] < 0.2), None)
        if pos and neg:
            S["cm1_linearity_L15"] = float(np.linalg.norm(cm1R(pos) + cm1R(neg)) / max(np.linalg.norm(cm1R(pos) - cm1R(neg)), 1e-12))
        if pos and small:
            S["cm1_scaling_S15"] = float((np.linalg.norm(cm1R(pos)) / keys[pos]["var5"]) / max(np.linalg.norm(cm1R(small)) / keys[small]["var5"], 1e-12))
        # growth: all normalised channels
        for nm in ("tiny1e-3", "tiny1e-4"):
            mem = br[nm]; dmem = ColumnData([mem], norm, dev)
            D = np.zeros(n + 1)
            for k in range(1, n + 1):
                diff = dmem.full(0, frame_at(mem, T0 + k * dt))[0] - u_ctrl[k]
                D[k] = float(torch.sqrt((diff ** 2).mean()))
            S["growth"][f"cm1_{nm}"] = {"D": D.tolist(), "rates": growth_rates(D, kk)}
        for eps in (1e-3, 1e-4):
            g = torch.Generator().manual_seed(0)
            u = u_ctrl[0].clone()
            for l in range(Z):
                u[ci + l] += torch.as_tensor((2 * torch.rand(u.shape[-1], generator=g) - 1) * eps / sc[ci + l], dtype=u.dtype)
            with torch.no_grad():
                zp0 = ae.encode(u[None]); wb = z_ctrl[None].clone(); wb[:, 0] = zp0
                zp = torch.cat([zp0, forced_halo_rollout(prop, wb, n, halo)[0]])
                w0 = z_ctrl[None].clone(); z0 = torch.cat([z_ctrl[:1], forced_halo_rollout(prop, w0, n, halo)[0]])
                D = np.array([float(torch.sqrt(((dec_all(ae, zp[i:i + 1]) - dec_all(ae, z0[i:i + 1])) ** 2).mean())) for i in range(n + 1)])
            S["growth"][f"model_tiny{eps:g}"] = {"D": D.tolist(), "rates": growth_rates(D, kk)}
        # leading finite-time Lyapunov exponent of the latent one-step map along the control rollout
        sl = slice(halo, P - halo); c = z_ctrl.shape[2]
        gen = torch.Generator().manual_seed(1)
        V = torch.randn(a.n_lyap_vec, (P - 2 * halo) * c, generator=gen)
        V /= V.norm(dim=1, keepdim=True)
        logs = []
        z = z_ctrl[0].clone()
        for k in range(n):
            zh = z_ctrl[k]
            def f(v, z=z, zh=zh):
                zz = torch.cat([zh[:halo], v.view(P - 2 * halo, c), zh[P - halo:]], 0)
                return prop(zz[None])[0][sl].reshape(-1)
            v0 = z[sl].reshape(-1)
            with torch.no_grad():
                Jv = torch.stack([torch.func.jvp(f, (v0,), (V[i],))[1] for i in range(V.shape[0])])
                Qm, Rm = torch.linalg.qr(Jv.T)
                znew = f(v0).view(P - 2 * halo, c)
            logs.append(torch.log(torch.abs(torch.diagonal(Rm))).numpy())
            V = Qm.T.contiguous()
            z = torch.cat([z_ctrl[k + 1][:halo], znew, z_ctrl[k + 1][P - halo:]], 0)
        logs = np.array(logs)                                  # (n, nvec) per-step log growth (per frame)
        rate_t = logs[:, 0] / (dt / 60.0)                       # per minute, leading direction (orthonormal iteration with QR)
        kk1 = np.arange(1, n + 1) * dt / 60.0
        S["lyap_leading_per_min"] = {f"{lo}-{hi}": float(rate_t[(kk1 > lo) & (kk1 <= hi)].mean()) for lo, hi in ((0, 15), (15, 30), (30, 60))}
        S["lyap_vec_mean_per_min"] = [float(x) for x in (logs.mean(0) / (dt / 60.0))]
        res["storms"][short] = S
        print(f"[paired] {short} done ({time.time() - t0w:.0f}s)", flush=True)
    json.dump(res, open(out / "paired.json", "w"), indent=1)
    print("[paired] done")


if __name__ == "__main__":
    main()
