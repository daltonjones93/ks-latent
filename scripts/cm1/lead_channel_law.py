"""FPe: a closed equation for the leading channels of the ORDERED (nested-dropout) latent.

FPc/FPd asked for a sparse law for all 32 channels and found it dense. Here the question is smaller: with channels
ordered by importance (fpc_nested_stagec: channel 0 alone reconstructs to nMSE 0.30), is there a short equation
for the first k channels in terms of themselves only? Sparse regression (bootstrap STLSQ over whole storms, as in
FP0) of dz_j/dt (centred difference, per minute) for j < k on a library built from the first k channels:
  ode   z_0..z_{k-1} and their quadratic products (a local ODE per site: no neighbours)
  pde   plus d1 z, d2 z (centred differences over 2 km sites) and products with z
  pde3, pde4  derivatives up to third / fourth order (dispersion, hyperdiffusion)
  +dd, +cubic, +mixed  derivative products, cubic terms, d2x(dz/dt) (BBM-type implicit law; ks_latent.cm1.leadlaw)
Reported for k = 1, 2, 4, 8 and each library, on held-out test storms:
  - tendency R^2 (one step), number of terms, the selected terms of channel 0 with selection frequency
  - CLOSED rollouts of the fitted k-channel system (Heun, 1 min, edge sites held at the start state, tendency
    capped like the law), from encoded test states at 60 / 120 / 180 min: RMSE of channel j against
    persistence of z_j and against the trained model's own rollout of all 32 channels (the dense law)
  - decoded skill of the k-channel rollout: decode with channels >= k zeroed (the nested decoder's training
    condition) and score nMSE against the truth, against the same decode of persistence
  - what the leading channels are: correlations with site-mean surface theta', column-max w, column condensate,
    low-level u, from the truth columns
With --time-orders 2 the law is second order in time (ks_latent.cm1.leadlaw): d2z/dt2 = F(z, dz/dt, ...).
  python scripts/cm1/lead_channel_law.py --run ~/cm1_runs/fpc_nested_stagec --source cache:A+A2+B --out ~/cm1_da/fpe
"""
from __future__ import annotations

import argparse
import itertools
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
from ks_latent.cm1.leadlaw import backward_velocity, fit, is_mixed, r2, refit_rollout, rollout, training_rows, with_mixed  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import _forecast  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--kinds", nargs="+", default=["ode", "pde"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    ap.add_argument("--n-boot", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--frames", default="30:330", help="frame range used for fitting (1-min frames)")
    ap.add_argument("--starts", type=int, nargs="+", default=[60, 120, 180])
    ap.add_argument("--leads", type=int, nargs="+", default=[5, 15, 30, 60])
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--time-orders", type=int, nargs="+", default=[1], choices=[1, 2],
                    help="2: target d2z/dt2 with dz/dt of the kept channels in the library (waves, memory)")
    ap.add_argument("--refit-rollout", type=int, default=0,
                    help="> 0: keep the STLSQ support and refit its coefficients on closed rollouts up to this many minutes")
    ap.add_argument("--refit-iters", type=int, default=1500)
    ap.add_argument("--mixed-l2", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                    help="+mixed kinds: fixed length scales l2 (sites^2) of (1 - l2 d2x) dz/dt, each fitted and scored")
    ap.add_argument("--t-smooth", type=float, default=0.0, help="Gaussian smoothing of z in time (frames) before differencing")
    ap.add_argument("--x-smooth", type=float, default=0.0,
                    help="Gaussian smoothing of the latent field over sites (sites) before everything; tests whether "
                         "d3/d4 terms act on a ~1-site front as a lagged gradient. Persistence and the full-model reference "
                         "are then also computed on the smoothed field: compare only within a run.")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))

    def enc(names):
        zs, us = [], []
        for n in names:
            d = ColumnData([load_member(files[n])], norm, dev)
            with torch.no_grad():
                zs.append(encode_members(ae, d)[0].cpu().numpy().astype(np.float64))
            us.append(d)
        return zs, us
    ztr, _ = enc(split["train"][: a.n_train])
    zte, dte = enc(split["test"][: a.n_test])
    if a.x_smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        ztr = [gaussian_filter1d(z, a.x_smooth, axis=1, mode="nearest") for z in ztr]
        zte = [gaussian_filter1d(z, a.x_smooth, axis=1, mode="nearest") for z in zte]
    c = ztr[0].shape[-1]
    print(f"[FPe] {Path(a.run).name}: {len(ztr)} training / {len(zte)} test storms, c {c}, halo {halo} [{time.time() - t_start:.0f}s]",
          flush=True)
    res = {"run": a.run, "fits": [], "meaning": {}}

    # what the leading channels are (truth diagnostics per site, test storms)
    Zl, p = dte[0].Z, ae.cfg.patch
    diag = {k: [] for k in ("theta_sfc", "w_max", "cond_col", "u_low")}
    zl = []
    for z, d in zip(zte, dte):
        for t in range(f0, min(f1, z.shape[0]), 10):
            u = d.full(0, t)[0].cpu().numpy()
            sc = np.repeat(np.asarray(norm.scale, float), Zl)[:, None]
            u = u * sc
            g = lambda v, lev: u[VARS.index(v) * Zl + lev]
            th = g("th", 0); w = u[VARS.index("w") * Zl: (VARS.index("w") + 1) * Zl].max(0)
            cond = (u[VARS.index("qliq") * Zl: (VARS.index("qliq") + 1) * Zl] ** 2).sum(0) + \
                   (u[VARS.index("qice") * Zl: (VARS.index("qice") + 1) * Zl] ** 2).sum(0)
            ul = u[VARS.index("u") * Zl: VARS.index("u") * Zl + 5].mean(0)
            for key, v in (("theta_sfc", th), ("w_max", w), ("cond_col", cond), ("u_low", ul)):
                diag[key].append(v.reshape(-1, p).mean(1))
            zl.append(z[t])
    Zs = np.concatenate(zl)
    for j in range(min(8, c)):
        res["meaning"][f"z{j}"] = {k: float(np.corrcoef(Zs[:, j], np.concatenate(v))[0, 1]) for k, v in diag.items()}
    print("[FPe] correlation of the leading channels with truth diagnostics (site means):")
    for j, m in list(res["meaning"].items())[:4]:
        print(f"   {j}: " + "  ".join(f"{k} {v:+.2f}" for k, v in m.items()))

    rng = np.random.default_rng(0)
    # reference: the trained model's own rollout of all channels, from the same encoded states
    ref = {}
    with torch.no_grad():
        for si, z in enumerate(zte):
            for t0 in a.starts:
                if t0 + max(a.leads) >= z.shape[0]:
                    continue
                Zt = torch.as_tensor(z[t0], dtype=torch.float32)[None]
                zz, traj = Zt, []
                for s in range(max(a.leads)):
                    zz = _forecast(prop, zz, torch.as_tensor(z, dtype=torch.float32), t0 + s, 1, halo, 0.0, None, hold=t0)
                    traj.append(zz[0].numpy())
                ref[(si, t0)] = np.stack(traj)
    for k in a.ks:
        if k > c:
            continue
        combos = [(o, kd, l2) for o in a.time_orders for kd in a.kinds
                  for l2 in (a.mixed_l2 if is_mixed(kd) else [0.0]) if not (o == 2 and is_mixed(kd))]
        for order, kind, l2 in combos:
            X, Y, G = [], [], []
            for si, z in enumerate(ztr):
                Xi, Yi, names = training_rows(z[f0 - 1: f1 + 1, :, :k], kind, order, halo, a.t_smooth, l2)
                X.append(Xi); Y.append(Yi); G.append(np.repeat(si, len(Xi)))
            X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
            if len(X) > a.max_rows:
                ii = rng.choice(len(X), a.max_rows, replace=False); X, Y, G = X[ii], Y[ii], G[ii]
            Xt, Yt = [], []                          # held-out targets (dz/dt, or d2z/dt2 for order 2)
            for z in zte:
                Xi, Yi, _ = training_rows(z[f0 - 1: f1 + 1, :, :k], kind, order, halo, a.t_smooth, l2)
                Xt.append(Xi); Yt.append(Yi)
            Xt, Yt = np.concatenate(Xt), np.concatenate(Yt)
            for thr in a.thresholds:
                xi, freq = fit(X, Y, G, thr, n_boot=a.n_boot)
                r2t = [r2(Yt[:, j], Xt @ xi[:, j]) for j in range(k)]      # +mixed: R^2 of the (1 - l2 d2x) dz/dt target
                tnames = names
                if is_mixed(kind):
                    xi, freq, tnames = with_mixed(xi, freq, names, l2)
                xi_rt = None
                if a.refit_rollout:
                    xi_rt = refit_rollout(xi, [z[f0 - 2: f1 + 1, :, :k] for z in ztr], kind, halo, order,
                                          k_max=a.refit_rollout, iters=a.refit_iters, cap=a.cap, log=print)
                errs = {L: {"law": [], "pers": [], "ref": [], "dec_law": [], "dec_pers": [], "law_rt": [], "dec_law_rt": []}
                        for L in a.leads}
                for si, z in enumerate(zte):
                    for t0 in a.starts:
                        if (si, t0) not in ref:
                            continue
                        zk = z[:, :, :k]
                        v0 = backward_velocity(zk, t0) if order == 2 else None
                        traj = rollout(xi, zk[t0], max(a.leads), kind, a.cap, halo, order, v0)
                        traj_rt = None if xi_rt is None else rollout(xi_rt, zk[t0], max(a.leads), kind, a.cap, halo, order, v0)
                        for L in a.leads:
                            tru = z[t0 + L, halo:-halo, 0]
                            errs[L]["law"].append(np.mean((traj[L - 1, halo:-halo, 0] - tru) ** 2))
                            errs[L]["pers"].append(np.mean((z[t0, halo:-halo, 0] - tru) ** 2))
                            errs[L]["ref"].append(np.mean((ref[(si, t0)][L - 1, halo:-halo, 0] - tru) ** 2))
                            if traj_rt is not None:
                                errs[L]["law_rt"].append(np.mean((traj_rt[L - 1, halo:-halo, 0] - tru) ** 2))
                            u_true = dte[si].full(0, t0 + L)[0]
                            cases = [("dec_law", traj[L - 1]), ("dec_pers", zk[t0])]
                            if traj_rt is not None:
                                cases.append(("dec_law_rt", traj_rt[L - 1]))
                            for key, zz in cases:
                                if not np.isfinite(zz).all():
                                    errs[L][key].append(np.nan); continue
                                zf = np.zeros_like(z[t0]); zf[:, :k] = zz
                                with torch.no_grad():
                                    dec = ae.decode(torch.as_tensor(zf, dtype=torch.float32)[None])[0]
                                errs[L][key].append(float(((dec - u_true) ** 2).mean()))
                skill = {L: {key: (float(np.nanmean(v)) if len(v) else None) for key, v in e.items()} for L, e in errs.items()}
                n_terms = int((xi != 0).sum())
                top0 = sorted([(tnames[i], float(xi[i, 0]), float(freq[i, 0])) for i in np.flatnonzero(xi[:, 0])],
                              key=lambda t: -abs(t[1]))[:12]
                rec = {"k": k, "kind": kind, "l2": l2, "time_order": order, "threshold": thr, "n_terms": n_terms,
                       "n_terms_z0": int((xi[:, 0] != 0).sum()), "r2_test": r2t, "skill": skill, "z0_terms": top0,
                       "library_size": X.shape[1],
                       "z0_terms_refit": None if xi_rt is None else sorted(
                           [(tnames[i], float(xi_rt[i, 0])) for i in np.flatnonzero(xi_rt[:, 0])], key=lambda t: -abs(t[1]))[:12],
                       "blowups": int(sum(np.isnan(v).sum() for v in
                                                                      [np.array(errs[max(a.leads)]["law"])]))}
                res["fits"].append(rec)
                s30 = skill.get(30, skill[max(skill)])
                lhs = "dz0/dt" if order == 1 else "d2z0/dt2"
                print(f"[FPe] k {k} {kind}{f' l2 {l2:g}' if is_mixed(kind) else ''} order {order} thr {thr}: {n_terms} terms ({rec['n_terms_z0']} for z0 of {X.shape[1]}), "
                      f"test R^2 z0 {r2t[0]:.3f}" + (f" mean {np.mean(r2t):.3f}" if k > 1 else "")
                      + f" | z0 MSE at 30 min: law {s30['law']:.4f} persistence {s30['pers']:.4f} full model {s30['ref']:.4f}"
                      + f" | decoded (first k ch) law {s30['dec_law']:.4f} persistence {s30['dec_pers']:.4f}"
                      + (f" | REFIT law {s30['law_rt']:.4f} decoded {s30['dec_law_rt']:.4f}" if xi_rt is not None else "")
                      + f" | blow-ups {rec['blowups']} [{time.time() - t_start:.0f}s]", flush=True)
                if xi_rt is not None:
                    print("      refit: " + " ".join(f"{v:+.3g} {n}" for n, v in rec["z0_terms_refit"][:10]), flush=True)
                print(f"      {lhs} = " + " ".join(f"{v:+.3g} {n} ({f:.1f})" for n, v, f in top0[:10]), flush=True)
                (out / "fpe.json").write_text(json.dumps(res, indent=1))
    from dataclasses import make_dataclass
    Args = make_dataclass("FPeArgs", [(k, object, tuple(v) if isinstance(v, list) else v) for k, v in vars(a).items()])
    write_provenance(out / "fpe.json", config=Args(), seed=0, device=str(dev), wall_time_s=time.time() - t_start)
    print(f"[FPe] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
