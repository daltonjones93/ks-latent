"""FPe: a closed equation for the leading channels of the ORDERED (nested-dropout) latent.

FPc/FPd asked for a sparse law for all 32 channels and found it dense. Here the question is smaller: with channels
ordered by importance (fpc_nested_stagec: channel 0 alone reconstructs to nMSE 0.30), is there a short equation
for the first k channels in terms of themselves only? Sparse regression (bootstrap STLSQ over whole storms, as in
FP0) of dz_j/dt (centred difference, per minute) for j < k on a library built from the first k channels:
  ode   z_0..z_{k-1} and their quadratic products (a local ODE per site: no neighbours)
  pde   plus d1 z, d2 z (centred differences over 2 km sites) and products with z
Reported for k = 1, 2, 4, 8 and each library, on held-out test storms:
  - tendency R^2 (one step), number of terms, the selected terms of channel 0 with selection frequency
  - CLOSED rollouts of the fitted k-channel system (Heun, 1 min, edge sites held at the start state, tendency
    capped like the law), from encoded test states at 60 / 120 / 180 min: RMSE of channel j against
    persistence of z_j and against the trained model's own rollout of all 32 channels (the dense law)
  - decoded skill of the k-channel rollout: decode with channels >= k zeroed (the nested decoder's training
    condition) and score nMSE against the truth, against the same decode of persistence
  - what the leading channels are: correlations with site-mean surface theta', column-max w, column condensate,
    low-level u, from the truth columns
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
from ks_latent.cm1.fieldpde import bootstrap_stlsq, r2  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import _forecast  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def derivs(z: np.ndarray):
    """z (..., P, k) -> d1, d2 over sites (interior; edges replicated), per site spacing."""
    zp = np.concatenate([z[..., :1, :], z, z[..., -1:, :]], axis=-2)
    return 0.5 * (zp[..., 2:, :] - zp[..., :-2, :]), zp[..., 2:, :] - 2 * z + zp[..., :-2, :]


def library(z: np.ndarray, kind: str):
    """z (..., P, k) -> features (..., P, F) and their names."""
    k = z.shape[-1]
    lin, names = [z], [f"z{j}" for j in range(k)]
    if kind == "pde":
        d1, d2 = derivs(z)
        lin += [d1, d2]
        names += [f"d1z{j}" for j in range(k)] + [f"d2z{j}" for j in range(k)]
    L = np.concatenate(lin, -1)
    nz = k
    quad, qn = [], []
    for a, b in itertools.combinations_with_replacement(range(L.shape[-1]), 2):
        if a >= nz and b >= nz:          # no products of two derivatives
            continue
        quad.append(L[..., a] * L[..., b]); qn.append(f"{names[a]}*{names[b]}")
    F = np.concatenate([np.ones(z.shape[:-1] + (1,)), L, np.stack(quad, -1)], -1)
    return F, ["1"] + names + qn


def tendency(xi: np.ndarray, z: np.ndarray, kind: str, cap: float) -> np.ndarray:
    F, _ = library(z, kind)
    f = F @ xi                                            # (..., P, k)
    return cap * np.tanh(f / cap) if cap else f


def rollout(xi, z0, steps, kind, cap, halo):
    z, out = z0.copy(), []
    edge = z0.copy()
    for _ in range(steps):
        k1 = tendency(xi, z, kind, cap)
        k2 = tendency(xi, z + k1, kind, cap)
        z = z + 0.5 * (k1 + k2)
        z[:halo], z[-halo:] = edge[:halo], edge[-halo:]
        if not np.isfinite(z).all():
            out.append(np.full_like(z, np.nan)); break
        out.append(z.copy())
    while len(out) < steps:
        out.append(np.full_like(z0, np.nan))
    return np.stack(out)


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
        for kind in a.kinds:
            X, Y, G = [], [], []
            for si, z in enumerate(ztr):
                zz = z[f0 - 1: f1 + 1, :, :k]
                dz = 0.5 * (zz[2:] - zz[:-2])                               # per minute
                F, names = library(zz[1:-1], kind)
                F, dz = F[:, halo:-halo], dz[:, halo:-halo]
                X.append(F.reshape(-1, F.shape[-1])); Y.append(dz.reshape(-1, k))
                G.append(np.repeat(si, X[-1].shape[0]))
            X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
            if len(X) > a.max_rows:
                ii = rng.choice(len(X), a.max_rows, replace=False); X, Y, G = X[ii], Y[ii], G[ii]
            # held-out tendencies
            Xt, Yt = [], []
            for z in zte:
                zz = z[f0 - 1: f1 + 1, :, :k]
                F, _ = library(zz[1:-1], kind)
                Xt.append(F[:, halo:-halo].reshape(-1, F.shape[-1]))
                Yt.append((0.5 * (zz[2:] - zz[:-2]))[:, halo:-halo].reshape(-1, k))
            Xt, Yt = np.concatenate(Xt), np.concatenate(Yt)
            for thr in a.thresholds:
                xi = np.zeros((X.shape[1], k)); freq = np.zeros_like(xi)
                mx = X[:, 1:].mean(0)
                for j in range(k):        # centred fit (the constant column has no spread); intercept from the means
                    b = bootstrap_stlsq(X[:, 1:] - mx, Y[:, j] - Y[:, j].mean(), G, thr, n_boot=a.n_boot)
                    xi[1:, j], freq[1:, j] = b["full"], b["freq"]
                    xi[0, j], freq[0, j] = Y[:, j].mean() - mx @ b["full"], 1.0
                r2t = [r2(Yt[:, j], Xt @ xi[:, j]) for j in range(k)]
                # closed rollouts
                errs = {L: {"law": [], "pers": [], "ref": [], "dec_law": [], "dec_pers": []} for L in a.leads}
                for si, z in enumerate(zte):
                    for t0 in a.starts:
                        if (si, t0) not in ref:
                            continue
                        traj = rollout(xi, z[t0, :, :k], max(a.leads), kind, a.cap, halo)
                        for L in a.leads:
                            tru = z[t0 + L, halo:-halo, 0]
                            errs[L]["law"].append(np.mean((traj[L - 1, halo:-halo, 0] - tru) ** 2))
                            errs[L]["pers"].append(np.mean((z[t0, halo:-halo, 0] - tru) ** 2))
                            errs[L]["ref"].append(np.mean((ref[(si, t0)][L - 1, halo:-halo, 0] - tru) ** 2))
                            u_true = dte[si].full(0, t0 + L)[0]
                            for key, zk in (("dec_law", traj[L - 1]), ("dec_pers", z[t0, :, :k])):
                                if not np.isfinite(zk).all():
                                    errs[L][key].append(np.nan); continue
                                zf = np.zeros_like(z[t0]); zf[:, :k] = zk
                                with torch.no_grad():
                                    dec = ae.decode(torch.as_tensor(zf, dtype=torch.float32)[None])[0]
                                errs[L][key].append(float(((dec - u_true) ** 2).mean()))
                skill = {L: {key: float(np.nanmean(v)) for key, v in e.items()} for L, e in errs.items()}
                n_terms = int((xi != 0).sum())
                top0 = sorted([(names[i], float(xi[i, 0]), float(freq[i, 0])) for i in np.flatnonzero(xi[:, 0])],
                              key=lambda t: -abs(t[1]))[:12]
                rec = {"k": k, "kind": kind, "threshold": thr, "n_terms": n_terms,
                       "n_terms_z0": int((xi[:, 0] != 0).sum()), "r2_test": r2t, "skill": skill, "z0_terms": top0,
                       "library_size": X.shape[1]}
                res["fits"].append(rec)
                s30 = skill.get(30, skill[max(skill)])
                print(f"[FPe] k {k} {kind} thr {thr}: {n_terms} terms ({rec['n_terms_z0']} for z0 of {X.shape[1]}), "
                      f"test R^2 z0 {r2t[0]:.3f}" + (f" mean {np.mean(r2t):.3f}" if k > 1 else "")
                      + f" | z0 MSE at 30 min: law {s30['law']:.4f} persistence {s30['pers']:.4f} full model {s30['ref']:.4f}"
                      + f" | decoded (first k ch) law {s30['dec_law']:.4f} persistence {s30['dec_pers']:.4f} [{time.time() - t_start:.0f}s]",
                      flush=True)
                print("      dz0/dt = " + " ".join(f"{v:+.3g} {n} ({f:.1f})" for n, v, f in top0[:8]), flush=True)
                (out / "fpe.json").write_text(json.dumps(res, indent=1))
    from dataclasses import make_dataclass
    Args = make_dataclass("FPeArgs", [(k, object, tuple(v) if isinstance(v, list) else v) for k, v in vars(a).items()])
    write_provenance(out / "fpe.json", config=Args(), seed=0, device=str(dev), wall_time_s=time.time() - t_start)
    print(f"[FPe] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
