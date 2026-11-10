"""Pre-registration 5l: adjoint sensitivity of the storm's propagation, checked against the paired CM1 runs.

Outcomes at +30 min from the decoded surface theta' (site means): O1 = cold-pool centroid (km), O2 = total cold-pool deficit,
with weight c = softplus(-(theta' + 1 K), beta 2). The gradient of O with respect to the input state (through encoder,
forced-halo propagator steps and decoder) is projected on the 5k bump (sigma 2 km, nlev levels, taper 1 - l/nlev), giving S_O(s):
the linear response of O to a 1 K bump at site s. Truth: the same functionals on the CM1 control and bump branches.
  python scripts/cm1/prop_sensitivity.py --run ~/cm1_runs/ens_A_cpu_k45 --base ~/cm1_cache_P --branches ~/cm1_cache_Q \
      --manifest ~/cm1_bundle/ens_Q/manifest.json --out ~/cm1_da/sens_k45
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import forced_halo_rollout  # noqa: E402
from paired_compare import frame_at, site_mean  # noqa: E402

SITE_KM = 2.0


def outcomes(th):
    """th (P,) site-mean surface theta' in K -> (O1 centroid km, O2 deficit)."""
    c = F.softplus(-(th + 1.0), beta=2.0)
    x = torch.arange(th.shape[0], dtype=th.dtype) * SITE_KM
    return (x * c).sum() / c.sum(), c.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--base", required=True); ap.add_argument("--branches", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lag", type=int, default=30); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    for q in list(ae.parameters()) + list(prop.parameters()):
        q.requires_grad_(False)
    halo, p = cfg["stage_b"]["halo_sites"], ae.cfg.patch
    man = json.loads(Path(a.manifest).expanduser().read_text())
    T0, n = man["t0_s"], a.lag
    sigma_m, nlev = man["sigma_m"], int(man["nlev"])
    res = {"lag": n, "storms": {}}
    for short, info in man["members"].items():
        base = load_member(next(Path(a.base).expanduser().glob(f"*__{short}.npz")))
        br = {nm: load_member(Path(a.branches).expanduser() / f"ens_Q__{b['dir']}.npz") for nm, b in info["branches"].items()}
        dt = base.dt
        d_base = ColumnData([base], norm, dev); Z = d_base.Z
        ci = VARS.index("th") * Z
        sc = np.asarray(norm.channel_scale(Z), float); sc_th = float(sc[ci])
        i0 = frame_at(base, T0)
        ctrl = br["ctrl"]; dctrl = ColumnData([ctrl], norm, dev)
        u0 = d_base.full(0, i0)[0]
        u_ctrl = [u0] + [dctrl.full(0, frame_at(ctrl, T0 + k * dt))[0] for k in range(1, n + 1)]
        with torch.no_grad():
            z_ctrl = ae.encode(torch.stack(u_ctrl))
        P, NX = z_ctrl.shape[1], u0.shape[-1]

        def thsite(zlast):
            return site_mean(ae.decode(zlast[None])[0, ci] * sc_th, p)

        def forward(u):
            zp0 = ae.encode(u[None])
            wb = torch.cat([zp0[:, None], z_ctrl[None, 1:n + 1]], 1)
            return thsite(forced_halo_rollout(prop, wb, n, halo)[0, -1])
        u = u0.clone().requires_grad_(True)
        th = forward(u)
        O1, O2 = outcomes(th)
        g1 = torch.autograd.grad(O1, u, retain_graph=True)[0]
        g2 = torch.autograd.grad(O2, u)[0]
        cells = torch.arange(NX, dtype=torch.float32)
        cen = torch.arange(P, dtype=torch.float32) * p + (p - 1) / 2.0
        G = torch.exp(-0.5 * ((cells[None] - cen[:, None]) * (SITE_KM * 1000.0 / p) / sigma_m) ** 2)   # (P, NX)
        def project(g):
            return sum((1 - l / nlev) / sc[ci + l] * (G @ g[ci + l]) for l in range(nlev)).numpy()
        S1, S2 = project(g1), project(g2)
        # model rollout of the control and CM1 truth functionals
        with torch.no_grad():
            th_c = forward(u0); O1c, O2c = outcomes(th_c)
        th_cm1_c = torch.as_tensor(site_mean(dctrl.full(0, frame_at(ctrl, T0 + n * dt))[0, ci].numpy() * sc_th, p), dtype=torch.float32)
        C1, C2 = outcomes(th_cm1_c)
        info_out = {"front_site": info["front_site"], "S_O1": S1.tolist(), "S_O2": S2.tolist(),
                    "O_model_ctrl": [float(O1c), float(O2c)], "O_cm1_ctrl": [float(C1), float(C2)], "cases": {}}
        for nm, b in info["branches"].items():
            if "var5" not in b:
                continue
            s_idx = int(round((b["var6"] - 3.5) / p))
            mem = br[nm]; dmem = ColumnData([mem], norm, dev)
            th_b = torch.as_tensor(site_mean(dmem.full(0, frame_at(mem, T0 + n * dt))[0, ci].numpy() * sc_th, p), dtype=torch.float32)
            B1, B2 = outcomes(th_b)
            ub = u0.clone()
            bump = np.exp(-0.5 * ((np.arange(NX) - b["var6"]) * (SITE_KM * 1000.0 / p) / sigma_m) ** 2)
            for l in range(nlev):
                ub[ci + l] += torch.as_tensor(b["var5"] * (1 - l / nlev) * bump / sc[ci + l], dtype=ub.dtype)
            with torch.no_grad():
                N1, N2 = outcomes(forward(ub))
            info_out["cases"][nm] = {"off": b["off"], "amp": b["var5"], "site": s_idx,
                                     "pred_O1": float(b["var5"] * S1[s_idx]), "pred_O2": float(b["var5"] * S2[s_idx]),
                                     "cm1_O1": float(B1 - C1), "cm1_O2": float(B2 - C2),
                                     "nl_O1": float(N1 - O1c), "nl_O2": float(N2 - O2c)}
        res["storms"][short] = info_out
        print(f"[sens] {short} done", flush=True)
    json.dump(res, open(out / "sens.json", "w"), indent=1)


if __name__ == "__main__":
    main()
