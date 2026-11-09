"""FPg-P: read the learned n-channel head (LatentLaw.head) in physical terms.

(1) Which physical quantity does each leading channel carry? Correlation (over training storms, every 5th frame, site means over
    the 2 km site) of z_i with: theta'_sfc, max-over-levels w, u'_sfc, u'_top, column condensate, q_v at the surface.
(2) The head's channel-0 equation dz0/dt = g_0(z_{0..n-1}): each term's contribution, as the standard deviation of coef x feature
    relative to the standard deviation of g_0 (test storms), the R2 of the z0-only terms against g_0 (no refit), and the z0-only
    coefficients in theta' units with the FPe-U gauge map (theta' = s z0 + b, s geometric slope):
        c(theta') = c0 + c1 theta' km/min (d theta'/dt = -c d_x theta' + ...), nu m^2/s, growth a0 per min.
  python scripts/cm1/law_head_physical.py --run ~/cm1_runs/fpg_n4_w01_stagec --source cache:A+A2+B --out DIR
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
from ks_latent.cm1.models import _d1_d2  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0
NU_UNIT = (SITE_KM * 1000.0) ** 2 / 60.0


def physical_scalars(d, norm, p, t):
    u = d.full(0, t)[0].cpu().numpy() * np.repeat(np.asarray(norm.scale, float), d.Z)[:, None]
    Zl = d.Z
    g = lambda v: u[VARS.index(v) * Zl: (VARS.index(v) + 1) * Zl]
    site = lambda a: a.reshape(-1, p).mean(1)
    cond = (g("qliq") ** 2).sum(0) + (g("qice") ** 2).sum(0)
    return {"theta_sfc": site(g("th")[0]), "w_max": site(g("w").max(0)), "w_min": site(g("w").min(0)),
            "u_sfc": site(g("u")[0]), "u_top": site(g("u")[-1]), "condensate": site(cond), "qv_sfc": site(g("qv")[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", default="30:330")
    a = ap.parse_args()
    t0 = time.time(); torch.set_num_threads(4)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    prop.eval()
    law = prop.law
    if law.head is None:
        raise ValueError("run has no head (--law-head-n)")
    n, h = law.head_n, law.head
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    f0, f1 = map(int, a.frames.split(":"))
    p = ae.cfg.patch

    def load(names, phys):
        Z, S = [], []
        for nm in names:
            d = ColumnData([load_member(files[nm])], norm, dev)
            with torch.no_grad():
                z = encode_members(ae, d)[0].cpu().numpy().astype(np.float64)
            Z.append(z)
            if phys:
                for t in range(f0, min(f1, z.shape[0] - 2), 5):
                    s = physical_scalars(d, norm, p, t)
                    S.append((z[t, :, :n], s))
        return Z, S
    ztr, S = load(split["train"][: a.n_train], True)
    zte, _ = load(split["test"][: a.n_test], False)
    res = {"run": a.run, "n": n}
    # (1) what each leading channel carries
    zs = np.concatenate([s[0] for s in S])
    keys = list(S[0][1])
    corr = {k: [float(np.corrcoef(zs[:, i], np.concatenate([s[1][k] for s in S]))[0, 1]) for i in range(n)] for k in keys}
    res["channel_correlations"] = corr
    print("[headP] correlation of z_i with physical site means (train storms)")
    print("        " + " ".join(f"{k:>10s}" for k in keys))
    for i in range(n):
        print(f"   z{i}   " + " ".join(f"{corr[k][i]:+10.2f}" for k in keys), flush=True)
    th = np.concatenate([s[1]["theta_sfc"] for s in S]); z0 = zs[:, 0]
    r = float(np.corrcoef(z0, th)[0, 1]); sg = float(np.sign(r) * th.std(ddof=1) / z0.std(ddof=1)); bg = float(th.mean() - sg * z0.mean())
    # (2) the head's channel-0 equation
    with torch.no_grad():
        co = {k: v.double().numpy() for k, v in h.effective_coefficients().items()}
        b = h.b.detach().double().numpy()
    A, Q1, Q2 = co["A"], co["Q1"], co["Q2"]
    iu = h.iu.numpy()
    feats = {}
    for z in zte:
        zt = torch.as_tensor(z[f0:min(f1, z.shape[0] - 2)], dtype=torch.float32)[..., :n]
        d1, d2 = _d1_d2(zt)
        zt, d1, d2 = (x.double().numpy()[:, halo:-halo].reshape(-1, n) for x in (zt, d1, d2))
        f = {"1": np.full(len(zt), b[0])}
        for j in range(n):
            f[f"z{j}"] = A[0, 0, j] * zt[:, j]; f[f"d1z{j}"] = A[1, 0, j] * d1[:, j]; f[f"d2z{j}"] = A[2, 0, j] * d2[:, j]
        for c, (i_, j_) in enumerate(zip(iu[0], iu[1])):
            f[f"z{i_}*z{j_}"] = Q1[0, c] * zt[:, i_] * zt[:, j_]
        for i_ in range(n):
            for j_ in range(n):
                f[f"z{i_}*d1z{j_}"] = Q2[0, i_ * n + j_] * zt[:, i_] * d1[:, j_]
        for k_, v in f.items():
            feats.setdefault(k_, []).append(v)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    g0 = sum(feats.values())
    sd = float(g0.std())
    only0 = [k for k in feats if k == "1" or (k[0] == "z" and "*" not in k and k[1:] == "0") or k in ("d1z0", "d2z0", "z0*z0", "z0*d1z0")]
    g_only0 = sum(feats[k] for k in only0)
    r2_only0 = float(1 - np.sum((g0 - g_only0) ** 2) / np.sum((g0 - g0.mean()) ** 2))
    res["head_channel0"] = {"sd_g0": sd, "R2_z0_only_terms_vs_g0": r2_only0,
                            "terms_sd_share": sorted(((k, float(v.std() / sd)) for k, v in feats.items()), key=lambda x: -x[1])[:12]}
    print(f"[headP] gauge map: r {r:+.3f}, s {sg:+.3f} K, b {bg:+.3f} K")
    print(f"[headP] head channel 0: R2 of its z0-only terms against the full g_0 (no refit): {r2_only0:.3f}")
    print("[headP] largest terms (sd of coef x feature / sd of g_0): " + ", ".join(f"{k} {v:.2f}" for k, v in res["head_channel0"]["terms_sd_share"][:10]))
    a0, a1, nu, q, a2 = A[0, 0, 0], A[1, 0, 0], A[2, 0, 0], Q1[0, 0], Q2[0, 0]
    res["z0_only_in_theta_units"] = {"c0_km_min": float(-(a1 - a2 * bg / sg) * SITE_KM), "c1_km_min_per_K": float(-a2 / sg * SITE_KM),
                                    "nu_m2_s": float(nu * NU_UNIT), "growth_per_min": float(a0), "coef_sites_frames": [float(x) for x in (a0, a1, nu, q, a2)]}
    u_ = res["z0_only_in_theta_units"]
    print(f"[headP] z0-only terms in theta' units: c(theta') = {u_['c0_km_min']:+.3f} {u_['c1_km_min_per_K']:+.4f} theta' km/min, "
          f"nu {u_['nu_m2_s']:.0f} m2/s, growth {u_['growth_per_min']:+.4f}/min")
    json.dump(res, open(out / "headP.json", "w"), indent=1)
    print(f"[headP] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
