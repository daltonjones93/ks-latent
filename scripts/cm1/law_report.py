"""FPc: read a trained latent law (`train_cm1.py --prop-body law`) on the test storms.

  ordering     reconstruction nMSE from the first k latent channels only (k = 1, 2, 4, 8, 16, all): with nested
               dropout the curve should fall steeply then flatten (an importance-ordered latent)
  residual     the residual network's share of the squared tendency on encoded test states (0 = closed form)
  terms        coefficient counts above thresholds, and the largest terms by name, in per-minute units
  dispersion   the linear part's symbol L(kappa) = A0 + i sin(kappa) A1 + (2 cos(kappa) - 2) A2 at wavelengths
               4-128 km: growth rate (1/min) and phase speed (km/min) of each latent mode, fastest and most
               unstable. Linearization about z = 0 (the latent's mean state), not about a storm.
  pruning      forecast skill (nMSE / persistence at 5 / 30 / 60 min) after zeroing every coefficient below a
               threshold, without retraining: skill against number of terms
  python scripts/cm1/law_report.py --run ~/cm1_runs/fpc_law_reuse --source cache:A+B+C
"""
import argparse
import copy
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
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.evaluate import forecast_report  # noqa: E402
from ks_latent.cm1.models import _d1_d2  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

SITE_KM = 2.0


def term_names(c: int, iu: torch.Tensor) -> dict:
    lin = [[f"{op}z{j}" for j in range(c)] for op in ("", "d1 ", "d2 ")]
    return {"A": lin, "Q1": [f"z{int(i)}*z{int(j)}" for i, j in zip(iu[0], iu[1])],
            "Q2": [f"z{i}*d1z{j}" for i in range(c) for j in range(c)], "G": [f"<z{j}>" for j in range(c)]}


def top_terms(law, n: int) -> list:
    names = term_names(law.c, law.iu)
    rows = []
    co = law.coefficient_tensors()
    A = co["A"].detach().cpu().numpy()
    for o in range(3):
        for i in range(law.c):
            for j in range(law.c):
                rows.append((abs(A[o, i, j]), f"dz{i}/dt += {A[o, i, j]:+.4f} {names['A'][o][j]}"))
    for k in ("Q1", "Q2", "G"):
        if k in co:
            Q = co[k].detach().cpu().numpy()
            for i in range(Q.shape[0]):
                for j in range(Q.shape[1]):
                    rows.append((abs(Q[i, j]), f"dz{i}/dt += {Q[i, j]:+.4f} {names[k][j]}"))
    rows.sort(key=lambda r: -r[0])
    return [r[1] for r in rows[:n]]


def dispersion(law, wavelengths_km: list) -> list:
    A = law.coefficient_tensors()["A"].detach().double().cpu()
    out = []
    for L in wavelengths_km:
        kappa = 2 * np.pi * SITE_KM / L                       # radians per site
        sym = A[0].to(torch.complex128) + 1j * np.sin(kappa) * A[1] + (2 * np.cos(kappa) - 2) * A[2]
        ev = torch.linalg.eigvals(sym).numpy()
        speed = -ev.imag / kappa * SITE_KM                    # km/min (phase speed of exp(i(kx - wt)))
        out.append({"wavelength_km": L, "max_growth_per_min": float(ev.real.max()),
                    "min_growth_per_min": float(ev.real.min()), "max_abs_speed_km_min": float(np.abs(speed).max()),
                    "speeds_km_min_of_3_least_damped": [float(s) for s in speed[np.argsort(-ev.real)[:3]]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    run = Path(a.run).expanduser()
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(run, dev)
    if cfg["prop"].get("body") != "law":
        raise ValueError(f"{run} is not a law run (prop body {cfg['prop'].get('body')})")
    law = prop.law
    files = member_files(a.source)
    te = split["test"][: a.max_test or None]
    test = ColumnData([load_member(files[n]) for n in te], norm, dev)
    halo = cfg["stage_b"]["halo_sites"]
    res = {"run": str(run), "test": te}
    # ordering
    c = ae.cfg.latent_channels
    ks = sorted({k for k in (1, 2, 4, 8, 16, c) if k <= c})
    err = {k: [] for k in ks}
    with torch.no_grad():
        for m in range(len(test.u)):
            for t in range(test.T[m] // 4, test.T[m], max(1, test.T[m] // 8)):
                u = test.full(m, t)
                z = ae.encode(u)
                for k in ks:
                    zk = z.clone(); zk[..., k:] = 0
                    err[k].append(((ae.decode(zk) - u) ** 2).mean().item())
    res["ordering_nmse_first_k"] = {str(k): float(np.mean(v)) for k, v in err.items()}
    # residual share on encoded test states
    zt = encode_members(ae, test)
    with torch.no_grad():
        num = den = 0.0
        for z in zt:
            zz = z[::5].to(dev)
            if law.R is None:
                break
            d1, d2 = _d1_d2(zz)
            r = law.R(torch.cat([zz, d1, d2], -1))
            f = law.tendency(zz)
            num += float((r ** 2).sum()); den += float((f ** 2).sum())
    res["residual_share"] = num / den if den else 0.0
    res["n_coefficients"] = sum(v.numel() for v in law.coefficient_tensors().values())
    res["terms_above"] = {f"{t:g}": law.n_terms(t) for t in (1e-4, 1e-3, 1e-2, 1e-1)}
    res["top_terms"] = top_terms(law, 25)
    if law.global_mean:   # size of the domain-mean coupling against the local linear coupling
        res["global_vs_local_A0_norm"] = float(law.G.norm() / law.A[0].norm().clamp(min=1e-12))
    res["dispersion"] = dispersion(law, [4, 8, 16, 32, 64, 128])
    # pruning path
    every = int(round(60 * 60 / test.dt))
    n_lead = int(round(60 * 60 / test.dt))
    if every + n_lead >= min(test.T):        # short runs (smoke): shrink the evaluation to fit
        n_lead = every = max(1, min(test.T) // 3)
    t0s = list(range(every, min(test.T) - n_lead, every))
    path = []
    for thr in a.thresholds:
        p2 = copy.deepcopy(prop)
        n = p2.law.prune(thr) if thr > 0 else p2.law.n_terms()
        with torch.no_grad():
            fc = forecast_report(ae, p2, test, zt, norm, t0s, n_lead, halo)
        lead = [round(s / 60) for s in fc["lead_s"]]
        sk = np.array(fc["nmse"]) / np.array(fc["nmse_pers"])
        row = {"threshold": thr, "n_terms": n, "skill_5_30_60": [float(sk[lead.index(L)]) for L in (5, 30, 60) if L in lead] or [float(sk[-1])]}
        path.append(row)
        print(f"[law] prune {thr:g}: {n} terms, skill 5/30/60 {' '.join('%.3f' % s for s in row['skill_5_30_60'])} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    res["pruning_path"] = path
    (run / "law_report.json").write_text(json.dumps(res, indent=1))
    print(f"\n[law] {run.name}: {res['n_coefficients']} coefficients; above 1e-4/1e-3/1e-2/1e-1: "
          f"{' / '.join(str(v) for v in res['terms_above'].values())}; residual share of the tendency {res['residual_share']:.3f}")
    print("[law] reconstruction nMSE from the first k channels: " +
          "  ".join(f"k={k} {v:.4f}" for k, v in res["ordering_nmse_first_k"].items()))
    print("[law] linear part about z = 0: wavelength -> max growth (1/min), fastest |phase speed| (km/min)")
    for d in res["dispersion"]:
        print(f"   {d['wavelength_km']:5.0f} km: growth {d['max_growth_per_min']:+.4f}  speed {d['max_abs_speed_km_min']:.2f}  "
              f"least-damped modes' speeds {', '.join('%+.2f' % s for s in d['speeds_km_min_of_3_least_damped'])}")
    print("[law] largest terms:\n   " + "\n   ".join(res["top_terms"][:15]))
    print(f"[law] done in {(time.time() - t0) / 60:.1f} min -> {run / 'law_report.json'}")


if __name__ == "__main__":
    main()
