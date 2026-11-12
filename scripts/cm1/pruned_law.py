"""5u: the explicit latent law of the pruned (first m channels) models, against the unpruned model's law restricted to the same channels.
For each run (a nested-dropout 'law' propagator): (1) the EFFECTIVE m-channel law, with the dropped channels substituted by their constant fill
(linear terms in the fill fold into A0/A1/b; verified numerically against the model's own tendency with the residual off); (2) channel 0-1 terms ranked
by contribution (|coefficient| x rms of the term over test states); (3) the linear dispersion relation of the m-channel law (growth rate and phase speed
against wavelength); (4) the residual MLP's share of the tendency per channel; (5) decoded forecast error of the explicit law alone (residual
switched off) against law + residual; (6) correlation of each retained channel with the same channel of the unpruned model's latent.
  python scripts/cm1/pruned_law.py --base ~/cm1_runs/fpc_nested_stagec --runs ~/cm1_runs/fpc_nested_p8_dft ... --source cache:... --m 8 --out DIR
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import _d1_d2  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from prop_clamp import roll, var_err, LAGS  # noqa: E402

SITE_KM = 2.0


def eff_law(law, m, fill):
    co = law.effective_coefficients()
    A, Q1, Q2 = co["A"].detach().double(), co["Q1"].detach().double(), co["Q2"].detach().double()
    if "G" in co:
        raise ValueError("global-mean law not supported")
    c, fill, b = law.c, fill.double(), law.b.detach().double()
    A0, A1, A2 = A[0][:m, :m].clone(), A[1][:m, :m].clone(), A[2][:m, :m].clone()
    bt = b[:m] + A[0][:m, m:] @ fill[m:]
    iu = law.iu
    pairs = [(int(j), int(k)) for j, k in zip(iu[0], iu[1])]
    keep1 = {}
    for p, (j, k) in enumerate(pairs):
        if k < m:
            keep1[(j, k)] = Q1[:m, p].clone()
        elif j < m:
            A0[:, j] += Q1[:m, p] * fill[k]
        else:
            bt += Q1[:m, p] * fill[j] * fill[k]
    keep2 = {}
    for j in range(c):
        for k in range(m):
            col = Q2[:m, j * c + k]
            if j < m:
                keep2[(j, k)] = col.clone()
            else:
                A1[:, k] += col * fill[j]
    return {"A0": A0, "A1": A1, "A2": A2, "b": bt, "Q1": keep1, "Q2": keep2, "m": m}


def eff_terms(E, z):
    """z (B, P, m) -> dict of every term's value (B, P, m_out) as {name: (coef (m_out,), value (B, P))}."""
    d1, d2 = _d1_d2(z); m = E["m"]; out = {}
    for j in range(m):
        out[f"z{j}"] = (E["A0"][:, j], z[..., j]); out[f"d1z{j}"] = (E["A1"][:, j], d1[..., j]); out[f"d2z{j}"] = (E["A2"][:, j], d2[..., j])
    out["1"] = (E["b"], torch.ones_like(z[..., 0]))
    for (j, k), q in E["Q1"].items():
        out[f"z{j}*z{k}"] = (q, z[..., j] * z[..., k])
    for (j, k), q in E["Q2"].items():
        out[f"z{j}*d1z{k}"] = (q, z[..., j] * d1[..., k])
    return out


def eff_tendency(E, z):
    f = 0
    for coef, val in eff_terms(E, z).values():
        f = f + val[..., None] * coef.to(val.dtype)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--m", type=int, default=8); ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--t0", default="120,180,240"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); t_start = time.time(); m = a.m
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cpu")
    files = member_files(a.source)
    models = {}
    for pth in [a.base] + a.runs:
        pth = Path(pth).expanduser(); ae, prop, norm, cfg, split = load_run(pth, dev); models[pth.name] = (ae, prop, norm, cfg, split)
    base_name = Path(a.base).expanduser().name
    names = models[base_name][4]["test"][: a.n_test]
    data, lat = {}, {}
    for nm in names:
        data[nm] = ColumnData([load_member(files[nm])], models[base_name][2], dev)
    for rn, (ae, prop, norm, cfg, split) in models.items():
        with torch.no_grad():
            lat[rn] = {nm: encode_members(ae, data[nm])[0] for nm in names}
    res = {"m": m, "base": base_name, "vars": list(VARS), "runs": {}}
    zb = torch.cat([lat[base_name][nm][60:].reshape(-1, lat[base_name][nm].shape[-1]) for nm in names])
    base_fill = zb.mean(0)
    free_base = None
    for rn, (ae, prop, norm, cfg, split) in models.items():
        law = prop.law; c = law.c; pm = prop.cfg.prefix_m
        fill = prop.prefix_fill.detach().cpu() if pm else base_fill
        E = eff_law(law, m, fill)
        # (1) verify the algebra against the model's tendency (residual and cap off)
        R0, cap0 = law.R, law.cap
        law.R, law.cap = None, 0.0
        zz = torch.randn(2, 30, c) * 0.7; zz[..., m:] = fill[m:]
        with torch.no_grad():
            ref = law.tendency(zz)[..., :m].double()
        got = eff_tendency(E, zz[..., :m].double())
        law.R, law.cap = R0, cap0
        err = float((ref - got).abs().max() / ref.abs().max())
        assert err < 1e-4, f"{rn}: effective law differs from the model's tendency ({err})"
        # states: true latents of the test members, tail at the fill
        Z = torch.cat([lat[rn][nm][60:].clone() for nm in names]); Z[..., m:] = fill[m:]
        zs = Z[..., :m].double()
        terms = eff_terms(E, zs)
        r = {"prefix_m": pm, "algebra_check_rel_err": err}
        tot = sum(v[..., None] * c_.to(v.dtype) for c_, v in terms.values())
        rank = {}
        for i in (0, 1, 2, 3):
            if i >= m:
                break
            rows = [(n, float(cf[i]), float((cf[i] * v).pow(2).mean().sqrt())) for n, (cf, v) in terms.items()]
            rows.sort(key=lambda x: -x[2])
            rank[f"ch{i}"] = {"total_rms": float(tot[..., i].pow(2).mean().sqrt()), "top": rows[:8]}
        r["top_terms"] = rank
        r["A_eff_diag"] = {"growth_per_min": [float(E["A0"][i, i]) for i in range(m)], "A1_diag": [float(E["A1"][i, i]) for i in range(m)],
                           "A2_diag_sites2_per_min": [float(E["A2"][i, i]) for i in range(m)], "b": [float(x) for x in E["b"]]}
        # (3) dispersion of the m-channel linear part
        disp = []
        for L in (6, 10, 20, 40, 80, 160):
            kap = 2 * np.pi * SITE_KM / L
            sym = E["A0"].to(torch.complex128) + 1j * np.sin(kap) * E["A1"] + (2 * np.cos(kap) - 2) * E["A2"]
            ev = torch.linalg.eigvals(sym).numpy(); o = np.argsort(-ev.real)
            disp.append({"wavelength_km": L, "max_growth_per_min": float(ev.real.max()), "speed_km_min_3_least_damped": [float(-ev.imag[k] / kap * SITE_KM) for k in o[:3]],
                         "growth_3_least_damped": [float(ev.real[k]) for k in o[:3]]})
        r["dispersion"] = disp
        # (4) residual share per retained channel (with the model's own inputs)
        if law.R is not None:
            with torch.no_grad():
                Zf = torch.cat([lat[rn][nm][60:] for nm in names])
                if pm:
                    Zf = Zf.clone(); Zf[..., m:] = fill[m:]
                d1, d2 = _d1_d2(Zf)
                rr = law.R(torch.cat([Zf, d1, d2], -1))[..., :m]
                law.R, R1 = None, law.R
                c1, law.cap = law.cap, 0.0
                f0 = law.tendency(Zf)[..., :m]
                law.R, law.cap = R1, c1
            r["resid_share_by_channel"] = [float(rr[..., i].pow(2).mean() / (f0[..., i] + rr[..., i]).pow(2).mean()) for i in range(m)]
        # (5) forecast error with and without the residual
        halo, p, m_sites = cfg["stage_b"]["halo_sites"], ae.cfg.patch, ae.cfg.reach_sites
        n = max(LAGS); cases = []
        for nm in names:
            d, z = data[nm], lat[rn][nm]
            for t0 in [int(x) for x in a.t0.split(",")]:
                if t0 + n + 1 > z.shape[0]:
                    continue
                u_true = torch.stack([d.full(0, t0 + k)[0] for k in LAGS])
                ev_ = lambda st: var_err(ae, torch.stack([st[k] for k in LAGS]), u_true, d.Z, halo, m_sites, p).mean(1).tolist()
                case = {"member": nm, "t0": t0, "with_residual": ev_(roll(prop, z, t0, n, halo, [], "free"))}
                R1 = law.R; law.R = None
                case["law_only"] = ev_(roll(prop, z, t0, n, halo, [], "free")); law.R = R1
                cases.append(case)
        r["cases"] = cases
        # (6) channel correspondence with the unpruned model
        if rn != base_name:
            cor = []
            for i in range(m):
                x = torch.cat([lat[rn][nm][60:, :, i].reshape(-1) for nm in names]); y = torch.cat([lat[base_name][nm][60:, :, i].reshape(-1) for nm in names])
                cor.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
            r["corr_with_base_channel"] = cor
        res["runs"][rn] = r
        print(f"[pruned_law] {rn} done ({time.time() - t_start:.0f}s)", flush=True)
    json.dump(res, open(out / "pruned_law.json", "w")); print("[pruned_law] done")


if __name__ == "__main__":
    main()
