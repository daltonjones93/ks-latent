"""SM0: where does the smoothing come from? Horizontal power spectra and extremes of truth, autoencoder
reconstruction, deterministic forecast and stochastic samples, at several leads, on unseen test storms.

Per case (storm x start time x lead): the physical fields of w, theta', u' and condensate (liquid + ice, g/kg)
at a low (~1 km) and a mid (~5 km) level; their horizontal power spectra (rfft over the 400 km domain,
mean removed), and the variance in three wavelength bands (< 4 km, 4-16 km, > 16 km) as a fraction of the
truth's. Extremes per case: 99.9th percentile of w and of condensate over the whole section, minimum
surface theta'. If the reconstruction already loses the small scales, the limit is the representation
(SM1); if the reconstruction keeps them and the forecast loses them, it is the objective (SM2-SM4).
  python scripts/cm1/smoothing_diag.py --run ~/cm1_runs/full_v1_stagec --run-stoch ~/cm1_runs/full_v1_stagec_crps \
      --source cache:A+B+C --out ~/cm1_da/sm0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, load_stochastic_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

FIELDS = ("w", "th", "u", "cond")
BANDS = ((0.0, 4.0), (4.0, 16.0), (16.0, 1e9))      # wavelength bands (km)


def fields(u: torch.Tensor, norm, Z: int, levels: dict) -> dict:
    """(C, NX) normalized -> {field: {level_name: (NX,) physical}} plus the full sections for extremes."""
    def var(v):
        i = norm.vars.index(v)
        return (u[i * Z:(i + 1) * Z].double() * float(norm.scale[i])).cpu().numpy()
    w, th, uu = var("w"), var("th"), var("u")
    cond = np.clip(var("qliq"), 0, None) ** 2 + np.clip(var("qice"), 0, None) ** 2
    sec = {"w": w, "th": th, "u": uu, "cond": cond}
    return {"lev": {f: {ln: sec[f][k] for ln, k in levels.items()} for f in FIELDS},
            "ext": {"w999": float(np.percentile(w, 99.9)), "cond999": float(np.percentile(cond, 99.9)),
                    "th_sfc_min": float(th[0].min())}}


def spectrum(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(x - x.mean())) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-stoch", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-storms", type=int, default=10)
    ap.add_argument("--t0-min", type=float, nargs="+", default=[60.0, 120.0])
    ap.add_argument("--leads-min", type=float, nargs="+", default=[5.0, 30.0, 60.0])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0, help="flow model only: scale of its source noise")
    ap.add_argument("--n-flow", type=int, default=0, help="flow model only: Euler steps when sampling (0: as trained)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.run).expanduser(), dev)
    ae_s, prop_s, norm_s, cfg_s, _ = load_stochastic_run(Path(a.run_stoch).expanduser(), dev)
    if a.temperature != 1.0:
        if not hasattr(prop_s, "temperature"):
            raise ValueError("--temperature applies to flow models only")
        prop_s.temperature = a.temperature
    if a.n_flow:
        prop_s.cfg.n_flow = a.n_flow
    if list(norm.scale) != list(norm_s.scale):
        raise ValueError("the two runs use different normalizers")
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    z_m = BaseState.from_sounding().z
    levels = {"low": int(np.argmin(np.abs(z_m - 1000))), "mid": int(np.argmin(np.abs(z_m - 5000)))}
    storms = [n for n in split["test"] if n.startswith("ens_A2_")][:a.n_storms // 2] + \
             [n for n in split["test"] if n.startswith("ens_B_")][:a.n_storms - a.n_storms // 2]
    kinds = ("truth", "recon", "det", "stoch_sample", "stoch_mean")
    spec = {L: {k: {f: {ln: 0 for ln in levels} for f in FIELDS} for k in kinds} for L in a.leads_min}
    ext = {L: {k: [] for k in kinds} for L in a.leads_min}
    n_cases = 0
    for sn in storms:
        d = ColumnData([load_member(files[sn])], norm, dev)
        Z, NX = d.Z, d.NX
        z, zs = encode_members(ae, d)[0], encode_members(ae_s, d)[0]
        dt_min = d.dt / 60
        for t0m in a.t0_min:
            t0 = int(round(t0m / dt_min))
            nl = int(round(max(a.leads_min) / dt_min))
            if t0 + nl >= d.T[0]:
                continue
            with torch.no_grad():
                lat = rollout_from(prop, z, t0, nl, halo, boundary="persistence")[:, 0]
                lats = rollout_from(prop_s, zs, t0, nl, halo, a.n_samples, gen=torch.Generator().manual_seed(0),
                                    boundary="persistence")                               # (L, S, P, c)
                for L in a.leads_min:
                    j = int(round(L / dt_min)) - 1
                    tr = d.full(0, t0 + j + 1)[0]
                    dec_s = ae_s.decode(lats[j])                                           # (S, C, NX)
                    cases = {"truth": tr, "recon": ae(tr[None])[0][0], "det": ae.decode(lat[j:j + 1])[0],
                             "stoch_sample": dec_s[0], "stoch_mean": dec_s.mean(0)}
                    for k, u in cases.items():
                        fz = fields(u, norm, Z, levels)
                        for f in FIELDS:
                            for ln in levels:
                                spec[L][k][f][ln] = spec[L][k][f][ln] + spectrum(fz["lev"][f][ln])
                        ext[L][k].append(fz["ext"])
            n_cases += 1
        print(f"[SM0] {sn} done [{time.time() - t_start:.0f}s]", flush=True)
    k_wave = np.fft.rfftfreq(NX, d=0.25)                          # cycles per km
    lam = np.where(k_wave > 0, 1.0 / np.maximum(k_wave, 1e-12), np.inf)
    np.savez(out / "spectra.npz", lam_km=lam, **{f"{L:g}|{k}|{f}|{ln}": spec[L][k][f][ln] / max(n_cases, 1)
                                                for L in a.leads_min for k in kinds for f in FIELDS for ln in levels})
    res = {"storms": storms, "n_cases": n_cases, "levels": levels, "bands_km": BANDS, "band_variance_ratio": {},
           "extremes": {}}
    for L in a.leads_min:
        res["band_variance_ratio"][str(L)] = {}
        for k in kinds:
            res["band_variance_ratio"][str(L)][k] = {}
            for f in FIELDS:
                for ln in levels:
                    tr = spec[L]["truth"][f][ln]
                    res["band_variance_ratio"][str(L)][k][f"{f}_{ln}"] = [
                        float(spec[L][k][f][ln][(lam >= lo) & (lam < hi)].sum() / tr[(lam >= lo) & (lam < hi)].sum())
                        for lo, hi in BANDS]
        res["extremes"][str(L)] = {k: {e: float(np.mean([x[e] for x in ext[L][k]])) for e in ext[L][k][0]} for k in kinds}
    (out / "sm0.json").write_text(json.dumps(res, indent=1))
    # summary
    print(f"\n[SM0] variance retained by wavelength band (< 4 km, 4-16 km, > 16 km), fraction of the truth's; {n_cases} cases")
    for L in a.leads_min:
        print(f"  lead {L:.0f} min")
        for f in FIELDS:
            for ln in levels:
                row = "    %-9s" % f"{f}_{ln}" + "".join(
                    "  %s %s" % (k[:12].ljust(12), " ".join("%.2f" % v for v in res["band_variance_ratio"][str(L)][k][f"{f}_{ln}"]))
                    for k in kinds if k != "truth")
                print(row)
    print("\n[SM0] extremes (mean over cases): w 99.9% (m/s), condensate 99.9% (g/kg), min surface theta' (K)")
    for L in a.leads_min:
        for k in kinds:
            e = res["extremes"][str(L)][k]
            print(f"  lead {L:4.0f} min {k:13s} w {e['w999']:6.2f}  cond {e['cond999']:5.2f}  th_sfc_min {e['th_sfc_min']:6.2f}")
    # figure: spectra at the longest lead, plus the reconstruction
    Lf = max(a.leads_min)
    fig, axs = plt.subplots(2, len(FIELDS), figsize=(10, 5), constrained_layout=True)
    for c_, f in enumerate(FIELDS):
        for r, ln in enumerate(levels):
            ax = axs[r, c_]
            for k, sty in (("truth", "k-"), ("recon", "C0-"), ("det", "C1-"), ("stoch_sample", "C2-"), ("stoch_mean", "C2--")):
                ax.loglog(lam[1:], spec[Lf][k][f][ln][1:] / n_cases, sty, lw=1, label=k)
            ax.invert_xaxis(); ax.set_title(f"{f}, {ln} level, lead {Lf:.0f} min", fontsize=8)
            ax.set_xlabel("wavelength (km)")
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.savefig(out / "fig_sm0_spectra.png", dpi=130)
    print(f"[SM0] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
