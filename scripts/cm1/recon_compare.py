"""I-PCA (3): reconstruction at equal dimension, local PCA-k against the trained autoencoders, on the test storms.

PCA is fitted once with k_max components on the reference run's training storms (same normalizer, same
50% storm-weighted second moment as `fit_pca_stage`); PCA-k for smaller k takes the leading k of the same
eigenvectors, identical to fitting with k. Every model is scored on the same test storms:
  - `reconstruction_report`: normalized MSE (all columns, storm columns), RMSE per variable, physics;
  - SM0's small-scale diagnostics on frames at --t-min: variance kept in the < 4 / 4-16 / > 16 km bands
    (fraction of the truth's) for w, theta', u', condensate at ~1 and ~5 km, and extremes
    (99.9% of w and condensate, minimum surface theta').
  python scripts/cm1/recon_compare.py --ref ~/cm1_runs/full_v1_stagec --source cache:A+B+C --out ~/cm1_da/recon_cmp \
      --ae-runs ~/cm1_runs/full_v1 ~/cm1_runs/full_v1_stagec ~/cm1_runs/sm1_c64_stagec --pca-k 16 32 64
"""
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
from smoothing_diag import BANDS, FIELDS, fields, spectrum  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member, positive_channel_mask  # noqa: E402
from ks_latent.cm1.evaluate import reconstruction_report  # noqa: E402
from ks_latent.cm1.models import ColumnAEConfig, PCAColumnAE  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import fit_pca_stage  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def truncate(pca: PCAColumnAE, k: int, eig: list) -> PCAColumnAE:
    cfg = ColumnAEConfig(**{**pca.cfg.__dict__, "latent_channels": k})
    out = PCAColumnAE(cfg, pca.positive_mask.cpu()).to(pca.mean.device)
    out.mean.copy_(pca.mean); out.components.copy_(pca.components[:k])
    out.scale.fill_(float(np.sqrt(np.mean(eig[:k])))); out.explained.copy_(pca.explained[:k])
    return out.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="run whose normalizer and split are used")
    ap.add_argument("--ae-runs", nargs="*", default=[])
    ap.add_argument("--pca-k", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-min", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    ap.add_argument("--max-train", type=int, default=0, help="fit on this many training storms (0: all; smoke)")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    _, _, norm, cfg, split = load_run(Path(a.ref).expanduser(), dev)
    models = {}
    for r in a.ae_runs:
        ae, _, n2, _, sp2 = load_run(Path(r).expanduser(), dev)
        if list(n2.scale) != list(norm.scale) or sp2["test"] != split["test"]:
            raise ValueError(f"{r} has a different normalizer or test split from {a.ref}")
        models[Path(r).name] = ae.eval()
    files = member_files(a.source)
    tr = [n for n in split["train"] if n in files][: a.max_train or None]
    if len(tr) < len(split["train"]) and not a.max_train:
        raise ValueError(f"training members missing from {a.source}: {sorted(set(split['train']) - set(files))[:5]}")
    print(f"[recon] fitting PCA-{max(a.pca_k)} on {len(tr)} training storms", flush=True)
    d_tr = ColumnData([load_member(files[n]) for n in tr], norm, torch.device("cpu"))
    d_va = ColumnData([load_member(files[n]) for n in split["val"]], norm, torch.device("cpu"))
    Z = d_tr.Z
    acfg = ColumnAEConfig(n_channels=d_tr.C, n_levels=Z, patch=cfg["ae"]["patch"], latent_channels=max(a.pca_k), kind="pca")
    pca = PCAColumnAE(acfg, positive_channel_mask(Z, norm.vars)).to(dev)
    info = fit_pca_stage(pca, d_tr, d_va, log=lambda s: print(s, flush=True))
    del d_tr, d_va
    eig = info["eigenvalues"]
    for k in sorted(a.pca_k):
        models[f"pca{k}"] = truncate(pca, k, eig)
    te = split["test"][: a.max_test or None]
    z_m = BaseState.from_sounding().z
    levels = {"low": int(np.argmin(np.abs(z_m - 1000))), "mid": int(np.argmin(np.abs(z_m - 5000)))}
    rec = {k: {"se_all": 0.0, "n": 0} for k in models}
    reports = {k: [] for k in models}
    spec = {k: {f"{f}_{ln}": 0 for f in FIELDS for ln in levels} for k in ["truth", *models]}
    ext = {k: [] for k in ["truth", *models]}
    for sn in te:
        d = ColumnData([load_member(files[sn])], norm, torch.device("cpu"))
        for k, m in models.items():
            with torch.no_grad():
                reports[k].append(reconstruction_report(m, d, norm))
        dt_min = d.dt / 60
        for tm in a.t_min:
            t = int(round(tm / dt_min))
            if t >= d.T[0]:
                continue
            u = d.full(0, t, device=dev)
            with torch.no_grad():
                cases = {"truth": u[0], **{k: m(u)[0][0] for k, m in models.items()}}
            for k, x in cases.items():
                fz = fields(x, norm, Z, levels)
                for f in FIELDS:
                    for ln in levels:
                        spec[k][f"{f}_{ln}"] = spec[k][f"{f}_{ln}"] + spectrum(fz["lev"][f][ln])
                ext[k].append(fz["ext"])
        print(f"[recon] {sn} done [{time.time() - t_start:.0f}s]", flush=True)
    NX = d.NX
    lam = 1.0 / np.maximum(np.fft.rfftfreq(NX, d=0.25), 1e-12)
    res = {"ref": a.ref, "test": te, "pca_fit": info, "levels": levels, "bands_km": BANDS, "models": {}}
    for k in models:
        rs = reports[k]
        r = {"nmse_all": float(np.mean([x["nmse_all"] for x in rs])),
             "nmse_storm": float(np.mean([x["nmse_storm"] for x in rs if "nmse_storm" in x])),
             "rmse_phys_storm": {v: float(np.mean([x["rmse_phys_storm"][v] for x in rs if "rmse_phys_storm" in x]))
                                 for v in norm.vars},
             "physics": {p: float(np.mean([x["physics"][p] for x in rs])) for p in rs[0]["physics"]},
             "band_variance_ratio": {fl: [float(spec[k][fl][(lam >= lo) & (lam < hi)].sum()
                                                / spec["truth"][fl][(lam >= lo) & (lam < hi)].sum()) for lo, hi in BANDS]
                                     for fl in spec["truth"]},
             "extremes": {e: float(np.mean([x[e] for x in ext[k]])) for e in ext[k][0]},
             "dim_per_site": int(models[k].cfg.latent_channels)}
        res["models"][k] = r
    res["extremes_truth"] = {e: float(np.mean([x[e] for x in ext["truth"]])) for e in ext["truth"][0]}
    (out / "recon_compare.json").write_text(json.dumps(res, indent=1))
    write_provenance(out / "recon_compare.json", config=vars(a), seed=0, device=str(dev),
                     wall_time_s=time.time() - t_start)
    et = res["extremes_truth"]
    print(f"\n[recon] {len(te)} test storms; PCA-{max(a.pca_k)} explains {100 * info['explained_total']:.1f}% "
          f"(storm-weighted); truth extremes: w99.9 {et['w999']:.2f}, cond99.9 {et['cond999']:.2f}, th_min {et['th_sfc_min']:.2f}")
    print("%-22s %4s %8s %8s   %-24s %-24s %6s %6s %6s" % ("model", "dim", "nMSE", "storm", "w_low  <4/4-16/>16 km",
                                                            "cond_mid <4/4-16/>16", "w999", "cnd999", "thmin"))
    for k, r in res["models"].items():
        b = r["band_variance_ratio"]; e = r["extremes"]
        print("%-22s %4d %8.4f %8.4f   %-24s %-24s %6.2f %6.2f %6.2f" % (
            k, r["dim_per_site"], r["nmse_all"], r["nmse_storm"], " ".join("%.2f" % v for v in b["w_low"]),
            " ".join("%.2f" % v for v in b["cond_mid"]), e["w999"], e["cond999"], e["th_sfc_min"]))
    print(f"[recon] done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
