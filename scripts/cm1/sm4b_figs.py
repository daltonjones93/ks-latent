"""Figures for SM4b (the rollout-trained flow model) against the deterministic and noise-channel models.

  fig_sm4b_sections_w.png / _cond.png  x-z sections of one test storm at three leads: truth, deterministic
                                       forecast, one noise-channel stochastic sample, one SM4b sample
  fig_sm4b_bands.png                   fraction of the truth's variance kept in the < 4 km and 4-16 km bands
                                       against lead, single samples of each model (SM0 json files)
  fig_sm4b_extremes.png                99.9th percentile of w and condensate, minimum surface theta', vs lead
  fig_sm4b_spectra.png                 horizontal power spectra of low-level w and mid-level condensate
  fig_sm4b_crps.png                    CRPS skill and spread/error against lead (SM4b's results.json)
  python scripts/cm1/sm4b_figs.py --det ~/cm1_runs/full_v1_stagec --stoch ~/cm1_runs/full_v1_stagec_crps \\
      --flow ~/cm1_runs/sm4b_flow_stagec --source cache:A+B+C --sm0 Deterministic=~/cm1_da/sm0x_v1:det \\
      "Noise-channel=~/cm1_da/sm0x_v1:stoch_sample" "SM2b=~/cm1_da/sm0x_sm2b:stoch_sample" \\
      "SM4b flow=~/cm1_da/sm0x_sm4b:stoch_sample" --out ~/cm1_figs/sm4b
"""
import argparse
import json
import sys
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

# reference categorical palette (dataviz skill, slots 1-4 in fixed order); truth in primary ink
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STYLES = ["-", "--", "-.", "-"]
MARKERS = ["o", "s", "^", "D"]
FIELD_LEVELS = ("w_low", "w_mid", "th_low", "th_mid", "u_low", "u_mid", "cond_low", "cond_mid")


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "legend.frameon": False, "figure.facecolor": "white"})


def phys(u: torch.Tensor, norm, Z: int):
    def var(v):
        i = norm.vars.index(v)
        return (u[i * Z:(i + 1) * Z].double() * float(norm.scale[i])).cpu().numpy()
    cond = np.clip(var("qliq"), 0, None) ** 2 + np.clip(var("qice"), 0, None) ** 2
    return {"w": var("w"), "cond": cond, "th": var("th")}


def sections(a, out: Path):
    dev = torch.device(a.device)
    ae, prop, norm, cfg, split = load_run(Path(a.det).expanduser(), dev)
    ae_s, prop_s, _, _, _ = load_run(Path(a.stoch).expanduser(), dev)
    ae_f, prop_f, _, _, _ = load_stochastic_run(Path(a.flow).expanduser(), dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    storm = a.storm or [n for n in split["test"] if n.startswith("ens_B_")][0]
    d = ColumnData([load_member(files[storm])], norm, dev)
    Z, dt_min = d.Z, d.dt / 60
    t0 = int(round(a.t0_min / dt_min))
    nl = int(round(max(a.leads) / dt_min))
    with torch.no_grad():
        runs = {}
        for name, (m_ae, m_prop, S) in {"det": (ae, prop, 1), "stoch": (ae_s, prop_s, 1), "flow": (ae_f, prop_f, 1)}.items():
            z = encode_members(m_ae, d)[0]
            kw = {"n_samples": S, "gen": torch.Generator().manual_seed(a.seed)} if name != "det" else {}
            lat = rollout_from(m_prop, z, t0, nl, halo, boundary="persistence", **kw)[:, 0]
            runs[name] = (m_ae, lat)
    zkm = BaseState.from_sounding().z / 1000.0
    rows = [("CM1 truth", None), ("Deterministic", "det"), ("Noise-channel sample", "stoch"), ("SM4b flow sample", "flow")]
    for field, (cmap, vmin, vmax, label) in {"w": ("RdBu_r", -10, 10, "w (m/s)"),
                                            "cond": ("Blues", 0, 8, "condensate (g/kg)")}.items():
        fig, axs = plt.subplots(len(rows), len(a.leads), figsize=(3.3 * len(a.leads), 1.75 * len(rows)),
                                sharex="col", sharey=True, constrained_layout=True)
        for c, L in enumerate(a.leads):
            j = int(round(L / dt_min))
            tr = d.full(0, t0 + j)[0]
            xc = int(d.cref[0][t0 + j].argmax())
            lo, hi = max(0, xc - 160), min(d.NX, xc + 160)                     # +- 40 km around the storm
            x = (np.arange(lo, hi) - xc) * 0.25
            for r, (rname, key) in enumerate(rows):
                if key is None:
                    u = tr
                else:
                    m_ae, lat = runs[key]
                    u = m_ae.decode(lat[j - 1:j])[0]
                f = phys(u, norm, Z)[field][:, lo:hi]
                ax = axs[r, c]
                im = ax.pcolormesh(x, zkm, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
                ax.set_ylim(0, 15); ax.grid(False)
                if r == 0:
                    ax.set_title(f"{L:g} min after {a.t0_min:g} min", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{rname}\nheight (km)", fontsize=8)
                if r == len(rows) - 1:
                    ax.set_xlabel("x relative to the storm (km)")
        fig.colorbar(im, ax=axs, shrink=0.6, label=label)
        fig.suptitle(f"{storm} (unseen test storm): {label.split(' (')[0]} sections", fontsize=10)
        fig.savefig(out / f"fig_sm4b_sections_{field}.png", dpi=160)
        plt.close(fig)
        print(f"-> {out / f'fig_sm4b_sections_{field}.png'}")


def load_sm0(spec: str):
    label, rest = spec.split("=", 1)
    d, kind = rest.rsplit(":", 1)
    d = Path(d).expanduser()
    r = json.loads((d / "sm0.json").read_text())
    sp = np.load(d / "spectra.npz") if (d / "spectra.npz").exists() else None
    return label, kind, r, sp


def bands_and_extremes(a, out: Path):
    sm0 = [load_sm0(s) for s in a.sm0]
    leads = sorted({float(L) for _, _, r, _ in sm0 for L in r["band_variance_ratio"]})
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    for b, (ax, title) in enumerate(zip(axs, ("< 4 km", "4-16 km"))):
        ax.axhline(1, color=INK, lw=1)
        ax.text(leads[-1], 1.02, "truth", color=MUTED, ha="right", va="bottom", fontsize=8)
        for i, (label, kind, r, _) in enumerate(sm0):
            Ls = [L for L in leads if str(L) in r["band_variance_ratio"]]
            y = [np.mean([r["band_variance_ratio"][str(L)][kind][fl][b] for fl in FIELD_LEVELS]) for L in Ls]
            ax.plot(Ls, y, STYLES[i], marker=MARKERS[i], color=SERIES[i], lw=2, ms=6, label=label)
        ax.set_title(f"Variance kept at wavelengths {title}", fontsize=9)
        ax.set_xlabel("lead (min)"); ax.set_ylabel("fraction of the truth's variance")
        ax.set_xticks(leads)
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Single forecasts or samples, mean over w, θ′, u, condensate at ~1 and ~5 km; test storms", fontsize=9)
    fig.savefig(out / "fig_sm4b_bands.png", dpi=160); plt.close(fig)
    print(f"-> {out / 'fig_sm4b_bands.png'}")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, (e, title) in zip(axs, (("w999", "w, 99.9th percentile (m/s)"), ("cond999", "condensate, 99.9th percentile (g/kg)"),
                                    ("th_sfc_min", "minimum surface θ′ (K)"))):
        _, _, r0, _ = sm0[0]
        Ls = [L for L in leads if str(L) in r0["extremes"]]
        ax.plot(Ls, [np.mean([c[e] for c in r0["extremes"][str(L)]["truth"]]) if isinstance(r0["extremes"][str(L)]["truth"], list)
                     else r0["extremes"][str(L)]["truth"][e] for L in Ls], "-", color=INK, lw=2, label="CM1 truth")
        for i, (label, kind, r, _) in enumerate(sm0):
            Ls_ = [L for L in leads if str(L) in r["extremes"]]
            ex = r["extremes"]
            ax.plot(Ls_, [ex[str(L)][kind][e] if isinstance(ex[str(L)][kind], dict) else np.mean([c[e] for c in ex[str(L)][kind]])
                          for L in Ls_], STYLES[i], marker=MARKERS[i], color=SERIES[i], lw=2, ms=6, label=label)
        ax.set_title(title, fontsize=9); ax.set_xlabel("lead (min)"); ax.set_xticks(leads)
    axs[0].legend(fontsize=7)
    fig.savefig(out / "fig_sm4b_extremes.png", dpi=160); plt.close(fig)
    print(f"-> {out / 'fig_sm4b_extremes.png'}")
    with_spec = [(lab, k, sp) for lab, k, _, sp in sm0 if sp is not None]
    if not with_spec:
        return
    panels = [(L, fl) for L in (30, 60) for fl in (("w", "low"), ("cond", "mid"))]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2), constrained_layout=True)
    for ax, (L, (f, ln)) in zip(axs, panels):
        lab0, _, sp0 = with_spec[0]
        key = lambda k: f"{L:g}|{k}|{f}|{ln}"
        if key("truth") not in sp0:
            continue
        lam = sp0["lam_km"][1:]
        ax.loglog(lam, sp0[key("truth")][1:], "-", color=INK, lw=2, label="CM1 truth")
        for i, (label, kind, sp) in enumerate(with_spec):
            if key(kind) in sp:
                ax.loglog(lam, sp[key(kind)][1:], STYLES[i], color=SERIES[i], lw=1.6, label=label)
        ax.invert_xaxis(); ax.axvline(4, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{'w' if f == 'w' else 'condensate'} at ~{1 if ln == 'low' else 5} km, {L} min", fontsize=9)
        ax.set_xlabel("wavelength (km)")
    axs[0].set_ylabel("power"); axs[0].legend(fontsize=7)
    fig.savefig(out / "fig_sm4b_spectra.png", dpi=160); plt.close(fig)
    print(f"-> {out / 'fig_sm4b_spectra.png'}")


def crps(a, out: Path):
    r = json.loads((Path(a.flow).expanduser() / "results.json").read_text())["comparison"]
    names = {"deterministic": "Deterministic", "stochastic": "Noise-channel", "flow": "SM4b flow"}
    colours = {"deterministic": SERIES[0], "stochastic": SERIES[1], "flow": SERIES[3]}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for i, (k, row) in enumerate((k, r[k]) for k in names if k in r):
        L = np.array(row["lead_s"]) / 60
        axs[0].plot(L, np.array(row["nmse"]) / np.array(row["nmse_pers"]), STYLES[i], color=colours[k], lw=2, label=names[k])
        axs[1].plot(L, np.array(row["crps"]) / np.array(row["crps_pers"]), STYLES[i], color=colours[k], lw=2, label=names[k])
        if row.get("spread_err"):
            axs[2].plot(L, row["spread_err"], STYLES[i], color=colours[k], lw=2, label=names[k])
    axs[2].axhline(1, color=INK, lw=1)
    for ax, t in zip(axs, ("Error of the mean (nMSE / persistence)", "CRPS / persistence MAE", "Spread / error (1 = calibrated)")):
        ax.set_title(t, fontsize=9); ax.set_xlabel("lead (min)")
    axs[0].legend(fontsize=7)
    fig.savefig(out / "fig_sm4b_crps.png", dpi=160); plt.close(fig)
    print(f"-> {out / 'fig_sm4b_crps.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--stoch", required=True)
    ap.add_argument("--flow", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--sm0", nargs="+", required=True, help="LABEL=DIR:KIND (KIND det or stoch_sample), plot order")
    ap.add_argument("--storm", default="")
    ap.add_argument("--t0-min", type=float, default=120.0)
    ap.add_argument("--leads", type=float, nargs="+", default=[5.0, 30.0, 60.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    style()
    bands_and_extremes(a, out)
    crps(a, out)
    sections(a, out)


if __name__ == "__main__":
    main()
