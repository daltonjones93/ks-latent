"""More pictures of the best CM1 models (8a/8b), for the held-out storms; run on Saidie (models, caches). Held boundaries (nothing after
the start is known), forecasts from --start-min. Models: label=run_dir; the LAST of --models is the featured (best) one in the ensemble figure.

  fig_best_sections_<storm>.png   truth / each model at +30 and +60 min (w, theta', liquid, ice), sample 0
  fig_best_hovmoller_<storm>.png  surface theta' and column-max condensate x-t over a 2-h forecast: truth and each model (sample 0)
  fig_best_ensemble_<storm>.png   the featured model at +30 min: |ensemble mean - truth|, ensemble std, and four samples (w and liquid)
  python scripts/cm1/make_best_model_figs.py --models orig=RUN a=RUN b=RUN --labels orig="..." --storms mem029 ens_A2_mem001 --out DIR"""
import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from make_status_figs import STYLE, phys, section  # noqa: E402


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--storms", nargs="+", default=["ens_A2_mem001"]); ap.add_argument("--start-min", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    names = [m.split("=", 1)[0] for m in a.models]; paths = {m.split("=", 1)[0]: Path(m.split("=", 1)[1]).expanduser() for m in a.models}
    text = {k: k for k in names}; text.update({m.split("=", 1)[0]: m.split("=", 1)[1] for m in a.labels})
    wrap = lambda t, w=22: textwrap.fill(t, w)
    torch.set_num_threads(16); dev = torch.device("cpu")
    files = member_files(a.source); zk = BaseState.from_sounding().z / 1000.0
    loaded = {nm: load_run(paths[nm], dev) for nm in names}
    norm0 = loaded[names[0]][2]
    for nm in names:
        if not np.allclose(loaded[nm][2].scale, norm0.scale):
            raise ValueError(f"{nm}: different normalizer")
    best = names[-1]
    for storm in a.storms:
        fc, truth, t0 = {}, {}, None
        for nm in names:
            ae, prop, norm, cfg, split = loaded[nm]
            halo = cfg["stage_b"]["halo_sites"]
            d = ColumnData([load_member(files[storm])], norm, dev); Z, NX = d.Z, d.NX; dtm = d.dt / 60
            z = encode_members(ae, d)[0]
            t0 = int(round(a.start_min / dtm)); n = int(round(120 / dtm))
            S = a.samples if getattr(prop.cfg, "noise_channels", 0) > 0 else 1
            torch.manual_seed(0)
            with torch.no_grad():
                lat = rollout_from(prop, z, t0, n, halo, n_samples=S, boundary="persistence")           # (n, S, P, c)
                dec = {30: torch.stack([ae.decode(lat[29, s:s + 1])[0] for s in range(S)]),
                       60: torch.stack([ae.decode(lat[59, s:s + 1])[0] for s in range(S)])}
                hov = torch.cat([ae.decode(lat[j:j + 20, 0]) for j in range(0, n, 20)])                   # (n, C, NX), sample 0
            fc[nm] = {"dec": dec, "hov": hov}
            truth = {30: d.full(0, t0 + 30)[0], 60: d.full(0, t0 + 60)[0], "hov": d.full(0, slice(t0 + 1, t0 + 1 + n))}
        x = (np.arange(NX) + 0.5) * 0.25
        W = min(fc[nm]["dec"][30].shape[-1] for nm in names)
        P = lambda f, v: phys(f, norm0, Z, v)

        # sections
        vs = ["w", "th", "qliq", "qice"]; ncol = 1 + len(names)
        fig, axs = plt.subplots(len(vs), 2 * ncol, figsize=(2.3 * 2 * ncol, 6.6), sharex=True, sharey=True, constrained_layout=True)
        for r, v in enumerate(vs):
            for h, lag in enumerate((30, 60)):
                c0 = h * ncol
                section(axs[r, c0], P(truth[lag], v), x, zk, v, title=f"truth +{lag} min" if r == 0 else "")
                for k, nm in enumerate(names):
                    mm = section(axs[r, c0 + 1 + k], P(fc[nm]["dec"][lag][0], v), x, zk, v, title=wrap(text[nm]) if r == 0 else "")
            fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0]); axs[r, 0].set_ylabel("z (km)")
            for ax in axs[r]:
                ax.set_xlim(100, 300)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.suptitle(f"{storm} (unseen): +30 min (left) and +60 min (right) from t = {a.start_min} min, held boundaries", fontsize=9)
        fig.savefig(out / f"fig_best_sections_{storm}.png", dpi=130); plt.close(fig)

        # hovmoller
        tt = (t0 + 1 + np.arange(n)) * dtm / 60
        sfc = lambda u: phys(u, norm0, Z, "th")[..., 0, :]
        cond = lambda u: np.sqrt(phys(u, norm0, Z, "qliq") + phys(u, norm0, Z, "qice")).max(axis=-2) ** 2
        cols = [("truth", truth["hov"])] + [(wrap(text[nm], 24), fc[nm]["hov"]) for nm in names]
        fig, axs = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.2), sharex=True, sharey=True, constrained_layout=True)
        for c, (lab, u_) in enumerate(cols):
            m1 = axs[0, c].pcolormesh(x, tt, sfc(u_), cmap="RdBu_r", vmin=-8, vmax=8, shading="auto", rasterized=True)
            m2 = axs[1, c].pcolormesh(x, tt, cond(u_), cmap="Blues", vmin=0, vmax=4, shading="auto", rasterized=True)
            axs[0, c].set_title(lab, fontsize=7); axs[1, c].set_xlabel("x (km)")
        axs[0, 0].set_ylabel("t (h)"); axs[1, 0].set_ylabel("t (h)")
        fig.colorbar(m1, ax=axs[0], label="surface θ′ (K)", shrink=0.85); fig.colorbar(m2, ax=axs[1], label="column-max condensate (g/kg)", shrink=0.85)
        fig.suptitle(f"Cold pool and precipitation over a 2-h forecast from t = {a.start_min} min ({storm}, unseen); held boundaries", fontsize=9)
        fig.savefig(out / f"fig_best_hovmoller_{storm}.png", dpi=130); plt.close(fig)

        # ensemble of the featured model
        S = fc[best]["dec"][30].shape[0]
        if S > 1:
            vs2 = ["w", "qliq"]; k_show = 4
            fig, axs = plt.subplots(len(vs2), 3 + k_show, figsize=(2.5 * (3 + k_show), 5.0), sharex=True, sharey=True, constrained_layout=True)
            for r, v in enumerate(vs2):
                fs = np.stack([P(fc[best]["dec"][30][s], v) for s in range(S)]); tr = P(truth[30], v)
                lab, cmap, (lo, hi), _ = STYLE[v]
                panels = [("truth", tr, cmap, lo, hi), ("|ensemble mean − truth|", np.abs(fs.mean(0) - tr), "magma", 0, hi / 2),
                          ("ensemble std", fs.std(0, ddof=1), "magma", 0, hi / 2)] + [(f"sample {s}", fs[s], cmap, lo, hi) for s in range(k_show)]
                for c, (ttl, f, cm, vlo, vhi) in enumerate(panels):
                    mm = axs[r, c].pcolormesh(x, zk, f, cmap=cm, vmin=vlo, vmax=vhi, shading="auto", rasterized=True)
                    axs[r, c].set_ylim(0, 13); axs[r, c].set_xlim(130, 220)
                    if r == 0: axs[r, c].set_title(ttl, fontsize=8)
                    if c in (0, 2): fig.colorbar(mm, ax=axs[r, c], shrink=0.8, label=lab if c == 0 else "")
                axs[r, 0].set_ylabel("z (km)")
            for ax in axs[-1]:
                ax.set_xlabel("x (km)")
            fig.suptitle(f"{text[best]}: +30 min ensemble ({S} samples), error vs spread ({storm}, held boundaries)", fontsize=9)
            fig.savefig(out / f"fig_best_ensemble_{storm}.png", dpi=130); plt.close(fig)
        print(f"[figs] {storm} done", flush=True)


if __name__ == "__main__":
    main()
