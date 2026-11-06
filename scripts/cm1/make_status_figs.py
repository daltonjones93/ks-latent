"""Figures for the CM1 / Paper III status summary (docs/cm1_status). Run where the models, caches and DA
outputs live (Saidie); writes PNGs to --out. Every number in a figure is computed here from saved models,
cached CM1 members or saved results; nothing is typed in.

  fig_storm_structure    x-z sections of one storm (all 7 variables) at 2 h
  fig_environments       composite reflectivity x-t for storms from three environments
  fig_forecast_sections  truth / 30-min / 60-min forecast / 60-min error for 5 variables (held boundaries)
  fig_hovmoller          surface theta' and column-max condensate x-t: truth, 2-h forecast, persistence
  fig_skill              skill by lead to 2 h (H2), advection, CRPS / spread of the stochastic model,
                         and skill per variable
  fig_latent             latent geometry: PCA of site vectors coloured by physics and by ensemble,
                         variance spectrum, and the leading latent component as an x-t field
  fig_da_example         truth / no DA / latent DA forecast fields after 2 h of cycling
  fig_da_results         paired per-storm DA, noise study, held-out environments, transfer pilot
  python scripts/cm1/make_status_figs.py --out ~/cm1_figs/status
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
from compare_studies import per_truth  # noqa: E402
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import ColumnData, load_member  # noqa: E402
from ks_latent.cm1.models import rollout_from  # noqa: E402
from ks_latent.cm1.physics import BaseState  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402

H = Path.home()
STYLE = {  # variable: label, colormap, (lo, hi) in plotted units, plot q (g/kg) from sqrt(q)?
    "u": ("u′ (m/s)", "RdBu_r", (-15, 15), False), "w": ("w (m/s)", "RdBu_r", (-10, 10), False),
    "th": ("θ′ (K)", "RdBu_r", (-8, 8), False), "qv": ("qv′ (g/kg)", "BrBG", (-4, 4), False),
    "prs": ("p′ (hPa)", "PuOr_r", (-2, 2), False), "qliq": ("liquid (g/kg)", "Blues", (0, 2), True),
    "qice": ("ice (g/kg)", "Purples", (0, 2), True)}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "figure.dpi": 150})


def phys(u_norm: torch.Tensor, norm, Z: int, v: str) -> np.ndarray:
    """(..., C, NX) normalized -> (..., Z, NX) physical field of variable v (g/kg for condensate; p′ is stored in hPa)."""
    i = norm.vars.index(v)
    f = (u_norm[..., i * Z:(i + 1) * Z, :].double() * float(norm.scale[i])).cpu().numpy()
    if STYLE[v][3]:
        f = np.clip(f, 0, None) ** 2
    return f


def section(ax, f, x, z, v, zmax=14.0, title=""):
    lab, cmap, (lo, hi), _ = STYLE[v]
    m = ax.pcolormesh(x, z, f, cmap=cmap, vmin=lo, vmax=hi, shading="auto", rasterized=True)
    ax.set_ylim(0, zmax)
    ax.set_title(title)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(H / "cm1_runs/full_v1_stagec"))
    ap.add_argument("--run-stoch", default=str(H / "cm1_runs/full_v1_stagec_crps"))
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--da", default=str(H / "cm1_da"))
    ap.add_argument("--storm", default="ens_A2_mem001")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    run = Path(a.run)
    ae, prop, norm, cfg, split = load_run(run, dev)
    halo = cfg["stage_b"]["halo_sites"]
    files = member_files(a.source)
    base = BaseState.from_sounding()
    zk = base.z / 1000.0
    m = load_member(files[a.storm])
    d = ColumnData([m], norm, dev)
    Z, NX = d.Z, d.NX
    x = (np.arange(NX) + 0.5) * 0.25
    dt_min = d.dt / 60
    z = encode_members(ae, d)[0]
    T2 = int(round(120 / dt_min))
    done = []

    # ---- storm structure ----
    if "structure" not in a.skip:
        u = d.full(0, T2)[0]
        fig, axs = plt.subplots(4, 2, figsize=(7.5, 7.0), sharex=True, constrained_layout=True)
        for ax, v in zip(axs.flat, ["u", "w", "th", "qv", "prs", "qliq", "qice"]):
            mm = section(ax, phys(u, norm, Z, v), x, zk, v, title=STYLE[v][0])   # sets 0-14 km
            fig.colorbar(mm, ax=ax, shrink=0.85)
        ax = axs.flat[-1]
        cr = m.cref[:T2 + 1]
        mm = ax.pcolormesh(x, np.arange(cr.shape[0]) * dt_min, cr, cmap="turbo", vmin=0, vmax=60, shading="auto", rasterized=True)
        ax.set_ylim(0, 120); ax.set_title("composite reflectivity (dBZ), x–t, 0–2 h"); ax.set_ylabel("t (min)")
        fig.colorbar(mm, ax=ax, shrink=0.85)
        for ax in list(axs[:, 0]) + [axs[0, 1], axs[1, 1], axs[2, 1]]:
            ax.set_ylabel("z (km)")
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.suptitle(f"One unseen test storm ({a.storm}) at t = 2 h: the 7 model variables (62 levels each)")
        fig.savefig(out / "fig_storm_structure.png"); plt.close(fig); done.append("structure")

    # ---- environments ----
    if "environments" not in a.skip:
        storms = [("ens_A2_mem001", "control environment"), ("ens_B_mem016", "dry, strong shear (held out in H3)"),
                  ("ens_B_mem044", "moist, strong shear"), ("ens_B_mem054", "mid moisture, weak shear")]
        fig, axs = plt.subplots(1, 4, figsize=(7.5, 3.0), sharey=True, constrained_layout=True)
        for ax, (sn, lab) in zip(axs, storms):
            mm_ = load_member(files[sn])
            mm = ax.pcolormesh(x, np.arange(mm_.cref.shape[0]) * dt_min / 60, mm_.cref, cmap="turbo", vmin=0, vmax=60,
                               shading="auto", rasterized=True)
            ax.set_title(f"{sn}\n{lab}", fontsize=7); ax.set_xlabel("x (km)")
        axs[0].set_ylabel("t (h)")
        fig.colorbar(mm, ax=axs, shrink=0.8, label="composite reflectivity (dBZ)")
        fig.suptitle("Squall lines in different environments (x–t of composite reflectivity, moving grid)")
        fig.savefig(out / "fig_environments.png"); plt.close(fig); done.append("environments")

    # ---- forecast sections ----
    t0 = int(round(60 / dt_min))
    with torch.no_grad():
        lat = rollout_from(prop, z, t0, int(round(120 / dt_min)), halo, boundary="persistence")[:, 0]   # (L, P, c)
        dec = torch.stack([ae.decode(lat[j:j + 1])[0] for j in range(lat.shape[0])])                      # (L, C, NX)
    if "sections" not in a.skip:
        vs = ["w", "th", "u", "qliq", "qice"]
        L30, L60 = int(round(30 / dt_min)) - 1, int(round(60 / dt_min)) - 1
        fig, axs = plt.subplots(len(vs), 4, figsize=(7.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
        for r, v in enumerate(vs):
            tr = phys(d.full(0, t0 + L60 + 1)[0], norm, Z, v)
            f30 = phys(dec[L30], norm, Z, v)
            f60 = phys(dec[L60], norm, Z, v)
            tr30 = phys(d.full(0, t0 + L30 + 1)[0], norm, Z, v)
            section(axs[r, 0], tr30, x, zk, v, title="truth, t = 1.5 h" if r == 0 else "")
            section(axs[r, 1], f30, x, zk, v, title="forecast, 30 min" if r == 0 else "")
            mm = section(axs[r, 2], f60, x, zk, v, title="forecast, 60 min" if r == 0 else "")
            section(axs[r, 3], tr, x, zk, v, title="truth, t = 2 h" if r == 0 else "")
            fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0])
            axs[r, 0].set_ylabel("z (km)")
            for ax in axs[r]:
                ax.set_xlim(100, 300)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        fig.suptitle(f"Forecasts from t = 1 h ({a.storm}, unseen), boundaries held at the start: no future information")
        fig.savefig(out / "fig_forecast_sections.png"); plt.close(fig); done.append("sections")

    # ---- Hovmoller: truth / forecast / persistence ----
    if "hovmoller" not in a.skip:
        L = dec.shape[0]
        tt = (t0 + 1 + np.arange(L)) * dt_min / 60
        tru = d.full(0, slice(t0 + 1, t0 + 1 + L))
        def sfc_th(u):
            return phys(u, norm, Z, "th")[..., 0, :]
        def cond(u):
            return np.sqrt(phys(u, norm, Z, "qliq") + phys(u, norm, Z, "qice")).max(axis=-2) ** 2
        p0 = d.full(0, t0)[0][None].expand(L, -1, -1)
        fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.0), sharex=True, sharey=True, constrained_layout=True)
        for c_, (lab, u_) in enumerate((("truth", tru), ("forecast from 1 h", dec), ("persistence of 1 h", p0))):
            m1 = axs[0, c_].pcolormesh(x, tt, sfc_th(u_), cmap="RdBu_r", vmin=-8, vmax=8, shading="auto", rasterized=True)
            m2 = axs[1, c_].pcolormesh(x, tt, cond(u_), cmap="Blues", vmin=0, vmax=4, shading="auto", rasterized=True)
            axs[0, c_].set_title(lab)
            axs[1, c_].set_xlabel("x (km)")
        axs[0, 0].set_ylabel("t (h)"); axs[1, 0].set_ylabel("t (h)")
        fig.colorbar(m1, ax=axs[0], label="surface θ′ (K)", shrink=0.85)
        fig.colorbar(m2, ax=axs[1], label="column-max condensate (g/kg)", shrink=0.85)
        fig.suptitle(f"Cold pool and precipitation over a 2-h forecast ({a.storm}); held boundaries")
        fig.savefig(out / "fig_hovmoller.png"); plt.close(fig); done.append("hovmoller")

    # ---- skill curves ----
    if "skill" not in a.skip:
        h2 = json.loads((run / "results_h2_lead120.json").read_text())["forecast"]
        h2s = json.loads((Path(a.run_stoch) / "results_h2_lead120.json").read_text())["forecast"]
        adv = json.loads((run / "h1a_advection.json").read_text())
        lead = np.array(h2["lead_s"]) / 60
        fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.7), constrained_layout=True)
        ax = axs[0]
        ax.plot(lead, np.array(h2["nmse"]) / np.array(h2["nmse_pers"]), label="surrogate (Stage C)", lw=2)
        ax.plot(lead, np.array(h2s["nmse"]) / np.array(h2s["nmse_pers"]), label="stochastic, mean of 8", lw=1.5)
        ax.plot(lead, adv["modes"]["per_level"]["skill"], label="advection extrapolation", lw=1.5, ls="--")
        ax.axhline(1, color="0.5", lw=1, ls=":"); ax.text(118, 1.03, "persistence", ha="right", color="0.4", fontsize=7)
        ax.set_xlabel("lead (min)"); ax.set_ylabel("nMSE / persistence nMSE"); ax.set_ylim(0, 1.4)
        ax.set_title("(a) skill, 22 unseen storms × 8 starts"); ax.legend(frameon=False, fontsize=6.5)
        ax = axs[1]
        ax.plot(lead, h2["nmse"], label="surrogate", lw=2); ax.plot(lead, h2["nmse_pers"], label="persistence", lw=1.5, color="0.5")
        ax.set_xlabel("lead (min)"); ax.set_ylabel("nMSE (normalized units)")
        ax.set_title("(b) error growth"); ax.legend(frameon=False, fontsize=6.5)
        ax = axs[2]
        ax.plot(lead, np.array(h2["crps"]) / np.array(h2["crps_pers"]), label="deterministic (CRPS = MAE)", lw=2)
        ax.plot(lead, np.array(h2s["crps"]) / np.array(h2s["crps_pers"]), label="stochastic, 8 samples", lw=1.5)
        ax.plot(lead, h2s["spread_err"], label="stochastic spread / error", lw=1, ls="--", color="k")
        ax.set_xlabel("lead (min)"); ax.set_ylim(0, 1.2); ax.set_title("(c) probabilistic skill and calibration")
        ax.legend(frameon=False, fontsize=6.5)
        fig.savefig(out / "fig_skill.png"); plt.close(fig)
        # skill per variable (6 unseen storms, starts at 1 h and 2 h, held boundaries)
        names = [n for n in split["test"] if n.startswith("ens_A2_")][:3] + [n for n in split["test"] if n.startswith("ens_B_")][:3]
        V = len(norm.vars)
        se_m, se_p = 0, 0
        n_lead = int(round(60 / dt_min))
        for sn in names:
            dd = ColumnData([load_member(files[sn])], norm, dev)
            zz = encode_members(ae, dd)[0]
            for t0_ in (int(round(60 / dt_min)), int(round(120 / dt_min))):
                with torch.no_grad():
                    la = rollout_from(prop, zz, t0_, n_lead, halo, boundary="persistence")[:, 0]
                    de = torch.stack([ae.decode(la[j:j + 1])[0] for j in range(n_lead)])
                tr = dd.full(0, slice(t0_ + 1, t0_ + 1 + n_lead))
                p0 = dd.full(0, t0_)[0][None]
                se_m = se_m + ((de - tr) ** 2).reshape(n_lead, V, Z, NX).mean(dim=(2, 3)).cpu().numpy()
                se_p = se_p + ((p0 - tr) ** 2).reshape(n_lead, V, Z, NX).mean(dim=(2, 3)).cpu().numpy()
        fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
        ll = np.arange(1, n_lead + 1) * dt_min
        for i, v in enumerate(norm.vars):
            ax.plot(ll, se_m[:, i] / se_p[:, i], label=STYLE[v][0].split(" (")[0])
        ax.axhline(1, color="0.5", lw=1, ls=":")
        ax.set_xlabel("lead (min)"); ax.set_ylabel("MSE / persistence MSE"); ax.set_ylim(0, 1.1)
        ax.set_title("Skill per variable (6 unseen storms, 2 starts each)"); ax.legend(frameon=False, ncol=2, fontsize=6.5)
        fig.savefig(out / "fig_skill_by_variable.png"); plt.close(fig); done.append("skill")

    # ---- latent geometry ----
    if "latent" not in a.skip:
        rng = np.random.default_rng(0)
        names = split["test"]
        Zs, cols = [], {"th_sfc": [], "w_max": [], "cond": [], "ens": []}
        p = ae.cfg.patch
        for sn in names:
            dd = ColumnData([load_member(files[sn])], norm, dev)
            zz = encode_members(ae, dd)[0]                                  # (T, P, c)
            for t_ in range(30, zz.shape[0], 20):
                u_ = dd.full(0, t_)[0]
                th0 = phys(u_, norm, Z, "th")[0].reshape(-1, p).mean(1)
                wm = phys(u_, norm, Z, "w").max(axis=0).reshape(-1, p).max(1)
                cd = (phys(u_, norm, Z, "qliq") + phys(u_, norm, Z, "qice")).max(axis=0).reshape(-1, p).max(1)
                Zs.append(zz[t_].cpu().numpy()); cols["th_sfc"].append(th0); cols["w_max"].append(wm); cols["cond"].append(cd)
                cols["ens"].append(np.full(zz.shape[1], 0 if sn.startswith("mem") else 1 if sn.startswith("ens_A2") else 2))
        Zs = np.concatenate(Zs); cols = {k: np.concatenate(v) for k, v in cols.items()}
        mu = Zs.mean(0); U, S, Vt = np.linalg.svd(Zs - mu, full_matrices=False)
        pcs = (Zs - mu) @ Vt[:3].T
        ev = S ** 2 / (S ** 2).sum()
        idx = rng.permutation(len(Zs))[:30000]
        storm = cols["cond"][idx] > 0.05
        fig, axs = plt.subplots(2, 3, figsize=(7.5, 5.0), constrained_layout=True)
        for ax, (k, lab, cm, lim) in zip(axs[0], (("th_sfc", "surface θ′ (K)", "RdBu_r", (-6, 6)),
                                                  ("w_max", "column max w (m/s)", "magma", (0, 8)),
                                                  ("cond", "column max condensate (g/kg)", "Blues", (0, 3)))):
            o = np.argsort(np.abs(cols[k][idx]))
            sc = ax.scatter(pcs[idx][o, 0], pcs[idx][o, 1], c=cols[k][idx][o], s=1.5, cmap=cm, vmin=lim[0], vmax=lim[1],
                            rasterized=True, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.8, label=lab)
            ax.set_xlabel("latent PC1"); ax.set_ylabel("latent PC2")
        ax = axs[1, 0]
        for e, lab, c_ in ((0, "ens_A (control)", "C0"), (1, "ens_A2 (control, new noise)", "C1"), (2, "ens_B (varied env.)", "C2")):
            sel = idx[(cols["ens"][idx] == e) & (cols["cond"][idx] > 0.05)]
            ax.scatter(pcs[sel, 0], pcs[sel, 2], s=1.5, color=c_, label=lab, rasterized=True, linewidths=0, alpha=0.6)
        ax.set_xlabel("latent PC1"); ax.set_ylabel("latent PC3"); ax.set_title("storm sites by ensemble")
        ax.legend(frameon=False, fontsize=6, markerscale=5)
        ax = axs[1, 1]
        ax.semilogy(np.arange(1, ev.size + 1), ev, "o-", ms=2)
        ax.set_xlabel("principal component"); ax.set_ylabel("variance fraction")
        ax.set_title(f"latent spectrum (c = {ev.size}): no gap"); ax.grid(alpha=0.3)
        ax = axs[1, 2]
        pc1 = ((z.cpu().numpy() - mu) @ Vt[0])                                  # (T, P)
        P = pc1.shape[1]
        xs = (np.arange(P) + 0.5) * p * 0.25
        mm = ax.pcolormesh(xs, np.arange(pc1.shape[0]) * dt_min / 60, pc1, cmap="RdBu_r", shading="auto", rasterized=True,
                           vmin=-np.percentile(np.abs(pc1), 99), vmax=np.percentile(np.abs(pc1), 99))
        ax.set_xlabel("x (km)"); ax.set_ylabel("t (h)"); ax.set_title(f"PC1 of the latent field, {a.storm}")
        fig.colorbar(mm, ax=ax, shrink=0.8)
        fig.suptitle(f"Latent space geometry: {len(Zs)} site vectors (2 km sites) from the 22 unseen test storms")
        fig.savefig(out / "fig_latent.png"); plt.close(fig); done.append("latent")

    # ---- DA example: truth / no DA / latent DA after 2 h of cycling ----
    if "da_example" not in a.skip:
        from ks_latent.cm1.letkf import LETKFConfig
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        from ks_latent.cm1.twin import TwinConfig, run_twin
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        rh = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev).eval(); rh.load_state_dict(ck["state_dict"])
        ens_names = sorted(np.random.default_rng(0).choice(split["train"], 32, replace=False).tolist())
        others = ColumnData([load_member(files[n]) for n in ens_names], norm, dev)
        Z0 = torch.stack([zz[t0] for zz in encode_members(ae, others)])
        tc = dict(t0=t0, n_cycles=36, steps_per_cycle=5, halo=halo, obs_every_cells=8, sigma_obs=0.5, model_noise=0.1,
                  refl_every_cells=8, sigma_refl=5.0, vr_tilts=(0.5, 2.0, 5.0), sigma_vr=2.0,
                  letkf=LETKFConfig(loc_km=5.0, rtps=0.9), boundary="persistence", forecast_from=(), seed=0, keep_fields=True)
        free = run_twin(ae, prop, d, z, Z0, norm, TwinConfig(**tc), assimilate=False, refl_head=rh)
        da = run_twin(ae, prop, d, z, Z0, norm, TwinConfig(**tc), refl_head=rh)
        k = 23                                                              # cycle ending at t0 + 120 min
        fr = int(free["cycles"][k]["frame"])
        vs = ["th", "w", "qliq"]
        fig, axs = plt.subplots(len(vs), 3, figsize=(7.5, 5.5), sharex=True, sharey=True, constrained_layout=True)
        for r, v in enumerate(vs):
            section(axs[r, 0], phys(d.full(0, fr)[0], norm, Z, v), x, zk, v, title="truth" if r == 0 else "")
            section(axs[r, 1], phys(free["cycles"][k]["prior_field"].float(), norm, Z, v), x, zk, v,
                    title="no DA (ensemble mean)" if r == 0 else "")
            mm = section(axs[r, 2], phys(da["cycles"][k]["prior_field"].float(), norm, Z, v), x, zk, v,
                         title="latent DA, 5-min forecast (prior mean)" if r == 0 else "")
            fig.colorbar(mm, ax=axs[r], shrink=0.9, label=STYLE[v][0])
            axs[r, 0].set_ylabel("z (km)")
            for ax in axs[r]:
                ax.set_xlim(100, 300)
        for ax in axs[-1]:
            ax.set_xlabel("x (km)")
        nf = free["cycles"][k]["prior"]["nmse_mean_dec"]; nd = da["cycles"][k]["prior"]["nmse_mean_dec"]
        fig.suptitle(f"{a.storm}, t = {fr * dt_min / 60:.1f} h, after 2 h of 5-min cycles (θ′ + reflectivity + radial velocity)\n"
                     f"nMSE {nf:.2f} without DA, {nd:.2f} with")
        fig.savefig(out / "fig_da_example.png"); plt.close(fig); done.append("da_example")

    # ---- DA results ----
    if "da_results" not in a.skip:
        D = Path(a.da)
        fig, axs = plt.subplots(1, 4, figsize=(7.5, 2.4), constrained_layout=True)
        r, _ = per_truth(D / "noise_det_0.1")
        nod = np.array([v["noDA"] for v in r.values()]); dav = np.array([v["prior"] for v in r.values()])
        ens = ["C2" if t.startswith("ens_B") else "C1" if t.startswith("ens_A2") else "C0" for t in r]
        ax = axs[0]
        ax.scatter(nod, dav, c=ens, s=12)
        lim = (0, max(nod.max(), dav.max()) * 1.05); ax.plot(lim, lim, color="0.5", lw=1, ls=":")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("no DA, nMSE"); ax.set_ylabel("latent DA, nMSE")
        ax.set_title(f"(a) 22 storms: −{100 * (1 - dav.mean() / nod.mean()):.0f}%")
        ax = axs[1]
        for name, lab, mk in (("noise_det_0.02", "det 0.02", "o"), ("noise_det_0.05", "det 0.05", "o"), ("noise_det_0.1", "det 0.1", "*"),
                              ("noise_stoch_0", "stoch, RTPS 0.9", "s"), ("noise_stoch_0_rtps0.5", "stoch, RTPS 0.5", "s"),
                              ("noise_stoch_0_rtps0", "stoch, RTPS 0", "s")):
            rr, _ = per_truth(D / name)
            ax.scatter(np.mean([v["sperr"] for v in rr.values()]), np.mean([v["prior"] for v in rr.values()]), marker=mk,
                       s=30 if mk == "*" else 14, label=lab)
        ax.axvspan(0.7, 1.3, color="0.9", zorder=0)
        ax.set_xlabel("spread / error at θ′ obs"); ax.set_ylabel("cycled nMSE"); ax.set_title("(b) model error")
        ax.legend(frameon=False, fontsize=5.5)
        ax = axs[2]
        vals = []
        for name in ("h3_da_indist", "h3_da_region"):
            s_ = json.loads((D / name / "study.json").read_text())
            vals.append((s_["free"]["nmse_mean_dec"]["mean"], s_["modes"]["latent"]["prior"]["nmse_mean_dec"]["mean"]))
        xx = np.arange(2)
        ax.bar(xx - 0.18, [v[0] for v in vals], 0.36, label="no DA", color="0.6")
        ax.bar(xx + 0.18, [v[1] for v in vals], 0.36, label="latent DA", color="C0")
        ax.set_xticks(xx); ax.set_xticklabels(["in-distribution", "held-out\nenvironment"])
        ax.set_ylabel("cycled nMSE"); ax.set_title("(c) held-out env. (H3)"); ax.legend(frameon=False, fontsize=6)
        ax = axs[3]
        t1 = json.loads((D / "t1_pilot" / "t1.json").read_text())
        for loc, lab in ((5.0, "localized (5 km)"), (10000.0, "no localization")):
            Ns = sorted({row["N"] for row in t1["da"]})
            mn = [np.mean([row["ratio_prior"] for row in t1["da"] if row["N"] == N and row["loc_km"] == loc]) for N in Ns]
            sd = [np.std([row["ratio_prior"] for row in t1["da"] if row["N"] == N and row["loc_km"] == loc]) for N in Ns]
            ax.errorbar(Ns, mn, yerr=sd, marker="o", ms=3, capsize=2, label=lab)
        ax.axhline(1, color="0.5", lw=1, ls=":"); ax.set_xscale("log", base=2)
        ax.set_xlabel("ensemble size"); ax.set_ylabel("error ratio, 2 storms / 1"); ax.set_title("(d) transfer (T1)")
        ax.legend(frameon=False, fontsize=6)
        fig.savefig(out / "fig_da_results.png"); plt.close(fig); done.append("da_results")
    print("made:", done, "->", out)


if __name__ == "__main__":
    main()
