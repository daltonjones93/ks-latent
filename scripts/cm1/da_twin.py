"""E3: latent DA twin experiment with a trained CM1 column local field.

Truth: a held-out member (default: the run's first test member). Initial ensemble: the encoded states
of training members at the start frame (other storms; nothing from the truth). Observations: surface
theta' from the TRUE columns every --obs-km km, with Gaussian error --sigma-obs K. Runs a free ensemble
(no DA) and the LETKF for every combination of --loc-km x --rtps x --obs-km, all from the same initial
ensemble, and reports scores over the second half of the cycles plus the autoencoder floor
(decode(encode(truth))) at the same frames.

  python scripts/cm1/da_twin.py --run ~/cm1_runs/d34_c32_kuniform --source cache:~/cm1_cache --out ~/cm1_da/e3_v1
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ks_latent.cm1.data import VARS, ColumnData, Normalizer, load_member, positive_channel_mask  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig  # noqa: E402
from ks_latent.cm1.models import ColumnAEConfig, ColumnFieldAE, StencilPropagator, StencilPropConfig  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import TwinConfig, run_twin, summarize_twin  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402


def member_files(source: str) -> dict:
    """name -> file for every cached member in the cache: directories (reads only each file's name)."""
    if not source.startswith("cache:"):
        raise ValueError("da_twin.py reads cached members: --source cache:DIR[,DIR2...]")
    out = {}
    for d in source[len("cache:"):].split(","):
        for f in sorted(Path(d).expanduser().glob("*.npz")):
            with np.load(f) as z:
                name = str(z["name"])
            if name in out:
                raise ValueError(f"member {name} in two cache files: {out[name]} and {f}")
            out[name] = f
    return out


def load_run(run: Path, dev):
    cfg = json.loads((run / "config.json").read_text())
    norm = Normalizer.from_dict(json.loads((run / "normalizer.json").read_text()))
    acfg, pcfg = ColumnAEConfig(**cfg["ae"]), StencilPropConfig(**cfg["prop"])
    ae = ColumnFieldAE(acfg, positive_channel_mask(acfg.n_levels)).to(dev)
    prop = StencilPropagator(pcfg).to(dev)
    ae.load_state_dict(torch.load(run / "ae.pt", map_location=dev)["state_dict"])
    prop.load_state_dict(torch.load(run / "prop.pt", map_location=dev)["state_dict"])
    split = json.loads((run / "results.json").read_text())["split"]
    return ae.eval(), prop.eval(), norm, cfg, split


def plot(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    free, best = res["free"], res["runs"][res["best"]]
    tt = lambda r: np.array([c["frame"] for c in r["cycles"]]) / 60.0
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
    ax[0].plot(tt(free), [c["prior"]["nmse_mean_dec"] for c in free["cycles"]], color="#888888", lw=2, label="no DA")
    ax[0].plot(tt(best), [c["prior"]["nmse_mean_dec"] for c in best["cycles"]], color="#2a6fdb", lw=2,
               label="LETKF, forecast (prior)")
    ax[0].plot(tt(best), [c["analysis"]["nmse_mean_dec"] for c in best["cycles"]], color="#1b9e77", lw=1.5,
               ls="--", label="LETKF, analysis")
    ax[0].plot(tt(best), res["ae_floor_nmse"], color="#d9822b", lw=1.5, ls=":", label="autoencoder floor")
    ax[0].set_xlabel("time (h)"); ax[0].set_ylabel("normalized MSE, all variables"); ax[0].legend(frameon=False, fontsize=7)
    sf, sb = res["summary"]["free"]["prior"]["rmse_phys"], res["summary"]["runs"][res["best"]]["prior"]["rmse_phys"]
    ax[1].bar(np.arange(len(VARS)), [sb[v] / sf[v] for v in VARS], color=["#2a6fdb" if v == "th" else "#8e6bd1" for v in VARS])
    ax[1].axhline(1, color="#888888", lw=1)
    ax[1].set_xticks(np.arange(len(VARS))); ax[1].set_xticklabels(["u′", "w", "θ′ (obs.)", "qv′", "p′", "liq", "ice"])
    ax[1].set_ylabel("forecast RMSE, LETKF / no DA (second half)")
    rh = np.array(res["summary"]["runs"][res["best"]]["prior"]["rank_hist"], float)
    ax[2].bar(np.arange(rh.size), rh / rh.sum(), color="#2a6fdb")
    ax[2].axhline(1 / rh.size, color="#888888", lw=1)
    ax[2].set_xlabel("rank of observation among forecast members"); ax[2].set_ylabel("frequency")
    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.suptitle(f"E3 twin: truth {res['truth']}, best config {res['best']}")
    fig.savefig(out / "fig_twin.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default=None, help="member name (default: the run's first test member)")
    ap.add_argument("--n-members", type=int, default=0, help="ensemble size (default: all training members)")
    ap.add_argument("--t0-min", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--cycle-min", type=float, default=5.0)
    ap.add_argument("--obs-km", type=float, nargs="+", default=[2.0, 10.0])
    ap.add_argument("--sigma-obs", type=float, default=0.5)
    ap.add_argument("--loc-km", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    ap.add_argument("--rtps", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--obs-sets", nargs="+", default=["th"], choices=["th", "refl", "th+refl"],
                    help="what is assimilated: surface theta', composite reflectivity (needs refl_head.pt), or both")
    ap.add_argument("--refl-km", type=float, default=2.0)
    ap.add_argument("--sigma-refl", type=float, default=5.0)
    ap.add_argument("--extra-props", nargs="*", default=[],
                    help="run dirs with propagators trained on the SAME autoencoder (--reuse-ae): multi-model ensemble")
    ap.add_argument("--model-noise", type=float, nargs="+", default=[0.0],
                    help="additive latent noise std per forecast step (spread maintenance); several = sweep")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    run = Path(a.run).expanduser()
    ae, prop, norm, cfg, split = load_run(run, dev)
    props = [prop]
    for d in a.extra_props:
        d = Path(d).expanduser()
        a2, p2, _, c2, _ = load_run(d, dev)
        if c2["ae"] != cfg["ae"] or any(not torch.equal(x, y) for x, y in zip(a2.state_dict().values(), ae.state_dict().values())):
            raise ValueError(f"{d} does not share the autoencoder of {run}: multi-model needs --reuse-ae runs")
        props.append(p2)
    refl_head = None
    if any("refl" in s for s in a.obs_sets):
        from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
        ck = torch.load(run / "refl_head.pt", map_location=dev)
        refl_head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])).to(dev)
        refl_head.load_state_dict(ck["state_dict"]); refl_head.eval()
    files = member_files(a.source)
    truth_name = a.truth or split["test"][0]
    ens_names = split["train"][: a.n_members or None]
    if truth_name in ens_names:
        raise ValueError("the truth must not be an ensemble member")
    print(f"[e3] run {run.name}: truth {truth_name}, {len(ens_names)} members from its training set, device {dev}")
    truth = ColumnData([load_member(files[truth_name])], norm, dev)
    dt = truth.dt
    others = ColumnData([load_member(files[n]) for n in ens_names], norm, dev)
    if others.dt != dt:
        raise ValueError("ensemble and truth frame spacings differ")
    z_truth = encode_members(ae, truth)[0]
    t0 = int(round(a.t0_min * 60 / dt)); spc = int(round(a.cycle_min * 60 / dt))
    n_cycles = int(round(a.hours * 60 / a.cycle_min))
    with torch.no_grad():
        Z0 = torch.stack([ae.encode(others.full(m, t0, device=dev))[0] for m in range(len(ens_names))])
        floor = []
        for k in range(n_cycles):
            u = truth.full(0, t0 + (k + 1) * spc, device=dev)
            floor.append(float(((ae(u)[0] - u) ** 2).mean()))
    del others
    halo = cfg["stage_b"]["halo_sites"]
    base = dict(t0=t0, n_cycles=n_cycles, steps_per_cycle=spc, sigma_obs=a.sigma_obs, halo=halo, seed=a.seed)
    res = {"truth": truth_name, "ensemble": ens_names, "run": str(run), "dt_s": dt, "ae_floor_nmse": floor,
           "n_models": len(props),
           "free": run_twin(ae, props, truth, z_truth, Z0, norm, TwinConfig(**base), assimilate=False,
                            refl_head=refl_head), "runs": {}}
    summ = {"free": summarize_twin(res["free"]), "runs": {}}
    print(f"[e3] no DA: forecast nMSE (2nd half) {summ['free']['prior']['nmse_mean_dec']:.3f}; "
          f"autoencoder floor {np.mean(floor[n_cycles // 2:]):.3f}  [{time.time() - t_start:.0f}s]")
    for oset, obs_km, loc, rtps, mn in itertools.product(a.obs_sets, a.obs_km, a.loc_km, a.rtps, a.model_noise):
        key = f"{oset}_obs{obs_km:g}km_loc{loc:g}km_rtps{rtps:g}_mn{mn:g}" + (f"_{len(props)}models" if len(props) > 1 else "")
        tc = TwinConfig(**base, obs_every_cells=int(round(obs_km / 0.25)), model_noise=mn,
                        assim_theta="th" in oset,
                        refl_every_cells=int(round(a.refl_km / 0.25)) if "refl" in oset else 0, sigma_refl=a.sigma_refl,
                        letkf=LETKFConfig(loc_km=loc, inflation=a.inflation, rtps=rtps))
        r = run_twin(ae, props, truth, z_truth, Z0, norm, tc, refl_head=refl_head)
        res["runs"][key] = r
        s = summarize_twin(r); summ["runs"][key] = s
        print(f"[e3] {key:28s} forecast nMSE {s['prior']['nmse_mean_dec']:.3f}  analysis {s['analysis']['nmse_mean_dec']:.3f}  "
              f"th obs RMSE/spread {s['prior']['th_obs_rmse']:.2f}/{s['prior']['th_obs_spread']:.2f} K  "
              f"w {s['prior']['rmse_phys']['w']:.2f} (no DA {summ['free']['prior']['rmse_phys']['w']:.2f})  "
              f"front |disp| {s['prior']['front_abs_disp_km_median']:.2f} km"
              + (f"  refl RMSE/echo {s['prior']['refl_obs_rmse']:.1f}/{s['prior']['refl_echo_rmse']:.1f} dBZ "
                 f"(no DA {summ['free']['prior']['refl_obs_rmse']:.1f}/{summ['free']['prior']['refl_echo_rmse']:.1f})"
                 if "refl_obs_rmse" in s["prior"] else "")
              + f"  qliq {s['prior']['rmse_phys']['qliq']:.3f} qice {s['prior']['rmse_phys']['qice']:.3f}"
              f" (no DA {summ['free']['prior']['rmse_phys']['qliq']:.3f}/{summ['free']['prior']['rmse_phys']['qice']:.3f})"
              f"  [{time.time() - t_start:.0f}s]", flush=True)
    res["summary"] = summ
    res["best"] = min(summ["runs"], key=lambda k: summ["runs"][k]["prior"]["nmse_mean_dec"])
    (out / "twin.json").write_text(json.dumps(res))
    (out / "summary.json").write_text(json.dumps({"truth": truth_name, "n_members": len(ens_names), "best": res["best"],
                                                  "ae_floor_nmse_2nd_half": float(np.mean(floor[n_cycles // 2:])),
                                                  **summ}, indent=1))
    write_provenance(out / "summary.json", config=TwinConfig(**base), seed=a.seed, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"args": vars(a)})
    plot(res, out)
    print(f"[e3] best: {res['best']}; done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
