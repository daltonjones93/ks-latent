"""Train and evaluate the CM1 column local field: Stage A (autoencoder), Stage B (propagator), test report.

Data sources
  --source local10        the 10 original members (cm1data/cm1r21.1/run_e*, 5-min frames)
  --source grads:DIR      an ensemble made by cluster/cm1 (DIR/members/mem*/cm1out_s.ctl, GrADS)
  --source netcdf:DIR     the same with NetCDF output (DIR/members/mem*/cm1out.nc)
  --source cache:DIR[:PATTERN][+DIR2[:PATTERN2]...]  float16 member files (build_cache.py, or an earlier run's
                          cache), from one or more directories separated by '+' (or ','; but sbatch --export
                          splits on commas, so use '+' there); member names must be unique across them
  --source synthetic      random fields with the right layout: tests the code path only, no physics
Members are split by whole member: the last --n-test are test, the --n-val before them validation.

Profiles
  smoke  3 members, 12 frames, tiny model, a few steps on CPU (< 1 min): the whole code path
  local  the 10 original members on this Mac (MPS)
  full   a cluster ensemble on a GPU (Saidie)

The propagator's reach per step is set from the frame interval: the locality test
(docs/cm1_notes.md) measured an information reach of ~8-12 km per 5-min frame, i.e.
~25 m/s; reach_sites = ceil(25 m/s * dt / site width) + 1 (margin), and the forced halo
equals it.

Outputs (--out): config.json, normalizer.json, ae.pt, prop.pt, history_{a,b}.json,
results.json (each artifact with a .meta.json), fig_forecast.png, fig_sections.png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ks_latent.cm1.data import _natural_key  # noqa: E402
from ks_latent.cm1.data import (VARS, ColumnData, Normalizer, load_cached_members, load_member,  # noqa: E402
                                member_from_cm1io, positive_channel_mask, save_member, split_members,
                                synthetic_members)
from ks_latent.cm1.evaluate import (forecast_report, reconstruction_report, summarize, summarize_fronts,  # noqa: E402
                                    summarize_physics)
from ks_latent.cm1.io import load_grads_run, load_netcdf_run  # noqa: E402
from ks_latent.cm1.models import ColumnAEConfig, StencilPropagator, StencilPropConfig, build_autoencoder  # noqa: E402
from ks_latent.cm1.train import StageAConfig, StageBConfig, encode_members, fit_pca_stage, train_stage_a, train_stage_b  # noqa: E402
from ks_latent.utils.device import get_device  # noqa: E402
from ks_latent.utils.io import write_provenance  # noqa: E402
from ks_latent.utils.seeding import set_seed  # noqa: E402

CM1 = ROOT / "cm1data" / "cm1r21.1"
INFO_SPEED = 25.0  # m/s, from the locality test and the spread-front speed


@dataclasses.dataclass
class RunConfig:
    profile: str
    source: str
    n_val: int
    n_test: int
    seed: int
    device: str
    max_frames: int
    ae: ColumnAEConfig
    prop: StencilPropConfig
    stage_a: StageAConfig
    stage_b: StageBConfig
    eval_lead_min: float
    eval_every_min: float


def load_source(source: str, cache: Path, max_frames: int, log) -> list:
    if source == "synthetic" or source.startswith("cache:"):
        if source == "synthetic":
            log("  WARNING: synthetic data (code-path test only; never report these numbers)")
            members = synthetic_members(4, T=max_frames or 24)
        else:
            members = []
            for part in source[len("cache:"):].replace("+", ",").split(","):
                d, _, pattern = part.partition(":")
                got = load_cached_members(d, pattern or "*.npz")
                log(f"  {len(got)} members from {d}")
                members += got
            names = [m.name for m in members]
            if len(set(names)) != len(names):
                raise ValueError("duplicate member names across cache directories")
            members.sort(key=lambda m: _natural_key(m.name))
        if max_frames:
            members = [dataclasses.replace(m, X=m.X[:max_frames], cref=m.cref[:max_frames], t=m.t[:max_frames])
                       for m in members]
        for m in members:
            log(f"  loaded {m.name}: {m.X.shape[0]} frames, dt {m.dt:.0f} s")
        return members
    cache.mkdir(parents=True, exist_ok=True)
    if source == "local10":
        items = [(f"run_e{i}", lambda r=f"run_e{i}": load_netcdf_run(CM1 / r / "cm1out.nc")) for i in range(1, 11)]
    else:
        kind, d = source.split(":", 1)
        mdirs = sorted(Path(d).glob("members/mem*"))
        if not mdirs:
            raise FileNotFoundError(f"no members/mem* in {d}")
        if kind == "grads":
            items = [(p.name, lambda p=p: load_grads_run(p / "cm1out_s.ctl")) for p in mdirs]
        elif kind == "netcdf":
            items = [(p.name, lambda p=p: load_netcdf_run(p / "cm1out.nc")) for p in mdirs]
        else:
            raise ValueError(f"unknown source kind {kind!r}")
        done = [p for p in mdirs if (p / "cm1.done").exists()]
        if len(done) < len(mdirs):
            log(f"  WARNING: {len(mdirs) - len(done)} members have no cm1.done (unfinished runs are skipped)")
            items = [it for it, p in zip(items, mdirs) if (p / "cm1.done").exists()]
    tag = source.replace("/", "_").replace(":", "_")
    members = []
    for name, loader in items:
        f = cache / f"{tag}__{name}.npz"
        if f.exists():
            m = load_member(f)
        else:
            m = member_from_cm1io(loader(), name)
            save_member(m, f)
        if max_frames:
            m = dataclasses.replace(m, X=m.X[:max_frames], cref=m.cref[:max_frames], t=m.t[:max_frames])
        members.append(m)
        log(f"  loaded {name}: {m.X.shape[0]} frames, dt {m.dt:.0f} s")
    return members


def build_config(a, dt: float | None) -> RunConfig:
    smoke = a.profile == "smoke"
    patch = a.patch
    site_m = patch * 250.0
    dt = dt or 300.0
    reach = a.prop_reach or (math.ceil(INFO_SPEED * dt / site_m) + 1)
    layers = 3
    radius = max(1, math.ceil(reach / layers))
    reach = layers * radius
    ae = ColumnAEConfig(patch=patch, hidden=(7 * 5 if a.encoder_groups > 1 else 32) if smoke else a.hidden,
                        latent_channels=a.latent_channels, n_mix=a.n_mix, mix_radius=1, groups=a.encoder_groups,
                        vertical_channels=a.vertical_channels, vertical_kernel=a.vertical_kernel, kind=a.ae_kind)
    prop = StencilPropConfig(latent_channels=a.latent_channels, hidden=32 if smoke else (a.prop_hidden or a.hidden), layers=layers,
                             body=a.prop_body, heads=a.prop_heads, law_quadratic=not a.law_no_quadratic,
                             law_residual_hidden=a.law_residual_hidden, law_integrator=a.law_integrator,
                             law_substeps=a.law_substeps, law_l1=a.law_l1, law_res_weight=a.law_res_weight,
                             law_global=a.law_global, law_cap=a.law_cap, law_lead=a.law_lead,
                             law_lead_nu_pos=a.law_lead_nu_pos,
                             law_head_n=a.law_head_n, law_head_weight=a.law_head_weight,
                             radius=radius, n_hist=a.n_hist, noise_channels=a.noise_channels,
                             global_kind=a.global_kind, global_k=a.global_k, global_hidden=a.global_hidden,
                             global_layers=a.global_layers, global_heads=a.global_heads, global_feat=a.global_feat)
    k_max = a.k_max or (4 if smoke else max(8, int(round(40 * 60 / dt)) if dt >= 300 else 20))
    sa = StageAConfig(window_sites=a.window_a, batch=4 if smoke else a.batch, steps=a.steps_a,
                      val_every=max(1, a.steps_a // 20), seed=a.seed, nested_frac=a.nested_frac)
    sb = StageBConfig(window_sites=max(a.window_b, 4 * reach + 4), halo_sites=reach, batch=4 if smoke else a.batch,
                      steps=a.steps_b, k_max=k_max, k_sampling=a.k_sampling, val_every=max(1, a.steps_b // 20),
                      seed=a.seed, crps_samples=a.crps_samples)
    if smoke:
        sa.val_batches = 2
        sa.warmup = sb.warmup = 1
    return RunConfig(profile=a.profile, source=a.source, n_val=a.n_val, n_test=a.n_test, seed=a.seed,
                     device=a.device, max_frames=a.max_frames, ae=ae, prop=prop, stage_a=sa, stage_b=sb,
                     eval_lead_min=a.eval_lead_min, eval_every_min=a.eval_every_min)


def level_heights_km(source: str, Z: int, log) -> np.ndarray:
    """Model level heights (km): the heights in an input sounding (CM1 writes zh there), taken from the
    ensemble's own member directories, or the original runs; level index if neither is found."""
    cands = [CM1 / "run" / "input_sounding"]
    if ":" in source:
        cands = sorted(Path(source.split(":", 1)[1]).glob("members/mem*/input_sounding"))[:1] + cands
    for f in cands:
        if f.exists():
            zc = np.loadtxt(f, skiprows=1)[:, 0] / 1000.0
            if len(zc) == Z:
                return zc
    log(f"  WARNING: no input_sounding with {Z} levels found; plotting against level index")
    return np.arange(Z, dtype=float)


def plot(results: dict, out: Path, sections: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fc = results["forecast"]
    lead = np.array(fc["lead_s"]) / 60
    col = {"model": "#2a6fdb", "pers": "#888888", "spread": "#d9822b"}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    for a_, (k, kp, ylab) in zip(ax[:2], [("nmse", "nmse_pers", "normalized MSE (all channels)"),
                                          ("th_sfc_rmse", "th_sfc_rmse_pers", "surface θ′ RMSE (K)")]):
        a_.plot(lead, fc[k], "-o", ms=3, lw=2, color=col["model"], label="learned model")
        a_.plot(lead, fc[kp], "--", lw=1.5, color=col["pers"], label="persistence")
        if k == "nmse" and "nmse_spread" in fc:
            a_.plot(lead, fc["nmse_spread"], ":", lw=2, color=col["spread"], label="truth-member spread")
    if "disp_abs_km_median" in fc:
        ax[2].fill_between(lead, fc["disp_abs_km_median"], fc["disp_abs_km_p90"], color=col["model"], alpha=0.15, lw=0,
                           label="model, median–90th pct")
        ax[2].plot(lead, fc["disp_abs_km_median"], "-", lw=2, color=col["model"], label="|displacement|, model")
        ax[2].plot(lead, fc["disp_pers_abs_km_median"], "--", lw=1.5, color=col["pers"], label="|displacement|, persistence")
        ax[2].plot(lead, fc["front_main_err_km_median"], ":", lw=1.5, color=col["model"], label="main-pool front error, model")
        ax[2].set_ylabel("front position error (km)")
        fk = "4km" if "4km" in fc["fss"]["0.3"] else sorted(fc["fss"]["0.3"])[0]
        for thr, ls in (("0.3", "-"), ("1", "--")):
            ax[3].plot(lead, fc["fss"][thr][fk], ls, lw=2, color=col["model"], label=f"model, q > {thr}")
            ax[3].plot(lead, fc["fss_pers"][thr][fk], ls, lw=1.5, color=col["pers"], label=f"persistence, q > {thr}")
        ax[3].set_ylabel(f"FSS, column-max condensate, {fk} window"); ax[3].set_ylim(0, 1)
        ax[2].legend(frameon=False, fontsize=7); ax[3].legend(frameon=False, fontsize=7)
    for a_ in ax:
        a_.set_xlabel("lead time (min)"); a_.grid(alpha=0.25)
    ax[0].set_ylabel("normalized MSE (all channels)"); ax[1].set_ylabel("surface θ′ RMSE (K)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Held-out members {', '.join(results['split']['test'])}: forecasts from the encoded truth, "
                 f"true state at the domain edges")
    fig.savefig(out / "fig_forecast.png", dpi=130)
    plt.close(fig)
    if sections:
        fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
        for a_, (lab, F) in zip(ax, sections["fields"].items()):
            m = a_.pcolormesh(sections["x"], sections["z"], F, cmap="RdBu_r", vmin=-8, vmax=8, shading="auto",
                              rasterized=True)
            a_.set_title(lab); a_.set_ylabel("z (km)"); a_.set_ylim(0, 12)
        ax[-1].set_xlabel("x (km)")
        fig.colorbar(m, ax=ax, label="θ′ (K)", shrink=0.8)
        fig.savefig(out / "fig_sections.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["smoke", "local", "full"], default="local")
    ap.add_argument("--source", default="local10")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=str(ROOT / "artifacts" / "cm1" / "members"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--val-members", default="", help="'+'-separated member names for validation (overrides --n-val)")
    ap.add_argument("--test-members", default="", help="'+'-separated member names for test (overrides --n-test)")
    ap.add_argument("--mirror", action="store_true",
                    help="D9: add the x-mirrored copy of every training member (reverse x, negate u')")
    ap.add_argument("--holdout-env", default="",
                    help="D10: environment box, e.g. 'qv_offset_gkg:-inf:-1.2+shear_scale:1.1:inf'; storms inside it are "
                         "kept out of training and validation and added to the test set (needs --manifest)")
    ap.add_argument("--manifest", default="",
                    help="'+'-separated PREFIX=PATH of ensemble manifest.csv files, e.g. ens_B_=~/cm1_env/ens_B_manifest.csv")
    ap.add_argument("--max-train", type=int, default=0,
                    help="use only this many training members (seeded random subset; val/test unchanged)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent-channels", type=int, default=16)
    ap.add_argument("--n-mix", type=int, default=2)
    ap.add_argument("--ae-kind", choices=["conv", "pca"], default="conv",
                    help="pca: fixed local PCA of each site's column block instead of the autoencoder (I-PCA)")
    ap.add_argument("--encoder-groups", type=int, default=1,
                    help="1: joint column encoder; 7: one encoder/decoder per variable (ablation)")
    ap.add_argument("--prop-reach", type=int, default=0, help="sites per step; 0 = from the frame interval")
    ap.add_argument("--window-a", type=int, default=32)
    ap.add_argument("--window-b", type=int, default=48)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps-a", type=int, default=None)
    ap.add_argument("--steps-b", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=0)
    ap.add_argument("--k-sampling", choices=["max", "uniform"], default="max",
                    help="rollout length per batch: always the curriculum maximum, or uniform in [2, max]")
    ap.add_argument("--eval-lead-min", type=float, default=60.0)
    ap.add_argument("--eval-boundary", choices=["truth", "persistence"], default="truth",
                    help="forecast boundary halo: the true state every step, or held at the start state (H0)")
    ap.add_argument("--eval-subset", default="",
                    help="'+'-separated name prefixes: evaluate only the test members starting with one of them "
                         "(e.g. ens_B_ or mem for ens_A); the split, and so the normalizer, is unchanged")
    ap.add_argument("--results-tag", default="",
                    help="write results_<tag>.json (and no figures) instead of results.json, e.g. for H0/H2 re-scores")
    ap.add_argument("--eval-every-min", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reuse-ae", default=None, help="run dir whose ae.pt to reuse (skips Stage A; Stage B still runs)")
    ap.add_argument("--n-hist", type=int, default=1, help="propagator history frames (D13)")
    ap.add_argument("--stage-c-crps", type=int, default=0,
                    help="Stage C with a stochastic propagator: fair CRPS over this many decoded rollouts")
    ap.add_argument("--stage-c-energy", type=float, default=0.0, help="SM2: weight of the energy score in Stage C")
    ap.add_argument("--stage-c-spectral", type=float, default=0.0, help="SM3: weight of the spectral term in Stage C")
    ap.add_argument("--prop-body", choices=["conv", "attention", "law"], default="conv",
                    help="propagator body: stencil convolutions, windowed self-attention (M1a), or an explicit latent "
                         "PDE (FPc: linear + quadratic terms + penalized residual)")
    ap.add_argument("--law-l1", type=float, default=0.0, help="FPc: L1 weight on the law's coefficients")
    ap.add_argument("--law-res-weight", type=float, default=0.0, help="FPc: weight of the residual's tendency share")
    ap.add_argument("--law-residual-hidden", type=int, default=64, help="FPc: residual MLP width (0: none)")
    ap.add_argument("--law-no-quadratic", action="store_true", help="FPc: linear law (+ residual) only")
    ap.add_argument("--law-cap", type=float, default=0.0,
                    help="FPc: saturating bound on the law's tendency, cap tanh(F/cap) per minute (0: off)")
    ap.add_argument("--law-global", action="store_true",
                    help="FPc: + G <z> (channel means over the sites the step sees); train with --window-b = domain width")
    ap.add_argument("--law-integrator", choices=["euler", "heun"], default="heun")
    ap.add_argument("--law-lead", choices=["", "z0"], default="",
                    help="FPf: channel 0's equation restricted to a closed short law in z0 alone (no residual)")
    ap.add_argument("--law-head-n", type=int, default=0, help="FPg: closed PDE head on the first n channels (0: off)")
    ap.add_argument("--law-head-weight", type=float, default=0.0, help="FPg: weight of the head's representation loss")
    ap.add_argument("--law-lead-nu-pos", action="store_true", help="FPf: channel 0's diffusion nu = softplus(.) >= 0")
    ap.add_argument("--law-substeps", type=int, default=1)
    ap.add_argument("--nested-frac", type=float, default=0.0,
                    help="FPc: nested dropout (importance-ordered channels) in Stages A and C: share of samples truncated")
    ap.add_argument("--prop-heads", type=int, default=4)
    ap.add_argument("--prop-hidden", type=int, default=0, help="propagator width (0: --hidden, which also sets the autoencoder's)")
    ap.add_argument("--global-kind", choices=["", "attn", "mean"], default="",
                    help="7a: global coupling of the leading channels (conv body); train with a wide --window-b")
    ap.add_argument("--global-k", type=int, default=8, help="7a: leading channels the global module sees")
    ap.add_argument("--global-hidden", type=int, default=64)
    ap.add_argument("--global-layers", type=int, default=2)
    ap.add_argument("--global-heads", type=int, default=4)
    ap.add_argument("--global-feat", type=int, default=16, help="7a: global features appended to the stencil input per site")
    ap.add_argument("--noise-channels", type=int, default=0, help="stochastic propagator noise channels (D14c)")
    ap.add_argument("--crps-samples", type=int, default=0, help="train the stochastic propagator with fair CRPS")
    ap.add_argument("--eval-samples", type=int, default=8, help="samples averaged when evaluating a stochastic propagator")
    ap.add_argument("--vertical-channels", type=int, default=0, help="vertical convolution front end (D14b)")
    ap.add_argument("--vertical-kernel", type=int, default=5)
    ap.add_argument("--init-from", default=None,
                    help="run dir: start from its trained autoencoder and propagator (skips Stages A and B)")
    ap.add_argument("--stage-c-steps", type=int, default=0, help="Stage C joint decoded-rollout fine-tuning (D12)")
    ap.add_argument("--stage-c-k", type=int, default=20)
    ap.add_argument("--stage-c-lr", type=float, default=1e-4)
    ap.add_argument("--stage-c-batch", type=int, default=12)
    ap.add_argument("--c-law-l1", type=float, default=-1.0, help="FPd: law L1 weight in Stage C (-1: the run's)")
    ap.add_argument("--c-law-res-end", type=float, default=-1.0,
                    help="FPd: ramp the law's residual-share weight to this over the first half of Stage C (-1: off)")
    ap.add_argument("--c-law-prune-every", type=int, default=0, help="FPd: sequential thresholding period (steps)")
    ap.add_argument("--c-law-prune-rel", type=float, default=0.02,
                    help="FPd: prune terms below this fraction of the largest coupling")
    ap.add_argument("--prefix-m", type=int, default=0,
                    help="5s: with --init-from and Stage C, prune the latent to its first M channels (the others become their training mean)")
    ap.add_argument("--distill-steps", type=int, default=0,
                    help="5t: with --prefix-m, first this many Stage C steps against the UNPRUNED model's rollouts (teacher), then the normal Stage C on CM1 data")
    ap.add_argument("--c-freeze-ae", action="store_true", help="Stage C updates the propagator only (M3: shared latent)")
    ap.add_argument("--c-site-smooth", type=float, default=0.0,
                    help="FPd: weight of the latent's second difference over sites (physical space, not channel index)")
    ap.add_argument("--w-sfc-th", type=float, default=0.0, help="Stage C extra weight on theta' in the lowest ~1 km")
    ap.add_argument("--env-file", default="",
                    help="environment features (scripts/cm1/build_env.py), '+'-separated files: condition the encoder (D11)")
    ap.add_argument("--average-props", nargs="*", default=[],
                    help="run dirs whose propagators share this autoencoder: evaluate the averaged forecast (D14a)")
    ap.add_argument("--eval-only", action="store_true", help="load ae.pt/prop.pt from --out, skip training")
    ap.add_argument("--data-device", default="same", help="'same' as the model, or 'cpu' (data in host memory)")
    a = ap.parse_args()
    defaults = {"smoke": (6, 6), "local": (4000, 4000), "full": (20000, 20000)}[a.profile]
    a.steps_a = a.steps_a or defaults[0]
    a.steps_b = a.steps_b or defaults[1]
    if a.profile == "smoke":
        a.max_frames = a.max_frames or 12
        a.device = "cpu" if a.device == "auto" else a.device  # smoke runs on CPU unless a device is asked for
        a.window_a, a.window_b = 12, 12
        a.n_val, a.n_test = 1, 1
        a.eval_lead_min, a.eval_every_min = 15, 15
    torch.set_num_threads(a.threads)
    out = Path(a.out or ROOT / "artifacts" / "cm1" / "runs" / f"{a.profile}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    t_start = time.time()
    set_seed(a.seed)
    dev = get_device(a.device)
    log(f"[cm1] profile {a.profile}, source {a.source}, device {dev}, out {out}")

    members = load_source(a.source, Path(a.cache), a.max_frames, log)
    if a.profile == "smoke":
        members = members[:3]
    names = [m.name for m in members]
    if a.val_members or a.test_members:
        va_n = [n for n in a.val_members.split("+") if n]
        te_n = [n for n in a.test_members.split("+") if n]
        missing = sorted(set(va_n + te_n) - set(names))
        if missing or set(va_n) & set(te_n) or not va_n or not te_n:
            raise ValueError(f"bad explicit split: missing {missing}, overlap {sorted(set(va_n) & set(te_n))}, "
                             f"{len(va_n)} val, {len(te_n)} test (both lists are needed)")
        tr_n = [n for n in names if n not in set(va_n + te_n)]
        if not tr_n:
            raise ValueError(f"explicit split leaves no training members ({len(names)} members: {names})")
    else:
        tr_n, va_n, te_n = split_members(names, a.n_val, a.n_test)
    if a.holdout_env:   # D10: a whole region of environment space out of training and validation
        from ks_latent.cm1.env import members_in_box, parse_box, read_manifest
        if not a.manifest:
            raise ValueError("--holdout-env needs --manifest")
        man = {}
        for item in a.manifest.split("+"):
            pre, _, path = item.partition("=")
            man.update(read_manifest(Path(path).expanduser(), pre))
        region = [n for n in members_in_box(man, parse_box(a.holdout_env)) if n in names]
        if not region:
            raise ValueError(f"no loaded storm lies in the box {a.holdout_env}")
        moved = [n for n in va_n if n in region]
        tr_n = [n for n in tr_n if n not in region]
        va_n = [n for n in va_n if n not in region]
        te_n = te_n + [n for n in region if n not in te_n]
        if not va_n or not tr_n:
            raise ValueError("the held-out region leaves no training or validation storms")
        log(f"[cm1] D10 held-out environment {a.holdout_env}: {len(region)} storms {region} out of training/validation"
            + (f" (moved from validation: {moved})" if moved else ""))
    if a.max_train and a.max_train < len(tr_n):
        rs = np.random.default_rng(1000 + a.seed)
        tr_n = sorted(rs.choice(tr_n, a.max_train, replace=False).tolist(), key=_natural_key)
    pick = lambda ns: [m for m in members if m.name in ns]
    if a.eval_only and (out / "normalizer.json").exists():   # exactly the scaling the model was trained with
        norm = Normalizer.from_dict(json.loads((out / "normalizer.json").read_text()))
        fit = Normalizer.fit(pick(tr_n))
        if not np.allclose(fit.scale, norm.scale, rtol=1e-6) or tuple(fit.vars) != tuple(norm.vars):
            raise ValueError(f"the split given refits a different normalizer ({fit.scale} vs saved {norm.scale}): "
                             "pass the run's own --val-members / --test-members")
    else:
        norm = Normalizer.fit(pick(tr_n))
    dt = members[0].dt
    cfg = build_config(a, dt)
    Z = members[0].X.shape[3]
    cfg.ae.n_channels, cfg.ae.n_levels = members[0].X.shape[2] * Z, Z  # before the config is saved (7 or 10 variables)
    if a.eval_only and (out / "config.json").exists() and not (out / "config.json.trained").exists():
        (out / "config.json.trained").write_text((out / "config.json").read_text())
    if not a.eval_only:
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    (out / "normalizer.json").write_text(json.dumps(norm.to_dict(), indent=1))
    log(f"[cm1] split train {tr_n} | val {va_n} | test {te_n}; dt {dt:.0f} s; scales "
        + ", ".join(f"{v} {s:.3g}" for v, s in zip(norm.vars, norm.scale)))
    log(f"[cm1] AE patch {cfg.ae.patch} ({cfg.ae.patch * 0.25:g} km sites), c {cfg.ae.latent_channels}, reach "
        f"{cfg.ae.reach_sites}; propagator reach {cfg.prop.reach_sites} sites/step "
        f"({cfg.prop.reach_sites * cfg.ae.patch * 0.25:g} km), k_max {cfg.stage_b.k_max}")
    data_dev = dev if a.data_device == "same" else torch.device(a.data_device)
    log(f"[cm1] data on {data_dev}, model on {dev}")
    env = None
    if a.env_file:
        from ks_latent.cm1.env import load_env_files
        env = load_env_files(a.env_file.split("+"))
        cfg.ae.env_raw_dim = len(next(iter(env.values())))
        log(f"[cm1] environment-conditioned encoder: {cfg.ae.env_raw_dim} features per member from {a.env_file}")
        if not a.eval_only and not a.init_from:
            (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    if a.mirror:   # training members only; val/test stay real storms
        from ks_latent.cm1.data import mirror_member
        mir = [mirror_member(m) for m in pick(tr_n)]
        members += mir
        tr_n = tr_n + [m.name for m in mir]
        if env is not None:
            from ks_latent.cm1.env import mirror_env
            for m in mir:
                env[m.name] = mirror_env(env[mirror_member(m).name])
        log(f"[cm1] D9 mirrors: {len(mir)} mirrored training members added ({len(tr_n)} in training)")
    te_eval = te_n
    if a.eval_subset:
        pre = tuple(p for p in a.eval_subset.split("+") if p)
        te_eval = [n for n in te_n if n.startswith(pre)]
        if not te_eval:
            raise ValueError(f"no test member starts with {pre}: {te_n}")
        log(f"[cm1] evaluating on {len(te_eval)} of {len(te_n)} test members: {te_eval}")
    parts = (("test", te_eval),) if a.eval_only else (("train", tr_n), ("val", va_n), ("test", te_eval))
    data = {k: ColumnData(pick(n), norm, data_dev, env=env) for k, n in parts}   # eval-only needs the test set only
    del members

    if a.init_from:   # architecture of the run we start from
        saved = json.loads((Path(a.init_from) / "config.json").read_text())
        cfg.ae, cfg.prop = ColumnAEConfig(**saved["ae"]), StencilPropConfig(**saved["prop"])
        cfg.stage_b = StageBConfig(**saved["stage_b"])
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    if a.eval_only:
        saved = json.loads((out / "config.json.trained").read_text() if (out / "config.json.trained").exists()
                           else (out / "config.json").read_text())
        cfg.ae, cfg.prop = ColumnAEConfig(**saved["ae"]), StencilPropConfig(**saved["prop"])
        cfg.stage_a, cfg.stage_b = StageAConfig(**saved["stage_a"]), StageBConfig(**saved["stage_b"])
        log(f"[cm1] eval only: architecture from the run's saved config (c {cfg.ae.latent_channels}, patch "
            f"{cfg.ae.patch}, hidden {cfg.ae.hidden}, halo {cfg.stage_b.halo_sites})")
    ae = build_autoencoder(cfg.ae, positive_channel_mask(Z, norm.vars)).to(dev)
    prop = StencilPropagator(cfg.prop).to(dev)
    if a.eval_only or a.init_from:
        src = out if a.eval_only else Path(a.init_from)
        for mod, f in ((ae, "ae.pt"), (prop, "prop.pt")):
            ck = torch.load(src / f, map_location=dev)
            mod.load_state_dict(ck["state_dict"])
        ra = json.loads((src / "history_a.json").read_text())
        rb = json.loads((src / "history_b.json").read_text())
        log(f"[cm1] loaded ae.pt and prop.pt from {src}")
        if a.prefix_m:
            if not a.init_from or cfg.ae.prefix_m:
                raise ValueError("--prefix-m needs --init-from an unpruned run")
            if a.distill_steps:
                import copy
                teacher = (copy.deepcopy(ae).eval(), copy.deepcopy(prop).eval())
            zt = torch.cat([x[60:].reshape(-1, x.shape[-1]) for x in encode_members(ae, data["train"])])
            fill = zt.mean(0).cpu()
            ae.set_prefix(a.prefix_m, fill); prop.set_prefix(a.prefix_m, fill)
            ae.to(dev); prop.to(dev)
            (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
            log(f"[cm1] 5s: latent pruned to the first {a.prefix_m} of {fill.numel()} channels; the rest fixed at their training mean")
        if a.init_from and a.stage_c_steps:
            from ks_latent.cm1.train import StageCConfig, train_stage_c
            sc = StageCConfig(window_sites=max(cfg.stage_b.window_sites, 4 * cfg.stage_b.halo_sites + 8),
                              halo_sites=cfg.stage_b.halo_sites, batch=a.stage_c_batch, steps=a.stage_c_steps,
                              lr=a.stage_c_lr, k_max=a.stage_c_k, w_sfc_th=a.w_sfc_th, seed=a.seed,
                              crps_samples=a.stage_c_crps, w_energy=a.stage_c_energy, w_spectral=a.stage_c_spectral,
                              val_every=max(1, a.stage_c_steps // 16), nested_frac=a.nested_frac,
                              law_l1=a.c_law_l1, law_res_weight_end=a.c_law_res_end,
                              law_prune_every=a.c_law_prune_every, law_prune_rel=a.c_law_prune_rel,
                              w_site_smooth=a.c_site_smooth, freeze_ae=a.c_freeze_ae)
            log(f"[C] joint decoded-rollout fine-tuning: {sc}")
            tc = time.time()
            if a.distill_steps:
                if not a.prefix_m:
                    raise ValueError("--distill-steps needs --prefix-m")
                import dataclasses as _dc
                sd = _dc.replace(sc, steps=a.distill_steps, val_every=max(1, a.distill_steps // 8))
                log(f"[D] distillation against the unpruned model: {a.distill_steps} steps")
                rd = train_stage_c(ae, prop, data["train"], data["val"], sd, log=log, teacher=teacher)
                (out / "history_distill.json").write_text(json.dumps(rd, indent=1))
                torch.save({"state_dict": ae.state_dict(), "cfg": dataclasses.asdict(cfg.ae)}, out / "ae_distilled.pt")
                torch.save({"state_dict": prop.state_dict(), "cfg": dataclasses.asdict(cfg.prop)}, out / "prop_distilled.pt")
                del teacher
            rcst = train_stage_c(ae, prop, data["train"], data["val"], sc, log=log)
            (out / "history_c.json").write_text(json.dumps(rcst, indent=1))
            for mod, f, c_ in ((ae, "ae.pt", cfg.ae), (prop, "prop.pt", cfg.prop)):
                torch.save({"state_dict": mod.state_dict(), "cfg": dataclasses.asdict(c_)}, out / f)
                write_provenance(out / f, config=cfg, seed=a.seed, device=str(dev), wall_time_s=time.time() - tc)
            for f, r_ in (("history_a.json", ra), ("history_b.json", rb)):
                (out / f).write_text(json.dumps(r_, indent=1))
        z = {k: encode_members(ae, d) for k, d in data.items()}
    else:
        # ---- Stage A ----
        ta = time.time()
        if a.reuse_ae:
            src = Path(a.reuse_ae)
            ck = torch.load(src / "ae.pt", map_location=dev)
            if ColumnAEConfig(**ck["cfg"]) != cfg.ae:  # compare with defaults filled (older runs lack newer fields)
                raise ValueError(f"autoencoder config in {src} differs from this run's: {ck['cfg']} vs {cfg.ae}")
            ae.load_state_dict(ck["state_dict"])
            ra = json.loads((src / "history_a.json").read_text())
            ra["reused_from"] = str(src)
            log(f"[A] reusing the autoencoder of {src}")
        elif cfg.ae.kind == "pca":
            ra = fit_pca_stage(ae, data["train"], data["val"], log=log)
        else:
            log(f"[A] autoencoder: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M parameters")
            ra = train_stage_a(ae, data["train"], data["val"], cfg.stage_a, log=log)
        torch.save({"state_dict": ae.state_dict(), "cfg": dataclasses.asdict(cfg.ae)}, out / "ae.pt")
        write_provenance(out / "ae.pt", config=cfg, seed=a.seed, device=str(dev), wall_time_s=time.time() - ta)
        (out / "history_a.json").write_text(json.dumps(ra, indent=1))

        # ---- Stage B (always run) ----
        z = {k: encode_members(ae, d) for k, d in data.items()}
        act = {k: d.active for k, d in data.items()}
        log(f"[B] propagator: {sum(p.numel() for p in prop.parameters()) / 1e6:.2f} M parameters")
        tb = time.time()
        rb = train_stage_b(prop, z["train"], z["val"], act["train"], act["val"], cfg.ae.patch, cfg.stage_b, log=log)
        torch.save({"state_dict": prop.state_dict(), "cfg": dataclasses.asdict(cfg.prop)}, out / "prop.pt")
        write_provenance(out / "prop.pt", config=cfg, seed=a.seed, device=str(dev), wall_time_s=time.time() - tb)
        (out / "history_b.json").write_text(json.dumps(rb, indent=1))

    # ---- test report ----
    n_lead = max(1, int(round(cfg.eval_lead_min * 60 / dt)))
    every = max(1, int(round(cfg.eval_every_min * 60 / dt)))
    T_test = min(data["test"].T)
    if every + n_lead >= T_test:  # a short run (smoke): shrink the evaluation to fit
        n_lead, every = max(1, T_test // 3), max(1, T_test // 3)
        log(f"  WARNING: test runs have only {T_test} frames; evaluating {n_lead} leads every {every} frames")
    t0s = list(range(every, T_test - n_lead, every))
    rec = reconstruction_report(ae, data["test"], norm)
    props = [prop]
    for d in a.average_props:
        ck_a = torch.load(Path(d) / "ae.pt", map_location=dev)
        if any(not torch.equal(ck_a["state_dict"][k].to(dev), v) for k, v in ae.state_dict().items()):
            raise ValueError(f"{d} does not share this autoencoder: averaging needs --reuse-ae runs")
        pc = json.loads((Path(d) / "config.json").read_text())["prop"]
        p2 = StencilPropagator(StencilPropConfig(**pc)).to(dev)
        p2.load_state_dict(torch.load(Path(d) / "prop.pt", map_location=dev)["state_dict"])
        props.append(p2)
    if len(props) > 1:
        log(f"[test] averaging the decoded forecasts of {len(props)} propagators")
    fc = forecast_report(ae, props if len(props) > 1 else prop, data["test"], z["test"], norm, t0s, n_lead,
                         cfg.stage_b.halo_sites, n_samples=a.eval_samples, boundary=a.eval_boundary)
    results = {"split": {"train": tr_n, "val": va_n, "test": te_n}, "eval_test": te_eval, "dt_s": dt, "reconstruction": rec,
               "forecast": fc, "stage_a_best_val": ra["best_val_recon"], "stage_b_best_val": rb["best_val_mse"]}
    res_name = f"results_{a.results_tag}.json" if a.results_tag else "results.json"
    (out / res_name).write_text(json.dumps(results, indent=1))
    write_provenance(out / res_name, config=cfg, seed=a.seed, device=str(dev),
                     wall_time_s=time.time() - t_start, extra={"eval_boundary": a.eval_boundary,
                                                               "eval_lead_min": a.eval_lead_min})
    log("[test] reconstruction RMSE (physical units), storm columns: "
        + ", ".join(f"{k} {v:.3g}" for k, v in rec.get("rmse_phys_storm", rec["rmse_phys_all"]).items()))
    log("[test] forecasts (mean over start times " + ", ".join(f"{t * dt / 3600:.1f} h" for t in t0s) + "):")
    log(summarize(fc))
    log(summarize_fronts(fc))
    log(summarize_physics(fc))
    if rec.get("physics"):
        log("[test] reconstruction physics: " + ", ".join(f"{k} {v:.3f}" for k, v in rec["physics"].items()))

    # a section: truth vs forecast theta' at the last lead from the first start time, first test member
    sections = None
    if t0s:
        with torch.no_grad():
            m, t0 = 0, t0s[0]
            from ks_latent.cm1.models import rollout_from
            lat = rollout_from(prop, z["test"][m], t0, n_lead, cfg.stage_b.halo_sites,
                               a.eval_samples if prop.cfg.noise_channels else 1, gen=torch.Generator().manual_seed(0))
            state = lat[-1]                                   # (samples, P, c) at the last lead
            ith = VARS.index("th")
            sc = float(norm.scale[ith])
            sl = slice(ith * Z, (ith + 1) * Z)
            truth = data["test"].full(m, t0 + n_lead, device=dev)[0, sl].cpu().numpy() * sc
            pred = ae.decode(state).mean(0)[sl].cpu().numpy() * sc
            e0 = data["test"].env_of([m])
            recon = ae(data["test"].full(m, t0 + n_lead, device=dev), None if e0 is None else e0.to(dev))[0][0, sl].cpu().numpy() * sc
            zc = level_heights_km(a.source, Z, log)
            sections = {"x": (np.arange(truth.shape[1]) + 0.5) * 0.25, "z": zc,
                        "fields": {f"truth, t = {(t0 + n_lead) * dt / 3600:.1f} h ({te_eval[m]})": truth,
                                   "autoencoder reconstruction of the truth": recon,
                                   f"forecast from t = {t0 * dt / 3600:.1f} h, lead {n_lead * dt / 60:.0f} min": pred}}
    if not a.results_tag:
        plot(results, out, sections)
    log(f"[cm1] done in {(time.time() - t_start) / 60:.1f} min; results in {out}")


if __name__ == "__main__":
    main()
