"""10c: cycled DA twin with the 4-model pool (independently trained F6 seeds), against one model and no DA.

The pooled forecast system has one latent space per model, so the analysis is done where the members meet: on the decoded
columns. Each cycle every model advances its own members (its learned noise, boundaries held at t0), the 32 members are
decoded into one physical ensemble (the 4 models share one normalizer), a per-cell LETKF updates them, and each analysis
member is re-encoded by the autoencoder of the model that carries it. Systems (--systems):
  pool_phys     4 models x 8 members, physical-space LETKF, re-encoded per model (the method)
  single_phys   model 0 alone, 32 members, the same physical-space LETKF (isolates the pooling)
  single_latent model 0 alone, 32 members, latent LETKF (the G0 headline analysis, isolates the analysis space)
  single_pff    model 0 alone, 32 members, domain-localized latent particle flow filter (ks_latent/cm1/pff_site.py)
  pool_pff      4 models x 8 members, one latent PFF per model: all 32 decoded members are encoded by model k (cross-encoding),
                so model k's prior covariance comes from 32 members; model k keeps the analysis of its own 8 (whose latents are its
                own, not re-encoded)
plus the no-DA runs of the pool and of model 0 from the same initial ensemble. Setup as the G0 headline: theta' (surface) +
composite reflectivity + radial velocity every 2 km, 5-min cycles for 3 h from t = 60 min, localization 5 km, held
boundaries, 32 members drawn from training storms (here: those present in --source). Reflectivity uses --refl-run's
learned operator (trained on true columns; inputs re-scaled from the pool's normalizer to its own).

Free forecasts from the analyses after --forecast-from cycles (60 min, G3) are scored for the ensemble mean, the 10 km scale
blend (mean > 10 km + member 0 < 10 km; the recommended single field, postproc_eval.py), and member 0: nMSE (normalized
units, condensate in sqrt) and power / truth at 2-10 km wavelength. The no-DA ensembles get the same forecast scores from
the same times.

  python scripts/cm1/da_pool.py --runs R0 R1 R2 R3 --refl-run ~/cm1_runs/full_v1_stagec --split test --out DIR --job 0 --n-jobs 11
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from da_twin import load_run, member_files  # noqa: E402
from ks_latent.cm1.data import VARS, ColumnData, Normalizer, _natural_key, load_member  # noqa: E402
from ks_latent.cm1.letkf import LETKFConfig, letkf_analysis_torch  # noqa: E402
from ks_latent.cm1.pff_site import PFFSiteConfig, pff_site_analysis  # noqa: E402
from ks_latent.cm1.train import encode_members  # noqa: E402
from ks_latent.cm1.twin import _forecast, _scores  # noqa: E402

DX = 0.25
SYSTEMS = ("pool_phys", "single_phys", "single_latent", "single_pff", "pool_pff")


def lowpass(f, sigma_km):
    """Gaussian low-pass along the last axis (edges replicated); the same filter as postproc_eval.py"""
    s = sigma_km / DX; r = int(np.ceil(3 * s))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=f.dtype, device=f.device) / s) ** 2); k = k / k.sum()
    sh = f.shape
    y = torch.nn.functional.pad(f.reshape(-1, 1, sh[-1]), (r, r), mode="replicate")
    return torch.nn.functional.conv1d(y, k.view(1, 1, -1)).reshape(sh)


def band_power(f, lo_km=2.0, hi_km=10.0):
    """power of (C, NX) along x in wavelengths [lo, hi] km, summed over channels"""
    F = torch.fft.rfft(f - f.mean(-1, keepdim=True), dim=-1)
    lam = 1.0 / torch.clamp(torch.fft.rfftfreq(f.shape[-1], d=DX), min=1e-9)
    sel = (lam >= lo_km) & (lam <= hi_km)
    return float((F[..., sel].abs() ** 2).sum())


class System:
    """models: list of (ae, prop, z_truth); owner: (N,) model index of each member"""

    def __init__(self, models, Z0s, space):
        self.models, self.space = models, space
        self.Z = [z.clone() for z in Z0s]

    def decode(self):
        return torch.cat([m[0].decode(z) for m, z in zip(self.models, self.Z)])

    def forecast(self, t, steps, halo, hold, gen):
        self.Z = [_forecast(m[1], z, m[2], t, steps, halo, 0.0, gen, hold=hold) for m, z in zip(self.models, self.Z)]

    def clamp(self, k, z):
        """clamp model k's latent to its training range (per channel); returns (z, fraction of entries clipped)"""
        b = self.models[k][3]
        if b is None:
            return z, 0.0
        zc = torch.maximum(torch.minimum(z, b[1]), b[0])
        return zc, float((zc != z).float().mean())

    def set_physical(self, Ua):
        i, fr = 0, []
        for k, (m, z) in enumerate(zip(self.models, self.Z)):
            n = z.shape[0]
            self.Z[k], f = self.clamp(k, m[0].encode(Ua[i:i + n])); i += n; fr.append(f)
        return float(np.mean(fr))


def main():
    H = Path.home()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--refl-run", default=f"{H}/cm1_runs/full_v1_stagec")
    ap.add_argument("--source", default=f"cache:{H}/cm1_cache+{H}/cm1_cache_A2+{H}/cm1_cache_B")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    ap.add_argument("--rtps", type=float, nargs="+", default=[0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--loc-km", type=float, default=5.0)
    ap.add_argument("--obs-km", type=float, default=2.0); ap.add_argument("--sigma-obs", type=float, default=0.5)
    ap.add_argument("--sigma-refl", type=float, default=5.0); ap.add_argument("--sigma-vr", type=float, default=2.0)
    ap.add_argument("--vr-tilts", type=float, nargs="+", default=[0.5, 2.0, 5.0]); ap.add_argument("--radar-x-km", type=float, default=330.0)
    ap.add_argument("--t0-min", type=float, default=60.0); ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--cycle-min", type=float, default=5.0)
    ap.add_argument("--forecast-from", type=int, nargs="*", default=[11, 17, 23, 29])
    ap.add_argument("--forecast-minutes", type=int, default=60)
    ap.add_argument("--n-cycles", type=int, default=0, help="override (smoke)")
    ap.add_argument("--max-truths", type=int, default=0)
    ap.add_argument("--truths", nargs="*", default=[], help="these storms only (overrides --split's list)")
    ap.add_argument("--verbose", action="store_true", help="per-cycle diagnostics")
    ap.add_argument("--clamp-margin", type=float, default=0.1,
                    help="clamp each analysis latent channel to the model's range on the ensemble's training storms, widened by "
                         "this fraction of the range on each side (negative: no clamp)")
    ap.add_argument("--pff-loc-km", type=float, default=3.0); ap.add_argument("--pff-taper-km", type=float, default=4.0)
    ap.add_argument("--pff-half-width", type=int, default=6); ap.add_argument("--pff-steps", type=int, default=12)
    ap.add_argument("--job", type=int, default=0); ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    dev = torch.device("cpu")
    t_start = time.time()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    loaded = [load_run(Path(r).expanduser(), dev) for r in a.runs]
    norm = loaded[0][2]
    for L in loaded[1:]:
        if not np.allclose(L[2].scale, norm.scale):
            raise ValueError("the pooled models must share one normalizer")
    split, halo = loaded[0][4], loaded[0][3]["stage_b"]["halo_sites"]
    files = member_files(a.source)
    # reflectivity operator, re-scaled from the pool's normalizer to its own
    from ks_latent.cm1.refl import ReflectivityHead, ReflHeadConfig
    rr = Path(a.refl_run).expanduser()
    ck = torch.load(rr / "refl_head.pt", map_location=dev)
    head = ReflectivityHead(ReflHeadConfig(**ck["cfg"])); head.load_state_dict(ck["state_dict"]); head.eval()
    nr = Normalizer.from_dict(json.loads((rr / "normalizer.json").read_text()))
    if tuple(nr.vars) != tuple(norm.vars):
        raise ValueError("reflectivity head variables differ from the pool's")
    truths = sorted([n for n in split[a.split] if n in files], key=_natural_key)
    missing = [n for n in split[a.split] if n not in files]
    if missing:
        print(f"[pool-da] {len(missing)} {a.split} storms not in --source, skipped: {missing}")
    truths = (list(a.truths) or truths[: a.max_truths or None])[a.job::a.n_jobs]
    pool_train = sorted([n for n in split["train"] if n in files], key=_natural_key)
    ens_names = sorted(np.random.default_rng(0).choice(pool_train, a.members, replace=False).tolist(), key=_natural_key)
    others = ColumnData([load_member(files[n]) for n in ens_names], norm, dev)
    Zl, NX, dt = others.Z, others.NX, others.dt
    ratio = torch.as_tensor(np.repeat(norm.scale / nr.scale, Zl), dtype=torch.float32)[None, :, None]
    refl_fn = lambda d: head(d * ratio)
    t0, spc = int(round(a.t0_min * 60 / dt)), int(round(a.cycle_min * 60 / dt))
    n_cycles = a.n_cycles or int(round(a.hours * 60 / a.cycle_min))
    nf = int(round(a.forecast_minutes * 60 / dt))
    nm = len(loaded); per = a.members // nm
    Z0_single = torch.stack([z[t0] for z in encode_members(loaded[0][0], others)])
    Z0_pool = []
    for k, L in enumerate(loaded):   # model k carries members k*per .. (k+1)*per - 1, encoded by its own autoencoder
        sub = ColumnData([load_member(files[n]) for n in ens_names[k * per:(k + 1) * per]], norm, dev)
        Z0_pool.append(torch.stack([z[t0] for z in encode_members(L[0], sub)]))
    bounds = []
    for k, L in enumerate(loaded):   # per-channel latent range of each model on the 32 training storms (every 5th frame)
        if a.clamp_margin < 0:
            bounds.append(None); continue
        zz = torch.cat([z[::5].reshape(-1, z.shape[-1]) for z in encode_members(L[0], others)])
        lo, hi = zz.min(0).values, zz.max(0).values; w = hi - lo
        bounds.append((lo - a.clamp_margin * w, hi + a.clamp_margin * w))
        print(f"[pool-da] model {k}: latent range {float(lo.min()):.2f} .. {float(hi.max()):.2f}", flush=True)
    from ks_latent.cm1.obs import RadialVelocity
    from ks_latent.cm1.physics import BaseState
    obs_every = int(round(a.obs_km / DX))
    obs_cols = np.arange(obs_every // 2, NX, obs_every)
    x_obs, x_cell = (obs_cols + 0.5) * DX, (np.arange(NX) + 0.5) * DX
    vr_op = RadialVelocity(norm, BaseState.from_sounding().z, NX, radar_x_km=a.radar_x_km, tilts_deg=a.vr_tilts,
                           every_cells=obs_every, col_offset=0)
    ith, s_th = VARS.index("th") * Zl, float(norm.scale[VARS.index("th")])
    pos_mask = loaded[0][0].positive_mask[None, :, None]
    P = Z0_single.shape[1]
    x_site = (np.arange(P) + 0.5) * loaded[0][0].cfg.patch * DX
    print(f"[pool-da] {nm} models, {a.members} members ({per} per model in the pool), {len(truths)} {a.split} truths "
          f"(job {a.job}/{a.n_jobs}), systems {a.systems}, rtps {a.rtps}, seeds {a.seeds}, {n_cycles} cycles", flush=True)

    def fc_scores(sysm, truth, t, gen):
        """free forecast from the system's current members; per lead: mean / blend10 / member nMSE and 2-10 km power ratio"""
        Zs = [z.clone() for z in sysm.Z]
        fs = System(sysm.models, Zs, sysm.space)
        rec = {k: [] for k in ("nmse_mean", "nmse_blend10", "nmse_member", "pw_mean", "pw_blend10", "pw_member")}
        for j in range(nf):
            fs.forecast(t + j, 1, halo, t0, gen)
            if (j + 1) % 5:
                continue
            dec = fs.decode(); u = truth.full(0, t + j + 1, device=dev)[0]
            mean = dec.mean(0); mem = dec[0]
            b10 = lowpass(mean, 1.9) + mem - lowpass(mem, 1.9)
            pt = band_power(u)
            for nme, f in (("mean", mean), ("blend10", b10), ("member", mem)):
                rec["nmse_" + nme].append(float(((f - u) ** 2).mean()))
                rec["pw_" + nme].append(band_power(f) / max(pt, 1e-12))
        return rec

    def run(sysm, truth, seed, rtps, assimilate, tag=""):
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        cfg = LETKFConfig(loc_km=a.loc_km, rtps=rtps)
        cycles, fcs = [], {}
        for k in range(n_cycles):
            t = t0 + k * spc; t1 = t + spc
            sysm.forecast(t, spc, halo, t0, gen)
            u_true = truth.full(0, t1, device=dev)[0]
            y = (u_true[ith, obs_cols] * s_th).double().numpy() + rng.normal(0.0, a.sigma_obs, obs_cols.size)
            cref_true = truth.cref[0][t1].clamp(min=0).double().numpy()
            y_refl = np.clip(cref_true[obs_cols] + rng.normal(0.0, a.sigma_refl, obs_cols.size), 0.0, None)
            vr_true = vr_op.truth_values(u_true); y_vr = vr_true + rng.normal(0.0, a.sigma_vr, vr_true.size)
            dec = sysm.decode()
            if a.verbose:
                zmax = max(float(z.abs().max()) for z in sysm.Z)
                print(f"    [{tag}] cycle {k}: prior nmse {float(((dec.mean(0) - u_true) ** 2).mean()):.3f}  max|dec| {float(dec.abs().max()):.3g}"
                      f"  max|z| {zmax:.3g}  member nmse max {float(((dec - u_true) ** 2).mean((1, 2)).max()):.3g}", flush=True)
            sk = dict(refl_head=refl_fn, refl_cols=obs_cols, cref_true=cref_true, y_refl=None, vr_op=vr_op, vr_true=vr_true)
            rec = {"prior": _scores(None, None, u_true, obs_cols, None, norm, Zl, dec=dec, **sk)}
            rec["prior"]["spread"] = float(dec.var(0, unbiased=True).mean())
            if assimilate:
                ops = [lambda d_: d_[:, ith, obs_cols] * s_th, lambda d_: refl_fn(d_)[:, obs_cols], lambda d_: vr_op._from_columns(d_)]
                Ys_ = [op(dec).double().numpy() for op in ops]
                Yall = np.concatenate(Ys_, 1)
                if a.verbose:
                    print(f"    [{tag}] cycle {k}: obs-space spread max th {Ys_[0].std(0).max():.3g} refl {Ys_[1].std(0).max():.3g} "
                          f"vr {Ys_[2].std(0).max():.3g}; finite {np.isfinite(Yall).all()}", flush=True)
                yall = np.concatenate([y, y_refl, y_vr])
                rall = np.concatenate([np.full(obs_cols.size, a.sigma_obs ** 2), np.full(obs_cols.size, a.sigma_refl ** 2),
                                       np.full(y_vr.size, a.sigma_vr ** 2)])
                xall = np.concatenate([x_obs, x_obs, vr_op.x_km])
                if sysm.space in ("pff", "pool_pff"):
                    pcfg = PFFSiteConfig(loc_km=a.pff_loc_km, taper_km=a.pff_taper_km, half_width=a.pff_half_width,
                                         n_steps=a.pff_steps, rtps=rtps)
                    clips, tp = [], time.time()
                    for mk_, (m_, z_) in enumerate(zip(sysm.models, sysm.Z)):
                        ae_ = m_[0]
                        obs_fn = lambda zz, ae_=ae_: torch.cat([op(ae_.decode(zz[None])) for op in ops], 1)[0]
                        if sysm.space == "pool_pff":     # model k sees all 32 members: its own latents + the others' decoded members encoded by it
                            n_ = z_.shape[0]; i0 = sum(z.shape[0] for z in sysm.Z[:mk_])
                            others_ = torch.cat([dec[:i0], dec[i0 + n_:]])
                            Zk = torch.cat([z_, sysm.clamp(mk_, ae_.encode(others_))[0]])
                        else:
                            Zk = z_
                        with torch.enable_grad():
                            Za, pinfo = pff_site_analysis(Zk, obs_fn, yall, rall, x_site, xall, pcfg, gen=gen)
                        sysm.Z[mk_], c_ = sysm.clamp(mk_, Za[: z_.shape[0]]); clips.append(c_)
                    clip = float(np.mean(clips))
                    if a.verbose:
                        print(f"    [{tag}] cycle {k}: PFF {time.time() - tp:.1f}s, {pinfo['n_iterations']} pseudo-steps, "
                              f"max obs/site {pinfo['n_obs_per_site_max']}", flush=True)
                    dec_a = sysm.decode()
                elif sysm.space == "latent":
                    Zb = sysm.Z[0]
                    Za = letkf_analysis_torch(Zb.double().numpy(), Yall, yall, rall, x_site, xall, cfg, device=dev)
                    sysm.Z[0], clip = sysm.clamp(0, torch.as_tensor(Za, dtype=Zb.dtype))
                    dec_a = sysm.decode()
                else:
                    Ua = letkf_analysis_torch(dec.permute(0, 2, 1).double().numpy(), Yall, yall, rall, x_cell, xall, cfg, device=dev)
                    Ua = torch.as_tensor(Ua, dtype=dec.dtype).permute(0, 2, 1).contiguous()
                    Ua = torch.where(pos_mask, Ua.clamp(min=0), Ua)
                    if a.verbose:
                        U0 = letkf_analysis_torch(dec.permute(0, 2, 1).double().numpy(), Yall, yall, rall, x_cell, xall,
                                                  LETKFConfig(loc_km=a.loc_km, rtps=0.0), device=dev)
                        U0 = torch.as_tensor(U0, dtype=dec.dtype).permute(0, 2, 1)
                        sb, sa0, sa = dec.std(0), U0.std(0), Ua.std(0)
                        V = len(norm.vars)
                        per_v = lambda t: [round(float(x), 2) for x in t.reshape(t.shape[0], V, Zl, -1).abs().amax((0, 2, 3))] if t.dim() == 3 else None
                        print(f"    [{tag}] cycle {k}: max|Ua| per var {per_v(Ua)}  no-RTPS {per_v(U0)}  prior {per_v(dec)}; "
                              f"max sa/sb {float((sa / sb.clamp(min=1e-12)).max()):.3g}, max sa0/sb {float((sa0 / sb.clamp(min=1e-12)).max()):.3g}", flush=True)
                        for mi, (m, z) in enumerate(zip(sysm.models, sysm.Z)):
                            n = z.shape[0]; sl = slice(mi * n, (mi + 1) * n)
                            print(f"      model {mi}: max|z| prior {float(z.abs().max()):.3g}, re-encoded analysis {float(m[0].encode(Ua[sl]).abs().max()):.3g}, "
                                  f"no-RTPS {float(m[0].encode(U0[sl]).abs().max()):.3g}, re-encoded prior {float(m[0].encode(dec[sl]).abs().max()):.3g}", flush=True)
                    clip = sysm.set_physical(Ua)
                    dec_a = sysm.decode()          # the re-encoded analysis: what the next forecast starts from
                rec["analysis"] = _scores(None, None, u_true, obs_cols, None, norm, Zl, dec=dec_a, **sk)
                rec["analysis"]["spread"] = float(dec_a.var(0, unbiased=True).mean())
                rec["analysis"]["clip_frac"] = clip
            for st in rec:
                rec[st].pop("_mean_dec", None)
            if k in a.forecast_from:
                fcs[k] = fc_scores(sysm, truth, t1, gen)
            cycles.append(rec)
        half = cycles[len(cycles) // 2:]
        summ = {}
        for st in ("prior", "analysis"):
            if st not in half[0]:
                continue
            g = lambda key: float(np.nanmean([c[st][key] for c in half]))
            summ[st] = {key: g(key) for key in ("nmse_mean_dec", "th_obs_rmse", "th_obs_spread", "refl_obs_rmse",
                                                 "refl_echo_rmse", "vr_obs_rmse", "vr_obs_spread", "spread")}
            summ[st]["front_abs_disp_km_median"] = float(np.nanmedian([abs(c[st]["front_disp_km"]) for c in half]))
            summ[st]["rmse_phys"] = {v: float(np.mean([c[st]["rmse_phys_mean_dec"][v] for c in half])) for v in norm.vars}
        summ["nmse_by_cycle"] = [c["prior"]["nmse_mean_dec"] for c in cycles]
        if assimilate:
            cf = [c["analysis"]["clip_frac"] for c in cycles]
            summ["clip_frac_mean"], summ["clip_frac_max"], summ["cycles_clipped"] = float(np.mean(cf)), float(np.max(cf)), int(np.sum(np.array(cf) > 0))
        summ["forecast"] = {str(k): v for k, v in fcs.items()}
        return summ

    res = {}
    for tn in truths:
        truth = ColumnData([load_member(files[tn])], norm, dev)
        zts = [encode_members(L[0], truth)[0] for L in loaded]
        pool_models = [(L[0], L[1], zt, b) for L, zt, b in zip(loaded, zts, bounds)]
        single = [(loaded[0][0], loaded[0][1], zts[0], bounds[0])]
        r = {"free_pool": [run(System(pool_models, Z0_pool, "physical"), truth, s, 0.0, False) for s in a.seeds],
             "free_single": [run(System(single, [Z0_single], "physical"), truth, s, 0.0, False) for s in a.seeds]}
        for sname in a.systems:
            for rt in a.rtps:
                mk = (lambda: System(pool_models, Z0_pool, "physical")) if sname == "pool_phys" else \
                     (lambda: System(single, [Z0_single], "physical")) if sname == "single_phys" else \
                     (lambda: System(single, [Z0_single], "latent")) if sname == "single_latent" else \
                     (lambda: System(single, [Z0_single], "pff")) if sname == "single_pff" else \
                     (lambda: System(pool_models, Z0_pool, "pool_pff"))
                r[f"{sname}|{rt}"] = [run(mk(), truth, s, rt, True, tag=f"{sname}|{rt}") for s in a.seeds]
        res[tn] = r
        print(f"[pool-da] {tn}: " + "  ".join(f"{k} {np.mean([x['prior']['nmse_mean_dec'] for x in v]):.3f}" for k, v in r.items())
              + f"  [{time.time() - t_start:.0f}s]", flush=True)
        (out / f"part_{a.job:02d}.json").write_text(json.dumps({"args": vars(a), "ens_names": ens_names, "results": res}))
    print(f"[pool-da] done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
