"""Task H9 (2026-10-09): latent ensemble filter on several shifted tilings, analyses blended.

Same trained 2D KS models, truth, observation network and per-site filter (LETKFLocal2D) as the
Part II runs (run_da_pff_2d_ltransfer.py); no retraining. K tilings are the patch lattice shifted
by (ox, oy) grid points. Tiling k carries its own latent ensemble of the field rolled by
(-ox, -oy), forecast with the learned propagator, and assimilates the SAME observations (rolled
into its frame; offsets must be multiples of the observation stride, so the network is identical
in every frame). Offsets need not be multiples of the observation stride: in tiling k's frame the
observed pixels are (x_obs - ox) mod NX, and the per-site observation map is recomputed from them
(the observation vector keeps the original order in every frame). After the K analyses, member i of
every tiling is decoded, rolled back and blended with Hann weights that peak at each tiling's
patch centres and vanish at its patch edges (they sum to 1 exactly for offsets 0 and p/2).

--feedback: re-encode each blended member on every tiling, so the K ensembles restart from the
same blended analysis (a projection). Without it the K filters run independently and only the
output is blended.

Reported (field units, steady = cycles 51-100): analysis RMSE of tiling 0 alone (the current
filter), of the blended analysis, decoder floors (1 tiling, blended), spline interpolation of the
same observations. Writes artifacts/experiments/da2d/h9_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from ks_latent.config import KSConfig2D
from ks_latent.da.forecast import forecast_ensemble
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed


def interp_periodic(obs: np.ndarray, s: int, NX: int) -> np.ndarray:
    m = obs.shape[0]
    xo = np.arange(-3, m + 3) * s
    sp = RectBivariateSpline(xo, xo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)
    x = np.arange(NX)
    return sp(x, x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p7b_ks2d")
    ap.add_argument("--L", type=float, default=44.0)
    ap.add_argument("--n-ensemble", type=int, default=64)
    ap.add_argument("--n-cycles", type=int, default=100)
    ap.add_argument("--n-prop-steps", type=int, default=5)
    ap.add_argument("--obs-stride", type=int, default=8)
    ap.add_argument("--obs-noise-std", type=float, default=0.1)
    ap.add_argument("--model-noise-std", type=float, default=0.02)
    ap.add_argument("--gc-c", type=float, default=1.5)
    ap.add_argument("--ic-perturb-std", type=float, default=0.1)
    ap.add_argument("--n-tilings", type=int, default=4, choices=[1, 2, 4, 16],
                    help="1: current filter; 2: offsets (0,0),(p/2,p/2); 4: half-patch grid; 16: quarter-patch grid.")
    ap.add_argument("--weights", default="hann", choices=["hann", "uniform"])
    ap.add_argument("--inflation", type=float, default=1.0)
    ap.add_argument("--rtps", type=float, default=0.0, help="Relaxation to prior spread in each tiling's filter (I21).")
    ap.add_argument("--no-free", action="store_true", help="Skip the free run (skill denominator).")
    ap.add_argument("--diag", action="store_true",
                    help="Task I8: per wavenumber band, the inter-tiling covariance C of the tilings' analysis-mean "
                         "errors, the band error of tiling 0, of the Hann blend and of the uniform average, the "
                         "uniform-average prediction 1'C1/K^2, and the error of re-encoding the blended mean on "
                         "tiling 0 (what --feedback would restart from).")
    ap.add_argument("--manifold-diag", action="store_true",
                    help="I30(d): per cycle, for tiling 0's analysis mean z_a: latent error |z_a - E(u)| (rms per "
                         "coordinate) and off-manifold distance |E(D(z_a)) - z_a|; same for one member.")
    ap.add_argument("--save-fields", type=int, default=0,
                    help="Task I9: save truth / tiling-0 analysis / blended analysis / floors at this many "
                         "steady-state cycles (npz next to the json).")
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--same-tiling-control", action="store_true",
                    help="Control: K independent ensembles all on the unshifted tiling, equal weights (separates "
                         "the effect of shifting the seams from that of K times more members).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    t0 = time.time()

    AE = f"artifacts/stage1_ae_kolmogorov_crop_{a.tag}.pt"
    PR = f"artifacts/stage2_prop_kolmogorov_crop_{a.tag}.pt"
    ck = torch.load(AE, map_location="cpu", weights_only=False)
    patch = ck["ae_config"].NX // ck["ae_config"].n_sites_x
    g = deployment_grid_2d(a.L, training_dx(ck["train_dataset"]), patch)
    NX, ns = g.NX, g.n_sites
    ae = load_autoencoder_checkpoint_resized_2d(AE, new_n_sites_x=ns, new_n_sites_y=ns, new_NX=NX, new_NY=NX)[0].eval()
    prop = load_propagator_checkpoint_resized_2d(PR, new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
    mean, std = float(ck["normalization_mean"]), float(ck["normalization_std"])

    dt = 0.02
    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                      seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    u0 = np.load(cache) if cache.exists() else spinup2d(kcfg, np.random.default_rng(a.seed + 500))
    truth = integrate2d(u0, kcfg, n_steps=a.n_prop_steps * a.n_cycles * kcfg.snapshot_every)
    truth = ((truth.reshape(truth.shape[0], -1) - mean) / std).reshape(-1, NX, NX)

    s, N, hp = a.obs_stride, a.n_ensemble, patch // 2
    q = patch // 4
    offs = {1: [(0, 0)], 2: [(0, 0), (hp, hp)], 4: [(0, 0), (hp, 0), (0, hp), (hp, hp)],
            16: [(ox, oy) for ox in range(0, patch, q) for oy in range(0, patch, q)]}[a.n_tilings]

    def hann(off):
        t = (np.arange(NX) - off) % patch
        return np.sin(np.pi * (t + 0.5) / patch) ** 2 if a.weights == "hann" else np.ones(NX)
    if a.same_tiling_control:
        offs = [(0, 0)] * len(offs)
    W = [hann(ox)[:, None] * hann(oy)[None, :] for ox, oy in offs]
    Wsum = sum(W); W = [w / Wsum for w in W]

    def enc(F, o):  # fields (B, NX, NX) in the original frame -> latents of tiling o
        X = torch.tensor(np.roll(F, (-o[0], -o[1]), axis=(1, 2)).reshape(len(F), -1), dtype=torch.float32)
        with torch.no_grad():
            return torch.cat([ae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])

    def dec(Z, o):  # latents of tiling o -> fields in the original frame
        with torch.no_grad():
            U = torch.cat([ae.decode(Z[i:i + 32]) for i in range(0, len(Z), 32)]).double().numpy()
        return np.roll(U.reshape(len(Z), NX, NX), (o[0], o[1]), axis=(1, 2))

    xg = np.arange(0, NX, s)  # observed rows/columns in the original frame
    def frame_obs(o):  # observed pixels and their sites in tiling o's frame, original obs order
        ix, iy = (xg - o[0]) % NX, (xg - o[1]) % NX
        site = ((ix[:, None] // patch) * ns + (iy[None, :] // patch)).reshape(-1)
        return torch.as_tensor(ix), torch.as_tensor(iy), torch.as_tensor(site, dtype=torch.int64)
    FO = {o: frame_obs(o) for o in set(offs)}
    assert torch.equal(FO[(0, 0)][2], obs_site_map_2d(NX, NX, ns, ns, s, s).to(torch.int64)), "site convention"

    def make_h(o):  # observation operator in tiling o's frame
        ix, iy = FO[o][0], FO[o][1]
        return lambda z: ae.decode(z.to(torch.float32).unsqueeze(0)).squeeze(0).reshape(NX, NX)[ix][:, iy].reshape(-1).double()

    site_of_obs = FO[(0, 0)][2]
    d = enc(truth[:1], (0, 0)).shape[1]
    c = d // (ns * ns)
    R_diag = torch.full((site_of_obs.shape[0],), a.obs_noise_std ** 2, dtype=torch.float64)

    gen = torch.Generator().manual_seed(a.seed)
    Zc = []
    for o in offs:
        z0 = enc(truth[:1], o)[0]
        Zc.append(z0 + a.ic_perturb_std * torch.randn(N, d, generator=gen))
    Zp = [z.clone() for z in Zc]
    rng = np.random.default_rng(a.seed + 1)

    rec = {k: [] for k in ("tiling0", "blend", "blend_of_means", "floor1", "floorK", "interp", "spread0",
                           "spread_blend", "blend_obs", "free", "prior_tiling0", "prior_blend",
                           "z_err0", "z_offman0", "z_offman_member0", "z_offman_truth")}
    Zfree_p, Zfree_c = Zp[0].clone(), Zc[0].clone()
    kk = 2 * np.pi * np.fft.fftfreq(NX, d=g.L / NX); KK = np.sqrt(kk[:, None] ** 2 + kk[None, :] ** 2)
    BANDS = [(0, 0.6), (0.6, 1.14), (1.14, 1.6), (1.6, 99)]
    bmask = [(KK >= lo) & (KK < hi) for lo, hi in BANDS]
    diag = {"C": [], "tiling0": [], "blend": [], "uniform": [], "reencode_err": [], "reencode_dist": []}
    late0 = a.n_cycles // 2
    save_at = set(np.linspace(late0, a.n_cycles - 1, a.save_fields).round().astype(int).tolist()) if a.save_fields else set()
    saved = {k: [] for k in ("truth", "tiling0", "blend", "floor1", "floorK", "cycle")}
    t = 0
    for cyc in range(a.n_cycles):
        Zf = []
        for k in range(len(offs)):
            fc = forecast_ensemble(prop, Zp[k], Zc[k], a.n_prop_steps, a.model_noise_std)
            Zp[k], Zf_k = fc.z_prev_final, fc.z_curr_final
            Zf.append(Zf_k)
        if not a.no_free:
            fcf = forecast_ensemble(prop, Zfree_p, Zfree_c, a.n_prop_steps, 0.0)
            Zfree_p, Zfree_c = fcf.z_prev_final, fcf.z_curr_final
        t += a.n_prop_steps
        u = truth[t]
        yg = u[::s, ::s] + a.obs_noise_std * rng.standard_normal((NX // s, NX // s))
        yv = torch.tensor(yg.reshape(-1))
        Uf = [dec(Zf[k], o).mean(0) for k, o in enumerate(offs)]  # forecast (prior) means, original frame
        Ua = []
        for k, o in enumerate(offs):
            filt = LETKFLocal2D(make_h(o), R_diag, FO[o][2], ns, ns, c, a.gc_c, inflation=a.inflation, rtps=a.rtps,
                                seed=a.seed * 1000 + cyc * 20 + k)
            Za = filt.analyze(Zf[k].double(), yv).Z_analysis.float()
            Zc[k] = Za
            Ua.append(dec(Za, o))  # (N, NX, NX)
        Ub = sum(w[None] * U for w, U in zip(W, Ua))  # member-wise blend
        if a.feedback and len(offs) > 1:
            for k, o in enumerate(offs):
                Zc[k] = enc(Ub, o)
        f = lambda v: float(np.sqrt(np.mean((v - u) ** 2)) * std)  # noqa: E731  field units
        rec["tiling0"].append(f(Ua[0].mean(0)))  # tiling 0 alone: the current filter
        rec["blend"].append(f(Ub.mean(0)))
        if a.manifold_diag:
            rms = lambda v: float(v.pow(2).mean().sqrt())  # noqa: E731
            za = Zc[0].mean(0, keepdim=True) if not a.feedback else Za.mean(0, keepdim=True)
            zt = enc(u[None], offs[0])
            with torch.no_grad():
                proj = lambda z: ae.encode(ae.decode(z))  # noqa: E731  (tiling 0's own frame)
                rec["z_err0"].append(rms(za - zt))
                rec["z_offman0"].append(rms(proj(za) - za))
                rec["z_offman_member0"].append(rms(proj(Zc[0][:1]) - Zc[0][:1]))
                rec["z_offman_truth"].append(rms(proj(zt) - zt))
        rec["prior_tiling0"].append(f(Uf[0])); rec["prior_blend"].append(f(sum(w * x for w, x in zip(W, Uf))))
        rec["blend_of_means"].append(f(sum(w * dec(Zc[k].mean(0, keepdim=True), o)[0]
                                           for k, (w, o) in enumerate(zip(W, offs)))))
        recs = [dec(enc(u[None], o), o)[0] for o in offs]
        rec["floor1"].append(f(recs[0])); rec["floorK"].append(f(sum(w * r for w, r in zip(W, recs))))
        rec["interp"].append(f(interp_periodic(yg, s, NX)))
        rec["spread0"].append(float(Ua[0].std(0).mean() * std))
        rec["spread_blend"].append(float(np.sqrt(np.mean(Ub.var(0))) * std))
        rec["blend_obs"].append(float(np.sqrt(np.mean((Ub.mean(0)[::s, ::s] - u[::s, ::s]) ** 2)) * std))
        if not a.no_free:
            rec["free"].append(f(dec(Zfree_c, (0, 0)).mean(0)))
        if a.diag and cyc >= late0:
            E = np.fft.fft2(np.stack([U.mean(0) - u for U in Ua]) * std)  # (K, NX, NX) tiling errors
            Eb = np.fft.fft2((Ub.mean(0) - u) * std); Eu = E.mean(0)
            diag["C"].append([np.real(np.einsum("kn,ln->kl", E[:, m], np.conj(E[:, m]))) / NX ** 4 for m in bmask])
            diag["tiling0"].append([float(np.sum(np.abs(E[0][m]) ** 2) / NX ** 4) for m in bmask])
            diag["blend"].append([float(np.sum(np.abs(Eb[m]) ** 2) / NX ** 4) for m in bmask])
            diag["uniform"].append([float(np.sum(np.abs(Eu[m]) ** 2) / NX ** 4) for m in bmask])
            rb = dec(enc(Ub.mean(0)[None], (0, 0)), (0, 0))[0]
            diag["reencode_err"].append(f(rb)); diag["reencode_dist"].append(f(rb - Ub.mean(0) + u))
        if cyc in save_at:
            for k, v in (("truth", u), ("tiling0", Ua[0].mean(0)), ("blend", Ub.mean(0)), ("floor1", recs[0]),
                         ("floorK", sum(w * r for w, r in zip(W, recs)))):
                saved[k].append((v * std + mean).astype(np.float32))
            saved["cycle"].append(cyc)
        if not np.isfinite(Ub).all():
            raise FloatingPointError(f"non-finite blended analysis at cycle {cyc}")
        if cyc % 10 == 9:
            print(f"cycle {cyc + 1}: " + " ".join(f"{k} {v[-1]:.3f}" for k, v in rec.items() if v), flush=True)
    late = slice(a.n_cycles // 2, None)
    steady = {k: float(np.mean(v[late])) for k, v in rec.items() if v}
    if "free" in steady:
        steady["skill_blend"] = steady["free"] / steady["blend"]
        steady["skill_tiling0"] = steady["free"] / steady["tiling0"]
    steady["calib_blend"] = steady["spread_blend"] / steady["blend"]
    import resource
    steady["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    if a.diag:
        C = np.mean(np.array(diag["C"]), axis=0)  # (bands, K, K)
        Kt = C.shape[1]
        corr = [C[b] / np.sqrt(np.outer(np.diag(C[b]), np.diag(C[b]))) for b in range(len(BANDS))]
        steady["diag"] = {
            "bands": BANDS,
            "tiling0_var": np.mean(diag["tiling0"], axis=0).tolist(),
            "blend_var": np.mean(diag["blend"], axis=0).tolist(),
            "uniform_var": np.mean(diag["uniform"], axis=0).tolist(),
            "uniform_var_predicted": [float(C[b].sum() / Kt ** 2) for b in range(len(BANDS))],
            "mean_offdiag_corr": [float((cb.sum() - np.trace(cb)) / (Kt * (Kt - 1))) if Kt > 1 else 1.0 for cb in corr],
            "C": C.tolist(),
            "reencode_err": float(np.mean(diag["reencode_err"])),
            "reencode_dist_from_blend": float(np.mean(diag["reencode_dist"])),
        }
    if a.save_fields:
        np.savez_compressed(f"artifacts/experiments/da2d/h9_{a.out}_fields.npz",
                            **{k: np.stack(v) for k, v in saved.items()}, NX=NX, n_sites=ns, patch=patch,
                            offsets=np.array(offs), L=g.L)
    out = {"args": vars(a), "steady": steady, "per_cycle": rec, "wall_s": time.time() - t0, "oi_climatological": 0.798}
    Path("artifacts/experiments/da2d").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/experiments/da2d/h9_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, {k: round(v, 3) for k, v in steady.items() if not isinstance(v, dict)}, f"{out['wall_s']:.0f}s")
    if "diag" in steady:
        print("diag", {k: (np.round(v, 3).tolist() if k != "C" else "...") for k, v in steady["diag"].items()})


if __name__ == "__main__":
    main()
