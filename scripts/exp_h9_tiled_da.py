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
from ks_latent.da.forecast import forecast_ensemble, forecast_ensemble_history
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import coloured_obs_jacobian_2d_compact, obs_site_map_2d
from ks_latent.da.letkf_local import LETKFLocal2D, LETKFLocal2DIterative
from ks_latent.da.nonlinearity import innovation_chi2_diag, jdelta_compact, linearization_stats, rank_histogram
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed


def interp_periodic(obs: np.ndarray, s: int, NX: int) -> np.ndarray:
    m = obs.shape[0]
    xo = np.arange(-3, m + 3) * s
    sp = RectBivariateSpline(xo, xo, np.pad(obs, 3, mode="wrap"), kx=3, ky=3)
    x = np.arange(NX)
    return sp(x, x)


def parse_memory(spec):
    """'eps,tau,kappa[,coupling]' -> MemoryParams (J17B), or None."""
    if not spec:
        return None
    from ks_latent.solver.ks2d_memory import MemoryParams
    f = spec.split(",")
    return MemoryParams(float(f[0]), float(f[1]), float(f[2]), f[3] if len(f) > 3 else "lap")


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
    ap.add_argument("--truth-lap-coef", type=float, default=1.0,
                    help="I28 imperfect-model twin: truth uses coefficient a on the nabla^2 term "
                         "(ks_latent/solver/twin.py); the models keep a = 1. Default 1: perfect-model twin.")
    ap.add_argument("--truth-advect-x", type=float, default=0.0, help="I28: truth has mean-flow advection c (x); models do not")
    ap.add_argument("--truth-forcing-amp", type=float, default=0.0, help="I28: truth has terrain-like forcing (std); models do not")
    ap.add_argument("--assumed-obs-noise-std", type=float, default=None,
                    help="I16b: observation-error std the FILTER assumes (R); the data keep --obs-noise-std")
    ap.add_argument("--ic-climatology", action="store_true",
                    help="I16a: initial ensemble = encoded states of an independent climatology (seed-3 run, 2 time "
                         "units apart), with no knowledge of the truth (instead of the encoded truth + noise)")
    ap.add_argument("--forecast-every", type=int, default=0,
                    help="I7: from steady-state analyses (cycles > n_cycles/2), every k cycles run deterministic "
                         "forecasts of each tiling's analysis mean, blended, out to --forecast-leads cycles")
    ap.add_argument("--forecast-leads", type=int, default=60)
    ap.add_argument("--truth-memory", default=None,
                    help="J17B: 'eps,tau,kappa[,coupling]' -- truth has the hidden memory field v (ks2d_memory.py)")
    ap.add_argument("--truth-ganiso", type=float, default=0.0, help="J17G: truth has -(delta/2)(u_x^2 - u_y^2)")
    ap.add_argument("--truth-cubic", type=float, default=0.0,
                    help="J17A: truth u_t = KS(u) - beta P[u'^3] (the latent model is trained on that system's data)")
    ap.add_argument("--tiling-reset", action="store_true",
                    help="Failsafe (2026-10-09): if one tiling's analysis mean disagrees with the median of the others "
                         "by more than --reset-factor x the median disagreement (and > --reset-min field units), "
                         "drop it from this cycle's blend and re-initialize it from the blend of the healthy "
                         "tilings (members re-encoded in its frame). Uses no truth.")
    ap.add_argument("--reset-factor", type=float, default=3.0)
    ap.add_argument("--reset-min", type=float, default=1.0, help="field units")
    ap.add_argument("--gate", action="store_true",
                    help="I31: manifold-gated propagator (ks_latent/da/gated.py). Experts: this tag's propagator "
                         "(on-manifold) and --gate-off-tag's (off-manifold), or 'project then on' if no off tag.")
    ap.add_argument("--gate-off-tag", default=None)
    ap.add_argument("--gate-quantile", type=float, default=0.95)
    ap.add_argument("--gate-width-frac", type=float, default=0.25)
    ap.add_argument("--project-members", action="store_true",
                    help="I30(d) baseline: after each forecast (incl. model noise), project every member back onto "
                         "the autoencoder's manifold, z <- E(D(z)), in its own tiling's frame. No second model.")
    ap.add_argument("--manifold-diag", action="store_true",
                    help="I30(d): per cycle, for tiling 0's analysis mean z_a: latent error |z_a - E(u)| (rms per "
                         "coordinate) and off-manifold distance |E(D(z_a)) - z_a|; same for one member.")
    ap.add_argument("--save-fields", type=int, default=0,
                    help="Task I9: save truth / tiling-0 analysis / blended analysis / floors at this many "
                         "steady-state cycles (npz next to the json).")
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--nl-diag", action="store_true",
                    help="Z20 E1: per cycle and tiling, on the forecast ensemble: linearization residual of H = H_obs o D "
                         "(eta, eta_o, per-site eta_o), Jensen gap in obs space, innovation consistency ratio; "
                         "observation rank histograms (tiling 0 and the blended prior), steady cycles only")
    ap.add_argument("--n-iter", type=int, default=1,
                    help="Z20 E3: Gauss-Newton relinearizations of the per-site update (1 = the standard update)")
    ap.add_argument("--iter-tol", type=float, default=1e-4)
    ap.add_argument("--filter", choices=["enkf", "kpff"], default="enkf",
                    help="Z20 E6: kpff = kernel particle flow filter (Hu & van Leeuwen 2021; ks_latent/da/pff_kernel.py) "
                         "per tiling, Schur-localized prior covariance with the same GC taper, per-particle gradients "
                         "through the nonlinear decoder")
    ap.add_argument("--kpff-alpha", type=float, default=None, help="kernel width factor (default 1/N)")
    ap.add_argument("--kpff-steps", type=int, default=200)
    ap.add_argument("--kpff-max-move", type=float, default=0.05)
    ap.add_argument("--kpff-tol", type=float, default=1e-4, help="stop when the ensemble mean changes less (0: run all steps)")
    ap.add_argument("--oracle-forecast", choices=["none", "project", "increment"], default="none",
                    help="Z20 E4: forecast every member with the TRUE PDE in physical space, encode it on each tiling, "
                         "analyse in latent space. project: next physical member = D(z_a); increment: u_f + D(z_a) - D(z_f) "
                         "(keeps the off-manifold part of the forecast). No model noise (perfect model).")
    ap.add_argument("--oracle-latent-noise", type=float, default=0.0,
                    help="Z20 E4: std of Gaussian noise added to each encoded oracle forecast member (latent units); "
                         "the learned-model filter adds 0.02 per step over 5 steps, i.e. about 0.045 per cycle")
    ap.add_argument("--save-priors", type=int, default=0,
                    help="Z20 E2: save tiling 0's forecast ensemble (latent, and decoded fields) at this many steady cycles")
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
    # J17B memory test (2026-10-10): a mode="history" propagator steps from the current state and
    # H-1 past states. Each member then carries its history; the filter analyses the history as
    # extra channels of each site (an augmented-state, lagged-smoother update with the same local
    # gain), observing only the newest state.
    H = prop.cfg.n_history if getattr(prop, "mode", None) == "history" else 1
    if H > 1 and (a.gate or a.forecast_every or a.tiling_reset or a.feedback or a.project_members
                  or a.manifold_diag):
        raise ValueError("a history propagator is implemented for the plain tiled filter only")
    if a.gate:
        from ks_latent.da.gated import GatedPropagator, calibrate_gate
        off = None
        if a.gate_off_tag:
            off = load_propagator_checkpoint_resized_2d(f"artifacts/stage2_prop_kolmogorov_crop_{a.gate_off_tag}.pt",
                                                        new_n_sites_x=ns, new_n_sites_y=ns)[0].eval()
        # calibrate on an independent truth (seed 3), never the DA truth
        cal = np.load(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt0.02_T200_seed3.npy")
        ccfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=0.02, snapshot_every=250, spinup_time=200.0, seed=3)
        Ucal = integrate2d(cal, ccfg, n_steps=20 * 250)[1:]
        with torch.no_grad():
            Zcal = ae.encode(torch.tensor(((Ucal - mean) / std).reshape(20, -1), dtype=torch.float32))
        d0, gw = calibrate_gate(Zcal, ae.encode, ae.decode, ns * ns, a.gate_quantile, a.gate_width_frac)
        print(f"[gate] d0 {d0:.4f} width {gw:.4f} off={'project' if off is None else a.gate_off_tag}", flush=True)
        prop = GatedPropagator(prop, off, ae.encode, ae.decode, ns * ns, d0, gw)

    dt = 0.02
    kcfg = KSConfig2D(Lx=g.L, Ly=g.L, NX=NX, NY=NX, dt=dt, snapshot_every=round(0.1 / dt), spinup_time=200.0,
                      seed=a.seed)
    cache = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed{a.seed}.npy")
    if not cache.exists():
        np.save(cache, spinup2d(kcfg, np.random.default_rng(a.seed + 500)))
    from ks_latent.solver.twin import twin_truth
    n_cyc_truth = a.n_cycles + (a.forecast_leads if a.forecast_every else 0)  # I7: truth beyond the last cycle
    if a.truth_memory:  # J17B: hidden memory field; only u is ever used (observed / scored)
        from ks_latent.solver.ks2d_memory import integrate_memory, memory_truth
        mp = parse_memory(a.truth_memory)
        u0, v0 = memory_truth(kcfg, a.seed, mp)
        truth = integrate_memory(u0, v0, kcfg, a.n_prop_steps * n_cyc_truth * kcfg.snapshot_every, mp)[0]
    else:
        u0, truth_kw = twin_truth(kcfg, a.seed, a.truth_lap_coef, a.truth_advect_x, a.truth_forcing_amp,
                                 cubic=a.truth_cubic, ganiso=a.truth_ganiso)
        truth = integrate2d(u0, kcfg, n_steps=a.n_prop_steps * n_cyc_truth * kcfg.snapshot_every, **truth_kw)
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
        if H > 1:  # augmented state (P, H, c) site-major: observe the newest state only
            return lambda z: ae.decode(z.to(torch.float32).reshape(ns * ns, H, -1)[:, -1].reshape(1, -1)).squeeze(0) \
                .reshape(NX, NX)[ix][:, iy].reshape(-1).double()
        return lambda z: ae.decode(z.to(torch.float32).unsqueeze(0)).squeeze(0).reshape(NX, NX)[ix][:, iy].reshape(-1).double()

    def make_hb(o):  # batched, differentiable observation operator (kernel PFF)
        ix, iy = FO[o][0], FO[o][1]
        return lambda Z: ae.decode(Z.to(torch.float32)).reshape(len(Z), NX, NX)[:, ix][:, :, iy].reshape(len(Z), -1).double()

    def to_aug(Zh):  # (N, H, P*c) -> (N, P*H*c), site-major
        return Zh.reshape(len(Zh), H, ns * ns, -1).permute(0, 2, 1, 3).reshape(len(Zh), -1)

    def from_aug(Za):  # inverse of to_aug
        return Za.reshape(len(Za), ns * ns, H, -1).permute(0, 2, 1, 3).reshape(len(Za), H, -1)

    site_of_obs = FO[(0, 0)][2]
    d = enc(truth[:1], (0, 0)).shape[1]
    c = d // (ns * ns)
    r_assumed = a.obs_noise_std if a.assumed_obs_noise_std is None else a.assumed_obs_noise_std
    R_diag = torch.full((site_of_obs.shape[0],), r_assumed ** 2, dtype=torch.float64)
    kpff_taper = None
    if a.filter == "kpff":
        if H > 1 or a.n_iter > 1:
            raise ValueError("--filter kpff: plain state, no --n-iter")
        from ks_latent.da.localization import build_latent_taper_matrix_2d
        from ks_latent.da.pff_kernel import KernelPFF
        kpff_taper = build_latent_taper_matrix_2d(ns, ns, channels=c, c=a.gc_c).to(torch.float64)

    gen = torch.Generator().manual_seed(a.seed)
    Zc = []
    if a.ic_climatology:  # I16a: members drawn from an independent climatology, no knowledge of the truth
        if a.truth_memory or a.truth_cubic or a.truth_ganiso or a.truth_lap_coef != 1.0 or a.truth_advect_x or a.truth_forcing_amp:
            raise ValueError("--ic-climatology is implemented for the standard truth only")
        cc = Path(f"artifacts/cache/ks2d_spinup_L{g.L:g}_NX{NX}_dt{dt:g}_T200_seed3.npy")
        if not cc.exists():
            np.save(cc, spinup2d(kcfg, np.random.default_rng(3 + 500)))
        clim = integrate2d(np.load(cc), kcfg, n_steps=N * 2 * 50, snapshot_every=100)[1:]  # N states, 2 units apart
        clim = ((clim.reshape(N, -1) - mean) / std).reshape(N, NX, NX)
    for o in offs:
        if a.ic_climatology:
            Zc.append(enc(clim, o))
            continue
        z0 = enc(truth[:1], o)[0]
        Zc.append(z0 + a.ic_perturb_std * torch.randn(N, d, generator=gen))
    Zp = [z.clone() for z in Zc]
    Zh = [z.unsqueeze(1).repeat(1, H, 1) for z in Zc] if H > 1 else None  # constant initial history
    rng = np.random.default_rng(a.seed + 1)

    rec = {k: [] for k in ("tiling0", "blend", "blend_of_means", "floor1", "floorK", "interp", "spread0",
                           "spread_blend", "blend_obs", "free", "prior_tiling0", "prior_blend",
                           "z_err0", "z_offman0", "z_offman_member0", "z_offman_truth", "gate_mean", "n_reset",
                           "tiling0_of_mean", "n_iter_used", "iter_last_change",
                           "nl_eta", "nl_eta_o", "nl_eta_o_site_median", "nl_eta_o_site_p90", "nl_eta_o_site_max",
                           "nl_jensen_obs", "nl_spread_obs", "nl_chi2", "nl_chi2_blend", "nl_eta_a", "nl_eta_o_a")}
    rank_hist = {"tiling0": None, "blend": None}
    rng_rank = np.random.default_rng(a.seed + 77)
    saved_priors = {"Z": [], "U": [], "truth": [], "cycle": []}
    oracle = a.oracle_forecast != "none"
    if oracle:
        if H > 1 or a.gate or a.project_members or a.feedback or a.tiling_reset or a.forecast_every:
            raise ValueError("--oracle-forecast is implemented for the plain tiled filter only")
        if a.truth_memory or a.truth_cubic or a.truth_ganiso or a.truth_lap_coef != 1.0 or a.truth_advect_x or a.truth_forcing_amp:
            raise ValueError("--oracle-forecast uses the standard KS solver as the forecast: perfect-model truth only")
        import importlib.util
        import scipy.fft as sfft
        from ks_latent.solver.ks2d import get_operator2d
        _sp = importlib.util.spec_from_file_location("exp_i11", Path(__file__).with_name("exp_i11_pde_letkf.py"))
        _i11 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_i11)
        op_pde = get_operator2d(kcfg)
        n_steps_cycle = a.n_prop_steps * kcfg.snapshot_every

        def pde_forecast(U):  # standardized (N, NX, NX) -> standardized, true PDE over one cycle
            X = U * std + mean
            X = X - X.mean(axis=(1, 2), keepdims=True)
            with sfft.set_workers(2):
                V = sfft.fft2(X)
                for _ in range(n_steps_cycle):
                    V = _i11.step_batch(V, op_pde)
                X = sfft.ifft2(V).real
            return (X - mean) / std
        Uphys = [dec(Zc[k], o) for k, o in enumerate(offs)]
        gen_on = torch.Generator().manual_seed(a.seed + 4242)  # physical members start from the decoded latent ensemble
    resets = []  # (cycle, tiling, disagreement, median disagreement)
    Zfree_p, Zfree_c = Zp[0].clone(), Zc[0].clone()
    Zfree_h = Zh[0].clone() if H > 1 else None
    kk = 2 * np.pi * np.fft.fftfreq(NX, d=g.L / NX); KK = np.sqrt(kk[:, None] ** 2 + kk[None, :] ** 2)
    BANDS = [(0, 0.6), (0.6, 1.14), (1.14, 1.6), (1.6, 99)]
    bmask = [(KK >= lo) & (KK < hi) for lo, hi in BANDS]
    diag = {"C": [], "tiling0": [], "blend": [], "uniform": [], "reencode_err": [], "reencode_dist": []}
    late0 = a.n_cycles // 2
    diverged_at = None
    prior_at = set(np.linspace(late0, a.n_cycles - 1, a.save_priors).round().astype(int).tolist()) if a.save_priors else set()
    save_at = set(np.linspace(late0, a.n_cycles - 1, a.save_fields).round().astype(int).tolist()) if a.save_fields else set()
    saved = {k: [] for k in ("truth", "tiling0", "blend", "floor1", "floorK", "cycle")}
    fcst = []  # I7 forecast rows
    t = 0
    for cyc in range(a.n_cycles):
        Zf = []
        for k in range(len(offs)):
            if oracle:
                Uphys[k] = pde_forecast(Uphys[k])
                Zo = enc(Uphys[k], offs[k])
                if a.oracle_latent_noise > 0:
                    Zo = Zo + a.oracle_latent_noise * torch.randn(Zo.shape, generator=gen_on)
                Zf.append(Zo)
                continue
            if H > 1:
                Zh[k] = forecast_ensemble_history(prop, Zh[k], a.n_prop_steps, a.model_noise_std).z_hist_final
                Zf_k = Zh[k][:, -1]
            else:
                fc = forecast_ensemble(prop, Zp[k], Zc[k], a.n_prop_steps, a.model_noise_std)
                Zp[k], Zf_k = fc.z_prev_final, fc.z_curr_final
            if a.project_members:
                with torch.no_grad():
                    Zf_k = torch.cat([ae.encode(ae.decode(Zf_k[i:i + 32])) for i in range(0, len(Zf_k), 32)])
            Zf.append(Zf_k)
        if not a.no_free:
            if H > 1:
                Zfree_h = forecast_ensemble_history(prop, Zfree_h, a.n_prop_steps, 0.0).z_hist_final
                Zfree_c = Zfree_h[:, -1]
            else:
                fcf = forecast_ensemble(prop, Zfree_p, Zfree_c, a.n_prop_steps, 0.0)
                Zfree_p, Zfree_c = fcf.z_prev_final, fcf.z_curr_final
        t += a.n_prop_steps
        u = truth[t]
        yg = u[::s, ::s] + a.obs_noise_std * rng.standard_normal((NX // s, NX // s))
        yv = torch.tensor(yg.reshape(-1))
        Uf = [dec(Zf[k], o).mean(0) for k, o in enumerate(offs)]  # forecast (prior) means, original frame
        if oracle:
            Uf = [U.mean(0) for U in Uphys]  # the physical forecast, not its re-encoding
        if a.nl_diag:  # Z20 E1, on each tiling's forecast ensemble (tiling-averaged)
            acc = []
            for k, o in enumerate(offs):
                hk = make_h(o)
                Zk = Zf[k].double()
                zb = Zk.mean(0)
                Jk = coloured_obs_jacobian_2d_compact(hk, zb, FO[o][2], c).detach()
                with torch.no_grad():
                    Hz = torch.stack([hk(Zk[e]) for e in range(N)])
                    st_k = linearization_stats(Hz, hk(zb), jdelta_compact(Zk - zb, Jk, FO[o][2], c), a.obs_noise_std,
                                               FO[o][2], ns * ns)
                st_k["chi2"] = innovation_chi2_diag(yv, Hz, a.obs_noise_std)
                acc.append(st_k)
                if k == 0 and cyc >= late0:
                    hcount = rank_histogram(yv.numpy(), Hz.numpy(), a.obs_noise_std, rng_rank)
                    rank_hist["tiling0"] = hcount if rank_hist["tiling0"] is None else rank_hist["tiling0"] + hcount
            for key in ("eta", "eta_o", "eta_o_site_median", "eta_o_site_p90", "eta_o_site_max"):
                rec["nl_" + key].append(float(np.mean([x[key] for x in acc])))
            rec["nl_jensen_obs"].append(float(np.mean([x["jensen_obs_over_sigma"] for x in acc])))
            rec["nl_spread_obs"].append(float(np.mean([x["spread_obs_over_sigma"] for x in acc])))
            rec["nl_chi2"].append(float(np.mean([x["chi2"] for x in acc])))
            # blended prior members at the observed points (what the blended analysis is scored like)
            Ubp = sum(w[None] * (Uphys[k] if oracle else dec(Zf[k], o)) for k, (w, o) in enumerate(zip(W, offs)))
            Hb = torch.tensor(Ubp[:, ::s, ::s].reshape(N, -1))
            rec["nl_chi2_blend"].append(innovation_chi2_diag(yv, Hb, a.obs_noise_std))
            if cyc >= late0:
                hb = rank_histogram(yv.numpy(), Hb.numpy(), a.obs_noise_std, rng_rank)
                rank_hist["blend"] = hb if rank_hist["blend"] is None else rank_hist["blend"] + hb
        if a.save_priors and cyc in prior_at:
            saved_priors["Z"].append(Zf[0].numpy().astype(np.float32))
            saved_priors["U"].append((Uphys[0] if oracle else dec(Zf[0], offs[0])).astype(np.float32))
            saved_priors["truth"].append(u.astype(np.float32)); saved_priors["cycle"].append(cyc)
        Ua = []
        it_used, it_change = [], []
        for k, o in enumerate(offs):
            if a.filter == "kpff":
                filt = KernelPFF(make_hb(o), R_diag, taper=kpff_taper, alpha=a.kpff_alpha, n_steps=a.kpff_steps,
                                 max_move=a.kpff_max_move, rtps=a.rtps, tol=a.kpff_tol)
            elif a.n_iter > 1:
                filt = LETKFLocal2DIterative(make_h(o), R_diag, FO[o][2], ns, ns, c * H, a.gc_c, inflation=a.inflation,
                                             rtps=a.rtps, seed=a.seed * 1000 + cyc * 20 + k, n_iter=a.n_iter,
                                             tol=a.iter_tol)
            else:
                filt = LETKFLocal2D(make_h(o), R_diag, FO[o][2], ns, ns, c * H, a.gc_c, inflation=a.inflation,
                                    rtps=a.rtps, seed=a.seed * 1000 + cyc * 20 + k)
            if H > 1:
                Zh[k] = from_aug(filt.analyze(to_aug(Zh[k]).double(), yv).Z_analysis.float())
                Za = Zh[k][:, -1]
            else:
                res_k = filt.analyze(Zf[k].double(), yv)
                Za = res_k.Z_analysis.float()
                if a.filter == "kpff":
                    it_used.append(res_k.n_steps); it_change.append(0.0 if res_k.converged else 1.0)
            if a.n_iter > 1:
                it_used.append(filt.iterations_used); it_change.append(filt.mean_changes[-1])
            Zc[k] = Za
            if oracle:
                Uz = dec(Za, o)
                Uphys[k] = Uz if a.oracle_forecast == "project" else Uphys[k] + Uz - dec(Zf[k], o)
                Ua.append(Uphys[k])
            else:
                Ua.append(dec(Za, o))  # (N, NX, NX)
        if it_used:
            rec["n_iter_used"].append(float(np.mean(it_used))); rec["iter_last_change"].append(float(np.max(it_change)))
        Ub = sum(w[None] * U for w, U in zip(W, Ua))  # member-wise blend
        if a.tiling_reset and len(offs) > 2:
            means = np.stack([U.mean(0) for U in Ua])  # (K, NX, NX), standardized units
            dis = np.array([np.sqrt(np.mean((means[k] - np.median(np.delete(means, k, 0), axis=0)) ** 2)) * std
                            for k in range(len(offs))])
            med = float(np.median(dis))
            bad = [k for k in range(len(offs)) if dis[k] > a.reset_factor * med and dis[k] > a.reset_min]
            if bad and len(bad) < len(offs):
                good = [k for k in range(len(offs)) if k not in bad]
                wsum = sum(W[k] for k in good)
                Ub = sum(W[k][None] * Ua[k] for k in good) / wsum[None]
                for k in bad:
                    Zc[k] = enc(Ub, offs[k]); Zp[k] = Zc[k].clone()
                    Ua[k] = Ub.copy()
                    resets.append((cyc + 1, k, float(dis[k]), med))
            rec["n_reset"].append(len(bad) if bad and len(bad) < len(offs) else 0)
        if a.feedback and len(offs) > 1:
            for k, o in enumerate(offs):
                Zc[k] = enc(Ub, o)
        f = lambda v: float(np.sqrt(np.mean((v - u) ** 2)) * std)  # noqa: E731  field units
        rec["tiling0"].append(f(Ua[0].mean(0)))  # tiling 0 alone: the current filter
        rec["blend"].append(f(Ub.mean(0)))
        rec["tiling0_of_mean"].append(f(dec(Zc[0].mean(0, keepdim=True), offs[0])[0]))  # D(z_bar), tiling 0
        if a.nl_diag:  # linearization residual on tiling 0's ANALYSIS ensemble (after RTPS)
            h0 = make_h(offs[0]); Z0a = Zc[0].double(); zb0 = Z0a.mean(0)
            J0 = coloured_obs_jacobian_2d_compact(h0, zb0, FO[offs[0]][2], c).detach()
            with torch.no_grad():
                Hza = torch.stack([h0(Z0a[e]) for e in range(N)])
                sa = linearization_stats(Hza, h0(zb0), jdelta_compact(Z0a - zb0, J0, FO[offs[0]][2], c),
                                         a.obs_noise_std, FO[offs[0]][2], ns * ns)
            rec["nl_eta_a"].append(sa["eta"]); rec["nl_eta_o_a"].append(sa["eta_o"])
        if a.gate:
            rec["gate_mean"].append(prop.last_mean_gate)
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
        if a.forecast_every and cyc >= late0 and (cyc - late0) % a.forecast_every == 0:
            # I7: deterministic forecasts of each tiling's analysis mean (no model noise), blended (Hann);
            # tiling 0 alone (single-tiling filter); the encoded TRUE state (decoder-limited ceiling); persistence.
            zt_c = {o: enc(u[None], o) for o in offs}
            zt_p = {o: enc(truth[t - a.n_prop_steps][None], o) for o in offs}
            cur = [(Zp[k].mean(0, keepdim=True), Zc[k].mean(0, keepdim=True)) for k in range(len(offs))]
            tru = [(zt_p[o], zt_c[o]) for o in offs]
            row = {"blend": [], "tiling0": [], "true_init_blend": [], "persistence": [], "clim": []}
            ub0 = Ub.mean(0)
            for lead in range(1, a.forecast_leads + 1):
                if t + lead * a.n_prop_steps >= truth.shape[0]:
                    break
                ut = truth[t + lead * a.n_prop_steps]
                nxt, nxt_t, fields, fields_t = [], [], [], []
                for k, o in enumerate(offs):
                    fc1 = forecast_ensemble(prop, cur[k][0], cur[k][1], a.n_prop_steps, 0.0)
                    fc2 = forecast_ensemble(prop, tru[k][0], tru[k][1], a.n_prop_steps, 0.0)
                    nxt.append((fc1.z_prev_final, fc1.z_curr_final)); nxt_t.append((fc2.z_prev_final, fc2.z_curr_final))
                    fields.append(dec(fc1.z_curr_final, o)[0]); fields_t.append(dec(fc2.z_curr_final, o)[0])
                cur, tru = nxt, nxt_t
                g_ = lambda v: float(np.sqrt(np.mean((v - ut) ** 2)) * std)  # noqa: E731
                row["blend"].append(g_(sum(w * x for w, x in zip(W, fields))))
                row["tiling0"].append(g_(fields[0]))
                row["true_init_blend"].append(g_(sum(w * x for w, x in zip(W, fields_t))))
                row["persistence"].append(g_(ub0))
                row["clim"].append(float(np.sqrt(np.mean(ut ** 2)) * std))
            fcst.append(row)
        if cyc in save_at:
            for k, v in (("truth", u), ("tiling0", Ua[0].mean(0)), ("blend", Ub.mean(0)), ("floor1", recs[0]),
                         ("floorK", sum(w * r for w, r in zip(W, recs)))):
                saved[k].append((v * std + mean).astype(np.float32))
            saved["cycle"].append(cyc)
        # Runaway check (2026-10-09): a tiling whose analysis error exceeds 5x the field's own std has
        # diverged; stop and record it rather than averaging a blow-up into the blend.
        tile_err = [f(U.mean(0)) for U in Ua]
        field_sd = float(u.std() * std)
        if max(tile_err) > 5 * field_sd:
            diverged_at = cyc + 1
            print(f"DIVERGED at cycle {diverged_at}: tiling errors {[round(e, 2) for e in tile_err]} "
                  f"(field std {field_sd:.2f})", flush=True)
            break
        if not np.isfinite(Ub).all():
            raise FloatingPointError(f"non-finite blended analysis at cycle {cyc}")
        if cyc % 10 == 9:
            print(f"cycle {cyc + 1}: " + " ".join(f"{k} {v[-1]:.3f}" for k, v in rec.items() if v), flush=True)
    late = slice(a.n_cycles // 2, None)
    steady = {k: float(np.mean(v[late])) for k, v in rec.items() if v and len(v) > late0}
    steady["diverged_at_cycle"] = diverged_at
    if diverged_at is not None:
        steady = {"diverged_at_cycle": diverged_at}
    if "free" in steady:
        steady["skill_blend"] = steady["free"] / steady["blend"]
        steady["skill_tiling0"] = steady["free"] / steady["tiling0"]
    if "blend" in steady:
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
    if a.tiling_reset:
        steady["n_resets_total"] = len(resets)
    if a.nl_diag:
        steady["rank_hist_tiling0"] = rank_hist["tiling0"].tolist() if rank_hist["tiling0"] is not None else None
        steady["rank_hist_blend"] = rank_hist["blend"].tolist() if rank_hist["blend"] is not None else None
    if a.save_priors:
        np.savez_compressed(f"artifacts/experiments/da2d/h9_{a.out}_priors.npz",
                            **{k: np.stack(v) for k, v in saved_priors.items()}, NX=NX, n_sites=ns, patch=patch,
                            channels=c, obs_stride=s, mean=mean, std=std, L=g.L)
    out = {"args": vars(a), "steady": steady, "resets": resets, "per_cycle": rec, "wall_s": time.time() - t0, "oi_climatological": 0.798}
    if fcst:  # I7: mean error vs lead (cycles of n_prop_steps snapshots) over the forecast starts
        nl = min(len(r["blend"]) for r in fcst)
        out["forecast"] = {k: np.mean([r[k][:nl] for r in fcst], axis=0).tolist() for k in fcst[0]}
        out["forecast"]["n_starts"] = len(fcst)
    Path("artifacts/experiments/da2d").mkdir(parents=True, exist_ok=True)
    Path(f"artifacts/experiments/da2d/h9_{a.out}.json").write_text(json.dumps(out, indent=2))
    print(a.out, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in steady.items() if not isinstance(v, (dict, list))},
          f"{out['wall_s']:.0f}s")
    if "diag" in steady:
        print("diag", {k: (np.round(v, 3).tolist() if k != "C" else "...") for k, v in steady["diag"].items()})


if __name__ == "__main__":
    main()
