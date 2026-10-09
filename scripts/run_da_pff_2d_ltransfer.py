"""2D analogue of `scripts/run_da_pff_ltransfer.py` (brief §15/Phase 13;
`docs/ks_local_field_ltransfer.tex` sec:consolidation Phase 4). Loads a
trained `local_field_2d` autoencoder + `local_mlp_2d` propagator (e.g.
`p7b_ks2d`) and runs localized DA cycling -- by default at the SAME size
the model was trained on, or at a LARGER size via `--old-L`/`--new-L`
(zero retraining, `load_autoencoder_checkpoint_resized_2d`/
`load_propagator_checkpoint_resized_2d`, the same mechanism
`exp_crop_eval_ks2d.py` already uses for Lyapunov transfer). Reuses
`ks_latent.da.cycling.run_da_experiment`/`ks_latent.da.letkf.LETKF`/
`ks_latent.da.pff.ParticleFlowFilter` exactly as-is (both are already
generic in physical/latent dimensionality -- nothing 2D-specific needed
there); the only genuinely new piece is `build_latent_taper_matrix_2d`
(`ks_latent/da/localization.py`), since the 1D taper cannot express a 2D
periodic lattice.

Deployment geometry (fixed 2026-10-07): the grid spacing is ALWAYS the
training dataset's spacing (read from the checkpoint's `train_dataset`
metadata), and `--L` is the physical side of the periodic domain. The
number of sites and grid points follow from that
(`ks_latent.da.geometry2d.deployment_grid_2d`). The earlier version took
the checkpoint's crop size (40 sites on 640 points) as the native
domain and paired it with L=44, which deployed the model at 5x the
training resolution; every run made with that version is invalid as a
measurement of the model.

Observations are a REGULAR 2D sparse grid (`--obs-stride-x`/`-y` in RAW
grid points, reshape-then-stride-then-flatten), not a 1D flat stride --
a flat stride over a 2D field would give a diagonal-looking, physically
arbitrary sampling pattern instead of a regular observation network.

Truth is a genuine 2D KS PDE integration (`ks_latent.solver.ks2d`), not
the model's own rollout -- the brief's own discipline (compare against
the real system, not the surrogate validating itself).
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import torch

from ks_latent.config import KSConfig2D
from ks_latent.da.var_recenter import VarRecenterConfig
from ks_latent.da.cycling import CycleConfig, run_da_experiment
from ks_latent.da.geometry2d import deployment_grid_2d, training_dx
from ks_latent.da.jacobian_coloring import assert_decoder_zero_reach, obs_site_map_2d
from ks_latent.da.localization import build_latent_taper_matrix_2d
from ks_latent.da.pff import PFFConfig
from ks_latent.da.sec import make_fixed_taper_localizer
from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.solver.ks2d import integrate2d, spinup2d
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def _sparse_obs_operator_2d(NX: int, NY: int, stride_x: int, stride_y: int):
    """`(NX*NY,) -> (m,)`: reshape to `(NX, NY)`, take a regular grid
    `[::stride_x, ::stride_y]`, flatten. A proper 2D observation network
    (unlike a flat stride, which samples a diagonal-ish, physically
    arbitrary pattern in 2D)."""
    def op(u_phys: torch.Tensor) -> torch.Tensor:
        return u_phys.reshape(NX, NY)[::stride_x, ::stride_y].reshape(-1)
    return op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--L", type=float, required=True,
                        help="Physical side of the square periodic domain; must be a whole number of sites.")
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--n-prop-steps", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument(
        "--method", choices=["pff", "letkf", "letkf_local2d", "letkf_local2d_etkf"], default="letkf",
        help="'letkf' (default here, cheaper per-cycle for this script's larger d_latent), 'pff', or "
             "'letkf_local2d' (T2, docs/ks_local_field_ltransfer.tex sec:c19_da_hypotheses -- the "
             "per-site local filter that avoids 'letkf's dense (d,d) covariance/taper; REQUIRED once "
             "d_latent makes the dense filter's memory impractical, e.g. beyond native size -- see H2 "
             "there for the GB-per-matrix numbers. Only valid for a zero-decoder-reach checkpoint; "
             "this script asserts that precondition and fails loudly otherwise.).",
    )
    parser.add_argument("--letkf-inflation", type=float, default=1.0)
    parser.add_argument("--obs-stride", type=int, default=32,
                        help="In grid points (grid spacing is fixed, so this is a fixed physical spacing); "
                             "32 = one observation per 2x2 block of 16-point sites.")
    parser.add_argument("--dt-snap", type=float, default=0.1, help="2D KS's own production dt_snap.")
    parser.add_argument("--localizer", choices=["none", "gaspari_cohn"], default="gaspari_cohn")
    parser.add_argument("--gc-c", type=float, default=3.0, help="In site units (sites have a fixed physical size).")
    parser.add_argument("--obs-noise-std", type=float, default=0.1)
    parser.add_argument("--filter-obs-std", type=float, default=None,
                        help="Observation-error std assumed by the filter (R); default = --obs-noise-std. "
                             "Larger values account for the decoder's representation error.")
    parser.add_argument("--model-noise-std", type=float, default=0.08)
    parser.add_argument("--ic-perturb-std", type=float, default=0.1)
    parser.add_argument("--spinup-time", type=float, default=200.0, help="Reduce for smoke-testing the DA machinery; 200.0 for a real run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--phys-rmse", action="store_true",
                        help="Task D6: also report physical-space RMSE of decode(mean z) against the truth "
                             "field at each analysis (all grid points and observed points separately), the "
                             "free run's, and the decoder's own floor |D(E(u)) - u|.")
    parser.add_argument("--phys-noise-std", type=float, default=0.0,
                        help="Accuracy study: physical-space random-field model noise (standardized units), "
                             "applied once per cycle before the analysis.")
    parser.add_argument("--phys-noise-len", type=float, default=4.0, help="Its correlation length (grid points).")
    parser.add_argument("--var-recenter", action="store_true",
                        help="Accuracy study: recentre the analysis on a variational fit "
                             "(ks_latent.da.var_recenter).")
    parser.add_argument("--var-beta", type=float, default=1.0)
    parser.add_argument("--var-gamma", type=float, default=1.0)
    parser.add_argument("--var-iters", type=int, default=30)
    parser.add_argument("--var-cov", choices=["diag", "ens_loc"], default="diag")
    parser.add_argument("--save-fields", type=int, default=0,
                        help="Task D6/D10: save truth / analysis / free / reconstruction fields at this many "
                             "evenly spaced analysis cycles (npz next to the JSON). 0 = none.")
    args = parser.parse_args()
    set_seed(args.seed)

    old_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    old_ae_cfg = old_ckpt["ae_config"]
    assert old_ae_cfg.n_sites_x == old_ae_cfg.n_sites_y, "script assumes a square-trained model (2D KS)"
    patch = old_ae_cfg.NX // old_ae_cfg.n_sites_x
    dx_train = training_dx(old_ckpt["train_dataset"])
    grid = deployment_grid_2d(args.L, dx_train, patch)
    new_n_sites, new_NX = grid.n_sites, grid.NX
    obs_stride_x = obs_stride_y = args.obs_stride
    assert new_NX % obs_stride_x == 0, f"NX={new_NX} not divisible by obs stride {obs_stride_x}"
    print(f"[da2d] L={grid.L} dx={grid.dx:.6f} (= training dx) NX={new_NX} n_sites={new_n_sites} "
          f"site_length={grid.site_length:.4f} obs_stride={obs_stride_x} "
          f"(obs spacing {obs_stride_x * grid.dx:.3f} units)")

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        args.ae_checkpoint, new_n_sites_x=new_n_sites, new_n_sites_y=new_n_sites, new_NX=new_NX, new_NY=new_NX,
    )
    prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d(
        args.prop_checkpoint, new_n_sites_x=new_n_sites, new_n_sites_y=new_n_sites,
    )
    ae.eval(); prop.eval()
    d = prop_cfg.d_latent
    print(f"[da2d] resized AE/propagator loaded with ZERO retraining (strict state_dict load), d_latent={d}")

    dt = 0.02  # 2D KS's own production dt (exp_crop_eval_ks2d.py)
    snapshot_every = round(args.dt_snap / dt)
    ks_cfg = KSConfig2D(Lx=grid.L, Ly=grid.L, NX=new_NX, NY=new_NX, dt=dt,
                        snapshot_every=snapshot_every, spinup_time=args.spinup_time, seed=args.seed)
    # The spin-up dominates run time and depends only on the domain, spinup and
    # seed, so it is cached and shared across ensemble sizes, filters and noise levels.
    cache = ARTIFACTS_DIR / "cache" / (
        f"ks2d_spinup_L{grid.L:g}_NX{new_NX}_dt{dt:g}_T{args.spinup_time:g}_seed{args.seed}.npy")
    if cache.exists():
        u0 = np.load(cache)
        print(f"[da2d] loaded cached spin-up state {cache}")
    else:
        u0 = spinup2d(ks_cfg, np.random.default_rng(args.seed + 500))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, u0)
    truth_np = integrate2d(u0, ks_cfg, n_steps=args.n_prop_steps * args.n_cycles * ks_cfg.snapshot_every)
    truth_traj = torch.tensor(truth_np.reshape(truth_np.shape[0], -1), dtype=torch.float32)  # (T+1, NX*NY)
    rss_scale = 1.0 if sys.platform == "darwin" else 1024.0  # ru_maxrss: bytes on macOS, KiB on Linux
    rss_after_truth_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale / 2**30

    norm_mean, norm_std = old_ckpt["normalization_mean"], old_ckpt["normalization_std"]
    truth_traj = (truth_traj - norm_mean) / norm_std

    encoder = lambda u: ae.encode(u)  # noqa: E731
    decoder = lambda z: ae.decode(z)  # noqa: E731
    obs_op = _sparse_obs_operator_2d(new_NX, new_NX, obs_stride_x, obs_stride_y)

    with torch.no_grad():
        z0 = encoder(truth_traj[0:1]).squeeze(0)
    rng = torch.Generator().manual_seed(args.seed)
    z0_ensemble = z0.unsqueeze(0) + args.ic_perturb_std * torch.randn(args.n_ensemble, d, generator=rng)
    z_minus1_ensemble = z0_ensemble.clone()

    localize_fn = None
    local_method = args.method in ("letkf_local2d", "letkf_local2d_etkf")  # D4 adds the deterministic variant
    if args.localizer == "gaspari_cohn" and not local_method:
        taper = build_latent_taper_matrix_2d(new_n_sites, new_n_sites, ae_cfg.local_channels, c=args.gc_c)
        localize_fn = make_fixed_taper_localizer(taper)

    site_of_obs = None
    if local_method:
        assert_decoder_zero_reach(ae_cfg)  # fails loudly if this checkpoint has nonzero decoder reach
        site_of_obs = obs_site_map_2d(new_NX, new_NX, new_n_sites, new_n_sites, obs_stride_x, obs_stride_y)
        print(f"[da2d] {args.method}: {site_of_obs.shape[0]} observations, gc_c={args.gc_c} (site units)")

    cfg = CycleConfig(
        n_prop_steps=args.n_prop_steps, n_cycles=args.n_cycles, n_ensemble=args.n_ensemble,
        model_noise_std=args.model_noise_std, obs_noise_std=args.obs_noise_std,
        filter_obs_std=args.filter_obs_std, record_decoded_member_mean=args.phys_rmse,
        phys_noise_std=args.phys_noise_std, phys_noise_len=args.phys_noise_len, phys_noise_shape=(new_NX, new_NX),
        var_recenter=(VarRecenterConfig(
            beta=args.var_beta, gamma=args.var_gamma, n_iter=args.var_iters, grid_shape=(new_NX, new_NX),
            cov=args.var_cov,
            taper=(build_latent_taper_matrix_2d(new_n_sites, new_n_sites, ae_cfg.local_channels, c=args.gc_c)
                   if args.var_cov == "ens_loc" else None)) if args.var_recenter else None),
        pff_config=PFFConfig(method="NAT", n_steps=100), seed=args.seed,
        localize_fn=localize_fn, method=args.method, letkf_inflation=args.letkf_inflation,
        site_of_obs=site_of_obs, n_sites_x=new_n_sites, n_sites_y=new_n_sites,
        local_channels=ae_cfg.local_channels, gc_c_local=args.gc_c,
    )
    result = run_da_experiment(prop, decoder, encoder, obs_op, truth_traj, z0_ensemble, z_minus1_ensemble, cfg)

    rmse_da = float(result.rmse_da.mean())
    rmse_free = float(result.rmse_free.mean())
    spread = float(result.spread.mean())
    # Per-cycle (not just the aggregate mean) diagnostic, added 2026-10-06
    # user-directed ("not sure how to interpret it honestly" re: modest DA
    # skill) -- the aggregate alone cannot distinguish "skill is flat/poor
    # from cycle 1" from "the ensemble is still finding its footing early
    # and skill improves over the run," which matters for deciding whether
    # weak skill is a tuning problem (e.g. no inflation) or a more basic
    # one. One entry per cycle, read at the analysis step (the last of
    # each cycle's `n_prop_steps` rows, cycling.py's own convention).
    n_prop_steps = args.n_prop_steps
    rmse_da_per_cycle = result.rmse_da[n_prop_steps - 1::n_prop_steps].tolist()
    rmse_free_per_cycle = result.rmse_free[n_prop_steps - 1::n_prop_steps].tolist()
    spread_per_cycle = result.spread[n_prop_steps - 1::n_prop_steps].tolist()
    report = {
        "rmse_da": rmse_da, "rmse_free": rmse_free, "spread": spread,
        "calibration_spread_over_rmse": spread / rmse_da if rmse_da > 0 else float("nan"),
        "skill_free_over_da": rmse_free / rmse_da if rmse_da > 0 else float("nan"),
        "n_ensemble": args.n_ensemble, "method": args.method, "localizer": args.localizer,
        "gc_c": args.gc_c if args.localizer == "gaspari_cohn" else None,
        "L": grid.L, "dx": grid.dx, "NX": new_NX, "n_sites": new_n_sites, "site_length": grid.site_length,
        "n_cycles": args.n_cycles, "n_prop_steps": args.n_prop_steps,
        "model_noise_std": args.model_noise_std, "obs_noise_std": args.obs_noise_std,
        "filter_obs_std": args.filter_obs_std,
        "seed": args.seed,
        "obs_stride_x": obs_stride_x, "obs_stride_y": obs_stride_y, "d_latent": d,
        "letkf_inflation": args.letkf_inflation,
        "rmse_da_per_cycle": rmse_da_per_cycle,
        "rmse_free_per_cycle": rmse_free_per_cycle,
        "spread_per_cycle": spread_per_cycle,
        "peak_rss_after_truth_GB": rss_after_truth_gb,
        "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale / 2**30,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    if args.phys_rmse or args.save_fields:
        # Analysis rows (last row of each cycle). Fields are standardized; x norm_std -> physical units.
        rows = torch.arange(n_prop_steps - 1, result.u_da.shape[0], n_prop_steps)
        u_t, u_a, u_f = result.u_truth[rows], result.u_da[rows], result.u_free[rows]
        with torch.no_grad():
            u_r = torch.cat([decoder(encoder(u_t[i:i + 10])) for i in range(0, len(rows), 10)])

        def rms(a):
            return (a.pow(2).mean(dim=1).sqrt() * norm_std).tolist()

        def rms_obs(a):
            return (torch.stack([obs_op(x) for x in a]).pow(2).mean(dim=1).sqrt() * norm_std).tolist()

        report.update({
            "phys_units": "field units (standardized RMSE x normalization_std)",
            "phys_rmse_da_per_cycle": rms(u_a - u_t), "phys_rmse_free_per_cycle": rms(u_f - u_t),
            "phys_rmse_recon_floor_per_cycle": rms(u_r - u_t),
            "phys_rmse_da_member_mean_per_cycle": (rms(result.u_da_member_mean - u_t)
                                                   if result.u_da_member_mean is not None else None),
            "phys_rmse_da_obs_per_cycle": rms_obs(u_a - u_t), "phys_rmse_free_obs_per_cycle": rms_obs(u_f - u_t),
            "phys_field_std_truth": float(u_t.std() * norm_std),
        })
        late = slice(args.n_cycles // 2, None)
        pa, pf = np.array(report["phys_rmse_da_per_cycle"])[late], np.array(report["phys_rmse_free_per_cycle"])[late]
        za = np.array(rmse_da_per_cycle)[late]; zf = np.array(rmse_free_per_cycle)[late]
        report["phys_skill_steady"] = float(np.mean(pf / pa))
        report["latent_skill_steady"] = float(np.mean(zf / za))
        report["phys_recon_floor_steady"] = float(np.mean(np.array(report["phys_rmse_recon_floor_per_cycle"])[late]))
        print(f"[da2d] D6: steady skill latent {report['latent_skill_steady']:.3f} phys {report['phys_skill_steady']:.3f}; "
              f"phys RMSE DA {pa.mean():.4f}, recon floor {report['phys_recon_floor_steady']:.4f}")
        if args.save_fields:
            pick = np.unique(np.linspace(0, len(rows) - 1, args.save_fields).round().astype(int))
            shape = (len(pick), new_NX, new_NX)
            np.savez_compressed(
                ARTIFACTS_DIR / "experiments" / "da2d" / f"da2d{suffix}_fields.npz",
                cycles=pick + 1, truth=(u_t[pick].numpy().reshape(shape) * norm_std + norm_mean).astype(np.float32),
                analysis=(u_a[pick].numpy().reshape(shape) * norm_std + norm_mean).astype(np.float32),
                free=(u_f[pick].numpy().reshape(shape) * norm_std + norm_mean).astype(np.float32),
                recon=(u_r[pick].numpy().reshape(shape) * norm_std + norm_mean).astype(np.float32),
                L=grid.L, NX=new_NX, n_sites=new_n_sites, obs_stride=obs_stride_x)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "experiments" / "da2d" / f"da2d{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
