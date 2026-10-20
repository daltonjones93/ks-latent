"""Phase F3 (`docs/steps_4-3.md`): full localized-DA-cycling L-transfer
test. Reuses `scripts/run_da_pff.py`'s own DA cycling machinery
(`CycleConfig`/`run_da_experiment`/`build_latent_taper_matrix`/
`make_fixed_taper_localizer`) exactly, but loads `LOCAL_AE`/
`TRANSFER_PROP` (Section 224, trained at `L=100`) at a LARGER `L` via
`load_autoencoder_checkpoint_resized`/`load_propagator_checkpoint_
resized` -- zero retraining -- same mechanism Phase F2
(`scripts/run_ltransfer_test.py`) already validated for free-running
forecast quality.

`--obs-stride` is interpreted in RAW GRID POINTS, same as `run_da_pff.py`
-- since `NX` scales with `L` to hold `dx` fixed (same reasoning as
`n_sites`/`patch_size`), `--obs-stride` should be scaled by the SAME
factor to keep observation density (physical spacing between
observations) constant across the comparison; this script does that
scaling itself from the checkpoint's own trained `NX` rather than
requiring the caller to compute it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ks_latent.da.cycling import CycleConfig, run_da_experiment
from ks_latent.da.localization import build_latent_taper_matrix
from ks_latent.da.pff import PFFConfig
from ks_latent.da.sec import make_fixed_taper_localizer
from ks_latent.models import load_autoencoder_checkpoint_resized, load_propagator_checkpoint_resized
from ks_latent.solver.ks import KSConfig, integrate, spinup
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def _sparse_obs_operator(stride: int):
    def op(u_phys: torch.Tensor) -> torch.Tensor:
        return u_phys[::stride]
    return op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--old-L", type=float, default=100.0)
    parser.add_argument("--new-L", type=float, default=200.0)
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--n-prop-steps", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument(
        "--method", choices=["pff", "letkf", "kpff"], default="pff",
        help="DA analysis scheme (added 2026-09-28): 'pff' (default, this project's primary NAT-PFF) "
        "or 'letkf' (ks_latent.da.letkf, a covariance-localized EnKF comparator -- same forecast, "
        "same localization taper, different analysis update).",
    )
    parser.add_argument("--letkf-inflation", type=float, default=1.0)
    parser.add_argument("--kpff-alpha", type=float, default=None, help="Z20 E7: kernel width factor (default 1/N)")
    parser.add_argument("--kpff-steps", type=int, default=200)
    parser.add_argument("--kpff-max-move", type=float, default=0.05)
    parser.add_argument("--kpff-rtps", type=float, default=0.0)
    parser.add_argument("--obs-stride-at-old-L", type=int, default=8,
                        help="Only with --legacy-scaled-stride: scaled by new_L/old_L (E2/E7 behaviour).")
    parser.add_argument("--obs-spacing-units", type=float, default=3.125,
                        help="Physical spacing between observations (task D1, 2026-10-08); the stride in grid "
                             "points is round(spacing / dx). 3.125 = stride 8 at L=100, NX=256.")
    parser.add_argument("--legacy-scaled-stride", action="store_true",
                        help="Reproduce E2/E7: stride = obs_stride_at_old_L * new_L/old_L. Since dx is held "
                             "fixed, this DOUBLES the physical observation spacing at every doubling of L "
                             "(the confound of review sec. 4.2a).")
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--localizer", choices=["none", "gaspari_cohn"], default="gaspari_cohn")
    parser.add_argument("--gc-c", type=float, default=3.0, help="In SITE units at the NEW (transferred) n_sites.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    set_seed(args.seed)

    scale = args.new_L / args.old_L
    old_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    old_ae_cfg = old_ckpt["ae_config"]
    new_n_sites = int(round(old_ae_cfg.n_sites * scale))
    new_NX = int(round(old_ae_cfg.NX * scale))
    dx = args.new_L / new_NX
    if args.legacy_scaled_stride:
        obs_stride = int(round(args.obs_stride_at_old_L * scale))
    else:
        obs_stride = max(1, int(round(args.obs_spacing_units / dx)))
    obs_spacing = obs_stride * dx
    assert new_NX % new_n_sites == 0, f"new_NX={new_NX} not divisible by new_n_sites={new_n_sites}"
    print(f"[ltransfer-da] old: L={args.old_L} NX={old_ae_cfg.NX} n_sites={old_ae_cfg.n_sites} "
          f"obs_stride={args.obs_stride_at_old_L}")
    print(f"[ltransfer-da] new: L={args.new_L} NX={new_NX} n_sites={new_n_sites} "
          f"obs_stride={obs_stride} -> obs spacing {obs_spacing:g} units "
          f"({'legacy scaled stride' if args.legacy_scaled_stride else 'fixed physical spacing'})")

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized(args.ae_checkpoint, new_n_sites=new_n_sites, new_NX=new_NX)
    d = ae_cfg.n_sites * ae_cfg.local_channels
    prop, prop_cfg, _ = load_propagator_checkpoint_resized(
        args.prop_checkpoint, new_d_latent=d, new_n_tokens=new_n_sites,
    )
    ae.eval(); prop.eval()
    print(f"[ltransfer-da] resized AE/propagator loaded with ZERO retraining (strict state_dict load), d_latent={d}")

    snapshot_every = round(args.dt_snap / 0.05)
    ks_cfg = KSConfig(L=args.new_L, NX=new_NX, dt=0.05, snapshot_every=snapshot_every, spinup_time=500.0, seed=args.seed)
    u0 = spinup(ks_cfg, np.random.default_rng(args.seed + 500))
    truth_np = integrate(u0, ks_cfg, n_steps=args.n_prop_steps * args.n_cycles * ks_cfg.snapshot_every)
    truth_traj = torch.tensor(truth_np, dtype=torch.float32)

    encoder = lambda u: ae.encode(u)  # noqa: E731
    decoder = lambda z: ae.decode(z)  # noqa: E731
    obs_op = _sparse_obs_operator(obs_stride)

    with torch.no_grad():
        z0 = encoder(truth_traj[0:1]).squeeze(0)
    rng = torch.Generator().manual_seed(args.seed)
    z0_ensemble = z0.unsqueeze(0) + 0.1 * torch.randn(args.n_ensemble, d, generator=rng)
    z_minus1_ensemble = z0_ensemble.clone()

    localize_fn = None
    if args.localizer == "gaspari_cohn":
        taper = build_latent_taper_matrix(ae_cfg.n_sites, ae_cfg.local_channels, c=args.gc_c)
        localize_fn = make_fixed_taper_localizer(taper)

    cfg = CycleConfig(
        n_prop_steps=args.n_prop_steps, n_cycles=args.n_cycles, n_ensemble=args.n_ensemble,
        pff_config=PFFConfig(method="NAT", n_steps=100), seed=args.seed,
        localize_fn=localize_fn, method=args.method, letkf_inflation=args.letkf_inflation,
        kpff_taper=(taper.to(torch.float64) if args.localizer == "gaspari_cohn" else None),
        kpff_alpha=args.kpff_alpha, kpff_steps=args.kpff_steps, kpff_max_move=args.kpff_max_move,
        kpff_rtps=args.kpff_rtps,
    )
    result = run_da_experiment(prop, decoder, encoder, obs_op, truth_traj, z0_ensemble, z_minus1_ensemble, cfg)

    rmse_da = float(result.rmse_da.mean())
    rmse_free = float(result.rmse_free.mean())
    spread = float(result.spread.mean())
    report = {
        "rmse_da": rmse_da,
        "rmse_free": rmse_free,
        "spread": spread,
        "calibration_spread_over_rmse": spread / rmse_da if rmse_da > 0 else float("nan"),
        "skill_free_over_da": rmse_free / rmse_da if rmse_da > 0 else float("nan"),
        "n_ensemble": args.n_ensemble,
        "method": args.method,
        "kpff": ({"alpha": args.kpff_alpha, "steps": args.kpff_steps, "max_move": args.kpff_max_move,
                  "rtps": args.kpff_rtps} if args.method == "kpff" else None),
        "localizer": args.localizer,
        "gc_c": args.gc_c if args.localizer == "gaspari_cohn" else None,
        "old_L": args.old_L,
        "new_L": args.new_L,
        "new_NX": new_NX,
        "new_n_sites": new_n_sites,
        "obs_stride": obs_stride,
        "obs_spacing_units": obs_spacing,
        "legacy_scaled_stride": bool(args.legacy_scaled_stride),
        "n_cycles": args.n_cycles,
        "n_prop_steps": args.n_prop_steps,
        "seed": args.seed,
        "rmse_da_per_step": [float(x) for x in result.rmse_da],
        "rmse_free_per_step": [float(x) for x in result.rmse_free],
        "spread_per_step": [float(x) for x in result.spread],
    }
    late = slice(len(result.rmse_da) // 2, None)  # steady state = second half of the analysed steps (cycles 51-100 of 100)
    da_l, fr_l, sp_l = (np.asarray(report[k])[late] for k in ("rmse_da_per_step", "rmse_free_per_step",
                                                               "spread_per_step"))
    report["skill_steady"] = float(np.mean(fr_l / da_l))
    report["calibration_steady"] = float(np.mean(sp_l / da_l))
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = ARTIFACTS_DIR / f"da_pff_ltransfer{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
