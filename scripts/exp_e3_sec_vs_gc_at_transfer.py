"""E3 (docs/ks_local_field_ltransfer.tex): does SEC's advantage over
Gaspari-Cohn at L=100 (docs/RESULTS.md Phase E1: 0/6 PASS, SEC wins by
5-25%) survive once the domain has grown past the training size?

Key realization while designing this (worth stating explicitly): Anderson
SEC's correction table (`ks_latent.da.sec.build_sec_table`) is a pure
Monte-Carlo function of ensemble size alone -- it draws bivariate-Gaussian
samples of a given `n_ens` and records the empirical sampling-error
correction, with NO dependence anywhere on the physical system, its
domain size, or its latent dimension. So "fit SEC fresh at L=200" doesn't
mean anything different from applying the existing table -- the table
IS already L-independent by construction. What actually changes at L=200
is that SEC can now be compared against Gaspari-Cohn on the SAME latent
representation (Section 224's local field, resized with zero retraining)
for the first time, instead of the L=100 comparison's global-vector-vs-
local-field confound (Table in docs/ks_local_field_ltransfer.tex Section
5: SEC ran on the 44-dim global latent, Gaspari-Cohn on the 48-dim local
one -- two different architectures, not just two localizers). This script
is a genuinely cleaner ablation of localizer choice alone.

Reuses `scripts/run_da_pff_ltransfer.py`'s exact DA-cycling body (same
resized-checkpoint loaders, same CycleConfig/PFFConfig), adding the `sec`
localizer option via `ks_latent.da.sec.build_sec_table`/
`make_sec_localizer` (both already used by `scripts/run_da_pff.py`'s own
`--localizer sec` path at L=100) -- no new DA logic, just a new localizer
choice wired into the L-transfer driver.
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
from ks_latent.da.sec import build_sec_table, make_fixed_taper_localizer, make_sec_localizer, make_sec_psd_localizer
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
    parser.add_argument("--new-L", type=float, required=True)
    parser.add_argument("--n-ensemble", type=int, default=64)
    parser.add_argument("--n-prop-steps", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument("--obs-stride-at-old-L", type=int, default=8)
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--localizer", choices=["none", "sec", "sec_psd", "gaspari_cohn"], default="sec",
                        help="sec_psd (task D3(i)): SEC followed by projection onto the PSD cone.")
    parser.add_argument("--method", choices=["pff", "letkf", "serial_sec_enkf"], default="pff",
                        help="serial_sec_enkf (task D3(ii)): SEC applied per observation in a serial EnKF; "
                             "--localizer is then ignored (the correction is built into the filter).")
    parser.add_argument("--obs-spacing-units", type=float, default=3.125,
                        help="Fixed physical observation spacing (task D1); see run_da_pff_ltransfer.py.")
    parser.add_argument("--legacy-scaled-stride", action="store_true", help="Reproduce the original E3 stride.")
    parser.add_argument("--gc-c", type=float, default=3.0)
    parser.add_argument("--sec-n-trials", type=int, default=2000)
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
    obs_stride = (int(round(args.obs_stride_at_old_L * scale)) if args.legacy_scaled_stride
                  else max(1, int(round(args.obs_spacing_units / dx))))
    assert new_NX % new_n_sites == 0, f"new_NX={new_NX} not divisible by new_n_sites={new_n_sites}"
    print(f"[e3] old: L={args.old_L} NX={old_ae_cfg.NX} n_sites={old_ae_cfg.n_sites}")
    print(f"[e3] new: L={args.new_L} NX={new_NX} n_sites={new_n_sites} obs_stride={obs_stride} (scale={scale}x)")

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized(args.ae_checkpoint, new_n_sites=new_n_sites, new_NX=new_NX)
    d = ae_cfg.n_sites * ae_cfg.local_channels
    prop, prop_cfg, _ = load_propagator_checkpoint_resized(
        args.prop_checkpoint, new_d_latent=d, new_n_tokens=new_n_sites,
    )
    ae.eval(); prop.eval()
    print(f"[e3] resized AE/propagator loaded with ZERO retraining, d_latent={d}, localizer={args.localizer}")

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
    sec_table = None
    if args.method == "serial_sec_enkf":
        sec_table = build_sec_table(n_ens=args.n_ensemble, n_trials=args.sec_n_trials, seed=args.seed)
    elif args.localizer in ("sec", "sec_psd"):
        table = build_sec_table(n_ens=args.n_ensemble, n_trials=args.sec_n_trials, seed=args.seed)
        localize_fn = make_sec_localizer(table) if args.localizer == "sec" else make_sec_psd_localizer(table)
    elif args.localizer == "gaspari_cohn":
        taper = build_latent_taper_matrix(ae_cfg.n_sites, ae_cfg.local_channels, c=args.gc_c)
        localize_fn = make_fixed_taper_localizer(taper)

    cfg = CycleConfig(
        n_prop_steps=args.n_prop_steps, n_cycles=args.n_cycles, n_ensemble=args.n_ensemble,
        pff_config=PFFConfig(method="NAT", n_steps=100), seed=args.seed,
        localize_fn=localize_fn, method=args.method, sec_table=sec_table,
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
        "localizer": args.localizer,
        "gc_c": args.gc_c if args.localizer == "gaspari_cohn" else None,
        "old_L": args.old_L,
        "new_L": args.new_L,
        "new_NX": new_NX,
        "new_n_sites": new_n_sites,
        "obs_stride": obs_stride,
        "obs_spacing_units": obs_stride * dx,
        "method": args.method,
        "n_cycles": args.n_cycles,
        "rmse_da_per_step": [float(x) for x in result.rmse_da],
        "rmse_free_per_step": [float(x) for x in result.rmse_free],
        "spread_per_step": [float(x) for x in result.spread],
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    late = slice(len(result.rmse_da) // 2, None)  # steady state: second half (cycles 51-100 of 100)
    report["skill_steady"] = float((result.rmse_free[late] / result.rmse_da[late]).mean())
    report["calibration_steady"] = float((result.spread[late] / result.rmse_da[late]).mean())
    out_path = ARTIFACTS_DIR / f"e3_sec_vs_gc_ltransfer{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
