"""Phase F2 (`docs/steps_4-3.md`): architecture-level L-transfer test, no
DA yet. Loads `LOCAL_AE`/`TRANSFER_PROP` (Section 224, trained at
L=100/NX=256/n_sites=16) at a LARGER L via `load_autoencoder_checkpoint_
resized`/`load_propagator_checkpoint_resized` -- ZERO retraining -- and
checks:

1. The decoded free-running rollout at the new L has a plausible energy
   spectrum (peaks near k ~= 1/sqrt(2), KS's own linear-instability
   wavenumber, independent of L).
2. The standalone D_KY measured at the new L, divided by L, is close to
   the L=100 model's own D_KY/100 ratio (both should also be close to
   the TRUE extensivity constant, 0.226).

User-directed (2026-09-25): "run a larger L without retraining. Make
certain that if L gets larger, the number of samples gets larger though
so the effective sample width remains the same" -- NX scales with L to
hold dx=L/NX fixed, which (since patch_size=NX/n_sites is held fixed by
`load_autoencoder_checkpoint_resized`'s own check) means n_sites scales
by the same factor.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator
from ks_latent.models import load_autoencoder_checkpoint_resized, load_propagator_checkpoint_resized
from ks_latent.solver.ks import KSConfig, integrate, spinup, wavenumbers


def energy_spectrum_peak_k(traj: np.ndarray, L: float, NX: int) -> float:
    spec = np.mean(np.abs(np.fft.fft(traj, axis=1)) ** 2, axis=0)
    k = wavenumbers(L, NX)
    pos = k > 0
    return float(k[pos][np.argmax(spec[pos])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--prop-checkpoint", required=True)
    parser.add_argument("--old-L", type=float, default=100.0)
    parser.add_argument("--new-L", type=float, default=200.0)
    parser.add_argument("--old-D-KY", type=float, required=True, help="Section 224's own measured D_KY at old_L.")
    parser.add_argument("--dt-snap", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-lyap-steps", type=int, default=2000)
    args = parser.parse_args()

    scale = args.new_L / args.old_L
    old_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    old_ae_cfg = old_ckpt["ae_config"]
    new_n_sites = int(round(old_ae_cfg.n_sites * scale))
    new_NX = int(round(old_ae_cfg.NX * scale))
    assert new_NX % new_n_sites == 0, f"new_NX={new_NX} not divisible by new_n_sites={new_n_sites}"
    print(f"[ltransfer] old: L={args.old_L} NX={old_ae_cfg.NX} n_sites={old_ae_cfg.n_sites} "
          f"d_latent={old_ae_cfg.n_sites*old_ae_cfg.local_channels}")
    print(f"[ltransfer] new: L={args.new_L} NX={new_NX} n_sites={new_n_sites} "
          f"d_latent={new_n_sites*old_ae_cfg.local_channels}  (scale={scale}x)")

    ae, ae_cfg, ae_ckpt = load_autoencoder_checkpoint_resized(
        args.ae_checkpoint, new_n_sites=new_n_sites, new_NX=new_NX,
    )
    d_latent = ae_cfg.n_sites * ae_cfg.local_channels
    prop, prop_cfg, prop_ckpt = load_propagator_checkpoint_resized(
        args.prop_checkpoint, new_d_latent=d_latent, new_n_tokens=new_n_sites,
    )
    ae.eval(); prop.eval()
    print(f"[ltransfer] resized AE/propagator loaded with ZERO retraining (strict state_dict load).")

    # Ground truth at the new L, zero-retraining rollout comparison.
    snapshot_every = round(args.dt_snap / 0.05)
    ks_cfg = KSConfig(L=args.new_L, NX=new_NX, dt=0.05, snapshot_every=snapshot_every, spinup_time=500.0, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    u0 = spinup(ks_cfg, rng)
    n_roll = 400
    truth = integrate(u0, ks_cfg, n_steps=n_roll * snapshot_every)  # (n_roll+1, NX)

    with torch.no_grad():
        z0 = ae.encode(torch.tensor(truth[0:1], dtype=torch.float32))
        z_roll = prop.rollout(z0, z0, k=n_roll)  # (1, n_roll+1, d_latent)
        u_roll = ae.decode(z_roll[0]).numpy()  # (n_roll+1, NX)

    finite = np.isfinite(u_roll).all()
    print(f"[ltransfer] decoded rollout finite everywhere: {finite}")
    if finite:
        max_abs = np.abs(u_roll).max()
        print(f"[ltransfer] decoded rollout max|u|: {max_abs:.4f} (true trajectory max|u|: {np.abs(truth).max():.4f})")

        k_peak_true = energy_spectrum_peak_k(truth, args.new_L, new_NX)
        k_peak_model = energy_spectrum_peak_k(u_roll, args.new_L, new_NX)
        k_expected = 1.0 / np.sqrt(2.0)
        print(f"[ltransfer] energy spectrum peak k: true={k_peak_true:.4f} model={k_peak_model:.4f} "
              f"(expected ~{k_expected:.4f}, L-independent)")

    # D_KY/L extensivity check.
    with torch.no_grad():
        z01 = ae.encode(torch.tensor(truth[0:2], dtype=torch.float32)).numpy()
    try:
        res = lyapunov_spectrum_latent_propagator(
            prop, z01, mode="single_state", n_directions=d_latent, n_steps=args.n_lyap_steps, qr_every=10,
            dt_snap=args.dt_snap, warmup_steps=200, seed=args.seed, max_abs_state=1e4,
        )
        d_ky_new = res.kaplan_yorke_dimension
        print(f"[ltransfer] new-L Lyapunov: lambda1={res.exponents[0]:.4f} n_positive={res.n_positive}/{res.n_directions} "
              f"D_KY={d_ky_new:.4f}")
        old_ratio = args.old_D_KY / args.old_L
        new_ratio = d_ky_new / args.new_L
        print(f"[ltransfer] D_KY/L: old={old_ratio:.4f} (L={args.old_L}) new={new_ratio:.4f} (L={args.new_L}) "
              f"[true KS extensivity constant ~= 0.226]")
    except RuntimeError as e:
        print(f"[ltransfer] Lyapunov computation FAILED at new L (reference trajectory diverged): {e}")

    print("[ltransfer] done.")


if __name__ == "__main__":
    main()
