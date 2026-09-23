"""One-off diagnostic (2026-09-04, user-directed): "I think 81 had the
best rollout error accuracy honestly. the trick is we want more smooth
embedding dynamics that could be modeled through a pde. this is just
hypothesis though, perhaps they are smooth enough, we just don't know"
followed by "yeah I'm seeing huge jumps in the gif for 82, so I don't
think increasing dimension is necessarily the answer either".

Direct smoothness measures, complementing
scripts/analyze_75_76_77_latent_geometry.py's covariance-conditioning-
based proxies (which Section 52's own catastrophic cond#=1.59e6-yet-
best-rollout result already falsified as a universal smoothness
predictor -- good conditioning is not necessary for good rollout fit,
and by the same token there is no a priori reason it should track
smoothness either):

1. ENCODED REAL-TRAJECTORY step size / curvature (encoder only, no
   propagator needed -- directly explains the GIF's visual "jumps",
   which are the ENCODER's placement of real states, not an artifact of
   any trained propagator): for each section's AE, encode real
   validation trajectories and measure ||z_{t+1}-z_t|| (one-step
   displacement) and ||z_{t+1}-2*z_t+z_{t-1}|| (discrete second
   difference / curvature) at every real time step, both raw and
   normalized by the trajectory's own RMS latent radius (so sections
   with very different overall latent scales -- e.g. Section 52's
   top eigenvalue ~17.7 vs. 75/81's well-conditioned ~20-22 -- remain
   comparable).
2. TRAINED-PROPAGATOR local Lipschitz constant: the spectral norm
   (largest singular value) of the trained propagator's own step
   Jacobian (d(z_next)/d(input), via torch.func.jacrev), evaluated at
   real points sampled from actual encoded rollout trajectories (on the
   attractor, not random off-manifold points) -- the literal quantity a
   "could a smooth PDE/ODE be fit to the LEARNED dynamics" question is
   asking about, distinct from #1 (which characterizes the ENCODING
   alone, before any propagator).

Handles both `mode="history"` (75/81/82: n_history=2, step via
step_history on a flattened (n_history*d_latent)-dim input) and
`mode="markovian"` (52: mlp backbone; 84 once trained: fourier_mlp
backbone, via step_one on a d_latent-dim input) propagators uniformly.

The two measures below moved to `ks_latent.analysis.diagnostics`
(2026-09-05, as "D9") so Gate 4 (`scripts/run_diagnostics.py`) can compute
the same thing for a single checkpoint without duplicating this logic --
this script now just imports them for the side-by-side comparison table
across multiple checkpoints at once, which Gate 4's one-checkpoint-at-a-
time report doesn't give you.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import torch

from ks_latent.analysis.diagnostics import (
    encoded_trajectory_smoothness,
    propagator_step_jacobian_spectral_norms,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint

SECTIONS = {
    "52 (vit AE, mlp/markovian prop, best rollout+DA, cond#=1.59e6)": (
        "artifacts/stage1_ae_patched_full_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep.pt",
        "artifacts/stage2_prop_patched_full_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep_warmstart_k12_300ep.pt",
    ),
    "75 (fourier_mlp AE+prop, baseline, cond#=20.98)": (
        "artifacts/stage1_ae_patched_full_section75_enc8_prop22_dec22_fourierifftreadout_wvar01_wspatialsigned01_logdet008_lambdaz00002_200ep.pt",
        "artifacts/stage2_prop_patched_full_section75_enc8_prop22_dec22_fourierifftreadout_wvar01_wspatialsigned01_logdet008_lambdaz00002_200ep_warmstart_k12_300ep.pt",
    ),
    "81 (vit+fourierifft hybrid AE, best val_kmax=0.0286, cond#=22.3)": (
        "artifacts/stage1_ae_patched_full_section81_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_section75regs_200ep.pt",
        "artifacts/stage2_prop_patched_full_section81_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_section75regs_200ep_warmstart_k12_300ep.pt",
    ),
    "82 (=81, d_latent=56, w_spatial/lambda_z x1.5, val_kmax=0.084, user saw jumps in GIF)": (
        "artifacts/stage1_ae_patched_full_section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep.pt",
        "artifacts/stage2_prop_patched_full_section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep_warmstart_k12_300ep.pt",
    ),
    "85 (=81, no lambda_z, w_var x2, w_spatial x1.2, +w_smooth=0.00075, val_kmax=0.0279)": (
        "artifacts/stage1_ae_patched_full_section85_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wvar002_wspatial0012_wsmooth00075_nolambdaz_200ep.pt",
        "artifacts/stage2_prop_patched_full_section85_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wvar002_wspatial0012_wsmooth00075_nolambdaz_200ep_warmstart_k12_300ep.pt",
    ),
    "86 (=52, w_spatial x1.3, +w_smooth=0.001, val_kmax=0.0283, worse than 52)": (
        "artifacts/stage1_ae_patched_full_section86_mlpmarkovian_wvar002_wspatial0013_wsmooth0001_logdet0035_200ep.pt",
        "artifacts/stage2_prop_patched_full_section86_mlpmarkovian_wvar002_wspatial0013_wsmooth0001_logdet0035_200ep_warmstart_k12_300ep.pt",
    ),
    "89 (=85, encoder enc_out_modes=8, local pooling w8, w_spatial=.014, w_smooth=.0009)": (
        "artifacts/stage1_ae_patched_full_section89_vitfourieriffthybrid_encoutmodes8_poollocalw8_wspatial014_wsmooth0009_200ep.pt",
        "artifacts/stage2_prop_patched_full_section89_vitfourieriffthybrid_encoutmodes8_poollocalw8_wspatial014_wsmooth0009_200ep_warmstart_k12_300ep.pt",
    ),
    "90 (=89, dec_fno_modes=12, dec_pool=mean, w_spatial=.019, w_smooth=.0005)": (
        "artifacts/stage1_ae_patched_full_section90_vitfourieriffthybrid_encoutmodes8_decfnomodes12_poollocalw8_decpoolmean_wspatial019_wsmooth0005_200ep.pt",
        "artifacts/stage2_prop_patched_full_section90_vitfourieriffthybrid_encoutmodes8_decfnomodes12_poollocalw8_decpoolmean_wspatial019_wsmooth0005_200ep_warmstart_k12_300ep.pt",
    ),
    "91 (75pct ViT, prop=52's mlp/markovian, w_spatial=.019, w_smooth=.0005, val_kmax=0.0162)": (
        "artifacts/stage1_ae_patched_full_section91_vitfourieriffthybrid_dmodel116_fourierhidden185_75pctvit_propmlpmarkovian52_wspatial019_wsmooth0005_200ep.pt",
        "artifacts/stage2_prop_patched_full_section91_vitfourieriffthybrid_dmodel116_fourierhidden185_75pctvit_propmlpmarkovian52_wspatial019_wsmooth0005_200ep_warmstart_k12_300ep.pt",
    ),
    "92 (=91, token_mlp both sides, val_kmax=0.0135)": (
        "artifacts/stage1_ae_patched_full_section92_vitfourieriffthybrid_dmodel116_fourierhidden185_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep.pt",
        "artifacts/stage2_prop_patched_full_section92_vitfourieriffthybrid_dmodel116_fourierhidden185_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep_warmstart_k12_300ep.pt",
    ),
    "93 (=92 scaled down to ~55% size, val_kmax=0.0118 BEST OF ARC)": (
        "artifacts/stage1_ae_patched_full_section93_vitfourieriffthybrid_dmodel84_fourierhidden130_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep.pt",
        "artifacts/stage2_prop_patched_full_section93_vitfourieriffthybrid_dmodel84_fourierhidden130_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep_warmstart_k12_300ep.pt",
    ),
    "94 (=93 scaled to ~61% of 93's size, val_kmax=0.0123, plateaued)": (
        "artifacts/stage1_ae_patched_full_section94_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep.pt",
        "artifacts/stage2_prop_patched_full_section94_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial019_wsmooth0005_200ep_warmstart_k12_300ep.pt",
    ),
    "95 (=94, w_spatial=.03, w_smooth=.0015, val_kmax=0.0198, regressed)": (
        "artifacts/stage1_ae_patched_full_section95_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial03_wsmooth0015_200ep.pt",
        "artifacts/stage2_prop_patched_full_section95_vitfourieriffthybrid_dmodel64_fourierhidden95_tokenmlp_propmlpmarkovian52_wspatial03_wsmooth0015_200ep_warmstart_k12_300ep.pt",
    ),
}


def main():
    with h5py.File("artifacts/datasets/stage1_trajectories_dtsnap1.h5", "r") as f:
        traj = torch.tensor(f["trajectories"][:20], dtype=torch.float32)  # (n, T, NX)

    print(f"{'section':70s}  {'step_med':>9s}  {'step_p95':>9s}  {'stepN_med':>10s}  "
          f"{'curv_med':>9s}  {'curv_p95':>9s}  {'curvN_med':>10s}  {'prop_jac_med':>13s}  {'prop_jac_p95':>13s}")
    for name, (ae_path, prop_path) in SECTIONS.items():
        ae, ae_cfg, _ = load_autoencoder_checkpoint(ae_path)
        ae.eval()
        n, T, NX = traj.shape
        with torch.no_grad():
            z_flat = ae.encode(traj.reshape(n * T, NX))
        z_seq_t = z_flat.view(n, T, -1)
        z_seq = z_seq_t.numpy()

        enc_stats = encoded_trajectory_smoothness(z_seq)

        prop, prop_cfg, _ = load_propagator_checkpoint(prop_path, device="cpu")
        jac_norms = propagator_step_jacobian_spectral_norms(prop, z_seq_t, n_samples=200, seed=0)

        print(f"{name:70s}  {enc_stats['step_med']:9.3f}  {enc_stats['step_p95']:9.3f}  "
              f"{enc_stats['step_norm_med']:10.4f}  {enc_stats['curv_med']:9.3f}  {enc_stats['curv_p95']:9.3f}  "
              f"{enc_stats['curv_norm_med']:10.4f}  {np.median(jac_norms):13.3f}  {np.percentile(jac_norms, 95):13.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
