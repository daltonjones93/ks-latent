"""One-off diagnostic (2026-09-04, user-directed): "design and run an
analysis to try to determine why we can get lower val_kmax_mse values
for something like 75, and not for 76/77 ... what is it about latent
space that makes predicting the next state hard?"

Compares Sections 75/76/77's AEs (encoder/decoder ONLY -- no trained
propagator needed, so this is a fair comparison even for 76/77 whose
Stage-2 propagators are killed/incomplete) on three propagator-
independent latent-geometry measures:

1. Covariance spectrum shape: condition number and participation ratio
   (effective number of independent directions carrying variance),
   computed from each AE's own encoded validation trajectories.
2. Encoder Jacobian amplification: the spectral norm of d(z)/d(u) at
   many sampled physical states, measuring how much a small physical
   perturbation gets amplified (or damped) once mapped into latent space.
3. Local one-step predictability of the TRUE (encoder-induced, no
   propagator) z_t -> z_{t+1} map: for k-NN pairs of time-indices with
   similar z_t (excluding temporally adjacent pairs), the ratio
   |z_{t+1}-z_{t'+1}| / |z_t-z_{t'}| -- >>1 means nearby latent states
   diverge sharply one step later (a locally rough/hard-to-fit map for
   ANY propagator architecture), ~1 means the one-step map is locally
   smooth/predictable. Also computed in raw physical (u) space as an
   intrinsic-dynamics baseline, so an "encoder amplification factor"
   (z-space ratio / u-space ratio) isolates what the ENCODING adds on
   top of KS's own intrinsic chaotic sensitivity.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import torch

from ks_latent.models import load_autoencoder_checkpoint

SECTIONS = {
    "75 (fourier_mlp, GOOD, val_kmax=0.039)": "artifacts/stage1_ae_patched_full_section75_enc8_prop22_dec22_fourierifftreadout_wvar01_wspatialsigned01_logdet008_lambdaz00002_200ep.pt",
    "77 (fourier_mlp, mid-training, val_k12~0.18@k_now=9)": "artifacts/stage1_ae_patched_full_section77_enc8_prop22_dec22_fourierifftreadout_wspatialsigned03_3x_lambdaz00002_200ep.pt",
    "76 (fourier_mlp, WORST, val_kmax~0.29)": "artifacts/stage1_ae_patched_full_section76_enc8_prop22_dec22_fourierifftreadout_wvar00005_wspatialsigned04_logdet004_lambdaz0005_200ep.pt",
    "78 (fourier_mlp, d_latent=64, PLATEAUED val_k12~0.143)": "artifacts/stage1_ae_patched_full_section78_dlatent64_enc8_prop32_dec32_fourierifftreadout_wspatial015_wlogdet016_lambdaz00002_200ep.pt",
    "65 (vit AE, mlp/markovian prop, GOOD val_kmax=0.089)": "artifacts/stage1_ae_patched_full_section65_mlpmarkovian_wvar0015_wspatialsigned0035_logdet0005_lambdaz0001fromepoch0_200ep.pt",
}


def participation_ratio(eig: np.ndarray) -> float:
    return float(eig.sum() ** 2 / (eig**2).sum())


def encoder_jacobian_spectral_norms(ae, u_samples: torch.Tensor) -> np.ndarray:
    """Spectral norm (largest singular value) of d(encode(u))/d(u) at
    each row of u_samples. d_latent~44, NX=256 -> full Jacobian via
    torch.func.jacrev is cheap per-sample."""
    from torch.func import jacrev

    def enc_fn(u_row):
        return ae.encode(u_row.unsqueeze(0)).squeeze(0)

    norms = []
    for i in range(u_samples.shape[0]):
        J = jacrev(enc_fn)(u_samples[i])  # (d_latent, NX)
        sv = torch.linalg.svdvals(J)
        norms.append(sv[0].item())
    return np.array(norms)


def knn_one_step_sensitivity(
    seq: np.ndarray, next_seq: np.ndarray, n_pairs: int = 4000, min_time_gap: int = 5, seed: int = 0
) -> np.ndarray:
    """`seq`/`next_seq`: (N, d) current/one-step-later states, any space
    (z or u). For n_pairs random anchor points, finds each anchor's
    nearest OTHER point (excluding a +-min_time_gap window around its own
    original time index, to avoid trivial temporal-autocorrelation
    matches) and returns the array of |next_seq[i]-next_seq[j]| /
    (|seq[i]-seq[j]|+eps) ratios."""
    rng = np.random.default_rng(seed)
    N = seq.shape[0]
    anchors = rng.choice(N, size=min(n_pairs, N), replace=False)
    ratios = []
    # brute-force pairwise distances against a random reference subset (cheap enough at N~a few thousand)
    ref_idx = rng.choice(N, size=min(N, 3000), replace=False)
    ref = seq[ref_idx]
    for a in anchors:
        d = np.linalg.norm(ref - seq[a], axis=-1)
        d_masked = d.copy()
        close_time = np.abs(ref_idx - a) <= min_time_gap
        d_masked[close_time] = np.inf
        j_local = np.argmin(d_masked)
        if not np.isfinite(d_masked[j_local]):
            continue
        j = ref_idx[j_local]
        num = np.linalg.norm(next_seq[a] - next_seq[j])
        den = np.linalg.norm(seq[a] - seq[j]) + 1e-8
        ratios.append(num / den)
    return np.array(ratios)


def main():
    with h5py.File("artifacts/datasets/stage1_trajectories_dtsnap1.h5", "r") as f:
        traj = torch.tensor(f["trajectories"][:20], dtype=torch.float32)  # (n, T, NX)
    n, T, NX = traj.shape
    u_flat_all = traj.reshape(n * T, NX)

    rng_torch = torch.Generator().manual_seed(0)
    jac_sample_idx = torch.randperm(u_flat_all.shape[0], generator=rng_torch)[:24]
    u_jac_samples = u_flat_all[jac_sample_idx]

    print(f"{'section':55s}  {'cond#':>10s}  {'top_eig':>8s}  {'partic.ratio':>12s}  "
          f"{'enc_jac_norm(median)':>20s}  {'z_ratio(med)':>12s}  {'u_ratio(med)':>12s}  {'amp_factor':>10s}")
    for name, ckpt_path in SECTIONS.items():
        ae, cfg, _ = load_autoencoder_checkpoint(ckpt_path)
        ae.eval()
        with torch.no_grad():
            z_flat = ae.encode(u_flat_all)
        z_np = z_flat.numpy()
        cov = np.cov(z_np, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        cond = eig[0] / eig[-1]
        pr = participation_ratio(eig)

        jac_norms = encoder_jacobian_spectral_norms(ae, u_jac_samples)

        # per-trajectory-run one-step pairs (t, t+1), excluding the run boundary
        z_seq = z_np.reshape(n, T, -1)
        u_seq = u_flat_all.numpy().reshape(n, T, NX)
        z_cur = z_seq[:, :-1].reshape(-1, z_seq.shape[-1])
        z_next = z_seq[:, 1:].reshape(-1, z_seq.shape[-1])
        u_cur = u_seq[:, :-1].reshape(-1, NX)
        u_next = u_seq[:, 1:].reshape(-1, NX)

        z_ratios = knn_one_step_sensitivity(z_cur, z_next)
        u_ratios = knn_one_step_sensitivity(u_cur, u_next)
        z_med, u_med = np.median(z_ratios), np.median(u_ratios)
        amp = z_med / u_med

        print(f"{name:55s}  {cond:10.2f}  {eig[0]:8.2f}  {pr:12.3f}  "
              f"{np.median(jac_norms):20.3f}  {z_med:12.3f}  {u_med:12.3f}  {amp:10.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
