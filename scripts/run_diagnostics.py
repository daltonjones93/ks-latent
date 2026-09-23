#!/usr/bin/env python
"""Phase 6 entry point: run diagnostics D1-D8 on a trained checkpoint (no
retraining) and emit docs/diagnostics_report.md (brief §8, Gate 4; D6 added
2026-08-30, D7 added 2026-08-31, D8 added 2026-09-01, all user-directed,
none part of the original brief -- see ks_latent.analysis.diagnostics's
D6/D7/D8 module docstrings).

    python scripts/run_diagnostics.py                  # full run, needs Stage-1/2 checkpoints
    python scripts/run_diagnostics.py --profile smoke   # <60s, trains tiny models first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from ks_latent.analysis.diagnostics import (
    coupling_graph_diagnostic,
    coupling_graph_diagnostic_history,
    decoder_sensitivity_map,
    encoder_sensitivity_map,
    local_dimension_vs_length,
    same_time_coupling_diagnostic,
    same_time_coupling_diagnostic_signed,
    smoothness_diagnostic,
    temporal_coherence_diagnostic,
    translation_representation,
    wavenumber_content,
)
from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    KSConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.propagator import AuxPropagator, LatentPropagator
from ks_latent.solver.dataset import generate_attractor_point_dataset, generate_trajectory_dataset
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = Path("docs/diagnostics_report.md")


def _smoke_setup():
    ks_cfg = KSConfig(L=22.0, NX=64, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=0)
    ae_cfg = AutoencoderConfig(
        NX=64, patch_size=8, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6,
    )
    aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
    device = torch.device("cpu")

    traj_path = ARTIFACTS_DIR / "datasets" / "smoke_diag_traj.h5"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    generate_trajectory_dataset(ks_cfg, traj_path, n_train=4, n_val=2, trajectory_time=15.0)
    with h5py.File(traj_path, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)

    ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    train_stage1(ae, aux, trajectories[:n_train], trajectories[n_train:],
                 Stage1TrainingConfig(epochs=2, batch_size=8), device)
    ae.eval()

    seq = encode_dataset_with_shifts(ae, trajectories, [0, ae_cfg.NX // 2], device)
    T = seq.shape[1]
    split = int(T * 0.7)
    prop = LatentPropagator(PropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1))
    train_stage2(prop, seq[:, :split], seq[:, split:],
                 Stage2TrainingConfig(epochs=2, batch_size=8, k_max=4, k_warmup_epochs=1), device)
    prop.eval()

    points_path = ARTIFACTS_DIR / "datasets" / "smoke_diag_points.h5"
    generate_attractor_point_dataset(ks_cfg, points_path, n_runs=150, spinup_discard_snapshots=2, post_spinup_time=1.0)
    with h5py.File(points_path, "r") as f:
        points_phys = torch.tensor(f["points"][:], dtype=torch.float32)

    return ae, prop, ks_cfg, trajectories, points_phys


def _full_setup(args):
    ae, ae_cfg, _ = load_autoencoder_checkpoint(args.ae_checkpoint)
    if args.latent_permutation:
        ae = PermutedAutoencoder(ae, load_latent_permutation(args.latent_permutation, args.latent_permutation_key))
    ae.eval()

    prop, prop_cfg, prop_ckpt = load_propagator_checkpoint(args.prop_checkpoint, device="cpu")
    prop.eval()

    ks_cfg = KSConfig(L=args.L, NX=ae_cfg.NX, dt=0.05, snapshot_every=5, spinup_time=500.0, seed=args.seed)
    with h5py.File(args.dataset, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)

    points_path = Path(args.points_dataset)
    if not points_path.exists():
        generate_attractor_point_dataset(ks_cfg, points_path, n_runs=2000, spinup_discard_snapshots=100, post_spinup_time=50.0)
    with h5py.File(points_path, "r") as f:
        points_phys = torch.tensor(f["points"][:], dtype=torch.float32)

    return ae, prop, ks_cfg, trajectories, points_phys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ae-checkpoint", default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument("--prop-checkpoint", default="artifacts/stage2_prop_patched_full.pt")
    parser.add_argument("--dataset", default="artifacts/datasets/stage1_trajectories_dtsnap1.h5")
    parser.add_argument("--points-dataset", default="artifacts/datasets/attractor_points.h5")
    parser.add_argument(
        "--L", type=float, default=100.0,
        help="Domain length used both (a) to auto-generate --points-dataset if it "
        "doesn't already exist, and (b) as the 'L' this script's diagnostics (e.g. "
        "wavenumber/extensivity ones) read for their own physical-unit conversions "
        "(added 2026-09-07 -- this was hardcoded at 100.0 regardless of the "
        "checkpoint's own training L, silently wrong for any non-canonical-L "
        "experiment, e.g. docs/sine_transform_pde_plan.md's reduced-L spectral_pde "
        "runs). MUST match the checkpoint's actual training KSConfig.L.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix appended to the output report filename (added 2026-08-30, "
        "matching train_stage1/2_patched.py's --tag) so comparing runs on "
        "different checkpoints doesn't overwrite the previous run's report.",
    )
    parser.add_argument(
        "--latent-permutation", type=str, default=None,
        help="Path to a saved D3 Fiedler permutation (added 2026-08-31, "
        "user-directed) -- either a bare .npy array or a "
        "diagnostics_arrays{tag}.npz (its 'd3_permutation' key is used). "
        "Wraps the loaded --ae-checkpoint in a PermutedAutoencoder "
        "(ks_latent/models/permuted_autoencoder.py) so downstream code sees "
        "latent vectors reindexed by this permutation -- no AE weights "
        "change, this is a pure relabeling. See "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 20.",
    )
    parser.add_argument(
        "--latent-permutation-key", type=str, default="d3_permutation",
        help="--latent-permutation .npz only. Which array to read: "
        "'d3_permutation' (default, propagator-Jacobian-derived), "
        "'d6_permutation' (raw-encoded-data TEMPORAL-coherence-derived, "
        "lagged correlation across time, user-directed 2026-08-31 -- no "
        "propagator needed, see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md "
        "Section 21), or 'd7_permutation' (raw-encoded-data SAME-TIME "
        "correlation-derived, no lag and no propagator, user-directed "
        "2026-08-31 after correctly distinguishing this from D6 -- see "
        "Section 22).",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.profile == "smoke":
        ae, prop, ks_cfg, trajectories, points_phys = _smoke_setup()
        n_null = 200
        patch_lengths = [2, 4, 8, 16]
    else:
        for p in [args.ae_checkpoint, args.prop_checkpoint, args.dataset]:
            if not Path(p).exists():
                raise FileNotFoundError(f"{p} not found; run Phase 3 training/dataset generation first.")
        ae, prop, ks_cfg, trajectories, points_phys = _full_setup(args)
        n_null = 1000
        patch_lengths = [5, 10, 20, 30, 40, 50]

    d, NX, L = ae.cfg.d_latent, ae.cfg.NX, ks_cfg.L
    rng = np.random.default_rng(args.seed)

    n_runs, T, _ = trajectories.shape
    u_flat = trajectories.reshape(n_runs * T, NX)
    u_idx = rng.choice(u_flat.shape[0], size=min(200, u_flat.shape[0]), replace=False)
    u_samples = u_flat[u_idx]
    with torch.no_grad():
        z_samples = ae.encode(u_samples)

    def decode_single(z):
        return ae.decode(z.unsqueeze(0)).squeeze(0)

    def encode_single(u):
        return ae.encode(u.unsqueeze(0)).squeeze(0)

    # D1
    d1_decoder = decoder_sensitivity_map(decode_single, z_samples, L)
    d1_encoder = encoder_sensitivity_map(encode_single, u_samples, L)

    # D2
    d2 = wavenumber_content(d1_decoder.S, L)

    # D3: needs (z_prev, z_curr) pairs -- or, for mode="history", an
    # n_history-length window -- from real encoded trajectories.
    is_history = getattr(prop, "mode", None) == "history"
    n_hist = prop.cfg.n_history if is_history else 2
    run_idx = rng.integers(0, n_runs, size=min(150, n_runs * (T - n_hist + 1)))
    t_idx = rng.integers(0, T - n_hist + 1, size=len(run_idx))
    if is_history:
        with torch.no_grad():
            u_hist = torch.stack([trajectories[run_idx, t_idx + h] for h in range(n_hist)], dim=1)
            z_hist = ae.encode(u_hist.reshape(-1, NX)).reshape(len(run_idx), n_hist, -1)
        d3 = coupling_graph_diagnostic_history(prop.step_history, z_hist, n_null=n_null, seed=args.seed)
    else:
        with torch.no_grad():
            u_prev = trajectories[run_idx, t_idx]
            u_curr = trajectories[run_idx, t_idx + 1]
            z_prev = ae.encode(u_prev)
            z_curr = ae.encode(u_curr)
        d3 = coupling_graph_diagnostic(prop.step, z_prev, z_curr, n_null=n_null, seed=args.seed)

    # D4
    shifts = list(range(0, NX, max(1, NX // 16)))
    d4 = translation_representation(ae.encode, u_samples, shifts, NX)

    # D5
    points_np = points_phys.numpy()
    d5 = local_dimension_vs_length(points_np, patch_lengths, rng)

    # D6 (user-directed 2026-08-30, not part of the original brief D1-D5 --
    # see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8 and
    # ks_latent.analysis.diagnostics's D6 module docstring). Needs a
    # genuine, ordered-in-time encoded trajectory (unlike D3's independent
    # (u_prev, u_curr) pairs) -- encode every frame of `trajectories` in
    # its own trajectory order.
    with torch.no_grad():
        z_traj = ae.encode(trajectories.reshape(n_runs * T, NX)).reshape(n_runs, T, d).numpy()
    d6_window, d6_max_lag = (100, 5) if args.profile == "full" else (20, 2)
    d6 = temporal_coherence_diagnostic(
        z_traj, window=d6_window, max_lag=d6_max_lag, n_null=n_null, seed=args.seed
    )

    # D7 (user-directed 2026-08-31, not part of the original brief -- see
    # ks_latent.analysis.diagnostics's D7 module docstring). Same-instant
    # (no lag, no propagator) channel correlation across real encoded
    # snapshots -- pool every frame of every trajectory as an i.i.d. sample.
    d7 = same_time_coupling_diagnostic(z_traj.reshape(n_runs * T, d), n_null=n_null, seed=args.seed)

    # D8 (user-directed 2026-09-01, not part of the original brief -- see
    # ks_latent.analysis.diagnostics's D8 module docstring). Same same-time
    # data as D7, but SIGNED correlation and NO Fiedler permutation search
    # (scored in the CURRENT, FIXED index order) -- "does the ACTUAL
    # ordering exhibit same-sign local coherence," distinct from D7's
    # "does some ordering reveal |correlation| coupling structure."
    d8 = same_time_coupling_diagnostic_signed(z_traj.reshape(n_runs * T, d), n_null=n_null, seed=args.seed)

    # D9 (user-directed 2026-09-05, not part of the original brief -- see
    # ks_latent.analysis.diagnostics's smoothness_diagnostic/SmoothnessResult
    # docstrings, and scripts/analyze_latent_smoothness.py's original
    # motivation: "please run visualizations, smoothness diagnostic + Gate
    # 3/4 (in the future add smoothness diagnostic to gate 4)"). Reuses the
    # SAME z_traj already encoded for D6/D7/D8 -- no new data loading.
    z_traj_t = torch.from_numpy(z_traj)
    d9 = smoothness_diagnostic(z_traj, z_traj_t, prop, n_samples=200, seed=args.seed)

    # --- Report ---
    lines = ["# Diagnostics Report (Phase 6, D1-D9)", "", f"Profile: `{args.profile}`", ""]

    lines += ["## D1: sensitivity maps", ""]
    lines += ["| latent k | centroid (x) | resultant length R | circular spread |", "|---|---|---|---|"]
    for k in d1_decoder.order_by_centroid:
        s = d1_decoder.per_latent[k]
        lines.append(f"| {k} | {s.centroid:.3f} | {s.resultant_length:.3f} | {s.spread:.3f} |")
    lines.append("")

    lines += ["## D2: wavenumber content", ""]
    lines += ["| latent k | spectral centroid \\|k\\| | spectral bandwidth | wavelet energy entropy |", "|---|---|---|---|"]
    for k in range(d):
        lines.append(f"| {k} | {d2.spectral_centroid[k]:.4f} | {d2.spectral_bandwidth[k]:.4f} | {d2.wavelet_energy_entropy[k]:.4f} |")
    lines.append("")
    mean_entropy = float(np.mean(d2.wavelet_energy_entropy))
    lines.append(f"Mean wavelet energy entropy: {mean_entropy:.3f} (0=maximally localized, 1=delocalized).")
    lines.append("")

    lines += ["## D3: Jacobian coupling graph", ""]
    lines.append(f"Bandedness (Fiedler-seriated): **{d3.bandedness_observed:.4f}**")
    lines.append(f"**p-value vs. {n_null} random permutations: {d3.bandedness_p_value:.4f}**")
    verdict_d3 = "SIGNIFICANT bandedness" if d3.bandedness_p_value < 0.05 else "NOT significant (dense/unstructured coupling)"
    lines.append(f"Verdict: {verdict_d3}")
    lines.append(f"Discovered ordering (Fiedler permutation): `{list(int(i) for i in d3.permutation)}`")
    lines.append("")
    lines.append("Note (brief §8 D3): the original seriation attempt used the *covariance*, "
                 "which is approximately I by construction because of L_decorr, so it could not "
                 "have found anything -- this diagnostic uses the propagator Jacobian instead.")
    lines.append("")

    lines += ["## D4: translation representation", ""]
    lines += ["| shift c | relative residual |", "|---|---|"]
    for c, r in zip(d4.shifts, d4.relative_residuals):
        lines.append(f"| {c} | {r:.4f} |")
    lines.append("")
    mean_resid = float(np.mean(d4.relative_residuals))
    mean_group_err = float(np.mean(d4.group_property_errors)) if len(d4.group_property_errors) else float("nan")
    lines.append(f"Mean relative residual: {mean_resid:.4f}")
    lines.append(f"Mean group-property error (||R(c1)R(c2) - R(c1+c2)|| / ||R(c1+c2)||): {mean_group_err:.4f}")
    verdict_d4 = (
        "Genuine (approximately) equivariant translation representation found"
        if mean_resid < 0.3 and mean_group_err < 0.3
        else "No clean linear translation representation found in the existing latent"
    )
    lines.append(f"**Verdict: {verdict_d4}**")
    lines.append("")

    lines += ["## D5: local intrinsic dimension vs. patch length", ""]
    lines += ["| patch length | d_local (two-NN) |", "|---|---|"]
    for ell, dl in zip(d5.lengths, d5.d_local):
        lines.append(f"| {int(ell)} | {dl:.3f} |")
    lines.append("")
    lines.append(f"Fitted slope: {d5.slope:.4f} (95% CI [{d5.slope_ci[0]:.4f}, {d5.slope_ci[1]:.4f}]); "
                 f"extensivity prediction: 0.226. Intercept: {d5.intercept:.4f}.")
    lines.append("")

    lines += ["## D6: temporal coherence structure (not part of the original brief)", ""]
    lines.append(f"Bandedness (Fiedler-seriated): **{d6.bandedness_observed:.4f}**")
    lines.append(f"**p-value vs. {n_null} random permutations: {d6.bandedness_p_value:.4f}**")
    verdict_d6 = "SIGNIFICANT bandedness" if d6.bandedness_p_value < 0.05 else "NOT significant (no detectable temporal coherence structure)"
    lines.append(f"Verdict: {verdict_d6}")
    lines.append(f"Discovered ordering (Fiedler permutation): `{list(int(i) for i in d6.permutation)}`")
    lines.append("")
    lines.append(
        "Note (user-directed 2026-08-30, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 8): "
        "D3's Jacobian-bandedness is blind to a real coupling structure through a broadband-but-"
        "still-ring-respecting operator (e.g. an FNO spectral conv with many active modes). D6 "
        "instead looks for emergent spatial structure directly in the raw ENCODED DATA -- lagged, "
        f"windowed cross-correlation (window={d6_window}, max_lag={d6_max_lag}) between latent "
        "channels along real trajectories through time, a statistic the L_decorr regularizer's "
        "same-time covariance constraint does not touch -- independent of any one propagator."
    )
    lines.append("")

    lines += ["## D7: same-time channel correlation (not part of the original brief)", ""]
    lines.append(f"Bandedness (Fiedler-seriated): **{d7.bandedness_observed:.4f}**")
    lines.append(f"**p-value vs. {n_null} random permutations: {d7.bandedness_p_value:.4f}**")
    verdict_d7 = "SIGNIFICANT bandedness" if d7.bandedness_p_value < 0.05 else "NOT significant (no detectable same-time correlation structure)"
    lines.append(f"Verdict: {verdict_d7}")
    lines.append(f"Discovered ordering (Fiedler permutation): `{list(int(i) for i in d7.permutation)}`")
    lines.append("")
    lines.append(
        "Note (user-directed 2026-08-31, docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 22): "
        "D3's own docstring claims the naive same-time covariance 'is approximately I by "
        "construction because of L_decorr, so it could not have found anything' -- this was "
        "checked empirically against real checkpoints and found to be an overstatement (mean "
        "off-diagonal |correlation| ~0.1, max ~0.5 on the checkpoints tested, not remotely flat). "
        "D7 seriates that real same-instant |Pearson correlation| matrix directly -- no "
        "propagator, no time lag, the most literal reading of 'do neighboring latent variables "
        "vary together.'"
    )
    lines.append("")

    lines += ["## D8: same-time SIGNED channel correlation (not part of the original brief)", ""]
    lines.append(f"Signed bandedness (fixed index order, no permutation search): **{d8.bandedness_observed:.4f}**")
    lines.append(f"**p-value vs. {n_null} random relabelings: {d8.bandedness_p_value:.4f}**")
    verdict_d8 = "SIGNIFICANT signed bandedness" if d8.bandedness_p_value < 0.05 else "NOT significant (no detectable same-sign local coherence)"
    lines.append(f"Verdict: {verdict_d8}")
    lines.append("")
    lines.append(
        "Note (user-directed 2026-09-01): D7 uses |correlation|, so an anti-correlated "
        "near-neighbor scores identically to a correlated one -- it cannot distinguish "
        "'coupled' from 'coherent'. D8 uses SIGNED correlation and scores the ACTUAL, "
        "fixed latent index order directly (no Fiedler search -- that assumes non-negative "
        "edge weights, which a signed correlation matrix does not satisfy), against a null "
        "of random relabelings of the same channels' signed pairwise correlations. This is "
        "the diagnostic-side counterpart to Stage1/Stage2TrainingConfig.spatial_signed."
    )
    lines.append("")

    lines += ["## D9: real-trajectory smoothness (not part of the original brief)", ""]
    lines += ["| measure | median | p95 |", "|---|---|---|"]
    lines.append(f"| step size \\|\\|z_t+1 - z_t\\|\\| (raw) | {d9.step_med:.4f} | {d9.step_p95:.4f} |")
    lines.append(f"| step size (normalized by trajectory RMS radius) | {d9.step_norm_med:.4f} | {d9.step_norm_p95:.4f} |")
    lines.append(f"| curvature \\|\\|z_t+1 - 2z_t + z_t-1\\|\\| (raw) | {d9.curv_med:.4f} | {d9.curv_p95:.4f} |")
    lines.append(f"| curvature (normalized) | {d9.curv_norm_med:.4f} | {d9.curv_norm_p95:.4f} |")
    if d9.propagator_jacobian_supported:
        lines.append(f"| propagator step-Jacobian spectral norm | {d9.propagator_jacobian_med:.4f} | {d9.propagator_jacobian_p95:.4f} |")
    else:
        lines.append(f"| propagator step-Jacobian spectral norm | not computed ({d9.propagator_jacobian_skip_reason}) | -- |")
    lines.append("")
    lines.append(
        "Note (user-directed 2026-09-04/05, docs/HANDOFF_2026-09-04_FOURIER_HYBRID.md and "
        "scripts/analyze_latent_smoothness.py): the encoder-side step size/curvature (rows 1-4) "
        "is a propagator-independent measure of whether the ENCODER's real-trajectory placement "
        "of physical states is smooth (motivated directly by a GIF-visible 'huge jumps' "
        "observation on Section 82, later confirmed quantitatively by this exact measure). The "
        "propagator step-Jacobian spectral norm (row 5, when supported) is the trained "
        "propagator's own local Lipschitz constant, sampled at real points on the attractor -- "
        "a mixture of WANTED chaotic expansion (KS genuinely has positive Lyapunov exponents) "
        "and UNWANTED roughness, so a lower value is not unconditionally better, unlike rows 1-4."
    )
    lines.append("")

    suffix = f"_{args.tag}" if args.tag else ""
    report_path = REPORT_PATH.with_name(f"{REPORT_PATH.stem}{suffix}{REPORT_PATH.suffix}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    # D6's discovered ordering/embedding (and D3's coupling matrices) aren't
    # fully representable in the markdown report -- save the raw arrays too,
    # e.g. for plotting the embedding to check by eye whether it traces a
    # ring, or for re-ordering the latent to exploit the discovered structure.
    npz_path = ARTIFACTS_DIR / f"diagnostics_arrays{suffix}.npz"
    np.savez(
        npz_path,
        d3_A_jacobian=d3.A_jacobian, d3_permutation=d3.permutation,
        d6_C=d6.C, d6_permutation=d6.permutation, d6_embedding=d6.embedding,
        d7_A=d7.A, d7_permutation=d7.permutation, d7_embedding=d7.embedding,
        d8_A=d8.A,
    )
    print(f"wrote {report_path}")
    print(f"wrote {npz_path}")
    print(f"D3 p-value: {d3.bandedness_p_value:.4f}")
    print(f"D4 verdict: {verdict_d4}")
    print(f"D6 p-value: {d6.bandedness_p_value:.4f}")
    print(f"D7 p-value: {d7.bandedness_p_value:.4f}")
    print(f"D8 p-value: {d8.bandedness_p_value:.4f}")
    if d9.propagator_jacobian_supported:
        print(f"D9 smoothness: step_norm_med={d9.step_norm_med:.4f}  curv_norm_med={d9.curv_norm_med:.4f}  "
              f"prop_jacobian_med={d9.propagator_jacobian_med:.4f}  prop_jacobian_p95={d9.propagator_jacobian_p95:.4f}")
    else:
        print(f"D9 smoothness: step_norm_med={d9.step_norm_med:.4f}  curv_norm_med={d9.curv_norm_med:.4f}  "
              f"prop_jacobian=not computed ({d9.propagator_jacobian_skip_reason})")


if __name__ == "__main__":
    main()
