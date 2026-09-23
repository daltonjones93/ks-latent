#!/bin/zsh
# User-directed 2026-09-07, "Section 101": "okay, so now we have two new
# options for training the pde propagator, correct? can we try both and
# see what we can learn. please do this, document all your choices and
# run as many diagnostic as possible to determine what we can improve."
#
# Executes Stages 0-3 of docs/sine_transform_pde_plan.md's execution plan:
# a reduced-L dataset (Stage 0), ONE shared spectral_field autoencoder
# (Stage 1), and TWO propagators trained on top of its frozen z --
# backbone=spectral_pde with --spectral-integrator rk4 (generic explicit
# integration, the ORIGINAL proposal) vs --spectral-integrator etdrk4
# (KS's own true linear operator integrated exactly, only the nonlinear
# residual learned -- see docs/sine_transform_pde_plan.md Section 5a) --
# a controlled comparison: IDENTICAL z representation (same trained AE),
# identical MLP capacity/curriculum, differing ONLY in integration scheme.
#
# === Choice 1: L=22 (not the canonical L=100) ===
# Reuses this codebase's OWN existing, already-validated Gate 1 benchmark
# (tests/replication/test_gate1_kaplan_yorke.py::test_L22_lyapunov):
# ground-truth D_KY in [5.2, 5.6], lambda_1 in [0.043, 0.05], at
# KSConfig(L=22.0, NX=128, dt=0.05, snapshot_every=5, spinup_time=200.0).
# This is a real, low-dimensional-but-genuinely-chaotic regime with a
# PRECISELY KNOWN ground truth already established in this repo -- no need
# to screen candidate L values ourselves (docs/sine_transform_pde_plan.md
# Section 6's suggestion), we can reuse this directly. NX kept at the
# project's --profile full default of 256 (not the Gate-1 test's own 128)
# for consistency with every other section's convention -- this only means
# the field is somewhat MORE spectrally over-resolved than Gate 1's own
# test, never a correctness concern.
#
# === Choice 2: dt_snap=0.2 (user-requested, vs. canonical 1.0) ===
# snapshot_every=4 at the solver's own dt=0.05 (KSConfig.dt_snap = dt *
# snapshot_every = 0.2). trajectory_time=50.0 (vs. canonical 250.0) chosen
# to give the SAME per-run snapshot COUNT (250) as every canonical dataset
# in this project, at the finer spacing -- a deliberate scoping choice to
# keep dataset size/training cost comparable to precedent, not a claim
# that 50 physical time units is generous for this system (the L=22
# system's own decorrelation time ~1/lambda_1~=22, so one run spans only
# ~2.3 such times -- n_train=50 independent, separately-spun-up runs is
# relied on for statistical diversity rather than within-run length).
# Dataset pre-generated this turn (9.1s, see artifacts/logs/gen_dataset_L22.log):
# artifacts/datasets/stage1_trajectories_L22_dtsnap02.h5
#
# === Choice 3: K=8 kept rFFT modes, N_w=32 (patch_size=8) ===
# L=22's system is low-dimensional (D_KY~5.4) -- K=8 keeps wavenumbers up
# to k=2*pi*7/22~=2.0, well beyond KS's own instability range (unstable for
# 0<k<1, peak growth at k=1/sqrt(2)~=0.71), giving several "spare" modes
# beyond the estimated handful of genuinely unstable/energetic ones.
# N_w=32 (patch_size=NX/N_w=256/32=8, matching this project's canonical
# patch_size=8 convention) is comfortably above 3*K=24 (the classical 2/3
# dealiasing margin flagged in docs/sine_transform_pde_plan.md Section 2.3).
#
# === Choice 4: regularizers -- w_var=0, w_spatial=0, w_decorr=0, w_logdet=0 ===
# w_var/w_spatial/w_decorr all exist to INDUCE decorrelated, organized
# latent channels in an otherwise-unconstrained latent -- a truncated
# orthogonal rFFT basis already has decorrelated, frequency-ordered
# channels BY CONSTRUCTION, so all three would be fighting a battle already
# won structurally (per docs/sine_transform_pde_plan.md Section 5, point 3).
# w_logdet is DROPPED here too (an extension beyond the user's literal
# request, reasoned through this turn): it exists to prevent ARBITRARY
# per-channel variance collapse during joint training, but a real KS
# field's amplitude spectrum genuinely, physically decays with
# wavenumber -- higher kept modes SHOULD have smaller variance than low
# ones, and w_logdet would fight that correct structure rather than an
# actual collapse. Verified via the covariance-spectrum diagnostic below,
# not merely assumed safe.
#
# === Choice 5: w_lowpass=0.003, lowpass_power=1.0 ===
# Verified by direct computation this turn (fresh, untrained
# SpectralFieldAutoencoderConfig(K=8, L=22.0) instance, 256 real states
# from this dataset): at init, l_recon=1.190, l_lowpass(power=1)=54.045
# (ratio ~45.4x). w_lowpass=0.003 gives a weighted contribution of
# ~0.162 (~14% of l_recon's own magnitude at init) -- a gentle secondary
# regularizer, not a dominant term, matching this project's established
# calibration practice for every other regularizer introduced this
# project. power=1.0 matches the user's literal "proportional to their
# frequency" request (docs/sine_transform_pde_plan.md Section 2.2 also
# discusses power=2, the classical H^1 Sobolev seminorm, as a documented
# follow-up ablation, not run this turn).
#
# === Choice 6: propagator sizing -- hidden=128, n_blocks=3 (both variants) ===
# Matches this project's own long-established "mlp"/markovian propagator
# convention (Section 52 and many since) for the pointwise MLP's capacity
# -- identical for BOTH --spectral-integrator rk4 and etdrk4, so capacity
# is not a confound between them.
#
# === Choice 7 (REVISED after direct timing): k_max=12/k_mid=8 (canonical,
# not the originally-planned 24), ode_substeps=1 (not 4), epochs=80
# (not 300), Stage 2 runs SEQUENTIAL not parallel ===
# Directly timed before committing: k_max=4/ode_substeps=4 came back at
# ~60s/epoch; even k_max=12/ode_substeps=1 (run in PARALLEL, contending for
# the same MPS device) still cost ~75s/epoch -- projecting the ORIGINALLY
# planned k_max=24/ode_substeps=4/300-epoch run to ~30 HOURS per propagator,
# clearly infeasible. Revised down:
#   - ode_substeps=1 for BOTH integrators (a real, informative choice, not
#     just a compute cut: ETDRK4's entire selling point in
#     docs/sine_transform_pde_plan.md Section 5a is that it does NOT need
#     fine sub-stepping for stability, unlike explicit RK4 -- comparing
#     both at the SAME ode_substeps=1 is exactly the comparison that
#     matters, and if plain "rk4" struggles at this coarseness while
#     "etdrk4" doesn't, that IS the result we're looking for, not a
#     confound to fix).
#   - k_max=12/k_mid=8, reverting to this project's long-established
#     canonical values rather than the reasoned-but-unaffordable k_max=24 --
#     the "shorter fraction of the system's own Lyapunov time than
#     canonical dt_snap=1.0 training gets" caveat from the original
#     reasoning STANDS (12*0.2=2.4 physical time units, ~0.11 Lyapunov
#     times at this L=22 system's lambda_1~=0.045) -- flagged as an
#     accepted limitation of this first pass, not a solved problem.
#   - epochs=80 (k_warmup_epochs=56, k_mid_epochs=46, same ~70%/58% ratios
#     as the canonical 300/210/175 schedule) -- justified empirically, not
#     just for speed: the 2-epoch timing runs already show
#     val_kmax_mse dropping from ~0.015-0.017 to ~0.002-0.004 within just 2
#     epochs on this simpler, low-dimensional L=22 system, i.e. this
#     appears to be a genuinely fast-converging optimization landscape
#     here, not merely truncated early.
#   - Stage 2 runs sequential (not parallel, unlike the timing check) --
#     both a fairer wall-clock comparison (no mutual MPS contention) and
#     the actual reason the timing check itself came back slower than
#     hoped.
# Estimated cost with these revisions: ~1-1.5 hours per propagator,
# ~2-3 hours total for both plus Gate 3/4 diagnostics -- still substantial,
# run as one long background job.
#
# === Aux propagator (Stage 1 L_pred only) ===
# Plain "mlp"/markovian (NOT spectral_pde -- AuxPropagatorConfig doesn't
# support this backbone, matching existing precedent for "node"/"cnn",
# which are ALSO full-propagator-only -- see PropagatorConfig's
# 'spectral_pde' docstring / this session's implementation notes). This is
# fine: Stage 1's aux propagator only provides a cheap joint-training
# regularization signal; the REAL propagator (either integrator) is built
# fresh in Stage 2 via --backbone spectral_pde, entirely independent of
# whatever architecture Stage 1's aux used.
# === Gate 3/4 script compatibility note (patched this turn) ===
# run_da_pff.py and run_diagnostics.py both HARDCODED KSConfig(L=100.0, ...)
# in their real (--ae-checkpoint-driven) setup paths, regardless of the
# checkpoint's actual training L -- silently wrong for this experiment
# (would compare an L=22-trained propagator against ground truth generated
# at L=100). Both now accept --L (default 100.0, unchanged behavior for
# every prior run) -- see each script's own --L help text. run_analysis_suite.py
# needed no patch (it never constructs its own KSConfig in the real path,
# only reads --dataset/--points-dataset directly) -- it just needs an L=22
# --points-dataset, generated below, since it has no auto-generation
# fallback at all (raises FileNotFoundError otherwise).
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap02.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
AE_TAG=section101_spectralfield_L22_K8_Nw32_wlowpass003
STAGE2_COMMON="--k-max 12 --k-mid 8 --k-warmup-epochs 56 --k-mid-epochs 46 --epochs 80 --ode-substeps 1"

if [ ! -f "$POINTS_DATASET" ]; then
  echo "=== [0/5] Generating L=22 attractor-points dataset (topology/dimension estimation) ==="
  mamba run -n da_env python -c "
from ks_latent.config import KSConfig
from ks_latent.solver.dataset import generate_attractor_point_dataset
cfg = KSConfig(L=22.0, NX=256, dt=0.05, snapshot_every=4, spinup_time=200.0, seed=0)
generate_attractor_point_dataset(cfg, '$POINTS_DATASET', n_runs=2000, spinup_discard_snapshots=100, post_spinup_time=50.0)
"
fi

echo "=== [1/5] Stage 1: spectral_field AE (K=8, N_w=32, L=22.0) + mlp/markovian aux, w_var=w_spatial=w_decorr=w_logdet=0, w_lowpass=0.003, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 0.2 \
  --encoder spectral_field --d-latent 32 --spectral-K 8 --spectral-L 22.0 \
  --aux-backbone mlp --mode markovian \
  --w-var 0 --w-spatial 0 --w-decorr 0 --w-logdet 0 \
  --w-lowpass 0.003 --lowpass-power 1.0 \
  --amp --epochs 200 --checkpoint-every 20 \
  --tag "$AE_TAG" \
  > artifacts/logs/stage1_${AE_TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${AE_TAG}.pt

echo "=== [2/5] Latent covariance spectrum (verifying the w_logdet=0 choice: is any collapse a physically-expected spectral decay, or pathological?) ==="
mamba run -n da_env python -c "
import h5py, numpy as np, torch
from ks_latent.models import load_autoencoder_checkpoint

with h5py.File('$DATASET','r') as f:
    traj = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, NX = traj.shape
ae, cfg, _ = load_autoencoder_checkpoint('$AE')
ae.eval()
with torch.no_grad():
    z = ae.encode(traj.reshape(n*T, NX))
z_np = z.numpy()
cov = np.cov(z_np, rowvar=False)
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print('spectrum: min_eig=%.4e  cond#=%.4e  top_eig=%.3f' % (eig[-1], eig[0]/eig[-1], eig[0]))
print('participation ratio: %.3f (out of d_latent=%d)' % (eig.sum()**2 / (eig**2).sum(), cfg.d_latent))
print('per-mode variance (should decay with k -- real modes 0..K-1, imag modes K..2K-1):')
print(np.array2string(z_np.var(axis=0), precision=4))
" > artifacts/logs/spectrum_${AE_TAG}.log 2>&1
cat artifacts/logs/spectrum_${AE_TAG}.log

echo "=== [3/5] Stage 2 (rk4): fresh spectral_pde propagator, --spectral-integrator rk4, hidden=128/n_blocks=3, k_max=12/ode_substeps=1, --amp, 80 epochs ==="
RK4_TAG="${AE_TAG}_spectralpde_rk4"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone spectral_pde --mode markovian --spectral-integrator rk4 \
  --hidden 128 --n-blocks 3 \
  --amp $STAGE2_COMMON \
  --tag "$RK4_TAG" \
  > artifacts/logs/stage2_${RK4_TAG}.log 2>&1

echo "=== [4/5] Stage 2 (etdrk4): fresh spectral_pde propagator, --spectral-integrator etdrk4, hidden=128/n_blocks=3, k_max=12/ode_substeps=1, --amp, 80 epochs ==="
ETD_TAG="${AE_TAG}_spectralpde_etdrk4"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --backbone spectral_pde --mode markovian --spectral-integrator etdrk4 \
  --hidden 128 --n-blocks 3 \
  --amp $STAGE2_COMMON \
  --tag "$ETD_TAG" \
  > artifacts/logs/stage2_${ETD_TAG}.log 2>&1

echo "=== [5/5] Gate 3/4 diagnostics on BOTH propagators (Lyapunov/D_KY via run_analysis_suite.py, DA skill via run_da_pff.py, D1-D9 via run_diagnostics.py) -- sequential PER TAG (the 3 scripts within one tag run in parallel, matching established precedent; the two TAGS run one after another to avoid 6-way MPS/CPU contention) ==="
for TAG in "$RK4_TAG" "$ETD_TAG"; do
  PROP=artifacts/stage2_prop_patched_full_${TAG}.pt
  mamba run -n da_env python scripts/run_analysis_suite.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 0.2 \
    --tag "$TAG" > artifacts/logs/gate3_analysis_${TAG}.log 2>&1 &
  mamba run -n da_env python scripts/run_da_pff.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --dt-snap 0.2 --L 22.0 \
    --tag "$TAG" > artifacts/logs/gate3_da_${TAG}.log 2>&1 &
  mamba run -n da_env python scripts/run_diagnostics.py \
    --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
    --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
    --tag "$TAG" > artifacts/logs/gate4_diagnostics_${TAG}.log 2>&1 &
  wait
done

echo "=== Section 101 complete ==="
echo "--- rk4 stage2 tail ---"; tail -3 artifacts/logs/stage2_${RK4_TAG}.log
echo "--- etdrk4 stage2 tail ---"; tail -3 artifacts/logs/stage2_${ETD_TAG}.log
