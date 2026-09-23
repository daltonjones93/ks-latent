#!/bin/zsh
# User-directed 2026-09-23: "alright, I want to try the self rollout
# spectrum-shape regularizer. But use it on L96 data in 16 dimensions,
# let the latent dimension be 8. Let the encoder, decoder and propagator
# be mlps. let the propagator have history. Suppose also that we use not
# only x, but x' (the time derivative of x) as a state, so ostensibly we
# have 32 dimensions to play with."
#
# Context: Section 203's rerun (docs/OPEN_QUESTIONS.md, docs/RESULTS.md)
# confirmed --multistep as the Stage-1 collapse cause (fixed: D_KY=2.117),
# but Stage 2 STILL collapsed to D_KY=0.00 even with a dedicated
# anti-collapse stack (w_varmatch + w_spatial + w_logdet_rollout_latent)
# -- the fourth independent confirmation of Stage-2 collapse on L96. The
# one remaining untried candidate flagged there: apply the pde_head's own
# self-rollout spectrum-shape mechanism (Section 193, built for the
# SEPARATE pde_head distillation target) directly to the MAIN propagator
# instead. Built today as Stage2TrainingConfig.w_spectrum_shape_self /
# ks_latent.training.loops._propagator_spectrum_shape_self_pool -- see
# those docstrings for the full mechanism. GENERALIZED (unlike the
# existing real-data-anchored w_spectrum_shape, which is mode="markovian"
# only) to mode="history" via a flatten-for-jacrev wrapper around
# propagator.step_history, since this experiment needs propagator memory.
#
# THREE independent new variables in this one experiment (all
# user-specified, not separately ablated here -- a genuinely combined
# test, not a controlled single-variable one):
#  1. New system SCALE: N=16 (every prior L96 section used N=64 or 256).
#  2. New DATA representation: --include-derivative appends the exact,
#     analytic dx/dt = l96_rhs(x, F) to every snapshot (ks_latent.solver.
#     lorenz96_dataset.generate_trajectory_dataset's new flag), doubling
#     the stored/encoded state from N=16 to 2N=32 -- "so ostensibly we
#     have 32 dimensions to play with". x and x' are different physical
#     quantities (different scales, x' std~11.7 vs x std~2.8 measured
#     directly on this dataset's train split) so they get their OWN
#     separate scalar normalization -- see that function's docstring.
#  3. New REGULARIZER: --w-spectrum-shape-self (this section's actual
#     point, above).
# Also: --encoder mlp / --aux-backbone mlp (Section 200's "back to the
# drawing board" architecture, not Section 201/203's ViT) and --mode
# history --n-history 6 (Section 201's history depth, ported to the mlp
# backbone -- config.py's own mode=="history" validation already allows
# backbone in ("mlp","vit","fno_vit","fourier_mlp"); train_stage1_
# patched.py's --mode help text says "requires --aux-backbone vit" but
# that documents what had been TRIED so far, not a real restriction --
# confirmed directly via a 2-epoch smoke run before this launch, see
# below).
#
# **F=6.0, NOT F=4.2 (every other L96 section's forcing) -- a genuine,
# necessary deviation, checked directly before writing this script.**
# Measured the TRUE (non-learned) Lorenz-96 N=16 Lyapunov spectrum at
# three candidate F via this project's own validated Benettin/QR code
# (ks_latent.analysis.lyapunov.benettin, the same machinery test_lorenz96.
# py's test_lyapunov_l96_n40_f8_matches_literature/test_l96_n256_is_
# chaotic use):
#   N=16 F=4.2: lambda1=-0.0000  n_positive=0/16  D_KY=0.0   (NOT chaotic)
#   N=16 F=6.0: lambda1=0.9801   n_positive=5/16  D_KY=9.39
#   N=16 F=8.0: lambda1=1.5197   n_positive=6/16  D_KY=10.83
# F=4.2 (this line's usual forcing, chosen originally for N=64) is a
# STABLE fixed point at N=16 -- there would be nothing chaotic to learn.
# F=6.0 chosen over F=8.0: its D_KY=9.39 is the closer match to this
# section's d_latent=8 (a latent CANNOT exceed D_KY=8 by construction, so
# the achievable target is capped at 8 regardless; F=6.0's true D_KY is
# the smaller overshoot of the two chaotic options, i.e. the fairer sizing
# for what an 8-dim latent could plausibly represent in full).
#
# Dataset generated today: ks_latent.solver.lorenz96_dataset.
# generate_trajectory_dataset(Lorenz96Config(N=16, F=6.0, dt=0.01,
# snapshot_every=10, spinup_time=200.0, seed=0), n_train=50, n_val=10,
# trajectory_time=200.0, include_derivative=True) -- every numeric
# parameter except N/F copied verbatim from the existing N=64/F=4.2
# dataset's own recorded metadata, for comparability. dt_snap=0.1, same
# as every other L96 section in this line.
#
# Encoder/decoder: --encoder mlp (Section 200's plain "Track A" MLP,
# NX=32 -> ... -> d_latent=8, mirrored decoder) -- the SIMPLEST
# architecture in this line, deliberately, so any effect seen is
# attributable to the new data representation + regularizer, not
# confounded with a ViT's own known idiosyncrasies (Sections 199/201/203
# all used --encoder vit).
#
# Stage 1 regularizers: w_var/w_spatial/w_logdet at the SAME small
# weights every prior section in this line has used (0.02/0.01 signed/
# 0.0035) -- not the object of this experiment, kept fixed for
# comparability.
#
# Stage 2: w_varmatch/w_spatial/w_logdet_rollout_latent carried forward
# UNCHANGED from Section 203's own (failed) stack, PLUS the new
# --w-spectrum-shape-self on top -- directly answers "does adding the
# self-rollout term on top of the stack that already failed finally
# prevent collapse," the exact question OPEN_QUESTIONS.md left open.
# --spectrum-shape-n-expand 3 (roughly the true system's own 5/16~31%
# positive fraction, scaled to d_latent=8: 0.31*8~2.5, rounded to 3)/
# --spectrum-shape-expand-target 1.1/--spectrum-shape-contract-floor 0.6/
# --spectrum-shape-two-sided all copied from Section 203's own Stage-1
# spectrum-shape targets (idea 1) for consistency across sections.
#
# Verified via a real (--epochs 2) dry run of BOTH stages before this
# launch (mlp+history Stage 1: recon 0.546->0.424 over 2 epochs, no
# error; Stage 2 with --w-spectrum-shape-self on: 2 epochs, no error) --
# confirms --aux-backbone mlp + --mode history trains cleanly (despite
# the CLI help text's "requires vit") and confirms
# _propagator_spectrum_shape_self_pool's flatten-for-jacrev wrapper
# around step_history is numerically sound end to end.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section204_lorenz96_n16_xxprime_selfrollout_spectrumshape
DATASET=artifacts/datasets/lorenz96_trajectories_n16_f6.0_xxprime.h5
N_HISTORY=6

echo "=== [1/3] Stage 1: plain MLP encoder/decoder (NX=32, x+x'), d_latent=8, mode=history (n_history=6) MLP-backbone propagator, w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=16 F=6.0 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 32 --d-latent 8 \
  --encoder mlp --aux-backbone mlp --mode history --n-history $N_HISTORY \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.01 --spatial-signed --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp \
  --epochs 40 --checkpoint-every 4 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt

_diagnose() {
  local PROP_PATH=$1
  local LABEL=$2
  mamba run -n da_env python -c "
import h5py, torch
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.analysis.lyapunov import lyapunov_spectrum_latent_propagator

ae, ae_cfg, _ = load_autoencoder_checkpoint('$AE')
prop, prop_cfg, _ = load_propagator_checkpoint('$PROP_PATH')
ae.eval(); prop.eval()
d_latent = prop.cfg.d_latent
n_history = prop.cfg.n_history
print('[$LABEL] d_latent:', d_latent, ' n_history:', n_history, ' mode:', prop.mode)

DATASET = '$DATASET'
with h5py.File(DATASET,'r') as f:
    traj20 = torch.tensor(f['trajectories'][:20], dtype=torch.float32)
n, T, N = traj20.shape
with torch.no_grad():
    z_all = ae.encode(traj20.reshape(n*T, N)).reshape(n, T, -1)
z_hist0 = z_all[:, :n_history, :]  # (20, n_history, d), oldest to newest

with torch.no_grad():
    traj_roll = prop.rollout_history(z_hist0, k=2000)
finite = torch.isfinite(traj_roll).all(dim=(0,2))
first_bad = int((~finite).float().argmax().item()) if (~finite).any() else -1
print('[$LABEL] multi-IC (20) rollout first non-finite step:', first_bad)
maxz_per_t = traj_roll.abs().amax(dim=(0,2))
for t in [0,50,100,300,600,1000,1500,1999]:
    if t < traj_roll.shape[1]:
        print('[$LABEL] t=%d  max|z| across all 20 ICs: %.4f' % (t, maxz_per_t[t].item()))
final_states = traj_roll[:,-1,:]
pairwise = torch.cdist(final_states, final_states)
print('[$LABEL] final-state pairwise distance (min excl. diag, max, mean):',
      (pairwise + torch.eye(20)*1e9).min().item(), pairwise.max().item(), pairwise.mean().item())

with h5py.File(DATASET,'r') as f:
    traj = torch.tensor(f['trajectories'][3], dtype=torch.float32)
with torch.no_grad():
    z_hist_single = ae.encode(traj[:n_history]).numpy()  # (n_history, d), oldest to newest
res = lyapunov_spectrum_latent_propagator(
    prop, z_hist_single, mode='history', n_directions=d_latent, n_steps=2000, qr_every=10,
    dt_snap=0.1, warmup_steps=200, seed=0, max_abs_state=1e3,
)
print('[$LABEL] lambda1:', res.exponents[0], ' n_positive:', res.n_positive, '/', res.n_directions)
try:
    print('[$LABEL] D_KY:', res.kaplan_yorke_dimension)
except Exception as e:
    print('[$LABEL] D_KY: could not bracket --', e)
print('[$LABEL] (true L96 N=16/F=6.0 reference (in x alone, 16-dim): lambda1=0.980, n_positive=5/16, D_KY=9.39 -- NOTE: d_latent=8 caps the achievable D_KY at 8)')
"
}

echo "=== [2/3] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== [3/3] Stage 2: warm-started from Stage 1, w_varmatch + w_spatial + w_logdet_rollout_latent (Section 203's own failed stack) PLUS w_spectrum_shape_self (the new self-rollout term, this section's actual test), --amp, k_max=12, 60 epochs ==="
STAGE2_TAG="${TAG}_warmstart_k12_60ep"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --dataset "$DATASET" \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --w-varmatch 0.02 --w-varmatch-adaptive \
  --w-spatial 0.01 --spatial-signed \
  --w-logdet-rollout-latent 0.0035 \
  --w-spectrum-shape-self 0.4 --spectrum-shape-self-rollout-k 20 --spectrum-shape-self-n-samples 16 \
  --spectrum-shape-n-expand 3 --spectrum-shape-expand-target 1.1 --spectrum-shape-contract-floor 0.6 --spectrum-shape-two-sided \
  --amp \
  --epochs 60 --k-max 12 --k-warmup-epochs 42 --k-mid 8 --k-mid-epochs 35 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

STAGE2_PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== Stage 2 diagnostics ==="
_diagnose "$STAGE2_PROP" "stage2" > artifacts/logs/lyapunov_${STAGE2_TAG}.log 2>&1
cat artifacts/logs/lyapunov_${STAGE2_TAG}.log

echo "=== Section 204 complete ==="
