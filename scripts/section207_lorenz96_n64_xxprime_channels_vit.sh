#!/bin/zsh
# RESULT (2026-09-23): reconstruction fixed (val_recon_final=0.0044,
# matching Section 201's 0.0028) but propagator chaos dropped ~5.5x:
# lambda1=0.00327, n_positive=1/20, D_KY=2.19 -- vs Section 201's
# D_KY=11.94 on the EXACT same recipe (x only). max|z| across 20 ICs
# actually SHRINKS over the 2000-step rollout (2.66->4.47->...->1.93)
# rather than sustaining; final-state pairwise spread tops out at 5.07
# (vs 201's 264) -- the ensemble converges toward each other rather than
# diverging chaotically. Not a full collapse (D_KY>0), but a real,
# substantial regression. Working hypothesis (not yet confirmed): x' is
# fully determined by x (no new information) but is a rougher, harder
# physical-space rollout-reconstruction target for the aux propagator's
# L_pred term, possibly pressuring the joint Stage-1 optimization toward
# a duller/more-damped propagator that still reconstructs x' adequately.
# Stage 2 deliberately NOT run against this checkpoint -- see
# docs/RESULTS.md's "Section 207" entry and docs/OPEN_QUESTIONS.md for
# the full writeup and the proposed follow-up (scope L_pred to x only,
# not x', to isolate the hypothesis).
#
# User-directed 2026-09-23: "can you think of a way of augmenting 201
# with x' that will work with the vit's assumptions. implement this and
# run it."
#
# Diagnosis this section fixes (docs/RESULTS.md's "Sections 204/205"
# correction entry, docs/OPEN_QUESTIONS.md): concatenating x and x' into
# one flat NX-dim vector (Sections 204-206, at N=16) breaks the ViT's
# patch-tokenization/positional-encoding assumption that adjacent tokens
# are adjacent PHYSICAL LOCATIONS -- an x'-block token is not "further
# along in space" than an x-block token, it is the SAME sites, a
# different quantity, and nothing in the architecture told it that.
# Measured directly: both the vit-propagator (205) and mlp-propagator
# (206) variants showed the IDENTICAL badly-converging reconstruction
# curve, ruling out the propagator and implicating the encoder/data
# representation.
#
# THE FIX, implemented today: give x/x' to the ViT as two CHANNELS per
# PHYSICAL SITE instead of two concatenated blocks.
#  - `ks_latent.solver.lorenz96_dataset.generate_trajectory_dataset(...,
#    include_derivative=True, derivative_layout="interleaved")`: stores
#    `[x_0, x'_0, x_1, x'_1, ..., x_{N-1}, x'_{N-1}]` (site-major/
#    channel-minor) instead of the old `[x_0..x_{N-1}, x'_0..x'_{N-1}]`
#    block layout. Verified directly (not just by shape): decoded
#    x'-channel matches l96_rhs(x, F) recomputed independently, bit for
#    bit, on a smoke-scale dataset before this launch.
#  - `ViTAutoencoderConfig.n_channels=2` (new field,
#    `ks_latent/models/autoencoder_vit.py`): `patch_size`/`token_window`
#    stay in PHYSICAL SITE units; each token's raw input becomes
#    `patch_size*n_channels` values (all channels of `patch_size`
#    CONSECUTIVE SITES, interleaved) instead of `patch_size` -- so
#    `n_tokens = (NX/n_channels)/patch_size` is exactly this section's
#    N=64/patch_size=8 -> 8 tokens, THE SAME token count Section 201's
#    own pure-x run had -- CircularPositionalEncoding/
#    LinearPositionalEncoding still see "one ring position per physical
#    site," now genuinely true again. Verified via
#    `tests/unit/test_autoencoder_vit.py`'s new `test_n_channels_*` tests
#    (5 tests, including a direct check that token 0 is exactly sites
#    0..patch_size-1's both channels interleaved, not a channel block).
#  - `--n-channels 2` (new CLI flag, `train_stage1_patched.py`).
#
# Everything else copied VERBATIM from Section 201 (the run that
# recovered genuine chaos, D_KY=11.94) for direct comparability: L96
# N=64, F=4.2 (this line's usual, established-chaotic forcing -- unlike
# the N=16 line, F=4.2 does NOT need replacing here), d_latent=20,
# --encoder vit --aux-backbone vit --mode history --n-history 6,
# pos_encoding=linear, FULL attention (no --attn-window), same Stage-1
# regularizers (w_var=0.02/w_spatial=0.01 signed/w_logdet=0.0035), same
# 40-epoch schedule. Only --nx (128, not 64, since NX counts raw scalars
# = N*n_channels) and the new --n-channels 2 differ from Section 201's
# own invocation.
#
# Verified via a real (--epochs 2) dry run before this launch: recon
# 0.163 -> 0.0265 over just 2 epochs -- already close to Section 201's
# FULLY CONVERGED final value (0.0028), and starting from a dramatically
# better place than Sections 205/206's 2-epoch dry runs (~0.28-0.29,
# final ~0.08-0.09 after 40 epochs) -- strong direct evidence the
# channel-based fix works before committing to the full 40-epoch run.
#
# Stage 1 ONLY this launch (matching this arc's own established "check
# chaos survives before running Stage 2" discipline, e.g. Section 194) --
# if this recovers D_KY comparable to Section 201's 11.94, Stage 2 (the
# actual --w-spectrum-shape-self test motivating this whole sub-thread)
# is the natural next step.
set -e
cd /Users/daltonjones/Documents/latent_DA

TAG=section207_lorenz96_n64_xxprime_channels_vit
DATASET=artifacts/datasets/lorenz96_trajectories_n64_f4.2_xxprime_interleaved.h5
N_HISTORY=6

echo "=== [1/2] Stage 1: ViT encoder/decoder with n_channels=2 (x/x' interleaved per site, NX=128), pos_encoding=linear, FULL attention -- no --attn-window, d_latent=20, mode=history (n_history=6) ViT-backbone propagator, w_var=0.02, w_spatial=0.01 signed, w_logdet=0.0035, on Lorenz-96 N=64 F=4.2 dt_snap=0.1, --amp, 40 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --nx 128 --n-channels 2 --d-latent 20 \
  --encoder vit --aux-backbone vit --mode history --n-history $N_HISTORY \
  --pos-encoding linear \
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
print('[$LABEL] (true L96 N=64/F=4.2 reference: lambda1=0.080, n_positive=5/64, D_KY=11.3)')
print('[$LABEL] (Section 201 comparison, SAME recipe on raw x only (no x augmentation): stage1 D_KY=11.94)')
"
}

echo "=== [2/2] Stage 1 diagnostics ==="
_diagnose "$AUX" "stage1" > artifacts/logs/lyapunov_${TAG}_stage1.log 2>&1
cat artifacts/logs/lyapunov_${TAG}_stage1.log

echo "=== Section 207 Stage 1 complete (Stage 2 deferred pending this result) ==="
