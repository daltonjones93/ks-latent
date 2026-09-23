#!/bin/zsh
# User-directed 2026-09-08, "Section 107" (REVISED -- the original
# vit_fourier_hybrid/L=40 launch, scripts/section107_L40_pdehead_mutual.sh,
# was killed mid-Phase-1 and superseded by this one). Builds on the full
# pde_head hybrid arc from this session (docs/sine_transform_pde_plan.md
# §19-23).
#
# User's revision, verbatim: "I told you to choose parameters that made
# sense for this setting. the pde coupling should do the work we were
# hoping that w_spatial would. we can set w_var = .01, w_spatial = .001,
# w_logdet = .008. please kill the training, and restart with L = 22. You
# can make d_latent 16." then, immediately after: "one thing to add, do
# everything I said above, and use the model, propagator from section 98."
#
# CLARIFIED (asked directly, given a real conflict: Section 98's actual
# checkpoint is FIXED at d_latent=44, plain encoder="vit" (NOT
# vit_fourier_hybrid), trained at L=100 -- none of which match "d_latent
# 16"/"L=22"): user chose "train fresh at d_latent=16, L=22, but copy
# Section 98's ARCHITECTURE FAMILY (plain vit + mlp/markovian) instead of
# vit_fourier_hybrid" -- i.e. reuse Section 98's own recipe/hyperparameter
# CHOICES, not its trained weights. So this is a FRESH-init run (matching
# this whole arc's "train from scratch" convention), on a DIFFERENT
# encoder family than the original section107 draft (plain "vit", not
# "vit_fourier_hybrid" -- see scripts/section98_no_fourier_branch.sh for
# the exact recipe this copies: --encoder vit --d-model 56 --pos-encoding
# linear --attn-window 4 --token-window 16 --pool token_mlp --dec-pool
# token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128, aux_backbone
# mlp/markovian, --full-propagator).
#
# ============================================================
# Domain: L=22 (this repo's own established Gate 1 benchmark domain,
# D_KY~5.2-5.6 -- tests/replication/test_gate1_kaplan_yorke.py::
# test_L22_lyapunov). Reuses EXISTING datasets (already generated earlier
# this session for Sections 103-106, no regeneration needed):
# artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5,
# artifacts/datasets/attractor_points_L22.h5.
#
# d_latent=16 (down from Section 98's own 44) -- user-directed.
#
# ============================================================
# Regularizers -- user explicitly corrected the first draft's "just copy
# Section 95's/98's own values verbatim" approach: "I told you to choose
# parameters that made sense for this setting. the pde coupling should do
# the work we were hoping that w_spatial would." I.e. w_spatial's original
# job (encourage banded/local structure in z, hopefully making it more
# PDE-like) is now REDUNDANT with pde_head+w_pde_distill, which does this
# MORE DIRECTLY and explicitly -- so w_spatial should be cut to near-
# vestigial rather than kept at its old tuned value.
#   w_var:     0.02 (Section 98) -> 0.01 (user-directed)
#   w_spatial: 0.04 (Section 98) -> 0.001 (user-directed, ~40x smaller --
#              near-off, pde_distill now carries this job)
#   w_logdet:  0.008 (user-directed -- happens to match Section 98's own
#              value already, unchanged)
#   w_smooth:  0.003 -- NOT mentioned by user, kept at Section 98's own
#              value (the relevant precedent now that we've switched to
#              its architecture family) rather than Section 95's 0.0015.
#   w_decorr/w_var_floor: 0 (off, matching both 95 and 98).
#   spatial_signed: kept True (--spatial-signed) -- only the WEIGHT was
#              asked to change, not the sign convention.
#
# pde_head: backbone=spectral_pde_raw, K=9 (=d_latent//2+1, no truncation
# at this new d_latent=16), L_pde=16 (=d_latent, pde_head's OWN self-FFT
# ring length for the latent index -- unrelated to the physical L=22
# above), integrator=euler, hidden=128/n_blocks=3 (matches aux capacity).
#
# w_pde_distill RECALIBRATED for this new architecture (real L=22 data,
# plain vit/d_model=56/d_latent=16, computed 2026-09-08): raw
# l_pde_distill (mutual)=0.381, weighted l_pred contribution
# (w_pred=0.5)=0.639, ratio ~60%.
#
# ============================================================
# REVISION 2 (2026-09-08, after watching Revision 1's actual trajectory --
# w_pde_distill=0.8, pde_head continuing into Phase 2 -- run to completion
# through Phase 1 and partway into Phase 2 before being killed and
# redesigned here): Phase 1 alone (200 epochs) showed NO collapse
# signature through completion (recon fell smoothly to 0.000128, loss to
# 0.000079, pde_distill tracking down to ~0.00008 alongside it -- not
# stuck, not exploding). User's read, agreed: "so what I'm seeing is we
# should increase the pde weight during stage 1, turn off the pde weight
# during stage 2. then see if we can still model a reasonable pde during
# stage 3" -- Stage 1 is the only phase where the encoder is trainable
# (the right place to push hardest for genuine PDE-friendliness, and it's
# tolerating pressure well); Stage 2 (extending rollout 2->12) is the
# highest collapse-risk window (longer horizon = more room for "stay near
# identity" to look cheap), so remove ALL pde pressure there and let it
# purely chase real long-horizon accuracy on whatever foundation Stage 1
# built; Stage 3 then tests whether that Stage-1-only foundation is
# enough for pde_head to still fit the FULLY-trained, long-horizon
# propagator well, in isolation.
#
#   w_pde_distill: 0.8 (Rev 1) -> 2.0 (Rev 2, Phase 1 only, --pde-mutual
#     still active -- user-directed, given Rev 1 showed no instability).
#   Phase 2: --init-pdehead-checkpoint/--w-pde-distill DROPPED entirely
#     (user-directed) -- Phase 2 is now IDENTICAL to Sections 95/98's own
#     untouched Phase 2 recipe (pure rollout-extension, no pde_head
#     involvement at all). pde_head is simply left as Phase 1 produced it
#     (--full-propagator's own AUX continues into Phase 2 as always; only
#     the pde_head SIDE-CHANNEL is removed).
#   Phase 3: refines Phase 1's OWN pde_head checkpoint (Phase 2 never
#     touches it) against Phase 2's now-FROZEN, long-horizon-capable
#     propagator -- both options at weight 1.0 (unchanged from Rev 1, no
#     more collapse risk to protect against once propagator is frozen).
#
# Phase 2/3 schedule otherwise IDENTICAL to Sections 95/98's own Phase 2
# (--epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8
# --k-mid-epochs 175); Phase 3 100 epochs (docs/sine_transform_pde_plan.md
# §23).
#
# Verified via a real smoke run before launching Rev 1 (2026-09-08): fresh
# vit/d_latent=16 + mlp aux + spectral_pde_raw pde_head (K=9/L=16) trains
# cleanly on real L=22 data, no NaN, pde_distill logged correctly. Rev 2's
# only changes (weight value, dropping two CLI flags from Phase 2, Phase
# 3's checkpoint source) are compositions of already-independently-tested
# pieces -- not smoke-tested again as a combination before this launch.
#
# ============================================================
# REVISION 3 (2026-09-08): Rev 2's Phase 2 result looked deceptively okay
# on error-growth alone (relative RMSE saturating just BELOW sqrt(2)) --
# but visualize_rollout.py's physical + latent Hovmoller plots showed this
# was a FALSE POSITIVE: the model's decoded rollout collapses almost
# immediately to a nearly flat, static field (latent Hovmoller: literal
# vertical stripes -- every channel snaps to a constant and stays there
# for all 200 steps), the classic H-PROP fixed-point collapse. A collapsed
# rollout trivially gives relative RMSE ~= ||u_true||/||u_true|| ~= 1,
# indistinguishable from genuine chaotic decorrelation on that plot alone
# -- this is why the Hovmoller check matters, not just the error-growth
# number. User: "well crap. kill the phase 3 training. we need to rerun
# this with a smaller pde term I suppose. please just replicate exactly
# the section 98 training, same regularizers and everything, just add
# smaller pde term (make it something like .3)."
#
# Regularizers REVERTED to Section 98's own exact, already-proven values
# (undoing Rev 1/2's "let pde_distill do w_spatial's job" experiment,
# which is the likely proximate cause -- w_spatial cut ~40x removed a
# real anti-collapse safety net Section 98 itself relied on):
#   w_var:     0.01 (Rev 1/2) -> 0.02 (Section 98's own value, restored)
#   w_spatial: 0.001 (Rev 1/2) -> 0.04 (Section 98's own value, restored)
#   w_logdet:  0.008 (unchanged -- already matched Section 98's own value)
#   w_smooth:  0.003 (unchanged -- already Section 98's own value)
# d_latent=16 and L=22 are KEPT (a separate, already-settled choice for
# this investigation, not something the "replicate Section 98" request
# was read as overriding -- "same regularizers" specifically, not "same
# d_latent"). w_pde_distill: 2.0 (Rev 2) -> 0.3 (user-directed, back near
# this project's usual gentle-regularizer convention rather than the
# "make it do real structural work" bet Rev 1/2 tried).
#
# --pde-mutual ALSO now turned OFF (user, immediately after: "oh yeah and
# detach the pde, let's see if that helps") -- back to the ORIGINAL,
# tested-safe design (train_stage1's default `pde_distill_detach_target
# =True`): gradient reaches only pde_head and, through z, the ENCODER;
# aux stays fully protected from any pde-related pressure, exactly as
# designed before the mutual experiment. Two independent levers were
# pulled back simultaneously this revision (weight 2.0->0.3, mutual->
# detached) -- if Rev 3 does NOT collapse, that alone won't say which
# change (or both) fixed it; a follow-up ablation would be needed to
# isolate that if it matters later.
set -e
cd /Users/daltonjones/Documents/latent_DA

DATASET=artifacts/datasets/stage1_trajectories_L22_dtsnap1.h5
POINTS_DATASET=artifacts/datasets/attractor_points_L22.h5
TAG=section107_vit_L22_dlatent16_pdehead_mutual

echo "=== [1/4] Phase 1 (DETACHED pde_head, Rev 3): plain vit (d_model=56, d_latent=16) + mlp/markovian aux (full-propagator), Section 98's OWN regularizers restored (w_var=0.02/w_spatial=0.04/w_logdet=0.008/w_smooth=0.003), w_pde_distill=0.3 (detached target -- aux protected), L=22, --amp, 200 epochs ==="
mamba run -n da_env python scripts/train_stage1_patched.py \
  --profile full --dataset "$DATASET" --dt-snap 1.0 --d-latent 16 \
  --encoder vit --aux-backbone mlp --mode markovian \
  --d-model 56 --pos-encoding linear --attn-window 4 --token-window 16 \
  --pool token_mlp --dec-pool token_mlp --token-mlp-reduction 8 --token-mlp-hidden 128 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0.04 --spatial-signed --w-var-floor 0 --w-logdet 0.008 \
  --w-smooth 0.003 \
  --pde-distill --w-pde-distill 0.3 --pde-hidden 128 --pde-n-blocks 3 \
  --full-propagator --amp \
  --epochs 200 --checkpoint-every 20 \
  --tag "$TAG" \
  > artifacts/logs/stage1_${TAG}.log 2>&1

AE=artifacts/stage1_ae_patched_full_${TAG}.pt
AUX=artifacts/stage1_prop_full_${TAG}.pt
PDEHEAD=artifacts/stage1_pdehead_full_${TAG}.pt

echo "=== [2/4] Phase 2 (extend rollout ONLY -- pde_head DROPPED entirely, per user direction after watching Phase 1's trajectory: isolate the highest collapse-risk window (longer rollout) from any pde pressure, matching Sections 95/98's own untouched Phase 2 recipe exactly): k_max=12, 300 epochs, --amp ==="
STAGE2_TAG="${TAG}_extended"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$AUX" \
  --amp \
  --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
  --tag "$STAGE2_TAG" \
  > artifacts/logs/stage2_${STAGE2_TAG}.log 2>&1

PROP=artifacts/stage2_prop_patched_full_${STAGE2_TAG}.pt

echo "=== [3/4] Phase 3 (docs/sine_transform_pde_plan.md §23): refine Phase 1's OWN pde_head (Phase 2 never touched it) against Phase 2's FROZEN, now-long-horizon-capable propagator -- the actual test of whether Phase 1's shaping alone transfers. Both options at weight 1.0 (no more collapse risk to protect). 100 epochs ==="
PHASE3_TAG="${TAG}_phase3refine"
mamba run -n da_env python scripts/train_stage2_patched.py \
  --ae-checkpoint "$AE" \
  --init-prop-checkpoint "$PROP" \
  --init-pdehead-checkpoint "$PDEHEAD" --freeze-propagator \
  --w-pde-distill 1.0 --w-pde-rollout 1.0 \
  --amp --epochs 100 --k-max 12 --k-warmup-epochs 1 \
  --tag "$PHASE3_TAG" \
  > artifacts/logs/stage2_${PHASE3_TAG}.log 2>&1

echo "=== [4/4] Gate 3/4 (on Phase 2's OWN propagator -- Phase 3 only refines pde_head, does not change this) ==="
mamba run -n da_env python scripts/run_analysis_suite.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --dt-snap 1.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_analysis_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_da_pff.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dt-snap 1.0 --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate3_da_${STAGE2_TAG}.log 2>&1 &
mamba run -n da_env python scripts/run_diagnostics.py \
  --ae-checkpoint "$AE" --prop-checkpoint "$PROP" \
  --dataset "$DATASET" --points-dataset "$POINTS_DATASET" --L 22.0 \
  --tag "$STAGE2_TAG" > artifacts/logs/gate4_diagnostics_${STAGE2_TAG}.log 2>&1 &
wait

echo "=== Section 107 complete ==="
