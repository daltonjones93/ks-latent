#!/bin/bash
# Paper II 1D per-patch model, option A (2026-10-11, user-approved "queue A"): the Paper I 1D recipe
# (Section 224: whole L = 100 periodic boxes, joint Stage 1 with a full-size auxiliary propagator,
# Stage 2 warm start, k 2 -> 8 -> 12 over 300 epochs) with three changes, to match the 2D DA model:
# encoder and decoder reach 0 (per-patch, so Paper II Prop. 1 holds in 1D), c = 6 channels, no
# spatial-coherence penalty, and training data carrying one fixed realization of Gaussian noise of
# std 0.1 (standardized units). Then: D_KY transfer at L = 100, 200, 400 from three initial
# conditions (scripts/eval_1d_transfer.py), and the full 1D DA set of Paper II Sec. 9
# (scripts/run_blockD_1d.sh with TAGP=pp1d_c6d10_). Starts once the J17G trainings are done (it runs
# beside the KDA training; the 1D model is small). Option B (window-trained) is
# scripts/train_ks1d_crop.py, not queued.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; TAG=pp1d_c6d10; S2=${TAG}_warmstart_k12_300ep
N=$A/datasets/stage1_trajectories_dtsnap1_noise0.1.h5; LOG=$A/logs/pp1d.log
AE=$A/stage1_ae_patched_full_$TAG.pt; AUX=$A/stage1_prop_full_$TAG.pt; PR=$A/stage2_prop_patched_full_$S2.pt
log() { echo "$(date '+%F %T') $*" >> $LOG; }
until grep -q "J17G_d0.3\] training done" $A/logs/j17g.log 2>/dev/null; do sleep 300; done
log "start"
[ -f $N ] || $PY scripts/make_noisy_dataset.py --src $A/datasets/stage1_trajectories_dtsnap1.h5 --noise 0.1 --out $N >> $LOG 2>&1
[ -f $AE ] || $PY scripts/train_stage1_patched.py --profile full --dataset $N --dt-snap 1.0 \
  --encoder local_field --local-field-n-sites 16 --local-field-channels 6 --local-field-n-mix-layers 0 \
  --aux-backbone local_mlp --mode markovian --aux-n-tokens 16 --attn-window 3 \
  --w-decorr 0 --w-var 0.02 --w-spatial 0 --w-var-floor 0 --w-logdet 0.0035 \
  --full-propagator --amp --epochs 200 --checkpoint-every 20 --tag $TAG > $A/logs/stage1_$TAG.log 2>&1
log "stage 1 rc=$? $(grep val_recon_final $A/logs/stage1_$TAG.log | tail -1)"
[ -f $PR ] || $PY scripts/train_stage2_patched.py --ae-checkpoint $AE --init-prop-checkpoint $AUX --dataset $N \
  --amp --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 --tag $S2 \
  > $A/logs/stage2_$S2.log 2>&1
log "stage 2 rc=$? $(grep best_val_kmax_mse $A/logs/stage2_$S2.log | tail -1)"
for s in 0 1 2; do
  $PY scripts/eval_1d_transfer.py --tag $TAG --L 100 200 400 --ic-seed $s --ic normalized >> $A/logs/eval1d_$TAG.log 2>&1
  log "D_KY seed $s rc=$?"
done
grep -h "D_KY" $A/logs/eval1d_$TAG.log | tail -9 >> $LOG
export AE PR TAGP=${TAG}_
for j in d1_pff_200 d1_pff_400 d1_letkf_200 d1_letkf_400 d2 d3; do bash scripts/run_blockD_1d.sh $j & done; wait
log "PP1D A DONE"
