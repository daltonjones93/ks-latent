#!/bin/bash
# 2D KS DA rerun at the correct deployment resolution (task E11, 2026-10-07).
# Every run: p7b_ks2d, 100 cycles x 5 steps, obs stride 8 grid points (2.75 units),
# obs noise 0.1, GC c=3 sites unless unlocalized. Peak RSS recorded via /usr/bin/time -l.
cd "$(dirname "$0")/.."
OUT=artifacts/experiments/da2d; LOG=artifacts/logs/da2d_rerun.log
mkdir -p $OUT artifacts/logs
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
log () { echo "[da2d $(date '+%H:%M:%S')] $*" >> $LOG; }

run () {  # tag then extra args
  tag=$1; shift
  if [ -f "$OUT/da2d_${tag}.json" ]; then log "skip (done): $tag"; return; fi
  log "running: $tag $*"
  /usr/bin/time -l mamba run -n da_env python scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 --obs-stride 8 "$@" --tag "$tag" \
    > "$OUT/da2d_${tag}.stdout" 2> "$OUT/da2d_${tag}.time" \
    && log "done: $tag" || log "FAILED: $tag"
}

# Block A: ensemble size x filter x localization at the three sizes with known truths.
for L in 44 66 88; do
  for N in 8 16 32 64 128; do run "A_L${L}_local_N${N}_s0" --L $L --n-ensemble $N --method letkf_local2d --seed 0; done
  for N in 16 64 128; do run "A_L${L}_denseGC_N${N}_s0" --L $L --n-ensemble $N --method letkf --seed 0; done
  for N in 16 64 128; do run "A_L${L}_denseNone_N${N}_s0" --L $L --n-ensemble $N --method letkf --localizer none --seed 0; done
done
# Second seed for the headline comparisons.
for L in 44 66 88; do
  run "A_L${L}_local_N64_s1" --L $L --n-ensemble 64 --method letkf_local2d --seed 1
  run "A_L${L}_denseNone_N64_s1" --L $L --n-ensemble 64 --method letkf --localizer none --seed 1
done
# Block B: model-noise sweep at L=44 (0.08 is the default, already in Block A).
for s in 0.02 0.04 0.16; do run "B_L44_local_N64_noise${s}_s0" --L 44 --n-ensemble 64 --method letkf_local2d --model-noise-std $s --seed 0; done
# Block C: large domains (d = 9600 / 21600 / 38400), local filter; dense at 220 for comparison.
run "C_L220_local_N64_s0" --L 220 --n-ensemble 64 --method letkf_local2d --seed 0
run "C_L220_denseGC_N64_s0" --L 220 --n-ensemble 64 --method letkf --seed 0
run "C_L330_local_N64_s0" --L 330 --n-ensemble 64 --method letkf_local2d --seed 0
run "C_L440_local_N64_s0" --L 440 --n-ensemble 64 --method letkf_local2d --seed 0
log "ALL DONE"
