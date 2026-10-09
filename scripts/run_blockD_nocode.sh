#!/bin/bash
# Block D runs that need no code changes (docs/full_paper_tasks.md), 2026-10-08.
# Same settings as scripts/run_da2d_rerun.sh: p7b_ks2d, 100 cycles x 5 steps, obs noise 0.1,
# GC c=3 sites unless unlocalized, obs stride 8 grid points (2.75 units) unless stated.
# Usage: run_blockD_nocode.sh small|large|d4   (the large lane runs one domain at a time)
# Lane d4 (added after the D4 code, letkf_local2d_etkf, 2026-10-08) runs the deterministic per-site filter.
# Lane d6 (after the D6 code, --phys-rmse / --save-fields): 44^2 and 88^2 now; 220^2 and 330^2 only after the
# large lane has logged "LANE DONE" (one large domain at a time).
cd "$(dirname "$0")/.."
OUT=artifacts/experiments/da2d; LOG=artifacts/logs/blockD_nocode.log
mkdir -p $OUT artifacts/logs
AE=artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt; PR=artifacts/stage2_prop_kolmogorov_crop_p7b_ks2d.pt
log () { echo "[blockD-$LANE $(date '+%H:%M:%S')] $*" >> $LOG; }
run () {  # tag, then extra args (an --obs-stride given here overrides the default 8)
  tag=$1; shift
  if [ -f "$OUT/da2d_${tag}.json" ]; then log "skip (done): $tag"; return; fi
  log "running: $tag $*"
  /usr/bin/time -l mamba run -n da_env python scripts/run_da_pff_2d_ltransfer.py \
    --ae-checkpoint $AE --prop-checkpoint $PR --n-cycles 100 --obs-stride 8 "$@" --tag "$tag" \
    > "$OUT/da2d_${tag}.stdout" 2> "$OUT/da2d_${tag}.time" \
    && log "done: $tag" || log "FAILED: $tag"
}
LANE=$1
if [ "$LANE" = small ]; then
  # D2 (2D half): obs spacing 1.375 / 5.5 / 11 units at 44^2 (2.75 = A_L44_local_N64_s0).
  for st in 4 16 32; do run "D2_L44_local_N64_st${st}_s0" --L 44 --n-ensemble 64 --method letkf_local2d --obs-stride $st --seed 0; done
  # D9: model noise 0.04 at 88^2.
  run "D9_L88_local_N64_noise0.04_s0" --L 88 --n-ensemble 64 --method letkf_local2d --model-noise-std 0.04 --seed 0
  # D5: inflation sweep at 44^2, N=64 (1.0 = Block A). Local = perturbed-obs per-site filter (D4 not built yet).
  for inf in 1.05 1.1 1.2; do
    run "D5_L44_denseNone_N64_inf${inf}_s0" --L 44 --n-ensemble 64 --method letkf --localizer none --letkf-inflation $inf --seed 0
    run "D5_L44_denseGC_N64_inf${inf}_s0" --L 44 --n-ensemble 64 --method letkf --letkf-inflation $inf --seed 0
    run "D5_L44_local_N64_inf${inf}_s0" --L 44 --n-ensemble 64 --method letkf_local2d --letkf-inflation $inf --seed 0
  done
elif [ "$LANE" = large ]; then
  # D9: model noise 0.04 at 220^2.
  run "D9_L220_local_N64_noise0.04_s0" --L 220 --n-ensemble 64 --method letkf_local2d --model-noise-std 0.04 --seed 0
  # D8: ensemble size at 220^2 (N=64 = C_L220_local_N64_s0).
  for N in 8 16 32 128; do run "D8_L220_local_N${N}_s0" --L 220 --n-ensemble $N --method letkf_local2d --seed 0; done
  # D7: second seed for the headline large-domain points (dense-none s1 at 44/88 exists in Block A).
  run "D7_L220_local_N64_s1" --L 220 --n-ensemble 64 --method letkf_local2d --seed 1
  run "D7_L330_local_N64_s1" --L 330 --n-ensemble 64 --method letkf_local2d --seed 1
elif [ "$LANE" = d4 ]; then
  for N in 8 16 32 64 128; do run "D4_L44_etkf_N${N}_s0" --L 44 --n-ensemble $N --method letkf_local2d_etkf --seed 0; done
  run "D4_L88_etkf_N64_s0" --L 88 --n-ensemble 64 --method letkf_local2d_etkf --seed 0
elif [ "$LANE" = d6 ]; then
  for L in 44 88; do run "D6_L${L}_local_N64_s0" --L $L --n-ensemble 64 --method letkf_local2d --phys-rmse --save-fields 10 --seed 0; done
  until grep -q "blockD-large .*LANE DONE" $LOG; do sleep 120; done
  for L in 220 330; do run "D6_L${L}_local_N64_s0" --L $L --n-ensemble 64 --method letkf_local2d --phys-rmse --save-fields 10 --seed 0; done
else
  echo "usage: $0 small|large|d4|d6"; exit 1
fi
log "LANE DONE"
