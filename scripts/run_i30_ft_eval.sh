#!/bin/bash
# Evaluate the noise-fine-tuned 12-channel propagator (p7c12n15_ks2d) once it exists.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/i30_ft_eval.log
PR=artifacts/stage2_prop_kolmogorov_crop_p7c12n15_ks2d.pt
until [ -f $PR ] && ! pgrep -f "train_kolmogorov_crop.py.*p7c12n15_ks2d" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') fine-tune done; evaluating" >> $LOG
run() { tag=$1; shift
  OMP_NUM_THREADS=6 $PY scripts/exp_h9_tiled_da.py --tag p7c12n15_ks2d --n-tilings 4 --out $tag "$@" > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(grep "^$tag " artifacts/logs/h9_${tag}.log)" >> $LOG; }
run L44_C12N15_st8_rtps0.7 --obs-stride 8 --rtps 0.7
run L44_C12N15_st16_rtps0.5 --obs-stride 16 --n-cycles 300 --rtps 0.5
run L44_C12N15_st16_base --obs-stride 16 --n-cycles 300
OMP_NUM_THREADS=6 $PY scripts/exp_i30_propagator_degradation.py --tag p7c12n15_ks2d --n-cycles 300 --every 10 \
  --out c12n15_st16_rtps0.5 > artifacts/logs/i30_degr_c12n15.log 2>&1
echo "$(date '+%F %T') done degradation $(tail -1 artifacts/logs/i30_degr_c12n15.log)" >> $LOG
OMP_NUM_THREADS=6 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_p7c12n15_ks2d.pt \
  --prop-ckpt $PR --scale 1.0 --seed 0 --n-directions 384 --tag p7c12n15_ks2d_x1.0_seed0 > artifacts/logs/crop_eval_p7c12n15_ks2d.log 2>&1
echo "$(date '+%F %T') done lyap $(grep -E 'rel_err_pct|model_D_KY' artifacts/logs/crop_eval_p7c12n15_ks2d.log | tr -d '\n ')" >> $LOG
echo "$(date '+%F %T') I30 FT EVAL DONE" >> $LOG
