#!/bin/bash
# KSEED (2026-10-13, referee M3): training seeds 1-2 for Paper I's three window-trained Kolmogorov models,
# evaluated like their seed-0 runs (exp_crop_eval.py, scales 1/2/3 x IC seeds 0-2; alpha 0.025 3x: ICs 0-1).
#   k_xs01_v2   (alpha 0.1)   kolmogorov_wide16_alpha0.1.h5 (unchanged since seed 0), k_max 16, w_logdet 0.005
#   c19_e0d0_c6 (alpha 0.05)  seed 1 exists (c19_e0d0_c6_s1, IC 0 evals only): train seed 2, add ICs 1-2 for s1
#   k_xs0025    (alpha 0.025) k_max 32, 9000 steps
# Data: the alpha 0.05 / 0.025 wide datasets were overwritten on 2026-10-06 with 8+2 trajectories; seed 0 (and
# c19 s1) trained on the original 4+1. The originals are rebuilt bit-for-bit from the saved parts traj0-4 by
# gen_kolmogorov_wide_dataset.py (--n-train 4 --n-val 1 --seed-base 500) as *_orig5.h5; the driver checks the
# normalization std equals the seed-0 checkpoint's.
# Code runs from a detached worktree (latent_DA_kseeds, commit 4a0b165) with PYTHONPATH set to it, so concurrent
# edits in the main tree cannot change these runs; artifacts/ is a symlink to the main tree's.
# After each training, the saved configs are diffed against seed 0's (only seed/dataset path may differ).
# Trainings one at a time; evals at most 2 at a time; 3 threads each; normal priority; no caffeinate.
# Waits for the reach test and box checks to finish first. Log: artifacts/logs/r4.log, tag [KSEED].
WT=/Users/daltonjones/Documents/latent_DA_kseeds
cd $WT || exit 1
export PYTHONPATH=$WT OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
A=artifacts; E=$A/experiments; L=$A/logs
log () { echo "$(date '+%F %T') [KSEED] $*" >> $L/r4.log; }
MIX="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6"

log "start; waiting for reach test / box checks / orig5 rebuilds"
while pgrep -f exp_reach_velocity > /dev/null || pgrep -f exp_box_check > /dev/null \
      || pgrep -f "_orig5.h5" > /dev/null; do sleep 60; done
for a in 0.05 0.025; do
  [ -f $A/datasets/kolmogorov_wide16_alpha${a}_orig5.h5 ] || { log "missing orig5 dataset alpha $a; abort"; exit 1; }
done
log "cores free; starting"

verify () {  # new tag vs seed-0 tag: every saved config field equal except seed / dataset path
  $PY - "$1" "$2" <<'EOF' 2>&1 | grep -v -i warn
import sys, torch
new, ref = sys.argv[1], sys.argv[2]
bad = []
for st, keys in (("stage1_ae", ("ae_config", "crop_ae_config", "normalization_mean", "normalization_std")),
                 ("stage2_prop", ("prop_config", "crop_prop_config"))):
    a = torch.load(f"artifacts/{st}_kolmogorov_crop_{new}.pt", map_location="cpu", weights_only=False)
    b = torch.load(f"artifacts/{st}_kolmogorov_crop_{ref}.pt", map_location="cpu", weights_only=False)
    for k in keys:
        x, y = a[k], b[k]
        x, y = (getattr(x, "__dict__", x), getattr(y, "__dict__", y))
        if isinstance(x, dict):
            for f in sorted(set(x) | set(y)):
                if f == "seed":
                    continue
                if f not in y:  # field added to the code after seed 0 was trained: report, not fatal
                    print(f"note {st}.{k}.{f}: new field = {x[f]!r}")
                elif x.get(f) != y.get(f):
                    bad.append(f"{st}.{k}.{f}: {x.get(f)!r} vs {y.get(f)!r}")
        elif x != y:
            bad.append(f"{st}.{k}: {x!r} vs {y!r}")
print("CONFIG OK" if not bad else "CONFIG MISMATCH " + "; ".join(bad))
EOF
}

train () { tag=$1; ref=$2; shift 2
  if [ -f $A/stage2_prop_kolmogorov_crop_$tag.pt ]; then log "skip train $tag (done)"; return; fi
  log "train $tag start"; t0=$(date +%s)
  $PY scripts/train_kolmogorov_crop.py --tag $tag "$@" > $L/crop_train_$tag.log 2>&1
  log "train $tag rc=$? [$(( $(date +%s) - t0 ))s] $(grep -E 'cropProp\] step' $L/crop_train_$tag.log | tail -1)"
  log "verify $tag vs $ref: $(verify $tag $ref | tr '\n' ' ')"
}

# ---- training lane ----
trains () {
  train k_xs01_v2_s1 k_xs01_v2 --dataset $A/datasets/kolmogorov_wide16_alpha0.1.h5 --seed 1 $MIX \
    --k-max 16 --k-mid 8 --noise-end 0.02 --prop-steps 9000 --w-logdet 0.005
  train k_xs01_v2_s2 k_xs01_v2 --dataset $A/datasets/kolmogorov_wide16_alpha0.1.h5 --seed 2 $MIX \
    --k-max 16 --k-mid 8 --noise-end 0.02 --prop-steps 9000 --w-logdet 0.005
  train c19_e0d0_c6_s2 c19_e0d0_c6 --dataset $A/datasets/kolmogorov_wide16_alpha0.05_orig5.h5 --seed 2 $MIX
  train k_xs0025_s1 k_xs0025 --dataset $A/datasets/kolmogorov_wide16_alpha0.025_orig5.h5 --seed 1 $MIX \
    --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000
  train k_xs0025_s2 k_xs0025 --dataset $A/datasets/kolmogorov_wide16_alpha0.025_orig5.h5 --seed 2 $MIX \
    --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000
  log "trainings done"
}

# ---- eval lane: at most 2 concurrent, in order (2x and 3x before 1x) ----
EVALS=""
for t in k_xs01_v2_s1 k_xs01_v2_s2; do for sc in 2 3 1; do for s in 0 1 2; do EVALS="$EVALS $t:$sc:$s"; done; done; done
for sc in 2 3 1; do for s in 1 2; do EVALS="$EVALS c19_e0d0_c6_s1:$sc:$s"; done; done
for sc in 2 3 1; do for s in 0 1 2; do EVALS="$EVALS c19_e0d0_c6_s2:$sc:$s"; done; done
for t in k_xs0025_s1 k_xs0025_s2; do
  for sc in 2 3 1; do for s in 0 1 2; do [ $sc = 3 ] && [ $s = 2 ] && continue; EVALS="$EVALS $t:$sc:$s"; done; done
done
evals () {
  for item in $EVALS; do
    IFS=: read tag sc s <<< "$item"
    out=$E/crop/${tag}_x${sc}_seed$s.json
    [ -f $out ] && continue
    while [ ! -f $A/stage2_prop_kolmogorov_crop_$tag.pt ] || pgrep -f "train_kolmogorov_crop.py --tag $tag " > /dev/null; do sleep 60; done
    while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 20; done
    ( t0=$(date +%s)
      $PY scripts/exp_crop_eval.py --ae-ckpt $A/stage1_ae_kolmogorov_crop_$tag.pt \
        --prop-ckpt $A/stage2_prop_kolmogorov_crop_$tag.pt --scale $sc --seed $s --tag ${tag}_x${sc}_seed$s \
        > $L/crop_eval_kseed_${tag}_x${sc}_seed$s.log 2>&1
      rc=$?
      d=$($PY -c "import json; print(round(json.load(open('$out'))['model_D_KY'], 2))" 2>/dev/null)
      log "eval ${tag} x$sc seed$s rc=$rc D_KY $d [$(( $(date +%s) - t0 ))s]" ) &
  done
  wait
  log "evals done"
}

trains &
evals &
wait
log "KSEED DONE"
