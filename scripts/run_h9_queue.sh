#!/bin/bash
# H9/Block I queue: shifted-tiling latent filter. One run at a time; each line: out-tag then flags.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/h9_queue.log
run() { tag=$1; shift
  [ -f artifacts/experiments/da2d/h9_${tag}.json ] && { echo "skip $tag" >> $LOG; return; }
  echo "$(date '+%F %T') start $tag $*" >> $LOG
  OMP_NUM_THREADS=6 /usr/bin/time -l $PY scripts/exp_h9_tiled_da.py --out $tag "$@" \
    > artifacts/logs/h9_${tag}.log 2>&1
  echo "$(date '+%F %T') done $tag rc=$? $(tail -c 400 artifacts/logs/h9_${tag}.log | grep "^$tag" )" >> $LOG; }
case "$1" in
  l220)
    run L220_T4 --L 220 --n-tilings 4
    run L220_T1 --L 220 --n-tilings 1 ;;
esac
echo "$(date '+%F %T') QUEUE $1 DONE" >> $LOG
