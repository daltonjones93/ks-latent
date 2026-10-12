#!/bin/bash
# R4 (user 2026-10-12): data for the reach test, generated now so it is ready if more 44^2 data does not close D's
# gap to the Paper II model (0.395-0.408 vs 0.324 at 44^2). Reach 5 needs an 11-site window (66^2 = 12 sites);
# reach 6 needs 13 sites without wrapping (77^2 = 14 sites; 66^2 would put the antipodal site in the window twice).
# Each set holds about the area of the 144 x 44^2 set: 64 x 66^2 and 48 x 77^2 training trajectories (+6 val),
# 25 time units each, seeds disjoint from the other sets. Starts after the 44^2 set.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python; L=artifacts/logs
log() { echo "$(date '+%F %T') [R4-gen] $*" >> $L/r4.log; }
until [ -f artifacts/datasets/ks2d_box44_large.h5 ] && ! pgrep -f "gen_ks2d_box_large.py --n-train 144" > /dev/null; do sleep 60; done
for spec in "66 64 2000" "77 48 3000"; do set -- $spec
  out=artifacts/datasets/ks2d_box${1}_large.h5; [ -f $out ] && continue
  $PY scripts/gen_ks2d_box_large.py --L $1 --n-train $2 --n-val 6 --seed-base $3 --workers 8 --out $out >> $L/gen_box${1}_large.log 2>&1
  log "data $out rc=$? $(tail -1 $L/gen_box${1}_large.log)"; done
log "reach-test data done"
