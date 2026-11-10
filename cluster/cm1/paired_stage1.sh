#!/bin/bash
# Paired runs, stage 1: re-run source members to t = 7260 s writing a restart at 7200 s (rstfrq), with the bump-patched build.
#   sbatch -p coe_cpu --cpus-per-task=4 --mem=8G -t 03:00:00 -o logs/paired1_%j.out kit/paired_stage1.sh
set -e
cd ~/cm1_bundle
export KIT_DIR=$HOME/cm1_bundle/kit; . kit/env.sh
[ -x build_l3/run/cm1.exe ] || { echo "build_l3 missing: run kit/bump_smoke.sh first"; exit 1; }
E=ens_P; mkdir -p $E/members
declare -A SRC=( [memA029]=ens_A/members/mem029 [memA030]=ens_A/members/mem030 [memB001]=ens_A2/members/mem001 [memB002]=ens_A2/members/mem002 )
for n in "${!SRC[@]}"; do
  D=$E/members/$n; s=${SRC[$n]}
  if [ ! -f $D/cm1.done ]; then
    rm -rf $D; mkdir -p $D
    (cd $s && tar cf - --exclude='cm1out*' --exclude=cm1.done --exclude=cm1.log --exclude=cm1.exe --exclude='cm1rst*' .) | (cd $D && tar xf -)
    cp build_l3/run/cm1.exe $D/
    python3 - "$D/namelist.input" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"(?m)^ *timax *=.*$", " timax = 7260.0,", s)
s = re.sub(r"(?m)^ *rstfrq *=.*$", " rstfrq = 7200.0,", s)
open(p, "w").write(s)
PY
    grep -E "^ *(timax|rstfrq|tapfrq|irst|var1|var3) " $D/namelist.input
  fi
done
printf "%s\n" "${!SRC[@]}" | xargs -P 4 -I{} bash kit/run_member.sh $PWD/$E/members/{}
ls $E/members/*/cm1rst_000001_s.dat
