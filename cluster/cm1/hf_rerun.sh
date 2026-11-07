#!/bin/bash
# Track FP3: re-run existing members with a shorter output interval (same namelist and sounding, so the same
# storm bit for bit; adapt_dt = 0, output does not feed back on the dynamics). Then FP0 on the new output at
# dt = 10 / 20 / 30 / 60 s, and a check that every 6th frame equals the original 1-min output.
#   (on Cashew) bash ~/latent_DA/cluster/cm1/hf_rerun.sh            # ens_A mem001 mem002, 10 s, 3 h
set -euo pipefail
SRC=${SRC:-$HOME/cm1_bundle/ens_A}
DST=${DST:-$HOME/cm1_bundle/ens_A_hf}
MEMS=${MEMS:-"mem001 mem002"}
TAPFRQ=${TAPFRQ:-10.0}; TIMAX=${TIMAX:-10800.0}
KIT=$HOME/cm1_bundle/kit
mkdir -p "$DST/members"
for m in $MEMS; do
  d=$DST/members/$m
  if [ -d "$d" ]; then echo "$d exists (kept)"; continue; fi
  mkdir -p "$d"
  for f in "$SRC/members/$m"/*; do
    case "$(basename "$f")" in cm1out*|cm1.done|cm1.queued|cm1.log|cm1.failed) continue;; esac
    cp -p "$f" "$d/"
  done
  sed -i -E "s/^( *tapfrq *= *)[0-9.]+/\1$TAPFRQ/; s/^( *timax *= *)[0-9.]+/\1$TIMAX/" "$d/namelist.input"
  grep -E "^ *(tapfrq|timax) *=" "$d/namelist.input"
done
cd "$HOME/cm1_bundle"
out=$(bash "$KIT/submit_ensemble.sh" "$DST" 2)
echo "$out"
jid=$(echo "$out" | sed -n 's/^job \([0-9]*\):.*/\1/p' | head -1)
[ -n "$jid" ] || { echo "no CM1 job submitted (already done?); submit FP0-hf by hand"; exit 0; }
cd "$HOME/latent_DA"
R="$DST/members/mem001 $DST/members/mem002"
prev=$jid
for k in 1 2 3 6; do
  j=$(sbatch --parsable --dependency=afterok:$prev --export=ALL,RUNS="$R",FIT=1,OUT=$HOME/cm1_da/f0_hf_dt$k,EXTRA="--dt-stride $k --t-range 60 170" \
      cluster/cashew_gpu/f0_pilot.slurm)
  echo "FP0-hf dt-stride $k: $j"
  prev=$j
done
j=$(sbatch --parsable --dependency=afterok:$jid --partition=coe_cpu --cpus-per-task=2 --mem=48G --time=01:00:00 \
    --job-name=cm1hfchk --output=logs/cm1hfchk_%j.out --wrap ". $HOME/venvs/cm1/bin/activate; export PYTHONPATH=$HOME/latent_DA
python - <<'PY'
import numpy as np
from ks_latent.cm1.io import read_grads
for m in ('mem001', 'mem002'):
    a = read_grads('$SRC/members/%s/cm1out_s.ctl' % m)
    b = read_grads('$DST/members/%s/cm1out_s.ctl' % m)
    n = min(a['th'].shape[0], b['th'][::6].shape[0])
    for v in ('th', 'qv', 'uinterp', 'winterp', 'qr'):
        d = np.abs(a[v][:n] - b[v][::6][:n]).max()
        print(m, v, 'frames', n, 'max |1-min output - every 6th 10-s frame|', d)
PY")
echo "consistency check: $j"
