#!/bin/bash
# J17B (2026-10-10): hidden-memory truth u_t = KS(u) + eps v, v_t = -v/tau + kappa lap u, (tau, kappa) = (2, -1),
# pre-registered levels eps = 0.1 and 0.2 (docs/full_paper_tasks.md J17B). Usage: run_j17b.sh <eps>
# data (5 trajectories in parallel; one level's generation at a time) -> train p7c12d10_memB<eps> (one MPS
# training at a time) while the physics filter (KS model, no v; all remedies) runs -> perfect-model floor
# (model carries v) -> latent filter.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
EPS=$1; MEM="$EPS,2,-1"; LOG=artifacts/logs/j17b.log; E=artifacts/experiments/da2d
NAME=memB$EPS; DS=artifacts/datasets/ks2d_${NAME}_wide6.h5; TAG=p7c12d10_$NAME; P=J17B_e$EPS
log() { echo "$(date '+%F %T') [$P] $*" >> $LOG; }
# --- B4 data ---
while pgrep -f "gen_ks2d_wide_dataset.py.*--memory" > /dev/null && [ ! -f artifacts/datasets/ks2d_${NAME}_wide6_parts/traj4.npy ]; do sleep 60; done
if [ ! -f $DS ]; then
  for i in 0 1 2 3 4; do [ -f artifacts/datasets/ks2d_${NAME}_wide6_parts/traj$i.npy ] || \
    $PY scripts/gen_ks2d_wide_dataset.py --scale 6 --memory $MEM --only $i --out $DS > artifacts/logs/gen_${NAME}_traj$i.log 2>&1 & done
  wait
  $PY scripts/gen_ks2d_wide_dataset.py --scale 6 --memory $MEM --out $DS >> $LOG 2>&1
fi
STD=$($PY -c "import h5py; print(h5py.File('$DS')['metadata'].attrs['normalization_std'])")
log "data assembled, std=$STD"
# --- B5 training, one at a time ---
if [ ! -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  ( while pgrep -f "train_kolmogorov_crop.py" > /dev/null; do sleep 120; done
    log "training started"
    $PY scripts/train_kolmogorov_crop.py --dataset $DS --tag $TAG --crop-sites 40 --crop-sites-y 40 --n-offsets 16 \
      --local-channels 12 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
      --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --data-noise 0.1 > artifacts/logs/crop_train_$TAG.log 2>&1
    log "training done rc=$?" ) &
fi
# truths (cached serially: parallel runs would race on the cache files)
$PY -c "
from ks_latent.config import KSConfig2D
from ks_latent.solver.ks2d_memory import MemoryParams, memory_truth
for L, NX, seeds in ((44.0, 128, (0, 1, 2, 3)), (220.0, 640, (0,))):
    cfg = KSConfig2D(Lx=L, Ly=L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=0)
    for s in seeds: memory_truth(cfg, s, MemoryParams($EPS, 2.0, -1.0))
" >> $LOG 2>&1
# --- B6 physics filter (KS model without v) ---
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f $E/i11_${tag}.json ] && return
  env $ENV nice -n 10 $PY scripts/exp_i11_pde_letkf.py --workers 2 --truth-memory $MEM --norm-std $STD --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep -E "^$tag |Error" artifacts/logs/i11_${tag}.log | cut -c1-200)"; }
best() { $PY - "$1" <<'PYEOF'
import glob, json, math, sys
r = []
for f in glob.glob(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"):
    e = json.load(open(f))["steady"].get("analysis")
    if e is not None and math.isfinite(e): r.append((e, f))
print(min(r)[1].split("i11_")[1][:-5] if r else "")
PYEOF
}
ME=$E/j16_model_error_$NAME.npz
[ -f $ME ] || env $ENV nice -n 10 $PY scripts/exp_j16_estimate_pde_model_error.py --truth-memory $MEM --n-starts 100 --out $NAME >> $LOG 2>&1
lane() { for r in "$@"; do for c in 8.25 16; do pde ${P}_pde_c${c}_rtps${r}_s0 --loc-c $c --rtps $r; done; done; }
lane 0.5 0.9 & lane 0.7 1.0 & wait
BT=$(best "${P}_pde_c*_rtps*_s0"); C=$(echo $BT | sed -E 's/.*_c([0-9.]+)_rtps.*/\1/'); R=$(echo $BT | sed -E 's/.*_rtps([0-9.]+)_s0/\1/')
log "best tuning: $BT"
( pde ${P}_pde_c${C}_rtps${R}_lap_s0 --loc-c $C --rtps $R --estimate-param lap --param-init-std 0.05
  pde ${P}_pde_c${C}_rtps${R}_lap_bc_q0_s0 --loc-c $C --rtps $R --estimate-param lap --param-init-std 0.05 \
      --model-error-file $ME --bias-correct --q-scale 0 ) &
( for q in 0 0.5 1; do pde ${P}_pde_c${C}_rtps${R}_bc_q${q}_s0 --loc-c $C --rtps $R --model-error-file $ME --bias-correct --q-scale $q; done ) &
wait
BP=$(best "${P}_pde_*_s0"); log "best physics config (seed 0): $BP"
ARGS=$($PY - "$BP" <<'PYEOF'
import json, sys
a = json.load(open(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json"))["args"]
o = f"--loc-c {a['loc_c']} --rtps {a['rtps']}"
if a.get("estimate_param", "none") != "none": o += f" --estimate-param {a['estimate_param']} --param-init-std {a['param_init_std']}"
if a.get("bias_correct"): o += f" --model-error-file {a['model_error_file']} --bias-correct --q-scale {a['q_scale']}"
print(o)
PYEOF
)
BASE=${BP%_s0}
pde ${BASE}_s1 $ARGS --seed 1 & pde ${BASE}_s2 $ARGS --seed 2 &
# --- B7 floor: the model carries v (true equations, v unobserved) ---
for sd in 0 1 2; do pde ${P}_floor_c${C}_rtps${R}_s$sd --loc-c $C --rtps $R --model-memory $MEM --seed $sd & done
wait
log "PHYSICS DONE"
# --- B7 latent filter ---
until [ -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ] && ! pgrep -f "train_kolmogorov_crop.py.*$TAG" > /dev/null; do sleep 120; done
lat() { out=$1; shift
  [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --truth-memory $MEM --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
lat ${P}_lat_T4_s0 --n-tilings 4 --seed 0 & lat ${P}_lat_T4_s1 --n-tilings 4 --seed 1 & lat ${P}_lat_T4_s2 --n-tilings 4 --seed 2 & wait
lat ${P}_lat_T1_s0 --n-tilings 1 --seed 0
log "DONE"
