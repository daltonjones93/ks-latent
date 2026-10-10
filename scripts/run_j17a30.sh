#!/bin/bash
# J17A A9 (2026-10-10): the pre-registered 30% level, beta = 2.2e-3. Copy of run_j17a.sh; generates its own data and
# waits for the 15% model's training to finish before training (one MPS training at a time).
# J17A (2026-10-10): missing cubic process. Truth u_t = KS(u) - beta P[u'^3], beta = 1.1e-3 (pre-registered,
# docs/full_paper_tasks.md J17). Chain: assemble data -> train p7c12d10_cubA (normal priority) while the physics
# filter (standard KS model, all J16 remedies) runs at nice 19 -> latent-filter runs after training.
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/miniforge/base/envs/da_env/bin/python
LOG=artifacts/logs/j17a30.log
B=0.0022
DS=artifacts/datasets/ks2d_cubA30_wide6.h5
TAG=p7c12d10_cubA30
E=artifacts/experiments/da2d
log() { echo "$(date '+%F %T') $*" >> $LOG; }
for i in 0 1 2 3 4; do [ -f artifacts/datasets/ks2d_cubA30_wide6_parts/traj$i.npy ] || \
  $PY scripts/gen_ks2d_wide_dataset.py --scale 6 --cubic $B --only $i --out $DS > artifacts/logs/gen_cubA30_traj$i.log 2>&1 & done
# --- A4: wait for the 5 parallel trajectories, then assemble the h5 (the generator skips finished parts) ---
until [ "$(ls artifacts/datasets/ks2d_cubA30_wide6_parts/traj?.npy 2>/dev/null | wc -l)" -ge 5 ] && \
      ! pgrep -f "gen_ks2d_wide_dataset.py.*cubA30" > /dev/null; do sleep 60; done
[ -f $DS ] || $PY scripts/gen_ks2d_wide_dataset.py --scale 6 --cubic $B --out $DS >> $LOG 2>&1
STD=$($PY -c "import h5py; f=h5py.File('$DS'); m=f['metadata']; print(m.attrs['normalization_std'] if 'normalization_std' in m.attrs else m['normalization_std'][()])")
log "A4 data assembled, std=$STD"
# --- A5: train (same recipe as p7c12d10_ks2d) ---
while pgrep -f "train_kolmogorov_crop.py.*p7c12d10_cubA " > /dev/null; do sleep 120; done
if [ ! -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ]; then
  $PY scripts/train_kolmogorov_crop.py --dataset $DS --tag $TAG --crop-sites 40 --crop-sites-y 40 --n-offsets 16 \
    --local-channels 12 --site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 \
    --k-max 32 --k-mid 16 --noise-end 0.02 --prop-steps 9000 --data-noise 0.1 > artifacts/logs/crop_train_$TAG.log 2>&1 &
  TRAIN_PID=$!; log "A5 training started pid $TRAIN_PID"
fi
# truths on the new attractor, cached serially first (parallel runs would race on the cache files)
$PY - <<PYEOF >> $LOG 2>&1
from ks_latent.config import KSConfig2D
from ks_latent.solver.twin import twin_truth
for L, NX, seeds in ((44.0, 128, (0, 1, 2, 3)), (220.0, 640, (0,))):
    cfg = KSConfig2D(Lx=L, Ly=L, NX=NX, NY=NX, dt=0.02, snapshot_every=5, spinup_time=200.0, seed=0)
    for s in seeds:
        twin_truth(cfg, s, cubic=$B); print("cached truth", L, s, flush=True)
PYEOF
# --- A6: physics filter (KS model, no u^3) ---
ENV="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2"
pde() { tag=$1; shift
  [ -f $E/i11_${tag}.json ] && return
  env $ENV nice -n 19 $PY scripts/exp_i11_pde_letkf.py --workers 2 --truth-cubic $B --norm-std $STD --out $tag "$@" > artifacts/logs/i11_${tag}.log 2>&1
  log "done $tag rc=$? $(grep "^$tag " artifacts/logs/i11_${tag}.log | cut -c1-170)"; }
best() { $PY - "$1" <<'PYEOF'
import glob, json, sys
r = [(json.load(open(f))["steady"].get("analysis", 1e9), f) for f in glob.glob(f"artifacts/experiments/da2d/i11_{sys.argv[1]}.json")]
r = [x for x in r if x[0] == x[0]]
print(min(r)[1].split("i11_")[1][:-5] if r else "")
PYEOF
}
[ -f $E/j16_model_error_cubA30.npz ] || env $ENV nice -n 19 $PY scripts/exp_j16_estimate_pde_model_error.py --truth-cubic $B --n-starts 100 --out cubA30 >> $LOG 2>&1
ME=$E/j16_model_error_cubA30.npz
# (1) tuning sweep, seed 0, two lanes
lane() { for r in "$@"; do for c in 8.25 16; do pde J17A30_pde_c${c}_rtps${r}_s0 --loc-c $c --rtps $r; done; done; }
lane 0.5 0.9 & lane 0.7 1.0 & wait
BT=$(best "J17A30_pde_c*_rtps*_s0"); C=$(echo $BT | sed -E 's/.*_c([0-9.]+)_rtps.*/\1/'); R=$(echo $BT | sed -E 's/.*_rtps([0-9.]+)_s0/\1/')
log "A6 best tuning: $BT (c=$C rtps=$R)"
# (2) remedies at the best tuning, seed 0: bias correction +- Q; nabla^2 surrogate estimation; all combined
( for q in 0 0.5 1 2; do pde J17A30_pde_c${C}_rtps${R}_bc_q${q}_s0 --loc-c $C --rtps $R --model-error-file $ME --bias-correct --q-scale $q; done ) &
( pde J17A30_pde_c${C}_rtps${R}_lap_s0 --loc-c $C --rtps $R --estimate-param lap --param-init-std 0.05
  for q in 0 1; do pde J17A30_pde_c${C}_rtps${R}_lap_bc_q${q}_s0 --loc-c $C --rtps $R --estimate-param lap --param-init-std 0.05 \
       --model-error-file $ME --bias-correct --q-scale $q; done ) &
wait
BP=$(best "J17A30_pde_*_s0"); log "A6 best physics config (seed 0): $BP"
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
pde ${BASE}_s1 $ARGS --seed 1 & pde ${BASE}_s2 $ARGS --seed 2 & wait
# --- A8: perfect-model floor (true equations) at the best tuning ---
for sd in 0 1 2; do pde J17A30_floor_c${C}_rtps${R}_s$sd --loc-c $C --rtps $R --model-cubic-coef $B --seed $sd & done; wait
log "A6/A8 PHYSICS DONE"
# --- A7: latent filter after training ---
until [ -f artifacts/stage2_prop_kolmogorov_crop_$TAG.pt ] && ! pgrep -f "train_kolmogorov_crop.py.*$TAG" > /dev/null; do sleep 120; done
log "A5 training done"
lat() { out=$1; shift
  [ -f $E/h9_${out}.json ] && return
  OMP_NUM_THREADS=3 $PY scripts/exp_h9_tiled_da.py --tag $TAG --obs-stride 8 --rtps 0.7 --truth-cubic $B --out $out "$@" > artifacts/logs/h9_$out.log 2>&1
  log "done $out rc=$? $(grep -E "^$out |DIVERGED" artifacts/logs/h9_$out.log | cut -c1-200 | tr '\n' ' ')"; }
lat J17A30_lat_T4_s0 --n-tilings 4 --seed 0 & lat J17A30_lat_T4_s1 --n-tilings 4 --seed 1 & lat J17A30_lat_T4_s2 --n-tilings 4 --seed 2 & wait
lat J17A30_lat_T1_s0 --n-tilings 1 --seed 0 &
TD=$($PY -c "import json; print(json.load(open('$E/j17a_truth_lyap_beta$B.json'))['D_KY'])" 2>/dev/null)
OMP_NUM_THREADS=4 $PY scripts/exp_crop_eval_ks2d.py --ae-ckpt artifacts/stage1_ae_kolmogorov_crop_$TAG.pt \
  --prop-ckpt artifacts/stage2_prop_kolmogorov_crop_$TAG.pt --scale 1.0 --seed 0 --n-directions 384 --cubic $B \
  ${TD:+--true-dky $TD} --tag ${TAG}_x1.0_seed0 > artifacts/logs/crop_eval_$TAG.log 2>&1
log "done lyap $(grep -E 'rel_err_pct|model_D_KY' artifacts/logs/crop_eval_$TAG.log | tr -d '\n ')"
wait
lat J17A30_lat_L220_T4_s0 --n-tilings 4 --seed 0 --L 220
log "J17A DONE"
