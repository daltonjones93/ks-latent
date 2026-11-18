#!/bin/bash
# 9b (user: "launch all three"): is the model under-trained or capacity-limited? Three arms on the F6 recipe (conv autoencoder + mixer with GELU + LayerNorm,
# Stage A prediction term, 573 training storms: ens_A, A2, B, C, D; the same 6 validation and 22 test storms):
#   F8_long       2x steps in every stage (A 120k, B 80k, C 60k) with the F6 sizes (conv 5.0 M, mixer 11.9 M); Stages A, B, C as separate jobs
#   F9_bigconv    conv hidden 640 (9.4 M) with the F6 mixer, F6 steps (A 60k, B 40k, C 30k); A+B in one job, then C
#   F10_bigmixer  mixer width 1024 (47.3 M) with the F6 conv, F6 steps; A, B, C as separate jobs (Stage B and C are 2-3x heavier)
# Time limits from F6 (A 52 min with the prediction term, B 75 min + encoding, C 90 min) with margin: a Stage B timeout is recoverable (the autoencoder is
# saved), a Stage C timeout is not. One evaluation per arm against F6, F4 and g_local48.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/scale_runs.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B+$HOME/cm1_cache_C+$HOME/cm1_cache_C2+$HOME/cm1_cache_D"
R=$HOME/cm1_runs
BASEC="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap --mixer-variant gelu_ln"
sb() { sbatch --parsable --mem=48G "$@"; }
job() { # name time dep extra-args-for-train_cm1
  local n=$1 t=$2 dep=$3; shift 3
  sb --time=$t ${dep:+--dependency=afterok:$dep} --job-name=$n --export=ALL,DATA="$DATA",RUN=$n,EXTRA="$*" cluster/saidie/train_cm1.slurm
}
ref="F6_conv416_mixer512_gelu_pred_all_stagec F4_conv416_mixer512_pred_stagec g_local48_stagec"
evalj() { sbatch --parsable --time=00:25:00 --dependency=afterany:$2 --export=ALL,OUT=v_eval_$1,RUNS="${1}_stagec $ref",EXTRA_EVAL="--ref-norm-run ${HOME}/cm1_runs/um_local_stagec" cluster/saidie/vit_eval.slurm; }
CA() { echo "--ae-kind conv --hidden $1 --prop-body mixer --prop-hidden $2 --window-b 48 --nested-frac 0.5"; }
CC() { echo "--stage-c-steps $1 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 4 --nested-frac 0.5 --stage-c-lr 1e-4"; }

# F8: long, stages split
n=F8_long
a=$(job ${n}_A 02:30:00 "" $BASEC $(CA 416 512) --steps-a 120000 --steps-b 80000 --w-pred-a 1 --pred-k-a 1 --stage-a-only)
# (run name for the autoencoder job is ${n}_A: Stage B reuses it)
b=$(job ${n}_B 03:30:00 $a $BASEC $(CA 416 512) --steps-b 80000 --reuse-ae $R/${n}_A)
c=$(job ${n}_stagec 03:50:00 $b $BASEC $(CA 416 512) --init-from $R/${n}_B --reuse-ae $R/${n}_B $(CC 60000))
e=$(evalj $n $c); echo "$n: A $a, B $b, C $c, eval $e"

# F9: bigger conv, A+B together
n=F9_bigconv
ab=$(job $n 03:40:00 "" $BASEC $(CA 640 512) --steps-a 60000 --steps-b 40000 --w-pred-a 1 --pred-k-a 1)
c=$(job ${n}_stagec 02:40:00 $ab $BASEC $(CA 640 512) --steps-b 40000 --init-from $R/$n --reuse-ae $R/$n $(CC 30000))
e=$(evalj $n $c); echo "$n: A+B $ab, C $c, eval $e"

# F10: bigger mixer, stages split
n=F10_bigmixer
a=$(job ${n}_A 01:30:00 "" $BASEC $(CA 416 1024) --steps-a 60000 --steps-b 40000 --w-pred-a 1 --pred-k-a 1 --stage-a-only)
b=$(job ${n}_B 03:30:00 $a $BASEC $(CA 416 1024) --steps-b 40000 --reuse-ae $R/${n}_A)
c=$(job ${n}_stagec 03:30:00 $b $BASEC $(CA 416 1024) --steps-b 40000 --init-from $R/${n}_B --reuse-ae $R/${n}_B $(CC 30000))
e=$(evalj $n $c); echo "$n: A $a, B $b, C $c, eval $e"
