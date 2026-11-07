#!/bin/bash
# H3 / G5 / D8: generalization to unseen environments. The full run's recipe and data (ens_A + ens_A2 + ens_B,
# d34 recipe + Stage C), but with a region of environment space held out of training and validation:
# dry, strongly sheared ens_B storms (qv offset < -1.2 g/kg and shear scale >= 1.1: 10 storms, the corner of
# the ens_B design). Then forecast scores and DA inside the region against the in-distribution ens_B test
# storms (the full run's 10 ens_B test storms minus the two inside the region). One GPU, all chained.
#   ssh saidie 'cd ~/latent_DA && AFTER=<job> bash -l cluster/saidie/h3_run.sh'
set -euo pipefail
cd "$HOME/latent_DA"
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
BOX="qv_offset_gkg:-inf:-1.2+shear_scale:1.1:inf"
MAN="${B}=$HOME/cm1_env/ens_B_manifest.csv"
REGION="" ; for i in 005 007 012 016 022 026 028 041 048 060; do REGION="$REGION+${B}mem$i"; done; REGION=${REGION#+}
INDIST="" ; for i in 003 049 054 017 050 030 011 044; do INDIST="$INDIST+${B}mem$i"; done; INDIST=${INDIST#+}
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
SPLIT="--val-members $VAL --test-members $TEST --holdout-env $BOX --manifest $MAN"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform $SPLIT"
R=$HOME/cm1_runs/h3_v1; RC=$HOME/cm1_runs/h3_v1_stagec
dep=${AFTER:+--dependency=afterany:$AFTER}
j1=$(sbatch --parsable ${dep:-} --mem=200G --export=ALL,DATA="$DATA",RUN=h3_v1,EXTRA="$BASE" cluster/saidie/train_cm1.slurm)
j2=$(sbatch --parsable --dependency=afterok:$j1 --mem=200G --export=ALL,DATA="$DATA",RUN=h3_v1_stagec,EXTRA="$BASE --init-from $R --reuse-ae $R --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2" cluster/saidie/train_cm1.slurm)
j3=$(sbatch --parsable --dependency=afterok:$j2 --mem=180G --export=ALL,RUN=h3_v1_stagec,DATA="$DATA" cluster/saidie/refl_head.slurm)
# forecast scores: inside the held-out region vs in-distribution ens_B storms, held boundaries
EVAL=""
for sub in region:$REGION indist:$INDIST; do
  EVAL="$EVAL python scripts/cm1/train_cm1.py --profile full --source $DATA --device cuda --threads 8 --out $RC --eval-only $SPLIT --eval-subset ${sub#*:} --eval-boundary persistence --results-tag h3_${sub%%:*} || exit 1;"
done
j4=$(sbatch --parsable --dependency=afterok:$j3 --job-name=cm1h3eval --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=8 --mem=200G --time=03:00:00 --output=logs/cm1h3eval_%j.out \
    --wrap "[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || exit 1; . ~/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA; $EVAL python scripts/cm1/compare_runs.py $RC/results_h3_indist.json $RC/results_h3_region.json")
# DA (G5): same setup as the G0 headline (held boundaries, th+refl+vr, 32 random members, 3 seeds)
COMMON="--modes latent --n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr --model-noise 0.05 --boundary persistence"
j5=$(sbatch --parsable --dependency=afterok:$j3,afterany:$j4 --time=08:00:00 --export=ALL,RUN=h3_v1_stagec,DATA="$DATA",OUT=h3_da_region,EXTRA="$COMMON --truths ${REGION//+/ }" cluster/saidie/da_study.slurm)
j6=$(sbatch --parsable --dependency=afterok:$j3,afterany:$j5 --time=08:00:00 --export=ALL,RUN=h3_v1_stagec,DATA="$DATA",OUT=h3_da_indist,EXTRA="$COMMON --truths ${INDIST//+/ }" cluster/saidie/da_study.slurm)
echo "h3_v1 $j1, stagec $j2, refl head $j3, forecast scores $j4, DA region $j5, DA in-distribution $j6"
