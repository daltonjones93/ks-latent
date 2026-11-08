#!/bin/bash
# Fable review section 6 (docs/cm1_fable_review.md; pre-registered in cm1_notes.md, G0 addendum 4). Two chains, one GPU
# each, so with the chains they follow at most 2 GPUs are in use:
#   chain A (after AFTER_A, the FPd chain): N5 localization -> N2 adaptive noise -> N4 ensemble size -> M1 sparse
#            observations x cycle length -> N6 cost table
#   chain B (after AFTER_B, the G2b chain): M3 stochastic propagator on full_v1_stagec's frozen autoencoder (Stage B,
#            then Stage C with --c-freeze-ae) -> M3 anchored-ensemble DA sweep
#   ssh saidie 'cd ~/latent_DA && AFTER_A=9038 AFTER_B=9072 bash -l cluster/saidie/fable_runs.sh'
set -euo pipefail
cd "$HOME/latent_DA"
python3 cluster/saidie/specs/make_specs.py >/dev/null
B=ens_B_; A2=ens_A2_
VAL="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST="$TEST+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST="$TEST+${B}mem$i"; done
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
BASE="--latent-channels 32 --k-max 45 --k-sampling uniform --val-members $VAL --test-members $TEST"
R=$HOME/cm1_runs
a=${AFTER_A:-}
for s in n5_loc n2_adapt n4_ens m1_sparse; do
  a=$(sbatch --parsable ${a:+--dependency=afterany:$a} --job-name=sw_$s --export=ALL,SPEC=cluster/saidie/specs/$s.json,ROOT=fable cluster/saidie/da_sweep.slurm)
  echo "chain A  $s: $a"
done
a=$(sbatch --parsable --dependency=afterany:$a --account=chen-saidie --partition=general --gres=gpu:h200:1 --cpus-per-task=8 \
      --mem=64G --time=01:00:00 --job-name=cm1cost --output=logs/cm1cost_%j.out \
      --wrap ". ~/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA; [ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ] || exit 1; cd \$HOME/latent_DA; python scripts/cm1/cost_table.py --run $R/full_v1_stagec --source $DATA --out \$HOME/cm1_da/fable/cost_h200 --device cuda")
echo "chain A  cost: $a"
b=${AFTER_B:-}
b=$(sbatch --parsable ${b:+--dependency=afterany:$b} --mem=200G --export=ALL,DATA="$DATA",RUN=m3_stoch_frozen,EXTRA="$BASE --reuse-ae $R/full_v1_stagec --noise-channels 4 --crps-samples 4 --eval-samples 8" cluster/saidie/train_cm1.slurm)
echo "chain B  m3_stoch_frozen (Stage B): $b"
b=$(sbatch --parsable --dependency=afterok:$b --mem=200G --export=ALL,DATA="$DATA",RUN=m3_stoch_frozen_stagec,EXTRA="$BASE --init-from $R/m3_stoch_frozen --reuse-ae $R/m3_stoch_frozen --stage-c-steps 15000 --stage-c-k 45 --w-sfc-th 2 --stage-c-crps 4 --stage-c-batch 6 --eval-samples 8 --c-freeze-ae" cluster/saidie/train_cm1.slurm)
echo "chain B  m3_stoch_frozen_stagec (Stage C, frozen AE): $b"
b=$(sbatch --parsable --dependency=afterok:$b --account=chen-saidie --partition=general --cpus-per-task=1 \
      --mem=2G --time=00:10:00 --job-name=cm1refl --output=logs/cm1refl_%j.out \
      --wrap "cp $R/full_v1_stagec/refl_head.pt* $R/m3_stoch_frozen_stagec/ && echo copied")
echo "chain B  reflectivity head copy (same autoencoder): $b"
b=$(sbatch --parsable --dependency=afterok:$b --job-name=sw_m3 --export=ALL,SPEC=cluster/saidie/specs/m3_anchored.json,ROOT=fable cluster/saidie/da_sweep.slurm)
echo "chain B  m3_anchored sweep: $b"
