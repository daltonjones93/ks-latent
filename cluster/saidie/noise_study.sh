#!/bin/bash
# G-noise: learned model error (stochastic propagator + Stage C) against hand-set latent noise on the
# deterministic Stage C model, in the DA twin. Same truths (the full run's 22 test storms: ens_A mem029-030,
# ens_A2 mem001-010, 10 ens_B), same 3 observation seeds, same 32 randomly drawn training members, theta' +
# reflectivity + radial velocity. Two chains, one GPU each, each starting with that model's reflectivity
# head:
#   deterministic full_v1_stagec, latent noise 0.02 / 0.05 / 0.1   (after DET_AFTER, default 8798)
#   stochastic full_v1_stagec_crps, latent noise 0 / 0.02           (after STOCH_AFTER, default 8820)
# Compare with: python scripts/cm1/compare_studies.py ~/cm1_da/noise_det_0.05 ~/cm1_da/noise_det_0.02 \
#     ~/cm1_da/noise_det_0.1 ~/cm1_da/noise_stoch_0 ~/cm1_da/noise_stoch_0.02
set -euo pipefail
cd "$HOME/latent_DA"
DATA="cache:$HOME/cm1_cache+$HOME/cm1_cache_A2+$HOME/cm1_cache_B"
COMMON="--include-test --modes latent --n-members 32 --random-members --seeds 0 1 2 --obs-set th+refl+vr"
chain() {   # chain AFTER RUN LABEL noise...
  local dep=$1 run=$2 label=$3; shift 3
  local r j
  r=$(sbatch --parsable --dependency=afterok:$dep --mem=180G --export=ALL,RUN=$run,DATA="$DATA" cluster/saidie/refl_head.slurm)
  echo "$run reflectivity head: job $r"
  j=$r
  for n in "$@"; do   # each study needs the head (afterok) and waits for the previous one (afterany): one GPU per chain
    j=$(sbatch --parsable --dependency=afterok:$r,afterany:$j --time=08:00:00 --export=ALL,RUN=$run,DATA="$DATA",OUT=noise_${label}_$n,EXTRA="$COMMON --model-noise $n" cluster/saidie/da_study.slurm)
    echo "$run noise $n: job $j"
  done
}
chain "${DET_AFTER:-8798}" full_v1_stagec det 0.05 0.02 0.1
chain "${STOCH_AFTER:-8820}" full_v1_stagec_crps stoch 0 0.02
