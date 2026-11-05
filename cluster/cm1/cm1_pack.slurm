#!/bin/bash
# Run several CM1 members in one Slurm job, one member per CPU at a time.
# Submitted by submit_ensemble.sh (do not sbatch by hand unless you mean to):
#   sbatch --cpus-per-task=C cm1_pack.slurm KIT_DIR ENS_DIR mem001 mem002 ...
# Partition coe_cpu: the general CPU nodes (the cluster default is coe_gpu, its only GPU node).
#SBATCH --job-name=cm1ens
#SBATCH --ntasks=1
#SBATCH --partition=coe_cpu
#SBATCH --mem-per-cpu=1G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cm1_%j.out
set -uo pipefail
KIT=$1; ENS=$2; shift 2
KIT_DIR=$KIT . "$KIT/env.sh"
echo "job $SLURM_JOB_ID on $(hostname): ${#@} members, ${SLURM_CPUS_PER_TASK} at a time: $*"
printf "%s\n" "$@" | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} bash "$KIT/run_member.sh" "$ENS/members/{}"
rc=$?
[ $rc -eq 0 ] && echo "all members in this job finished" || echo "some members FAILED (xargs exit $rc); run status.sh"
exit $rc
