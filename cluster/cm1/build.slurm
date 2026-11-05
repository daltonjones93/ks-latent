#!/bin/bash
# Build CM1 as a Slurm job (submitted by go.sh).
#SBATCH --job-name=cm1build
#SBATCH --ntasks=1
#SBATCH --partition=coe_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cm1_build_%j.out
set -euo pipefail
KIT_DIR=$1; BUNDLE=$2
. "$KIT_DIR/env.sh"
cd "$BUNDLE"
JOBS=2 bash "$KIT_DIR/build_cm1.sh" src build
