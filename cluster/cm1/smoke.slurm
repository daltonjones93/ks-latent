#!/bin/bash
# 5-minute test run of member 1's setup with the new build (submitted by go.sh).
# The ensemble jobs only start if this succeeds.
#SBATCH --job-name=cm1smoke
#SBATCH --ntasks=1
#SBATCH --partition=coe_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cm1_smoke_%j.out
set -euo pipefail
KIT_DIR=$1; BUNDLE=$2
. "$KIT_DIR/env.sh"
cd "$BUNDLE"
FMT=2; [ "$NONETCDF" = 1 ] && FMT=1
rm -rf ens_smoke
python3 "$KIT_DIR/make_ensemble.py" --template template --exe build/run/cm1.exe --out ens_smoke \
    --n 0 --seed 0 --timax 300 --output-format $FMT --reference-sounding template/input_sounding_member1
bash "$KIT_DIR/run_member.sh" ens_smoke/members/mem000
ls ens_smoke/members/mem000/cm1out* >/dev/null
s=$(cat ens_smoke/members/mem000/cm1.done)
echo "SMOKE OK: 5 simulated min took $s s; a 6 h member should take roughly $((s * 72 / 60))-$((s * 72 * 2 / 60)) min"
