#!/bin/bash
# Run one CM1 member in its own directory. Safe to re-run: a finished member
# (cm1.done present) is skipped, a failed one is re-run from the start.
# Usage: bash run_member.sh /path/to/ens/members/mem001
set -euo pipefail
D="$1"
cd "$D"
if [ -f cm1.done ]; then echo "$D already done"; exit 0; fi
rm -f cm1out* cm1.failed
export OMP_NUM_THREADS=1
start=$(date +%s)
./cm1.exe > cm1.log 2>&1 || true
end=$(date +%s)
if grep -q "Program terminated normally" cm1.log; then
    echo "$((end - start))" > cm1.done
    echo "$D finished in $((end - start)) s"
else
    touch cm1.failed
    echo "$D FAILED after $((end - start)) s; last lines of cm1.log:" >&2
    tail -20 cm1.log >&2
    exit 1
fi
