#!/bin/bash
# Summarize an ensemble: done / failed / not finished, and run times.
# Usage: bash status.sh /path/to/ens_A
ENS=$1; done=0; failed=0; other=0; times=()
for d in "$ENS"/members/mem*; do
  if [ -f "$d/cm1.done" ]; then done=$((done+1)); times+=("$(cat "$d/cm1.done")")
  elif [ -f "$d/cm1.failed" ]; then failed=$((failed+1)); echo "FAILED: $d"
  else other=$((other+1)); fi
done
echo "done $done, failed $failed, running or not started $other"
[ ${#times[@]} -gt 0 ] && printf "%s\n" "${times[@]}" | sort -n | awk '{a[NR]=$1} END{printf "run time (min): fastest %.0f, median %.0f, slowest %.0f\n", a[1]/60, a[int((NR+1)/2)]/60, a[NR]/60}'
du -sh "$ENS" 2>/dev/null
