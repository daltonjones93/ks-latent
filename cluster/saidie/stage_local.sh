#!/bin/bash
# 10b: copy the CM1 caches to the GPU node's local disk (/tmp on saidie-dgx1 persists across jobs; 1.6 TB free). Training from the shared
# NFS /home stalled for hours on 2026-11-21 (small random mmap reads; 7-120 MB/s), so training jobs read from this copy instead.
# Four rsyncs in parallel; verified by size; idempotent (rsync skips files already copied). Remove with: rm -r /tmp/c838596879_cm1
set -euo pipefail
DST=/tmp/c838596879_cm1; mkdir -p $DST
ls -d $HOME/cm1_cache $HOME/cm1_cache_A2 $HOME/cm1_cache_B $HOME/cm1_cache_C $HOME/cm1_cache_C2 $HOME/cm1_cache_D | xargs -P 4 -I{} rsync -a {} $DST/
for d in cm1_cache cm1_cache_A2 cm1_cache_B cm1_cache_C cm1_cache_C2 cm1_cache_D; do
  a=$(du -sb $HOME/$d | cut -f1); b=$(du -sb $DST/$d | cut -f1); n1=$(ls $HOME/$d/*.npz | wc -l); n2=$(ls $DST/$d/*.npz | wc -l)
  echo "$d: $n2/$n1 files, $b/$a bytes"; [ "$a" = "$b" ] && [ "$n1" = "$n2" ] || { echo "MISMATCH $d"; exit 1; }
done
df -h /tmp | tail -1
