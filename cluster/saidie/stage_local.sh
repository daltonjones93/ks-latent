#!/bin/bash
# 10b: copy the CM1 caches to the GPU node's local disk (/tmp on saidie-dgx1 persists across jobs; 1.6 TB free). Training from the shared
# NFS /home stalled for hours on 2026-11-21 (small random mmap reads; 7-120 MB/s), so training jobs read from this copy instead.
# Four rsyncs in parallel; verified by size; idempotent (rsync skips files already copied). Remove with: rm -r /tmp/c838596879_cm1
set -euo pipefail
DST=/tmp/c838596879_cm1; mkdir -p $DST
for d in ${DIRS:-cm1_cache cm1_cache_A2 cm1_cache_B cm1_cache_C cm1_cache_C2 cm1_cache_D}; do echo $HOME/$d; done | xargs -P 4 -I{} rsync -a {} $DST/
for d in ${DIRS:-cm1_cache cm1_cache_A2 cm1_cache_B cm1_cache_C cm1_cache_C2 cm1_cache_D}; do
  # sizes of the .npz files only (a killed rsync leaves hidden partial files that du would count)
  a=$(stat -c %s $HOME/$d/*.npz | paste -sd+ | bc); b=$(stat -c %s $DST/$d/*.npz | paste -sd+ | bc); n1=$(ls $HOME/$d/*.npz | wc -l); n2=$(ls $DST/$d/*.npz | wc -l)
  echo "$d: $n2/$n1 files, $b/$a bytes"; [ "$a" = "$b" ] && [ "$n1" = "$n2" ] || { echo "MISMATCH $d"; exit 1; }
done
df -h /tmp | tail -1
