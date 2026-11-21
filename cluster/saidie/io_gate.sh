#!/bin/bash
# 10b: wait until the GPU node reads the shared /home (NFS) fast enough to train from memory-mapped caches. On 2026-11-21 the node read
# at 7-10 MB/s (login node 636 MB/s): every training job stalled for hours in disk wait. Reads 512 MB of a randomly chosen cache file
# (unlikely to be in the page cache) every 10 min; exits 0 once the rate is >= MIN_MBS (default 200).
MIN=${MIN_MBS:-200}
while true; do
  f=$(ls $HOME/cm1_cache_C/*.npz $HOME/cm1_cache_D/*.npz | shuf -n 1)
  s=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M count=32 2>/dev/null; e=$(date +%s.%N)
  r=$(echo "512 / ($e - $s)" | bc); echo "$(date +%H:%M) $(basename $f): $r MB/s"
  [ "$r" -ge "$MIN" ] && exit 0
  sleep 600
done
