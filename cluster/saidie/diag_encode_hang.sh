#!/bin/bash -l
# 2026-11-25 diagnostic: F16s1 (9917) and F17 (9927) hung for 3.5 h / 5 h right after "[A] reusing the autoencoder", in
# encode_members (100% of one core, GPU idle, no "[B] propagator" line); the same step took seconds on 11-23/24.
# Reruns F17's command (Stage B 300 steps) into a scratch run dir with a 35-min timeout (SIGABRT -> faulthandler traceback),
# sampling user/system CPU time, page faults and the cgroup's memory every 60 s.
#   ssh saidie 'cd ~/latent_DA && bash -l cluster/saidie/diag_encode_hang.sh [MEM]'     (MEM default 64G, as the hung jobs)
set -euo pipefail
cd "$HOME/latent_DA"
MEM=${1:-64G}
B=ens_B_; A2=ens_A2_
VAL0="mem028+${A2}mem011+${B}mem025+${B}mem053+${B}mem055+${B}mem058"
TEST0="mem029+mem030"
for i in 001 002 003 004 005 006 007 008 009 010; do TEST0="$TEST0+${A2}mem$i"; done
for i in 003 049 016 054 022 017 050 030 011 044; do TEST0="$TEST0+${B}mem$i"; done
NV=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['val']))")
NT=$(python3 -c "import json;print('+'.join(json.load(open('docs/cm1_split_E_final.json'))['test']))")
L_=/tmp/c838596879_cm1; R=$HOME/cm1_runs
NEW="cache:$L_/cm1_cache+$L_/cm1_cache_A2+$L_/cm1_cache_B+$L_/cm1_cache_C+$L_/cm1_cache_C2+$L_/cm1_cache_D"
for d in cm1_cache_E1 cm1_cache_E_pilot4 cm1_cache_E2 cm1_cache_E3 cm1_cache_E4 cm1_cache_E5; do NEW="$NEW+$L_/$d"; done
ARGS="--latent-channels 32 --k-max 45 --k-sampling uniform --noise-channels 4 --crps-samples 4 --eval-samples 8 --tf32 --mmap"
ARGS="$ARGS --steps-b 300 --mixer-variant gelu_ln --val-members $VAL0+$NV --test-members $TEST0+$NT --normalizer-from $R/F16"
ARGS="$ARGS --ae-kind conv --hidden 416 --prop-body mixer --prop-hidden 1024 --window-b 48 --nested-frac 0.5 --steps-a 60000"
ARGS="$ARGS --w-pred-a 1 --pred-k-a 1 --seed 0 --reuse-ae $R/F17"
JOB=logs/diag_enc_job_$MEM.sh
cat > "$JOB" <<JOBEOF
#!/bin/bash
set -u
echo CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES mem=$MEM; free -g | head -2
. \$HOME/venvs/cm1/bin/activate; export PYTHONPATH=\$HOME/latent_DA OMP_NUM_THREADS=16
OUT=\$HOME/cm1_runs/diag_enc_\$SLURM_JOB_ID
timeout -s ABRT 2100 python -X faulthandler scripts/cm1/train_cm1.py --profile full --source '$NEW' --device cuda --threads 16 \\
  --cache \$HOME/cm1_cache_build --out \$OUT $ARGS &
P=\$!
CG=/sys/fs/cgroup\$(cut -d: -f3 /proc/self/cgroup | head -1)
while kill -0 \$P 2>/dev/null; do
  sleep 60
  C=\$(pgrep -P \$P python || echo \$P)
  read -r ut st mf <<<"\$(awk '{print \$14, \$15, \$12}' /proc/\$C/stat 2>/dev/null)"
  echo "[diag] \$(date +%H:%M:%S) utime \$ut stime \$st majflt \$mf state \$(awk '{print \$3}' /proc/\$C/stat 2>/dev/null) cg_mem \$(cat \$CG/memory.current 2>/dev/null || cat \$CG/memory.usage_in_bytes 2>/dev/null) | \$(tail -1 \$OUT/log.txt 2>/dev/null | cut -c1-80)"
done
wait \$P; echo rc=\$?
JOBEOF
bash -n "$JOB"
sbatch --parsable --job-name=diag_enc --account=chen-saidie --partition=general --nodelist=saidie-dgx1 --gres=gpu:h200:1 \
  --cpus-per-task=16 --mem="$MEM" --time=00:40:00 --output="logs/diag_enc_%j.out" "$JOB"
