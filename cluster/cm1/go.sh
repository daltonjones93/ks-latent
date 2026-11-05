#!/bin/bash
# One command on the Cashew login node:  bash kit/go.sh
# Finds modules, then submits (as Slurm jobs, chained):  build -> smoke test -> ensemble.
# Nothing heavy runs on the login node. Safe to re-run: finished steps are skipped,
# unfinished members are resubmitted.
# Settings (environment variables, all optional):
#   N=30 SEED=1 ENS=ens_A CPUS=8 TIMAX=21600 TAPFRQ=60 EMAIL=you@colostate.edu
#   EXTRA_ARGS="--qv-offset -1.5 1.0 --shear-scale 0.6 1.4"   (Phase B environments)
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd); BUNDLE=$(cd "$KIT_DIR/.." && pwd); cd "$BUNDLE"
N=${N:-30}; SEED=${SEED:-1}; ENS=${ENS:-ens_A}; CPUS=${CPUS:-8}
TIMAX=${TIMAX:-21600}; TAPFRQ=${TAPFRQ:-60}; EMAIL=${EMAIL:-}; EXTRA_ARGS=${EXTRA_ARGS:-}
say() { echo "[go] $*"; }
die() { echo "[go] ERROR: $*" >&2; exit 1; }
[ -d src ] && [ -d template ] || die "run from the unpacked cm1_bundle (src/ and template/ not found)"
command -v sbatch >/dev/null || die "sbatch not found: this must run on the cluster login node"
command -v python3 >/dev/null || die "python3 not found (try: module load python, then re-run)"
mkdir -p logs

# ---- 1. modules (recorded in kit/cashew.conf; edit that file to override) ----
if [ ! -f "$KIT_DIR/cashew.conf" ]; then
  . "$KIT_DIR/env.sh"   # makes `module` available
  pick() { module -t avail 2>&1 | grep -i -E "^$1(/|$)" | sed 's/(default)//; s/:$//' | sort -V | tail -1; }
  GCC=""; NF=""; MODS=""
  if command -v module >/dev/null 2>&1 || type module >/dev/null 2>&1; then
    GCC=$(pick gcc || true)
    [ -n "$GCC" ] && module load "$GCC" 2>/dev/null || true
    NF=$(pick netcdf-fortran || true)
    [ -z "$NF" ] && NF=$(pick netcdf || true)
    MODS="$GCC${NF:+ $NF}"
    [ -n "$NF" ] && module load "$NF" 2>/dev/null || true
  fi
  command -v gfortran >/dev/null || die "no gfortran found (modules seen: '${MODS}'). Put MODULES=\"...\" in kit/cashew.conf"
  NONET=0
  if ! command -v nf-config >/dev/null; then
    NONET=1; say "WARNING: netcdf-fortran not found; building without NetCDF (GrADS output, readable on the Mac)"
  fi
  printf 'MODULES="%s"\nNONETCDF=%s\n' "$MODS" "$NONET" > "$KIT_DIR/cashew.conf"
  say "modules: '${MODS:-none needed}', NONETCDF=$NONET (saved to kit/cashew.conf)"
fi
. "$KIT_DIR/cashew.conf"
FMT=2; [ "$NONETCDF" = 1 ] && FMT=1

MAIL=(); [ -n "$EMAIL" ] && MAIL=(--mail-type=END,FAIL --mail-user="$EMAIL")
DEP=()

# jobs of ours already in the queue with this name (so re-runs never duplicate them)
queued() { squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null | head -1; }

# ---- 2. build (skipped if cm1.exe exists or a build job is already queued) ----
if [ -x build/run/cm1.exe ]; then
  say "build: build/run/cm1.exe exists, skipping"
elif jb=$(queued cm1build) && [ -n "$jb" ]; then
  say "build: job $jb already queued, reusing it"
  DEP=(--dependency=afterok:$jb --kill-on-invalid-dep=yes)
else
  jb=$(sbatch --parsable "$KIT_DIR/build.slurm" "$KIT_DIR" "$BUNDLE" | cut -d";" -f1)
  say "build: job $jb"
  DEP=(--dependency=afterok:$jb --kill-on-invalid-dep=yes)
fi

# ---- 3. smoke test (skipped if it already passed) ----
if [ -f ens_smoke/members/mem000/cm1.done ]; then
  say "smoke test: already passed, skipping"
elif js=$(queued cm1smoke) && [ -n "$js" ]; then
  say "smoke test: job $js already queued, reusing it"
  DEP=(--dependency=afterok:$js --kill-on-invalid-dep=yes)
else
  js=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} "$KIT_DIR/smoke.slurm" "$KIT_DIR" "$BUNDLE" | cut -d";" -f1)
  say "smoke test: job $js (starts after the build)"
  DEP=(--dependency=afterok:$js --kill-on-invalid-dep=yes)
fi

# ---- 4. ensemble members (generated once, on the login node: only writes small files) ----
if [ ! -d "$ENS/members" ]; then
  python3 "$KIT_DIR/make_ensemble.py" --template template --exe build/run/cm1.exe --out "$ENS" \
      --n "$N" --seed "$SEED" --timax "$TIMAX" --tapfrq "$TAPFRQ" --output-format $FMT \
      --reference-sounding template/input_sounding_member1 --exe-may-not-exist-yet $EXTRA_ARGS
else
  say "ensemble: $ENS exists, resubmitting unfinished members only"
fi

# ---- 5. submit packed jobs ----
bash "$KIT_DIR/submit_ensemble.sh" "$ENS" "$CPUS" ${DEP[@]+"${DEP[@]}"} ${MAIL[@]+"${MAIL[@]}"}
say "all submitted. Watch with:  squeue -u \$USER    and    bash kit/status.sh $ENS"
say "logs are in $BUNDLE/logs/. Re-run 'bash kit/go.sh' any time to resubmit what is unfinished."
