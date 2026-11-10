#!/bin/bash
# Build CM1 (serial, NetCDF output) from the bundle's src/ into BUILD_DIR.
# Run inside an interactive Slurm job, not on the login node:
#   srun --account=ACCOUNT --partition=PARTITION --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash
#   module load <fortran compiler> <netcdf-fortran>      # names from discover.sh
#   bash build_cm1.sh src build
# Needs gfortran (or set FC=ifort/ifx) and nf-config from netcdf-fortran on PATH.
# NONETCDF=1 builds without NetCDF (GrADS output; then use --output-format 1).
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)   # before any cd: $0 may be a relative path
SRC=$(cd "$1" && pwd); BUILD=$2
FC=${FC:-gfortran}; JOBS=${JOBS:-2}
command -v "$FC" >/dev/null || { echo "Fortran compiler '$FC' not on PATH (module load it)"; exit 1; }
mkdir -p "$BUILD/run"; BUILD=$(cd "$BUILD" && pwd)
rm -rf "$BUILD/src"; cp -R "$SRC" "$BUILD/src"; cd "$BUILD/src"
rm -f ./*.o ./*.mod ./*.f90
# our source patches (patches/apply_patches.py), applied to this copy only; recorded for make_ensemble.py
python3 "$KIT/patches/apply_patches.py" "$BUILD/src"
echo "seed_var1 two_storms_var2 tiny_pert_var3 bump_restart_var5" > "$BUILD/PATCHES"
M=Makefile
if [ "${NONETCDF:-0}" = 1 ]; then
    sed -i.bak -E 's/^(OUTPUTINC|OUTPUTLIB|LINKOPTS|OUTPUTOPT)/#\1/' $M
else
    command -v nf-config >/dev/null || { echo "nf-config not on PATH: module load netcdf-fortran (or NONETCDF=1)"; exit 1; }
    NFINC=$(nf-config --includedir)
    NFLIBS=$(nf-config --flibs)
    NFLIBDIR=$(nf-config --prefix)/lib
    NCLIBDIR=$(nc-config --libdir 2>/dev/null || echo "$NFLIBDIR")
    # rpath so cm1.exe finds the NetCDF libraries at run time even without module load
    sed -i.bak -E \
      -e "s|^#?OUTPUTINC.*|OUTPUTINC = -I$NFINC|" \
      -e "s|^#?OUTPUTLIB.*|OUTPUTLIB = |" \
      -e "s|^#?OUTPUTOPT.*|OUTPUTOPT = -DNETCDF -DNCFPLUS|" \
      -e "s|^#?LINKOPTS.*|LINKOPTS  = $NFLIBS -lnetcdf -Wl,-rpath,$NFLIBDIR -Wl,-rpath,$NCLIBDIR|" $M
fi
if [ "$FC" = gfortran ]; then
    OPTS="-ffree-form -ffree-line-length-none -O2 -finline-functions"
else   # Intel ifort/ifx (the Makefile's own Intel settings)
    OPTS="-O3 -assume byterecl -fp-model precise -ftz -no-fma"
fi
echo "building with FC=$FC, $JOBS jobs"; grep -E "^(OUTPUT|LINKOPTS)" $M || true
make -j"$JOBS" FC="$FC" OPTS="$OPTS" > "$BUILD/make.log" 2>&1 || { tail -30 "$BUILD/make.log"; echo "BUILD FAILED, see $BUILD/make.log"; exit 1; }
test -x "$BUILD/run/cm1.exe" && echo "built $BUILD/run/cm1.exe"
ldd "$BUILD/run/cm1.exe" 2>/dev/null | grep -i -E "netcdf|not found" || true
