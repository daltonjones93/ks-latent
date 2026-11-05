#!/bin/bash
# (Run on the Mac.) Pack everything the cluster needs into one tarball:
# CM1 source (unmodified .F files + Makefile), the run template (member-1
# namelist, the unperturbed control sounding, lookup tables, and member 1's
# sounding for a validation run) and this kit.
# Usage: bash make_bundle.sh [/path/to/cm1data/cm1r21.1] [out.tar.gz]
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
CM1=${1:-$KIT/../../cm1data/cm1r21.1}; CM1=$(cd "$CM1" && pwd)
OUT=${2:-$CM1/../cm1_bundle.tar.gz}
T=$(mktemp -d); B=$T/cm1_bundle; mkdir -p $B/src $B/template $B/kit
cp "$CM1"/src/*.F "$CM1"/src/Makefile $B/src/
cp "$CM1"/run_e1/namelist.input $B/template/
cp "$CM1"/run/input_sounding $B/template/input_sounding
cp "$CM1"/run_e1/input_sounding $B/template/input_sounding_member1
for f in $(python3 -c "import sys; sys.path.insert(0,'$KIT'); import make_ensemble as m; print(' '.join(m.TABLES))"); do
  cp "$CM1"/run_e1/$f $B/template/
done
cp "$KIT"/*.sh "$KIT"/*.py "$KIT"/*.slurm "$KIT"/README.md $B/kit/ 2>/dev/null || true
mkdir -p $B/kit/patches && cp "$KIT"/patches/*.py $B/kit/patches/
tar -C $T -czf "$OUT" cm1_bundle
rm -rf $T
ls -lh "$OUT"
