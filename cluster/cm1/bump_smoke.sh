#!/bin/bash
# Smoke test of the bump_restart_var5 patch: base run writes a restart at 300 s; two branches restart from it to 480 s, one
# with var5 = 0 (control), one with a 0.5 K bump; a second control checks bitwise reproducibility. Run inside a Slurm job:
#   sbatch -p coe_cpu --cpus-per-task=4 --mem=8G -t 00:40:00 -o logs/bump_smoke_%j.out cluster/cm1/bump_smoke.sh
set -e
cd ~/cm1_bundle
export KIT_DIR=$HOME/cm1_bundle/kit; . kit/env.sh
rm -rf build_l3
NONETCDF=1 JOBS=4 bash kit/build_cm1.sh src build_l3 > logs/l3build_make.log 2>&1
cat build_l3/PATCHES
C="--template template --exe build_l3/run/cm1.exe --n 1 --seed 9 --timax 360 --tapfrq 60 --output-format 1 --noise-seed member --noise-offset 6000"
rm -rf ens_L3smoke; python3 kit/make_ensemble.py $C --out ens_L3smoke
B=ens_L3smoke/members/mem001
sed -i 's/^ *rstfrq *=.*/ rstfrq = 300.0,/' $B/namelist.input
(cd $B && ./cm1.exe > cm1.log 2>&1); grep -E "Program terminated normally" $B/cm1.log; ls $B | grep -i rst
for br in ctrl ctrl2 bump tiny; do
  D=ens_L3smoke/members/br_$br; rm -rf $D; mkdir -p $D; cp $B/cm1.exe $B/namelist.input $B/input_sounding $B/RRTMG* $D/ 2>/dev/null || cp $B/cm1.exe $B/namelist.input $B/input_sounding $D/
  cp $B/*lookup* $D/ 2>/dev/null || true
  cp $B/cm1rst_* $D/
  sed -i 's/^ *irst *=.*/ irst = 1,/; s/^ *rstnum *=.*/ rstnum = 1,/; s/^ *timax *=.*/ timax = 480.0,/; s/^ *rstfrq *=.*/ rstfrq = -900.0,/' $D/namelist.input
  if [ $br = bump ]; then sed -i 's/^ *var5 *=.*/ var5 = 0.5,/; s/^ *var6 *=.*/ var6 = 800.0,/; s/^ *var10 *=.*/ var10 = 2000.0,/; s/^ *var11 *=.*/ var11 = 6.0,/' $D/namelist.input; fi
    if [ $br = tiny ]; then sed -i 's/^ *var3 *=.*/ var3 = 0.001,/' $D/namelist.input; fi
  (cd $D && ./cm1.exe > cm1.log 2>&1 || true) &
done; wait
for br in ctrl ctrl2 bump tiny; do echo "== $br"; grep -E "bump added|tiny perturbation added|Program terminated normally|restart_read time" ens_L3smoke/members/br_$br/cm1.log; ls ens_L3smoke/members/br_$br | grep cm1out_s; done
cmp ens_L3smoke/members/br_ctrl/cm1out_s.dat ens_L3smoke/members/br_ctrl2/cm1out_s.dat && echo "CONTROL REPRODUCES BITWISE" || echo "CONTROL NOT REPRODUCIBLE"
cmp -s ens_L3smoke/members/br_ctrl/cm1out_s.dat ens_L3smoke/members/br_bump/cm1out_s.dat && echo "BUMP IDENTICAL (patch inactive?)" || echo "bump differs (expected)"
cmp -s ens_L3smoke/members/br_ctrl/cm1out_s.dat ens_L3smoke/members/br_tiny/cm1out_s.dat && echo "TINY IDENTICAL (patch inactive?)" || echo "tiny differs (expected)"
