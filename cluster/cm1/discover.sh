#!/bin/bash
# Print what the cluster offers for building and running CM1. Read-only; run
# on the login node (it only queries, it computes nothing). Paste the output
# back to Claude so the job script can be filled in.
echo "== host / OS";            hostname; uname -srm; cat /etc/os-release 2>/dev/null | head -2
echo "== scheduler";            which sbatch squeue sinfo 2>&1 | head -3
echo "== partitions (name, time limit, nodes, CPUs, memory MB, GPUs)"
sinfo -o "%P %l %D %c %m %G" 2>&1 | head -20
echo "== my Slurm accounts";    sacctmgr -nP show assoc user=$USER format=account,partition,qos 2>&1 | head -20
echo "== compilers on PATH";    for c in gfortran ifort ifx nvfortran mpif90; do printf "%-10s " $c; which $c 2>/dev/null || echo "-"; done
echo "== NetCDF on PATH";       for c in nf-config nc-config; do printf "%-10s " $c; which $c 2>/dev/null || echo "-"; done
echo "== modules matching compilers / netcdf / python"
( module -t avail 2>&1 || true ) | grep -i -E "gcc|gnu|fortran|intel|nvhpc|netcdf|hdf5|python|conda|anaconda" | sort -u | head -60
echo "== python3";              which python3 && python3 --version
echo "== disk";                 df -h $HOME 2>/dev/null | tail -1; ls -d /scratch* /work* 2>/dev/null
