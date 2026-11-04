# Running the CM1 squall-line ensemble on CSU's Cashew cluster

This kit generates more members of the 2D CM1 squall line in `cm1data/`. The setup is the same as before; only the run length, output interval and perturbations change.

- **Why Cashew.** CM1 is CPU-only Fortran, so GPUs don't help. Cashew's default partition (`coe_cpu`) is the general College of Engineering CPU pool.
- **What a member costs.** One CPU, about 0.4 GB of memory, and an estimated 1–2 h for 6 simulated hours. On the Mac, 90 min took 7.6 min, and cluster cores may be slower.
- **The job cap.** Cashew allows at most **20 submitted plus pending jobs per partition**. The kit therefore packs several members into each job (one member per CPU) instead of one job per member.

Everything was tested end to end on the Mac:
- bundle, build, member generation, packed submission (with a stand-in for `sbatch`), status and resubmission;
- the validation member (`mem000`) reproduces member 1's initial state bit for bit.

Things the kit cannot know are marked **FILL**: the login host name and the module names.

---

## 0. Once: get access
- **Eligibility.** Engineering students are pre-approved. Ask ETS to add you to the cluster users group (the request form on the Cashew page). Faculty and staff need to own a node.
- **VPN.** Off campus, connect to the CSU VPN first. Both ssh and the OnDemand web portal need it.
- **Login host.** Find the host name in the Cashew page's "Connecting to the Cluster" section. Below it is written `CASHEW_HOST` (**FILL**). `USER` is your CSU eID.

## 1. On the Mac: make the bundle (about 1 min, 327 MB)
```bash
cd ~/Documents/latent_DA
bash cluster/cm1/make_bundle.sh          # writes cm1data/cm1_bundle.tar.gz
```
The bundle holds:
- the unmodified CM1 source;
- the run template: member 1's namelist, the unperturbed base sounding, member 1's sounding (for validation) and the lookup tables;
- this kit.

## 2. Copy it to Cashew and unpack
```bash
scp cm1data/cm1_bundle.tar.gz USER@CASHEW_HOST:~/
ssh USER@CASHEW_HOST
tar xzf cm1_bundle.tar.gz && cd cm1_bundle      # all later commands run from here
```

## 3. See what the cluster offers (read-only, fine on the login node)
```bash
bash kit/discover.sh | tee discover.txt
overview                                         # Cashew's own node/queue summary
```
From `discover.txt`, pick a **Fortran compiler module** (gcc/gfortran preferred) and a matching **netcdf-fortran module** (it provides `nf-config`). If unsure, paste `discover.txt` to Claude.

## 4. Build CM1 inside an interactive job (about 2–5 min). Never build on the login node.
```bash
srun --cpus-per-task=2 --mem=4G --time=01:00:00 --pty bash
module load FILL_GCC_MODULE FILL_NETCDF_FORTRAN_MODULE
JOBS=2 bash kit/build_cm1.sh src build           # -> build/run/cm1.exe
```
- The script finds NetCDF through `nf-config` and bakes the library path into `cm1.exe`, so jobs don't need `module load` later.
- If NetCDF won't cooperate, `NONETCDF=1 bash kit/build_cm1.sh src build` builds a version that writes GrADS binary instead. Then add `--output-format 1` in steps 5 and 6. The Mac can read both formats.

## 5. Smoke test, still inside the interactive job (about 1 min)
```bash
python3 kit/make_ensemble.py --template template --exe build/run/cm1.exe \
    --out ens_smoke --n 1 --seed 0 --timax 300 \
    --reference-sounding template/input_sounding_member1
bash kit/run_member.sh ens_smoke/members/mem000
```
- **Expected:** `.../mem000 finished in N s`, and `ens_smoke/members/mem000/cm1out.nc` exists.
- **Cost estimate:** a 6 h member costs roughly 72 × N s, more once the storm fills the domain. On the Mac, N was 26 s.

Then leave the interactive job with `exit`.

## 6. Generate the main ensemble (Phase A: original environment, 1-min output)
```bash
python3 kit/make_ensemble.py --template template --exe build/run/cm1.exe \
    --out ens_A --n 30 --seed 1 \
    --reference-sounding template/input_sounding_member1
```
This makes 31 members:
- `mem000` is a validation copy of the existing member 1. We compare it with the original at matching times to check the cluster build.
- `mem001`–`mem030` are new members, perturbed the way the original 10 were: independent θ perturbations with σ = 0.1 K at the surface and below 500 m.

Every member runs 6 h and writes output every 60 s. Settings and random draws are recorded in `ens_A/manifest.json` and `manifest.csv`.

## 7. Submit
```bash
bash kit/submit_ensemble.sh ens_A 8 --mail-type=END --mail-user=FILL_EMAIL
```
- This makes 4 jobs of up to 8 CPUs each (31 members). Each job runs its members side by side, so a job takes about as long as one member.
- Use a larger second argument for fewer, bigger jobs, e.g. 16 makes 2 jobs. The script refuses if your queued jobs would exceed 18, leaving headroom under the cap of 20.
- There is no `--partition`: the default `coe_cpu` is right. Don't use `all`, because jobs there are killed and requeued when a group's own jobs need the node.

## 8. Watch
```bash
squeue -u $USER          # or: overview
bash kit/status.sh ens_A # done / failed / not finished, run times, disk use
tail logs/cm1_*.out
```

## 9. If a member fails or a job times out
```bash
bash kit/status.sh ens_A                    # lists FAILED members
tail -40 ens_A/members/mem0XX/cm1.log       # CM1's own error message
bash kit/submit_ensemble.sh ens_A 8         # resubmits only unfinished members
```
- A member counts as done only when CM1 prints "Program terminated normally". Finished members are never rerun.
- If jobs hit the 8 h limit, edit `#SBATCH --time` in `kit/cm1_pack.slurm`.

## 10. Copy results back to the Mac
From the Mac, with the VPN on if off campus:
```bash
bash cluster/cm1/fetch_results.sh USER@CASHEW_HOST cm1_bundle/ens_A    # -> cm1data/ens_A/
```
- **What's copied:** only outputs, namelists, soundings, logs and the manifest.
- **Size:** about 0.6–1 GB per member, so roughly 20–30 GB for `ens_A`. The Mac has 669 GB free.
- **Large transfers:** for much bigger ensembles, ask ETS about the Globus data transfer node.

`cm1data/` is ignored by git, so nothing lands in the repository.

---

## Later: Phase B, varied environments (for testing generalization)
```bash
python3 kit/make_ensemble.py --template template --exe build/run/cm1.exe \
    --out ens_B --n 60 --seed 2 --qv-offset -1.5 1.0 --shear-scale 0.6 1.4
```
- `--qv-offset` shifts boundary-layer moisture, in g/kg below 1 km, tapered to zero by 1.5 km.
- `--shear-scale` multiplies the wind profile.
- Any member whose sounding would be supersaturated is refused.
- Caveat: the grid moves at a fixed 12.5 m/s. Strongly changed shear changes the storm's speed, so the storm may drift toward a boundary over 6 h. Check the first few members before running many.

## Notes
- **CM1's random noise uses a fixed seed** (`use_truly_random_pert = .false.` in `src/init3d.F`). All members get the same noise; they differ through the sounding, as the original 10 do.
- **Output dropped by default:** subgrid TKE, eddy viscosities, staggered u/v/w, and v (zero in 2D). Use `--no-trim-output` to keep them.
- **Be good to each other:** the cluster is shared. 31 members use 31 CPUs for 1–2 h.
