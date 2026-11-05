# Training the CM1 model on Saidie (GPU)

**Not yet tested.** Saidie refused ssh connections at the time of writing; the same code is
tested on the Mac (`tests/unit/test_cm1.py`, `train_cm1.py --profile smoke` and `local`).

Saidie (Engineering DGX, 8 x H200) is the place for training; Cashew is the place for CM1 runs
(CPU). The training code needs only numpy and torch.

1. **Code** (Mac): `bash cluster/saidie/sync_code.sh`. This copies `ks_latent/`, `scripts/cm1/`, this kit and the CM1 tests to `~/latent_DA` on Saidie. Re-run it after code changes.
2. **Environment** (Saidie, once): `bash ~/latent_DA/cluster/saidie/setup_env.sh`.
3. **Account and partition**: run `sinfo` and `sacctmgr -nP show assoc user=$USER format=account,partition`, then fill `FILL_ACCOUNT` / `FILL_PARTITION` in `train_cm1.slurm`. Use one GPU per job; the node is shared.
4. **Data**: put the ensemble (`members/mem*/cm1out_s.ctl`, from `cluster/cm1`) on Saidie. If Saidie and Cashew share a filesystem, nothing needs copying. Otherwise copy Cashew -> Saidie, **only when Dalton says so**.
5. **Train**:
   ```bash
   mkdir -p ~/latent_DA/logs && cd ~/latent_DA
   sbatch --export=ALL,DATA=$HOME/cm1_bundle/ens_A,RUN=ens_A_v1 cluster/saidie/train_cm1.slurm
   ```
   The job does three things:
   - runs the CM1 unit tests first;
   - trains Stage A then Stage B (20 000 steps each by default);
   - writes `~/cm1_runs/ens_A_v1/`: checkpoints, `results.json`, figures, `log.txt`.

   With 1-min frames, the propagator reach is set automatically to 3 sites (6 km) per step: 25 m/s x 60 s rounded up, plus a margin, at 2 km per site, as 3 layers of radius 1. `k_max` is 20 steps.
6. **Watch**: `squeue -u $USER`, `tail -f ~/latent_DA/logs/cm1train_*.out`.
