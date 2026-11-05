# CM1 plan: more data on Cashew → Saidie → a set of trained models → latent DA

Status as of 2026-11-05.
- Findings and numbers live in `docs/cm1_notes.md`; this file is the to-do list. We tackle one task at a time.
- Each task has an owner (**D** = Dalton, **C** = Claude), its dependencies (`deps`), and a "done when" check.
- **Standing rule:** ensemble data never moves between machines without Dalton's go-ahead (marked 🔒).

## Where we stand
- **Data.**
  - The original 10 members are at 5-min output.
  - `ens_A` is done on Cashew: 31 members, 6 h at 1-min output, 64 GB of GrADS. A float16 cache of it is in `~/cm1_cache` on Cashew, about 15 GB.
  - All members share one environment.
- **Model.** `ks_latent/cm1`: a column local field (2 km sites) plus a local stencil propagator, with forced-halo training. Best so far (`ens_A_cpu_k45`, trained on Cashew CPU) has forecast nMSE 0.105 / 0.217 / 0.249 at 5 / 30 / 60 min. That is below the truth-member spread (~0.26) for the whole hour. The front error is ≤0.8 km for the first 20 min and ≤3.3 km over the hour.
- **Infrastructure.**
  - Cashew runs CM1 ensembles and CPU training with one command each.
  - Saidie is reachable without a password: account `chen-saidie`, partitions `general` (2 d) and `long-runs` (7 d), one node with 8 × H200, 224 CPUs and 1.8 TB RAM. Saidie and Cashew share no filesystem, but Saidie can ssh to Cashew.
  - Saidie is set up: code, environment and job scripts are in place, and the tests and smoke run pass on an H200 (section A).

---

## A. Saidie setup — done 2026-11-05
- [x] **A1 (C)** Saidie job scripts in `cluster/saidie/`: `train_cm1.slurm`, `gpu_check.slurm` and `smoke.slurm`.
  - Settings: account `chen-saidie`, partition `general`, `--gres=gpu:h200:1`, login shell (`#!/bin/bash -l`).
  - **Saidie shows all 8 GPUs to every job.** Each script prints `CUDA_VISIBLE_DEVICES` and refuses to run if Slurm did not set it, so torch can only use the assigned GPU. In tests Slurm set it (GPU 2).
- [x] **A2 (C)** Code synced to `~/latent_DA` on Saidie (`bash cluster/saidie/sync_code.sh saidie`, code only). Re-run it after code changes.
- [x] **A3 (C)** Environment `~/venvs/cm1`: torch 2.11.0+cu128, numpy, matplotlib, pytest.
  - **Driver 570.172 supports CUDA ≤ 12.8.** PyPI's default torch (CUDA 13.2) would not run, so `setup_env.sh` takes `TORCH_INDEX=https://download.pytorch.org/whl/cu128`.
  - **Saidie's system Python has no `ensurepip`.** The script uses the `python312` module.
  - GPU job 8677: `cuda available: True`, NVIDIA H200 (143 GB).
- [x] **A4 (C)** GPU job 8679: 15 CM1 tests pass and the smoke profile runs on `cuda` (synthetic data, since no CM1 data is on Saidie yet), writing `results.json` and the checkpoints.
  - Fixed along the way: the smoke profile had forced CPU even with `--device cuda`.
- [x] **A5 (C)** New data sources in `train_cm1.py`:
  - `cache:DIR[:PATTERN]` reads the float16 member files in natural member order (`mem2` before `mem10`), so the train/val/test split is the intended one;
  - `synthetic` provides random fields for code-path tests where no data exist.

  Unit tests pass. A run on the local cache reproduces `local_v1`'s split and scales exactly.
  - On Saidie, train with `DATA="cache:$HOME/cm1_cache"` once the ens_A cache is copied there (C1/C2).

## B. More data on Cashew
- [ ] **B1 (C)** Validate the Cashew build: `mem000` against the original `run_e1`, at matching 5-min times, as a job on Cashew.
  - Done when: a short note in `cm1_notes.md` with the RMS difference over time. It should look like two runs whose tiny differences grow, not a different model.
- [ ] **B2 (C)** Per-member random-noise seed.
  - Use CM1's spare namelist variable `var1` as a seed offset in `init3d.F`. Members can then differ in the bubble noise as well as the sounding.
  - Done when: two 5-min test runs with different `var1` differ at t=0, and identical `var1` gives bitwise-equal output.
- [ ] **B3 (D+C)** Decide the next ensembles. Proposal:
  - `ens_A2`: 30 more members in the same environment, with new noise seeds. Its purpose is a 1-min learning curve: 14 / 28 / 56 members.
  - `ens_B`: 60 members in varied environments, low-level moisture offset −1.5…+1.0 g/kg and shear scale 0.6…1.4. Its purpose is generalization across environments, holding out whole environment regions.

  Done when: Dalton approves the design (sizes, ranges, output interval 60 s).
- [ ] **B4 (C)** Pilot for `ens_B`: 4 members at the corners of the environment ranges, 6 h.
  - Check that the storm stays inside the domain (the grid moves at a fixed 12.5 m/s) and that no member fails.
  - deps: B2, B3. Done when: storm tracks are plotted and the ranges are confirmed or narrowed.
- [ ] **B5 (D or C)** Launch `ens_A2` and `ens_B` with `launch_cashew.sh`, using a new `ENS` name and `SEED` for each.
  - Respect the 20-job cap; run the two ensembles one after the other.
  - deps: B4. Done when: `status.sh` shows all members done.
- [ ] **B6 (C)** Build the float16 member caches on Cashew, as a CPU job.
  - deps: B5. Done when: one `.npz` per member, with sizes and frame counts checked.

## C. Moving data to Saidie 🔒
- [ ] **C1 (D)** Go-ahead to copy the caches from Cashew to Saidie.
  - About 15 GB for `ens_A`, then about 45 GB more for `ens_A2` and `ens_B`.
- [ ] **C2 (C)** Copy them with `rsync` run on Saidie, pulling from Cashew. That needs either your agent forwarded (`ssh -A`) or a key on Saidie for Cashew.
  - Done when: file counts and checksums match on both sides.

## D. A set of models on Saidie
Each task is a job script with 1 GPU per run and at most ~4 GPUs at once, since the node is shared. Results go to `~/cm1_runs/<name>`, and summaries are copied to `artifacts/cm1/runs/`.

- [ ] **D1 (C)** Reproduce `ens_A_cpu_k45` on the GPU.
  - deps: A4, A5, C2. Done when: nMSE is within 0.01 of the Cashew run at 5, 30 and 60 min.
- [ ] **D2 (C)** Seeds: 3 seeds of the D1 recipe.
  - Done when: a mean ± spread table. Every later comparison is judged against this spread.
- [ ] **D3 (C)** Rollout curriculum: a mix of short and long rollouts, to keep the 1–5 min skill of k_max 20 and the 60-min skill of k_max 45.
  - Done when: better than both at all leads, or the trade-off is documented.
- [ ] **D4 (C)** Capacity sweep: latent channels 16 / 32, sites 2 km / 1 km, propagator width 256 / 512.
  - Done when: a table of reconstruction error and forecast nMSE at 5 / 30 / 60 min.
- [ ] **D5 (C)** Learning curve at 1-min: train on 7 / 14 / 28 / 56 members, test on fixed held-out members.
  - deps: `ens_A2`. Done when: a plot, and whether forecast skill is still rising at 56 members.
- [ ] **D6 (C)** Physics checks:
  - mass divergence ∂(ρu)/∂x + ∂(ρw)/∂z of decoded states against the truth;
  - non-negativity and totals of condensate;
  - cold-pool strength and depth.

  Done when: added to `evaluate.py`, with a test against an analytic field.
- [ ] **D7 (C)** Ablation: one encoder per variable against the joint column encoder.
  - Done when: reconstruction and forecast tables. Do the variables need to be encoded jointly?
- [ ] **D8 (C)** Generalization across environments: train on `ens_B` excluding one region of moisture × shear, test inside it.
  - deps: `ens_B`. Done when: error in held-out environments against in-distribution ones.

## E. Latent DA twin experiments (the Paper III core)
- [ ] **E1 (C)** Observation operators.
  - (a) Surface θ′ at selected columns: linear in the decoded state.
  - (b) Reflectivity: a small learned column head from decoded condensate to dBZ, trained against CM1's own `dbz`. Its error is measured.
  - Done when: both operators are tested, with the reflectivity head's error reported.
- [ ] **E2 (C)** Per-site latent LETKF for open boundaries: adapt `ks_latent/da/letkf_local.py`, which assumes periodic rolls, and use the true state at the domain edges as boundary conditions.
  - Done when: in a linear-Gaussian test, the analysis matches the Kalman answer, as in the KS tests.
- [ ] **E3 (C)** Twin experiment 1: a held-out member is the truth, surface θ′ every 2 km every 5 min is observed, with a 30-member latent ensemble.
  - Done when: analysis and forecast error against a free run and against the truth-member spread, consistency ratio, rank histograms.
- [ ] **E4 (C)** Twin experiment 2: add reflectivity observations (nonlinear, bounded). Report the nonlinearity index from Paper II.
  - deps: E1b, E3. Done when: same table as E3, plus the nonlinearity index.
- [ ] **E5 (D+C)** Decide the Paper III framing from E3–E4: what is claimed, against which baselines.

## F. Writing
- [x] **F1 (C)** `docs/cm1_summary/cm1_summary.tex`: a short illustrated summary of what we have found so far.
- [ ] **F2 (C)** Update the summary after each of B1, D2, D5, D8 and E3.
