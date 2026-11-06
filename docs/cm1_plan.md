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
- **DA so far (E3/E4):** latent LETKF with θ′ + reflectivity cuts forecast nMSE 0.329 → 0.271, the front error to ~0.2 km and echo reflectivity error 7.7 → 4.3 dBZ (one truth, one seed). The path to a publishable Paper III is section G.
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
- [x] **B1 (C)** Validate the Cashew build — done 2026-11-05 (Cashew job 745020, `scripts/cm1/validate_cashew_build.py`, figure `docs/cm1_summary/figs/validate_build.pdf`).
  - `mem000` is **bitwise identical to `run_e1` at t=0**.
  - Build-level differences start **~50× smaller than the difference between two original members**: nMSE 1.7e-8 vs 8e-7 at 5 min, 6e-6 vs 2e-4 at 30 min, 4e-3 vs 6e-2 at 1 h. They then grow chaotically and reach member-to-member size by ~2 h.
  - Gust fronts agree within 1 km for 3 h.
  - **Verdict:** same model, not a bit-copy. A rebuilt run is effectively a new member after ~2 h.
- [x] **B2 (C)** Per-member noise seed — done 2026-11-05.
  - The patch `cluster/cm1/patches/apply_patches.py` (seed_var1) is applied by `build_cm1.sh` to its own copy of the source and recorded in `build/PATCHES`. The original source in `cm1data/` is untouched.
  - Namelist `var1` ≠ 0 scrambles CM1's noise seed. That seed drives **both** the bubble noise (±0.2 K) **and** a domain-wide ±0.25 K θ noise (`irandp = 1`). The original 10 members and `ens_A` all share one noise field.
  - `make_ensemble.py --noise-seed member` sets `var1` to the member number (`mem000` keeps 0). It refuses builds without the patch.
  - Tests:
    - `var1=0` reproduces run_e1's t=0 bitwise;
    - `var1=1` vs `2` differ by up to 0.78 K (max possible 0.9 K);
    - the same `var1` twice is bitwise identical in every field and time.
  - Note: Cashew's existing build (`~/cm1_bundle/build`) predates the patch. Delete it before a seeded ensemble, and `go.sh` rebuilds it.
- [x] **B3 (D+C)** Decide the next ensembles. **Approved by Dalton 2026-11-05 as proposed.** Noise-seed offsets keep ensembles apart: ens_A2 1000+, ens_B 2000+, pilot 3000+.
  - **`ens_A2`: 30 members, same environment as ens_A, per-member noise seeds (B2).**
    - Purpose: 1-min learning curve to 56–58 members (D5), and more independent truths for DA.
    - Caveat: these differ from ens_A in noise as well as sounding, so they are slightly more varied. Report learning-curve points by ensemble.
  - **`ens_B`: 60 members in varied environments, with per-member noise seeds.**
    - Low-level moisture offset uniform in **−2.0…+1.0 g/kg**. **+1.0 is the limit**: +1.5 supersaturates at shear 1.4, and the generator refuses it.
    - Shear scale uniform in **0.6…1.4**.
    - Hold out the high-moisture, high-shear corner (offset > +0.5 and scale > 1.2) for testing generalization (D8).
  - **Common settings:** 6 h, 60 s output, the original θ perturbations (σ = 0.1 K below 500 m).
  - **Cost:**
    - 90 members × ~1.5 h on 1 core each. In 8-CPU packs that is 12 jobs, under Cashew's 20-job cap, and about 1.5–3 h wall if the CPU nodes are free.
    - Storage: ~200 GB raw on Cashew (~74 GB used of 1 TB) and ~45 GB of float16 caches to copy to Saidie (2 TB, 15 GB used).
- [x] **B4 (C)** Pilot for `ens_B` — **done 2026-11-05. Range narrowed to qv −2.0…+0.25 g/kg, shear 0.6…1.4.** 4 members at the corners of the environment ranges, 6 h.
  - **Pilot 1 done 2026-11-05** (`ens_B_pilot`, job 745140, 73–87 min per member; `scripts/cm1/pilot_check.py`, `artifacts/cm1/pilot_check.json`):
    - **qv −2.0:** a single squall line at both shears.
      - Shear 0.6: slower than the 12.5 m/s grid; the front stays at 160–180 km, and only the trailing stratiform echo comes within 30 km of the west edge after 3.5 h, which is harmless with open boundaries.
      - Shear 1.4: faster; the front reaches 298 km by 6 h.
    - **qv +1.0 is NOT a squall line.** Echo covers 0–400 km within 40–60 min: in a near-saturated boundary layer (max RH 0.99), CM1's domain-wide ±0.25 K noise triggers convection everywhere. The upper end of the moisture range must come down.
  - **Pilot 2** (`ens_B_pilot2`, job 745484, 77–86 min per member):
    - **+0.25 g/kg is a squall line at both shears** (front 143–183 km and 200 → 244 km);
    - **+0.5 g/kg** is a squall line at shear 0.6 but breaks down by 2 h at shear 1.4 (echo 40–400 km, front at the east edge).
    - The weak-shear members' light trailing echo exits the open west boundary late; the gust fronts stay well inside, so this is harmless.
  - **Edge criterion refined:** what matters is the main-pool front and convective cores, not the trailing light echo.
  - Check that the storm stays inside the domain (the grid moves at a fixed 12.5 m/s) and that no member fails.
  - deps: B2, B3. Done when: storm tracks are plotted and the ranges are confirmed or narrowed.
- [~] **B5 (D or C)** **`ens_A2` done (30/30). `ens_B` launched 2026-11-05:** 60 members, qv −2.0…+0.25, shear 0.6…1.4, noise seeds 2001–2060, Cashew jobs 745725–745732. Hold-out region for D10: the strict corner (qv > 0, shear > 1.2) has only 4 members, so widen it (e.g. qv > −0.5, shear > 1.1, ~7–8 members). Earlier: **`ens_A2` launched 2026-11-05** (30 members, seeds 1001–1030, Cashew jobs 745136–745139); `ens_B` waits for the pilot. Launch `ens_A2` and `ens_B` with `launch_cashew.sh`, using a new `ENS` name and `SEED` for each.
  - Respect the 20-job cap; run the two ensembles one after the other.
  - deps: B4. Done when: `status.sh` shows all members done.
- [ ] **B6 (C)** Build the float16 member caches on Cashew, as a CPU job (`scripts/cm1/build_cache.py`, ensemble-prefixed names). **ens_A2: 30/30 members done; cache job 745513 running, then an automatic copy to `saidie:~/cm1_cache_A2` with checksums.**
  - deps: B5. Done when: one `.npz` per member, with sizes and frame counts checked.

## C. Moving data to Saidie 🔒
- [x] **C1 (D)** Go-ahead to copy the caches from Cashew to Saidie. Given 2026-11-05 for `ens_A`.
  - About 15 GB for `ens_A`, then about 45 GB more for `ens_A2` and `ens_B`.
- [x] **C2 (C)** `ens_A` cache copied — done 2026-11-05.
  - `rsync` ran on Saidie, pulling from Cashew, with the Mac's agent forwarded on a fresh connection (`ssh -A -o ControlPath=none`; the shared connection ignores `-A`).
  - 31 files, 15,613,803,542 bytes on both sides, all 31 md5 checksums identical. The data is in `saidie:~/cm1_cache`.
  - `ens_A2` and `ens_B` will need the same copy once they exist (and a go-ahead).

## D. A set of models on Saidie
Each task is a job script with 1 GPU per run and at most ~4 GPUs at once, since the node is shared. Results go to `~/cm1_runs/<name>`, and summaries are copied to `artifacts/cm1/runs/`.

- [–] **D1 (C)** Reproduce `ens_A_cpu_k45` on the GPU. **Dropped (Dalton, 2026-11-05).** A ~0.02 gap at 60 min between CPU and GPU runs is not unexpected.
  - The run (Saidie job 8680, `ens_A_gpu_k45`) serves as the GPU reference for D3/D4: nMSE 0.110 / 0.213 / 0.270 at 5 / 30 / 60 min.
  - Treat differences of ~0.02 at 60 min as within run-to-run noise.
- [–] **D2 (C)** Seeds. **Dropped (Dalton, 2026-11-05):** go straight to training.
- [x] **D3 (C)** Rollout curriculum — done 2026-11-05 (Saidie job 8681, `--k-sampling uniform`: each batch draws its rollout length in [2, curriculum max]).
  - Short leads recovered: nMSE 0.096 at 5 min vs 0.110 for the reference, at the k ≤ 20 level. The 60-min error is unchanged (0.267 vs 0.270).
  - Surface θ′ and front errors are slightly worse at 45–60 min.
- [x] **D4 (C)** Capacity sweep — done 2026-11-05 (jobs 8682–8685, 12–16 min each on one H200). nMSE at 1 / 5 / 15 / 30 / 45 / 60 min:

  | Run | 1 | 5 | 15 | 30 | 45 | 60 min |
  |---|---|---|---|---|---|---|
  | reference | 0.053 | 0.110 | 0.185 | 0.213 | 0.251 | 0.270 |
  | c = 32 | 0.048 | 0.106 | 0.174 | 0.201 | **0.229** | **0.245** |
  | 1 km sites | 0.057 | 0.107 | 0.166 | 0.199 | 0.240 | 0.256 |
  | width 512 | 0.046 | 0.099 | 0.186 | 0.237 | 0.270 | 0.283 |
  | **c = 32 + uniform k** | **0.040** | **0.092** | **0.166** | **0.196** | 0.240 | 0.262 |

  - **Verdict:** adopt **c = 32 with uniform k** as the default, since it is best at 1–30 min, the leads that matter for DA cycles. c = 32 with k at the maximum is best at 45–60 min, but within ~noise.
  - **Width 512 is worse beyond 15 min:** the model is data-limited, not capacity-limited, so more members should help more than more parameters (D5).
  - **1 km sites** improve reconstruction of w and condensate, with no clear forecast gain at 2× the sites.
  - **Caveats:**
    - single seed (D2 dropped), and differences ≤ 0.02 at 60 min are within run-to-run noise;
    - the gust-front error at 45–60 min (~2–3.5 km) was dominated by detector jumps. **Fixed 2026-11-05:**
      - new metrics in `ks_latent/cm1/verify.py`: main-pool front, threshold-free displacement, FSS;
      - all runs re-scored (job 8687): median displacement ≤ 0.5 km to 30 min and 0.8–1.7 km at 60 min, against 6 / 11.5 km for persistence;
      - c = 32 has the smallest worst cases. See `cm1_notes.md`.
- [x] **D5 (C)** **Done 2026-11-05. The learning curve flattens after ~28 storms in this environment.**
  - Corrected 58-member run: Saidie job 8706, 31 + 30 members loaded, 58 trained.
  - **Normalized errors are not comparable across runs with different training sets:** each fits its own per-variable scales (u 4.98 vs 5.15 m/s here). Compare skill relative to persistence, or physical units.
  - **Results, 28 vs 58 members:**
    - error / persistence at 5 / 30 / 60 min: 0.214 / 0.332 / 0.350 vs 0.206 / 0.328 / 0.347 (1–4% better);
    - surface θ′ RMSE at 30 / 60 min: 0.59 / 0.83 vs 0.64 / 0.89 K (slightly worse);
    - low-level-rain FSS at 30 / 60 min: 0.831 / 0.696 vs 0.849 / 0.714.

    All of this is within single-seed noise.
  - **Verdict:** big gains from 7 → 14 → 28, then essentially none. Same-environment data is no longer the bottleneck; environmental variety (ens_B) and model changes (G10, the cold-pool bias) are.
  - Earlier notes:
  - **Bug:** `DATA=cache:A,B` inside `sbatch --export` was cut at the comma, so every D5 run trained on ens_A only. The 58 point duplicated the 28 point. Fixed: `+` separates cache directories, and the log reports members per directory. The 58 run is resubmitted as job 8706; the invalid one is kept as `d5_train58_INVALID_onlyensA`.
  - **Valid points (subsets of ens_A; c = 32 + uniform k):**

    | Train members | nMSE 30 / 60 min | Low-level-rain FSS at 60 min | Condensate-mass error at 60 min |
    |---|---|---|---|
    | 7 | 0.271 / 0.328 | 0.48 | 27% |
    | 14 | 0.230 / 0.277 | 0.62 | 18% |
    | 28 | **0.196 / 0.262** | **0.70** | **7.5%** |

    Skill is still rising at 28 members (−15% at 30 min from 14 to 28), so more storms should keep helping.
  - Original task: learning curve at 1-min: train on 7 / 14 / 28 / 56 members, test on fixed held-out members.
  - deps: `ens_A2`. Done when: a plot, and whether forecast skill is still rising at 56 members.
- [x] **D6 (C)** Physics checks — done 2026-11-05 (`ks_latent/cm1/physics.py`; analytic tests; Saidie re-score job 8690). Best model `d34_c32_kuniform`:
  - **Mass continuity is respected.** The divergence ratio of forecasts is 0.23–0.25 against the truth's 0.21–0.22, and reconstructions 0.234 against 0.223. The 16-channel reference is worse (0.284).
  - **Condensate mass** relative error is 1.8 / 3.4 / 5.7 / 11.8 / 7.5% at 1 / 5 / 15 / 30 / 60 min. That is better than persistence only at 60 min, because the domain total changes slowly.
  - **Systematic bias: the cold pool does not strengthen.** Max C in the forecasts is 23–25 m/s while the truth's grows from 25 to 28 m/s, ending 17% too weak at 60 min. Track it through later training changes.
  - **FSS of low-level liquid (q > 1 g/kg, 1 km) discriminates where column-max condensate did not:** 0.93 / 0.84 / 0.83 / 0.70 at 5 / 15 / 30 / 60 min, against persistence's 0.82 / 0.61 / 0.59 / 0.22.

  Original list:
  - mass divergence ∂(ρu)/∂x + ∂(ρw)/∂z of decoded states against the truth;
  - non-negativity and totals of condensate;
  - cold-pool strength and depth.

  Done when: added to `evaluate.py`, with a test against an analytic field.
- [x] **D7 (C)** **Done 2026-11-05 (Saidie jobs 8688 and 8689; c 28, width 252, k ≤ 45 uniform). Joint encoding wins.**
  - **Reconstruction, storm-column RMSE:** u 1.02 vs 1.72 m/s, w 0.42 vs 0.83 m/s, θ′ 0.55 vs 0.78 K. The per-variable encoder is worst on the coupled variables.
  - **Forecast nMSE at 1 / 5 / 15 / 60 min:** 0.041 / 0.092 / 0.172 / 0.257 vs 0.071 / 0.118 / 0.192 / 0.273.
  - **Main-pool front error at 30 min:** 0.88 vs 1.50 km.
  - **Caveat:** the per-variable encoder has 7× fewer parameters (0.36 M vs 2.53 M), so part of the gap is capacity.
  - Originally: ablation: one encoder per variable against the joint column encoder.
  - Done when: reconstruction and forecast tables. Do the variables need to be encoded jointly?
- [ ] **D8 (C)** (deps: D10, D11) Generalization across environments: train on `ens_B` excluding one region of moisture × shear, test inside it.
  - deps: `ens_B`. Done when: error in held-out environments against in-distribution ones.

- [x] **D9 (C)** **Tested on ens_B (D16 S2/S3): mirrors hurt (+0.03 skill at 5 min, worse rain FSS); not in the recipe.** **Code done 2026-11-05:** `--mirror` adds the mirrored copy of each training member; `mirror_member` and `env.mirror_env` (u_m = 2·umove − u, umove 12.5 m/s) are tested involutions. The namelist has icor 0, free-slip bottom and no surface fluxes, so the symmetry is exact in the grid frame. Being tested in the ens_B screen (D16). Mirror-image augmentation, after `ens_B` exists (D8).
  - With icor = 0 the equations are exactly symmetric under x → −x, u → −u. A mirrored member is a valid storm in mirrored shear.
  - Add an option to `train_cm1.py` that adds mirrored copies of the training members: reverse x, negate u′, reverse `cref`. Test it with a unit test that mirroring twice is the identity.
  - On `ens_A` alone (5-min data, locality test), mirrors did *not* help predict the same environment, because they live in reversed shear.
  - With `ens_B`'s spread of shears, they may help generalization. Train D8's held-out-environment split with and without mirrors.
  - deps: D8. Done when: held-out-environment errors with and without mirrors, with a verdict.

- [x] **D10 (C)** **Done 2026-11-05 (`--holdout-env`, `--manifest`; used by H3).** **Environment-based split** for ens_B: `--holdout-env` reads each member's moisture offset and shear scale from the ensemble's `manifest.csv` and keeps a region entirely out of training and validation. Members are otherwise split by name, which would leak held-out environments into training. Needed before D8 and G5.
- [x] **D11 (C)** **Tested on ens_B (D16 S1): neutral on skill, worse rain FSS; not in the recipe.** **Environment conditioning. Code done 2026-11-05.**
  - `ks_latent/cm1/env.py`: features from each member's input sounding minus the reference, (θ, qv, u) on 62 levels, scaled 1 K / 1 g/kg / 5 m/s.
  - The encoder projects them (186 → 16) and broadcasts them as input channels, threaded through Stages A/C, the reports and the twin. `--env-file`, `scripts/cm1/build_env.py`.
  - Files `ens_A.json` and `ens_A2.json` are in `~/cm1_env` on Cashew and Saidie.
  - **Still to do for DA with conditioned models:** pass the environment in `da_twin.py` / `da_study.py` (they fail loudly until then).
  - Original task:
  - Problem: columns are perturbations from each member's own far-field t=0 column, which removes the base moisture and shear that distinguish ens_B storms.
  - Fix: append the member's base profiles (u₀(z), qv₀(z), θ₀(z)) as extra input channels, constant in x and not reconstructed, to the encoder and propagator; or condition the propagator on a small learned environment embedding.
  - Test on ens_A (should be neutral) and ens_B (should help). Needed for a fair D8/G5.
- [ ] **D12 (C)** **Round 1 (jobs 8719/8720, k ≤ 20, 4000 steps): skill only at 5 min (0.191 vs 0.214); 30/60 unchanged, fronts worse, the near-surface weight does not reduce the cold-pool bias. Retry with k ≤ 45, 15 000 steps: job 8753.** **Round 2 (8753): it works. Skill 0.187 / 0.316 / 0.310 vs 0.214 / 0.332 / 0.351, cold-pool bias −13.7% vs −17.3%, rain FSS 0.773 vs 0.696; 30-min front worse (0.69 vs 0.43 km). In the recipe; being checked on ens_B (D16).** **Stage C: joint fine-tuning on a decoded multi-step rollout loss** (encoder + propagator + decoder; error on physical columns over k steps).
  - Why: Stage B minimizes latent error, weighting latent directions equally rather than by physical importance. Decoded-space targets also avoid the documented latent-target collapse.
  - Keep the log-det anti-collapse term on the latents. Optionally up-weight θ′ below ~1 km against the cold-pool bias (G10).
  - Done when: skill vs persistence at 5 / 30 / 60 min and max-C bias, against `d34_c32_kuniform`.
- [ ] **D13 (C)** **Round 1 (job 8721): best long-lead fronts (0.72 vs 1.21 km at 60 min), skill 0.345 vs 0.351 at 60, slightly worse at 30. Averaging 4 history propagators: jobs 8754–8757.** **Round 2: the history average (0.192 / 0.319 / 0.332) is no better than the plain average; dropped.** **History in the propagator:** 2–3 past frames as input, reusing the codebase's history-mode propagator ideas. Unresolved sub-site convection appears as memory (Mori–Zwanzig).
  - Done when: same table as D12.
- [ ] **D14 (C)** **Round 1:**
  - (a) **averaging 4 propagators is the best single idea:** skill 0.193 / 0.315 / 0.330 vs 0.214 / 0.332 / 0.351;
  - (c) the stochastic CRPS propagator is worse on mean error but halves the cold-pool bias (−10% vs −17%);
  - (b) the vertical front end is neutral.

  Details in `cm1_notes.md`. **Lower-cost improvements:**
  - (a) average the forecasts of several independently trained propagators (the 4 E3b ones exist), as a forecast ensemble mean;
  - (b) vertical 1D convolutions within each variable's column before the encoder mixes channels;
  - (c) a stochastic propagator trained on CRPS, which learns its own spread for DA and usually lowers ensemble-mean error.

  Report each against the same baseline.
- [–] **D15 (C)** **Deferred (Dalton, 2026-11-05): saved for later; the split-cache build was cancelled. The code stays in place.** **Rain, snow and graupel as separate model variables** (added 2026-11-05).
  - Why: the 7-variable state lumps cloud water + rain into `qliq` and cloud ice + snow + graupel into `qice`, so the model cannot forecast rain or snow separately, and the reflectivity head has to guess the species mix. Hydrometeor species are also what a radar operator and a precipitation forecast need.
  - State: 10 variables, `VARS_SPLIT = (u, w, θ′, qv′, p′, qc, qr, qi, qs, qg)`, each condensate species as sqrt(g/kg); 620 channels instead of 434. The first five variables are unchanged.
  - **Code: done.** `io.transform(split=True)`, `data.vars_for(V)` (picks the set from the data's shape), `Normalizer.vars`, positivity mask, reports, physics (condensate sums all species present), twin scores, `plot_sections.py` styles for qc/qr/qi/qs/qg, `build_cache.py --split`. Test `test_split_species_pipeline_and_physics`: the species sums reproduce the 7-variable condensate mass and low-level liquid exactly; on the 1-min scratch run the split recombines to 2e-7.
  - [ ] Build split caches from the raw GrADS on Cashew (`~/cm1_cache_split`, `~/cm1_cache_A2_split`; ens_B once done), copy to Saidie.
  - [ ] Train `d15_split` with the `d34_c32_kuniform` recipe (c=32, uniform k ≤ 45) on the split caches.
  - Done when: skill vs persistence at 5 / 30 / 60 min against `d34_c32_kuniform`, scored on the 7 combined variables (sum the species back) so the comparison is like for like; per-species RMSE and storm-column FSS for qr and qs; rain and snow section figures. Watch for: rain and graupel are sparse and fast-falling, so the per-variable RMS scaling may give them too little weight (try a floor on the scale), and c=32 may now be too few latent channels (try c=48).

- [ ] **D16 (C)** **Screen done (8761–8765): Stage C wins (0.237 / 0.383 / 0.370 vs 0.265 / 0.400 / 0.385); environment conditioning neutral; mirrors hurt. Full run with d34 recipe + Stage C: jobs 8797 → 8798. Stochastic + Stage C on ens_A: job 8794.** **Pick the best model on ens_B, then one full run on all the data** (added 2026-11-05, Dalton: pick the best model before the full run).
  - **Screen on ens_B alone** (60 storms over moisture −2.0…+0.2 g/kg × shear 0.63…1.40), where the environment varies. The split is explicit (`--val-members` / `--test-members`) and spread over a 3 × 3 grid of moisture × shear:
    - test (10): ens_B mem003, 049, 016 (dry: low / mid / high shear), 054, 022, 017 (middle), 050, 030, 011 (moist), and 044 (the moist, high-shear corner);
    - val (4): mem025, 053, 055, 058;
    - train: the other 46.
  - Runs, all with the `d34_c32_kuniform` recipe (c 32, k ≤ 45 uniform): S0 baseline; **S0 + Stage C** (rollout fine-tuning, k ≤ 45, 15 000 steps, as job 8753 on ens_A); S1 + environment conditioning (D11); S2 + mirrors (D9); S3 both.
  - In parallel on the other GPU: round 2 on ens_A (Stage C k ≤ 45, job 8753; history propagators averaged, 8754–8757).
  - Then pick the recipe: the best of S0–S3, plus Stage C or history if round 2 shows a real gain. Averaging 4 propagators (D14a) is applied to the final model either way.
  - **Full run:** ens_A + ens_A2 + ens_B (121 storms). Test: ens_A mem029–030, ens_A2 mem001–010 (the G1 DA truths, so DA results stay paired) and the 10 ens_B test storms. Val: ens_A mem028, ens_A2 mem011 and the 4 ens_B val storms.
  - Done when: skill vs persistence at 5 / 30 / 60 min, cold-pool bias and fronts for S0–S3 in one table, and the full run scored on the same ens_B test storms.

## E. Latent DA twin experiments (the Paper III core)
- [x] **E1 (C)** Observation operators — done 2026-11-05.
  - (a) `obs.py SurfaceTheta`.
  - (b) `refl.py ReflectivityHead`: a per-column net to composite reflectivity, trained on true columns of the training members (Saidie job 8696, 8000 steps). Test RMSE is 1.34 dBZ on true columns and 1.63 on decoded columns (echo columns: 2.00 / 2.47).
  - (a) Surface θ′ at selected columns: linear in the decoded state.
  - (b) Reflectivity: a small learned column head from decoded condensate to dBZ, trained against CM1's own `dbz`. Its error is measured.
  - Done when: both operators are tested, with the reflectivity head's error reported.
- [x] **E2 (C)** Per-site latent LETKF — done 2026-11-05 (`ks_latent/cm1/letkf.py`).
  - Method: Hunt et al. 2007, in ensemble space, R-localization with Gaspari–Cohn, plain distances (open boundaries, no wrap), inflation, RTPS. No Jacobian, so it works with nonlinear operators and decoder reach > 0.
  - Tests:
    - equals the Kalman update on the sample covariance to 1e-10 without localization;
    - sites beyond 2c are untouched, including the far edge;
    - RTPS restores the prior spread.

  Originally: adapt `ks_latent/da/letkf_local.py`, which assumes periodic rolls, and use the true state at the domain edges as boundary conditions.
  - Done when: in a linear-Gaussian test, the analysis matches the Kalman answer, as in the KS tests.
- [x] **E3 (C)** **Done 2026-11-05 (Saidie job 8695, 48 configs in 1.4 min).** 28 members, 1-min model `d34_c32_kuniform`, truth mem029. Second-half means:
  - **All-variable forecast nMSE:** no DA 0.329; best LETKF 0.302 (−8%; obs 2 km, loc 10 km, RTPS 0.9, model noise 0.05). Most configs are 0.32–0.33.
  - **Observed surface θ′ RMSE:** 0.73 → 0.46–0.48 K.
  - **Front displacement:** ~1–1.8 km → 0.2–0.3 km.
  - **Unobserved w:** 1.10 → 1.08 m/s, i.e. no real change. Condensate aloft is likewise unchanged.
  - **Spread:** model noise 0.05–0.1 is needed against collapse (spread 0.26–0.40 K vs ~0.1 without); 0.2 over-disperses and hurts.
  - **Verdict:** the filter pins what it sees (cold pool, gust front) but surface θ′ alone barely constrains the storm aloft. Observations aloft (E4, reflectivity) are the priority.
  - Details: Code in `ks_latent/cm1/twin.py` and `scripts/cm1/da_twin.py` (tests in `tests/unit/test_cm1_twin.py`). **Running: Saidie job 8695 (`~/cm1_da/e3_v1`).**
  - **Design:**
    - truth = held-out mem029, observations from the true columns;
    - ensemble = encoded states of the 28 training members at 1 h;
    - 5-min cycles for 3 h, truth at the domain edges;
    - sweep: observations every 2 / 10 km × localization 5 / 10 / 20 km × RTPS 0.5 / 0.9 × model-error noise 0 / 0.05 / 0.1 / 0.2.
  - **Local first run** (5-min model, 7 members):
    - the deterministic propagator collapses the spread (0.05 K vs 0.8 K error at the observations), hence the model-error noise option;
    - noise 0.1 + RTPS 0.9 gives forecast nMSE 0.333 vs 0.431 without DA and a front displacement of 0.25 km;
    - the unobserved w is not improved yet, and the analysis is sometimes worse than the prior over all variables. Both are expected with 7 members; the 28-member run decides.
  - Original task: a held-out member is the truth, surface θ′ every 2 km every 5 min is observed, with a 30-member latent ensemble.
  - Done when: analysis and forecast error against a free run and against the truth-member spread, consistency ratio, rank histograms.
- [x] **E3b (C)** **Done 2026-11-05 (Saidie job 8705).** 4 propagators share the `d34_c32_kuniform` autoencoder (seeds 0–3); θ′ + reflectivity, loc 5 km, RTPS 0.9.
  - **Model noise 0.05:** forecast / analysis nMSE 0.263 / 0.257, spread 0.29 K at the observations. Single model with the same noise: 0.271 / 0.263, spread 0.26 K. So −3%, slightly better calibrated.
  - **No noise:** 0.279 / 0.277, spread 0.19 K vs error 0.52 K.
  - **Verdict:** model disagreement roughly doubles the deterministic spread (single model ~0.1 K) but does not replace inflation. Useful in combination; G4 (adaptive inflation) is still needed. Was: **Running:** 3 propagators on `d34_c32_kuniform`'s autoencoder (seeds 1–3; Saidie jobs 8702–8704; a first attempt failed on a too-strict config check, now fixed), then the multi-model twin (8705: θ′ + reflectivity, model noise 0 and 0.05). Spread from model disagreement: a multi-model ensemble that cycles members through the 6 trained propagators (D3/D4 variants) instead of, or as well as, additive noise. Compare calibration and error with E3.
- [ ] **E3c (C)** Shifted tilings (Paper II): 4 offsets of the patch lattice (2/4/6 cells; non-periodic, so pad at the edges), each cycling independently and blended only for output.
  - Expectation: a modest gain only. The autoencoder floor (0.022) is small against the forecast error (~0.30), and tilings do not add model-error spread; each tiling collapses on its own.
- [x] **E4 (C)** **Done 2026-11-05 (Saidie job 8697, 36 configs in 1.3 min).** 28 members, truth mem029, observations every 2 km every 5 min (θ′ σ 0.5 K; reflectivity σ 5 dBZ, clipped at 0). Second-half means:

  | Observed | Forecast nMSE | Analysis nMSE | θ′ | Front | Echo dBZ | Liquid / ice | w |
  |---|---|---|---|---|---|---|---|
  | none | 0.329 | — | 0.73 K | ~1–1.8 km | 7.7 | 0.126 / 0.126 | 1.10 |
  | θ′ | 0.302 | 0.300 | 0.47 | 0.25 | 7.6 | 0.120 / 0.119 | 1.08 |
  | reflectivity | 0.295 | 0.291 | **0.96 (worse)** | 1.0 | 4.6 | 0.114 / 0.104 | 1.08 |
  | **both** | **0.271** | **0.261** | **0.46** | **0.17–0.23** | **4.3** | **0.107 / 0.100** | 1.05 |

  - **The two types are complementary:** θ′ pins the cold pool and front, reflectivity the condensate aloft.
  - **Reflectivity alone degrades θ′** through spurious ensemble correlations; θ′ observations anchor it.
  - **w is barely constrained:** neither observation type sees the wind.
  - **Best setting:** loc 5–10 km, RTPS 0.9, model noise 0.05.
- [x] **E4b (C)** **Done 2026-11-05 (Saidie job 8701). Radial velocity is the most valuable observation.**
  - Setup: 28 members, loc 5 km, RTPS 0.9, model noise 0.05; radar at x = 330 km, tilts 0.5 / 2 / 5°, σ 2 m/s.

    | Observed | Forecast nMSE | Analysis nMSE | u m/s | w m/s | Liquid / ice | Echo dBZ |
    |---|---|---|---|---|---|---|
    | none | 0.329 | — | 2.20 | 1.10 | 0.126 / 0.126 | 7.7 |
    | θ′ + reflectivity | 0.271 | 0.263 | 1.92 | 1.05 | 0.108 / 0.100 | 4.4 |
    | θ′ + radial velocity | 0.239 | 0.226 | 1.72 | 1.01 | 0.102 / 0.097 | 5.9 |
    | **all three** | **0.229 (−30%)** | **0.212 (−36%)** | **1.70** | **1.00** | **0.095 / 0.089** | **4.2** |

  - Model noise 0.1 is worse than 0.05 with radial velocity.
  - Earlier note: (`obs.py RadialVelocity`, known-field test). Local test, 7 members: θ′ + reflectivity + radial velocity gives forecast nMSE 0.307 vs 0.333 for θ′ alone, u 2.84 → 2.23 m/s, w unchanged. **Running: Saidie job 8701.** Doppler radial-velocity observations. In the 2D slice a low-elevation beam measures ≈ u cos α + w sin α: linear in the decoded state, with α from radar geometry (a radar at a fixed x, beam elevation 0.5–2°). Add it to the twin with θ′ + reflectivity, and see whether w and u improve.
  - Original E4 task: add reflectivity observations (nonlinear, bounded). Report the nonlinearity index from Paper II.
  - deps: E1b, E3. Done when: same table as E3, plus the nonlinearity index.
- [ ] **E5 (D+C)** Decide the Paper III framing from E3–E4: what is claimed, against which baselines.

## H. Forecasting as a result of its own (added 2026-11-05)
Paper III's second headline, alongside the DA: a local latent surrogate emulates a cloud-resolving model's full multivariate state (7 variables × 62 levels, 434 channels) for an hour, from a few dozen simulations, and stays physically consistent. Locality is what makes this possible: every 2 km site at every minute is a training sample, ~2 million local samples from 28 storms. Frame it as the condition that makes equation-free DA possible, so the paper does not read as two halves. Each task answers a referee's question.

- [~] **H0 (C)** **Forecasts: passed (Cashew 746659). Holding the boundaries at their start state costs 1–2% skill and 3–5% CRPS at 60 min; cold pool, fronts and rain FSS are unchanged, so the effect is confined to the edges. Report held-boundary numbers in the paper. Still to do: the DA twin's halo and the full-run models.** **Boundary check: do this first; every forecast number depends on it.**
  - The evaluation sets the 3-site (6 km) halo at each domain edge to the true state at every step, and the propagator's reach is 3 sites (6 km) per 1-min step, so in principle edge information can cross the domain within the hour. A real forecast does not know the future boundary state.
  - Re-score with the halo held at its state at the start time (`--eval-boundary persistence`, `rollout_from(boundary=...)`, tested; writes `results_h0_persistence.json`). The far field is quiet, so expect a small change; a large one means the truth boundaries are doing the work.
  - The DA twin also takes its halo from the truth (`twin._forecast`): check that too, with a fixed boundary.
  - Done when: skill / CRPS / physics with true against held boundaries, for `d34_c32_kuniform` and `d12_stagec_k45` (ens_A) and the full-run models.
- [ ] **H1 (C)** **(a) code done (`ks_latent/cm1/baselines.py`, `scripts/cm1/advection_baseline.py`, tested; global and per-level shifts), queued on Cashew (746836) on H2's cases. (b) not started.** **Baselines stronger than persistence.**
  - (a) Advection extrapolation: each field translated by its own best-fit shift at t0 (the nowcasting standard, Lagrangian persistence).
  - (b) A non-local model of similar size on the whole domain (a global encoder, or a 1D U-Net over the full 400 km), trained on the same storms. It shows what locality buys.
  - Done when: both in the skill / CRPS table at 5 / 30 / 60 min.
- [ ] **H2 (C)** **Queued on Cashew CPU (746832, `cluster/eval_h2.sh`): full-run Stage C models, 2 h leads, held boundaries.** **Where skill runs out.** Leads to 2 h (`--eval-lead-min 120 --results-tag h2_lead120`): when does skill vs persistence reach 1, and when does the stochastic model's spread saturate?
- [ ] **H3 (C)** **D10 split done (`--holdout-env`, `--manifest`, tested). Queued on Saidie (`cluster/saidie/h3_run.sh`, jobs 8868–8873): held-out corner qv < −1.2 g/kg and shear ≥ 1.1 (10 ens_B storms), training + Stage C + forecast scores + DA inside vs outside the region.** **Generalization to unseen environments** (= D8 / G5, with D10's split): train with a moisture × shear region of ens_B held out, test inside it. The test that separates "learned convection" from "fits these storms".
- [ ] **H4 (C)** **All seven variables:** skill, CRPS and spectra (variance retained by scale) per variable at 5 / 30 / 60 min, condensate included; 3 seeds for error bars on the main models.
- [ ] **H5 (D)** **Literature check** on ML emulation of storm-scale and cloud-resolving models before claiming anything is new (and the locality / sample-efficiency argument in particular).

## T. Transfer: more storms, bigger domains, no retraining (added 2026-11-05)
The CM1 analogue of Paper I's L-transfer. CM1 is not extensive chaos, so there is no Lyapunov spectrum to match, but the model is local and translation-equivariant, so the same two claims can be tested with storms in place of domain length: (1) a model trained on one squall line in 400 km forecasts two lines in 800 km without retraining, each as well as alone; (2) with localization, the ensemble size DA needs for a fixed skill does not grow with the number of storms ("ensemble size per storm" in place of D_KY per unit length), while without localization it does.

- [ ] **T1 (C)** **Pilot, no new simulations** (`scripts/cm1/transfer_pilot.py`): two test storms of the same environment (ens_A test + ens_A2) side by side, 3200 cells = 800 km. The domain edges are quiescent, so this is approximately a two-storm state until the outflows meet; scores near the junction are not meaningful and are excluded.
  - (a) Forecast: skill of the combined forecast against the mean of the two single-storm forecasts, held boundaries.
  - (b) DA: latent LETKF with 5 km localization against effectively none (very wide localization), N = 8 / 16 / 32 members (each member two training storms side by side), 1 storm against 2. The claim holds if the localized error ratio (2 storms / 1 storm) stays ~1 at every N while the unlocalized one grows, most at small N.
- [ ] **T2 (C, needs CM1 runs on Cashew; awaiting Dalton's go-ahead)** **Real two-storm CM1 runs.** Repeat T1 on real storms, including the region where they meet.
  - **Patch done (2026-11-05, not yet built):** `two_storms_var2` in `cluster/cm1/patches/apply_patches.py`. iinit = 8 starts one 2 K line thermal (radius 10 km × 1.5 km, at z = 1.5 km, x = domain centre) with ±0.2 K noise in it; var2 ≠ 0 replaces it with two identical thermals at centre ∓ var2/2. var2 = 0 is bit-for-bit unchanged CM1 (the random stream is untouched); CM1 stops if the thermals overlap or leave the domain. `init3d.F` compiles (partial gfortran build, not linked or run).
  - **Layout:** nx = 3200, var2 = 400 000 → thermals at 200 and 600 km, each 200 km from its outer edge and owning a 400 km half, like two single runs side by side; same dx, moving grid, boundaries. 6–10 members (var1 seeds), ~2× an ens_A run each.
  - **When the storms meet (measured on one single run):** deep gravity waves cross to the other storm after ~3 h; storm 2's rear cold pool (spreading west at ~55 km/h) reaches the midline at ~4 h and storm 1's inflow region by 5–6 h. So up to ~3 h the halves behave like independent storms; after that the run tests the model on interacting storms, which it never saw in training. For 6 h with no interaction: nx = 4000, var2 = 600 000 (~2.5× a single run).
  - **Before launching:** rebuild cm1.exe on Cashew (the current one ignores var2 and would silently give one storm at 400 km); have `make_ensemble.py` require `two_storms_var2` in `build/PATCHES` when var2 is set and check `cm1.log` for "latent_DA: two thermals at x ="; add nx / var2 options (or a two-storm template).
- [ ] **T3 (C)** **Four storms** (1600 km), if T1/T2 hold: the ensemble-size-per-storm curve with 1 / 2 / 4 storms.
- [ ] **T4 (C)** **Transfer to unseen environments** = H3 / D8 (moisture × shear region held out).

## F. Writing
- [x] **F1 (C)** `docs/cm1_summary/cm1_summary.tex`: a short illustrated summary of what we have found so far.
- [ ] **F2 (C)** Update the summary after each of B1, D2, D5, D8 and E3.

## G. Paper III: from promising to publishable (added 2026-11-05)
Working title: **equation-free ensemble data assimilation of a squall line with a learned local latent surrogate.** It is the sequel to Papers I–II on a full cloud model, with open boundaries and learned observation operators. Candidate venues: JAMES or MWR.

**Framing:** CM1 is the *nature run* only, the stand-in for reality that produces the truth and observations. The DA system and every baseline see only data, never the equations.

Each task below answers a specific objection a referee would raise. They come in priority order: **tier 1** is essential, **tier 2** strengthens the paper a lot, **tier 3** is a stretch goal or polish.

### G0. Pre-register the claims (before the final runs; ground rule from CLAUDE.md)
- [~] **G0 (D+C)** **Updated draft in `cm1_notes.md` ("G0 pre-registration — DRAFT", 2026-11-05), awaiting Dalton's sign-off; it supersedes the proposal below.** Write the paper's decision rules into `docs/cm1_notes.md` *before* running G1–G5. Original proposal:
  - **DA works:** θ′ + reflectivity beats no DA by ≥ 10% in all-variable forecast nMSE, averaged over truths, with a 95% confidence interval excluding 0. Front median |displacement| < 1 km.
  - **Calibrated:** spread/error at the observations is in 0.7–1.3, and rank histograms are not U-shaped (a χ² test, stated).
  - **Better than a static method:** latent LETKF beats EnOI (G2) by ≥ 10%.
  - **Generalizes:** in held-out environments (G5) the DA gain is ≥ half its in-distribution value.

  Every outcome is reported, including failures.

### Tier 1: essential
- [x] **G1 (C)** **Done (job 8717, 12 truths × 3 seeds, `th+refl+vr`): forecast nMSE 0.236 ± 0.027 vs no DA 0.380 ± 0.099; latent beats no DA on 12/12 truths; physical-space analysis 0.245, surrogate EnOI 0.331. See `cm1_notes.md`.** **Many truths, with error bars.**
  - The 28-member model (`d34_c32_kuniform`) never saw `ens_A2`, so its 30 members are 30 independent truths, plus mem029–030.
  - Run the best DA settings on ≥ 10 truths × 3 observation-noise seeds.
  - Report mean ± 95% CI, and paired differences (DA − no DA) per truth.
  - Done when: the tables and figures of E4/E4b have error bars.
- [ ] **G1b (C)** **Learned against hand-set model error** (added 2026-11-05). Deterministic Stage C model with latent noise 0.02 / 0.05 / 0.1, against the stochastic Stage C model (learned noise) with 0 / 0.02, on the full run's 22 test storms × 3 seeds, 32 random training members, `th+refl+vr`. `cluster/saidie/noise_study.sh` (jobs 8838–8844, after the full-run training); compare with `scripts/cm1/compare_studies.py` (paired per truth).
  - Why: in G1 the ensemble's spread at the θ′ observations is only 0.5 of its error (spread/error 0.505 ± 0.03), so the hand-set noise leaves the ensemble twice too confident; the stochastic model's own forecasts are calibrated (0.93).
  - Done when: cycled nMSE, spread/error and 5 / 30 / 60-min forecast nMSE for all five, with paired differences against noise 0.05.
- [ ] **G2 (C)** **(a) and (b) run in job 8717: latent 0.236 vs physical-space 0.245 vs surrogate EnOI 0.331; the physical-archive EnOI (alpha 0.5) is worse than no DA (0.732) and needs tuning before it is a fair baseline. (c) still to do.** **Conventional (equation-free) baselines with the same observations and members.**
  - (a) **EnOI:** a static physical-space covariance from the training members' states at the same time, with the same localization. It answers "what does a forecast model add?".
  - (b) **Physical-space LETKF on the decoded surrogate ensemble:** the same forecasts, but analysed in physical space instead of latent space. It isolates what the latent update itself buys: dimension, locality, and staying on the decoder's manifold. Paper II found a cost from returning to the decoder range, so this needs checking here.
  - (c) **Spline / optimal interpolation of the observations alone** (no ensemble), as in Paper II.
  - Done when: one table with the latent LETKF vs (a)–(c), on the G1 truths.
- [ ] **G2b (C)** **Equation-free forecast-model baselines with the same LETKF and observations.**
  - (a) **Analog forecasting:** each member's forecast is what followed its nearest neighbours in the training archive. Distance in latent space, k neighbours, averaged increments; matched by time since initiation. It asks whether a learned surrogate beats looking things up.
  - (b) **Persistence ensemble:** members do not evolve between analyses. This is the floor for any forecast model.
  - Done when: both appear in the G2 table on the G1 truths.
- [ ] **G3 (C)** **DA-initialized forecasts.** From analyses at 1.5, 2, 2.5 and 3 h, run 0–60 min ensemble forecasts. Compare them with no-DA forecasts and with persistence of the analysis: nMSE, front displacement, low-level-rain FSS, reflectivity ≥ 35 dBZ.
  - Done when: a "forecast skill vs lead after DA" figure. This is the result forecasters care about: how long DA keeps paying off.
- [ ] **G4 (C)** **Calibration without hand tuning.**
  - Replace the fixed model-noise level with adaptive inflation driven by the innovations (e.g. Desroziers-type / Anderson 2009 adaptive inflation; *verify the exact scheme before citing*), and/or the multi-model spread of E3b.
  - Report spread/error, rank histograms (θ′, reflectivity, radial velocity), and **reliability of reflectivity exceedance probabilities**: Brier score and reliability diagram for ≥ 35 dBZ.
  - Done when: calibration meets G0 with no per-experiment tuning.
- [ ] **G5 (C)** **Unseen environments (needs `ens_B`).**
  - Train on `ens_B` minus a held-out region of moisture × shear.
  - The truth comes from the held-out region; the ensemble from training environments, i.e. an imperfect surrogate *and* a mismatched prior.
  - Done when: DA gain in held-out vs in-distribution environments, with the G1 error bars.

### Tier 2: strengthens the paper
- [ ] **G6 (C)** **Ensemble size:** 8 / 16 / 28 / 58 members (`ens_A` + `ens_A2` as the pool, with the D5 58-member model). Plot error and calibration against N; this is the classic EnKF figure, and it connects to Paper II's local-dimension argument.
- [ ] **G7 (C)** **Localization against the reach rule.** Sweep the localization radius against the measured information reach (8–12 km per 5 min, i.e. ~2 km per 1-min step, times the cycle length) and the decoder reach. Check whether Paper II's rule (localization ≥ reach) predicts the best radius here. This ties the DA design to the data-only locality analysis.
- [ ] **G8 (C)** **Observation realism and sensitivity:**
  - surface network density 2 / 5 / 10 / 20 km;
  - radar coverage limited to range (and a blocked sector);
  - observation-error sweeps;
  - **operator error:** the learned reflectivity operator against a perfect one (using the truth's own reflectivity at analysis time, as an upper bound).

  Done when: a sensitivity table, and which observations matter most.
- [ ] **G9 (C)** **Cost table.** Wall time per cycle for the latent DA (forecast + analysis) against CM1 for the same ensemble (~90 min per member per 6 h on one core, measured), in core-hours and GPU-seconds.
- [ ] **G10 (C)** **Fix the weak cold pool (D6 finding).** Try:
  - a decoded-space loss term in Stage B (physical-space rollout loss on the lowest ~1 km of θ′);
  - longer rollouts with surface weighting.

  Done when: max C bias at 60 min < 5%, without losing the D3/D4 forecast skill.

### Tier 3: stretch and polish
- [–] **G11** ~~CM1-EnKF~~. **Dropped (Dalton, 2026-11-05):** the paper's premise is DA *without* the equations, so CM1 is only the nature run and is never used by the DA system or its baselines. Every baseline must be equation-free; see G2 and G2b.
- [ ] **G12 (C)** **Literature review (verify every citation).** Latent-space DA (latent EnKF/ETKF, ROAD-EnKF, LAE-EnKF, score-based latent filters), ML emulators of cloud-resolving models, convective-scale radar DA (reflectivity and radial-velocity assimilation in EnKF systems). Confirm the novelty claim: convective-scale, equation-free latent DA with learned reflectivity on a full cloud model.
- [ ] **G13 (C)** **Shifted tilings (E3c)** and **multi-model ensembles (E3b)**: include whichever helps, as ablations.
- [ ] **G14 (D+C)** **Paper skeleton** `docs/paperIII/` (same layout as Papers I–II): intro; data and data-only locality analysis; model; forecast skill and physics; DA method; twin experiments (θ′, reflectivity, radial velocity, N, localization, observations); baselines; unseen environments; cost; limitations (2D, cold-pool bias, twin setting); conclusions. Draft only after G0–G5.
- [ ] **G15 (C)** **Reproducibility:**
  - every figure from a script and config;
  - the Cashew kit regenerates all ensembles from the bundle;
  - a data availability statement (CM1 r21.1, our namelists, soundings and seeds).

**Suggested order:** G0 → G1 + G2 (now, with `ens_A`/`ens_A2`) → G3 → G4 → G6/G7 (cheap sweeps) → G5 (when `ens_B` exists) → G8–G10 → G12, G14.
