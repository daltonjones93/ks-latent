# CM1 squall-line data: findings and decisions

Data: `cm1data/cm1r21.1/`, which is gitignored and never committed.
- 2D x–z squall line, nx=1600 (dx=250 m), nz=62 stretched levels (20 km top).
- Open lateral boundaries; grid moves east at 12.5 m/s.
- Morrison two-moment microphysics (ptype=5); TKE turbulence; no Coriolis (icor=0).
- 10 members (`run_e1`–`run_e10`), each 6 h at 5-min output, plus a 12 h control (`run`) at 10-min output.
- Members differ only in low-level θ of the input sounding (about 0.1 K, lowest 1.5 km).

Scripts are in `scripts/cm1/`; outputs go to `artifacts/cm1/`.

## Running CM1 locally (2026-11-04)
- The supplied `cm1.exe` fails only because it was linked to Homebrew NetCDF, which is not installed.
- Rebuilt from `src/` without NetCDF (GrADS output, `output_format=1`, read by `grads_io.py`) using `make -j1`: 1 min 45 s.
- The t=0 state is bitwise equal to member 1's, including CM1's random noise, which uses a fixed seed.
- At 300 s fields differ by ~1e-3 of their range, from compiler differences. A rebuild run is the same model, not a bit-copy.
- Cost on one core: 90 simulated minutes took 7.6 min. A 6 h member is estimated at 30–60 min.
- The build currently lives in the session scratch directory. Move it somewhere persistent before relying on it.

## Findings
1. **Reach.** Two independent estimates agree:
   - one-step skill saturates at a window half-width of 8–12 km per 5-min frame (`locality_test*.json`);
   - the member-spread front moves 19–23 m/s grid-relative, about 6–7 km per frame (`viz_overview.json`).

   Gust front alone: 2.5–3.3 m/s grid-relative.
2. **Uncertainty concentrates at the gust front.** Surface-θ spread exceeds 3 K at the leading edge, so the errors are position errors.
3. **Data.** With 1→7 training members, one-step skill (storm columns) went:

   | Model | 1 member | 5 members | 7 members | Trend |
   |---|---|---|---|---|
   | Ridge | 0.600 | 0.654 | 0.658 | saturated |
   | MLP | 0.568 | 0.656 | 0.662 | about +0.03 per doubling to 5 members, flattening after |

   - The spread across member draws is ±0.005.
   - Mirror-image members (valid under icor=0) do not help: they are a different, reversed-shear environment.
   - Targets decorrelate within 2 frames and 2–4 km, about 1.5 k independent storm samples per member.
   - Inputs stay correlated for over 1 h and 25 km.
4. **5-min output is too coarse** (`sampling_test.json`, 1-min rerun of member 1's setup, storm columns 30–90 min):

   | Δ | Successive-increment correlation |
   |---|---|
   | 1 min | +0.37 (top PCs +0.5) |
   | 2 min | −0.01 |
   | 5 min | −0.53 |

   - The original 5-min files give −0.48 over the same window.
   - D(Δ) = E|Δx|² saturates by 5 min, and D(10)≈D(5).
   - So column states go through a cycle of roughly 10 min, sampled at about Nyquist by 5-min output.
   - 1-min output resolves it.

## Decisions
- Request **1-min output** (`tapfrq=60`), at least 2-min, from cluster runs. Reason: finding 4.
- Request 20–40 members in the current environment first, then several environments for generalization. Reason: finding 3.

## Cluster runs (2026-11-04)
- **Saidie is the wrong cluster.** It is an AI/GPU system (one 8×H200 node), and CM1 is CPU-only. **Use Cashew** (College of Engineering CPU cluster, Slurm, default partition `coe_cpu`).
- **Cashew caps jobs** at 20 submitted+pending per partition. The kit therefore packs one member per CPU into a few jobs; it does not use job arrays.
- **Kit:** `cluster/cm1/` (README has step-by-step instructions). Tested end to end on the Mac:
  - the validation member `mem000` reproduces member 1's initial state bitwise;
  - each run uses ~0.4 GB of memory.
- **Phase A plan:** 30 new members, original perturbation scheme (σθ = 0.1 K, z < 500 m), 6 h, 1-min output, plus `mem000`.
- **Local build** moved to `cm1data/cm1r21.1/build_local/`, which is gitignored.

## First learned model (2026-11-04): `ks_latent/cm1`, `scripts/cm1/train_cm1.py`
**Design.**
- **Autoencoder.** Columns as channels (7 vars × 62 levels = 434 per cell). 8-cell patches give 2 km sites. Two local mixing layers, so the encoder reach is 2 sites. Non-circular padding; condensate decoded ≥ 0.
- **Propagator.** Residual stencil, identity at init, reach 6 sites (12 km) per 5-min step, forced halo.
- **Training.** Stage A on 64 km windows. Stage B on 96 km windows of the encoded full domain, k ≤ 8.
- **Testing.** Full-domain forecasts from the encoded truth, with the true state at the domain edges.

**Data:** train run_e1–7, val run_e8, test run_e9–10; 5-min frames; MPS, about 5 min per run.

| c | Recon nMSE, all / storm | Forecast nMSE at 5 / 30 / 60 min | Persistence at 5 / 30 / 60 | Truth-member spread | Front error, 5–45 min | Persistence front error |
|---|---|---|---|---|---|---|
| 16 | 0.036 / 0.100 | 0.141 / 0.251 / 0.271 | 0.422 / 0.593 / 0.712 | ~0.20–0.25 | 0.65–1.2 km (3 km at 50–60 min) | 0.8 → 10.6 km |
| 32 | 0.023 / 0.065 | 0.151 / 0.245 / 0.260 | (same) | (same) | 0.95–1.4 km | (same) |

- **One-step skill vs persistence is 0.67.** That equals the model-free ceiling from the locality test (0.66–0.68), so the model has reached what 5-min data allow.
- **More latent channels improve reconstruction, not forecasts.** Forecasts are limited by the dynamics at 5-min frames and by the 7 training members, not by autoencoder capacity. This is consistent with findings 3–4: the 1-min ensemble is the next step.
- **Forecast error reaches the truth-member spread after about 25 min.** After that, the forecast is as far from the truth as another member is.

## Cashew ensemble ens_A (2026-11-04)
- **31 members** done on coe_cpu, 72–93 min each:
  - mem000 is a validation copy of run_e1's sounding;
  - mem001–030 use the original perturbation scheme;
  - 6 h each at 1-min output, 361 frames.
- **Output:** GrADS (no netcdf-fortran on Cashew), 2.19 GB per member, 64 GB in total, all on Cashew.
- **Data stays on Cashew.** It is not copied back without the user's go-ahead. Only run summaries (figures, JSON, logs) come back.

## Model trained on Cashew CPU (job 743446, `cluster/cashew_gpu/train_cm1_cpu.slurm`)
- **Setup:** 32 cores, ~36 Stage-A steps/s, Stage A 9 min + Stage B 26 min. No GPU needed.
- **Data:** train mem000–027, val mem028, test mem029–030; 1-min frames.
- **Model:** propagator reach 3 sites (6 km) per step; k_max 20.

| Lead | 1-min, 28 members | 5-min, 7 members (local_v1) | Persistence | Front error, 1-min | Front error, local_v1 |
|---|---|---|---|---|---|
| 1 min | 0.047 | — | 0.105 | 0.32 km | — |
| 5 min | **0.090** | 0.141 | 0.43 | **0.43 km** | 1.08 km |
| 15 min | 0.176 | 0.172 | 0.51 | **0.43 km** | 1.17 km |
| 30 min | **0.230** | 0.251 | 0.59 | 1.42 km | 0.65 km |
| 60 min | 0.319 | **0.271** | 0.75 | 2.50 km | 2.88 km |

- **Reconstruction nMSE:** 0.030 all / 0.088 storm (local_v1: 0.036 / 0.100).
- **Skill vs persistence at 5 min:** 0.79, up from 0.67.
- **Error vs truth-member spread:** model error stays below the spread for ~35 min.
- **Diagnosis:** 60 one-minute steps exceed the training horizon of k ≤ 20 steps, so the long-lead error builds up past what training controlled. Testing k_max 45 on the same autoencoder (job 743451).

## Longer training rollouts (job 743451): same autoencoder, propagator k_max 45 instead of 20 (53 min)

| Lead (min) | 1 | 5 | 10 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|
| nMSE, k ≤ 20 | 0.047 | 0.090 | 0.137 | 0.192 | 0.230 | 0.285 | 0.319 |
| nMSE, k ≤ 45 | 0.053 | 0.105 | 0.142 | 0.207 | **0.217** | **0.243** | **0.249** |

- **Truth-member spread** ≈ 0.25–0.27. With k ≤ 45 the forecast now stays below it for the full hour.
- **Surface θ′ RMSE at 60 min:** 0.98 → 0.80 K.
- **Front error at 45 / 60 min:** 3.5 / 2.5 km → 3.0 / 1.8 km.
- **Trade-off:** error is 10–15% higher at 1–5 min leads and 22% lower at 60 min.
- **Decision:** keep k_max 45 for forecasting and DA cycles longer than ~20 min. A curriculum that keeps some short rollouts, or a longer run, may recover the short leads.

## Saidie GPU run (D1, job 8680): ens_A, k_max 45, own autoencoder, one H200, 15.4 min
- **nMSE at 1 / 5 / 30 / 45 / 60 min:** 0.053 / 0.110 / 0.213 / 0.251 / 0.270. Cashew CPU k45 had 0.053 / 0.105 / 0.217 / 0.243 / 0.249.
- **Reconstruction:** val 0.0479 vs 0.0481.
- **The 60-min gap (+0.021) is above the 0.01 target.** It is attributed provisionally to run-to-run variation: different autoencoder, GPU non-determinism. To be confirmed by the seed spread (D2).
- **Speed:** about 6× faster than Cashew CPU.

## B1: Cashew build validation
- mem000 is bitwise equal to run_e1 at t=0.
- Build differences start ~50× below member-to-member differences, grow chaotically, and reach member size by ~2 h.
- Fronts agree within 1 km for 3 h.
- Same model; a rebuilt run is a new member after ~2 h.

## B2: noise seed
- CM1's fixed seed drives both the bubble noise (±0.2 K) and a domain-wide ±0.25 K θ noise (irandp = 1). All existing members share it.
- The seed_var1 patch makes it per member.

## D3/D4 on Saidie (2026-11-05): ens_A, single seed each, ~12–16 min per run on one H200
- **Recipe:**
  - random rollout lengths up to 45 (`--k-sampling uniform`) recover short-lead skill without hurting 60 min;
  - 32 latent channels improve reconstruction (storm RMSE 15–20% lower for every variable) and the 45–60 min forecasts.
- **Best at 1–30 min: c = 32 + uniform k.** nMSE 0.040 / 0.092 / 0.166 / 0.196, against the reference's 0.053 / 0.110 / 0.185 / 0.213.
- **Wider networks hurt beyond 15 min.** The model is data-limited.
- **1 km sites:** no clear forecast gain.
- **Front metric:** errors at 45–60 min are dominated by detector jumps when new cells form ahead of the front.

## New front metrics (2026-11-05, `ks_latent/cm1/verify.py`)
- **Main-pool front:** the leading edge of the main cold pool. Among pools with ≥ 25% of the largest pool's cold mass the easternmost is taken; pools narrower than 2 km are ignored and gaps narrower than 2 km merged.
- **Displacement:** threshold-free, signed alignment shift of surface θ′ around the true front.
- **FSS** of column-max condensate.
- **On the 10 original truth runs:**
  - threshold sensitivity (−0.5 / −1 / −2 K) 90th pct is 0.25 km, against 1.0 km for the easternmost-cell front;
  - the largest jump is 22 km, against 31;
  - jumps remain in ~3% of 5-min frames because squall lines advance discretely (new cells merge), so displacement is the headline error.

## Re-scored D3/D4 runs (Saidie job 8687)
**|displacement| median (90th pct), km:**

| Run | 30 min | 60 min |
|---|---|---|
| persistence | 5.95 | 11.48 |
| reference | 0.52 (1.10) | 0.80 (3.76) |
| c = 32 | **0.37 (0.75)** | 1.05 (**2.56**) |
| c = 32 + uniform k | 0.43 (0.86) | 1.21 (3.13) |
| 1 km sites | 0.66 (1.24) | 1.41 (4.52) |
| width 512 | 0.56 (1.53) | 1.72 (2.80) |

- **c = 32 has the smallest tails.**
- **Bias:** |bias| ≤ 0.3 km for all runs except the reference (−1.2 km at 60 min, a lag).
- **FSS barely discriminates:**
  - q > 0.3 at 4 km: 0.92–0.97 for models, 0.83–0.96 for persistence;
  - q > 1 at 1 km: 0.89–0.97 for models, 0.80–0.95 for persistence.

  Column-max condensate is dominated by the large anvil and stratiform region. **Follow-up:** FSS of low-level condensate or surface rain, where convective cells are.

## E3 latent DA twin (2026-11-05, Saidie job 8695)
- **Setup:** 28 training members as the ensemble, truth mem029, surface θ′ observations (σ 0.5 K) every 2 or 10 km every 5 min, 3 h of cycling from 1 h.
- **Results** (second-half means):
  - all-variable forecast nMSE: no DA 0.329; best LETKF 0.302;
  - observed θ′ RMSE 0.73 → 0.47 K;
  - front displacement ~1–1.8 → 0.2–0.3 km;
  - unobserved w unchanged (1.10 → 1.08 m/s).
- **Spread** collapses without model-error noise (spread ~0.1 K vs error ~0.7 K). Noise std 0.05–0.1 in latent units restores 0.26–0.40 K; 0.2 is too much.
- **Conclusion:** surface θ′ constrains the cold pool and front but not the storm aloft, so reflectivity observations are next (E4).
- **Shifted tilings (Paper II) are unlikely to cure collapse.** Each tiling cycles independently and collapses on its own, and blending averages spreads down. A multi-model ensemble over the 6 trained propagators is the principled spread source to try (E3b).

## D6 physics (2026-11-05), best model d34_c32_kuniform
- **Mass divergence:** forecasts 0.23–0.25 against the truth's 0.21–0.22 (CM1's own value from interpolation and compressibility).
- **Condensate mass error:** 2–12% over 1–60 min.
- **Cold pool too weak:** forecast max C stays at 23–25 m/s while the truth's strengthens to 28 m/s.
- **FSS of low-level rain** beats persistence strongly from 5 min on (0.70 vs 0.22 at 60 min).

## D7 joint vs per-variable encoder
Joint wins: w reconstruction error is 2× lower with the joint encoder, and forecast nMSE at 5 min is 0.092 vs 0.118. The per-variable encoder has 7× fewer parameters, which is part of the gap.

## E4 reflectivity + surface θ′ (2026-11-05, Saidie jobs 8696/8697)
- **Reflectivity operator:** test RMSE 1.63 dBZ on decoded columns (2.47 in echo columns).
- **θ′ + reflectivity** is best: forecast / analysis nMSE 0.271 / 0.261, against 0.329 without DA (−18 / −21%).
  - θ′ 0.46 K, front 0.17–0.23 km, echo reflectivity 7.7 → 4.3 dBZ, ice RMSE −21%.
  - w only −5%.
- **Reflectivity alone degrades surface θ′** (0.73 → 0.96 K): a spurious cross-variable update.
- **Next observation type:** Doppler radial velocity (E4b), for the wind.

## E4b radial velocity (2026-11-05, Saidie job 8701)
- **θ′ + reflectivity + Doppler radial velocity:** forecast / analysis nMSE 0.229 / 0.212, against 0.329 without DA (−30 / −36%).
  - Wind: u 2.20 → 1.70, w 1.10 → 1.00 m/s.
  - Condensate: liquid / ice −25 / −29%.
  - Echo reflectivity: 7.7 → 4.2 dBZ.
- **θ′ + radial velocity alone** reaches 0.239, so radial velocity is the single most valuable addition.

## D5 learning curve (valid part, ens_A subsets)
- nMSE at 30 / 60 min: 0.271 / 0.328 (7 members), 0.230 / 0.277 (14), 0.196 / 0.262 (28). Still rising.
- The 58-member point (ens_A + ens_A2) is being rerun (job 8706) after the `sbatch --export` comma bug.

## D5 complete: the learning curve saturates near 28 storms
- Skill relative to persistence at 5 / 30 / 60 min: 28 members 0.214 / 0.332 / 0.350, 58 members 0.206 / 0.328 / 0.347.
- θ′ and front metrics are mixed. ens_A2 adds members with other noise seeds.
- **Caution:** raw nMSE across runs with different training sets is not comparable, because the per-variable normalization differs.

## Model ideas, round 1 (2026-11-05, Saidie jobs 8718–8723; ens_A, single seed, baseline d34_c32_kuniform)
Skill = forecast nMSE / persistence nMSE (lower is better). Single-seed noise is ~±0.01–0.02.

| Run | Skill 5 / 30 / 60 min | θ′ at 60 min | Cold-pool max-C bias | Front 30 / 60 min | Rain FSS at 60 min |
|---|---|---|---|---|---|
| baseline | 0.214 / 0.332 / 0.351 | 0.83 K | −17.3% | 0.43 / 1.21 km | 0.696 |
| **D14a, average of 4 propagators** | **0.193 / 0.315 / 0.330** | **0.78** | −19.6% | 0.47 / 1.06 | **0.704** |
| D12, Stage C (k ≤ 20, 4000 steps) | 0.191 / 0.337 / 0.350 | 0.86 | −17.3% | 0.56 / 1.53 | 0.678 |
| D12, Stage C + θ′ sfc weight 2 | 0.192 / 0.338 / 0.351 | 0.85 | −17.1% | 0.57 / 1.57 | 0.678 |
| D13, 2-frame history | 0.207 / 0.347 / 0.345 | 0.80 | −18.6% | 0.47 / **0.72** | 0.701 |
| D14c, stochastic CRPS (mean of 8) | 0.257 / 0.383 / 0.369 | 0.83 | **−10.1%** | 0.71 / 1.94 | 0.647 |
| D14b, vertical front end | 0.203 / 0.337 / 0.346 | 0.84 | −17.5% | 0.43 / 1.22 | 0.683 |

- **Averaging is best and consistent:** 5–10% lower error at all leads.
- **History gives the best long-lead fronts.**
- **Stage C at k ≤ 20 helps only at 5 min, and the cold-pool weighting does nothing.** It is probably misconfigured: the baseline was trained to k 45. Retry with k ≤ 45 and 15 000 steps (job 8753).
- **The stochastic propagator is worse on mean error** but halves the cold-pool bias; its samples keep intensity.
- **The vertical front end is neutral.**
- **Next:** Stage C with k ≤ 45 (8753), and an average of 4 history propagators (8754–8757).

## G1 + G2: DA over 12 truths × 3 observation seeds (2026-11-05, Saidie job 8717)
Setup: `d34_c32_kuniform` (28 ens_A training members, never saw ens_A2); truths are 10 ens_A2 storms plus test members mem029–030. Observations θ′ (sfc) + reflectivity + radial velocity (`th+refl+vr`), every 2 km, 5-min cycles, 36 cycles, LETKF loc 5 km, RTPS 0.9, model noise 0.05. Score: all-variable forecast nMSE over the second half of the cycles (60-min forecasts from cycles 11/17/23/29), mean ± 95% CI over truths.

| Method | Forecast nMSE | vs no DA (paired) | Front |disp| |
|---|---|---|---|
| no DA | 0.380 ± 0.099 | — | — |
| **latent LETKF** | **0.236 ± 0.027** | **−0.144 ± 0.076** | 0.42 km |
| physical-space LETKF on the same ensemble (G2b) | 0.245 ± 0.031 | −0.135 ± 0.072 | 0.45 km |
| EnOI, surrogate forecast + static archive covariance (G2a) | 0.331 ± 0.065 | −0.049 ± 0.037 | 0.82 km |
| EnOI, physical archive (alpha 0.5) | 0.732 ± 0.064 | +0.352 ± 0.042 | 1.16 km |

Autoencoder floor 0.024.
- **Latent DA beats no DA on 12 of 12 truths.** The mean gain is 38%, inflated by two truths where the free forecast goes badly wrong (ens_A2 mem001 0.80 → 0.34, mem003 0.67 → 0.31). The median ratio latent / no DA is about 0.70, so a typical gain is ~30%.
- **Latent vs physical-space analysis:** latent is lower on 9 of 12 truths, but only by 0.009 on average, within the noise. The latent update costs nothing and is a little better; it is not a big win on its own.
- **Latent LETKF beats the surrogate EnOI by 29%**, so the flow-dependent ensemble matters.
- **The physical-archive EnOI is worse than no DA.** Its increments from a ±30-min climatological archive at alpha 0.5 are destructive. As it stands it is a strawman; tune alpha (and the archive window) before it goes in a paper, or drop it.
- **Against the proposed G0 rules** (not yet signed off, so the run is not pre-registered): ≥ 10% gain with a CI excluding 0 ✓; front < 1 km ✓; ≥ 10% better than EnOI ✓. Calibration and rank histograms were not part of this run.

## Model ideas, round 2 (2026-11-05, Saidie jobs 8753–8757; ens_A, single seed)
| Run | Skill 5 / 30 / 60 min | θ′ at 60 min | Cold-pool max-C bias | Front 30 / 60 min | Rain FSS at 60 min |
|---|---|---|---|---|---|
| baseline d34_c32_kuniform | 0.214 / 0.332 / 0.351 | 0.83 K | −17.3% | 0.43 / 1.21 km | 0.696 |
| D14a, average of 4 propagators | 0.193 / 0.315 / 0.330 | 0.78 | −19.6% | 0.47 / 1.06 | 0.704 |
| **D12, Stage C, k ≤ 45, 15 000 steps** | **0.187 / 0.316 / 0.310** | **0.76** | **−13.7%** | 0.69 / 0.94 | **0.773** |
| D13, 2-frame history | 0.207 / 0.347 / 0.345 | 0.80 | −18.6% | 0.47 / 0.72 | 0.701 |
| D13, average of 4 history propagators | 0.192 / 0.319 / 0.332 | 0.77 | −19.7% | 0.47 / 0.94 | 0.643 |

- **Stage C works once its rollouts match the base model's (k ≤ 45).** One fine-tuned model beats the 4-propagator average at 5 and 60 min (−12% at 60 min against the baseline) and ties it at 30. It also has the smallest cold-pool bias and much the best low-level rain FSS. The cost is a worse 30-min front (0.69 vs 0.43 km). Round 1's k ≤ 20 was the problem.
- **History adds nothing once propagators are averaged** (0.192 / 0.319 / 0.332 vs 0.193 / 0.315 / 0.330), and the twin cannot use it yet. Dropped from the recipe.
- **Averaging and Stage C don't combine directly:** Stage C also fine-tunes the autoencoder, so 4 Stage-C models have 4 latent spaces. Averaging decoded forecasts works for forecasts; a multi-model DA ensemble needs one shared latent space (fine-tune 4 propagators against one shared autoencoder in Stage C).

## D16 ens_B screen (2026-11-05, Saidie jobs 8761–8765; 46 training storms, 10 test storms spread over moisture × shear, single seed)
| Run | Skill 5 / 30 / 60 min | θ′ at 60 min | Cold-pool max-C bias | Front 30 / 60 min | Rain FSS at 60 min | Recon |
|---|---|---|---|---|---|---|
| S0 baseline (d34 recipe) | 0.265 / 0.400 / 0.385 | 0.83 K | −17.7% | 0.66 / 1.72 km | 0.540 | 0.079 |
| **S0 + Stage C (k ≤ 45)** | **0.237 / 0.383 / 0.370** | **0.79** | **−14.7%** | 0.77 / 1.87 | **0.569** | 0.083 |
| S1 + environment conditioning (D11) | 0.265 / 0.399 / 0.388 | 0.83 | −19.1% | 0.76 / 2.04 | 0.496 | 0.079 |
| S2 + mirrors (D9) | 0.297 / 0.402 / 0.399 | 0.87 | −21.5% | 0.86 / 2.11 | 0.470 | 0.085 |
| S3 + both | 0.294 / 0.407 / 0.397 | 0.89 | −17.5% | 0.81 / 1.88 | 0.359 | 0.085 |

- **ens_B is harder than ens_A** (baseline 0.265 / 0.400 / 0.385 vs 0.214 / 0.332 / 0.351): more varied storms from fewer storms per environment.
- **Stage C helps again** (−11% / −4% / −4%), with a smaller cold-pool bias and better rain FSS, and slightly worse fronts, as on ens_A. In the recipe.
- **Environment conditioning is neutral** on skill and worse on rain FSS. At these leads the storm's own columns seem to carry what the model needs; the environment enters through them.
- **Mirrors hurt** (+0.03 at 5 min, worse rain FSS), as on ens_A: mirrored storms live in reversed shear, which no test storm has, so they spend capacity on states that never occur. D9 is closed unless D8's held-out-environment test asks for it.
- **Recipe for the full run:** d34 recipe + Stage C, no environment conditioning, no mirrors (jobs 8797 → 8798).

## Stochastic propagators scored as ensembles (2026-11-05, Saidie jobs 8793 rescore, 8794 stochastic Stage C; ens_A test)
CRPS / persistence MAE (a deterministic forecast's CRPS is its MAE), 8 samples per stochastic forecast.

| Run | Skill 5 / 30 / 60 | CRPS 5 / 30 / 60 | Spread/error 60 | Cold-pool bias | Front 30 / 60 |
|---|---|---|---|---|---|
| deterministic baseline | 0.214 / 0.332 / 0.351 | 0.448 / 0.476 / 0.512 | — | −17.3% | 0.43 / 1.21 km |
| deterministic + Stage C | 0.187 / 0.316 / 0.310 | 0.421 / 0.453 / 0.476 | — | −13.7% | 0.69 / 0.94 |
| stochastic (D14c) | 0.257 / 0.383 / 0.369 | 0.344 / 0.331 / 0.344 | 0.99 | −10.1% | 0.71 / 1.94 |
| **stochastic + Stage C (decoded CRPS)** | 0.228 / 0.360 / 0.347 | **0.297 / 0.309 / 0.324** | 0.93 | −11.0% | 1.01 / 1.92 |

- **Stochastic + Stage C is the best probabilistic forecast:** CRPS 30–32% below deterministic Stage C at every lead, with calibrated spread (0.93). Stage C cut both its CRPS (−8–14%) and its mean error (to the deterministic baseline's at 60 min).
- Deterministic Stage C still has the best ensemble-mean error and fronts.
- **Caveat:** ensemble CRPS against a point forecast's MAE favours the ensemble by construction. The fair test for DA is the learned noise against the hand-set latent noise (0.05) on a deterministic model, in the twin.
- Full-data stochastic branch queued: jobs 8819 → 8820.

## H0 boundary check (2026-11-05, Cashew CPU job 746659; ens_A test)
Edge halo (3 sites = 6 km each side) set to the true state every step ("truth", as in every forecast score so far) against held at its state at the forecast start ("persistence": nothing after t0 is known).

| Run | Boundary | Skill 5 / 30 / 60 | CRPS 5 / 30 / 60 | θ′ 60 | Cold-pool bias | Front 30 / 60 | Rain FSS 60 |
|---|---|---|---|---|---|---|---|
| d34_c32_kuniform | truth | 0.214 / 0.332 / 0.351 | 0.448 / 0.476 / 0.512 | 0.83 K | −17.3% | 0.43 / 1.21 km | 0.696 |
| | held | 0.216 / 0.338 / 0.358 | 0.454 / 0.493 / 0.535 | 0.86 | −17.3% | 0.43 / 1.21 | 0.696 |
| d12_stagec_k45 | truth | 0.187 / 0.316 / 0.310 | 0.421 / 0.453 / 0.476 | 0.76 | −13.7% | 0.69 / 0.95 | 0.774 |
| | held | 0.188 / 0.321 / 0.316 | 0.427 / 0.469 / 0.497 | 0.79 | −13.7% | 0.69 / 0.95 | 0.774 |

- **The true boundaries are not doing the work.** Holding them fixed costs 1–2% in skill and 3–5% in CRPS at 60 min. The storm metrics (cold pool, fronts, rain FSS) are identical to three decimals, so the small loss is confined to near the edges (outgoing gravity waves), not the storm.
- The ranking is unchanged (Stage C still best), and the CPU re-score with true boundaries reproduces the GPU numbers exactly.
- For the paper, report the held-boundary numbers as the main ones (no future information at all). Still to check: the DA twin's halo, which also comes from the truth.

## G0 pre-registration — APPROVED by Dalton, 2026-11-05 (before any headline run)
Fixed before the headline runs. Every outcome is reported, including failures; a missed threshold is written up as such, not tuned away.

**Headline setup.**
- Model: the full run (ens_A + ens_A2 + ens_B, 93 training storms) with Stage C. Deterministic or stochastic is decided by G1b (lower paired cycled nMSE over the 22 test storms; if the difference's CI includes 0, the deterministic model, as the simpler one). Decided and recorded before any run below.
- Truths: the full run's 22 test storms (ens_A mem029–030, ens_A2 mem001–010, 10 ens_B spread over moisture × shear), 3 observation-noise seeds each.
- Observations: surface θ′ (σ 0.5 K), composite reflectivity (σ 5 dBZ), radial velocity at 3 tilts (σ 2 m/s), every 2 km, every 5 min. 36 cycles from t = 60 min; scores are second-half means.
- Ensemble: 32 training storms drawn at random (seed 0). Localization 5 km, RTPS 0.9.
- Boundaries held at the truth at t0 (no future information; H0).
- Statistics: mean ± 95% CI over truths (seed-averaged within a truth), and paired differences per truth.

**Claims and thresholds.**
1. **DA works.** Cycled forecast nMSE ≥ 10% below no DA, with the paired 95% CI excluding 0, and lower than no DA on ≥ 80% of truths. Median |front displacement| < 1 km.
2. **Calibrated.** Prior spread / error at the θ′ observations in 0.7–1.3, and the rank histogram not U-shaped (χ² test against flat, p > 0.01, reported with the histogram).
3. **Better than static and observation-only methods.** ≥ 10% lower cycled nMSE than each of: surrogate EnOI, physical-archive EnOI (both with alpha tuned on one truth outside the 22), and observation-only interpolation (G2c).
4. **Generalizes.** In the held-out environment region (H3/G5: dry, strongly sheared ens_B storms that no training or validation storm comes from), the DA's relative gain over no DA is ≥ half of its in-distribution gain, and forecast skill vs persistence at 30 min is within 25% of its in-distribution value.
5. **Forecasts (H).** With held boundaries: skill vs persistence (nMSE ratio) < 0.5 at 30 min and < 1 at 60 min, and lower than advection extrapolation (H1a) at 30 and 60 min.
6. **Transfer (T).** With 5 km localization, two storms side by side give an error ratio (2 storms / mean of 1-storm) ≤ 1.1 at every ensemble size; at N = 8 the ratio without localization is larger than with it.

**Exploratory, no threshold:** latent against physical-space analysis; learned against hand-set model error (G1b, beyond the model choice above); four storms (T3).

## Full run on all three ensembles (2026-11-05, Saidie 8797/8798/8819/8820; 93 training storms; 22 test storms from all three ensembles, true boundaries)
| Run | Skill 5 / 30 / 60 | CRPS 5 / 30 / 60 | Spread/error 60 | θ′ 60 | Cold-pool bias | Front 30 / 60 | Rain FSS 60 |
|---|---|---|---|---|---|---|---|
| full_v1 | 0.237 / 0.382 / 0.364 | 0.495 / 0.511 / 0.527 | — | 0.80 K | −20.5% | 0.76 / 1.65 km | 0.638 |
| full_v1 + Stage C | **0.213 / 0.365 / 0.344** | 0.475 / 0.498 / 0.517 | — | **0.73** | −17.4% | **0.62 / 1.39** | **0.590** |
| full_v1 stochastic | 0.287 / 0.415 / 0.384 | 0.384 / 0.368 / 0.376 | 0.91 | 0.78 | −15.2% | 0.91 / 1.61 | 0.544 |
| full_v1 stochastic + Stage C | 0.267 / 0.404 / 0.385 | **0.342 / 0.344 / 0.360** | 0.93 | 0.80 | **−9.9%** | 0.87 / 2.02 | 0.571 |

- The ens_A pattern holds on all the data: Stage C helps both models; the stochastic one has ~30% lower CRPS, calibrated spread and the smallest cold-pool bias, the deterministic one the lowest mean error and best fronts.
- The test set mixes ensembles, so these are not comparable with the ens_A-only or ens_B-only tables; per-ensemble scores (true and held boundaries) are running on Cashew (job 746827, `cluster/eval_full_subsets.sh`).

## G1b: learned against hand-set model error, in the DA (2026-11-05, Saidie 8860–8866)
Full run + Stage C, 22 test storms × 3 seeds, θ′ + reflectivity + radial velocity, 32 random members, loc 5 km, RTPS 0.9, boundaries held at t0 (the G0 headline setup). No DA: 0.469 ± 0.061.

| Model, added latent noise | Cycled nMSE | Paired vs det 0.05 | Wins vs det 0.05 | Spread/error (θ′ obs) | Front | Forecast 5 / 30 / 60 min |
|---|---|---|---|---|---|---|
| deterministic, 0.05 | 0.277 ± 0.023 | — | — | 0.43 | 0.37 km | 0.272 / 0.314 / 0.337 |
| deterministic, 0.02 | 0.292 ± 0.024 | +0.015 ± 0.004 | 5% | 0.27 | 0.45 | 0.282 / 0.322 / 0.352 |
| deterministic, 0.1 | 0.273 ± 0.022 | −0.004 ± 0.004 | 64% | **0.71** | 0.38 | 0.270 / 0.316 / 0.337 |
| stochastic, 0 (learned only) | 0.312 ± 0.022 | +0.035 ± 0.007 | 5% | 1.18 | 0.40 | 0.299 / 0.340 / 0.370 |
| stochastic, 0.02 | 0.312 ± 0.022 | +0.035 ± 0.008 | 9% | 1.19 | 0.42 | 0.298 / 0.339 / 0.368 |
| deterministic, 0.05, **true** boundaries | 0.250 ± 0.023 | −0.027 ± 0.003 | 100% | 0.46 | 0.37 | 0.250 / 0.287 / 0.305 |

- **DA works on the headline setup:** 0.277 vs 0.469 with no DA (−41%, paired −0.192 ± 0.054), better on all 22 storms, front 0.37 km. G0 claim 1 met.
- **The learned model error loses in the DA** by 0.035 (12%), on 91–95% of storms, despite the stochastic model's calibrated free forecasts. In cycling it is now over-dispersive (spread/error 1.18): RTPS 0.9, tuned for the deterministic model, relaxes the analysis spread back towards a prior spread that the learned noise has already made large. RTPS was not re-tuned for it, so this comparison is not yet fair to it. Exploratory follow-up: stochastic with RTPS 0 and 0.5.
- **G0 model choice (pre-registered rule: lower paired cycled nMSE; deterministic if the CI includes 0): deterministic.** The stochastic model is worse with a CI excluding 0.
- **Hand-set noise:** 0.1 and 0.05 are tied on error (−0.004 ± 0.004); 0.1 is much better calibrated (0.71 vs 0.43). 0.02 is worse on both.
- **Boundaries matter more in DA than in forecasts:** true boundaries lower the cycled error by 10% (0.250), against 1–2% for 60-min forecasts. Over 3 h of cycling the held halo drifts from the truth. Held boundaries stay the headline, as pre-registered.

## Full run scored per ensemble (2026-11-05, Cashew 746827, `cluster/eval_full_subsets.sh`)
Skill 5 / 30 / 60 min (true boundaries; held boundaries cost 1–3% at 60 min in every ensemble, storm metrics unchanged):

| Test storms | Specialised model | Full run + Stage C |
|---|---|---|
| ens_A mem029–030 | ens_A-only Stage C (d12_stagec_k45): 0.187 / 0.316 / 0.310 | 0.183 / 0.320 / 0.316 |
| ens_B, 10 storms | ens_B-only Stage C (D16 screen): 0.237 / 0.383 / 0.370 | 0.239 / 0.376 / 0.366 |
| ens_A2, 10 storms | — | 0.202 / 0.367 / 0.334 |

- **One model on all the data matches each specialised model on its own storms**: pooling neither helps nor hurts, consistent with D5's saturation at ~28 storms per environment. The extra storms buy coverage, not accuracy within an environment.
- The stochastic Stage C model has 20–30% lower CRPS, spread/error 0.91–0.99 and the smallest cold-pool bias in every ensemble.
- **Follow-up (exploratory; Saidie 8876–8877), stochastic model with RTPS re-tuned:** RTPS 0.5 gives 0.298 ± 0.024 (paired +0.021 ± 0.007 vs deterministic 0.05; lower on 18% of storms) with spread/error **1.04** — calibrated with no noise level to tune — and fronts 0.36 km; RTPS 0 gives 0.320 (spread/error 0.86). The remaining ~7% error cost matches the stochastic propagator's weaker mean forecasts. Trade-off to report: hand-set noise is more accurate (0.273 at 0.1), learned noise is calibrated out of the box.

## H2 (2 h leads) and H1a (advection extrapolation) (2026-11-05, Cashew 746832, 746836)
Full run + Stage C, all 22 test storms × 8 start times (176 cases), boundaries held at t0. Skill = nMSE / persistence nMSE.

| Lead (min) | 5 | 30 | 60 | 90 | 120 |
|---|---|---|---|---|---|
| deterministic Stage C, skill | 0.220 | 0.357 | 0.361 | 0.352 | 0.350 |
| deterministic Stage C, nMSE | 0.088 | 0.211 | 0.269 | 0.316 | 0.358 |
| persistence, nMSE | 0.397 | 0.593 | 0.745 | 0.898 | 1.023 |
| stochastic Stage C, skill | 0.272 | 0.399 | 0.406 | 0.392 | 0.379 |
| stochastic Stage C, CRPS skill | 0.344 | 0.355 | 0.385 | 0.390 | 0.387 |
| stochastic Stage C, spread/error | 0.86 | 0.91 | 0.91 | 0.89 | 0.89 |
| advection extrapolation, global shift | 0.960 | 1.212 | 1.249 | 1.227 | 1.214 |
| advection extrapolation, per-level shifts | 0.957 | 1.198 | 1.256 | 1.234 | 1.212 |

- **Skill does not run out within 2 h:** relative to persistence it settles at ~0.35 from 30 min on, and the model's error at 2 h (0.358) is below persistence's at 5 min (0.397). The stochastic model stays calibrated (0.89) out to 2 h.
- **Advection extrapolation is worse than persistence** beyond 5 min (skill 1.2–1.25, both variants). In the moving grid the storm is nearly stationary; the fitted shifts (mean −3 to −3.6 cells per 5 min) follow the part that moves fastest (the cold pool spreading rearward) and displace everything else. So persistence is the stronger simple baseline here, and the surrogate beats both by ~3×.
- **G0 claim 5 met:** skill 0.357 at 30 min (< 0.5) and 0.361 at 60 min (< 1), below advection at both.

## H3 / G5: unseen environments (2026-11-05, Saidie 8868–8873)
Full-run recipe with the dry, strongly sheared corner of ens_B (qv offset < −1.2 g/kg, shear ≥ 1.1; 10 storms) out of training and validation. Held boundaries.

| | Skill 5 / 30 / 60 | CRPS 5 / 30 / 60 | θ′ 60 | Cold-pool bias | Front 30 / 60 | Rain FSS 60 |
|---|---|---|---|---|---|---|
| in-distribution ens_B test storms (8) | 0.246 / 0.392 / 0.388 | 0.529 / 0.550 / 0.581 | 0.82 K | −18.1% | 0.73 / 1.80 km | 0.454 |
| held-out region (10) | 0.261 / 0.405 / 0.420 | 0.547 / 0.573 / 0.605 | 0.86 | −19.8% | 0.96 / 3.50 | 0.394 |

DA (G0 setup, noise 0.05): held-out region 0.224 ± 0.018 vs no DA 0.551 ± 0.084 (−59%, paired −0.326 ± 0.079, 10/10 storms, front 0.42 km); in-distribution 0.265 ± 0.042 vs 0.508 ± 0.141 (−48%, 8/8 storms, front 0.47 km).

- **G0 claim 4 met.** Forecast skill at 30 min in the unseen region is within 3% of in-distribution (threshold 25%); at 60 min 8% worse, and the 60-min front error roughly doubles (3.5 vs 1.8 km), the clearest sign of extrapolation. The DA's relative gain is larger in the unseen region (59%) than in distribution (48%): where the model alone is weaker, the observations correct it more, and the localized analysis does not depend on having seen the environment.

## T1: transfer pilot, two storms side by side (2026-11-05, Saidie 8867)
Full run + Stage C (trained on single 400 km storms), 3 pairs of same-environment test storms joined into 800 km, ±20 km around the junction left out of every score (and the same columns in the single runs). Held boundaries.

Forecast, nMSE ratio joined / mean of singles: 1.000 at 5 min, 1.014–1.022 at 30, 1.032–1.050 at 60. The joined state is discontinuous at the junction (two unrelated far fields meet), and what the model does there spreads past the ±20 km band within the hour; the pilot cannot separate that artifact from genuine non-locality. Real two-storm runs (T2) would.

DA, prior nMSE ratio 2 storms / 1 storm (3 pairs × 2 seeds):

| N | loc 5 km | no localization (10 000 km) |
|---|---|---|
| 8 | 1.034 ± 0.060 (1-storm 0.384) | 1.122 ± 0.097 (1-storm 0.429) |
| 16 | 1.053 ± 0.031 (0.362) | 1.091 ± 0.074 (0.424) |
| 32 | 1.065 ± 0.014 (0.348) | 1.049 ± 0.109 (0.415) |

- **G0 claim 6 met as worded, but weakly.** Localized ratio ≤ 1.1 at every N; at N = 8 the unlocalized ratio is larger (1.12 vs 1.03), but the CIs overlap, so it is not a significant difference. The unlocalized ratio falls with N as expected (more members can cover two storms).
- The localized ratio rises slightly with N (1.03 → 1.07, tight CI at N = 32): a fixed junction artifact becomes a larger share as the DA error falls. Consistent with the forecast pilot, and another reason for T2.
- Localization matters even for one storm: the unlocalized 1-storm error is 12–19% higher at every N.

## G0 addendum (2026-11-06, before the runs it governs): settings are tuned on the validation storms
The latent noise level for the headline DA, and alpha for the two EnOI baselines (claim 3), are chosen on the full run's 6 validation storms (ens_A mem028, ens_A2 mem011, ens_B mem025/053/055/058; used only for checkpoint selection, never trained on), 3 seeds, same setup as the headline, by lowest cycled nMSE. The 22 test storms do not enter any choice. Candidates: noise 0.05 / 0.1 / 0.15; alpha 0.1 / 0.25 / 0.5 / 1.0. (`da_study.py --tuning`.)
- **G0 addendum 2 (2026-11-06, before the claim-3 runs):** the observation-only baseline (G2c) has no forecast model, so before the observations its estimate is just the archive mean. For it, claim 3 compares **analysis nMSE** (each method's estimate right after the current observations), not the cycled forecast nMSE. The two EnOI baselines are compared on cycled forecast nMSE, as worded. The observation-only baseline's alpha is tuned on the validation storms like the EnOI ones (0.1 / 0.25 / 0.5 / 1.0).
- **Validation tuning, noise level (2026-11-06, Saidie 8886–8888; 6 validation storms × 3 seeds):** cycled nMSE 0.260 ± 0.026 (0.05), **0.258 ± 0.023 (0.1)**, 0.275 ± 0.025 (0.15); spread/error 0.42 / 0.71 / 0.95. By the G0 addendum's rule (lowest cycled nMSE on validation), the headline noise level is **0.1**. Headline numbers on the 22 test storms at 0.1: 0.273 ± 0.022 vs no DA 0.469 ± 0.061, spread/error 0.71 (claim 2's spread criterion within 0.7–1.3; its rank-histogram test still to compute).

- **G0 addendum 3 (2026-11-07, before any of the runs it governs; `cluster/saidie/g_paper.sh`):**
  - *Claim 2, rank histogram.* Tested on a rerun of the headline (noise 0.1; identical setup, which also checks that it reproduces 0.273), with `scripts/cm1/rank_test.py`. The primary histogram adds observation error (σ 0.5 K) to the members before ranking (Hamill 2001; ranking noisy observations among noise-free members makes even a perfect ensemble U-shaped). The plain histogram is reported too. Chosen before any histogram was computed: no rank histogram has been looked at in this project. The pre-registered χ² test (p > 0.01) pools all ranks, which are strongly correlated in space and time, so it will reject almost any real ensemble. Reported as pre-registered regardless. Exploratory beside it: the same test on a decorrelated subsample (every 10th observation, i.e. 20 km; every 3rd cycle; seed 0), plus the outer-bin ratio (> 1 U-shaped, < 1 dome-shaped).
  - *G2b, equation-free forecast-model baselines* (same LETKF, observations, members and boundaries; only the forecast model differs). Persistence ensemble: members do not evolve, spread from added latent noise, 0.1 / 0.2 / 0.3. Local analog: each site's next change is that of one of its 10 nearest 3-site latent patches in the 93 training storms' archive, 0 / 0.05 / 0.1 added noise. For each baseline the noise level with the lowest cycled nMSE on the 6 validation storms is the one reported (the rule claim 3 used). Reading, borrowed from claim 3 and not a G0 claim: the latent DA ≥ 10% lower cycled nMSE than each, paired CI excluding 0.
  - *G3, forecasts from the analyses* (exploratory, no threshold): ensemble-mean forecasts from the analyses at 90 / 120 / 150 / 180 min to 60 min, against no-DA forecasts from the same times and persistence of the analysis. Reported: nMSE, front displacement and reflectivity ≥ 35 dBZ FSS (10 km) by lead, and the lead at which the DA-initialized forecast's nMSE advantage over the no-DA forecast falls below 10%.
  - *Claim 6, extended* (the original T2 result stands as run): 8 more two-storm runs (16 in all), seeds 0 / 1 / 2, N = 8 / 16 / 32, the same rule (localized ratio ≤ 1.1 at every N; at N = 8 the unlocalized ratio larger).

### G0 addendum 3b: post-hoc G2b grid extension (2026-11-07, written after seeing the first validation runs, before any extension result)
The pre-registered grids (persistence noise 0.1 / 0.2 / 0.3, analog 0 / 0.05 / 0.1 with one drawn neighbour) gave validation analyses far worse than no DA at all. nMSE: persistence 2.2 / 4.4 / 7.8, rising with noise; analog 14.2 at noise 0, against 0.45 for no DA. The θ′ fit in observation space was moderate. The cause is in the design, not a bug: with persistence, the per-step noise random-walks the latent off the training manifold in directions the observations do not constrain, and no dynamics pull it back. With one drawn neighbour per site, neighbouring sites take incoherent increments from different storms. So the grids were too narrow to give these baselines a fair chance. **Added (post hoc, disclosed): persistence noise 0 / 0.02 / 0.05, and the analog with the mean of the 10 neighbours' increments (smooth, deterministic) at noise 0.02 / 0.05 / 0.1.** Selection rule unchanged: the best validation analysis nMSE over the pre-registered and the added candidates together, reported on the test storms. Adding weaker-noise and smoother variants can only strengthen the baselines against the latent model.

### G0 addendum 4 (2026-11-07, before any of the runs it governs): the Fable-review section-6 sweeps
Source: `docs/cm1_fable_review.md` §6. Common protocol for every study below: the G0 headline setup (full_v1_stagec unless stated, 32 random training members, θ′ + reflectivity + radial velocity, 5-min cycles over 3 h, RTPS 0.9, localization 5 km, held boundaries, 3 seeds). Every grid point runs on the 6 validation storms. The point with the lowest validation prior nMSE is chosen (`scripts/cm1/da_sweep.py`, recorded in `choice.json`), then run on the 22 test storms. Paired comparisons are against the headline (0.273, noise 0.1). Out: `~/cm1_da/fable/<name>/`.
- **N2, adaptive additive noise** (`--adaptive-noise global|site`, `twin.adapt_noise`). Each cycle, the noise amplitude is rescaled by clip(ratio, 1/4, 4)^(γ/2), γ = 0.5, bounded to 0.2-5× the base level. The ratio is Σw(d²/R − 1) / Σw(s²/R) over the prior innovations; w are the LETKF's Gaspari-Cohn weights per site, or all ones for the global variant. Base level grid 0.05 / 0.1 / 0.2. **Reading:** adopted for the paper if spread/error at the θ′ observations is in 0.9-1.1, the perturbed-member rank histogram's outer-bin ratio (decorrelated) is < 1.1, and the paired cycled nMSE is no more than 3% above the headline. The analysis cold-pool bias is reported too (Sobash & Wicker 2015 warn that additive noise biases it).
- **N4, ensemble size** (N 8 / 16 / 32 / 64 / 93, noise grid 0.05 / 0.1 / 0.15 per N, for full_v1_stagec and the 64-channel sm1_c64_stagec). **Reading:** if the 64-channel penalty (+0.023 at N = 32) falls to within its CI of 0 at N ≥ 64, the "richer latent hurts DA" result is stated as a sampling-error effect. The error curve against N is reported as the classic EnKF figure; no threshold. (93 is the size of the training pool, so 128 is not possible without reusing storms.)
- **N5, localization radius** (2 / 5 / 10 / 20 / 40 km at noise 0.1; every radius also on the test storms, for the curve). **Reading:** the reach rule (Paper II: localization ≥ information reach ≈ 2 km/min × 5 min = 10 km) predicts the optimum at ≥ 10 km. Report whether the validation optimum agrees, and the test curve. Note: the headline's 5 km was chosen in E3/E4 on one truth, before G0.
- **M1, observation spacing × cycle length** (spacing 2 / 4 / 8 / 16 km for all three observation types × cycles of 5 / 10 / 15 / 30 min, 3 h of cycling, localization fixed at 5 km). Two models: deterministic with the noise tuned per cell (0.05 / 0.1 / 0.2), and the stochastic full_v1_stagec_crps with no added noise and RTPS tuned per cell (0.5 / 0.9). **Hypothesis (assimilation in the unstable subspace):** the deterministic model's advantage (+7% for the stochastic one at the dense, 5-min headline) shrinks and reverses as observations thin and cycles lengthen. **Reading:** a reversal is claimed in a cell if the stochastic model's paired test cycled nMSE is lower with the CI excluding 0. The map of the paired difference over the 16 cells is the deliverable either way. The 30-min cycle has only 6 cycles (3 scored), so it is the noisiest cell.
- **M3, mean-anchored ensemble** (`--modes anchored`): the mean is advanced by the deterministic full_v1_stagec, the perturbations by a stochastic propagator (`m3_stoch_frozen_stagec`; 4 noise channels, Stage B, then a Stage C CRPS fine-tune with the autoencoder frozen, `--c-freeze-ae`, so the latent is shared and checked bit for bit). RTPS grid 0 / 0.5 / 0.9, no added noise. The frozen stochastic model alone is a control. **Reading:** adopted if spread/error is in 0.9-1.1 and the paired cycled nMSE is no more than 2% above the headline. That would be calibration without a tuned noise level, at the deterministic model's accuracy.
- **N6, cost table** (`scripts/cm1/cost_table.py`): forecast hour and DA cycle wall times on one H200 and on Cashew CPU, against CM1's measured ~14.5 core-min per member-hour. No threshold.
- **Not queued, with reasons:**
  - N8a: L1 already ran the propagator in float64 on CPU (`lyapunov_cm1.py` casts the propagator and trajectory to double), so the numerical-noise check is satisfied.
  - N3, a non-local DA baseline: needs an adapter for S55's global latent. S55's forecasts already exist as a non-local forecast baseline: skill 0.48 / 0.57 / 0.51 at 5 / 30 / 60 min, against 0.21 / 0.37 / 0.34, with ~70 M parameters against ~3 M.
  - N7 / L2b: needs a CM1 patch that applies the tiny perturbation after a restart read.
  - M2 (PFF), M4 (pair-divergence loss), M5 (waits on FPd), M6-M8: not yet coded.

### G0 addendum 5 (2026-11-08, before the full run): FPe, a closed equation for the leading channels
User direction (2026-11-08): FPd showed the 32-channel law cannot be pruned to a readable size. Ask a smaller question instead: on the ordered (nested-dropout) latent fpc_nested_stagec, is there a short equation for the first k channels in terms of themselves only? `scripts/cm1/lead_channel_law.py`:
- **Fit:** bootstrap STLSQ over whole storms (FP0's tool, centred fit) of dz_j/dt (centred difference, per minute), j < k, on 30 training storms, frames 30-330. Two libraries in the first k channels: "ode" (z and quadratic products: a local ODE per site) and "pde" (plus ∂x z and ∂²x z and their products with z). k = 1 / 2 / 4 / 8, thresholds 0.01 / 0.03 / 0.1.
- **Scoring,** on 10 test storms: held-out tendency R²; closed rollouts of the k-channel system from 60 / 120 / 180 min (Heun, tendency cap 5, edges held), scored by channel 0's MSE against persistence and against the trained model's own all-channel rollout; decoded nMSE with channels ≥ k zeroed.
- **Control:** the same on fpc_law_reuse_stagec, whose latent is not ordered.
- **Reading:** the result is "a usable leading-channel law" if, for some k ≤ 4, the pde fit for channel 0 has ≤ 30 terms and held-out R² ≥ 0.5, and its closed rollout beats persistence of channel 0 at 30 min. Otherwise report the best R² and rollout at each k: how many channels a closed law needs.
- From the smoke test (2 storms, not a result): channel 0 correlates +0.79 with surface θ′, −0.85 with column condensate and −0.72 with column-max w, so it is a cold-pool / convection coordinate. Its largest pde term is z0 ∂x z0.

### G0 addendum 5b (2026-11-08, before the runs): FPe with higher spatial order, second order in time, and rollout refits
- **What the first-order full run showed (9094),** so this addendum is written after seeing it:
  - The one-step fits explain up to 65% of channel 0's held-out tendency, with pde terms only; local ODEs explain ≤ 20%.
  - Their closed rollouts are unstable: worse than persistence at 30 min.
  - The k = 1 law with 3 terms is a nonlinear advection: ∂t z0 = −(0.057 + 0.083 z0) ∂x z0. Its constant speed, 0.11 km/min, is Q2's storm propagation speed.
- **Added, at user direction:**
  - spatial derivatives up to 3rd and 4th order (`pde3`, `pde4`: dispersion and hyperdiffusion, as in KS);
  - second order in time (d²z/dt² with dz/dt in the library);
  - a rollout refit (`--refit-rollout 30`): the STLSQ support is kept and its coefficients are refit on closed rollouts up to 30 min, which addresses the instability.
- All tested on known lattice laws (advection-diffusion, the wave equation, hyperdiffusion). The refit test also checks that the input coefficients are not modified, after an aliasing bug in the first smoke run.
- **Runs (CPU):** time order 1 and 2 separately × kinds pde / pde3 / pde4 × k 1 / 2 / 4 × thresholds 0.03 / 0.1.
- **Reading:** as addendum 5 (≤ 30 terms for channel 0, held-out R² ≥ 0.5, closed rollout better than persistence of channel 0 at 30 min), now judged on the refit law. Reported in addition: whether pde4 or time order 2 improves the refit rollout over pde at equal k.

### G0 addendum 5c (2026-11-08, before the runs): FPe with derivative-product and cubic terms
User direction: two library options, `+dd` and `+cubic` (`ks_latent.cm1.leadlaw.parse_kind`), both tested on known laws. They answer questions the quadratic library cannot.
- **`+dd`:** products of two spatial derivatives, e.g. the KPZ term (∂x z)², which coarse-grained KS flows to. The test recovers D∂²z + λ(∂x z)².
- **`+cubic`:** cubic terms with at most one derivative, e.g. z0³ or z0² ∂x z0 (Ginzburg-Landau-type saturation). The test recovers az − bz³ + D∂²z.
- **Runs (CPU, after 9100-9103):**
  - time order 1, k 1 / 2 / 4;
  - time order 2, k 1 / 2;
  - each with kinds pde+dd, pde+cubic, pde+dd+cubic, pde4+dd, pde4+cubic, pde4+dd+cubic, thresholds 0.03 / 0.1, rollout refit to 30 min;
  - k = 8 left out: its library would be 1-8 k columns.
- **Reading:** as addendum 5b, judged on the refit law; in addition, whether a selected dd or cubic term survives the refit with a stable coefficient (the interpretability criterion: [[project-interpretable-latent-pde-goal]] in memory; coefficients in physical units, stable between the one-step and refit fits).

### G0 addendum 5d (2026-11-08, before the runs): FPe with mixed derivatives (BBM / Boussinesq-type law)
User direction: allow products of time and space derivatives. The first-order library is extended with the implicit operator (1 − ℓ²∂²x) ∂t z = F(z) (`+mixed`), the form of BBM / Boussinesq equations. In a horizontal-only description it is also how nonhydrostatic gravity-wave dispersion appears, and it damps the response of the shortest waves without dissipating energy. (Products of ∂t z with spatial derivatives already exist in the second-order-in-time library as v_i ∂ⁿz_j.)
- **A fitting trap, found in the smoke test and fixed before any result.** Fitting ℓ² as a regression coefficient of ∂²x(∂t z) gave ℓ² ≈ −0.3: unphysical, and a near-singular operator whose rollouts exploded. The bias arises because ∂²x(∂t z) at a site contains −2 ∂t z of that site, so the target is regressed on itself; the "R² 0.80-0.88" of that run is this artefact. Now ℓ² ≥ 0 is FIXED during the sparse fit (target (1 − ℓ²∂²x) ∂t z) and scanned. The rollout refit may adjust each channel's ℓ², projected onto ℓ² ≥ 0. Tested on a known BBM-type lattice law: the true ℓ² gives the smallest residual and the coefficients are recovered.
- **Interpretation:** ℓ = √ℓ² × 2 km is a dispersion length, to compare with CM1's vertical scales; with the advection term it gives a wave speed and dispersion relation.
- **Runs (CPU, after 9104-9105):** kinds pde+mixed and pde4+mixed, ℓ² 0.1 / 0.3 / 1.0 sites² (ℓ 0.6 / 1.1 / 2 km), k 1 / 2 / 4 / 8, thresholds 0.03 / 0.1, rollout refit to 30 min.
- **Reading:** as addendum 5b. In addition, ℓ² is chosen by the refit rollout skill on the test storms. This is reported as exploratory, since ℓ² is a hyperparameter scored on test storms; a validation-storm choice would need a rerun. Also whether the refit moves ℓ² far from its starting value, which would mean ℓ² is a stability knob rather than a physical scale.

### G0 addendum 5e (2026-11-08, before the run): FPe-S, does the fitted advection speed track each storm's measured speed?
User-approved test of whether an FPe coefficient carries CM1 physics, independent of the latent's gauge. `scripts/cm1/storm_speed_law.py`:
- **Per-storm fit** (40 storms: 30 training + 10 test, nothing trained, frames 30-330, interior sites): dz0/dt = c0 + a1 ∂x z0 + a2 z0 ∂x z0. **c_eff** = (∂x z0)²-weighted mean of −(a1 + a2 z0), in km/min. It is gauge-free (a constant shift of z0 changes a1, not a1 + a2 z0 at the actual z0).
- **Measured speed** from the truth columns, Theil-Sen slope of centroid position against time (every 5 min): (i) condensate-weighted centroid, (ii) cold-pool centroid (weights max(0, −θ′_sfc)).
- **Reading, fixed now:** the advection coefficient "carries physics" if Pearson r(c_eff, measured speed) ≥ 0.5 for at least one of the two measures, with the bootstrap 95% CI over storms excluding 0 and a permutation p < 0.05; and the OLS slope of measured on c_eff within 0.5-2 (the law's speed is of the right size). r between 0.3 and 0.5 is reported as weak. Below 0.3, or CI covering 0: the advection coefficient does not track storm speed and the 0.12 km/min match was a coincidence.
- Both measures and the smoothed variants (σ = 1) are reported whatever the outcome; storms pooled over train and test, so no tuning on the test set.

### G0 addendum 5f (2026-11-09, before the run): FPe-U, the channel-0 advection law in physical units
User direction: interpretability matters, rollout reliability does not. `scripts/cm1/law_physical_units.py`:
- **Gauge map** θ′_sfc = s z0 + b fitted on the 30 training storms (site means, every 5th frame); s from the geometric-mean slope, with the OLS and inverse-OLS slopes as bounds.
- **Law, one-step OLS, no sparsity, no rollout refit** (frames 30-330, interior sites): dz0/dt = k0 + a1 ∂x z0 + a2 z0 ∂x z0 + ν ∂²x z0. In K: c(θ′) = c0 + c1 θ′ (km/min, with ∂t θ′ = −c ∂x θ′ + ...), ν in m²/s, the source in K/min.
- **Uncertainty:** bootstrap over the 30 training storms; the same law from the 10 test storms with the training map.
- **Comparison:** density-current scaling |c| ∝ √|θ′|, i.e. d|c|/d|θ′| = |c| / (2|θ′|) at the data's mean |c| and mean |θ′| (weighted by (∂x θ′)²).
- **Reading, fixed now:** the law is physically interpretable if (i) the bootstrap 95% CI of c1 excludes 0 with c1 > 0 and c0 < 0 (the colder the pool, the faster the front moves toward −x); (ii) c1 is within a factor 3 of the density-current value, for the geometric slope and both bounds; (iii) the test-storm c1 lies inside the training bootstrap CI; (iv) the retrained seed-1 nested autoencoder (`cluster/saidie/fpc_nested_s1.sh`) gives the same signs and c1 within a factor 2. ν is reported and compared with FP0's ~30 m²/s but not interpreted physically if it is far above (the latent smooths). Failing any of (i)-(iii): the advection coefficient is a fitting artefact of the latent, not a statement about CM1's cold pools.

### G0 addendum 5g (2026-11-09, before the run): FPf-D, distilling a short channel-0 law from the trained propagator
User question: learn the propagator first (with nested dropout), then a PDE from the propagator. The trained LatentLaw of `fpc_nested_stagec` is already a dense explicit PDE; FPe fitted to the encoded data instead. `scripts/cm1/law_distill.py`:
- **Targets** on the encoded states of the 30 training and 10 test storms (frames 30-330, interior sites): the data's centred difference of z0 (FPe-U's target), and the propagator's own dz0/dt = `law.tendency(z)[..., 0]`, which is exact in time and depends on every channel; also the propagator's tendency at perturbed states z + 0.1 sd N(0, 1) (off the data).
- **Libraries:** law4 = [1, ∂x z0, z0 ∂x z0, ∂²x z0] (FPe-U) and law6 = [1, z0, ∂x z0, ∂²x z0, z0², z0 ∂x z0] (the FPf constrained row, 5h). OLS fitted on training storms, R² on both splits (test out of sample). Also reported: R² of the dense law's own z0-alone terms against its full tendency; law4 in θ′ units via the FPe-U gauge map, with a bootstrap over storms.
- **Reading, fixed now** (reference: law4 R² on the data, train 0.19):
  - test R² of law4 on the propagator target ≥ 0.4 → the data fits were **noise-limited**: a short law exists for the model's dynamics even though finite differences hide it;
  - < 0.3 → **closure-limited**: the propagator's own dz0/dt needs the other channels, and distillation cannot give a short law. 0.3-0.4 is ambiguous.
  - The distilled c1 is "consistent with FPe-U" if it lies inside FPe-U's training bootstrap CI.
  - Nothing here is a test of CM1 physics beyond 5f; it decides whether option 1 is worth pursuing.

### G0 addendum 5h (2026-11-09, before the runs): FPf, a constrained closed law for channel 0 learned with the model
User question: learn the propagator and a PDE for it at the same time. Option 2, as a structural constraint rather than an extra loss: `--law-lead z0 --law-lead-nu-pos` (`LatentLaw(lead="z0")`, `tests/unit/test_cm1_lead_law.py`). Channel 0's equation is restricted to
dz0/dt = b0 + a0 z0 + a1 ∂x z0 + ν ∂²x z0 + q z0² + a2 z0 ∂x z0, with ν = softplus(·) ≥ 0, no other channel and no residual. Channels 1-31 keep the full law plus residual. The cap (5) still applies.
- **Runs** (`cluster/saidie/fpf_lead.sh`, one GPU, sequential):
  - `fpf_lead`: Stage B only, reusing `fpc_nested`'s autoencoder (frozen), with the same recipe as `fpc_nested` (law-res-weight 0.1, cap 5, law-l1 1e-5, k-max 45). This measures the constraint's cost with the encoder fixed.
  - `fpf_lead_stagec`: Stage C from it (15000 steps, k 45, w-sfc-th 2, nested 0.5), the same recipe as `fpc_nested_stagec`. Here the encoder can move to a coordinate that obeys the law.
  - Comparisons: `fpf_lead` vs `fpc_nested`, and `fpf_lead_stagec` vs `fpc_nested_stagec`.
- **Reading, fixed now (Stage C run):**
  - (a) **The law is learned, not just imposed:** law4 or law6 OLS on the new encoder's *data* (centred differences, test storms) has R² ≥ 0.4, against 0.19 for `fpc_nested_stagec`. If this fails, the propagator uses the closed law but the encoded z0 does not actually evolve by it, so the constraint is cosmetic.
  - (b) **Cost:**
    - test `reconstruction.nmse_all` within 10% of 0.0287;
    - k=1 (channel 0 alone) reconstruction NMSE within 25% of `fpc_nested_stagec`'s;
    - `forecast.disp_abs_km_median` at 30 and 60 min within 25% of `fpc_nested_stagec`'s.
    - If the cost fails while (a) holds, the result is still a trade-off worth reporting, but not adopted as the model.
  - (c) **Physics:** the learned channel-0 coefficients are converted to θ′ units with the FPe-U gauge map. Reported: c(θ′) = c0 + c1 θ′, ν (m²/s) and the ratio of c1 to the density-current value (3.3-3.5 in 5f). "Closer to CM1 physics" means c1 > 0 and the ratio < 2. Signs and the ratio are reported whatever they are.
  - A seed-1 repeat is run only if (a) holds. The gauge map r is also reported: if channel 0 stops tracking θ′ (|r| < 0.7), (c) is not interpretable.

## E3c: shifted tilings in the DA (2026-11-06, Saidie 8893; 22 test storms × 3 seeds, noise 0.1, held boundaries)
| | Cycled nMSE (prior) | Paired vs single | Storms lower | Analysis nMSE |
|---|---|---|---|---|
| single tiling, 32 members | 0.271 ± 0.021 | — | — | 0.257 |
| 4 tilings (offsets 0/2/4/6 cells) × 32 members | **0.261 ± 0.021** | −0.010 ± 0.002 (−4%) | 95% | 0.245 (−4.6%) |
| 4 tilings × 8 members (equal cost) | 0.286 ± 0.024 | +0.015 ± 0.006 | 9% | 0.275 |

- A small, very consistent gain at 4× the cost; at equal cost the smaller per-tiling ensembles lose more than the seams cost.
- Not yet attributable to seams: averaging 4 independent DA systems also averages sampling noise. Control running (Cashew 749619, `da_tiling.py --control`): 4 systems at offset 0 with distinct model noise, against 4 tilings with the same distinct noise, all on identical observations.

## L1 / L5: finite-time Lyapunov spectrum of the latent model, and perturbation spreading (2026-11-06, Cashew 749449)
full_v1_stagec (deterministic), 6 unseen storms (3 ens_A2, 3 ens_B), full 6 208-dimensional spectrum, boundary sites held, 16-min warm-up. Mean ± sd over storms:

| Window | Linearization | λ1 (per min) | Doubling | Positive exponents | D_KY |
|---|---|---|---|---|---|
| 60–120 min | along the encoded CM1 trajectory | 0.0052 ± 0.0043 | ~130 min | 4 ± 3 | 9 ± 6 |
| 60–120 min | along the surrogate's own run | 0.0009 ± 0.0020 | ~13 h | 1 ± 1 | 2 ± 1 |
| 120–180 min | along the encoded CM1 trajectory | 0.0103 ± 0.0077 | ~67 min | 5 ± 2 | 12 ± 6 |
| 120–180 min | along the surrogate's own run | −0.0010 ± 0.0054 | — | 1 ± 1 | 2 ± 3 |

- **The surrogate is far less chaotic than CM1.** Along its own trajectory it is nearly neutral (0–2 positive exponents, doubling of hours); evaluated on the true states its tangent map is more unstable (4–5 positive, D_KY ~10, doubling ~1–2 h), which grows as the storm matures. CM1's own small perturbations doubled in ~2–3 min during initiation (L2 smoke); the full L2 comparison (749729) will give CM1's rates in the same windows.
- Consistent with the MSE objective (SM): a model trained to forecast the conditional mean is dissipative and damps the small-scale instabilities, and its own trajectory drifts towards smooth states where the tangent map is even more stable.
- **Spreading (L5):** a one-site perturbation spreads at 3.2–3.3 km/min (threshold 1e-3; 2.8–3.9 across thresholds), the same in the storm and in the quiescent far field. That is faster than the data-only reach (1.6–2.4 km/min) and about half the network's maximum (6 km/min), and the lack of a storm/far-field difference says it reflects the network's linear coupling (gravity-wave-like), not convective dynamics.

## L2 / SM5: CM1 error growth from twin runs, against the surrogate (2026-11-06, Cashew 749729; figure `artifacts/cm1/figs/fig_l2_growth.png`)
8 control/perturbed pairs (1e-3 K tiny field at t = 0, ens_A environment, 4 h) + 2 pairs at 1e-4 K. Normalized RMS difference (model units, all channels, whole domain).

| t (min) | 0 | 10 | 20 | 30 | 45 | 60 | 90 | 120 | 180 |
|---|---|---|---|---|---|---|---|---|---|
| twin difference (RMS) | 1.5e-4 | 3.0e-4 | 3.9e-3 | 3.0e-2 | 0.11 | 0.24 | 0.56 | 0.65 | 0.83 |
| independent storms (saturation) | 0.05 | 0.07 | 0.38 | 0.49 | 0.47 | 0.55 | 0.76 | 0.78 | 0.96 |

- **CM1 growth rates** (while < 10% of saturation): 0.13 /min at 5–15 min (doubling 5.4 min), **0.21 /min at 15–30 min (3.4 min)**, 0.08 /min at 30–60 min (8.4 min); later windows are saturated. The 1e-4 K pairs agree after the initial transient (0.18 and 0.08 /min) and lag by ~10 min, as a 10× smaller perturbation should. Tiny errors grow by ~3 orders of magnitude in the first hour and saturate by ~2–3 h.
- **The deterministic surrogate does not amplify a tiny latent perturbation at all** (5e-4 → 1e-3 → 5e-4 over 3 h): 20–60× slower than CM1 even counting L1's along-the-truth exponents. **The stochastic surrogate's sample pairs** start large (0.19 after one step) and follow the independent-storm saturation curve from ~30 min: it represents the uncertainty of forecasting from an encoded state, not the growth of tiny errors.
- **L2c, a rough intrinsic floor:** half the twin nMSE approximates the error of the best possible deterministic forecast. At 120 / 180 min of storm life that is ~0.22 / ~0.35, against our forecasts' 0.27 (60-min lead from 1 h) and 0.36 (120-min lead). So our forecasts are roughly at the floor set by CM1's own chaos, *if* growth from 1 h on resembles growth from t = 0. It may not: the twins start at initiation, the most sensitive phase, and our forecasts start from the much larger autoencoder error at 1 h. **L2b** (perturb at a 60-min restart) is needed before claiming it.

## SM0: where the smoothing comes from (2026-11-06, Cashew 749738; 10 test storms × 2 starts; figure `artifacts/cm1/figs/fig_sm0_spectra.png`)
Variance retained by wavelength band (< 4 / 4–16 / > 16 km) as a fraction of the truth's, ranges over w, θ′, u′, condensate at low and mid levels:

| | 5 min | 30 min | 60 min |
|---|---|---|---|
| reconstruction | 0.27–0.84 / 0.67–0.99 / ~1 | same | same |
| deterministic forecast | 0.10–0.21 / 0.46–0.69 / 0.86–1.00 | 0.00–0.11 / 0.07–0.22 / 0.61–0.95 | 0.00–0.09 / 0.02–0.15 / 0.53–0.90 |
| stochastic sample | 0.18–0.63 / 0.82–1.34 / ~1 | 0.17–0.48 / 0.53–1.54 / 0.83–1.22 | 0.23–0.45 / 0.58–1.14 / 0.54–1.13 |
| stochastic mean (8) | as smooth as the deterministic forecast | | |

Extremes at 60 min (truth / reconstruction / deterministic / stochastic sample): w 99.9% 10.95 / 10.94 / 4.90 / 10.01 m/s; condensate 99.9% 6.01 / 5.77 / 3.80 / 5.62 g/kg; minimum surface θ′ −8.25 / −8.05 / −7.36 / −8.30 K.

- **The smoothing is the propagator's objective, not the representation.** The reconstruction keeps the 4–16 km band (0.7–1.0) and the extremes (w 10.94 against 10.95); the deterministic forecast loses most of the < 16 km variance within 30 min and halves the w extremes. The < 4 km band is partly lost already in the reconstruction (θ′ 0.27–0.31, condensate 0.36–0.53), so the representation sets a floor there.
- **The stochastic samples keep realistic structure and intensity:** the 4–16 km band near or above the truth's, w extremes 10.0 m/s, the cold pool exactly (−8.30 K). Their average is as smooth as the deterministic forecast, as it should be.
- **Consequence:** SM1 (more channels, 1 km sites) can only help the < 4 km band; SM2–SM4 (sample-based objectives, generative propagator) are the lever for sharp, accurate forecasts.
- **Tuning grid extended (2026-11-06, validation storms only, before any claim-3 run on the test storms):** both EnOI baselines got worse from alpha 0.1 to 0.25 (surrogate 0.337 → 0.353, physical archive 0.566 → 0.606), so their optimum is probably below the original grid. Added alpha 0.05 and 0.02 for the EnOI baselines (Saidie 8918, 8919; the alpha 1.0 run, 8903, cancelled) and 0.05 for the observation-only baseline (8920). Observation-only at 0.1: analysis nMSE 0.299 (latent DA's analysis on the same storms: 0.246).
- **Control (Cashew 749619, 22 storms × 3 seeds, identical observations):** 4 independent 32-member DA systems on the SAME lattice (own model noise), averaged: 0.2596 (−0.0113 ± 0.0017, −4.2%, lower on 100% of storms); 4 SHIFTED lattices with the same distinct noise: 0.2575 (−0.0134 ± 0.0018, −4.9%). **The seam effect itself (shifted minus same lattice) is −0.0021 ± 0.0007 (−0.8%), shifted lower on 91% of storms.** So almost all of the tiling gain is averaging several independent ensembles (less sampling noise), not seam removal; the seams are real but small. Implication: the cheap route to the 4% is a larger ensemble (I1), not tilings; I5 (cheap tilings) is not worth building.

## T2: transfer to real two-storm CM1 runs (2026-11-06, Cashew 749808; 8 two-storm truths, 800 km, two lines 400 km apart; model full_v1_stagec trained on single 400 km storms, no retraining)
References: 8 single-storm ens_A2 test storms (same control environment). Unpaired (different truths), boundaries held at the start.

Forecasts, skill (nMSE / persistence) at 5 / 30 / 60 min, and the error ratio two-storm / single:

| Start | Two-storm | Single | Error ratio |
|---|---|---|---|
| 1 h | 0.245 / 0.356 / 0.378 | 0.275 / 0.364 / 0.372 | 0.98 / 0.99 / 1.01 |
| 2 h | 0.281 / 0.396 / 0.409 | 0.211 / 0.385 / 0.384 | 0.97 / 1.00 / 1.04 |
| 3 h | 0.231 / 0.367 / 0.358 | 0.189 / 0.337 / 0.301 | 1.17 / 1.03 / 1.14 |
| 4 h | 0.235 / 0.413 / 0.448 | 0.169 / 0.424 / 0.343 | 1.09 / 0.83 / 1.15 |

DA (θ′ + reflectivity, noise 0.1, 36 cycles from 1 h, so the scored second half, 2.5–4 h, includes the interaction phase), cycled nMSE ± 95% CI:

| N | Localization | Two-storm | Single | Ratio |
|---|---|---|---|---|
| 8 | 5 km | 0.392 ± 0.031 | 0.395 ± 0.030 | **0.99** |
| 32 | 5 km | 0.336 ± 0.023 | 0.337 ± 0.024 | **1.00** |
| 8 | none | 0.477 ± 0.062 | 0.440 ± 0.078 | 1.08 |
| 32 | none | 0.417 ± 0.061 | 0.401 ± 0.031 | 1.04 |

- **Forecasts transfer before the storms interact** (starts at 1–2 h: error ratio 0.97–1.04). From 3 h, when outflows and gravity waves from the two lines meet, two-storm errors are 10–15% higher at 60 min, but skill stays 0.36–0.45 against persistence: the model handles interacting storms it never saw, with a modest penalty.
- **Localized DA transfers perfectly** (ratio 0.99 / 1.00 at N = 8 / 32), including the interaction phase; without localization it is worse in absolute terms (0.42–0.48 against 0.34–0.39) and degrades with two storms (1.04–1.08), most at small N. The ensemble needed per storm does not grow with the number of storms when localized: the CM1 analogue of Paper I's result, now on real CM1 solutions, not the T1 pilot's joined seam.
- **G0 claim 6 met** (localized ratio ≤ 1.1 at every N; at N = 8 the unlocalized ratio is larger, 1.08 against 0.99), and more convincingly than in T1. Caveats: unpaired, 8 storms each (the ratio's CI is roughly ±10%), one seed, two ensemble sizes; θ′ + reflectivity only (the radar geometry was set up for 400 km).

## SM2a and SM4, first results (2026-11-06, Saidie 8912, 8917; 22 test storms)
| Run | Skill 5 / 30 / 60 | CRPS 5 / 30 / 60 | Spread/error 60 | Cold-pool bias | Front 30 / 60 |
|---|---|---|---|---|---|
| deterministic Stage C (full_v1_stagec) | 0.213 / 0.365 / 0.344 | 0.475 / 0.498 / 0.517 | — | −17.4% | 0.62 / 1.39 km |
| stochastic Stage C (CRPS) | 0.267 / 0.404 / 0.385 | 0.342 / 0.344 / 0.360 | 0.93 | −9.9% | 0.87 / 2.02 |
| SM2a: stochastic Stage C, CRPS + energy score | 0.260 / 0.405 / 0.384 | 0.344 / 0.348 / 0.362 | 0.93 | −11.5% | 1.04 / 2.33 |
| SM4: flow-matching propagator (pilot; held boundaries, own eval) | 0.247 / 0.478 / 0.491 | 0.383 / 0.460 / 0.509 | 0.84 | | |

(SM4's own comparison, same cases, held boundaries: deterministic 0.215 / 0.371 / 0.354 and stochastic 0.267 / 0.410 / 0.395, CRPS 0.348 / 0.363 / 0.388.)
- **SM2a: the energy score changes nothing** measurable (skill and CRPS within 0.01 of the CRPS-only model; fronts slightly worse).
- **SM4, the flow pilot, is worse** than the stochastic model at 30–60 min (CRPS 0.46 / 0.51 against 0.36 / 0.39) and under-dispersed (spread/error 0.59–0.84). It is trained on one-step pairs only, with no multi-step rollout training and no Stage C, so its errors compound over a rollout, whereas the stochastic model had the k-curriculum and Stage C. A fair flow model needs rollout training (backpropagating through sampled multi-step rollouts, or a Stage-C-like fine-tuning on decoded samples). Whether its samples are sharper (SM0 diagnostics) is still to check.
- Claim-3 baselines tuned on validation (rule: lowest cycled nMSE; observation-only by analysis nMSE): surrogate EnOI alpha 0.02 (0.321; still falling slowly at the grid edge: 0.337 / 0.329 / 0.321 at 0.1 / 0.05 / 0.02, far from the latent DA's 0.258), physical-archive EnOI alpha 0.05 (0.565; worse than no DA at every alpha), observation-only alpha 0.5 (analysis 0.284; latent DA analysis 0.246). Test-storm runs: Saidie 8947–8949.

## P, first look (2026-11-06, Cashew 750124: small test, 6 training storms, 3 test storms)
- **P1: the latent encodes about as much as a linear PCA of the physical columns (32 components, 92% of the column variance), not more.** Test R² (latent site / latent ±2 sites / PCA site / PCA ±2): log rain rate 0.91 / 0.93 / 0.94 / 0.95; w_max 0.86 / 0.86 / 0.87 / 0.88; log qr low 0.89 / 0.92 / 0.95 / 0.96; log qs aloft 0.92 / 0.93 / 0.93 / 0.95; log qg aloft 0.80 / 0.83 / 0.83 / 0.86; surface θ′ 0.95 / 0.96 / 0.96 / 0.96. **Rain vs cloud water is separable** (AUC 0.943 latent, 0.946 PCA, 0.912 for the physical summary) although the model's inputs lump them. (The summary's R² of 1.0 for w and surface θ′ is trivial: those quantities are in it.)
- **P2: eight latent regimes with clear physics:** a convective core (5% of sites; w 12 m/s, rain 36 mm/h, graupel 2.7 g/kg aloft, θ′ −7.7 K), two raining cold-pool regimes (w 2.6–3.3, rain 5.5 mm/h), a dry cold pool (θ′ −6.4, no rain), anvil (snow aloft 0.2 g/kg, no rain), and quiet air; each persists over 5 min with probability 0.84–0.94.
- **P3 as first written was trivial:** every predictor, including a 6-number physical summary and the surrogate's forecast, scored AUC ≈ 0.98–0.999, because "quiet" candidates include the whole far field, which never rains. Fixed: the main question now uses only quiet sites with the event active within ±20 km (is the storm about to start HERE?); controls in P4 likewise; the far-field version is kept for reference. P7 also gets max-pooling (a mean over the domain cannot represent a storm maximum: pooled-latent R² for storm max w was 0.10). Rerun: Cashew 750202 (test) → 750203 (full).

## P, full run (2026-11-06, Cashew 750203; probes fitted on 30 training storms, scored on the 22 test storms and the 8 two-storm T2 runs; figures `artifacts/cm1/figs/fig_p4_epoch.png`, `fig_p5_traverse_*.png`)
- **P1: the latent ≈ a 32-component PCA of the site columns (92.5% of their variance), slightly below it** on linear readout: R² for log rain rate 0.94 (latent ±2 sites) / 0.96 (PCA ±2), w_max 0.87 / 0.88, log qr 0.92 / 0.96, log qs aloft 0.93 / 0.95, log qg aloft 0.85 / 0.87, surface θ′ 0.96 / 0.96. Rain vs cloud water separable (AUC 0.95 / 0.95; physical summary 0.92) although the inputs lump them, so D15 is not needed for these questions.
- **P2: eight regimes with clear physics, each persisting 0.88–0.93 over 5 min:** convective core (3.5% of sites; w 10.4 m/s, rain 28 mm/h, graupel 1.9 g/kg aloft), heavy trailing rain (w 4.5, rain 12 mm/h, snow 2.2 g/kg, θ′ −8.5 K), light rain over the cold pool (rain 4 mm/h), dry cold pool (θ′ −6 K, no rain), anvil (snow aloft, no rain), and three quiet regimes.
- **P3, the hard question (quiet sites with the event active within 20 km), AUC at 5 / 10 / 20 / 30 min:**

| Event | latent + motion | latent | PCA-32 | physical summary | surrogate forecast | two-storm runs (latent + motion) |
|---|---|---|---|---|---|---|
| rain onset (> 1 mm/h) | 0.977 / 0.973 / 0.960 / 0.952 | 0.972 / 0.970 / 0.956 / 0.948 | 0.976 / 0.971 / 0.957 / 0.946 | 0.928 / 0.933 / 0.932 / 0.919 | 0.954 / 0.948 / 0.942 / 0.938 | 0.969 / 0.963 / 0.942 / 0.930 |
| updraft (> 5 m/s) | 0.928 / 0.922 / 0.909 / 0.904 | 0.924 / 0.918 / 0.905 / 0.898 | 0.920 / 0.918 / 0.909 / 0.902 | 0.941 / 0.924 / 0.904 / 0.892 | 0.930 / 0.905 / 0.874 / 0.862 | 0.900 / 0.892 / 0.877 / 0.879 |
| snow aloft | 0.941 / 0.931 / 0.906 / 0.898 | 0.912 / 0.912 / 0.890 / 0.887 | 0.930 / 0.926 / 0.899 / 0.894 | 0.777 / 0.798 / 0.813 / 0.819 | 0.946 / 0.940 / 0.923 / 0.917 | 0.906 / 0.904 / 0.874 / 0.844 |
| cold-pool arrival | 0.99 for all | | | | 0.98–0.99 | 0.98 |

  Brier skill (latent + motion / PCA): rain 0.57–0.64 / 0.55–0.62; updraft 0.37–0.49 / 0.35–0.49; snow 0.47–0.51 / 0.42–0.50.
  - **Learned latent ≈ PCA of the columns for prediction too** (within 0.01 AUC); both beat the physical summary for rain and especially snow aloft.
  - **Motion adds a little** (+0.005 for rain, up to +0.03 for snow): the latent's recent velocity carries some information beyond the state.
  - **Neighbourhood context matters most:** the site's own latent alone loses 0.08–0.13 AUC.
  - **The surrogate's own forecast is competitive** (best for snow aloft at 5–30 min, a little below the probes for rain and updrafts), so rolling the dynamics forward is as good as a trained snapshot probe.
  - **The probes transfer to the two-storm runs** with a 0.01–0.05 AUC loss.
- **P4 (superposed epoch, 1 799 rain onsets against 1 799 matched near-storm controls):** the latent probe's probability of rain within 10 min rises 0.08 → 0.13 → 0.22 → 0.48 → 0.79 at −30 / −20 / −15 / −10 / −5 min, against 0.03–0.06 for the controls; separated already at −30 min (the earliest lag computed). A real early signal, though part may be that onset sites sit on the storm's advancing side; a longer window and matching on distance to the active region would settle it.
- **P5 (traversal along the rain-onset direction):** moving a dry site's latent along the probe direction adds liquid at 1–4 km and, more strongly, ice at 8–10 km, with a vertically alternating w structure: the onset direction is a deepening cloud with ice aloft (consistent with rain forming through the ice phase), not low-level drizzle.
- **P7 (storm scale from pooled features), R² now / 30 min ahead (pooled latent | PCA | physical summary):** total rain 0.79 / 0.50 | 0.86 / 0.51 | 0.92 / 0.50; minimum surface θ′ 0.92 / 0.83 | 0.93 / 0.85 | 1.00 / 0.86; max w 0.69 / 0.30 | 0.78 / 0.32 | 1.00 / 0.29. For 30-min-ahead storm totals the pooled local latent is as good as either physical baseline, so a unified latent space is not needed for these (P8 not triggered); "now" favours the summary because it contains the targets.

## Q1–Q3: what the propagator alone says about CM1's dynamics (2026-11-06, Cashew 750253; 10 test storms; figures `artifacts/cm1/figs/fig_q1_dispersion.png`, `fig_q2_q3.png`)
**Q1, the quiescent environment.** The homogeneous far-field latent state is close to a fixed point (relative residual 5%). Its stencil symbol (32 modes per wavenumber):
- strongly damped at short wavelengths: growth −0.064 /min at 4–8 km (e-folding ~16 min), −0.027 /min at 8–16 km;
- weakly unstable at long wavelengths: +0.02 /min above 32 km (e-folding ~45 min), a slow drift of the far field;
- almost no neutral (wave) modes, and the few are slow: |c| ≤ 0.46 km/min, against the first hydrostatic gravity-wave speed N H / π = 2.46 km/min (N = 0.0107 /s).
- In nonlinear runs the waves exist (forced by the storm): power-weighted phase speed of far-field w at 5 km 1.04 / 1.04 / 1.25 km/min (surrogate) against 1.24 / 1.31 / 1.86 (CM1) at 8–16 / 16–32 / 32–64 km wavelength: **15–35% too slow, worst for the long, fast waves.**
- So the propagator's linear dynamics of the environment do not carry fast gravity waves: it damps them and propagates the storm-forced waves too slowly. Gravity waves are deterministic and organized, so this is a counterexample to the section's hypothesis that the MSE-trained model gets all the organized dynamics right; the damping of small scales (SM0) extends to fast linear waves.

**Q2, the squall line as a travelling wave (20 mature states, T = 10 min).**
- Best-fit propagation speed, model against the encoded CM1 trajectory: correlation 0.61, within 0.05 km/min in 12 of 20 states, means 0.125 against 0.118 km/min. The model propagates the storm at about the right speed. Neither matches the surface gust-front speed well (correlations 0.12 / 0.31; mean front speed 0.17 km/min): the whole-state shift and the gust front are different things.
- Neither state is a clean travelling wave (residual ~0.6 model, ~0.7 CM1), but **the propagator has a nearly exact travelling-wave solution close to every storm state** (refined with a penalty that keeps it near the storm: residual 0.62 → 0.095).
- Decoded cold-pool strength of those solutions C = 15–29 m/s against 0–2.5 km shear Δu = 7–12 m/s: C / Δu = 1.3–3.6 (mean 2.5), cold-pool-dominated (RKW: > 1 means upshear-tilted, typical of mature squall lines), with no relation to the shear across environments (correlation −0.14). An RKW-like balance is not something the model's steady solutions encode.

**Q3, a regeneration period.** The residual of "repeats after T min up to a shift" rises smoothly with T for both, with no dip at any T from 1 to 30 min: no regeneration period is visible at the whole-domain level, in the model or in CM1. The model's residual is lower than CM1's at every T (0.62 against 0.73 at 10 min): it is more persistent, consistent with its weaker chaos. A period, if present, would need a local (storm-window) version of the test.
- **SM2b (Saidie 8913 → 8914): 8 noise channels instead of 4, then Stage C with CRPS + energy score:** skill 0.259 / 0.408 / 0.384, CRPS 0.343 / 0.349 / 0.363, spread/error 0.98 (against 0.93), cold-pool bias −12.7%, fronts 0.82 / 1.84 km, rain FSS 0.594 (against 0.571). Same accuracy, slightly better calibration, fronts and rain; the stochastic family's mean-accuracy gap (~12% behind deterministic Stage C at 60 min) is untouched by objective tweaks so far.
- **SM2 small scales (Cashew 750266; SM0 diagnostics and SM5 growth on SM2a / SM2b):** single samples keep more fine-scale variance (< 4 km band, mean over w, θ′, u, condensate at two levels, fraction of the truth's, 5 / 30 / 60 min): original stochastic 0.31 / 0.29 / 0.31, SM2a 0.41 / 0.40 / 0.44, SM2b 0.39 / 0.42 / 0.54; the 4–16 km band stays near 1 (SM2b overshoots, 1.34 at 60 min). w 99.9% at 30 min (truth 11.5 m/s): 8.6 / 9.6 / 9.8. SM2b overdoes condensate (6.6 against 6.0 g/kg) and the cold pool (−8.75 against −8.25 K) at 60 min. Ensemble means as smooth as before. Sample-pair spread starts at ~0.2 after one step and saturates within an hour for all three: none reproduces CM1's growth from tiny errors. So sample-based objectives make samples more realistic at small scales at the same accuracy and calibration, without improving the predictability of those scales; the < 4 km band is approaching what the autoencoder represents (θ′ ~0.3), so further gains need SM1.

## Claim 3: latent DA against static and observation-only methods (2026-11-07, Cashew 750274–750276; 22 test storms × 3 seeds, settings tuned on validation, G0 setup)
| Method | Cycled forecast nMSE | Analysis nMSE | Front |disp| | Latent DA − method (paired) | Latent better on |
|---|---|---|---|---|---|
| no DA | 0.469 ± 0.061 | — | — | | |
| **latent LETKF (headline, noise 0.1)** | **0.273 ± 0.022** | **0.260** | 0.38 km | | |
| surrogate EnOI (alpha 0.02) | 0.356 ± 0.027 | 0.357 | 1.09 km | −0.083 ± 0.012 (−23%) | 22 / 22 |
| physical-archive EnOI (alpha 0.05) | 0.578 ± 0.041 (worse than no DA) | 0.564 | 1.53 km | −0.305 ± 0.031 (−53%) | 22 / 22 |
| observation-only analysis (alpha 0.5) | 0.443 (prior = archive mean) | 0.311 | 7.55 km | analysis: −0.051 ± 0.022 (−16%) | 20 / 22 |

**G0 claim 3 met** (≥ 10% better than each, CIs excluding 0). The cycled ensemble beats the same model with a static covariance by 23%; without a model (physical archive) EnOI is worse than doing nothing; the observations alone reconstruct the state at the analysis time 16% worse and put the gust front 7.5 km off.

## Overnight model results (2026-11-06/07, Saidie; 22 test storms, true boundaries)
**SM2c / SM3** (stochastic Stage C variants; reference: stochastic Stage C, CRPS 0.342 / 0.344 / 0.360):
- SM2c (30 000 steps): CRPS 0.336 / 0.349 / 0.366, skill 0.249 / 0.410 / 0.390, spread/error 0.97, fronts 0.79 / 1.77 km: no meaningful change.
- SM3 (+ spectral term on samples): CRPS 0.386 / 0.372 / 0.383, spread/error 0.81: **worse**; not adopted.
- So the stochastic family's ~12% mean-accuracy gap to deterministic Stage C is untouched by objective tweaks (energy score, noise channels, longer training, spectral term).

**SM1** (representation; deterministic, + Stage C; reference full_v1_stagec 0.213 / 0.365 / 0.344, reconstruction nMSE 0.074):
- 64 channels: reconstruction 0.047 (−36%), forecast 0.215 / 0.372 / 0.357.
- 1 km sites: reconstruction 0.051 (−31%), forecast 0.210 / 0.381 / 0.359, fronts 0.65 / 1.40.
- A richer representation reconstructs much better but does not forecast better at 30–60 min (slightly worse): forecast error is set by the dynamics and predictability, not by the representation. It could still lower the DA's floor (untested).

**M1a** (attention propagator on full_v1's autoencoder, + Stage C):

| Run | Skill 5 / 30 / 60 | Rain FSS 60 | Cold-pool bias |
|---|---|---|---|
| conv, seed 0 (full_v1_stagec) | 0.213 / 0.365 / 0.344 | 0.590 | −17.4% |
| conv, seed 1 | 0.212 / 0.368 / 0.342 | 0.577 | −18.0% |
| attention, 128 wide (matched parameters), seeds 0 / 1 | 0.206 / 0.377 / 0.363, 0.206 / 0.378 / 0.356 | 0.567 / 0.591 | −16.5 / −16.8% |
| attention, 256 wide (1.6 M parameters) | **0.188** / 0.369 / 0.348 | **0.639** | −15.3% |

- Seed spread of the conv model is tiny (±0.003). At matched size attention is better at 5 min but worse at 30–60 min by more than that: **not adopted** under the pre-registered rule.
- At 256 wide it is 12% better at 5 min, equal at 30–60 min, with better rain FSS and a weaker cold-pool bias. Under the rule (30 and 60 min) it is not adopted for forecasting, **but the DA cycles every 5 min, where it is clearly better, so it is worth a DA test.**

## Capacity, attention-in-DA, representation-in-DA and the PCA baseline (2026-11-07, Saidie 8966–8994, Cashew 750295–750296)
DA rows: 22 test storms × 3 seeds, θ′ + reflectivity + radial velocity, latent noise tuned on the 6 validation storms (`pick_tuned.py`, lowest validation prior nMSE), paired against the headline (noise_det_0.1, 0.273 ± 0.022).

**M1c, capacity control** (conv propagator 512 wide, 1.64 M parameters, matched to attn256's 1.60 M; one seed):

| Propagator | Parameters | Skill 5 / 30 / 60 | Rain FSS 60 | Cold-pool bias |
|---|---|---|---|---|
| conv 128 (full_v1_stagec; seed 1: 0.212 / 0.368 / 0.342) | 0.43 M | 0.213 / **0.365** / **0.344** | 0.590 | −17.4% |
| conv 512 (m1c_conv512_s0_stagec) | 1.64 M | 0.199 / 0.378 / 0.354 | 0.617 | −15.6% |
| attention 256 (m1a_attn256_s0_stagec) | 1.60 M | **0.188** / 0.369 / 0.348 | **0.639** | −15.3% |

- Capacity explains about half of attention's 5-min gain (0.213 → 0.199 from size alone, → 0.188 with attention) and most of the rain-FSS and cold-pool gains. At 30–60 min neither large model beats the small conv model (conv-512 is 3% worse). Attention adds ~5% at 5 min over a conv model of the same size.

**M1a-DA, attention propagator in DA** (tuned noise 0.05): cycled 0.274 ± 0.024, analysis 0.261; paired against the headline **+0.001 ± 0.011** (lower on 12 / 22), 5 / 30 / 60 min forecasts −0.003 / −0.002 / +0.001. **No DA gain**: the 12% better 5-min forecast does not show up in cycling (observations every 5 min correct what the model gets wrong at 5 min). Spread/error 0.44 against the headline's 0.71 (its validation-best noise is lower). Not adopted.

**SM1-DA, richer representation in DA** (64-channel model, reconstruction −36%; tuned noise 0.05): cycled 0.296 ± 0.023, paired **+0.023 ± 0.006 (worse, 21 / 22 storms)**, 60-min forecasts +0.056. A better autoencoder makes DA worse, not better (more latent dimensions per site for 32 members to constrain). **M1b's gate fails**: attention in the autoencoder is not worth building for DA.

**I-PCA, is the nonlinear autoencoder needed?** **Yes, clearly.**
1. *Linear local-latent model* (PCA-32 per 2 km site, same propagator, Stage B + C, tuned noise 0.02, the bottom of the grid, but validation nMSE is flat there, 0.350 / 0.353 at 0.02 / 0.05):
   - Forecasts: skill 0.306 / 0.420 / 0.399 against 0.213 / 0.365 / 0.344 (44 / 15 / 16% worse); fronts 1.09 / 2.02 km against 0.62 / 1.39; rain FSS 0.49 against 0.59; divergence ratio 0.61 against the truth's 0.22 (PCA cannot keep the anelastic balance).
   - DA: cycled **0.372 ± 0.026**, paired **+0.100 ± 0.012** (worse on 22 / 22), fronts 0.55 against 0.38 km. Still beats no DA (−0.245), but is **worse than both claim-3 baselines that use the learned model** (surrogate EnOI 0.356, observation-only analysis 0.311): the learned autoencoder is what makes the latent DA beat them.
2. *Nonlinear probes* (MLP vs linear, test R²; Cashew 750295): MLPs read both latents better (log rain rate: latent 0.916 → 0.972, PCA 0.948 → 0.977; w_max: 0.847 → 0.950, PCA 0.856 → 0.964), and the gap between the latent and PCA closes to ≤ 0.01 everywhere. So the latent's information is all there, but nonlinearly encoded; the P-probe finding "latent ≈ PCA-32" was a linear-readout artefact. Event prediction (P3): latent and PCA AUCs within 0.01 with or without MLP.
3. *Reconstruction at equal dimension* (22 test storms):

| Model | Dim / site | nMSE | Storm nMSE | w 1 km, < 4 / 4–16 km | Cond 5 km, < 4 / 4–16 km | w 99.9% (truth 11.0) |
|---|---|---|---|---|---|---|
| autoencoder (full_v1) | 32 | **0.022** | **0.071** | 0.73 / 0.97 | 0.48 / 0.92 | 10.78 |
| autoencoder 64 ch (sm1_c64) | 64 | 0.014 | 0.044 | 0.86 / 0.98 | 0.59 / 0.96 | 10.87 |
| PCA-16 | 16 | 0.116 | 0.303 | 0.53 / 0.63 | 0.51 / 0.60 | 10.13 |
| PCA-32 | 32 | 0.061 | 0.170 | 0.59 / 0.91 | 0.65 / 0.68 | 10.56 |
| PCA-64 | 64 | 0.029 | 0.085 | 0.80 / 0.98 | 0.60 / 0.82 | 10.75 |

   The autoencoder at 32 beats PCA at 64 (half the dimension); PCA-32's error is 2.7× larger, worst in the 4–16 km condensate band (0.68 against 0.92). PCA-64 explains 95.4% of the storm-weighted variance.

**M1a-T (part 1), does attention transfer?** T2 two-storm runs with attn256 (Cashew 750296; unpaired, 8 storms, one seed; conv t2_v1 in brackets):
- Forecast error ratio two-storm / single at 5 / 30 / 60 min: before interaction 0.93 / 0.99 / 1.01 (0.98 / 0.99 / 1.01) and 1.03 / 1.00 / 1.13 (0.97 / 1.00 / 1.04); after it 1.23 / 1.03 / 1.22 (1.17 / 1.03 / 1.14) and 1.15 / 0.90 / 1.28 (1.09 / 0.83 / 1.15). Two-storm 60-min skill 0.362 / 0.438 / 0.384 / 0.491 against conv 0.378 / 0.409 / 0.358 / 0.448: worse in 3 of 4 starts.
- Localized DA ratio 1.005 (N = 8) / 1.069 (N = 32) against conv 0.99 / 1.00; absolute two-storm 0.397 / 0.347 against 0.392 / 0.336.
- So attention still transfers (claim 6 holds: ratios ≤ 1.1), but somewhat less well than the conv model, mostly at 60 min after the storms interact (ratio ~1.25 against ~1.15), consistent with state-dependent neighbour weights generalizing worse to configurations not seen in training. Within the ±10% unpaired uncertainty for DA. L1 / L5 spectra and spreading pending (Cashew 750297).

**Summary.** The original small conv propagator on the learned 32-channel autoencoder remains the model: bigger or attention propagators help only at 5 min, which DA does not need; a richer autoencoder hurts DA; a linear (PCA) latent is much worse at everything. The nonlinear autoencoder and locality are both load-bearing.

## SM4b: flow-matching propagator with rollout training (2026-11-07, Saidie 8995)
Pointwise velocity network (reach 3 sites/min), 4 Euler steps, one-step flow-matching pretraining, then the stochastic model's Stage C (decoded 45-step rollouts, 4 samples, fair CRPS, autoencoder updated jointly). Test storms, held boundaries, leads 5 / 30 / 60 min:

| Model | Skill (nMSE / persistence) | CRPS / persistence MAE | Spread / error |
|---|---|---|---|
| flow, SM4b | 0.245 / 0.425 / 0.403 | 0.346 / 0.375 / 0.395 | 0.82 / 0.91 / 0.93 |
| deterministic Stage C | 0.215 / 0.371 / 0.354 | 0.481 / 0.515 / 0.543 | — |
| stochastic Stage C (CRPS) | 0.267 / 0.410 / 0.395 | 0.348 / 0.363 / 0.388 | 0.86 / 0.91 / 0.92 |

Rollout training turned the pilot (worse, under-dispersed) into a model within 5% of the stochastic one in CRPS at every lead, with the same calibration. The SM4b success rule also asks for sharper samples; SM0/SM5 on its samples are not run yet.

## FP0: known-answer pilot for a transport + sources law (2026-11-07, Cashew 750306, 3 min)
Hand-built fields from raw CM1 output; laws fitted on ens_A mem001–004, scored on ens_A2 mem001–002 and ens_B mem003, mem016 (held out); frame pairs every 3 min from 1 to 4 h, dt = 60 s, storm columns ± 10 km. Skill against persistence (1 = perfect, 0 = persistence):

| Variable | Transport (semi-Lagrangian) | + sparse sources | Residual R² (held out) | Terms at threshold 0.05 |
|---|---|---|---|---|
| θ | 0.546 | 0.867 | 0.70 | 7 |
| q_v | 0.784 | 0.937 | 0.71 | 6 |
| θ_il (conserved) | 0.650 | 0.868 | 0.62 | 6 |
| q_t (conserved) | 0.814 | 0.922 | 0.58 | 5 |
| η (vorticity) | 0.878 | **0.947** | 0.56 | **3** |
| q_p (precipitation) | 0.471 | 0.767 | 0.56 | 7 |
| q_cl (cloud) | 0.533 | 0.657 | 0.27 | 6 |

Selected laws (bootstrap frequency ≥ 0.8, coefficients in SI units):
- **η: Dη/Dt = −0.80 ∂B/∂x + 28.5 ∇²η + 1.07 η w dlnρ0/dz.** Three terms, every one physical: the baroclinic source (known −1), the anelastic compression term (known +1), and an eddy viscosity of 28.5 m²/s (plausible for CM1's subgrid mixing at 250 m).
- **q_t: fall-out +5.74 ∂q_p/∂z** (a fall speed of 5.7 m/s, right for rain/graupel), plus mixing (35 m²/s) and small cloud–precipitation terms.
- **θ_il: fall-out −1.83e4 ∂q_p/∂z**, consistent with q_t's: −(Lv/(cp π)) × 5.74 m/s ≈ −1.6e4 to −2.0e4. But also a condensation term (+414): θ_il is not exactly conserved here (ice processes, linearization).
- **θ and q_v: the condensation term is selected with the right signs, but attenuated:** q_v's coefficient 0.22 (known +1), θ's −912 (known −Lv/(cp π) ≈ −2500 to −3500), a consistent factor ~0.25–0.35 for both. Most likely regression dilution: the proxy w · 1[cloud] · dq_sat/dz is noisy (cloud edges, w at scalar points, liquid-only saturation, 1-min steps), and a noisy regressor's coefficient shrinks towards 0.

Against the pre-registered checks: η's dB/dx within ±0.3 of −1 ✓ (−0.80); q_t's fall-out positive at a few m/s ✓ (5.7 m/s); conserved variables transport better than their non-conserved counterparts ✓ (θ_il 0.65 vs θ 0.55; q_t 0.81 vs q_v 0.78, small margin); **q_v's condensation within ±0.3 of +1 ✗ (0.22)**. So the transport, buoyancy and fall-out parts of the pipeline are trustworthy; the condensation proxy is not, and a learned law should not be judged on condensation terms until it is fixed (FP3's 10-s data tests the time-resolution part; a better proxy would use the saturation adjustment itself).

Also: sources explain only 56–71% of the Lagrangian residual even with every term; the rest is unexplained physics or noise at 1-min resolution (FP3 will separate the two).

**SM4b sharpness and growth (Cashew 750307; SM0 and SM5 on its samples, same cases as SM2):**
- < 4 km variance kept by single samples (mean over w, θ′, u, condensate at ~1 and ~5 km, fraction of the truth's) at 5 / 30 / 60 min: **0.63 / 0.73 / 0.85**, against the stochastic model's 0.31 / 0.29 / 0.31 and SM2b's 0.39 / 0.42 / 0.54. The 4–16 km band: 0.93 / 1.13 / **1.53** (overshoots at 60 min, most in condensate, 2.0–2.4×).
- Extremes of single samples, 30 min: w 99.9% 12.5 m/s (autoencoder reconstruction of the truth 11.2; stochastic model 8.6, SM2b 9.8); condensate 6.5 g/kg (reconstruction 5.9); surface θ′ minimum −7.7 K (−7.1). At 60 min condensate 7.4 against 5.8 and θ′ −8.9 against −8.1: too much convection and too cold a pool by then.
- Ensemble means as smooth as the other models' (0.10–0.18 of the < 4 km variance).
- Sample-pair spread from the encoded truth at 1 / 5 / 15 / 30 / 60 / 90 min: 0.18 / 0.33 / 0.45 / 0.60 / 0.69 / 0.80 (stochastic 0.19 / 0.37 / 0.52 / 0.56 / 0.78 / 0.86): the same immediate jump and saturation, no CM1-like growth from small differences.
- **Verdict under its rule (CRPS within 5% with sharper samples): met.** SM4b's samples are the most realistic of any model at 5–30 min (2–2.5× the fine-scale variance of the stochastic model, extremes at the truth's level) at equal CRPS and calibration; at 60 min they overshoot (mesoscale condensate variance, cold pool). Cost: ~4× per step.

## FP1: 2D latent-field autoencoders, reconstruction only (2026-11-07, Saidie 8996, 77 min; 22 test storms)
Config m : stride x : stride z : decoder; "num/pt" = latent numbers per data grid point (column autoencoder full_v1: 0.065, nMSE 0.022 / storm 0.071, divergence ratio 0.236, truth 0.224). Probe = pointwise MLP R² of a hand-built field from φ at the same point.

| Config | num/pt | nMSE | Storm | Div. ratio | Probe θ_il / q_t / η / B |
|---|---|---|---|---|---|
| 1:1:1 jet / point | 1 | 0.051 / 0.102 | 0.144 / 0.273 | 0.69 / 1.09 | 0.40 0.54 0.31 0.73 / 0.75 0.89 0.33 0.86 |
| 2:1:1 jet / point | 2 | 0.014 / 0.050 | 0.042 / 0.133 | 0.37 / 1.03 | |
| 4:1:1 jet / point | 4 | 0.0019 / 0.0091 | 0.006 / 0.029 | 0.28 / 0.54 | 0.86 0.92 0.41 0.84 / 0.97 0.99 0.45 0.96 |
| 6:1:1 jet / point | 6 | 0.0003 / 0.0009 | 0.001 / 0.003 | 0.24 / 0.24 | |
| 2:2:1 jet | 1 | 0.014 | 0.041 | 0.46 | |
| 8:2:1 jet | 4 | 0.0008 | 0.002 | 0.28 | |
| 8:4:2 jet | 1 | 0.0045 | 0.014 | 0.49 | |
| 8:8:4 jet | 0.25 | 0.019 | 0.062 | 0.74 | |
| 4:8:4 jet | 0.125 | 0.027 | 0.089 | 0.81 | |
| 2:8:4 jet | 0.062 | 0.047 | 0.148 | 0.80 | |

- **Derivatives carry variables:** at equal m the jet decoder's error is 2-5x lower, and it keeps the anelastic mass balance (divergence ratio near the truth's 0.22 from m = 4; a pointwise decoder is far off until m = 6).
- **No saturation near the physics count (~4):** error keeps falling with m; reconstruction alone cannot count independent fields (FP1b with propagation is the test).
- **More fields on a coarser grid beat fewer at full resolution** at equal size (1 num/pt: 0.051 at 1:1:1, 0.014 at 2:2:1, 0.0045 at 8:4:2).
- **At the column model's size it loses:** 2:8:4 (0.062 num/pt) 0.047 against the column autoencoder's 0.022; matching it takes ~2x the numbers (4:8:4, 0.027). Flattening the column into channels compresses the vertical better than a field with z as a coordinate.
- Probes: η is poorly read pointwise from φ in every model (0.31-0.66): with the jet decoder vorticity lives in φ's derivatives, not its values. Reconstruction alone does not single out conserved variables (θ_il and q_t are no better read than θ and q_v).

## FP3: does 1-min output limit the tendency fits? (2026-11-07, Cashew 750300-750305)
ens_A mem001, mem002 re-run with 10-s output for 3 h. **Every 6th 10-s frame equals the original 1-min output exactly** (max difference 0.0 for θ, q_v, u, w, q_r in both storms): same storms, bit for bit. FP0 on them (mem001 fits, mem002 held out; 37 pairs from 60-170 min) at dt = 10 / 20 / 30 / 60 s:

| Variable | Transport skill 10 / 20 / 30 / 60 s | + sources 10 / 60 s | Residual R² 10 / 60 s |
|---|---|---|---|
| θ | 0.716 / 0.708 / 0.690 / 0.617 | 0.927 / 0.884 | 0.74 / 0.69 |
| q_v | 0.758 / 0.756 / 0.746 / 0.698 | 0.944 / 0.923 | 0.77 / 0.74 |
| θ_il | 0.691 / 0.687 / 0.677 / 0.634 | 0.915 / 0.884 | 0.72 / 0.68 |
| q_t | 0.808 / 0.807 / 0.803 / 0.779 | 0.947 / 0.925 | 0.72 / 0.66 |
| η | 0.941 / 0.935 / 0.925 / 0.871 | 0.979 / 0.943 | 0.65 / 0.56 |
| q_p | 0.536 / 0.529 / 0.515 / 0.455 | 0.858 / 0.798 | 0.69 / 0.63 |

- Finer output helps modestly, almost all of it between 60 and 30 s (transport +0.03-0.08, with sources +0.02-0.04); 10 s adds little beyond 30 s. **1-min output is adequate for FP2; 30 s would be a small gain.**
- Coefficients are stable across dt: η's dB/dx −0.80 to −0.82, anelastic term 0.87-1.04, eddy viscosity 34-37 m²/s; fall speed 6.3-6.6 m/s.
- **The condensation coefficient is not a time-resolution problem:** q_v's stays 0.26-0.31 at every dt (known +1). It is the proxy. FP0b replaces it with the Lagrangian change of q_sat along trajectories that stay in cloud (saturated air keeps q_v = q_sat, so the known coefficient is exactly +1); queued, Cashew 750310 (8 storms, 60 s) and 750311 (10 s).
- The unexplained part of the Lagrangian residual (residual R² 0.65-0.77 at 10 s) is not mainly time resolution either.

**FP0b: Lagrangian condensation term (Cashew 750310, same 8 storms and protocol as FP0; 750311 on the 10-s storms).** New regressor: the change of q_sat along trajectories that stay in cloud.
- q_v's condensation coefficient **+0.77** (known +1; was 0.22 with the Eulerian proxy), at 10 s +0.67 (+0.06 from the old proxy, still selected). **The fourth pre-registered check now passes (within ±0.3 of +1)**: all four FP0 checks pass.
- q_v with sources 0.937 → 0.961, residual R² 0.71 → 0.82 (10 s: 0.85); θ 0.867 → 0.887, residual R² 0.70 → 0.75 (10 s: 0.81).
- θ's latent heating: −1560 on the new term plus −405 on the old proxy; the ratio to q_v's coefficient, about −2000 to −2500 K, against the known −Lv/(cp π) ≈ −2500 to −3500 K: still ~20-30% short.
- A flag: q_t also selects the new term (+0.48; physically condensation leaves q_t unchanged), so it is partly standing in for something else that happens in cloud (precipitation formation or evaporation); the q_t law is not yet clean.
- η's law unchanged (−0.80 ∂B/∂x, +1.07 anelastic, 28.5 m²/s).

## FP1b: 2D latent fields trained with a short-horizon decoded forecast (2026-11-07, Saidie 8997, 220 min; 22 test storms)
Same sweep as FP1 with the local latent stepper trained jointly (decoded k-step forecast, k ramped 1 → 5 min). Forecast skill = decoded nMSE / persistence's (1-3 h starts); the column model full_v1_stagec under its own protocol: 0.213 / 0.365 / 0.344 at 5 / 30 / 60 min.

| Config | num/pt | nMSE | Probe θ_il / q_t / η | Skill 1 / 5 / 15 / 30 / 60 min |
|---|---|---|---|---|
| 1:1:1 jet | 1 | 0.060 | 0.35 0.60 0.28 | 0.82 / 0.35 / 0.64 / 2.4 / 5.6 |
| 2:1:1 jet | 2 | 0.022 | 0.69 0.76 0.47 | 0.33 / 0.23 / 2.7 / 98 / 814 |
| 3:1:1 jet | 3 | 0.011 | 0.81 0.89 0.50 | 0.21 / 0.19 / 0.61 / 2.4 / 163 |
| 4:1:1 jet | 4 | 0.008 | 0.89 0.93 0.53 | 0.18 / 0.19 / 0.66 / 37 / 1e5 |
| 6:1:1 jet | 6 | 0.005 | 0.90 0.95 0.68 | 0.14 / 0.18 / 1.5 / 93 / 2e6 |
| 2:1:1 point | 2 | 0.063 | 0.81 0.93 0.37 | 0.92 / 0.39 / 0.66 / 1.15 / 1.31 |
| 4:1:1 point | 4 | 0.016 | 0.95 0.99 0.47 | 0.27 / 0.22 / 0.66 / 3.4 / 2e3 |
| 2:2:1 jet | 1 | 0.021 | 0.73 0.77 0.42 | 0.32 / 0.21 / 0.52 / 0.74 / 1.64 |
| 4:2:1 jet | 2 | 0.0075 | 0.88 0.91 0.57 | 0.16 / 0.17 / 0.51 / 0.74 / 2.09 |
| 8:2:1 jet | 4 | 0.0038 | 0.93 0.97 0.67 | 0.11 / 0.14 / 0.83 / 5.2 / 114 |
| 4:4:2 jet | 0.5 | 0.012 | 0.88 0.93 0.49 | 0.20 / 0.14 / 0.40 / 0.62 / 0.73 |
| 8:4:2 jet | 1 | 0.009 | 0.92 0.97 0.64 | 0.18 / 0.14 / 0.59 / 2.0 / 21 |

- **Compression caveat (read first):** data = 7 numbers per grid point, so compression = 7 / num/pt: 7x at m = 1, 1.75x at m = 4 full resolution, 14x for 4:4:2, against the column model's 108x. Most of these latents nearly are the data.
- **Within the training horizon the low-compression field models are very good:** 5-min skill 0.14-0.19 for most configs, numerically better than the column model's 0.213, but at 8-60x less compression. This is not a like-for-like win.
- **Beyond it they are unstable:** every config is worse than persistence by 30 min, most diverge (skill 10²-10⁶ at 60 min). Short-horizon training (k ≤ 5) does not constrain the long rollout; the column model needed Stage C (k 45) for the same reason. More numbers per point → better short forecasts and faster blow-up.
- Coarser latent grids are the most stable (4:4:2, 0.5 num/pt: 0.62 / 0.73 at 30 / 60 min).
- Training with propagation raises the probes of conserved variables (θ_il 0.35-0.90 and q_t 0.60-0.95 rise with m; η stays hard, ≤ 0.68), but no m saturation is visible in the short-range skill (5-min: 0.35 / 0.23 / 0.19 / 0.19 / 0.18 for m = 1-6 at full resolution; it flattens from m = 3).
- **Closing note:** this track (FP2) was dropped for FPc. The lessons that carry over: the 5-min plateau from m ≈ 3-4 fields, and that long-horizon decoded training is needed before any rollout claim.

## SM4c: fixing SM4b's 60-min overshoot (2026-11-07, Saidie 8998; 20 SM0 cases, 22 test storms for CRPS)
All arms start from sm4b_flow_stagec. Ratios are a single sample's band variance (or extreme) over the truth's; CRPS is the absolute CRPS at 5 / 30 / 60 / 90 min. Changes are against T1.0 (SM4b rescored to 90 min).

| Arm | 4-16 km, mean of fields 60 / 90 | Condensate 4-16 km 60 / 90 | < 4 km at 30 | w 99.9 60 / 90 | Cond 99.9 60 / 90 | θ_sfc min 60 / 90 | CRPS change 5 / 30 / 60 |
|---|---|---|---|---|---|---|---|
| T1.0 (SM4b) | 1.53 / 1.30 | 2.20 / 1.90 | 0.73 | 1.14 / 1.10 | 1.24 / 1.13 | 1.08 / 1.12 | 0 |
| T0.85 | 1.20 / 1.06 | 1.67 / 1.26 | 0.56 | 1.14 / 0.98 | 1.19 / 1.00 | 0.99 / 1.02 | +1.0 / +1.2 / +2.0% |
| T0.7 | 1.06 / 0.85 | 1.64 / 1.06 | 0.41 | 0.97 / 0.98 | 1.10 / 1.00 | 0.93 / 0.94 | +3.5 / +4.6 / +7.0% |
| T0.5 | 0.76 / 0.76 | 1.24 / 1.07 | 0.28 | 0.90 / 0.85 | 1.03 / 0.98 | 0.85 / 0.85 | +9.5 / +12 / +17% |
| E8 (8 Euler steps) | 1.74 / 1.67 | 2.09 / 2.72 | 0.85 | 1.32 / 1.21 | 1.33 / 1.17 | 1.12 / 1.18 | +0.9 / +3.0 / +4.5% |
| k90 (Stage C to 90 min) | 1.38 / 1.17 | 1.87 / 1.39 | 0.67 | 1.16 / 1.09 | 1.25 / 1.10 | 1.03 / 1.01 | +0.7 / −0.9 / −0.8% |
| es (CRPS + energy score) | 1.68 / 1.53 | 2.26 / 2.39 | 0.97 | 1.18 / 1.10 | 1.22 / 1.10 | 1.13 / 1.13 | −1.0 / −0.8 / +0.1% |
| k90 + es | 1.62 / 1.44 | 1.97 / 1.86 | 0.87 | 1.32 / 1.20 | 1.29 / 1.09 | 1.07 / 1.12 | +1.0 / +0.2 / +0.4% |

Pre-registered success (all at 60 and 90 min unless stated): 4-16 km 0.8-1.2; condensate's ≤ 1.3; the three extremes within 10%; CRPS ≤ 2% worse at 5-60; < 4 km ≥ 0.6 at 30. **No arm passes (final, all 8 arms).** k90 + es (8998's last arm) is worse than k90 alone on every sharpness measure: the energy score's added variance undoes the longer horizon's gain.
- **Temperature trades sharpness for skill and cannot fix it alone.** T0.85 brings the mean 4-16 km variance into range and the extremes nearly (w 1.14 at 60), at +2% CRPS, but the < 4 km variance drops to 0.56 and condensate stays 1.7×. Lower temperatures cost CRPS fast (T0.7 +7%).
- **Not integration error:** 8 Euler steps make the overshoot worse (1.74 / 1.67). The coarse 4-step sampler was damping a learned step distribution that is itself too wide.
- **Training horizon matters a little:** k90 is the only arm that improves both sharpness (1.53 → 1.38 at 60) and CRPS (−0.9% at 30-60). It is not enough on its own.
- **The energy score does not help:** es adds small-scale variance (< 4 km 0.97) and leaves the mesoscale overshoot unchanged.
- **The excess sits in condensate:** with variance in range for w, θ and u (T0.85), condensate's 4-16 km variance is still 1.6-1.7×. The flow makes too many mid-sized cloud and rain features. Likely next: k90 at T ≈ 0.85, or the mean-anchored hybrid (idea 5); a condensate-specific fix is a candidate if neither works.

## FPc: the column model with an explicit latent law (2026-11-07, Saidie 9017-9031; 22 test storms, starts every hour, `forecast_report` as in train_cm1)
The neural propagator is replaced by dz/dt = A0 z + A1 d1z + A2 d2z + b + Q1[z z] + Q2[z d1z] + R(z, d1z, d2z), Heun, tendency cap 5 (after 8999 diverged), trained Stage B then Stage C with the autoencoder in the loop. Skill is decoded nMSE / persistence's (lower is better); full_v1_stagec has 0.365 / 0.344 at 30 / 60 min.

| Arm | Skill 5 / 30 / 60 | Residual share | Terms > 1e-3 / > 1e-2 | Channels unused (ablation < 1e-3) | Pruning, no retraining: last stable / first unstable (60-min skill > 1) |
|---|---|---|---|---|---|
| reuse (full_v1 AE, L1 1e-5) | 0.227 / 0.377 / 0.362 | 0.014 | 46 761 / 14 334 | 0 of 32 | 46 761 terms (0.371) / 37 072 still 0.745, 14 340 diverges |
| L1 1e-4 | 0.231 / 0.385 / 0.367 | 0.029 | 30 219 / 4 195 | 0 of 32 | 30 220 (0.393) / 18 805 (49) |
| nested dropout AE | 0.244 / 0.377 / 0.357 | 0.011 | 42 724 / 10 231 | 5 of 32 | 49 388 (0.361) / 42 724 (3.3) |
| whole-domain windows (200 sites; arm 4's control, Saidie 9028) | 0.226 / 0.371 / 0.354 | 0.008 | 45 346 / 12 283 | 0 of 32 | 45 346 (0.371) / 34 624 (137) |
| + domain mean G⟨z⟩ (arm 4, Saidie 9031) | 0.221 / 0.365 / 0.348 | 0.008 | 46 254 / 12 781 | 0 of 32 | 46 254 (0.364) / 35 387 (112) |

**Pre-registered reading (plan, FPc): met on skill and residual, not on readability.** Every arm is within 10% of full_v1_stagec at 30 and 60 min (bounds 0.402 / 0.378) with a residual share of 0.01-0.03 (bound 0.3): an explicit local quadratic PDE in the 32 latent channels forecasts CM1 squall lines as well as the neural propagator, and the MLP residual carries only 1-3% of the tendency. **But the law is dense.** 30 000-47 000 of its 52 736 coefficients are above 1e-3, and pruning without retraining has no knee: skill holds until a few thousand to ~30 000 terms are removed, then the 60-min forecast blows up. There is no law of a readable size (hundreds of terms) on this latent. That is the question FPd asks (thresholding during training, with the autoencoder adapting).
- **Nested dropout orders the channels** (1 channel: reconstruction 0.30, 4: 0.15, 8: 0.086, against 1.10 / 0.97 / 0.74 without), at no skill cost (0.377 / 0.357). The largest terms then act on the first channels (d2 z0, d2 z1, z1 d1z1): diffusion-like and advection-like terms of the leading modes. 5 trailing channels are nearly unused (ablation 0.0008-0.001), as expected when the ordering pushes content forward.
- **Linear part about z = 0:** damped at 4-8 km (growth −0.01 to −0.18 /min), weakly growing at 32-128 km (+0.01 to +0.09 /min); fastest phase speeds 0.4-0.5 km/min at 4 km rising to 2.5-3.2 km/min at 128 km. That last value is between the cold-pool speed and the gravity-wave speeds (Q1). z = 0 is the climatological mean latent, not a rest state, so these are indicative only.
- **Arm 4, the domain mean as a global input:** skill 0.365 / 0.348 against the whole-domain local control's 0.371 / 0.354 (1.5-2% better, about two conv seed spreads of ±0.003), and ‖G‖ / ‖A0‖ = 0.90: the 1024 global coefficients are as large in norm as the local linear coupling, yet buy almost no skill. The pre-registered reading (G ≈ 0 and no gain → local; gain with sizeable G → a domain-wide part) is not clean either way: G is sizeable but nearly inert. The likely reason is that the domain mean of each channel varies slowly, so G⟨z⟩ is close to a learned bias (degenerate with b) and L1 1e-5 is too weak to remove it. Read as: no evidence that a domain-wide term is needed; locality stands. A fair test of a domain-wide component would penalize G more strongly or ablate it after training.
- The largest single terms are diagonal second derivatives (d2 z_k on dz_k/dt, +0.10 to +0.19: diffusion of each channel) and first derivatives (advection). Channel couplings are spread thin, and none dominates.

## G3 and claim 2's rank test (2026-11-07, Saidie 9048; 22 test storms x 3 seeds, the G0 headline setup)
The headline reproduces: analysis-cycle prior nMSE 0.274 ± 0.023 (0.273 before), no DA 0.469, latent better on 22 of 22.

**G3, forecasts from the analyses at 90 / 120 / 150 / 180 min.** Leads 1 / 11 / 21 / 31 / 41 / 51 min:

| | nMSE | Front displacement (km) | Reflectivity ≥ 35 dBZ FSS (10 km) |
|---|---|---|---|
| From the analyses | 0.25 / 0.30 / 0.30 / 0.31 / 0.33 / 0.33 | 0.4 / 1.0 / 1.7 / 2.2 / 3.0 / 3.1 | 0.98 / 0.96 / 0.94 / 0.94 / 0.93 / 0.93 |
| Persistence of the analysis | 0.26 / 0.38 / 0.42 / 0.47 / 0.52 / 0.56 | 0.4 / 1.7 / 3.0 / 4.5 / 6.2 / 7.9 | 0.98 / 0.96 / 0.94 / 0.91 / 0.88 / 0.85 |
| No DA (free ensemble) | 0.40 / 0.43 / 0.43 / 0.44 / 0.46 / 0.46 | 6.5 / 6.7 / 7.0 / 7.1 / 7.3 / 7.3 | 0.83 / 0.84 / 0.85 / 0.84 / 0.84 / 0.84 |

DA-initialized forecasts beat both references at every lead after the first minutes. At 50 min they are still clearly better than no DA (nMSE 0.33 against 0.46; the front 3 km off, not 7). They also beat persistence of the analysis by 40% in nMSE, with a front error less than half of persistence's. The forecast model moves the line correctly.

**Claim 2, rank histogram (members + N(0, σ_obs), θ′ at the observation sites; `scripts/cm1/rank_test.py`).**
- **Pre-registered pooled χ²:** rejects flatness, as G0 addendum 3 expected for 237 600 correlated ranks (p ≈ 0).
- **Decorrelated (every 10th site, every 3rd cycle, seed 0; exploratory):** also rejects (n 2 640, p 1.6e-16).
- **The shape:** mildly U-shaped. The outer bins hold 1.34× the flat frequency pooled and 1.11× decorrelated, with mean rank bias +0.02.
- **Without the observation-error perturbation:** a strong U (6.2×), the expected artefact.
- **Reading:** slight underdispersion, consistent with spread/error 0.71. The rank criterion is not met as written: formally rejected, the histogram close to flat. Claim 2 stays half met, and the paper reports it so.

## Overnight 2026-11-07/08: G2b, T2 extension, FPd, and the Fable section-6 sweeps (Saidie 9032-9082, Cashew 750365-750376)
All DA rows: 22 test storms × 3 seeds, the G0 setup, the setting chosen on the 6 validation storms by lowest prior nMSE (`choice.json` in `~/cm1_da/fable/<name>/`). Headline: 0.273 ± 0.022, no DA 0.469.

**G2b, equation-free forecast-model baselines (pre-registered grid + addendum 3b), test at the validation pick:**

| Baseline | Validation pick | Test cycled nMSE | vs no DA | spread/error |
|---|---|---|---|---|
| persistence ensemble | noise 0.05 (grid 0-0.3; val 1.62) | 1.551 ± 0.115 | +1.08 | 0.34 |
| analog, one drawn neighbour | noise 0.05 (val 14.0) | 14.5 | +14.0 | 1.08 |
| analog, mean of 10 neighbours | noise 0.05 (val 6.22) | 6.80 ± 0.83 | +6.33 | 0.99 |

- The latent DA beats both by a factor of 5 or more (the addendum-3 reading: ≥ 10% better, CI excluding 0; met trivially). Both baselines are **worse than no DA**. Persistence without noise is a frozen 1-h ensemble; its spread collapses (0.08) and it cannot follow the storm (1.59). With noise it random-walks off the manifold. Local analogs are incoherent between neighbouring sites even when averaged over 10 neighbours.
- So these are not strong baselines. The informative equation-free comparison stays claim 3 (surrogate EnOI 0.356, physical EnOI 0.578, obs-only 0.311). Report G2b as "a forecast model is needed, and a local look-up is not one". A fair analog would match whole-domain states (Lguensat et al. 2017) or kernel-weight them (Zhao & Giannakis 2016). It is not run.

**T2 extension (claim 6; 16 two-storm truths × 3 seeds, Cashew 750368):** localized DA error ratio two-storm / single 1.002 / 1.008 / 1.020 at N = 8 / 16 / 32 (≤ 1.1 at every N ✓); unlocalized 1.056 / 1.062 / 1.087, larger than localized at N = 8 ✓. **Claim 6 met on the extended set.** Unlocalized errors are 0.43-0.45 against localized 0.34-0.39. Forecasts: error ratio 0.96-1.06 at starts of 1-2 h (before the storms interact), 1.15-1.20 at 60 min from 3-4 h. The single-storm references are the 10 ens_A2 test storms (unpaired).

**FPd (law-shaped latent, Saidie 9032-9038):**

| Arm | Skill 5 / 30 / 60 | Terms kept | Residual share | Unused channels |
|---|---|---|---|---|
| fpd_reuse | 0.241 / 0.394 / 0.374 | 15 356 | 0.000 | 0 of 32 |
| fpd_smooth (+ site smoothness) | 0.241 / 0.396 / 0.375 | 15 403 | 0.000 | 0 |
| fpd_scratch | identical to fpd_smooth (see below) | | | |

- **Pre-registered reading: not met** on the term count (15.4k against ≤ 500). The other three criteria are met: residual share 0.000 < 0.1; 30-min skill 0.394 within 10% of 0.365 (bound 0.402); every channel used. Sequential thresholding during training cut the law 3.4× (52.7k → 15.4k) and closed it (the MLP residual is gone), at +8% error at 30 min. But the remaining terms are all needed: the next pruning level (1% of the largest coupling, 6.2k terms) blows up the 60-min forecast. A readable law is not reached on this latent.
- **fpd_scratch is not a scratch model.** Stage A with the same recipe and seed reproduced full_v1's autoencoder bit for bit on the GPU (identical md5), and every later stage then repeated fpd_smooth. That is a reproducibility check, not a third arm. Rerun with `--seed 1`: Saidie 9089-9091.

**N5, localization radius (noise 0.1):** validation 2 / 5 / 10 / 20 / 40 km: 0.262 / **0.259** / 0.266 / 0.275 / 0.295; test 0.275 / **0.271** / 0.279 / 0.292 / 0.307. The optimum is 5 km on both. **The reach rule's prediction (optimum ≥ 10 km) is not supported:** 10 km is 3% worse than 5 km, and larger radii are monotonically worse. Spread/error falls with radius (0.72 → 0.62). With 32 members and 32 channels per site, sampling noise in the cross-covariances seems to bind before the information reach does. Sobash & Stensrud (2013) found larger horizontal radii best at 2-3 km grid spacing, so this differs from the physical-space EnKF experience.

**N2, adaptive additive noise:** global: val flat at 0.256 for every base level → 0.1, test **0.272 ± 0.022**, spread/error 0.61. Site: val → 0.05, test **0.274 ± 0.022**, spread/error 0.60. Accuracy ties the headline, but **calibration is worse than the hand-tuned 0.1 (0.71), so the pre-registered reading fails** (it needed 0.9-1.1). The likely reason: the innovation ratio pools θ′, reflectivity and radial velocity in units of R, and reflectivity and radial velocity are ~2/3 of the observations. So the ensemble is calibrated in their mixture, not at the θ′ observations, where claim 2 measures it. Calibrating per observation type would be the follow-up.

**N4, ensemble size** (test cycled nMSE at the validation-chosen noise):

| N | 8 | 16 | 32 | 64 | 93 |
|---|---|---|---|---|---|
| 32 channels | 0.305 | 0.287 | 0.277 (0.05) | 0.264 | 0.261 |
| 64 channels | 0.340 | 0.311 | 0.297 | 0.290 | 0.284 |
| 64-channel penalty | +0.035 | +0.024 | +0.020 | +0.026 | +0.023 |

- **The 64-channel penalty does not vanish with N** (+0.02-0.03 at every N up to the full pool), so the pre-registered sampling-error reading is not supported. More latent channels per site make DA worse independently of ensemble size up to 93. The candidate explanation is the observation operator: decoded columns from a 64-channel latent are less smooth in the latent, so the linear (ensemble-space) update fits worse. That is untested.
- The 32-channel error falls 14% from N = 8 to 93 and flattens above 64 (0.264 → 0.261). At N = 32 the validation pick was 0.05 this time (0.258 against 0.259 at 0.1, a tie), so this row is 0.277 rather than the headline's 0.273.

**M1, observation spacing × cycle length, deterministic (noise tuned) vs stochastic (RTPS tuned), paired per storm:**

| obs km \ cycle | 5 | 10 | 15 | 30 |
|---|---|---|---|---|
| 2 | 0.277 / 0.299 (+8%) | 0.291 / 0.317 (+9%) | 0.303 / 0.328 (+8%) | 0.325 / 0.351 (+8%) |
| 4 | 0.286 / 0.312 (+9%) | 0.301 / 0.321 (+7%) | 0.315 / 0.335 (+6%) | 0.333 / 0.354 (+6%) |
| 8 | 0.300 / 0.320 (+7%) | 0.307 / 0.329 (+7%) | 0.319 / 0.345 (+8%) | 0.343 / 0.363 (+6%) |
| 16 | 0.319 / 0.342 (+7%) | 0.333 / 0.353 (+6%) | 0.341 / 0.364 (+7%) | 0.366 / 0.390 (+7%) |

(deterministic / stochastic cycled nMSE; the paired difference's CI excludes 0 in every cell; the stochastic model is lower on 1-7 of 22 storms).
- **Pre-registered reading: no reversal in any cell; the hypothesis is rejected over this range.** Down to 16 km spacing with 30-min cycles, the deterministic model with hand-set noise stays 6-9% better, about the stochastic model's mean-forecast deficit (7-12%). Thinner observations and longer cycles raise both errors by similar amounts (2 km/5 min → 16 km/30 min: +32% and +30%). The stochastic model is calibrated everywhere (spread/error 0.87-1.03). The deterministic one is under-dispersive everywhere (0.42-0.65), because validation always picks the smallest noise in the grid, 0.05, so its optimum may lie lower still.
- Caveat for the interpretation: neither model has CM1's error growth (SM5), so this tests learned noise against hand-set noise, not realistic against unrealistic dynamics. The unstable-subspace argument would need a surrogate whose spread grows like CM1's (M4).

**N6, cost** (`scripts/cm1/cost_table.py`; 32 members, 200 × 32 latent; excludes data loading):

| | One forecast hour | One DA cycle (θ′ + refl + Vr, LETKF) | 3-h DA experiment (36 cycles) |
|---|---|---|---|
| H200 | 0.0095 s | 0.019 s | 0.67 s |
| Cashew CPU, 16 threads | 0.32 s | 0.39 s | 14 s |
| CM1, 32 members, one core each (measured ~14.5 core-min per member-hour) | 7.7 core-hours | | |

That is ~86 000× in wall time on 16 CPU threads, ~5 400× per core. (Both runs crashed after writing `cost.json`, in the provenance call; fixed.)

**M3 (mean-anchored ensemble):** the stochastic propagator on the frozen autoencoder trained (9079-9080). The DA sweep failed on a state-dict comparison of boolean buffers (fixed) and is rerunning: Saidie 9088.
- **M3 result (9088; 22 test storms × 3 seeds):**
  - Validation picks RTPS 0.5: prior nMSE 0.295, spread/error 0.97, so it is calibrated.
  - Test cycled nMSE is **0.312 ± 0.022**, against the headline's 0.274. That is +14%; the pre-registered limit was +2%.
  - **Not adopted.**
  - The frozen stochastic model alone gives 0.307 on test. It is calibrated at validation (spread/error 1.00, RTPS 0.5), but it has the same ~12% accuracy deficit as before.
  - So anchoring the mean to the deterministic model does not recover its accuracy. The anchored mean is advanced from the analysis mean, so the stochastic perturbations, through the analysis update, still steer it.

## FPe results, first batch (2026-11-08, Saidie 9095, 9098, 9100; 30 training / 10 test storms; z0 MSE at 30 min, persistence 0.196)
**Unordered control (9095, fpc_law_reuse_stagec):**
- Channel 0 has no physical meaning: |r| ≤ 0.32 with every diagnostic.
- Held-out R² is ≤ 0.12 for k ≤ 4, and 0.29 at k = 8.
- No law beats persistence.
- **The ordering is what makes a leading-channel law possible.**

**Second order in time (9098, t-smooth 2, no refit):**
- R² for z0 is 0.31-0.34 (k = 1), 0.38-0.46 (k = 2) and 0.51-0.60 (k = 4). That is below first order at every k.
- Every closed rollout explodes (z0 MSE 10²-10⁴).
- Not pursued without the refit; 9101 and 9103 have the refit.

**First order with rollout refit (9100), channel 0 (cold-pool / convection coordinate):**
- r = +0.80 with θ′_sfc, −0.86 with column condensate, −0.72 with w_max.

| k | kind | thr | terms (z0) | R² z0 | one-step law | refit law | refit, decoded |
|---|---|---|---|---|---|---|---|
| 1 | pde | 0.1 | 3 | 0.21 | 1.44 | 0.187 | 0.539 |
| 1 | pde3 | 0.1 | 5 | 0.41 | **0.150** | 0.147 | 0.493 |
| 1 | pde4 | 0.1 | 3 | 0.41 | **0.170** | 0.166 | 0.530 |
| 1 | pde | 0.03 | 7 | 0.22 | 1.33 | 0.095 | 0.461 |
| 2 | pde3 | 0.03 | 20 | 0.60 | 3.8 | 0.076 | 0.408 |
| 4 | pde3 | 0.03 | 43 | 0.69 | 33 | 0.078 | 0.377 |
| 4 | pde | 0.03 | 35 | 0.65 | 19 | 0.079 | 0.371 |

Persistence (decoded, first k channels) is 0.51-0.54. The full model's z0 is 0.034.

- **Pre-registered reading (5b): met.**
  - k = 2, pde3, thr 0.03: 20 terms for z0, held-out R² 0.60.
  - Its refit rollout is 0.076 against persistence's 0.196, 2.6× better.
  - It is still 2.2× worse than the full model.
- **The k = 1 laws with a third derivative are the first one-step fits that beat persistence without a refit.** pde4, thr 0.1, 3 terms:
  - ∂t z0 = +0.073 z0 ∂³x z0 + 0.015 z0 ∂x z0 − 0.004.
  - z0 MSE is 0.170 (persistence 0.196). The refit barely changes the coefficients (0.039 / 0.022 / −0.004) or the skill (0.166).
  - It is stable between the one-step and refit fits, so by the interpretability criterion it is a candidate.
  - **But its leading term needs care.** A wide 5-point ∂³ stencil acting on a front about one site wide behaves like a lagged gradient, so z0 ∂³z0 may stand in for front propagation on an unresolved front, not for physical dispersion. With ∂³ in the library, the advection term z0 ∂x z0 changes sign and shrinks (from −0.083 to +0.015). That points to the two terms sharing one role.
  - **To check before interpreting:** fit the same law on the smoothed field, or with a compact stencil.
- **Refits inflate diffusion, as feared.** With thr 0.03 the refit raises ∂²z0 from 0.018 to 0.20-0.32 sites²/min.
  - That is ~1.3-2.1 × 10⁴ m²/s, against FP0's ~30 m²/s.
  - The one-step 0.018 is ~1 200 m²/s.
  - These refit diffusivities are stability terms, not physical eddy mixing, and should not be reported as such.
- **The k = 1 advection law (3 terms, thr 0.1)** is ∂t z0 = −(0.059 + 0.082 z0) ∂x z0 − 0.004.
  - The base speed is 0.12 km/min (Q2's storm speed ~0.12).
  - The speed rises 0.16 km/min per unit of z0, with warm (z0 > 0) air moving faster.
  - Its closed one-step rollout fails (1.44). The refit stabilizes it but shrinks the speed to 0.04 km/min, so the refit coefficients lose the physical match.
- Pending: 9101 / 9103 (time order 2 with refit), 9102 (k = 8), 9104-9105 (+dd / +cubic), 9108 (+mixed).

**FPd scratch, seed 1 (9089-9090):** trained. The law replication on it is running (9091). At prune ≤ 0.001 it is still all 17 770 terms (skill 0.24 / 0.39 / 0.37 at 5 / 30 / 60 min).

## FPe: smoothed-field check of the z0 ∂³z0 term, and second order in time with refit (Saidie 9109, 9110, 9101; 2026-11-08)
**Smoothing check** (Gaussian over sites, σ = 1 / 2; k = 1, thr 0.1; persistence and the full model are recomputed on the smoothed field, so compare only within a run):

| σ (sites) | kind | R² z0 | one-step law MSE (persistence) | z0 ∂³z0 coefficient, one-step / refit |
|---|---|---|---|---|
| 0 | pde4 | 0.41 | 0.170 (0.196) | 0.073 / 0.039 |
| 1 | pde3 | 0.27 | 0.110 (0.152) | 0.099 / 0.089 |
| 2 | pde3 | 0.14 | 0.302 (0.116) | 0.211 / 0.116 |

- **The term is selected at every smoothing**, the only third-order term in each law, and the advection term z0 ∂x z0 stays small positive (0.02-0.05). So it is not an artefact of one front sharpness.
- **But its coefficient is not a constant:** it doubles from σ = 1 to σ = 2, and one-step skill over persistence is lost at σ = 2. A physical dispersion coefficient would not depend on the smoothing width; this one does. Reading: it describes front sharpening / propagation on a ~1-site feature, not dispersion. **Not reported as a physical coefficient.**
- Without ∂³ in the library the second-order laws lose the advection term at σ ≥ 1 (∂²z0 and z0 ∂²z0 take over).
- Refit at k = 2 beats persistence at σ = 2 (z0 MSE 0.042-0.050 against 0.116), but with the usual inflated diffusion.

**Second order in time with refit (9101, k 1/2/4 done):** k = 1 refit laws are worse than persistence (0.10-1.7 against 0.196, one at 0.10); several k = 2 refits explode (10⁵). First order with refit remains better. Second order is not pursued unless +dd/+cubic changes this.

## FPe-S: per-storm advection speed against measured storm speed (Saidie 9111, 9112; 40 storms; pre-registration 5e)
| smoothing | measure | Pearson r (95% CI) | permutation p | Spearman | slope measured/law | mean law vs measured (km/min) |
|---|---|---|---|---|---|---|
| none | condensate centroid | +0.40 (−0.07, +0.79) | 0.010 | +0.09 | 0.19 | −0.277 vs −0.023 |
| none | cold-pool centroid | +0.35 (−0.26, +0.87) | 0.029 | +0.60 | 0.17 | −0.277 vs −0.286 |
| σ = 1 | condensate centroid | +0.49 (+0.01, +0.79) | 0.001 | +0.19 | 0.35 | −0.074 vs −0.023 |
| σ = 1 | cold-pool centroid | +0.46 (−0.11, +0.84) | 0.003 | +0.44 | 0.33 | −0.074 vs −0.286 |

- **Pre-registered reading: NOT met.** Needed r ≥ 0.5 with the CI excluding 0 and a slope in 0.5-2. Best case r = 0.49 with slope 0.35; the unsmoothed r is 0.35-0.40 with CIs covering 0. By the rule: weak correlation, slope off by a factor 3-6.
- **What is real:** the **mean** effective speed of the unsmoothed law (−0.277 km/min) equals the mean cold-pool speed (−0.286). But the storm-to-storm variation of the law's speed does not track the storms' speeds (slope 0.2). The earlier "+0.12 km/min = Q2 storm speed" match is withdrawn: that was the gauge-dependent speed at z0 = 0.
- Pearson and Spearman disagree (0.35 vs 0.60 for cold pool), so a few storms drive the correlation.

## FPe: k = 8, second order with +dd / +cubic, first order with +dd / +cubic (Saidie 9102-9105; z0 MSE at 30 min after the refit, persistence 0.196)
- **k = 8 (first order):** R² for z0 rises to 0.63-0.82, but only one refit law beats persistence (pde, thr 0.1, 13 terms, 0.18). Eight channels do not close better than two.
- **First order, k = 1:** `+dd` and `+cubic` make **every** pde refit law beat persistence (0.09-0.19) and raise R² a little (0.21 → 0.25; pde4: 0.41 → 0.46). They do not beat the best earlier law (k = 2 pde3, 0.076). Best compact: k = 1 pde+dd, thr 0.1, 7 terms, refit 0.135.
- **First order, k = 2:** pde+dd thr 0.03 (23 terms, R² 0.575) beats persistence even one-step (0.187), refit 0.092. At k ≥ 4 all refits blow up.
- **The KPZ-type term (∂x z0)² keeps its sign:** −0.046 one-step and −0.110 refit in pde+dd; −0.041 and −0.117 in pde4+dd (selected in 70% of bootstraps). The sign is stable across libraries and fits; the refit makes it 2.5× larger. `+cubic` adds nothing: z0² ∂x z0 is small (+0.01 → −0.001).
- **Second order in time:** R² is higher (0.5-0.63 at k = 1) but one-step rollouts all explode, and only a few refits (thr 0.03) beat persistence (0.11-0.18). First order stays better.
- **Pending:** 9108 (+mixed).

## FPe-U: the channel-0 law in physical units (Saidie 9148; pre-registration 5f)
**Gauge map** (30 training storms): θ′_sfc = 3.48 z0 − 1.55 K, r = +0.79 (slope bounds: OLS 2.76, inverse OLS 4.38 K per unit z0). θ′ has mean −2.15 K and sd 3.28 K.

**Law** (one-step OLS, 4 terms), rewritten in K: ∂t θ′ = −c(θ′) ∂x θ′ + ν ∂²x θ′ + S.

| | R² | c0 (km/min) | c1 (km/min per K) | ν (m²/s) | S (K/min) | c_eff at the front (km/min) |
|---|---|---|---|---|---|---|
| train (30 storms) | 0.19 | +0.202 | +0.0490 | −614 | −0.014 | −0.277 at θ′ = −9.8 K |
| bootstrap 95% CI | | +0.168, +0.236 | +0.0462, +0.0517 | −671, −551 | | −0.313, −0.232 |
| test (10 storms) | 0.21 | +0.173 | +0.0476 | −628 | −0.015 | −0.303 at θ′ = −10.0 K |

- **What it says:**
  - The front speed is linear in the cold-pool strength.
  - The speed is zero at θ′ ≈ −4.1 K. Colder air moves toward −x, the storm's direction, and at the front's typical −10 K it moves at −0.28 to −0.30 km/min, the cold-pool centroid speed (FPe-S, −0.29).
  - Ambient air (θ′ = 0) moves the other way, at +0.2 km/min.
  - c1 is tight: CI ±6%, the test storms agree to 3%.
- **ν is negative (−614 m²/s, CI excludes 0):** the one-step law is anti-diffusive. It sharpens the front, so as a PDE it is ill-posed.
  - That explains why every unrefit closed rollout blew up.
  - It also explains why the rollout refit adds large positive diffusion: it compensates.
  - The earlier 7-term sparse fit had +0.018 on ∂²z0, but with z0 ∂²z0 also present; this 4-term fit has no such term.
- **Pre-registered reading (5f): NOT met.**
  - (i) c1 > 0 with CI excluding 0: yes. But c0 < 0 was required, and c0 = +0.20 (CI excludes 0), so (i) fails as written.
  - (ii) Density-current scaling predicts d|c|/d|θ′| = |c|/(2|θ′|) = 0.014 km/min per K. The law's c1 is 0.049, 3.5× steeper, and 2.7-4.4× across the slope bounds: outside a factor 3. Fails.
  - (iii) Test c1 inside the training CI: met.
  - (iv) Seed-1 autoencoder: pending (9149-9150).
- **Reading:** the coefficient is stable and has the physically expected sign of the cold-pool dependence (colder → faster toward the storm's direction). But its size is not √-scaling, and the zero crossing at −4 K means the linear form is an effective fit across the front, not density-current theory. R² 0.2: this is one fifth of channel 0's tendency.

**FPe-U, seed-1 autoencoder (9149-9152; fpc_nested_s1_stagec, same recipe, seed 1):**
- **Ordering reproduces exactly:** nMSE with channel 0 alone is 0.301 (seed 0: 0.296); the curves agree to 0.01 for every k.
- **Gauge map:** θ′ = −3.46 z0 − 1.61 K, r = −0.81. Channel 0 has the OPPOSITE sign, which is the gauge freedom in action.
- **Law in K:**
  - c(θ′) = +0.225 + 0.0578 θ′ km/min (CI of c1 0.0548-0.0610);
  - ν = −963 m²/s;
  - R² 0.24;
  - test storms: c1 = 0.0567, R² 0.27;
  - c_eff at the front −0.34 km/min.
- **Agreement with seed 0:** same form and signs after gauge fixing; c1 is 18% larger and ν 57% more negative. The bootstrap CIs of c1 do not overlap, so the autoencoder spread of the coefficients is ~20%.
- **Criterion 5f(iv):** same signs, c1 within a factor 2: met.
- **Overall 5f: not met.** (i) and (ii) fail on both seeds: c0 > 0, and c1 is 3.3-3.5× the density-current value.
- **FPe-S on seed 1:** r = 0.31 with both measures, slope 0.12. Not met.

## FPf-D: short channel-0 law distilled from the trained propagator (Saidie 9162; pre-registration 5g)
`law_distill.py` on `fpc_nested_stagec`, 30 training / 10 test storms, frames 30-330. Gauge map r +0.794, s +3.48 K.
| target | law4 R² train / test | law6 R² train / test |
|---|---|---|
| data, centred difference (FPe-U) | 0.193 / 0.212 | 0.194 / 0.213 |
| propagator dz0/dt on the data | 0.187 / 0.191 | 0.188 / 0.191 |
| propagator dz0/dt, perturbed states (0.1 sd) | 0.267 / 0.263 | 0.272 / 0.267 |
- The propagator's dz0/dt explains R² 0.47 of the data's centred difference (train and test).
- The dense law's own z0-alone terms explain R² 0.002 of its full channel-0 tendency. The trained law routes channel 0's dynamics almost entirely through coupling to other channels and the quadratic cross terms.
- **Pre-registered reading (5g): closure-limited** (test R² 0.19 < 0.3). The short law fits the propagator no better than the data, so the data fits were not held back by finite-difference noise. Distillation gives no short law.
- The advection coefficient is the same: c1 = +0.0490 km/min per K (bootstrap 95% CI 0.0468-0.0517), inside FPe-U's CI, with c0 = +0.149 km/min.
- ν changes sign: +2915 m²/s (CI 2787-3039) from the propagator target, against −614 from the data. The data fit's anti-diffusion therefore comes from the finite-difference target, not from the model's dynamics. Neither value is physical (FP0 ~30 m²/s).
- **Consequence for 5h:** a closed short law for channel 0 is not present in the current coordinate. Only the encoder-free Stage C (`fpf_lead_stagec`) can test whether one exists in another coordinate.
