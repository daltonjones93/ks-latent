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
