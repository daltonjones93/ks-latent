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

### G0 addendum 5i (2026-11-09, before the runs): FPg, a closed PDE on the n leading channels trained jointly with the dense propagator
User direction: the joint scheme should keep a dense propagator F and add a separate PDE g, with a representation loss, and g should be a function of the n most important channels (the dropout's ordering), not of channel 0 alone. Not the 5h structural constraint (that makes channel 0's row itself the law). `LatentLaw(head_n=n)`, flags `--law-head-n n --law-head-weight w`.
- **Model:** the dense law F is unchanged. A separate head g (same library: 1, z, ∂x z, ∂²x z, z_i z_j, z_i ∂x z_j; no residual, no cap) acts on channels 0..n-1 only and gives dz_i/dt = g_i for i < n.
- **Loss:** w · mean (F_{<n}(z) − g(z_{<n}))² / mean F_{<n}(z)², added to every training loss that already has the law penalty (Stage B and C); gradients flow into both g and F (and into the encoder in Stage C, through z). Over all steps of the rollout, both Heun stages. The sparsity weight law_l1 also applies to g.
- **Runs** (`cluster/saidie/fpg_run.sh`, one GPU, sequential, after 9161): n = 4, w ∈ {0.1, 1.0}. For each: Stage B on the frozen `fpc_nested` autoencoder (`fpg_n4_w*`), then Stage C from it (`fpg_n4_w*_stagec`) with the same recipe as `fpc_nested_stagec`. The weight is not tuned: both are reported, and if both pass the lower weight is the adopted one.
- **Measurement** (`scripts/cm1/law_head_report.py`, test storms): the closure R² = R² of an OLS fit of F_i (i < n) on the head's library built from the first n channels only (fitted on train storms, scored on test), averaged over the n channels; the same on `fpc_nested_stagec` is the baseline (how closed F is without the loss). Also the head's R² against F and against the data.
- **Reading, fixed now (Stage C runs):**
  - (a) **Closure is gained:** mean closure R² on test ≥ 0.70 and ≥ 0.15 above the `fpc_nested_stagec` baseline. If not, the loss does not make the leading channels closed.
  - (b) **Cost:** test nmse_all within 10% of 0.0287; |disp| median at 30 and 60 min within 25% of 0.59 and 1.12 km. If (b) fails while (a) holds, it is reported as a trade-off and not adopted.
  - (c) **The head is the law:** head-vs-data R² on test ≥ 0.4 (it was 0.19 for the 4-term channel-0 law on data), averaged over the n channels.
  - (d) **Physics:** if (a)-(c) hold, the head's channel-0 equation is read in the z0 gauge map's units (θ′ for the z0 terms) and its advection and diffusion terms reported; interpretation is not claimed unless the signs and sizes are consistent with 5f (c1 > 0). A second seed only if (a)-(c) hold.
  - All three of (a)-(c) must hold for "a closed short law exists for the n leading channels"; otherwise the plain statement is that the encoder cannot make them closed at this cost.

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

## FPf Stage B: constrained channel-0 law with the autoencoder frozen (Saidie 9160; pre-registration 5h)
`fpf_lead` (frozen `fpc_nested` autoencoder) against `fpc_nested` (same autoencoder, dense law), test storms:
| run | nmse_all | \|disp\| median 30 / 60 min (km) | main-front error median 30 / 60 min (km) |
|---|---|---|---|
| fpc_nested (dense law) | 0.0267 | 1.01 / 2.15 | 1.00 / 2.25 |
| fpf_lead (closed z0 law) | 0.0267 | 1.75 / 5.12 | 1.38 / 4.38 |
| fpc_nested_stagec (reference for 5h) | 0.0287 | 0.59 / 1.12 | 0.75 / 1.25 |
- In the frozen coordinate the closed law costs forecast skill: displacement error is 1.7× worse at 30 min and 2.4× worse at 60 min.
- The learned channel-0 law (sites, frames) is almost pure diffusion: b −0.0044, a0 +0.0037, a1 +0.0048, ν +0.066 (≈ 4400 m²/s), q +0.0020, a2 +0.0064.
  - The advection is about 0.01 km/min, against storm speeds of about 0.3 km/min.
  - With only z0 available, the best rollout-trained closed law smooths channel 0 instead of propagating it. This agrees with 5g's closure-limited result.
- Not a 5h decision (the decision is on Stage C), but it is the baseline Stage C must improve on: can the encoder move so that z0 propagates by itself?

## FPg baseline: closure of the dense law on its 4 leading channels, no loss (Saidie 9180; pre-registration 5i)
`fpc_nested_stagec`, OLS of F_i on the 39-term library of channels 0-3 only, fitted on train storms, scored on test:
| channel | closure R² test | F vs data R² test |
|---|---|---|
| 0 | 0.509 | 0.477 |
| 1 | 0.360 | 0.453 |
| 2 | 0.490 | 0.530 |
| 3 | 0.446 | 0.608 |
| mean | **0.451** | |
- Four leading channels together are much more closed (0.45) than channel 0 alone (0.19 for the 4-term law, 5g), and the largest terms are advection-like (z0 ∂x z0, z0 ∂x z1) and diffusion (∂²x z0, ∂²x z1).
- 5i (a) therefore requires closure R² ≥ 0.70 on the trained runs (baseline + 0.15 = 0.60, so the absolute bar binds).

## FPf Stage C: closed channel-0 law with the encoder free (Saidie 9161, fit 9181; pre-registration 5h)
`fpf_lead_stagec` against `fpc_nested_stagec`:
- Cost: nmse_all 0.0294 vs 0.0287 (+2%); |disp| median 0.81 / 1.48 km at 30 / 60 min vs 0.59 / 1.12 (+37% / +32%). The k=1 reconstruction NMSE was not measured.
- Gauge map r +0.821 (channel 0 still tracks θ′).
- Law fitted to the encoded **data** (centred difference of z0): law4 R² train 0.097 / test 0.109, law6 0.099 / 0.111, against 0.19 / 0.21 before. The channel-0 equation the propagator uses (its own tendency) is fitted by law4 with R² 0.92 and by law6 with R² 1.00 (it is the law by construction), but the propagator's dz0/dt explains R² −0.15 of the data's centred difference.
- Law the propagator uses, in θ′ units (law4 on its own tendency): c(θ′) = −0.061 + 0.0017 θ′ km/min, ν = +6190 m²/s: a smoother, with almost no advection (c_eff −0.076 km/min against the −0.28 measured for the cold pool).
- **Pre-registered reading (5h): (a) fails** (data R² 0.11, bar 0.4): the constraint is cosmetic, the encoded z0 does not evolve by the law the propagator uses. **(b) fails** (displacement +37% / +32% against a 25% limit). (c): c1 = 0.0017 km/min per K is far below the density-current 0.0042-0.009 value and the FPe-U value; not interpretable. No second seed.
- Reading: forcing channel 0 alone to be closed does not make the data's z0 closed; rollout training satisfies the constraint by smoothing channel 0 and moving the advection into other channels. Next: FPg (5i), where g acts on the 4 leading channels and is matched to the dense law by a loss rather than imposed on it.

## FPg, weight 0.1: closed PDE head on the 4 leading channels (Saidie 9176/9177, report 9184; pre-registration 5i)
`fpg_n4_w01_stagec` (n = 4, w = 0.1) against `fpc_nested_stagec`, test storms:
| channel | closure R² (OLS on channels 0-3) | head vs F | head vs data | F vs data |
|---|---|---|---|---|
| 0 | 0.819 | 0.789 | 0.399 | 0.390 |
| 1 | 0.890 | 0.878 | 0.264 | 0.317 |
| 2 | 0.985 | 0.984 | −0.139 | −0.087 |
| 3 | 0.996 | 0.995 | 0.308 | 0.330 |
| mean | **0.922** (baseline 0.451) | | 0.208 | |
- Cost: nmse_all 0.0300 (+4.5%); |disp| median 0.65 / 1.54 km at 30 / 60 min vs 0.59 / 1.12 (+10% / +37%).
- **Pre-registered reading:** (a) **met**: mean closure 0.922 (≥ 0.70 and 0.47 above the baseline). (b) **not met**: the 60-min displacement error is 37% worse (limit 25%); nmse and the 30-min error are inside. (c) **not met**: head-vs-data R² averages 0.21 (bar 0.4); channel 0 reaches 0.399, channel 2 is negative.
- Reading: the loss does make the leading channels closed (0.45 → 0.92) at a small reconstruction cost, but closure is a property of the propagator, not of the data. The dense law's own tendency on channels 0-3 explains the data's centred difference less well than before (channel 0: 0.48 → 0.39; channel 2: 0.53 → −0.09), so the model has become closed by moving away from the observed one-step change of some channels. A closed law for the model's leading channels exists; it is not yet a law for CM1's data.
- Not adopted. Weight 1.0 (9178/9179) pending; no second seed (c fails).

## FPg, weight 1.0, and the physical reading of the heads (Saidie 9178/9179, reports 9190-9192; pre-registration 5i)
**w = 1.0** (`fpg_n4_w1_stagec`): closure R² 0.997 / 1.000 / 1.000 / 1.000 (mean 0.999), but F-vs-data R² on test is −0.21 / −0.62 / −0.14 / +0.13 and head-vs-data averages −0.22. Cost: nmse_all 0.0307 (+7%), |disp| median 0.83 / 1.77 km at 30 / 60 min (+41% / +58%). **(a) met, (b) not met, (c) not met.** Neither weight is adopted. Stronger weight buys closure by detaching the model from the data's one-step change (the head's channel-0 equation is then dominated by ∂²x z0: 0.96 of the tendency's sd, ν ≈ 12900 m²/s).

**What the leading channels carry** (`law_head_physical.py`, correlation of z_i with site means over the training storms, every 5th frame; w = 0.1 / w = 1.0 runs agree to ±0.04):
| channel | θ′_sfc | w max | w min | u′_sfc | condensate | q_v,sfc |
|---|---|---|---|---|---|---|
| z0 | +0.82 | −0.69 | +0.65 | −0.22 | −0.83 | +0.65 |
| z1 | −0.51 | −0.20 | +0.04 | −0.50 | −0.07 | −0.49 |
| z2 | −0.10 | −0.29 | +0.27 | +0.22 | −0.25 | −0.13 |
| z3 | −0.41 | +0.05 | −0.01 | −0.19 | 0.00 | −0.36 |
- z0 is a cold-pool / precipitation-core amplitude (cold surface, rain, updraft/downdraft together). z1 is a surface-outflow variant (θ′ and u′_sfc, q_v), z3 a θ′-q_v pattern, z2 weakly updraft/condensate. They are partially mixed (a latent has no unique physical meaning), so these are labels, not definitions.

**The head's channel-0 equation** (w = 0.1): z0-only terms explain R² 0.095 of g_0 without refit; the largest terms by sd share are ∂²x z0 (0.59), ∂x z0 (0.56), then couplings z1 ∂x z2, z0 ∂x z2, ∂x z1, ∂²x z1, z0 ∂x z1 (0.29-0.33). Channel 0's tendency is mostly diffusion/advection of z0 plus advective coupling to z1 and z2. In θ′ units the z0-only terms give c(θ′) = −0.276 + 0.0070 θ′ km/min, ν = 7300 m²/s. At w = 1.0 they give c(θ′) = −0.189 − 0.0052 θ′ and ν = 12900 m²/s.
- The sign of c1 flips between the two weights (+0.0070 vs −0.0052), and FPe-U's data fit gave +0.049; so the advection's dependence on θ′ is **not** a stable property of the closed head. The w = 0.1 value is within a factor 2 of the density-current scale (0.009-0.014), but that was noticed after the fact, the z0-only terms carry little of g_0, and the sign is not reproduced at w = 1.0: not claimed.
- The robust statements: (i) the leading channels are physically recognisable; (ii) their dynamics is advection + diffusion with advective coupling between them; (iii) the c1 > 0 advection found in FPe/FPf-D from the data and from the dense propagator is the only θ′-dependence of the speed that is reproduced across the data and the unconstrained model.

## FPe mixed-derivative run 9108 (partial: 37 of 48 fits; pre-registration 5d)
Complete for k = 1, 2, 4 (36 fits), one k = 8 fit; the remaining k = 8 fits (about 7 h each) cannot finish in the 36 h limit. Test storms, z0 error at 30 min, persistence 0.196.
- **One-step laws** beat persistence in 5 of 37 fits; **after the rollout refit** 35 of 37 do (0.067-0.18; the exceptions: one at 0.218, one blow-up at 4920). An earlier remark that this was the first time refit laws beat persistence was wrong: ordinary libraries reach 0.045-0.19 after the same refit at k ≤ 4 (unsmoothed 0.076-0.19); at k = 8 they blow up.
- **No systematic gain over the ordinary libraries** (mixed best 0.067 at k = 4 pde4, l2 0.3, thr 0.03; ordinary best unsmoothed 0.076 at k = 2 pde3, 0.078 at k = 4 pde3). One real effect: pde4 at k = 4, thr 0.03 blows up without the mixed operator (16235) and gives 0.067 / 0.074 / 0.087 with it.
- **Refit ℓ² (channel 0 only; other channels' ℓ² were not saved):** median 0.51 sites² (ℓ ≈ 1.4 km), IQR 0.19-0.82, range 0.08-3.6 over 35 fits; within a factor 2 of the starting value in 40%; correlation with the start 0.20 (logs), with refit skill 0.09. **Pre-registered reading (5d): ℓ² is a stability / fitting knob, not a physical scale.** The choice of ℓ² is scored on test storms, so exploratory.
- Terms most often among the top 6 of channel 0's refit law (35 fits): d2tz0 (33), z0 ∂x z0 (18), z0 ∂²x z0 (15), z1 ∂²x z1 (15), ∂x z0 (13), ∂²x z0 (11).
- Data copied to the Mac: only the small result JSONs (`docs/fpe_report/data/`); figures from `scripts/cm1/fpg_report_figs.py`.

### FPe control (2026-11-10): the channel-0 library fitted directly to site-mean surface θ′
Rule stated before the run: the latent adds nothing if its c1 lies inside the θ′ fit's bootstrap 95% CI and the R² values are within ~0.05; it differs if c1 lies outside the CI or its R² is clearly higher. `scripts/cm1/law_theta_control.py` (Saidie job 9217, CPU, 236 s): same storms, frames, interior sites, centred-difference target, library [1, ∂x θ′, θ′∂x θ′, ∂²x θ′], 1000-storm bootstrap.

| | R² train / test | c0 | c1 (km/min per K) | ν (m²/s) | c_eff (km/min) |
|---|---|---|---|---|---|
| θ′ directly | 0.26 / 0.28 | +0.065 (CI 0.029, 0.104) | +0.103 (CI 0.096, 0.110) | −2344 (CI −2574, −2140) | −0.41 (CI −0.45, −0.37) |
| latent channel 0 (FPe-U) | 0.19 / 0.21 | +0.202 | +0.049 (CI 0.046, 0.052) | −614 | −0.277 |

- **Verdict by the rule: the latent differs.** Same signs and form (c1 > 0, anti-diffusive one-step ν), but c1 is 2.1× larger on θ′ directly, the CIs are far apart, and θ′ R² is higher (0.26 vs 0.19).
- **Not a statement about which is right.** The θ′ c_eff (−0.41) overshoots the measured cold-pool centroid speed (−0.29, FPe-S); the latent's (−0.277) matches it. z0 carries only r² ≈ 0.65 of θ′, so errors-in-variables and the latent's smoothing both bias the latent's c1 low; I have not separated these.
- **Consequence:** c1 = +0.049 is a property of the latent coordinate expressed in θ′ units, not of θ′. The only claim the control supports is that the qualitative form (speed increasing with cold-pool strength) appears in both.
- Exploratory: one autoencoder, one seed.

## Interpreting the propagator through what it does to physical fields (2026-11-10)
Why: term-level PDE fitting has reached its ceiling. Channel 0's own terms explain 0.002 of its propagator tendency (closure-limited), and the θ′ control shows the channel-0 coefficients belong to the latent coordinate, not to θ′. The physics is in the coupling between channels, so the interpretation has to be done on the propagator's action in physical space, which is gauge-free (decode ∘ propagate ∘ encode). Five analyses, in order of priority:
1. **Physical response function (Green's function).** Perturb a real state by a small physical θ′ bump, encode, roll the propagator, decode, subtract the unperturbed rollout. Gives spreading speed, amplification, decay and linearity as a function of distance and lag, and of where the bump sits relative to the gust front. Truth side: the data's front speed and ambient advection; paired CM1 perturbation runs would be the real truth and are not available yet.
2. **Linearisation and modes.** Jacobian of the one-step latent map at a storm state; eigenvalues in physical units (growth rate, frequency), eigenvectors decoded to θ′, phase speed ω/k. Truth side: DMD of the encoded data over the same window, compared in the data's POD subspace.
3. **Jacobian Lyapunov spectrum against ensemble spread.** The propagator's exponents are gauge-invariant; compare with how fast the truth ensemble members diverge (ens_A and ens_A2: same environment, different noise).
4. **Adjoint sensitivity.** Gradient of a physical outcome (storm speed, front position at +30 min) with respect to the initial physical field through the decoder; check against squall-line theory (RKW balance).
5. **Channel clamping.** Hold channel k fixed during a rollout and decode, to see which physical fields lose their dynamics; the physical counterpart of the dropout ordering.
Status: 1 and 2 are run now (`scripts/cm1/prop_response.py`, pre-registered below); 3-5 queued.

### G0 addendum 5j (2026-11-10, before the run): propagator response function and linearisation
Model `fpc_nested_stagec`; the 10 test storms; base frames t0 ∈ {120, 180, 240} (mature line). A case is used only if the surface θ′ minimum is below −2 K (a cold pool exists) and the perturbation sits clear of the halo.
**Experiment 1.**
- Perturbation: Gaussian bump of σ = 2 km in θ′ over the lowest 6 levels with linear taper, amplitudes ±0.5 K and ±0.1 K, at offsets {−40, −20, −10, −4, 0, +4, +10, +20, +40} km from the gust front x_f (steepest |∂x θ′_sfc| of the truth; negative offset = ahead, since the storm moves toward −x). The bump amplitude is in K at the surface level; the 6-level taper is my choice and is not tuned.
- Response R(x, k) = site-mean surface θ′ of the perturbed minus the unperturbed rollout (halo held at the true state in both), lags k up to 60 min. Lag 0 is the autoencoder's own round trip, so gain G(k) = ‖R(k)‖/‖R(0)‖ uses the decoded lag-0 difference, not the input amplitude.
- Metrics: response centroid velocity (linear fit of the |R|-weighted centroid over k = 5-30 min), gain, linearity L = ‖R₊ + R₋‖/‖R₊ − R₋‖ at k = 15, and amplitude scaling S = (‖R(0.5)‖/0.5)/(‖R(0.1)‖/0.1) at k = 15.
- Truth reference: the data's front speed v_f (linear fit of the steepest-gradient position over t0..t0+30 min).
- **Pre-registered readings:**
  - (a) At the front (offset 0), the response centroid velocity is within 30% of v_f in at least 70% of cases. Met means the propagator moves a front perturbation with the front.
  - (b) Linear: median L < 0.3 and median S in 0.7-1.4.
  - (c) Reported without a pass mark, since no paired truth exists: gain G(k), response speed ahead of and behind the front, and spreading width. Paired CM1 runs (same member, one bump) on Cashew would be the truth for these; they are not run here.
  - Failing (a): the propagator does not carry front perturbations with the front, whatever its forecast skill.
**Experiment 2.**
- Jacobian J of the one-step latent map over the interior sites (halo fixed at truth) at z(t0), same storms and frames; eigenvalues λ → growth rate σ = ln|λ|/Δt and frequency ω = arg λ/Δt per minute; top-10 modes by σ decoded to θ′ (small-amplitude difference), dominant wavenumber by FFT, phase speed |ω|/(2π k), localisation by participation ratio.
- Truth: exact DMD of the encoded data over [t0, t0+30 min], POD rank r = 8; the tangent operator is projected onto the same POD basis, Ĵ = ΦᵀJΦ, and the 4 leading DMD eigenvalues (by mode amplitude) are matched to the nearest Ĵ eigenvalue.
- **Pre-registered reading:** matched with |Δσ| < 0.05 /min and |Δω| < 0.05 rad/min for at least 70% of the leading modes, and a median match distance |λ_J − λ_DMD| below 0.5 × the distance of the persistence null (λ = 1). Met means the propagator's linearisation reproduces the data's slow linear dynamics in its dominant subspace.
- **Limit I know about in advance:** DMD describes the trajectory, J the tangent map; they coincide only for the slowly translating part of the flow. Failure therefore does not show that the dynamics are wrong; success shows the translation/slow-decay structure is right. Also reported with no pass mark: number of unstable eigenvalues, maximum growth rate, e-folding time.
Exploratory: one autoencoder and propagator, one seed; test storms only.

### 5j result (2026-11-10, Saidie job 9218, 1821 s, 30 cases = 10 test storms × t0 ∈ {120, 180, 240}; data in `docs/fpe_report/data/prop_response.json`)
**Experiment 1 (response function), against the pre-registered readings:**
- **(a) not met, and the reference was bad.** The front detector (steepest |∂x θ′_sfc|) gave a data front speed of +0.20 km/min (median; IQR 0.17-0.28) with outliers above +1, the opposite sign to the independently measured cold-pool speed of −0.29 km/min (FPe-S). So "within 30% of v_f" holds in 0/30 cases, and the test says nothing about the propagator. Against −0.29 the response centroid at the front (median −0.58 km/min) is within 30% in 5/30. I have not changed the detector and re-scored: that would be a post-hoc choice. A corrected front definition needs its own pre-registration.
- **(b) met, but weakly informative.** Linearity L = 0.02 median (p90 0.06) and S = 1.000. The perturbations are small enough that the propagator responds linearly, as any smooth map does; this licenses reading the Jacobian (experiment 2), not more.
- **Autoencoder floor (important).** The decoded lag-0 difference keeps only 5-40% of the input bump (40% at −40 km ahead, 17% at the front, 5-9% behind the front). A 2 km surface-layer bump is largely outside what the autoencoder represents, so the gains below are relative to the decoded amplitude, not the input.
- **(c) descriptive, no truth.** Response centroid velocity at amplitude 0.5 K (median, km/min; negative = toward −x): −0.94 at −40 km (ahead), −0.77 at −20, −0.21 at −10, −0.40 at −4, −0.58 at 0, −0.22 at +4, −0.17 at +10, −0.47 at +20, −0.61 at +40. Gain at 30 min relative to the decoded lag 0: 0.27-0.61 ahead of and at the front, 2.7-4.0 at +4 to +20 km behind it. Every response moves toward the storm's direction, including 40 km ahead at ~1 km/min; I cannot say whether that is physical without paired CM1 runs.
**Experiment 2 (linearisation vs DMD): not met.**
- **Match with |Δσ| < 0.05/min and |Δω| < 0.05 rad/min:** 41% of 120 leading DMD modes (need ≥ 70%). **Median match distance** 0.076 against 0.148 for persistence: ratio 0.51 (need < 0.5). Both marginal misses; the propagator's tangent map is closer to the data's linear dynamics than persistence is, but not by the registered margin.
- **Spectrum of J** (6208 interior degrees of freedom): median 19 unstable eigenvalues, maximum growth rate 0.035/min median (0.155 maximum), e-folding about 28 min. The leading unstable modes are near-stationary (median |ω| 0.009 rad/min) and localised (participation ratio about 15 sites ≈ 30 km). Their wavenumber from the decoded field sits in the lowest FFT bin (0.008 cycle/km), so the phase speed (0.52 km/min) is not resolved; I would not quote it.
- DMD subspace holds 94% of the window's energy, so the comparison is not limited by rank.
**What this does and does not show.** The propagator responds linearly to small physical perturbations, with weak, localised, slow-growing unstable modes (28 min e-folding): the right kind of object for a squall line, but nothing here checks the numbers. The pre-registered truth checks failed (a, with a faulty reference) or narrowly missed (exp 2). The decisive truth is paired CM1 runs (same member, one bump), which would also remove the autoencoder-floor problem if the bump is placed through the model's own representation (e.g. a perturbation of the encoded state decoded to θ′). Exploratory: one model, one seed.

### G0 addendum 5k (2026-11-10, before the runs): paired CM1 runs as truth for the propagator's response and growth
Why: 5j's truth checks were weak (no paired runs; a bad front detector). This builds the real truth: same storm, same restart, with and without a bump.
**Infrastructure (tested, Cashew job 759910).** CM1 patch `bump_restart_var5` (cluster/cm1/patches): after a restart is read, var5 adds a Gaussian θ′ bump (amplitude K, centre cell var6, σ = var10 m, lowest nint(var11) levels, linear taper 1 − (k−1)/nlev), and var3 adds the tiny hashed field of `tiny_pert_var3`. Both zero leaves CM1 unchanged. Smoke test: restart from a 300 s restart, two controls bitwise identical, bump and tiny branches differ.
**Design.** Four test storms (ens_A mem029, mem030; ens_A2 mem001, mem002) re-run from t = 0 with the bump-patched build to 7260 s, restart written at 7200 s (t0 = 120 min). The re-run is the truth: build-level differences grow to member-to-member size by ~2 h (B1), so the cached members at 120 min are not those storms. Branches restart from the same file to 10800 s, 1-min output: ctrl; +0.5 K bumps (σ 2 km, 6 levels) at offsets {−20, −10, 0, +10} km from the gust front; −0.5 K at 0; +0.05 K at 0; tiny fields 1e-3 and 1e-4 K.
**Front definition.** Chosen by calibration on the 40 cached storms at t0 ∈ {120, 180, 240} (job 9229), BEFORE any paired run: the candidate (gradient, or the −0.5 / −1 / −2 K crossing walked from the cold-pool minimum toward −x) with the highest fraction of speeds within 30% of the measured −0.29 km/min. If none reaches 60%, offsets are measured from the cold-pool minimum site instead. The calibration reference (−0.29 km/min) is independent of the paired runs.
**Model-side input.** The propagator is started from the CM1 states, not from an abstract bump: u_pert(t0) = u_ctrl(t0) + the same bump (same centre, σ, taper) added to the θ′ channels, encoded, rolled with the true halo, decoded. This includes the autoencoder floor, so the comparison is the model's response against CM1's response to the same physical input.
**Pre-registered readings** (median over 4 storms × the 4 positive-amplitude offsets, surface θ′ site means):
- (a) Pattern: spatial correlation between the model's response R_model(k) and CM1's R_CM1(k) at k = 15 min ≥ 0.5 means the model reproduces where the perturbation goes; 0.3-0.5 weak; below 0.3 no.
- (b) Amplitude: ‖R_model‖ / ‖R_CM1‖ at k = 15 within [0.5, 2] in at least 70% of cases.
- (c) Speed: response-centroid velocity (k = 5-30 min) of the model within 0.15 km/min of CM1's, in the median.
- (d) Reported, no pass mark: CM1's own linearity (±0.5 K at offset 0; 0.5 vs 0.05 K), which tells whether comparing with a linearised model is fair.
- (e) Growth (item 3): CM1 finite-size growth rate of the normalised all-field difference between tiny-field and control branches over 5-15, 15-30, 30-60 min (both amplitudes; they should agree if the growth is linear), against the propagator's finite-time Lyapunov exponent along its rollout from the same state. Met: within a factor 2 in the 15-30 min window, median over the 4 storms.
Four storms, one model, one seed: exploratory.

### 5k calibration result and the rule applied (2026-11-10, Cashew 759935, 40 storms × t0 ∈ {120, 180, 240} = 120 cases)
Front speed over t0..t0+30 min by detector: gradient +0.19 km/min (IQR 0.16-0.25, 0% within 30% of −0.29); −0.5 K crossing −0.91 (IQR −1.03 to 0, 0%); −1 K crossing −0.91 (−1.10 to 0, 0%); −2 K crossing −1.00 (−1.25 to 0, 0.8%). None reaches 60%, so by the registered rule the paired-run offsets are measured from the **cold-pool minimum site**, not from a front. Two things I note and do not act on: the leading-edge detectors give about −0.9 km/min (a plausible squall-line gust-front speed, ~15 m/s) while −0.29 km/min was the cold-pool *centroid* speed (FPe-S), so my calibration target was probably the wrong quantity for a leading edge; and the IQR upper end of 0 means a quarter of the edge-detector tracks do not move (stuck at the search bound). The fallback needs no front, so the paired runs go ahead with it.
Saidie was refusing connections at the time (job 9229 duplicate, not used); calibration ran on Cashew with the same script.

### 5k amendment (2026-11-10, before the comparison): model used
Saidie is unreachable, and the user directed to use Cashew only. `fpc_nested_stagec` (the model of 5j) lives on Saidie, so the paired comparison runs on the best model on Cashew, `ens_A_cpu_k45`: conv stencil propagator (16 latent channels, 3 layers, radius 1, replicate padding), 2 km sites, halo 3, trained on ens_A mem000-027, 1-min frames, k_max 45. Its test members are mem029-030, which are two of the four paired storms; the two ens_A2 storms were not in its training. Consequences: the pre-registered readings (a)-(e) are unchanged, but the result is about this model, not the nested-dropout law model of 5j, and the two sets of numbers must not be pooled. The same comparison on `fpc_nested_stagec` is a rerun once Saidie is back and the checkpoint can be copied to Cashew.

### 5k result (2026-11-10, Cashew job 759987, model `ens_A_cpu_k45`; 4 storms × 4 positive-amplitude offsets = 16 response cases; data `docs/fpe_report/data/paired_k45.json`)
**Truth infrastructure checks.** CM1 restart is exact (control at 7260 s equals the base run's frame bitwise, all variables). CM1's own response is linear at 0.5 K (L15 = 0.03-0.05; 0.5 vs 0.05 K scaling 0.95-1.00), so comparing with a linearised model is fair. CM1's growth from the two tiny amplitudes agrees (e.g. 15-30 min: 0.218 vs 0.248 /min), so it is the linear regime.
**Response (a)-(c), against the registered readings:**
- **(a) met, narrowly:** median spatial correlation at 15 min = 0.52 (0.54, 0.51, 0.55, 0.43 at offsets −20, −10, 0, +10 km). It falls to 0.2-0.35 at 30 min and ~0 at 60 min, so the model gets the early pattern but not the later one.
- **(b) not met, 0 of 16 in [0.5, 2]:** ‖R_model‖/‖R_CM1‖ at 15 min is 0.02-0.15. Most of this is the autoencoder, not the dynamics: it retains only about 5% of the bump at lag 0 (floor 0.02-0.09). Relative to that retained amplitude the model keeps 0.6-0.8 of its response at 15 min ahead of and at the cold-pool core, and amplifies it 6.6× behind it (+10 km). The behind-the-core amplification was also seen in 5j with a different model.
- **(c) not met:** median |Δv| of the response centroid = 0.28 km/min (per offset: 0.24, 0.08, 0.32, 0.29; the −10 km offset is within the 0.15 tolerance, the others not). Centroid speeds: model −0.92 / −0.75 / −0.72 / −0.31, CM1 −0.83 / −0.91 / −0.63 / −0.65 km/min at −20 / −10 / 0 / +10 km. Both move toward −x at ~0.6-0.9 km/min, so the direction and rough size are right for three of four positions; the +10 km (behind the core) speed is not.
**Growth (e), item 3: not met by about a factor 10.**
- **CM1:** finite-size growth of the all-channel difference from a tiny θ′ field 0.12-0.25 /min at 15-30 min (e-folding 4-9 min), 0.00-0.15 at 5-15, 0.03-0.09 at 30-60.
- **Propagator:** leading finite-time Lyapunov exponent along the rollout, 15-30 min: 0.010-0.040 /min (median ratio to CM1 = 0.10); negative at 0-15 min (−0.03 to −0.045 /min); the same tiny field encoded and rolled grows at −0.01 to +0.04 /min. The registered mark was within a factor 2.
- **What it means:** the deterministic propagator is an order of magnitude too stable compared with CM1's own error growth at the mature stage. That fits an ensemble of this model being underdispersive and needing stochastic forcing or inflation, but I have not tested that here. The CM1 numbers also agree with the earlier t = 0 twin rates (0.13-0.21 /min), so the mature-stage growth is not much slower than at initiation.
**Reading.** The model gets the early direction and speed of a perturbation's spread roughly right, but it barely sees the perturbation (autoencoder) and does not amplify perturbations like CM1 does. Exploratory: four storms, one model, one seed; the nested-dropout law model has not been run through this comparison (Saidie was unreachable).

### G0 addendum 5l (2026-11-10, before the run): adjoint sensitivity of the storm's propagation (item 4), checked against the paired CM1 runs
Chosen because it is the remaining item with truth available (the 4 paired storms × 4 bump positions) and runs on Cashew. My expectation, stated before the run: a clean, interpretable map concentrated near the cold-pool edge, and agreement with CM1 at the level of 5k (a) (pattern right, amplitude lowered by the autoencoder floor). I do not expect it to fix the growth deficit. `scripts/cm1/prop_sensitivity.py`, model `ens_A_cpu_k45` (Cashew), the same 4 base states (t0 = 7200 s).
- **Outcomes** (surface θ′ site means from the decoded state at +30 min, with c(x) = softplus(−(θ′ + 1 K), β = 2) as the cold-pool weight; the 1 K offset keeps ambient noise from diluting the centroid; amended before the run): O1 = cold-pool centroid position Σ x c / Σ c (km, + = toward +x); O2 = total cold-pool deficit Σ c (K·sites).
- **Sensitivity:** the gradient of O with respect to the input state through encoder, 30 forced-halo propagator steps and decoder, projected on the 5k bump (σ = 2 km, 6 levels, taper), giving S_O(s) = the linear response of O to a unit 1 K bump at site s. Maps over all sites saved.
- **Truth:** ΔO_CM1 = O(bump branch at 30 min) − O(control at 30 min), the same functionals on the CM1 surface θ′ site means; model prediction ΔO = 0.5 K × S_O(s) at the 4 branch positions (−20, −10, 0, +10 km from the cold-pool minimum), plus the model's own finite-amplitude rollout difference as a linearity check.
- **Pre-registered readings** (16 cases, per outcome):
  - (a) sign agreement of ΔO_pred with ΔO_CM1 in at least 75% of cases whose |ΔO_CM1| is above 10% of its largest value;
  - (b) magnitude: |ΔO_pred / ΔO_CM1| within [0.5, 2] in at least 50% of those cases (the autoencoder floor, 2-9% of the bump at lag 0, will lower it; the ratio is reported with and without that floor);
  - (c) theory reading, qualitative: S_O1 near the cold-pool edge (offsets −10 and 0) is positive in at least 75% of storms (a warm anomaly at the pool's leading edge weakens it and moves its centroid toward +x, i.e. slows the storm's motion toward −x), and |S_O1| is largest within 20 km of the cold-pool minimum.
Exploratory: four storms, one model.

### 5l result (2026-11-10, Cashew job 760008, `ens_A_cpu_k45`, 4 storms × 4 positive-amplitude positions = 16 cases; data `docs/fpe_report/data/sens_k45.json`)
**Control outcomes (model vs CM1 at +30 min):** cold-pool centroid 157.1 / 156.3 / 123.4 / 155.1 km against CM1 157.7 / 157.6 / 123.0 / 155.0 (within 1.3 km); deficit 266.9 / 267.7 / 165.5 / 249.5 against 267.6 / 253.6 / 205.7 / 225.4 (0.3%, 6%, 20%, 11% off).
- **(a) met.** Sign agreement of the predicted with the CM1 outcome change: O1 (centroid) 12 of 13 cases above the 10% threshold (0.92); O2 (deficit) 15 of 15 (1.00). O2's sign is a weak test: any warm bump reduces a cold-pool deficit, so it is always negative.
- **(b) not met.** |ΔO_pred/ΔO_CM1| has median 0.050 (O1) and 0.037 (O2); none is in [0.5, 2]. The median autoencoder retention of the bump at lag 0 is 0.047, so the shortfall is about the size of the floor and I cannot separate the autoencoder from the propagator here; the 5k response numbers (ratio relative to the floor 0.6-0.8) say the propagator transmits most of what the autoencoder lets through.
- **Linear:** the finite-amplitude model rollout changes O by the gradient's prediction (ratio 1.00-1.01), so the gradient map is the model's response to this perturbation size.
- **(c) not met as written.** S_O1 (km per K) at −10 km is positive in 4 of 4 storms (+0.005, +0.006, +0.028, +0.006), at −20 km positive in 4 of 4, at +10 km negative in 4 of 4, but at 0 km it is mixed (+0.002, −0.002, −0.059, −0.004). The registered rule was positive at offsets −10 and 0: 5 of 8 = 62% < 75%. The largest |S_O1| is not within 20 km of the cold-pool minimum: it sits 72-78 km ahead (toward −x) for three storms and 48 km for the fourth, with maximum |S| 0.11-0.15 km/K. I do not know why; it is where the cold-pool weight is marginal (θ′ near −1 K), where a small change in θ′ moves the centroid most, which would make it a property of the outcome definition rather than of the storm.
- **CM1's own response** (for reference): the centroid moves toward +x for bumps ahead (+0.05 to +0.49 km at −20/−10 km) and toward −x behind (−0.06 to −0.20 km at +10 km), mixed at 0; the deficit falls by 0.15-2.4 K·sites. The model's signs follow this but its sizes are 20-30× too small.
**Reading.** The model's sensitivity has the right sign structure for the centroid (ahead/behind), but the registered theory check on the cold-pool edge fails at the core, the magnitude is about the autoencoder floor, and the maximum sensitivity sits in a place I cannot explain. Exploratory: four storms, one model.

### G0 addendum 5m (2026-11-10, before the run): does a higher autoencoder floor change the paired-run comparison?
Same paired storms, same scripts (5k `paired_compare.py`, 5l `prop_sensitivity.py`), model `d34_c32_kuniform` instead of `ens_A_cpu_k45`: the same conv propagator and data, but 32 latent channels instead of 16 (k-uniform rollout sampling); its test members are mem029-030 as before. Question: if the autoencoder keeps more of the bump, does the model's response amplitude (5k (b)) and sensitivity magnitude (5l (b)) move toward CM1's, and does the growth deficit (5k (e)) change?
- **Prediction, stated before the run:** the lag-0 retention rises (from ~0.05) but stays well below 0.5; the response amplitude ratio at 15 min stays below 0.5 in most cases (so (b) still not met); the leading Lyapunov exponent at 15-30 min stays within the same order as before (0.01-0.04 /min, not within a factor 2 of CM1's 0.12-0.25); pattern correlation (a) stays about 0.5.
- **Reading:** a change counts only if the median retention at least doubles AND the response/sensitivity ratios scale with it (ratio per unit retention within 30% of the 16-channel model's). Otherwise the amplitude shortfall is not a channel-count effect. Growth counts as changed only if the 15-30 min Lyapunov median reaches 0.06 /min (half CM1's lowest).
Exploratory: one more model, same four storms.

### 5m result (2026-11-10, Cashew 760009 / 760010, `d34_c32_kuniform` against `ens_A_cpu_k45`; data `docs/fpe_report/data/{paired,sens}_d34.json`)
| | 16 channels (k45) | 32 channels (d34) |
|---|---|---|
| lag-0 retention of the bump (median) | 0.047 | **0.184** (3.9×) |
| response amplitude ratio at 15 min (5k b) | 0.053, none in [0.5, 2] | 0.089, none in [0.5, 2] |
| response ratio per unit retention | 0.68 | 0.54 (−21%) |
| pattern correlation at 15 min (5k a) | 0.52 | 0.46 |
| median \|Δ centroid speed\| (5k c) | 0.28 km/min | 0.23 |
| leading Lyapunov, 15-30 min (median; per storm) | 0.015 (0.040, 0.015, 0.010, 0.016) | **0.006** (0.025, −0.010, −0.017, 0.023) /min |
| sensitivity O1: sign agreement / ratio (in [0.5, 2]) | 0.92 / 0.050 (0 of 13) | 0.85 / 0.195 (2 of 13) |
| sensitivity O1 ratio per unit retention | 1.07 | 1.06 |
| sensitivity O2: sign / ratio | 1.00 / 0.037 | 0.93 / 0.066 |
| sensitivity O2 ratio per unit retention | 0.79 | 0.36 |
| model control centroid vs CM1 (km) | within 1.3 | within 6.7 (storm memB001: 129.7 against 123.0) |
- **Reading under 5m.** Retention rose 3.9× (the "at least doubles" condition is met). The ratios scale with it for the response amplitude (per-retention −21%, inside 30%) and the O1 sensitivity (1.07 vs 1.06) but not for the O2 sensitivity (0.79 → 0.36). So most of the amplitude shortfall behaves like an autoencoder floor: with the retained amplitude as the unit, the propagator transmits about 0.5-1 of it in both models. But even at retention 0.18 the absolute ratios stay below 0.2, so (b) is still not met by either model.
- **Growth: not changed.** The 15-30 min Lyapunov median is 0.006 /min, far below the 0.06 mark (CM1 0.12-0.25); two of four storms are negative. The deficit is not a channel-count effect. The model's own tiny-field growth was 0.056 and 0.035 in two storms and ~0 in the others, a different measure that I do not read as growth.
- **The sign structure of the sensitivity is not stable across models.** S_O1 at the +10 km position is negative in all 4 storms for the 16-channel model and positive in all 4 for the 32-channel one; at the core it is mixed in both; at −20 and −10 km it is positive in both, and larger (0.03-0.22 vs 0.005-0.056 km/K) for 32 channels. The maximum |S_O1| is still 46-68 km ahead of the cold-pool minimum. Pattern correlation falls from 0.52 to 0.46, so (a) is no longer met.
- **Prediction check (stated before the run):** retention rose but stayed below 0.5 ✓; response ratio stayed below 0.5 ✓; Lyapunov stayed in the same order ✓; pattern correlation stayed near 0.5 ✓ (0.46, slightly lower).
**Reading.** The amplitude gap is largely the autoencoder, and wider latents shrink it only in proportion; the growth gap and the unstable sign structure are properties of the dynamics models, not of the bottleneck. Exploratory: two models, four storms.

### G0 addendum 5n (2026-11-10, before the run): growth-matching fine-tune of the propagator on the paired runs
Question: can the propagator's missing perturbation growth (leading Lyapunov 0.006-0.015 /min against CM1's 0.12-0.25) be trained in from the paired tiny-perturbation runs, and does it transfer to storms it was not trained on?
- **Method** (`scripts/cm1/growth_train.py`): model `d34_c32_kuniform`, autoencoder frozen, propagator only. Windows of 12 steps from CM1 lag k0 in 10-40 min: encode the control frame and the tiny-branch frame, roll both with the control's latents as halo, decode; loss = (log[D_m(j)/D_m(0)] - log[D_cm1(k0+j)/D_cm1(k0)])^2 (ratios, so the autoencoder floor on the amplitude drops out) + the decoded control-rollout MSE against the CM1 control (anchor, normalised by its starting value). 600 steps, lr 1e-4. Two folds over the four storms: train on {A029, B001}, test on {A030, B002}, and the swap.
- **Held-out measures:** (i) the window growth loss and the median log-growth slope per step against CM1's, (ii) the anchor MSE, (iii) the full paired comparison (5k) on the fine-tuned checkpoint: leading Lyapunov 15-30 min and the response metrics on the held-out storms only.
- **Prediction, stated before the run:** the training-storm growth loss falls by more than half; on the held-out storms it falls by less (I expect 20-50%) because four storms of one regime is little data; the anchor MSE rises by under 20%; the held-out leading Lyapunov at 15-30 min stays below 0.06 /min (the loss acts on finite decoded differences, the Lyapunov is a tangent-space quantity, and the autoencoder may not carry the growing modes); the response amplitude ratios (5k b) stay below 0.5.
- **Reading:** growth counts as learned only if the held-out window growth loss falls by at least 50% AND the held-out leading Lyapunov at 15-30 min reaches 0.06 /min in both folds, with the anchor MSE up by less than 20%. A drop in window loss without the Lyapunov rising means the model learned the finite-size growth of CM1's own differences (a fit to the data) but not an unstable direction of its own dynamics. A held-out loss that does not fall means it is memorising the training storms.
Exploratory: one model, four storms, two folds.

### 5n result (2026-11-10, Cashew 760089/760090 training, 760121/760122 paired comparison; data `docs/fpe_report/data/{growth_train,paired_d34_growth}_f{1,2}.json`)
Held-out storms only unless marked. Fold 1 trains on A029+B001 and tests on A030+B002; fold 2 the swap. 600 steps, propagator only.
| | fold 1 (test A030, B002) | fold 2 (test A029, B001) |
|---|---|---|
| window growth loss, train storms | 1.17 → 0.011 | 0.92 → 0.017 |
| window growth loss, held-out | 0.92 → 0.50 (−45%) | 1.17 → 0.65 (−44%) |
| median log-growth slope per step, held-out (CM1 ~0.10) | 0.005 → 0.044 | −0.027 → −0.001 |
| anchor MSE (decoded control rollout), held-out | 0.104 → 0.136 (+30%) | 0.128 → 0.158 (+23%) |
| leading Lyapunov 15-30 min, held-out storms, before → after (/min) | A030 −0.010 → 0.060; B002 0.023 → 0.068 | A029 0.025 → 0.135; B001 −0.017 → −0.003 |
| leading Lyapunov 15-30 min, train storms, after | A029 0.115, B001 0.107 | A030 0.004, B002 0.053 |
| leading Lyapunov 0-15 min, all storms | −0.035…−0.043 → −0.023…−0.002 | same pattern |
| response amplitude ratio at 15 min (median over cases) | 0.057-0.09 → 0.055-0.087 | 0.057-0.09 → 0.046-0.07 |
| response pattern correlation at 15 min, held-out | A030 0.37 → 0.62; B002 0.54 → 0.62 | A029 0.28 → 0.37; B001 0.89 → 0.47 |
- **Reading against the rule stated before the run:** growth counted as learned only with (i) held-out growth loss down by ≥50%, (ii) held-out Lyapunov at 15-30 min ≥0.06 in both folds, (iii) anchor up by <20%. (i) is not met (44-45%), (iii) is not met (23-30%), (ii) is met for 3 of 4 held-out storms (fold medians 0.064 and 0.066, but B001 stays at −0.003). So the pre-registered criterion is **not met**.
- **What did move.** The leading exponent over 15-30 min rises on every held-out storm (by 0.05-0.11 /min), to 0.06-0.135 /min, the lower end of CM1's 0.12-0.25. The rise is concentrated in 15-30 min, which is where CM1's own growth is concentrated (CM1 rates: 0.06-0.15 at 5-15, 0.12-0.25 at 15-30, 0.03-0.09 at 30-60), so the model has picked up some of the timing of the growth, not a uniform exponent. Over 0-15 min the exponent is still negative.
- **What did not.** Training-storm window loss falls to ~0.01, held-out only to ~0.5: the fit is mostly memorisation of two storms. The train-storm Lyapunov is not uniformly high either (A030 0.004 in fold 2). Response amplitude is unchanged (the autoencoder floor), and pattern correlation moves both ways (+0.25 on A030, −0.42 on B001). The control forecast gets worse on held-out storms by 23-30%.
- **Prediction check:** held-out loss fell by 20-50% ✓ (44-45%); anchor up <20% ✗ (23-30%); held-out Lyapunov stayed below 0.06 ✗ (3 of 4 reached it, median over the four 0.064); amplitude ratios stayed below 0.5 ✓.
**Reading.** A finite-size growth target can be partly trained into the propagator, and it appears as a larger tangent-space exponent on storms it never saw, so the earlier deficit is not forced by the autoencoder. But with four storms and 600 steps it costs forecast accuracy and does not transfer reliably (B001). Whether more storms would fix both is untested; the paired data cannot supply them, and each new storm needs the CM1 restart pair on the supercomputer. Exploratory: one model, four storms, one seed per fold.

### G0 addendum 5o (2026-11-11, before the run): channel clamping in propagator rollouts (item 5)
Question: which latent channels does the propagator's dynamics depend on, and which physical fields lose skill when a channel's evolution is removed? Script `scripts/cm1/prop_clamp.py`. From a real state at t0 (10 test storms × t0 ∈ {120, 180, 240} min), roll the propagator 30 steps with the true latents as halo and decode at lags 5/15/30 with the channel set K (a) free, (b) FROZEN at its t0 value, (c) FORCED to the encoded truth (teacher forcing of K only). Error per physical variable v ∈ (u, w, θ′, q_v, p, q_liq, q_ice) = decoded-vs-true mean squared error in normalised units (interior sites). Frozen damage D(K) = err_frozen/err_free; forced gain G(K) = err_forced/err_free. K = single channels 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31 and the groups 0-3, 8-11, 28-31. Models: `fpc_nested_stagec` (nested dropout, ordered) and a control with no ordering, `d34_c32_kuniform`, scored on ten ens_A2 members it never saw.
- **Predictions, stated before the run.** (i) Ordering: for the ordered model the Spearman correlation across the single channels between k and log D (all-variable error, lag 30) is ≤ −0.6; for the unordered control |ρ| < 0.4. (ii) Channel 0 in the ordered model: D > 2 at lag 30, and θ′ is among the two variables with the largest D. (iii) Group 28-31 in the ordered model: D < 1.2 (the late channels barely drive the dynamics). (iv) Forcing channel 0 to the truth helps the other channels' forecast: G < 0.8 at lag 30; forcing 28-31 does not: G > 0.95.
- **Reading:** (i)-(iv) all met ⇒ the dropout ordering is also an ordering of dynamical importance, and the leading channels carry the information the others need. (i) failing ⇒ the ordering reflects reconstruction importance only, not the propagator's dependence. If D is large for channel 0 but G ≈ 1, channel 0 is slaved to the others (a diagnostic, not a driver). The floor (decode of the encoded truth) separates the decoder's contribution from the dynamics'. Exploratory: two models, one seed each, 30 cases (6 for the control if fewer members are found).

### 5p result (2026-11-11, Cashew 761411 / 761412): the nested-dropout model `fpc_nested_stagec` through the paired comparison and the adjoint sensitivity (same four storms, same scripts as 5k/5l; data `docs/fpe_report/data/{paired,sens}_nested.json`)
Run as the pending rerun, with no separate pre-registration; the readings of 5k/5l/5m apply unchanged. The table recomputes all three models with one script over all cases (medians over the 16 response cases; sign agreement and ratio over cases with a non-zero CM1 change, no 10% threshold), so the 16- and 32-channel columns differ slightly from the 5k/5m tables, which used per-case thresholds.
| | 16 ch (k45) | 32 ch (d34) | 32 ch nested |
|---|---|---|---|
| lag-0 retention of the bump (median) | 0.034 | 0.095 | **0.215** |
| response amplitude ratio at 15 min | 0.034 | 0.077 | 0.112 |
| response pattern correlation at 15 min | 0.52 | 0.46 | 0.50 |
| median \|Δ centroid speed\| (km/min) | 0.29 | 0.22 | 0.14 |
| leading Lyapunov, 15-30 min, per storm (/min; CM1 0.12-0.25) | 0.040, 0.015, 0.010, 0.016 | 0.025, −0.010, −0.017, 0.023 | **−0.003, −0.005, 0.004, −0.006** |
| sensitivity O1 sign agreement / ratio | 0.71 / 0.039 | 0.75 / 0.111 | 0.88 / 0.135 |
| sensitivity O2 sign agreement / ratio | 1.00 / 0.032 | 0.92 / 0.048 | 0.88 / 0.101 |
| control cold-pool centroid error (km, 4 storms) | 0.1-1.2 | 0.3-6.7 | 0.2-2.4 |
- **Reading.** The nested model keeps the most of the bump (retention 0.215) and has the best centroid-speed agreement (0.14 km/min) and the highest sensitivity ratios, again consistent with the amplitude shortfall being the autoencoder floor. Pattern correlation is unchanged (0.5). **It is the least unstable of the three:** its leading exponent at 15-30 min is within ±0.006/min of zero on every storm, 30-60 times below CM1. So the growth gap is not a feature of one model class either: the three models differ in latent width and ordering and are all near-neutral or slightly unstable. Exploratory: one nested model, four storms.

### 5o result (2026-11-11, Saidie 9245, 62 s; ordered `fpc_nested_stagec` on its 10 test storms, unordered `d34_c32_kuniform` on ens_A2 mem001-010; 30 cases each; data `docs/fpe_report/data/clamp_{nested,unordered}.json`)
All-variable decoded error at lag 30 (normalised units, mean over cases): ordered model free 0.259 (floor of encode/decode 0.036, persistence 0.668); unordered free 0.241 (floor 0.027, persistence 0.547). D = frozen/free, G = forced/free.
| K | ordered D | ordered G | unordered D | unordered G |
|---|---|---|---|---|
| ch 0 | **4.10** | 0.73 | 1.53 | 1.10 |
| ch 1 | 1.51 | 0.73 | 1.37 | 1.08 |
| ch 2 / 3 / 4 | 1.52 / 1.45 / 1.39 | 0.84 / 0.81 / 0.75 | 1.71 / 1.69 / 1.45 | 0.99 / 1.03 / 0.91 |
| ch 8 / 12 / 16 | 1.14 / 1.12 / 1.15 | 0.90 / 0.92 / 0.94 | 1.46 / 1.31 / 1.26 | 0.96 / 0.93 / 0.95 |
| ch 24 / 31 | 1.08 / 1.07 | 0.98 / 1.00 | 1.80 / 1.71 | 1.04 / 0.93 |
| group 0-3 | 4.25 | **0.44** | 2.89 | 1.01 |
| group 8-11 | 1.48 | 0.75 | 2.08 | 0.91 |
| group 28-31 | 1.33 | 0.91 | 2.43 | 0.76 |
- **Prediction check (stated before the run).** (i) **Met:** Spearman(k, log D) across the single channels = −0.96 for the ordered model and +0.04 for the unordered control. (ii) **Half met:** D(ch 0) = 4.1 > 2 ✓, but θ′ is not among the two most damaged variables: ordered by D, ch 0 hurts p (6.4), w (4.4), q_v (3.9), u (3.9), θ′ (3.5), q_ice (3.4), q_liq (3.2), so θ′ is fifth; the damage is spread over all variables (3-6×). (iii) **Not met:** group 28-31 D = 1.33 (> 1.2), although no single late channel exceeds 1.14 (c31 1.07), so the late channels together matter more than any one. (iv) **Half met:** G(ch 0) = 0.73 < 0.8 ✓; G(group 28-31) = 0.91, not > 0.95, so forcing the late channels to the truth also helps a little (9%).
- **What it shows.** In the ordered model the propagator's dependence on a channel falls steadily with the dropout rank, and channel 0 carries most of it: freezing it alone raises the lag-30 error 4.1×, freezing channels 0-3 does the same (4.25×), and teacher-forcing channels 0-3 to the truth cuts the error by 56% (G = 0.44). The leading channels hold both the information the others need and a large part of what the dynamics depends on. In the unordered control the dependence is flat across indices (D 1.3-1.8, no trend) and forcing a channel to the truth does not help (G ≈ 1 for most), so no channel there is the one the others wait for. The dropout ordering is thus an ordering of dynamical importance in the propagator, not only of reconstruction.
- **What it does not show.** The damage is spread over every variable, so channel 0 is not "the cold-pool variable" in the dynamics: freezing it degrades w, pressure and moisture as much as θ′. The p ratio is large partly because the free-run p error is small. Each channel is tested alone or in groups of four; interactions (e.g. channels 0 and 1 together) were not scanned beyond 0-3. Frozen means held at the t0 value at every site, which also removes advection of the channel, so D measures both loss of its dynamics and loss of its propagation. Exploratory: one model of each kind, one seed.

### 5q. What the leading channels are (channel 0 characterisation; script scripts/cm1/channel0_char.py, data docs/fpe_report/data/ch0_{nested,unordered}.json)
Test split, 10 members, 1070 frames, 200 sites of 2 km; nested-dropout model vs the unordered 32-channel control (ens_A2 members). Run after 5o; no pre-registered thresholds, exploratory.
- **Scale ordering.** Power-weighted spatial wavelength of z_k: nested c0 101 km, c1 100, c2 53, c3 41, c8 33, c31 28; unordered 21-27 km for every channel. Temporal e-folding (censored at 41 min): nested c0/c1 >= 41, c3 13, c8 17, c31 9; unordered 10-19 for all. The dropout ordering is a large-to-small scale ordering in both space and time; the control has no such ordering.
- **Content of channel 0.** Per-site covariates explain z_0 with held-out R^2 0.98 (control 0.49). Top correlates: min u -0.89..0.89 (sign), column-mean qice -0.88, qliq -0.87, i.e. condensate loading/storm presence; channel 1 correlates with column-mean qv (-0.69), level-0 pressure and u. The control's channels have no covariate above |r| 0.35-0.64 apart from c8.
- **Variance explained (others at their dataset mean).** Channel 0 alone: u 0.62, w 0.35, th 0.46, qv 0.71, prs 0.43, qliq 0.81, qice 0.79. Channels 0-3: 0.68-0.92. 0-7: 0.80-0.95. Control: channel 0 alone 0.01-0.05; 0-7 0.26-0.52. Removing channel 0 raises the decoded error 4.5x (w) to 38x (qice).
- **Footprint of a one-std perturbation at one site.** Channel 0: w peak 2.2 and qliq 1.1 in normalised units, 50%-width 2-6 km, 10%-width ~10 km (local response even though the field is large-scale); channel 8: 0.26, 8x smaller. A uniform channel-0 shift changes prs by 1.1x its data rms.
- **Reading.** Channel 0 is a ~100 km, >40 min amplitude field of condensate/wind loading (storm presence and organisation); it is not the cold pool. Pressure and w are damaged by freezing it (5o) because they are diagnostic of that organisation, not because channel 0 is them.
- **Caveats.** Correlation, not mechanism; temporal e-folds censored at 41 min; one model of each kind; 10 members.

### 5r. Prefix truncation of the nested model at inference (scripts/cm1/prop_prefix.py, data docs/fpe_report/data/prefix_nested.json)
Only the first m channels evolve; channels >= m held at their dataset mean (MEAN), frozen at t0 (FROZEN) or forced to truth (oracle, uninformative: at small m it is mostly the truth). 30 cases (10 members x t0 120/180/240). Error ratio to the full free rollout (mean over 7 variables), lags 5/15/30 min; free error 0.115/0.196/0.259, persistence 0.51/0.59/0.67.
- MEAN: m=2 3.2/2.3/2.0; m=4 2.6/2.1/2.0; m=8 2.0/1.9/1.8; m=16 1.6/1.6/1.6; m=24 1.3/1.3/1.3. No elbow: every truncation costs, and the error falls slowly towards m=32. At m=8 the lag-30 error (1.8x free = 0.47) is still better than persistence (0.67).
- Truncation floor decode(truth with channels >= m at the mean) as ratio to free: m=8 0.85/0.50/0.39, m=16 0.53/0.30/0.24. At lag 5 the reconstruction loss alone is most of the degradation for m <= 8; at lag 30 it is a small part, so the rest is the propagator's dependence on the dropped channels.
- Conclusion: inference-time pruning of a trained wide model is not free at any m < 24. The dynamical weight of the late channels (5o) is small each but sums; a smaller m would have to be trained (or fine-tuned) as its own model, which this does not test.

### 5s. Fine-tuning after pruning (pre-registration; written before the runs)
Question: how much of the 5r truncation penalty (no-fine-tune error 1.6-2x the full model at m = 16, 2x at m = 8, lag 30) is recovered by training for the pruned latent? Pruned = channels >= m replaced by their training mean in the autoencoder output (`ColumnAEConfig.prefix_m`) and in the propagator's input and output (`StencilPropConfig.prefix_m`); the logdet/variance terms see only the first m channels. From fpc_nested_stagec, Stage C recipe (8000 steps, k 45, w_sfc_th 2, nested_frac 0.5), m = 4, 8, 12, 16, two variants: `prop` (autoencoder frozen, `--c-freeze-ae`) and `full` (autoencoder and propagator). Runs fpc_nested_p{m}_{prop,full}; evaluated on the 5r cases by scripts/cm1/prefix_eval.py against the 5r table (error ratio to the full 32-channel free rollout, mean over variables, lags 5/15/30).
Predictions: (i) `prop` improves on 5r at lag 30 for every m but is bounded below by the 5r truncation floor (m=8: 0.39x, m=16: 0.24x at lag 30; 0.85x and 0.53x at lag 5) - ratios at lag 5 stay > 1.0 for m <= 8. (ii) `full` beats `prop` at lag 5 for m <= 8 (the autoencoder can move information into the prefix) and is within 1.3x of the full model at lag 30 for m = 16. (iii) No m < 12 reaches 1.0x at any lag. (iv) A pruned model that beats persistence (0.67 at lag 30) at m = 4.
Not tested here: a model trained at m from scratch; different step counts; the effect on the physical-variable (w) error specifically.

### 5s result (data docs/fpe_report/data/prefix_ft.json; scripts/cm1/prefix_eval.py, cluster/saidie/prefix_ft.sh)
Error of the free rollout relative to the unpruned model (0.115 / 0.196 / 0.259 at lags 5 / 15 / 30; persistence 0.51 / 0.59 / 0.67), same 30 cases as 5r, mean over 7 variables. Lags 5 / 15 / 30:

| m | no fine-tune (5r) | propagator only | whole model (Stage C) | truncation floor, whole model |
|---|---|---|---|---|
| 4  | 2.64 / 2.11 / 2.01 | 2.09 / 1.61 / 1.35 | 1.70 / 1.43 / 1.26 | 1.11 / 0.65 / 0.52 |
| 8  | 2.04 / 1.92 / 1.83 | 1.55 / 1.35 / 1.23 | 1.33 / 1.25 / 1.16 | 0.72 / 0.41 / 0.33 |
| 12 | 1.82 / 1.74 / 1.71 | 1.36 / 1.27 / 1.17 | 1.17 / 1.16 / 1.10 | 0.56 / 0.32 / 0.25 |
| 16 | 1.58 / 1.58 / 1.57 | 1.25 / 1.18 / 1.12 | 1.11 / 1.10 / 1.07 | 0.48 / 0.27 / 0.21 |

w error (lag 30): 1.03-1.07 for the whole-model runs, 1.04-1.10 for propagator only.
Predictions: (i) met - propagator-only improves lag 30 for every m and stays above the floor, lag-5 ratio > 1 for m <= 8. (ii) met - whole model beats propagator-only at lag 5 for m <= 8 (1.70 vs 2.09, 1.33 vs 1.55) and is 1.07x the full model at lag 30 for m = 16. (iii) met - no m < 12 reaches 1.0 (best 1.16-1.26 at lag 30). (iv) met - m = 4 beats persistence at every lag (0.33 vs 0.67 at lag 30).
Reading: fine-tuning recovers most of the pruning penalty; the whole-model variant recovers more because the autoencoder moves information into the prefix (floor at m = 8 falls from 0.85 to 0.72 at lag 5). At lag 30 m = 16 is within 7% and m = 8 within 16% of the 32-channel model.
Caveats: the pruned runs get 8000 extra Stage C steps the unpruned model did not, so part of the recovery may be extra training, not the pruning (control: 8000 more steps on the unpruned model - not run); one seed; test members only; fine-tuning is of this model, not a model trained at m from scratch.

### 5t. Control for 5s and distill-then-fine-tune (pre-registration; written before the runs)
(a) Control: the unpruned fpc_nested_stagec given the same 8000 Stage C steps as the 5s runs (`fpc_nested_ctl8k_full`: whole model; `fpc_nested_ctl8k_prop`: propagator only). Purpose: 5s ratios divide by the unpruned model's error; if more training alone lowers that error, the 5s recovery is partly extra training, and the 5s ratios must be restated against the control.
(b) Distill then fine-tune (`fpc_nested_p{4,8,12,16}_dft`): prune to m (as 5s), 4000 Stage C steps whose targets are the UNPRUNED model's outputs (its decoded forced-halo rollout from the same true windows, and its reconstruction of the first frame; autoencoder and propagator trained), then the same 8000 CM1 Stage C steps as 5s `full`. The checkpoint after the distillation phase is saved (ae_distilled.pt, prop_distilled.pt) and evaluated as `<run>:distilled`. All use the 5r cases (30 cases), error ratio to the unpruned model's free rollout, lags 5/15/30.
Predictions. (i) Control: the unpruned model's error changes by less than 3% at every lag (it was already trained 15000 steps), so 5s stands. (ii) Distilled checkpoint (before CM1 fine-tune), vs 5r no-fine-tune: better at every lag and m; vs 5s whole: worse at lag 30 (it never sees the truth, so cannot do better than the teacher, whose own lag-30 error is the reference 1.0) - i.e. ratio >= 1.0 at lag 30 for every m. (iii) Distill+CM1 vs CM1 only (5s whole): no better than 0.03 absolute in the ratio at lag 30 for any m; any gain is at lag 5 for m <= 8 (reconstruction floor), >= 0.05 there for at least one of m = 4, 8. (iv) For m = 16 distill+CM1 stays within 1.07 +/- 0.03 at lag 30. Differences smaller than the bootstrap spread over the 30 cases (computed in the analysis, resampling members) are reported as ties.
Not tested: different distillation lengths or losses (latent-space targets, mixed truth+teacher); a from-scratch model at m.

### 5t result (data docs/fpe_report/data/prefix_dft.json; cluster/saidie/prefix_dft.sh, prefix_eval_dft.slurm)
Error ratio to the unpruned model's free rollout, 30 cases, lags 5 / 15 / 30 (bootstrap sd over the 10 members in brackets, lag 5 / 30).
Control (unpruned model, 8000 more Stage C steps): whole 0.980 / 0.994 / 1.004 (0.005 / 0.007); propagator only 0.984 / 0.979 / 1.001. So extra training changes the unpruned error by <= 2.1%; 5s's recovery is not an extra-training artefact (its ratios would move by at most 2%).

| m | distilled only (no CM1 data) | distill, then CM1 (5t) | CM1 only (5s whole) |
|---|---|---|---|
| 4  | 1.78 / 1.48 / 1.29 (0.033 / 0.020) | 1.66 / 1.41 / 1.25 | 1.70 / 1.43 / 1.26 |
| 8  | 1.41 / 1.28 / 1.18 (0.014 / 0.015) | 1.29 / 1.22 / 1.15 | 1.33 / 1.25 / 1.16 |
| 12 | 1.25 / 1.19 / 1.12 (0.014 / 0.012) | 1.15 / 1.14 / 1.08 | 1.17 / 1.16 / 1.10 |
| 16 | 1.16 / 1.13 / 1.08 (0.012 / 0.010) | 1.09 / 1.07 / 1.06 | 1.11 / 1.10 / 1.07 |

Predictions: (i) met (control within 2.1%). (ii) met: distilled beats the no-fine-tune 5r numbers at every lag and m, and is >= 1.0 at lag 30 for every m (1.07-1.29). (iii) lag-30 half met (gain over CM1-only 0.013-0.020 for every m, below 0.03); lag-5 half NOT met (gain 0.041 at m = 4, 0.037 at m = 8, below the 0.05 threshold). (iv) met: m = 16 at lag 30 is 1.057.
Findings: (a) distillation alone, with no CM1 data after pruning, reaches within 0.03 of the CM1 fine-tune at lag 30 for every m (e.g. m = 16: 1.075 vs 1.070; m = 8: 1.18 vs 1.16); at lag 5 it is worse by 0.04-0.12. (b) distill-then-CM1 is lower than CM1-only in all 12 (m, lag) cells, by 0.01-0.04, comparable to the bootstrap sd of a single ratio (0.01-0.03), so it is a small, consistent, not clearly significant gain; it also uses 4000 more steps (control: extra steps alone are worth <= 2%).
Caveats: one seed; teacher = the unpruned model itself; distillation targets only the decoded rollout and first-frame reconstruction; the bootstrap treats the 10 members as independent.

### 5u. The explicit law of the pruned 8-channel models (exploratory; scripts/cm1/pruned_law.py, data docs/fpe_report/data/pruned_law_m8.json)
The nested model's propagator is an explicit latent law (A0 z + A1 d1z + A2 d2z + b + quadratic terms + a small residual MLP). With channels >= 8 fixed at their mean, the first 8 channels obey an effective 8-channel law (terms in the fill fold into A0, A1, b; checked against the model's own tendency to 2e-7). Compared: the unpruned fpc_nested_stagec (restricted to channels 0-7, tail at its mean) and the pruned `p8_prop` (autoencoder frozen), `p8_full`, `p8_dft`. Test states of the 10 test storms.
- **Same coordinates.** Channel-wise correlation of the pruned models' latents with the unpruned model's: 1.0 for prop-only; whole-model 1.00, 1.00, 0.98, 0.97, 0.98, 0.93, 0.94, 0.93 (dft 1.00, 1.00, 0.98, 0.96, 0.97, 0.92, 0.93, 0.93). Fine-tuning the autoencoder does not re-mix the leading channels, so coefficients are directly comparable.
- **Same law for the leading channels.** Channel 0 (rms tendency 0.064 / 0.063 / 0.068 / 0.067 for unpruned / prop / full / dft): top terms by |coefficient| x term rms are the same, in the same order, in all four: d2z0 (+0.118 / 0.120 / 0.119 / 0.119), d2z1 (-0.072 / -0.072 / -0.068 / -0.067), d2z4 (+0.039 / 0.033 / 0.034 / 0.034), z1*z1, d1z0 (+0.098 / 0.102 / 0.110 / 0.119). Channel 1: z0*z1 (+0.03), d1z0 (-0.23 to -0.24), d2z1 (+0.11), d2z4. Channel 2: z0*d1z0 (-0.128 in all). Channel 0's own diffusion coefficient 0.118 sites^2/min = 0.47 km^2/min ~ 8e3 m^2/s; d1z0 +0.1 sites/min is a drift of about 0.2 km/min (3 m/s) towards -x.
- **Dispersion of the linear 8-channel part** (growth per min of the least-damped mode at wavelengths 6 / 10 / 20 / 40 / 80 / 160 km): unpruned -0.184 / -0.079 / -0.024 / +0.002 / +0.016 / +0.021; p8_full -0.179 / -0.082 / -0.011 / +0.009 / +0.015 / +0.019; p8_dft -0.176 / -0.084 / -0.008 / +0.009 / +0.013 / +0.018. Damping below ~30 km, weak growth (e-folds of 50-500 min) above ~40 km. The pruned models agree with the unpruned one to ~0.01/min; the main change is a +0.01/min shift of channel 0's own growth (0.002 -> 0.012) from folding the dropped channels' mean into A0.
- **The law is not closed.** Residual MLP share of the tendency variance, channels 0 / 1: 0.14 / 0.14 (unpruned), 0.12 / 0.14 (full), 0.11 / 0.13 (dft); 0.02-0.07 for channels 2-7. Switching the residual off raises the lag 5 / 15 / 30 error ratio from 1.0 to 1.27 / 1.31 / 1.37 (unpruned) and from 1.29 / 1.22 / 1.15 to 1.52 / 1.54 / 1.66 (p8_dft). The explicit law alone is not a forecast model; the residual is largest for channels 0 and 1.
Reading: the interpretable structure of the leading channels (diffusion, neighbour coupling, quadratic self-interaction, slow advective drift) survives pruning and fine-tuning essentially unchanged; pruning mainly shifts channel 0's growth rate. The residual share of the two leading channels is the part not explained by the polynomial library.
Caveats: exploratory, no pre-registration; coefficients are in latent units (the latent's gauge is fixed here only by the coupling to the unpruned model, and correlation 0.92-0.93 for channels 5-7 means those are less stable); one seed.

### 5v. Does the pruned latent help DA at small ensemble sizes? (pre-registration; written before the runs)
Motivation: the 5s/5t pruned models give up 6-26% of forecast accuracy (m = 16 ... 4, lag 30). A smaller latent state could need fewer ensemble members (SM1-DA already found that a richer, 64-channel latent made DA worse with 32 members: +0.023 +- 0.006, 21/22 storms). Test: latent LETKF twin, the headline setup (theta' + reflectivity + radial velocity, 2 km obs spacing, 5-min cycles, loc 5 km, RTPS 0.9, boundaries held at t0, 22 test storms x 3 seeds), models fpc_nested_stagec (32 channels) and the distill-then-CM1 pruned models fpc_nested_p{16,12,8,4}_dft; ensemble sizes N = 8, 16, 32, 64 (members drawn at random from the 93 training storms); model noise tuned per (model, N) on the 6 validation storms from {0.05, 0.1, 0.15} (lowest validation prior nMSE), a reflectivity head per model (train_refl_head.py). The twin's additive model noise is projected back to the fill on the dropped channels (ks_latent/cm1/twin.py `_forecast`; tested). Primary metric: cycled forecast (prior) nMSE, mean +- 95% CI over storms (seed-averaged), paired against the 32-channel model at the same N (cluster/saidie/da_prune.sh).
Predictions: (i) at N = 8 at least one pruned model (m = 8 or 16) beats the 32-channel model by >= 5% with the paired CI excluding 0. (ii) At N = 64 the 32-channel model beats m = 4 and m = 8 by >= 3%. (iii) For m = 8 the paired difference to the 32-channel model changes sign between N = 8 and N = 64. (iv) m = 4 is the worst model at N = 64. Reading: if the 32-channel model is better or tied (within 2%) at every N, pruning has no DA value in this setup and the per-site PFF stays at 32 channels; if a pruned model wins at N = 8 or 16, the per-site PFF is run at that m and at 32 for comparison.
Not tested: other observation sets or densities; localization radius and RTPS re-tuned per model; unpruned models trained at the smaller m from scratch.

## 6a. Two-scale model: deterministic large scales + stochastic small scales (pre-registration; written before the runs)
Idea (user, 2026-11-12): the latent model forecasts the large scales well but is not chaotic and loses the small scales. CM1's small perturbations grow at 0.1-0.25 / min after ~15 min (5k), so convective-scale detail is predictable for only ~10-20 min and an MSE-trained deterministic model regresses it to its conditional mean. Split the field: the large scales are the existing deterministic pruned model `fpc_nested_p8_dft` (first 8 channels, ~40-100 km, slow; 5q, 5u); the small scales are the residual it misses, r = u - D_L(E_L(u)) (so u = D_L + r exactly), carried by a separate small-scale autoencoder (16 channels, 2 km sites, no positivity) and a STOCHASTIC small-scale propagator g(z_S, z_L(t), z_L(t+1) - z_L(t), 8 noise channels), one-way coupled (large drives small), trained on the decoded residual with the fair CRPS + energy score + 0.1 x log-spectrum mismatch (S = 4 samples, k ramped to 30 min, large latent teacher-forced). Code: ks_latent/cm1/twoscale.py (TwoScaleAE / TwoScaleProp expose the usual interface on z = [z_L, z_S], so the rollout, paired and DA code run unchanged), scripts/cm1/train_twoscale.py, scripts/cm1/twoscale_eval.py; tests tests/unit/test_cm1_twoscale.py. A filter-based split (low-pass ~20-30 km) is the alternative if the residual split fails; feedback (small -> large) and joint fine-tuning come only after the one-way model passes.
Tests (10 test storms x t0 120/180/240 min, lags 5/15/30/60 min, 8 samples; baselines: large-only `fpc_nested_p8_dft`, the 32-channel `fpc_nested_stagec`, and the one-model stochastic `full_v1_stagec_crps` where its test split matches):
- T-spec (primary): horizontal power spectrum / truth's per variable (levels averaged) in the 2-5, 5-10, 10-20 km bands at 30 and 60 min. Pass: two-scale within [0.8, 1.25] in all three bands for u, w, th and qliq, while large-only is below 0.8 in at least two of them (the deficit exists).
- T-chaos (paired CM1 runs, 4 storms, Cashew): sample divergence D(k) = rms(sample_i - sample_j)/sqrt 2 vs CM1's rms(tiny1e-3 - ctrl). Pass: median over storms of D_model(30 min) / D_cm1(30 min) in [0.5, 2].
- T-prob: mean over variables of CRPS (normalized units) <= the large-only model's MAE at 30 and 60 min, and spread / RMSE of the sample mean in [0.7, 1.3] at 30 and 60 min.
- T-large: RMSE of the sample mean after a Gaussian low-pass (sigma 10 km) within 5% of large-only at every lag (one-way: the small scales must not bias the large ones).
- T-front: median |gust-front position error| of the samples <= 1.2 x large-only's at 30 and 60 min.
- T-DA (after the above): the two-scale model as the LETKF forecast model in the headline twin, against the 32-channel and large-only models. Reported, no threshold set yet (the one-model stochastic propagator lost 7-12% in cycled nMSE).
Expected regardless: no gain in RMSE of the mean beyond ~15 min (the gain, if any, is in spectra, divergence, spread and CRPS).

### 6a baselines (before the two-scale model exists; data docs/fpe_report/data/ts_eval_base.json, scripts/cm1/twoscale_report.py)
30 cases (10 test storms x 3 start times), interior columns. Power ratio forecast / truth (levels averaged) at 30 min, bands 2-5 / 5-10 / 10-20 / 20-50 km:
- large-only `fpc_nested_p8_dft`: u 0.15 / 0.15 / 0.29 / 0.63, w 0.01 / 0.03 / 0.16 / 0.48, th 0.05 / 0.06 / 0.17 / 0.36, qliq 0.01 / 0.04 / 0.21 / 0.78. The deficit grows with lag (w 5-10 km: 0.53 at 5 min, 0.13 at 15, 0.03 at 30).
- 32-channel `fpc_nested_stagec`: same picture (w 0.01 / 0.05 / 0.23 / 0.61 at 30 min).
- one-model stochastic `full_v1_stagec_crps` (8 samples): 5-10 and 10-20 km at or slightly above truth at every lag (1.0-1.45), 2-5 km at 0.3-0.5 from 5 min on (its autoencoder's floor); spread / RMSE 0.88-0.92; mean CRPS 0.073 / 0.096 / 0.119 / 0.145 at 5 / 15 / 30 / 60 min against the deterministic models' MAE 0.114-0.143 / 0.150-0.184 / 0.184-0.219 / 0.215-0.244; RMSE of the mean and low-pass RMSE equal to the 32-channel model's; median front error 0.25 / 0.50 / 1.06 / 2.25 km.
Consequences for 6a: the deficit the two-scale model targets is real and large in the deterministic models (T-spec's "large-only < 0.8" condition holds in every band below 20 km). But an existing one-model stochastic propagator already closes it at 5-20 km and is calibrated, so the two-scale model has to be judged against `full_v1_stagec_crps` as well, not only against the deterministic models. Added comparison (reported, no threshold): two-scale vs one-model stochastic on T-spec, T-prob, T-front, and in the DA (where the one-model stochastic lost 7-12%). What the two-scale design could add over it: a deterministic, interpretable large-scale law left untouched, and an explicit split of predictable and unpredictable scales.

### 6a result, version 1 (`ts_cpu`: reduced CPU training because the Saidie GPU queue was blocked; data docs/fpe_report/data/ts_eval_cpu.json, ts_chaos.json)
Training (CPU, 64 cores): small-scale autoencoder 6000 steps, 16 channels, explains 49% of the residual variance (val 0.057 of 0.112); propagator 3000 steps, 3 samples, k <= 15 min, w_spec 0.1 (the pre-registered design used 4 samples, k <= 30, 8000 steps; it stays queued as Saidie 9309). 30 cases, 8 samples.
Pre-registered tests:
- **T-spec: FAIL.** Power ratio at 30 min, bands 2-5 / 5-10 / 10-20 km: u 1.21 / 0.72 / 0.63; w 0.78 / 0.19 / 0.25; th 0.46 / 0.42 / 0.49; qliq 0.66 / 0.33 / 0.48 (pass needed [0.8, 1.25] in all, all four variables). The deficit the model targets does exist in the large-only model (w 0.01 / 0.03 / 0.16) and is partly closed (w 2-5 km 0.01 -> 0.78, th 0.05 -> 0.46), but least in the 5-20 km bands, and the samples' power decays with lead (w 10-20 km: 0.77 / 0.44 / 0.25 / 0.14 at 5 / 15 / 30 / 60 min).
- **T-chaos: FAIL, and as pre-registered the test is ill-posed.** The pre-registered quantity (divergence between samples from one state) is set by the injected noise: 0.27-0.36 from minute 5, 8-20 x CM1's tiny-perturbation divergence at 30 min (0.016-0.039). The meaningful version (added afterwards, flagged as such): ctrl and tiny1e-3 states encoded, the model rolled from each with IDENTICAL noise draws, log-growth per min between 15 and 30 min of the rollout, from rollout starts 10 / 20 / 30 min into the storm. CM1: +0.06 to +0.11 (+0.06 to +0.11 at starts 10 and 20, ~0 or +0.06 at start 30). two-scale v1: -0.055 to -0.001 (negative in all 12 cases); large-only p8_dft: -0.058 to -0.003; one-model stochastic full_v1_stagec_crps: -0.015 to +0.048 (median ~ -0.002). The stochastic samplers add spread, not chaos; no model reproduces CM1's perturbation growth.
- **T-prob: split.** CRPS 0.113 / 0.156 / 0.192 / 0.218 at 5 / 15 / 30 / 60 min, below the large-only model's MAE at 30 and 60 min (0.219, 0.244): pass. Spread / RMSE 0.68 / 0.58 / 0.54 / 0.49: FAIL (needed 0.7-1.3 at 30 and 60); under-dispersed and falling with lead.
- **T-large: FAIL (marginal).** Low-pass RMSE relative to large-only: 0.91 / 1.04 / 1.08 / 1.12 at 5 / 15 / 30 / 60 min (within 5% needed at every lag): the small-scale model adds large-scale error, probably because the residual r = u - D_L(E_L(u)) contains D_L's own large-scale reconstruction error and the small-scale model predicts that part badly.
- **T-front: pass.** 1.12 km at 30 min (large-only 1.12), 1.75 km at 60 min (1.62, 1.08 x).
Against the one-model stochastic baseline (`full_v1_stagec_crps`, not part of the pass/fail): it is clearly better on CRPS (0.096 / 0.119 / 0.145 at 15 / 30 / 60 min vs 0.156 / 0.192 / 0.218), has about the right power at 5-20 km (1.0-1.45), and is calibrated (0.88-0.92).
Reading: version 1 is a partial result: it recovers part of the missing small-scale power at the smallest scales and improves CRPS over the deterministic models, but it does not match CM1's spectra, spread or perturbation growth, and it does not beat the existing single stochastic model. Diagnosed weaknesses: the residual autoencoder captures only half of the residual variance (and includes the large model's large-scale error); the one-way, teacher-forced, noise-input propagator damps rather than sustains small-scale variance; training was reduced. Version 2 (`ts_cpu_v2`, Saidie 9315): 32 small-scale channels, 8000 + 6000 steps, w_spec 1.0 (changed after seeing v1, so reported as exploratory).

## 6b. Filter-based two-scale model with small -> large feedback (pre-registration; written before the runs; user request 2026-11-12)
Changes from 6a: (1) FILTER split, as originally intended: u_L = G u (Gaussian low-pass in x, sigma 5 km, half-power wavelength ~38 km), u_S = u - u_L (carries ~22% of the normalized variance); the small-scale autoencoder (32 channels) and propagator see only u_S, so the large-scale reconstruction error of 6a's residual cannot leak into them. Large scales: the frozen `fpc_nested_p8_dft` autoencoder encodes u_L, its law propagator advances z_L. (2) FEEDBACK: z_L(t+1) = P_L(z_L(t)) + F(z_S(t), z_L(t)), F a zero-initialized stencil net on the 8 kept channels, so the coupled model starts exactly decoupled and the explicit law is untouched. (3) Joint stage J: free rollouts (no teacher forcing) of [z_L, z_S], full-field fair CRPS + energy + log-spectrum (weight 1.0) + MSE of the low-passed sample mean against u_L. Stages: A (autoencoder, 8000), S (small propagator, large latent teacher-forced, 3000), J (joint, 3000), 3 samples, k <= 15 min, CPU (the GPU queue is still blocked). Runs: `tf_v1` (feedback, large propagator frozen), `tf_v1_unfrozen` (feedback, large law propagator also fine-tuned in J), `tf_v1_nofb` (no feedback: control). Code: scripts/cm1/train_twoscale_filter.py, ks_latent/cm1/twoscale.py (FilterTwoScaleAE, FeedbackNet), tests/unit/test_cm1_twoscale.py.
Same tests as 6a (30 cases, 8 samples) plus the common-noise growth test (log-growth per min, 15-30 min of the rollout, tiny1e-3 vs ctrl, identical noise draws; CM1: +0.06 to +0.11 at rollout starts 10 and 20).
Predictions: (i) the filter split fixes T-large without feedback: `tf_v1_nofb` low-pass RMSE within 5% of the large-only model at every lag. (ii) T-spec improves on 6a v1 (1 of 12 variable x band cells within [0.8, 1.25] at 30 min, bands 2-5 / 5-10 / 10-20 km, u / w / th / qliq): at least 5 of 12 for the best variant. (iii) Feedback produces growth: median common-noise log-growth (15-30 min, starts 10 and 20) of `tf_v1` at least 0.02 higher than `tf_v1_nofb` and above -0.01. (iv) spread / RMSE of the sample mean in [0.7, 1.3] at 30 and 60 min for the feedback variants. (v) front error <= 1.2 x large-only at 30 and 60 min, and CRPS below the large-only model's MAE (0.219 / 0.244). (vi) with feedback, the low-pass RMSE of the sample mean stays within 5% of large-only (feedback must not hurt the large scales). Expected to remain true regardless: the one-model stochastic baseline (`full_v1_stagec_crps`) may still beat it on CRPS.

### 6a result, version 2 (`ts_cpu_v2`: 32 small-scale channels, w_spec 1.0, 8000 + 6000 steps; exploratory, changed after v1; data docs/fpe_report/data/ts_eval_v2.json)
Worse than v1 on the pre-registered tests. 30 cases, 8 samples. CRPS 0.139 / 0.188 / 0.229 / 0.266 at 5 / 15 / 30 / 60 min (v1: 0.113 / 0.156 / 0.192 / 0.218; large-only MAE 0.219 at 30, 0.244 at 60: now FAILS T-prob's CRPS condition); spread / RMSE 0.43 / 0.37 / 0.38 / 0.43; low-pass RMSE vs large-only 1.16 / 1.28 / 1.33 / 1.44 (v1: 0.91 / 1.04 / 1.08 / 1.12): the large-scale leak is larger. Spectra at 30 min (2-5 / 5-10 / 10-20 km): u 1.08 / 0.73 / 0.61, w 0.77 / 0.23 / 0.28, th 0.84 / 0.78 / 0.50, qliq 0.99 / 0.34 / 0.38: the 2-5 km band is now close to truth (w 0.77, th 0.84, qliq 0.99) but 5-20 km is still low, and the power below 2 km overshoots and grows with lead (w 0.5-2 km: 1.75 / 1.98 / 2.58 / 5.49 at 5 / 15 / 30 / 60 min; at 60 min the 2-5 km band also overshoots, 1.4-1.8 for u, w, th, qliq): spurious small-scale power accumulating in the one-way rollout. More channels and a heavier spectral weight did not fix the residual-split design; they traded the dissipation for a growth of unphysical power.

## 7a. Global coupling of the leading channels in the one-model stochastic system (pre-registration, 2026-11-13)
Request (advisor, via user): the original one-model system with global coupling, e.g. nested dropout plus coupling between the most important channels of each patch. The two-scale line (6a/6b) is closed as negative: the filter split restores 5-10 km power, but the small scales come out as smeared texture and the samples barely spread, because the frozen deterministic large model carries the uncertainty (figures docs/fpe_report/figs_tf/).

**Design.** Every arm uses the same frozen nested-dropout autoencoder (`fpc_nested_stagec`; channels ordered by importance, 5q), so all arms forecast the same latent and differ only in the propagator. Propagator: the stochastic conv stencil of `full_v1_crps` (4 noise channels, fair CRPS, reach 3 sites = 6 km per 1-min step), plus an optional global module (`StencilPropConfig.global_kind`, ks_latent/cm1/models.py). The module takes the first k channels at every site and returns 16 features per site, appended to the stencil input. Two kinds:
- `attn`: 2 layers of full self-attention over all sites, with a signed, log-binned relative-position bias. It has no per-site parameters and runs on any width.
- `mean`: an MLP of the domain mean, broadcast to every site.

The stencil's weights on the global features start at 0, so each arm starts as the local model. Pipeline: Stage B (20k steps), then Stage C decoded CRPS (15k steps, k ≤ 45, autoencoder frozen). Tests are in tests/unit/test_cm1_global.py.

| arm | global | k | window (sites) | seed |
|---|---|---|---|---|
| g_local48 | none | - | 48 (recipe) | 0 |
| g_local128, g_local128_s1 | none | - | 128 | 0, 1 |
| g_mean8 | mean | 8 | 128 | 0 |
| g_attn4 | attn | 4 | 128 | 0 |
| g_attn8, g_attn8_s1 | attn | 8 | 128 | 0, 1 |
| g_attn32 | attn | 32 (all) | 128 | 0 |

**Evaluation** (cluster/saidie/global_eval.slurm): twoscale_eval on the 30 test cases with 8 samples (CRPS, spread/RMSE, band spectra, low-pass RMSE, front error at 5/15/30/60 min) against `full_v1_stagec_crps` and `ts_v1`. The same is repeated with persistence boundaries (halo held at t0) and with the global features zeroed at inference (`@globaloff`). scripts/cm1/global_diag.py measures:
- the share of the one-step Jacobian norm beyond the stencil reach (exactly 0 for a local model; checked on full_v1_stagec_crps);
- the attention mass by distance.

**Prior.** Two reasons to expect a small effect:
- Deep gravity waves (30-50 m/s, 2-3 km per step) fit inside the 6 km reach per step. The anelastic pressure response is the only instantaneous long-range coupling, and the latent may encode it implicitly.
- FPc arm 4 (domain mean of all channels in the law) gave 1.5-2% skill, which is close to noise.

Expected: CRPS gain under 5%.

**Decision rules** (CRPS averaged over all variables, at 30 and 60 min):
1. *Global coupling helps* if g_attn8 and g_attn8_s1 both beat both g_local128 seeds, and the mean gain is ≥ 3% and larger than the larger seed spread.
2. *It is coupling, not capacity:* the share beyond reach is ≥ 0.05 and `@globaloff` loses ≥ 3% CRPS. A gain without these is attributed to training, not distant information.
3. *Not boundary leakage:* the gain persists (≥ 2/3 of it) with persistence boundaries.
4. *Leading channels suffice* (the advisor's hypothesis) if g_attn4 or g_attn8 is within the seed spread of g_attn32. If g_attn32 is better by more than the spread, the global information is not confined to the leading channels.
5. *Structure vs summary:* g_mean8 against g_attn8.
6. *Window effect:* g_local48 against g_local128, which is a confound check for all of the above.

The Jacobian and attention distance profiles are read for a characteristic coupling length, for example upstream gravity-wave or pressure scales of 20-50 km.

## 8a. More data and a 200 M-parameter ViT autoencoder (plan and pre-registration, 2026-11-13)
Request (user and advisor): a much larger training set and a much larger model. The model is a ViT-style encoder and decoder with SwiGLU nonlinearities and a decent up-projection, about 4 layers and about 200 M parameters. The propagator may be a transformer, a dense MLP or the current stencil, and tests should decide.

Context: the small conv autoencoder (2.6 M parameters) stopped improving at about 28 storms (D5). Either data or capacity could be the limit, so the plan tests both.

**Data (Cashew, existing kit, GrADS output; noise-seed offsets chosen so no seed repeats an earlier ensemble):**
- `ens_C`: 360 storms, random boundary-layer moisture −2.0…+0.25 g/kg and shear ×0.6…1.4 (ens_B's range), seeds 10001-10360.
- `ens_D`: 120 storms in the original environment, seeds 11001-11120.

About 1.6-1.9 h per storm, in 16 jobs of 32 CPUs. Caches: one 8-shard `build_cache.py` job per ensemble after the CM1 jobs finish, writing ~/cm1_cache_C and ~/cm1_cache_D on Cashew (about 0.5 GB per storm). These are then pulled to Saidie by rsync, as before. In total 601 storms, against 121.

**Loading:** training memory-maps the uncompressed .npz caches (`--mmap`; `load_member(mmap=True)`, `ColumnData` lazy mode reads and normalizes windows on demand). 600 storms are about 300 GB in float16, which does not fit in a shared job.

**Model** (`ColumnAEConfig.kind = "vit"`, `ViTColumnAE`, ks_latent/cm1/models.py):
- Tokens are the 2 km sites (8 columns × 434 channels).
- 4 encoder and 4 decoder blocks, each pre-RMSNorm attention plus SwiGLU with hidden = 4 × d, at d = 1216 with 19 heads.
- A learned relative-position bias per head, and no per-site parameters.
- The latent is still 32 channels per site, so every propagator, rollout and DA path runs unchanged.
- 197.8 M parameters.
- Attention is either windowed (radius 2 sites per layer, reach 8 sites = 16 km, so the latent stays a local field) or global (every site of the window).

Tests are in tests/unit/test_cm1_vit.py: size, locality, shift equivariance, positivity, mmap equivalence, and the MLP propagator. A dense-MLP propagator body (`body = "mlp"`) has been added: a per-site SwiGLU MLP on the flattened neighbourhood, with the same 3-site reach as the conv stencil and zero-initialized.

**Pilot on the existing 121 storms** (cluster/saidie/vit_pilot.sh, same split as the full run). Every arm uses nested dropout 0.5, Stage A 40k steps at lr 3e-4, then Stage B (stochastic CRPS, 4 noise channels), then Stage C decoded CRPS with the autoencoder frozen. This is the recipe of 7a's `g_local48` (conv autoencoder), the control.
- `v_win`: 200 M windowed, conv stencil propagator.
- `v_win_attn`, `v_win_mlp`: the same autoencoder with the windowed-attention and dense-MLP propagator bodies.
- `v_glob`: 200 M global, trained on whole-domain windows (200 sites) in all stages.
- `v_small`: windowed, d = 512, about 37 M parameters.

Added the same day (cluster/saidie/vit_pilot_add.sh): the scaled-up original local model, with the original propagator.
- `c_wide`: `ColumnFieldAE`, hidden 3840, 2 mixing convs, reach 2 sites as in the original, 204 M.
- `c_deep`: hidden 2048, 7 mixing convs, reach 7 sites, close to the windowed ViT's 8, 191 M.

Both use the original propagator (conv stencil, hidden 256, 3 layers, radius 1, 4 noise channels, CRPS), so only the autoencoder differs from g_local48. `v_win_attn`'s attention propagator now has SwiGLU FFNs (4×). The ViT blocks have had SwiGLU with a 4× up-projection from the start.

Evaluation (vit_eval.slurm): twoscale_eval at lags 0 (reconstruction), 5, 15, 30 and 60 min, against g_local48_stagec and full_v1_stagec_crps.

**Decision rules:**
0. *Scaled-up conv vs ViT:* if c_wide or c_deep is within 3% of the best ViT in 30-min CRPS, the conv model is kept as the simpler, already validated design. Otherwise the best ViT is kept.
1. *Windowed vs global:* take the windowed model unless global is better by more than 5% in 30-min CRPS. Below that margin the local-field property (DA localization, the transfer results) is worth more.
2. *Capacity:* if v_small is within 3% of v_win at 30 and 60 min on the 121 storms, the large model is data-limited at this size. The data-scaling step then decides.
3. *Propagator:* keep the body with the lowest 30-min CRPS. The conv stencil wins ties within 3%, as the simplest.
4. Reconstruction at lag 0 must be no worse than the conv autoencoder's. Otherwise the large autoencoder is not doing its job, whatever the forecast scores.

**Then data scaling,** once the caches are on Saidie: the chosen autoencoder and propagator trained on 93 (the current training set), about 280, and about 560 training storms, with the same 22 test storms plus ~20 held-out ens_C/ens_D storms. This shows whether the large model keeps improving with data, where the small one stopped at about 28 storms.

### 7a result (2026-11-14; Saidie 9370, 30 test cases each, 8 samples; data docs/fpe_report/data/g_eval.json, g_eval_pers.json, g_diag.json)
Fair CRPS averaged over variables, at 15 / 30 / 60 min, and relative to the matched local control `g_local128` (seed 0: 0.0980 / 0.1189 / 0.1434; the other control seed is within 0.6%).

| arm | 15 min | 30 min | 60 min | spread/RMSE at 60 | share of Jacobian beyond 6 km |
|---|---|---|---|---|---|
| g_local48 (recipe window) | +1.5% | +2.6% | +1.2% | 0.95 | 0 |
| g_local128_s1 | −0.6% | +0.6% | −0.3% | 0.95 | 0 |
| g_mean8 | −0.7% | +3.5% | +4.2% | 0.94 | 0.07 |
| g_attn4 | +0.6% | +3.4% | +5.8% | 0.90 | 0.19 |
| g_attn8 | −1.5% | +2.6% | +5.5% | 0.92 | 0.19 |
| g_attn8_s1 | −2.7% | +1.1% | +3.6% | 0.92 | 0.18 |
| g_attn32 | **−5.6%** | **−3.0%** | **−2.9%** | 0.96 | 0.19 |
| full_v1_stagec_crps (original) | −2.0% | +0.3% | +1.2% | 0.92 | 0 |

**Reading against the pre-registration.**
1. *Rule 1 (global coupling helps) fails.* g_attn8 does not beat both local controls on either seed at 30 or 60 min; it is better only at 15 min. The Stage C validation gains (2-4.6%) did not transfer to test CRPS; validation was 6 storms with decoded k = 45 error, not CRPS.
2. *Rule 2 (it is coupling, not capacity):* the models do use distant information (13-19% of the one-step Jacobian norm lies beyond the 6 km stencil reach, nearly all of it beyond 60 km; g_mean8 7%). That did not buy skill. The switch-off ablation (global features zeroed at inference) costs +40% to +155% CRPS, but this is out of distribution for a model trained with the features present, so it shows dependence, not benefit. A fairer ablation would substitute the features' data mean.
3. *Rule 3 (not boundary leakage):* the only winner, g_attn32, loses most of its gain with persistence boundaries (30 min: −3.0% true, −0.8% persistence; 60 min: −2.9% → −0.3%). Its gain is mostly at 15 min and partly from the true lateral boundaries.
4. *Rule 4 (leading channels suffice):* no. attn4 and attn8 are worse than local at 30-60 min, attn32 is better, by more than the seed spread; but attn32 has one seed.
5. *Rule 5 (structure vs summary):* g_mean8 (+3.5 / +4.2%) is about the same as g_attn8 (+2.6 / +5.5 / seed 1: +1.1 / +3.6%). The attention adds nothing over a domain mean.
6. *Rule 6 (window effect):* g_local48 is 2.6% worse than g_local128 at 30 min: a longer training window alone buys about as much as any global arm.

**Conclusion.** Global coupling of the leading channels does not improve the one-model forecast; it slightly hurts the 30-60 min CRPS and the calibration. Coupling all 32 channels helps by 3-6%, mainly at short lead and with true boundaries, on one seed. Next steps if pursued: a seed repeat of g_attn32, and its persistence-boundary result is the figure to trust. The result is a negative for the advisor's hypothesis as stated (leading channels only) and a weak positive for coupling every channel.

### 8a pilot result (2026-11-14; 121 storms, Saidie v_eval 9392, 30 test cases, 8 samples; data docs/fpe_report/data/v_eval.json)
Reference `g_local48_stagec` (2.6 M conv autoencoder, nested dropout, original stochastic stencil; the 7a control). CRPS relative to it; reconstruction is the RMSE of decode(encode(u)) at lag 0, averaged over variables.

| arm | autoencoder | propagator | recon RMSE | CRPS 15 | CRPS 30 | CRPS 60 |
|---|---|---|---|---|---|---|
| v_win | ViT 198 M, windowed | conv stencil | −28% | +6.3% | +7.7% | +8.5% |
| v_win_attn | same | attention (SwiGLU) | −28% | −0.9% | +3.5% | +4.8% |
| v_win_mlp | same | dense SwiGLU MLP | −28% | −0.3% | +3.4% | +6.5% |
| v_small | ViT 37 M, windowed | conv stencil | −27% | +6.0% | +6.6% | +7.3% |
| c_wide | conv 204 M, reach 2 | original stencil | −25% | −1.2% | **−0.8%** | +0.6% |
| c_deep | conv 191 M, reach 7 | original stencil | −25% | +5.0% | +5.0% | +5.7% |
| full_v1_stagec_crps | conv, no nested dropout | original stencil | −12% | −3.4% | −2.3% | 0.0% |
| v_glob | ViT 198 M, global | conv stencil | Stage C ran out of GPU memory; rerun queued (9393, eval 9394) | | | |

**Reading against the pre-registration.** (Single seed per arm; the seed spread of the control is about 0.6%.)
- *Rule 4 (reconstruction no worse):* met by every large model: −25 to −28%.
- *Rule 0 (scaled conv within 3% of the best ViT):* the conv model is better. c_wide at 30 min is 0.1211 against the best ViT's 0.1262, and equals the original. The conv model is kept.
- *Rule 2 (capacity):* v_small (37 M) is as good as v_win (198 M) in reconstruction (−27% vs −28%) and in CRPS (+6.6% vs +7.7%): at 121 storms, the ViT gains nothing from 5× more parameters. The scaled conv does gain in reconstruction (−25% from 2.6 M to 204 M).
- *Rule 3 (propagator body):* on the ViT latent, the windowed-attention and dense-MLP bodies are about 4% better than the conv stencil at 30 min (3.5% and 3.4% against 7.7%); neither beats the original model.
- *Rule 1 (windowed vs global):* open (v_glob). Its Stage A validation reconstruction is 0.0163 against 0.0273 for the windowed ViT (windows of 200 vs 64 sites, so not matched).

**Conclusion.** A much larger autoencoder reconstructs 25-28% better but does not forecast better at 121 storms. The forecast is not limited by the autoencoder at this data size. The data-scaling step with the new storms is the real test.

### 8a follow-up: Stage C with the autoencoder unfrozen (queued 2026-11-14; cluster/saidie/vit_unfrozen.sh)
The pilot froze the autoencoder in Stage C (copied from 7a). The original best model, full_v1_stagec_crps, ran Stage C with it free, so its latent was fine-tuned to be forecastable; the pilot's ViT latent never was. Reruns of Stage C from the pilot's Stage B outputs, autoencoder unfrozen, nested dropout 0.5 kept, batch 4, lr 5e-5 (200 M models) or 1e-4 (control): u_win (ViT + stencil), u_win_mlp (ViT + dense MLP), u_wide (conv 204 M + stencil), u_local48 (control, conv 2.6 M). Evaluated with vit_eval into ~/cm1_da/v_eval_unfrozen against the frozen-Stage-C results.
Pre-registered reading: if u_win closes the gap to u_local48 (within 3% at 30 and 60 min), the ViT autoencoder with the original stencil is viable and the pilot's gap was the frozen latent. If it stays worse by more than 3%, the ViT latent is harder to forecast and the pairing is rejected. If a run runs out of GPU memory, rerun with a smaller batch.

### 8a follow-up: prediction term in Stage A (queued 2026-11-14; cluster/saidie/vit_stagea_pred.sh)
Idea (user): penalize one-step-ahead prediction in the autoencoder's own phase. Stage A had no prediction term (reconstruction + variance + log-determinant only); the propagator first appears in Stage B. Implemented in `train_stage_a(..., aux=)`: a small deterministic auxiliary stencil propagator, trained jointly and discarded after Stage A, rolls `--pred-k-a` steps from the encoded first frame with the true encoded halo, and the decoded rollout is scored against the true columns in physical space (not latent targets: the KS runs found those invite collapse). Only the first `--pred-batch-a` (8) windows of each batch carry the extra frames, so the reconstruction path is unchanged. Tests in tests/unit/test_cm1_vit.py (w_pred = 0 reproduces the old run exactly).

Arms on the ViT 198 M windowed autoencoder with the original stochastic stencil propagator; Stage C unfrozen as `u_win`, so each arm differs from `u_win` only in Stage A: `a_k1` (weight 1, one step ahead), `a_k4` (weight 1, four steps), `a_k4w5` (weight 5, four steps). Evaluated with vit_eval into v_eval_stagea, with u_win and u_local48 for reference.
Pre-registered reading: the prediction term helps if an arm beats `u_win` by more than 3% in CRPS at 30 and 60 min (seed spread about 0.6%) without losing more than 10% in reconstruction RMSE at lag 0. One step ahead moves the latent very little from persistence (the 1 min step), so the k = 4 arms are the stronger test. If the unfrozen Stage C already closes the ViT gap to the control, a Stage A term is less needed and this is judged on whether it adds anything beyond that.
Added after the first three (user: more rollout steps in Stage A): `a_k8` (weight 1, 8 steps, 8 windows carry the extra frames) and `a_k16` (weight 1, 16 steps, 4 windows), cluster/saidie/vit_stagea_pred2.sh, evaluated into v_eval_stagea2 with `a_k4`, `u_win` and `u_local48`. Windows per batch are cut as k grows so the encoder's activation memory stays about the same; Stage A cost grows roughly with k. Same decision rule. Risk to watch: at k = 16 (16 min) the rollout error is several times the reconstruction error, so at weight 1 the prediction term can dominate and cost reconstruction; the 10% reconstruction guard decides that.

### 8a: 75 M encoder for the Stage A prediction arms (user, 2026-11-14: 200 M is probably too much)
`vit_dim` 736, 23 heads of 32, 4+4 blocks, SwiGLU x4: 74.5 M parameters (test_75M_configuration). The 198 M Stage A arms (a_k1 ... a_k16, jobs 9400-9411) were cancelled after 4 minutes and replaced by cluster/saidie/vit75.sh: `w75` (matched control, no prediction term), `p75_k1`, `p75_k4`, `p75_k4w5`, `p75_k8`, `p75_k16`, windowed, original stochastic stencil propagator, Stage C unfrozen (lr 5e-5, batch 4), evaluated into v_eval_75 with `u_win` (198 M) and `u_local48`. The pilot's `v_small` (37 M) matched the 198 M model, so a 75 M encoder is not expected to reconstruct worse; the unfrozen `u_win` result (198 M) says whether the ViT + stencil pairing works at all.
Decision rule as for the Stage A arms: an arm helps if it beats `w75` by more than 3% in CRPS at 30 and 60 min and keeps reconstruction within 10%.

## 8b. Propagator structure: the Paper I/II local_mlp block on CM1 (pre-registration, 2026-11-14)
Why: in the pilot, the propagator body mattered more than most of what we varied (on the ViT latent: stencil +7.7% CRPS at 30 min, attention and dense MLP about +3.5%). The CM1 stencil (3 convs with GELU, width 256, 0.43 M) is of the same family as Paper I/II's propagator (`local_mlp` / `local_mlp_2d`: local, shared across sites, residual, zero-initialized, Markovian), but differs in three ways: Paper I/II separates linear windowed spatial mixing (conv, no nonlinearity) from a wide per-site MLP (4x), with pre-norm residual blocks; the stencil fuses mixing and nonlinearity, has no norms and no wide MLP; and CM1 adds 4 noise input channels trained with CRPS.

New body `StencilPropConfig.body = "mixer"` (ks_latent/cm1/models.py, `_MixerBlock`): embed Linear(c + noise -> H); `layers` blocks, each [pre-RMSNorm, Conv1d(H, H, 2r+1), residual; pre-RMSNorm, SwiGLU(H, 4H), residual]; RMSNorm; Linear(H -> c), zero-initialized (identity at init). 3 blocks of radius 1: reach 3 sites (6 km) per 1-min step, as the stencil. Tests: tests/unit/test_cm1_vit.py::test_mixer_propagator.

Arms (cluster/saidie/mixer_run.sh), on the same frozen autoencoder as the control `g_local48` (fpc_nested_stagec, nested dropout), same recipe (Stage B stochastic CRPS, Stage C decoded CRPS with the autoencoder frozen): `m128` (width 128, 0.75 M, 1.7x the stencil: the main arm, per the guidance to keep the propagator at 1-2x), `m256` (3.0 M, capacity check), `mlp256` (dense SwiGLU MLP on the 7-site neighbourhood, 1.65 M). Evaluated with vit_eval into v_eval_mixer.
Decision rule: the mixer helps if m128 beats g_local48 by more than 3% in CRPS at both 30 and 60 min (seed spread of a control about 0.6%) without losing spread/RMSE calibration (g_local48: 0.93-0.95); it is the better propagator structure only if it also does not lose at 15 min. If m256 helps and m128 does not, the gain is capacity, not structure, and conflicts with the simplicity preference. This is the AE-fixed test on the conv latent; the ViT latent is a later step.

### 8a follow-up result: Stage C with the autoencoder unfrozen (2026-11-14; Saidie 9399, 30 test cases, 8 samples; data docs/fpe_report/data/v_eval_unfrozen.json, v_eval_glob.json)
CRPS relative to `g_local48` (frozen Stage C, original stencil, 2.6 M conv autoencoder); reconstruction is RMSE at lag 0. Single seed per arm; control seed spread about 0.6%.

| arm | autoencoder | propagator | Stage C | recon | CRPS 15 | CRPS 30 | CRPS 60 |
|---|---|---|---|---|---|---|---|
| g_local48 | conv 2.6 M | stencil | frozen | 0% | 0% | 0% | 0% |
| u_local48 | conv 2.6 M | stencil | unfrozen | −3% | −1.6% | −1.3% | −0.1% |
| v_win | ViT 198 M | stencil | frozen | −28% | +6.3% | +7.7% | +8.5% |
| u_win | ViT 198 M | stencil | unfrozen | −25% | −1.9% | +2.6% | +5.3% |
| v_win_mlp | ViT 198 M | dense MLP | frozen | −28% | −0.3% | +3.4% | +6.5% |
| **u_win_mlp** | ViT 198 M | dense MLP | unfrozen | −26% | **−8.4%** | **−4.1%** | **−1.6%** |
| c_wide | conv 204 M | stencil | frozen | −25% | −1.2% | −0.8% | +0.6% |
| u_wide | conv 204 M | stencil | unfrozen | −23% | −6.0% | −3.4% | −1.9% |
| v_glob | ViT 198 M global | stencil | frozen | −25% | +5.3% | +5.3% | +7.6% |
| full_v1_stagec_crps | conv, no nested dropout | stencil | unfrozen | −12% | −3.4% | −2.3% | 0.0% |

**Reading against the pre-registration.**
- *The frozen autoencoder was a confound:* unfreezing Stage C improves every large arm (u_win vs v_win at 30 min −4.7%, u_win_mlp vs v_win_mlp −7.3%, u_wide vs c_wide −2.6%); the small control gains only 1.3%.
- *u_win (the pairing asked about) is not viable:* it stays 4.0% and 5.3% worse than the matched unfrozen control u_local48 at 30 and 60 min (the rule: within 3%). The stencil propagator is the weak link on the ViT latent.
- *Best arm: u_win_mlp* (ViT 198 M + dense SwiGLU MLP propagator, unfrozen): −4.1% / −1.6% at 30 / 60 min against g_local48, −2.8% / −1.6% against u_local48, and −1.9% / −1.7% against the previous best (full_v1_stagec_crps). u_wide is next. The margin over the unfrozen control is modest at 60 min and single seed.
- Spread/RMSE at 60 min is 0.92-0.96 for all unfrozen arms, no calibration loss.
- *Windowed vs global (rule 1):* v_glob is 5.3% / 7.6% worse than the control, no better than the windowed v_win (+7.7 / +8.5%): the windowed (local) autoencoder is kept. v_glob's reconstruction is −25%.
- Reconstruction cost of unfreezing: the first-frame training reconstruction rises (ViT 0.016 -> 0.066, under nested dropout), while lag-0 test RMSE barely moves (-28% -> -25%).
Pending: the 75 M ViT arms (Stage A prediction term) and the mixer propagator arms (8b).

### 8a figures: does the ViT + dense-MLP model capture more small-scale structure? (2026-11-14; scripts/cm1/make_model_compare_figs.py, plot_spectra_compare.py; figures docs/fpe_report/figs_vit/)
Storm ens_A2_mem001 (unseen), forecasts from t = 60 min with held boundaries (nothing after the start is known), original conv 2.6 M + stencil (Stage C unfrozen) against ViT 198 M + dense MLP (unfrozen). `fig_sections_orig.png` and `fig_sections_vitmlp.png` use the layout of docs/cm1_status/figs/fig_forecast_sections.png; `fig_compare_sections.png` puts truth and both models side by side at +30 and +60 min; `fig_compare_zoom.png` zooms on the storm at +30 min with two samples per model; `fig_compare_spectra.png` is the band-power ratio to truth from the 30-case evaluation.
Reading: power / truth at 30 min, 2-5 km band: w 0.71 (ViT + MLP) against 0.35-0.43 (original, conv 204 M, ViT + stencil: 0.29-0.31); theta' 0.62 against 0.29-0.37; liquid 0.72 against 0.35-0.42. At 0.5-2 km it reaches 0.64 (w) against 0.25-0.31 for the original (the ViT + stencil and the 204 M conv also reach 0.6-0.65 there but sit at 0.3 in 2-5 km). It also has the smallest overshoot at 10-20 km (w 1.2 against 1.3-1.65). It remains below truth at scales under 5 km (0.6-0.7). Visually the cores of w and liquid are sharper and more distinct than the original's smooth, broad structures at +30 min, but the cells are not in the right places and at +60 min the forecast concentrates into one tall, too-intense updraft / downdraft pair and theta' maximum, where the truth has several tilted cells. More small-scale power is not the same as more correct small-scale structure.

### 8b result: the Paper I/II mixer propagator on the conv latent (2026-11-14; Saidie 9431, 30 test cases, 8 samples; data docs/fpe_report/data/v_eval_mixer.json)
Same frozen conv autoencoder (fpc_nested_stagec, lag-0 reconstruction identical), same recipe as the control `g_local48` (frozen Stage C); only the propagator differs. CRPS relative to g_local48 (control seed spread about 0.6%):

| arm | propagator | parameters | CRPS 15 | CRPS 30 | CRPS 60 | RMSE of mean, 30 | spread/RMSE at 60 |
|---|---|---|---|---|---|---|---|
| g_local48 | stencil | 0.43 M | 0% | 0% | 0% | 0% | 0.96 |
| m128 | mixer, width 128 | 0.75 M | −6.2% | **−4.4%** | **−3.9%** | −2.8% | 0.97 |
| m256 | mixer, width 256 | 3.0 M | −8.2% | −5.1% | −3.6% | −2.8% | 0.93 |
| mlp256 | dense SwiGLU MLP, 7 sites | 1.65 M | −6.2% | −5.3% | −5.8% | −2.3% | 0.95 |
| full_v1_stagec_crps | stencil, unfrozen, no nested | | −3.4% | −2.3% | 0% | −0.3% | 0.93 |

Reading against the pre-registration: m128 beats the control by more than 3% at both 30 and 60 min with calibration intact (0.97): the rule is met, at 1.7x the stencil's size, inside the 1-2x preference. m256 is no better than m128 (−5.1 / −3.6% against −4.4 / −3.9%): the gain is the structure (separate linear spatial mixing and wide per-site SwiGLU MLP), not capacity. The dense MLP over the 7-site neighbourhood is as good at 30 min and better at 60 min (−5.8%, about 3 control-seed spreads; 2.2x the size of m128). On the same conv autoencoder these propagators gain more (−4 to −6%) than the large ViT autoencoder with the dense MLP and Stage C unfrozen gained over the same control (−4.1 / −1.6%): the propagator structure is a larger lever than autoencoder size so far.
Queued next (cluster/saidie/mixer_unfrozen.sh, jobs 9435-9441): m128 with Stage C unfrozen on the conv 2.6 M, ViT 75 M (w75) and ViT 198 M (v_win) autoencoders (um_local, um_w75, um_win), evaluated into v_eval_mixer_unfrozen with u_win_mlp, u_local48 and m128.

### 8a result: prediction term in Stage A, 75 M windowed ViT (2026-11-14; Saidie 9424, 30 test cases; data docs/fpe_report/data/v_eval_75.json)
Original stochastic stencil propagator, Stage C unfrozen; each arm differs from the matched control `w75` only in Stage A. CRPS relative to `w75` (and to g_local48), lag-0 reconstruction RMSE relative to g_local48 and w75:

| arm | pred weight, steps | recon vs w75 | CRPS 15 | CRPS 30 | CRPS 60 | spread/RMSE at 60 |
|---|---|---|---|---|---|---|
| w75 (control) | none | 0% (−26% vs g_local48) | −0.1% | +2.7% | +4.8% | 0.99 |
| p75_k1 | 1, 1 step | +1.9% | −5.0% (−5.0% vs w75) | −2.6% vs w75 | −3.2% vs w75 | 0.95 |
| p75_k4 | 1, 4 steps | +8.1% | −8.0% (−7.9% vs w75) | −4.9% vs w75 | −2.3% vs w75 | 0.93 |
| p75_k4w5 | 5, 4 steps | +15% | −7.5% | −4.7% vs w75 | −3.0% vs w75 | 0.92 |
| p75_k8 | 1, 8 steps | +12.6% | −6.3% | −4.0% vs w75 | −0.7% vs w75 | 0.88 |
| p75_k16 | 1, 16 steps | +19.7% | −4.6% | −4.0% vs w75 | −1.8% vs w75 | 0.85 |
(`u_win`, the 198 M ViT with the same stencil and unfrozen Stage C, equals w75 within 0.5%: 75 M is as good as 198 M here.)

Reading against the pre-registration (an arm helps if it beats w75 by more than 3% at both 30 and 60 min and keeps reconstruction within 10%): no arm meets both. k = 1 (the cheapest, +1.9% reconstruction) gives −2.6% / −3.2%, just short at 30 min; k = 4 gives −4.9% / −2.3%, short at 60 min, +8% reconstruction; w = 5 and k = 16 fail the reconstruction guard (+15%, +20%). Longer rollouts or a heavier weight do not help beyond k = 4: they cost reconstruction and calibration (spread/RMSE 0.85-0.88 at k = 8 and 16). All prediction arms improve the 15- and 30-min CRPS by 3-8% relative to w75, and the 60-min CRPS by 1-3%, against a control seed spread of about 0.6%: the term helps at short lead, not at 60 min. The stencil propagator remains the limit: no arm beats the original g_local48 at 60 min (the best, k = 1, +1.5%).
Next: the mixer propagator on these autoencoders (um_w75, um_win queued) and, if wanted, the Stage A term together with the mixer.

### 8a result: the dense-MLP propagator without noise channels (2026-11-14; Saidie 9434, 30 test cases; data docs/fpe_report/data/v_eval_det.json; figures docs/fpe_report/figs_vit/det/)
`d_win_mlp`: same ViT 198 M autoencoder (v_win), same dense SwiGLU MLP body, no noise inputs, MSE loss in latent space in Stage B and in decoded space in Stage C (autoencoder unfrozen as `u_win_mlp`). One forecast per case, against the mean of 8 samples of `u_win_mlp` (RMSE of the forecast mean, normalized units):

| lead | deterministic | stochastic (mean of 8) | difference | original g_local48 |
|---|---|---|---|---|
| +5 min | 0.2442 | 0.2715 | −10.1% | 0.2917 |
| +15 min | 0.3584 | 0.3677 | −2.5% | 0.3793 |
| +30 min | 0.4368 | 0.4497 | −2.9% | 0.4511 |
| +60 min | 0.4866 | 0.4982 | −2.3% | 0.4948 |

The 10 km low-pass RMSE is not better for the deterministic model (+30 min: 0.1262 against 0.1212, +60 min: 0.1763 against 0.1690), so its mean-error advantage is in the small scales, where it predicts the conditional mean. Power relative to truth at +30 min: w 0.02 / 0.02 / 0.07 / 0.33 (bands 0.5-2 / 2-5 / 5-10 / 10-20 km) against 0.64 / 0.71 / 1.05 / 1.21 for the stochastic model; liquid 0.02 / 0.02 / 0.08 / 0.38 against 0.51 / 0.72 / 1.02 / 1.10. The sections (`det/fig_compare_sections.png`) show a smooth forecast with the updrafts, liquid and ice cores washed out, and no spread. CRPS is not comparable (the deterministic value is a mean absolute error: 0.1748 at +30 min against 0.1170).
Reading: the noise channels cost little mean skill here (2-3%, against the 12% of the earlier conv stencil comparison), buy all the power below 10 km, calibrated spread and the sharp convective cores, and are kept. A deterministic model remains the choice for a pure mean forecast, with ensemble spread from added model noise.

### 8b result: the mixer with Stage C unfrozen on three autoencoders (2026-11-14; Saidie 9441, 30 test cases; data docs/fpe_report/data/v_eval_mixer_unfrozen.json)
Mixer propagator (width 128, 0.75 M), Stage C autoencoder unfrozen. CRPS relative to `g_local48` (stencil, conv 2.6 M, frozen); power / truth at +30 min.

| arm | autoencoder | CRPS 15 | CRPS 30 | CRPS 60 | spread/RMSE at 60 | w 2-5 km | theta' 2-5 km |
|---|---|---|---|---|---|---|---|
| **um_local** | conv 2.6 M | −7.3% | **−5.4%** | **−5.3%** | 0.90 | 0.73 | 1.12 |
| um_w75 | ViT 75 M | −7.7% | −2.6% | +0.2% | 0.92 | 0.97 | 1.52 |
| um_win | ViT 198 M | −6.8% | −2.5% | +0.7% | 0.93 | 0.82 | 1.10 |
| u_win_mlp | ViT 198 M, dense MLP | −8.4% | −4.1% | −1.6% | 0.94 | 0.71 | 0.62 |
| m128 (mixer, frozen conv) | conv 2.6 M | −6.2% | −4.4% | −3.9% | 0.97 | 0.72 | 0.46 |
| u_local48 (stencil, unfrozen) | conv 2.6 M | −1.6% | −1.3% | −0.1% | 0.92 | 0.35 | 0.30 |

Reading: the best CRPS overall is the small autoencoder with the mixer and Stage C unfrozen (−5.4% / −5.3% at 30 / 60 min, about 5% ahead of the previous best full_v1_stagec_crps), with a 2.6 M autoencoder and a 0.75 M propagator. The mixer's gain stacks a little with unfreezing on the conv model (m128 −4.4 / −3.9% to −5.4 / −5.3%), but it does not stack with the large ViT autoencoders: um_w75 and um_win are about 3% worse at 30 min and at the control level at 60 min, and over-produce theta' power at 2-5 km (1.5x and 1.1x truth). Consistent with the earlier data-limited reading (D5: skill saturates near 28 storms): at 121 storms the large autoencoders reconstruct 25% better and forecast no better. The new data are the real test of the large models.
