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
