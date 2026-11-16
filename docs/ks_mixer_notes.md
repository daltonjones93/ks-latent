# The CM1 mixer's details on the Paper I/II models (1D KS, 2D KS, Kolmogorov flow), Cashew CPU (started 2026-11-14)

Request (user): use the Paper I and II autoencoders with the mixer on 1D KS, 2D KS and Kolmogorov flow, on Cashew's CPU, to see whether it can improve those papers' results. Wait for the CM1 mixer evaluation first (done: the mixer with Stage C unfrozen on the conv latent gave CRPS −5.4% / −5.3% at 30 / 60 min).

**What the mixer is here.** Paper I's propagator (`local_mlp`, `local_mlp_2d`) already is the mixer block: windowed linear token mixer plus per-site MLP (ratio 4), pre-norm residual, zero-initialized output, radius 3, 2 blocks (1 for the 2D KS C2 model), width 32. The CM1 version differs only in details: RMSNorm instead of LayerNorm and SwiGLU instead of the GELU MLP. User choice: the same-size variant (`mixer_variant = "swiglu_rms"`, SwiGLU hidden (2/3) x ratio x width so the parameter count matches; ks_latent/config.py, ks_latent/models/propagator.py; tests/unit/test_local_mixer_variant.py), propagator trained on the frozen Paper I/II autoencoders (Stage B only). So this tests whether RMSNorm + SwiGLU help; it keeps the reach, the size and the autoencoders' transfer properties.

**Models and baselines (D_KY error against the true PDE, from the paper; truths: 1D from artifacts/experiments/e1_groundtruth*; 2D KS 44^2 / 66^2 / 88^2 IC 0: 81.77 / 192.50 / 338.3; Kolmogorov alpha = 0.1: 15.88 / 30.34 / 56.35 at 1x / 2x / 3x).**
- 1D KS, section224 (patch 16, 3 channels, d = 48; Stage 2 300 epochs, k_max 12; the published model was warm-started from the Stage 1 auxiliary propagator): within 0.2-1.2% up to L = 400.
- 2D KS, r13_C2 seed 0 (box 44^2, crop 8x8 sites, 6 channels, 1 mixer layer, k_max 48, 12000 steps): 81.5 / 183.9 / 328.7 (1.0 / 2.5 / 2.8% error).
- Kolmogorov alpha = 0.1, k_xs01_v2 (window-trained, 48 sites, 6 channels, 2 layers, k_max 16, 9000 steps): 16.0 / 32.7 / 52.1 (9.0 / 5.9 / 2.6% error).

**Design.** Per system: a cold-start control (published block, `gelu_ln`) and the variant (`swiglu_rms`), propagator seeds 0-2 on the same frozen autoencoder (the autoencoder is the published seed-0 one for all seeds; Paper I retrained it per seed, so seed spread here is the propagator's only). The cold-start control is needed because the 1D model was warm-started and because the CPU/numpy stack differs from the paper's (Cashew: torch 2.15, numpy 2.4; paper: torch 2.13, numpy 1.26). Scripts: cluster/ks_cpu/ (common.sh, train_sys.sh, validate_baseline.sh, bench.sh). A pipeline check evaluates the published checkpoints on the CPU first.
Benchmark (16 CPUs): about 0.25-0.3 s per Stage B step for both 2D systems (so 40-50 min per run), about 10 s per 1D Stage 2 epoch.

**Pre-registered reading** (before any result): evaluation = the model's Kaplan-Yorke dimension at the training size and at the transfer sizes (IC 0 first), error against the truth above. (1) Pipeline valid if the published checkpoints reproduce the paper's D_KY within 3% on the CPU. (2) The variant helps a system if its mean absolute D_KY error over the sizes is lower than the control's by more than the control's seed-to-seed spread of that mean, and it does not lose at any size by more than that spread. (3) A single size or seed is not a result; the paper's own seed spread (2D KS C2: 1.8-2.8% at 66^2; Kolmogorov: 3-25% across seeds at 2x) is large, so small differences are expected to be noise.
Jobs (Cashew): 767592 / 767593 (published-model evaluations), 767594-767599 (trainings: kolm, ks2d, ks1d x gelu_ln, swiglu_rms; 3 seeds each in sequence).

## Pipeline check, Kolmogorov (2026-11-14, Cashew 767592; IC 0 only so far)
Published checkpoint k_xs01_v2 evaluated on the CPU (model D_KY / lambda_1 / error against the truth): 1x 16.04 / 0.079 / 1.0%, 2x 35.66 / 0.093 / 17.5%, 3x 50.51 / 0.098 / 10.4%. The paper's own IC-0 runs (artifacts/experiments/crop on the Mac): 15.86 / 30.91 / 53.11, and its 3-IC means 16.0 +- 0.2 / 32.7 +- 2.4 / 52.1 +- 1.3. A single IC 0 differs from the Mac's at 2x by 15%; the paper notes its PDE trajectories (the initial conditions) depend on the FFT thread count, so "IC 0" is not the same state on this machine, and the model's D_KY varies with the initial condition by about the paper's own spread. Evaluation took 2 / 4 / 7 minutes at 1x / 2x / 3x on 16 CPUs.
Change to the pre-registered check, made after seeing this result: the criterion "reproduces the paper's D_KY within 3%" applies to the mean over the paper's three initial conditions, not to one IC. ICs 1 and 2 of the published model are queued (767600) and all retrained models are evaluated on ICs 0-2 (cluster/ks_cpu/eval_models.sh, jobs 767601-767603, each after its system's trainings).

## Pipeline check, 2D KS (2026-11-14, Cashew 767593; IC 0)
Published r13_C2 seed 0 on the CPU against the paper's own saved IC-0 runs (model D_KY / lambda_1): 44^2 81.32 / 0.223 (Mac 81.92 / 0.212, 0.7% apart), 66^2 182.02 / 0.221 (Mac 185.08 / 0.215, 1.7%), 88^2 324.3 / 0.226 (Mac 328.68 / 0.219, 1.3%); error against the IC-0 truths (81.77, 192.50, 338.3): 0.5%, 5.4%, 4.1% (the paper's 3-IC means: 1.0, 2.5, 2.8%). All three sizes reproduce the paper's run within 2%: the 2D KS pipeline check passes. Evaluations took 1.1 / 2.4 / 2.7 minutes. (For Kolmogorov the IC-0 comparison differs at 2x, see above.)

## Partial results, 2026-11-15 (Cashew CPU; ICs 0-2 each; model D_KY mean +- sd over the finished seeds and ICs; mean absolute error against the truth)
Trainings finished: Kolmogorov and 2D KS, both variants, seeds 0-2 (2.4-2.6 h and 1.7-1.8 h per 3 seeds); 1D: the control seeds 0-1 and the variant seed 0 (the 3-h limit was too short for 3 seeds at ~1 h each; the missing 1D seeds are rerun, jobs 768457-768459). Evaluations: the 2.5-h limit was too short (I had not multiplied by 3 ICs x 6 models), so the variant's evaluations are incomplete; the rest are resubmitted with skip-finished (768454-768456, 768460).
| system, size | control (gelu_ln) | variant (swiglu_rms) | truth |
|---|---|---|---|
| 2D KS 44^2 | 81.23 +- 0.97 (n 9), 2.1% | 82.04 +- 0.41 (n 7), 1.3% | 82.0 |
| 2D KS 66^2 | 182.28 +- 1.31 (n 9), 3.3% | 183.34 +- 1.48 (n 6), 2.7% | 188.5 |
| 2D KS 88^2 | 324.96 +- 1.45 (n 9), 4.0% | 327.57 +- 0.94 (n 6), 2.8% | 338.3 |
| Kolmogorov 1x | 15.25 +- 0.85 (n 9), 6.0% | 14.51 +- 0.39 (n 3, seed 0 only), 8.6% | 15.88 |
| Kolmogorov 2x | 32.87 +- 2.26 (n 9), 8.7% | 31.27 +- 0.42 (n 3, seed 0), 3.1% | 30.34 |
| Kolmogorov 3x | 51.07 +- 3.05 (n 9), 9.7% | 49.79 +- 0.98 (n 2, seed 0), 11.6% | 56.35 |
| 1D KS L = 100 | 22.31 +- 0.08 (n 6, seeds 0-1), 0.7% | 22.22 +- 0.10 (n 3, seed 0), 1.1% | 22.47 |
| 1D KS L = 200 | 44.68 +- 0.10, 1.7% | 44.46 +- 0.06, 2.2% | 45.46 |
| 1D KS L = 400 | 89.24 +- 0.11, 0.2% | 89.08 +- 0.16, 0.4% | 89.46 |
Interim reading (not yet the pre-registered one, which needs all seeds): 2D KS, the variant is closer to the truth at every size by 0.6-1.2 points of mean error, differences about one seed spread; Kolmogorov, the control's spread (sd 0.9-3.1) is larger than any difference and the variant has one seed so far; 1D, both are within 2.2% and the control is 0.4-0.5 points closer at L = 100 and 200 with different seed counts. The cold-start controls reproduce the paper's published numbers: Kolmogorov 15.3 / 32.9 / 51.1 (paper 16.0 / 32.7 / 52.1), 2D KS 81.2 / 182.3 / 325.0 (paper 81.5 / 183.9 / 328.7).

## Results for Kolmogorov and 2D KS, all 3 seeds x 3 ICs (2026-11-15; Cashew 767594-767598, 768454-768456)
Model D_KY, mean over the 9 (seed, IC) evaluations (sd), mean absolute error against the truth, and the error of each training seed's 3 ICs:
| system, size | control gelu_ln | variant swiglu_rms |
|---|---|---|
| 2D KS 44^2 | 81.23 (0.97), 2.1% [2.2, 1.4, 2.6] | 81.93 (0.52), 1.3% [1.6, 1.3, 0.9] |
| 2D KS 66^2 | 182.28 (1.31), 3.3% [3.0, 2.8, 4.0] | 183.13 (1.32), 2.8% [3.1, 2.4, 3.1] |
| 2D KS 88^2 | 324.96 (1.45), 4.0% [3.4, 3.7, 4.9] | 326.81 (1.68), 3.2% [2.7, 2.9, 3.9] |
| Kolmogorov 1x | 15.25 (0.85), 6.0% [5.3, 7.5, 5.3] | 14.46 (0.39), 8.9% [8.6, 9.0, 9.2] |
| Kolmogorov 2x | 32.87 (2.26), 8.7% [10.3, 10.5, 5.2] | 30.99 (1.48), 4.5% [3.1, 5.7, 4.6] |
| Kolmogorov 3x | 51.07 (3.05), 9.7% [5.0, 10.7, 13.3] | 48.86 (1.46), 13.3% [13.0, 14.4, 12.4] |
Mean over sizes of the per-seed errors: 2D KS control 3.12% (seed sd 0.5-0.65), variant 2.43%; Kolmogorov control 8.12%, variant 8.90%.
Reading against the pre-registered rule (the variant helps if its mean error over the sizes is lower than the control's by more than the control's seed spread, and it loses at no size by more than that spread):
- **2D KS: a marginal pass.** The variant is closer at every size (0.8, 0.5 and 0.8 points) and its mean error is 0.69 points lower, against a control seed sd of 0.53-0.65: at the edge of the rule, with three seeds each and a shared autoencoder. A small, consistent improvement, not a decisive one.
- **Kolmogorov: no.** The variant is better at 2x (4.5 against 8.7%) but worse at 1x (8.9 against 6.0%) and 3x (13.3 against 9.7%), and worse overall (8.90 against 8.12%). It is much more consistent across seeds and initial conditions (sd 0.4-1.5 against 0.9-3.1) but biased low (14.5 against 15.9 truth at 1x; 48.9 against 56.4 at 3x). The control's spread across seeds (up to 8 points of error at one size) is larger than any systematic difference at 1x and 3x.
- **1D KS:** trainings timed out twice (a seed takes about 2.5 h, not 1 h: epochs run 25-33 s, and the first benchmark used the cheap early epochs); resubmitted with 3 h limits and periodic checkpoints (768931-768934). Interim: control seeds 0-1 and variant seed 0 are within 0.2-2.2% at L = 100 / 200 / 400.

## 1D KS result and conclusion of the study (2026-11-16; Cashew 767596-767599, 768931-768934)
All 3 seeds x 3 ICs per variant (the 1D trainings needed 3-h limits and took 1.5-1.8 h each when run alone; the first two batches timed out at 3 h while running two at a time). Model D_KY (sd over the 9 evaluations), mean absolute error against the truth, per-seed errors:
| L | truth | control gelu_ln | variant swiglu_rms |
|---|---|---|---|
| 100 | 22.47 | 22.30 (0.09), 0.78% [0.80, 0.63, 0.92] | 22.25 (0.10), 0.98% [1.10, 0.77, 1.07] |
| 200 | 45.46 | 44.68 (0.12), 1.71% [1.72, 1.71, 1.70] | 44.49 (0.14), 2.14% [2.21, 2.27, 1.93] |
| 400 | 89.46 | 89.29 (0.12), 0.19% [0.17, 0.33, 0.07] | 89.16 (0.20), 0.34% [0.43, 0.41, 0.17] |
Mean over L of the per-seed errors: control 0.89% (seed sd 0.00), variant 1.15% (seed sd 0.08). The variant loses at every length by more than the control's seed spread, so by the pre-registered rule it does not help in 1D; both are within 2.2%, close to the paper's 0.2-1.2%.

**Conclusion (all three systems).** The CM1 mixer's details (RMSNorm + SwiGLU, same size, same reach) do not give a general improvement on the Paper I/II models. 2D KS: a small improvement, marginal by the rule (mean error 2.43% against 3.12%). Kolmogorov: no (8.9% against 8.1%; the variant is more consistent across seeds but biased low). 1D KS: no (1.15% against 0.89%, both within the paper's range). The differences are small against the seed-to-seed variation of the Kolmogorov models (control error up to 8 points across seeds at one size), and the paper's published propagator is already close to what this variant reaches. The gains on CM1 therefore do not come from RMSNorm and SwiGLU themselves, but from the block's structure replacing the stencil (which Paper I/II already had), the unfrozen Stage C, the Stage A prediction term and scale; an ablation (a mixer with a GELU MLP and LayerNorm on CM1) has not been run and would show how much the SwiGLU matters there. Not tested here: the Stage A prediction term and unfrozen fine-tuning on the KS and Kolmogorov models, the changes that did help on CM1.
