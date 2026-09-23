# Handoff: Fourier/ViT Hybrid Architecture Investigation (2026-09-04)

**Purpose of this document**: a self-contained briefing for picking up
this specific investigation with zero prior context. It explains the
project, the vocabulary used throughout (`Section N`, Stage 1/Stage 2,
Gate 3/Gate 4), the full narrative of what was tried and why, the
verified numeric results, exactly what is running or queued right now,
and the open questions. Everything numeric here was pulled directly from
log files this session, not from memory -- see "How to verify anything
in this document" at the end.

For the full experiment-by-experiment log across the whole project
(Sections 1-83), see `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` -- this
document is a distilled, narrative summary of just the most recent arc
(Sections 71-83) plus everything needed to understand it without reading
the other 4000+ lines.

## 1. Project background

`ks-latent` (this repo, `/Users/daltonjones/Documents/latent_DA`) builds
a latent-space data-assimilation pipeline for the Kuramoto-Sivashinsky
(KS) PDE, a 1D chaotic PDE (`u_t + u*u_x + u_xx + u_xxxx = 0`, periodic
boundary conditions, `ks_latent/solver/ks.py`). The pipeline:

1. An **autoencoder (AE)** maps a high-dimensional PDE state `u` (a
   vector of `NX` grid points) to a low-dimensional **latent vector** `z`
   (dimension `d_latent`, typically 44 in this arc) and back
   (`ae.encode(u)`, `ae.decode(z)`).
2. A **propagator** learns to predict the latent state's time evolution
   directly in latent space (`z_{t+1} = propagator(z_t, ...)`), so that
   forecasting/data-assimilation can be done cheaply in the low-dim
   latent space instead of the full PDE state space.

The central open question motivating this whole arc: **can we get a
latent space that is simultaneously (a) a faithful, low-error
autoencoder, (b) easy for a propagator to learn accurate multi-step
rollouts in, and (c) geometrically "smooth" enough that a continuous
latent PDE/ODE could eventually be fit to it?** Property (b) has been
achievable for a while; (c) is the newer target driving this specific
arc, which starts from the user's framing: *"really what I want is to
have lots of the properties of 75 (the rollout was the best I've seen)
but also have smoothness in the embedding space to the point where we
can learn a pde governing the evolution in latent space."*

## 2. Vocabulary

- **"Section N"**: this project numbers every distinct experiment
  sequentially ("Section 75", "Section 81", etc.) regardless of what
  changed -- architecture, hyperparameters, regularizer weights. Each
  section typically has a launch script `scripts/sectionN_*.sh` with an
  extensive docstring comment at the top explaining exactly what changed
  and why (quoting the user's request verbatim), a Stage-1 log
  (`artifacts/logs/stage1_section{N}_*.log`), a Stage-2 log
  (`artifacts/logs/stage2_section{N}_*.log`), and often a spectrum/
  diagnostics log (`artifacts/logs/spectrum_section{N}_*.log`) and Gate
  3/4 outputs. Not every section reaches every stage (some are killed
  early when a smoke test or partial run looks bad -- e.g. Sections 73
  and 80 have no Stage-2 log at all).
- **Stage 1** (`scripts/train_stage1_patched.py`): trains the
  autoencoder AND an auxiliary propagator jointly. The AE's loss is
  reconstruction (`recon`, MSE of `decode(encode(u))` vs `u`) plus
  several regularizer terms (see below) plus an auxiliary k-step
  prediction loss (`l_pred`, weight `w_pred`, default 0.5, over `k_pred`
  steps, default 2) that nudges the encoder toward a "propagatable"
  latent space even before Stage 2 begins. `w_pred`/`k_pred` are active
  in every run in this arc without being explicitly set on the command
  line. Output: an AE checkpoint (`stage1_ae_patched_full_{TAG}.pt`) and
  an auxiliary propagator checkpoint (`stage1_prop_full_{TAG}.pt`).
- **Stage 2** (`scripts/train_stage2_patched.py`): freezes the Stage-1
  AE and trains (or warm-starts, via `--init-prop-checkpoint`, from
  Stage 1's own auxiliary propagator) a propagator on the frozen latent
  space, using a k-step rollout curriculum: `k_max` (final rollout
  length), `k_warmup_epochs` (epochs to linearly ramp from k=1 up),
  optionally a two-segment ramp via `k_mid`/`k_mid_epochs`. The headline
  metric is `val_kmax_mse` -- **pooled MSE over the ENTIRE k_max-length
  rollout**. **Critical gotcha**: `val_kmax_mse` is NOT comparable
  across runs with different `k_max` (a `k_max=20` run's `val_kmax_mse`
  includes 8 more chaotic-error-growth steps than a `k_max=12` run's, so
  it will look worse even if the underlying propagator is equally good
  or better at the steps they share). To fix this, this session added
  `--compare-k N` (`Stage2TrainingConfig.compare_k`, validated
  `1<=compare_k<=k_max`): it pools MSE over just the first `N` steps of
  the same rollout and logs it separately as `val_k{N}_mse`, which IS
  comparable across different `k_max` settings. Sections 76 onward pass
  `--compare-k 12`, so their logs show both `val_kmax_mse` (full,
  curriculum-length-dependent) and `val_k12_mse` (first 12 steps only,
  comparable to Section 75's `k_max=12` runs).
- **Regularizer terms** (Stage 1 losses, all summed into the total loss):
  `w_var`/`w_var_floor` (per-channel variance floor -- **do not enable
  `RegConfig.lambda_z`'s banded-latent variant, per persistent memory**;
  a different `lambda_z` field used throughout this arc is a *log-det*
  regularizer, not the banded one -- see `w_logdet` below, they are
  related but the banded-regularizer memory is about a different,
  specifically-avoided mechanism), `w_decorr` (decorrelation between
  latent channels, 0 throughout this arc), `w_spatial`/`--spatial-signed`
  (`spatial_coherence_loss`: rewards nearby-latent-index channels being
  correlated, weighted by a Gaussian kernel over CIRCULAR index distance
  -- **known to collapse latent effective dimensionality via
  correlation-driven redundancy when set too high**, e.g. Section 76's
  `w_spatial=0.4` broke Stage-2 learnability entirely), `w_logdet`
  (log-determinant-of-covariance regularizer, encourages the latent
  covariance to stay well-conditioned/full-rank), `lambda_z` (in this
  arc's CLI, this is `--lambda-z`, paired with `w_logdet` -- keep this
  distinct from the persistent-memory warning about `RegConfig.lambda_z`
  in a *different*, banded-latent-regularizer sense; nothing in this arc
  uses that banded variant).
- **Gate 3 / Gate 4**: post-hoc diagnostic suites run on a finished
  (Stage-1 AE + Stage-2 propagator) pair:
  - `scripts/run_analysis_suite.py` ("Gate 3 full analysis"): topology
    (persistence diagrams), correlation dimension, and **Lyapunov
    spectrum** (`lyapunov_history_D_KY`, `lyapunov_history_n_positive`,
    `lyapunov_history_lambda1`) computed from the learned latent
    propagator's own rollout dynamics. Slow (~15-25+ min). The key
    fidelity check: does the learned system's Kaplan-Yorke dimension
    `D_KY` match the TRUE KS attractor's `D_KY ~ 22` (see
    `project_ks_true_dky_benchmark` memory -- NOT ~4-5, which was an
    earlier, incorrect assumption).
  - `scripts/run_da_pff.py` ("Gate 3 DA"): particle-filter-free data
    assimilation skill test -- `rmse_da` (assimilated forecast error),
    `rmse_free` (free-running forecast error, no assimilation),
    `spread` (ensemble spread), `calibration_spread_over_rmse`,
    `skill_free_over_da` (how much DA improves over free-running; >1
    means DA helps).
  - `scripts/run_diagnostics.py` ("Gate 4"): structural diagnostics D3
    (some structural null-hypothesis test), D4 (equivariant-translation-
    representation verdict -- checks if the latent representation
    respects the PDE's spatial-translation symmetry), D6/D7/D8
    (same-time coupling / "bandedness" of the latent covariance --
    D7 is unsigned, D8 is signed).
- **"Effectively fully dense" `attn_window`**: several architectures in
  this family use a masked linear layer where each output channel only
  attends to input channels within a circular-ring distance
  `attn_window` of it (in latent-index space). Because the ring has
  `d_latent` positions, once `attn_window >= d_latent // 2` every
  position is already within window of every other position, so the
  mask is all-True and the layer is effectively a normal dense layer,
  just implemented via the masked code path. This saturation point moves
  whenever `d_latent` changes (22 at `d_latent=44`, 28 at `d_latent=56`,
  32 at `d_latent=64`) -- always re-verify by direct instantiation
  (`mask.all()`) when changing `d_latent`, do not assume the same
  absolute window size still saturates.

## 3. The other reference point: Section 52

Before Section 75 became the masked-`fourier_mlp` baseline, **Section 52**
(2026-09-01, `scripts/section52_spatial_signed_tuned5.sh`) produced a
different architecture that the user considers **"the other best model
we've seen"** and is worth keeping in mind alongside Section 75/81 as a
comparison point, not just a superseded earlier step.

**Architecture**: `encoder=vit` (a ViT-for-function-space autoencoder,
`d_model=96`, `pos_encoding=linear`, `attn_window=4`, `token_window=16`
-- this is "Section 52's ViT structure," the same one reused as the ViT
branch inside the Section 80/81/82 hybrid AE), `aux_backbone=mlp`,
`mode=markovian` (the propagator predicts the next latent state from
only the CURRENT state, no history frames -- unlike Sections 71-83's
`mode=history`/`n_history=2`). Regularizers: `w_var=0.02`,
`w_spatial=0.01` (signed), `w_decorr=0`, `w_var_floor=0`,
`w_logdet=0.0035`, no `lambda_z`/delta_cap. AE size: **816,342 params**
(smaller than Section 75's 1.0M). Stage 2: warm-started, `k_max=12`,
`k_warmup_epochs=210`/`k_mid=8`/`k_mid_epochs=175`, 300 epochs -- the
same curriculum shape reused by every section since.

**Verified results** (from
`artifacts/logs/{stage1,stage2,spectrum,gate3,gate4}_section52_mlpmarkovian_wvar002_wspatialsigned01_logdet0035_200ep*.log`):

- Stage 1: `val_recon_final = 0.000292` (worse reconstruction than
  75/81's ~0.00003-0.00005, but still small in absolute terms).
- Stage 2: **`best_val_kmax_mse = 0.025623`** at `k_max=12` -- **better
  than both Section 75 (0.0392) and Section 81 (0.0286)** at the
  identical curriculum. This is the best raw rollout-MSE number in this
  entire comparison set.
- Latent spectrum: **`cond#=1.59e6`** (catastrophic -- six orders of
  magnitude worse than 75/81's ~21-22), with two eigenvalues near-zero
  (`4.4e-5`, `1.1e-5`) alongside two dominant ones (17.7, 15.2) -- a
  near-degenerate/collapsed latent by every conditioning-based diagnostic
  used elsewhere in this document, yet it produces the best rollout fit
  of the group. D8 signed bandedness=0.654 (between 75/81's 0.34 and
  71/72/74's ~0.82).
- **Gate 3 full analysis**: uses a different Lyapunov estimator than
  Sections 75/81's Gate 3 runs -- the log reports
  `lyapunov_single_state_D_KY` (not `lyapunov_history_D_KY`; the
  `two_step` variant failed to converge, "cumulative sum of all 20
  exponents is still non-negative"). **`lyapunov_single_state_D_KY =
  21.419`**, `n_positive=11` -- close to Section 75/81's ~21 and the
  true ~22 benchmark, though not the identical estimator so treat the
  comparison as approximate.
- **Gate 3 DA skill**: `rmse_da=0.2832`, `rmse_free=0.8843`,
  `spread=0.0859`, `calibration_spread_over_rmse=0.3035`,
  **`skill_free_over_da=3.123`** -- substantially better DA skill than
  Section 81's 1.305 (assimilation helps much more here).
- **Gate 4 diagnostics**: D3 p-value=0.184, **D4 verdict: "No clean
  linear translation representation found in the existing latent"**
  (unlike Section 81's "Genuine (approximately) equivariant" verdict),
  D6/D7/D8 p-values all 0.0000.

**Why this matters for the current investigation**: Section 52 is
essentially the SAME falsification case as "Section 65's ViT AE" cited
earlier in this project's conditioning-hypothesis walkback (see
`docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md`) -- a ViT-based, markovian-mode
architecture with terrible covariance conditioning that nonetheless
produces excellent rollout accuracy and DA skill, better on both counts
than the well-conditioned Section 75/81 masked-`fourier_mlp`/hybrid
lineage. It's a live counter-example to treat carefully when interpreting
Section 82/83's conditioning numbers: **good conditioning is not
necessary for a good propagator**, and Section 52's raw performance
means the `fourier_mlp`/hybrid lineage (75->81->82->83) has not yet
strictly surpassed the best `mode=markovian` ViT result on rollout MSE or
DA skill -- only on conditioning/interpretability-flavored diagnostics
and (for 81) the D4 equivariance verdict. If the goal shifts from
"smooth, structured latent" back toward "best raw forecast skill," Section
52's markovian-ViT recipe is the number to beat, not Section 75/81.

## 4. The narrative arc (Sections 71-83)

**Where this arc starts**: after a long prior investigation (Sections
1-70, see `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` for the full
history) diagnosed that the masked `fourier_mlp` AE/propagator family's
`w_spatial` regularizer, at high weight, collapses latent effective
dimensionality via correlation-driven redundancy (verified with
covariance-conditioning/participation-ratio/encoder-Jacobian
diagnostics, see `scripts/analyze_75_76_77_latent_geometry.py`) -- but
that this diagnostic does NOT generalize across architectures (Section
65's ViT-based AE had catastrophic conditioning, `cond#=12,120,047`, yet
predicted well, falsifying a "conditioning universally predicts
learnability" hypothesis).

**Sections 71-74** iterated on the masked `fourier_mlp` architecture's
attention-window sizing, converging on `attn_window=8` (encoder) /
`22` (propagator, decoder) -- the "effectively fully dense" convention
-- plus `fourier_ifft_readout` (the masked path's final layer replaced
by an explicit `irfft` of predicted frequency-domain coefficients, the
`FourierIFFTBody` mechanism). None of 71-74 converged well in Stage 2
(`best_val_kmax_mse` in the 0.15-0.24 range).

**Section 75** paired that architecture with a revised regularizer
recipe (`w_var=0.01`, `w_spatial=0.01` signed, `w_logdet=0.008`,
`lambda_z=0.0002`, all from epoch 0) and got the best result of the
whole masked-`fourier_mlp` family: `best_val_kmax_mse=0.039243`
(`k_max=12`), `cond#=20.98` (well-conditioned), Gate 3 `D_KY=21.089`
(within ~4% of the true ~22 benchmark). **This became the reference
point/baseline for everything that follows** -- "Section 75's
regularizers" or "Section 75's recipe" means exactly these five weights.
Per the user, qualitatively: *"the rollout was the best I've seen, it
really captured the dynamics the best I've observed."*

**Sections 76-79** explored pushing regularizers or capacity further
from the Section 75 baseline, all as attempts to get "smoother"
embeddings (the (c) property from Section 1 above) without giving up
75's rollout quality:
- **76**: `w_spatial x40` (0.01->0.4) -- **broke Stage-2 learnability
  entirely** (`val_kmax_mse` stuck ~0.91-0.92 after 98/500 epochs).
  Reinforces the standing "don't push `w_spatial` too high" finding.
- **77**: `w_spatial x3` (0.01->0.03), `k_max` extended to 20 -- much
  better than 76 but still clearly worse than 75 at the steps they
  share (`val_k12_mse~0.234` vs 75's `val_kmax_mse=0.039` at the
  identical `k_max=12`... not a perfectly fair comparison across
  `k_max`, but directionally consistent).
- **78**: `d_latent` 44->64 (more room, hoping to reduce
  correlation-driven collapse at higher `w_spatial`) -- produced the
  **best-conditioned latent of the whole family** (`cond#=11.7`,
  participation ratio 37.4/64, essentially no near-collapsed
  directions) but Stage-2 fit was still mediocre (`val_k12_mse~0.142`).
  This is the clearest example of "great encoder diagnostics, mediocre
  downstream fit" and is hypothesized (not yet tested) to be a
  **propagator-capacity confound**: the same propagator architecture
  (`hidden=480/n_blocks=2`) now has to predict a 50%-larger latent, and
  the bottleneck may simply be that the propagator itself is now
  too small, not that the encoder geometry is bad. CLI flags for a
  "does a bigger propagator fix this" test were added to
  `scripts/train_stage2_patched.py` (`--fourier-ifft-readout`,
  `--nonexpansive` for fresh non-warm-started propagators) but **this
  test was never actually launched** -- see Open Questions.
- **79**: isolates `w_pred` (Stage-1 auxiliary-propagator loss weight)
  doubling (0.5->1.0), everything else identical to 77. Had essentially
  no effect on the static encoder geometry (as expected, since `w_pred`
  only touches the Stage-1 auxiliary loss) and only a modest effect on
  Stage-2 fit. Deprioritized once the ViT-hybrid direction opened up.

**The pivot to a hybrid architecture**: the user asked *"would there be a
way to constrain the fourier_mlp to be nonexpansive?"* (aiming for
smoothness via a Lipschitz-1 constraint). This was implemented via
manual power-iteration spectral normalization
(`ks_latent/models/spectral_norm.py`, `SpectralNormLinear`; needed as a
standalone leaf module to avoid a circular import between
`propagator.py` and `autoencoder_masked_mlp.py`; needed a manual
implementation rather than PyTorch's built-in
`torch.nn.utils.parametrizations.spectral_norm` because the built-in one
crashes under `--amp` on MPS with
`RuntimeError: Failed to create function state object for:
div_true_strided_float_bfloat`). **Direct empirical testing showed this
crippled capacity**: a 40-epoch smoke test plateaued at `val_recon~0.36`
vs the unconstrained architecture's `~0.0001`. Per the user's explicit
direction after seeing this result -- *"Abandon nonexpansive for now,
keep exploring other levers"* -- this path was abandoned (code kept,
since it's tested and harmless when unused, but not pursued further).

**The hybrid ViT+Fourier architecture** (Sections 80-83) came next, from
the user's explicit design: *"I want a Vit for function space, the same
structure as section 52 plus I want a Fourier mlp that only acts on
frequency space... just add the output of the ViT and the Fourier mlp."*
`encode(u) = vit.encode(u) + fourier_encoder(fourier_features(u))`,
mirrored for decode
(`ks_latent/models/autoencoder_vit_fourier_hybrid.py`,
`KSAutoencoderViTFourierHybrid`/`ViTFourierHybridAutoencoderConfig`,
registered as encoder choice `"vit_fourier_hybrid"`).
- **Section 80** used a new `ConservedFourierMLP` module (an L1-
  conservation law: each layer's output is rescaled so its L1 norm
  exactly matches the original input Fourier features' L1 norm, checked
  at every intermediate layer) as the Fourier branch. Optimization was
  rocky (large initial loss spike) and the user judged the run "doesn't
  look very good" before a real Stage-2 run was launched.
- **Section 81** swapped in `FourierIFFTBody` (Section 75's own
  frequency-domain mechanism) as the Fourier branch instead, per the
  user: *"can we have the same kind of hybrid vit fourier mlp, just
  using the exact fourier mlp from 75 except only using the frequency
  components? make it 1.5 million params total."* This is the
  **architecture-comparison headline result**: `best_val_kmax_mse =
  0.028618` at `k_max=12` (vs. Section 75's `0.039243` at the identical
  curriculum), Gate 3 `D_KY=20.914` (matching Section 75's 21.089, both
  near the true ~22 benchmark -- rollout accuracy improved without
  sacrificing Lyapunov fidelity), Gate 4 D4 verdict "Genuine
  (approximately) equivariant translation representation found." AE size:
  1,509,438 params (vs Section 75's 1,002,332) -- **1.5x bigger**, which
  the user immediately and correctly flagged as a confound: *"we don't
  really know if the hybrid is the improvement, the models did get
  larger."*
- **Section 82** takes Section 81's architecture and pushes further
  toward smoothness per the user's plan (*"increase w_spatial for 81 to
  .015 and lambda_z to .0003 ... we could try this with embedding
  dimension 56"*), gated on 81 actually beating 75 (confirmed
  mid-training before launch: 81 vs 75 at matched curriculum point,
  0.035 vs 0.053). **Currently running** (see Section 5 below).
- **Section 83** is the direct answer to the Section 81 confound: grow
  Section 75's ORIGINAL architecture (same masking, same propagator,
  same regularizers -- literally nothing else changed) to Section 81's
  exact parameter count (`fourier_mlp_hidden` 224->282, verified
  1,509,368 vs 81's 1,509,438, ratio 1.0000) and compare
  `best_val_kmax_mse` directly. **Currently queued** (see Section 5).
  A 4-epoch smoke test suggested (not conclusive) that scaling up this
  architecture's raw capacity does NOT help and may hurt
  (`val_kmax_mse=0.188` at 4 epochs, worse than either reference's own
  early trajectory) -- consistent with 77/78's finding that more
  capacity within the masked-`fourier_mlp` family alone doesn't
  straightforwardly help.

## 5. Results table (all numbers verified directly from logs this session)

| Section | Architecture | AE params | `best_val_kmax_mse` (k_max) | `val_k12_mse` | Latent `cond#` | D8 signed bandedness | Gate 3 `D_KY` |
|---|---|---|---|---|---|---|---|
| **52** | **ViT (d_model=96) + mlp/markovian propagator ("the other best model")** | **816,342** | **0.0256 (12)** | -- | **1.59e6** (catastrophic) | 0.654 | 21.42 (single_state variant) |
| 71 | masked fourier_mlp, attn=4 | ~1.0M | ~0.23-0.24 (12, not converged) | -- | 310 | 0.816 | -- |
| 72 | masked fourier_mlp, attn=12 | ~1.0M | 0.1485 (12) | -- | 365 | 0.825 | -- |
| 73 | masked enc + dense prop + IFFT dec | -- | killed, no Stage 2 | -- | -- | -- | -- |
| 74 | attn=8/22/22 + fourier_ifft_readout | ~1.0M | 0.1815 (12, not fully converged) | -- | 382 | 0.831 | -- |
| **75** | **= 74 + revised regularizers (baseline)** | **1,002,332** | **0.0392 (12)** | -- | **20.98** | **0.341** | **21.089** |
| 76 | = 75, w_spatial x40, k_max=20 | ~1.0M | not converged (~0.91 at ep 98/500) | ~0.40 | 462 | 0.846 | -- |
| 77 | = 75, w_spatial x3, k_max=20 | ~1.0M | ~0.51 (20, ep 274/500) | ~0.234 | 122 | 0.696 | -- |
| 78 | = 77, d_latent=44->64 | ~1.0M+ | ~0.385 (20, ep 376/500) | ~0.142 | **11.7** (best cond, PR=37.4/64) | 0.211 | -- |
| 79 | = 77, w_pred x2 | ~1.0M | 0.330 (best) | 0.113 | 122.6 | 0.697 | -- |
| 80 | ViT + ConservedFourierMLP hybrid | ~1.0M | killed, no Stage 2 | -- | -- | -- | -- |
| **81** | **ViT + FourierIFFTBody hybrid ("only frequency")** | **1,509,438** | **0.0286 (12)** | -- | **22.3** | **0.341** | **20.914** |
| 82 | = 81, d_latent=56, w_spatial x1.5, lambda_z x1.5 | 1,552,374 | **running** (Stage 1 ep 130/200) | -- | -- | -- | -- |
| 83 | Section 75 architecture grown to 81's size (control) | 1,509,368 | **queued** | -- | -- | -- | -- |

Notes: `val_kmax_mse` is only directly comparable between rows with the
same `k_max` (75 vs 81 vs 83 target, all `k_max=12`); rows with
`k_max=20` are cross-referenced via `val_k12_mse` where available (added
via `--compare-k 12` starting at Section 76). Section 71/74 did not fully
converge by the end of their logged Stage-2 runs, so their numbers are
directional, not final.

## 6. Current live state (verify with `ps aux` and log files -- this changes)

As of the last check this session:

- **Section 82** (`scripts/section82_dlatent56_more_spatial.sh`,
  PID chain: shell script PID 93083, `mamba run` wrapper PID 93483,
  actual python training process PID 93484) is running Stage 1, last
  seen at **epoch 130/200, `recon=0.000103`**, decreasing smoothly. Log:
  `artifacts/logs/stage1_section82_dlatent56_vitfourieriffthybrid_dmodel92_fourierhidden270blocks2_wspatial015_lambdaz0003_200ep.log`.
  Once Stage 1 finishes, the same script automatically runs the
  covariance-spectrum/D7-D8 check and then Stage 2
  (`--k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175`,
  warm-started from Stage 1's own auxiliary propagator, 300 epochs) --
  check `artifacts/logs/stage2_section82_..._warmstart_k12_300ep.log`
  once it appears.
- **Section 83** (`scripts/section83_fourier_mlp_1_5m_control.sh`) is
  **queued**, not yet started. A chain-waiter shell process (PID 93835,
  `artifacts/logs/section83_chain_waiter.log`, currently empty) is
  polling every 30s and will auto-launch Section 83 once BOTH PID 93083
  (Section 82's outer pipeline script) and PID 93461 (Section 81's Gate
  3/4 analysis job, which has already finished) have exited -- so in
  practice it is waiting only on Section 82's full pipeline (Stage 1 +
  spectrum check + Stage 2) to complete. **Do not manually launch
  Section 83** -- it will start automatically; check
  `ps aux | grep 93835` to confirm the waiter is still armed, and check
  for the appearance of
  `artifacts/logs/stage1_section83_fouriermlp_1_5m_hidden282_section75regs_200ep.log`
  to know it has started.
- **Section 81's Gate 3/4 diagnostics are complete** (this was
  double-checked this session after an earlier reply had incorrectly
  said they were "still running"): full results are in Section 3/4
  above and in `artifacts/logs/gate3_analysis_section81_..._warmstart_k12_300ep.log`,
  `gate3_da_section81_..._warmstart_k12_300ep.log`,
  `gate4_diagnostics_section81_..._warmstart_k12_300ep.log`. A
  visualization (`scripts/visualize_rollout.py`) and latent-state GIF
  (`scripts/make_latent_gif.py`) were also generated for Section 81 as
  part of the same batch script (`scripts/section81_viz_gate34.sh`).

## 7. Open questions / suggested next steps

1. **The Section 81-vs-75 confound (primary open question)**: is
   Section 81's improvement over 75 (`best_val_kmax_mse` 0.0286 vs
   0.0392) due to the hybrid architecture itself, or just having 1.5x
   more parameters? Section 83 is designed to answer this directly and
   is queued to run automatically -- when it finishes, compare its
   `best_val_kmax_mse` (at `k_max=12`, same as 75/81) against both
   references. The 4-epoch smoke test hinted size alone might actually
   hurt (not help), which would argue FOR the hybrid architecture being
   the real driver -- but that smoke signal is weak (4 epochs only) and
   should not be treated as settled until the real run completes.
2. **Section 82's outcome**: once it finishes, check whether pushing
   `w_spatial`/`lambda_z` up 1.5x from Section 81's already-good recipe,
   combined with a larger `d_latent=56`, actually delivers the
   "progressively smoother embeddings while keeping 75/81's rollout
   quality" goal that motivated this whole arc, or whether it starts
   trending back toward Section 76/77's degradation pattern (higher
   `w_spatial` hurting learnability). Compare its eventual
   `best_val_kmax_mse`/`val_k12_mse`/`cond#`/D8-bandedness/Gate-3 `D_KY`
   against Section 81's numbers in the table above.
3. **Section 78's unresolved propagator-capacity hypothesis**: never
   actually tested (CLI flags exist in `train_stage2_patched.py` but no
   run was launched). Worth revisiting if `d_latent` is grown again
   (e.g. after Section 82) and Stage-2 fit still lags the encoder's own
   diagnostic quality -- try training a larger propagator
   (`--aux-hidden`/`--aux-blocks` increased) on the SAME frozen AE to see
   if that alone closes the gap. Do not launch this without a specific
   trigger (e.g. Section 82 showing the same "good encoder, mediocre
   propagator fit" pattern as 78) since it wasn't in the user's most
   recent explicit request list.
4. **If Section 83 confirms the hybrid architecture is genuinely
   better** (not just bigger): a natural next step would be applying the
   same "smoothness push" (higher `w_spatial`/`lambda_z`, larger
   `d_latent`) that Section 82 is testing on top of Section 81, and/or
   trying the ViT-hybrid architecture at Section 75's ORIGINAL size
   (~1.0M, i.e. the reverse control -- shrink 81 down to 75's size rather
   than growing 75 up to 81's) to further disentangle size from
   architecture in both directions.
5. **Latent-smoothness metric still undefined**: this whole arc has been
   optimizing rollout accuracy and conditioning as PROXIES for "can we
   eventually learn a continuous latent PDE/ODE" -- no direct smoothness
   metric (e.g. local Lipschitz constant of the true continuous flow map
   restricted to the data manifold, or finite-difference second-derivative
   estimates along real trajectories in latent space) has been computed
   yet. If the user wants to move toward literally fitting a latent PDE,
   this is likely the next methodological gap to fill.

## 8. Key files

- `ks_latent/models/autoencoder_vit_fourier_hybrid.py` -- the hybrid AE
  (`KSAutoencoderViTFourierHybrid`, `ConservedFourierMLP`).
- `ks_latent/models/propagator.py` -- `FourierIFFTBody` (reused by both
  the pure `fourier_mlp` family and the hybrid AE's Fourier branch),
  plus the (implemented, tested, currently unused) `nonexpansive`
  spectral-norm plumbing.
- `ks_latent/models/spectral_norm.py` -- standalone `SpectralNormLinear`
  (manual power iteration, autocast-safe).
- `ks_latent/config.py` -- `ViTFourierHybridAutoencoderConfig`
  (`fourier_kind: "conserved"|"ifft"`), `Stage2TrainingConfig.compare_k`.
- `ks_latent/training/loops.py` -- `eval_stage2_kmax` (now returns
  `(kmax_mse, compare_mse)` tuple when `compare_k` is set).
- `scripts/train_stage1_patched.py`, `scripts/train_stage2_patched.py` --
  main training entry points; see each section's launch script for the
  exact CLI flags used.
- `scripts/analyze_75_76_77_latent_geometry.py` -- reusable latent
  geometry diagnostic (cond#, participation ratio, encoder Jacobian
  spectral norm, k-NN sensitivity ratio) -- reliable only WITHIN the
  `fourier_mlp` family, not universally (falsified by Section 65's ViT
  AE).
- `scripts/section{N}_*.sh` for N in 71-83 -- one launch script per
  section, each with a detailed docstring comment explaining what
  changed and why, quoting the user's request.
- `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` -- the full project history,
  now including Sections 71-83 appended in the same session that
  produced this handoff document.

## 9. How to launch a new training run

Every run in this arc follows the same three-command pattern, executed
inside a zsh script under `scripts/`. There is no separate "config file"
to edit -- everything is passed as CLI flags to
`scripts/train_stage1_patched.py` then `scripts/train_stage2_patched.py`.
The fastest way to launch a new variant is to copy the most similar
existing `scripts/sectionN_*.sh` file, change the flags that differ, and
run it -- do not write Stage 1/Stage 2 commands from scratch.

**Environment**: all training/analysis commands run inside the `da_env`
mamba/conda environment via `mamba run -n da_env python ...` (never bare
`python`). Training runs on MPS (Apple Silicon GPU) by default; Gate 3/4
diagnostic scripts often fall back to CPU implicitly for parts that don't
support MPS ops (hence the `_scaled_dot_product_flash_attention_for_cpu`
warnings seen in Gate 3/4 logs -- these are harmless).

**The three-step pattern** (see any `scripts/sectionN_*.sh` for a
complete worked example, e.g. `scripts/section81_vit_fourier_ifft_hybrid.sh`):

1. **Stage 1** -- trains the AE + auxiliary propagator jointly:
   ```bash
   mamba run -n da_env python scripts/train_stage1_patched.py \
     --profile full \
     --encoder <fourier_mlp|vit|vit_fourier_hybrid> --aux-backbone <fourier_mlp|mlp> \
     --mode <history|markovian> [--n-history 2] \
     [architecture-specific flags -- attn windows, d_model, fourier hidden/blocks, etc.] \
     --w-decorr 0 --w-var <..> --w-spatial <..> [--spatial-signed] --w-var-floor 0 --w-logdet <..> \
     --lambda-z <..> --reg-start-epoch 0 \
     --full-propagator --amp \
     --epochs 200 --checkpoint-every 20 \
     --tag "<TAG>" \
     > artifacts/logs/stage1_<TAG>.log 2>&1
   ```
   Produces `artifacts/stage1_ae_patched_full_<TAG>.pt` (AE) and
   `artifacts/stage1_prop_full_<TAG>.pt` (auxiliary propagator).
2. **(Optional but standard) spectrum/bandedness check** -- a short
   inline Python block (copy verbatim from any section script) that
   loads the Stage-1 AE checkpoint, encodes a validation trajectory
   batch, and prints the latent covariance spectrum (`cond#`,
   participation ratio), D7/D8 bandedness, and raw spatial-coherence
   loss values -- written to `artifacts/logs/spectrum_<TAG>.log`.
3. **Stage 2** -- freezes the AE, trains/warm-starts the propagator with
   a k-step rollout curriculum:
   ```bash
   mamba run -n da_env python scripts/train_stage2_patched.py \
     --ae-checkpoint artifacts/stage1_ae_patched_full_<TAG>.pt \
     --init-prop-checkpoint artifacts/stage1_prop_full_<TAG>.pt \
     --amp \
     --epochs 300 --k-max 12 --k-warmup-epochs 210 --k-mid 8 --k-mid-epochs 175 \
     [--compare-k 12] \
     --tag "<TAG>_warmstart_k12_300ep" \
     > artifacts/logs/stage2_<TAG>_warmstart_k12_300ep.log 2>&1
   ```
   `--init-prop-checkpoint` is optional -- omit it to train a fresh
   (non-warm-started) propagator from scratch instead. Produces
   `artifacts/stage2_prop_patched_full_<STAGE2_TAG>.pt`.

**Gate 3/4 diagnostics**, once Stage 2 finishes, all take the same
`--ae-checkpoint`/`--prop-checkpoint` pair and a `--tag` (see
`scripts/section81_viz_gate34.sh` for the exact invocations, which also
shows how to run all three in parallel with `&`/`wait`):
```bash
mamba run -n da_env python scripts/run_analysis_suite.py --ae-checkpoint <AE> --prop-checkpoint <PROP> --tag <TAG>   # Gate 3 full (slow, ~15-25+ min)
mamba run -n da_env python scripts/run_da_pff.py         --ae-checkpoint <AE> --prop-checkpoint <PROP> --tag <TAG>   # Gate 3 DA
mamba run -n da_env python scripts/run_diagnostics.py    --ae-checkpoint <AE> --prop-checkpoint <PROP> --tag <TAG>   # Gate 4
mamba run -n da_env python scripts/visualize_rollout.py  --ae-checkpoint <AE> --prop-checkpoint <PROP> --tag <TAG>   # rollout plot
mamba run -n da_env python scripts/make_latent_gif.py    --ae-checkpoint <AE> --tag <TAG> --fps 12 --style line      # latent-state GIF (AE only, no propagator needed)
```

**Practical conventions established this arc, worth reusing**:
- Always run a cheap 4-epoch smoke test of both stages (same commands,
  `--epochs 4`) before committing to a full 200+300-epoch run, and
  directly instantiate the model in a one-off `python -c` snippet to
  verify parameter counts and (if `attn_window` or `d_latent` changed)
  that masks saturate as expected (`mask.all()`), before launching for
  real. Delete smoke-test artifacts/logs afterward so they don't get
  confused with the real run.
- To sequence runs without wasting GPU time on concurrent contention, use
  a "chain waiter": `nohup zsh -c 'while kill -0 <PID> 2>/dev/null; do sleep 30; done; <next command>' &` --
  this is exactly the mechanism currently queuing Section 83 behind
  Section 82 (see Section 6 above).
- Every section script's docstring comment at the top should state, in
  the user's own words where possible, what changed and why, plus any
  verified sizing/smoke-test numbers -- this is what makes the
  `docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md` writeups possible without
  re-deriving context later.

## 10. How to verify anything in this document

Every number above was extracted directly from log files, e.g.:

```bash
cd /Users/daltonjones/Documents/latent_DA
grep -E "val_kmax_mse|val_k12_mse|best_val" artifacts/logs/stage2_section81_*.log | tail -5
cat artifacts/logs/spectrum_section75_*.log
grep -E "D_KY|lambda1|n_positive" artifacts/logs/gate3_analysis_section81_*.log
ps aux | grep -E "train_stage|section8"
```

Do not trust this document's numbers blindly if it's been more than a
short while since 2026-09-04 -- Sections 82 and 83 in particular were
still in progress when this was written; re-check their logs for final
results before citing them as complete.
