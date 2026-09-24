#!/usr/bin/env python
"""Phase 3 entry point: train the Stage-1 patched-transformer autoencoder
(brief §5.1).

    python scripts/train_stage1_patched.py                  # full run (~30min on MPS)
    python scripts/train_stage1_patched.py --profile smoke   # <60s at tiny dims

Device policy (brief §1.3.8, measured in docs/RESULTS.md): Stage-1 AE trains
~2.9x faster on MPS than CPU at full scale, so `full` defaults to MPS;
`smoke` stays on CPU (tiny enough that device choice doesn't matter and CPU
keeps CI simple).

**Chosen recipe, 2026-08-29 (brief §5.1 addendum + docs/RESULTS.md
"Collapse follow-up"):** the original `mode="two_step"`, `dt_snap=0.25`
recipe reliably collapsed at the brief's originally-stated canonical
NX=1024 scale and never recovered within a 60-epoch budget.
`mode="markovian"` + a larger `dt_snap=1.0` (`snapshot_every=20`) was
chosen instead: `dt_snap` is the variable that measurably speeds up
escaping the collapse fixed point (see the step-resolved diagnostic in
docs/RESULTS.md); `markovian` was chosen on top of it for the
physical-modeling reason discussed with the user (the KS PDE is
first-order in time, so a two-step history is an unmotivated crutch), even
though a single-seed factorial comparison found it roughly
parameter-count-for-parameter-count neutral, not a decisive win on its
own. **Separately, the canonical `NX` itself was changed from 1024 to 256**
(see `KSConfig`'s docstring in `ks_latent/config.py`): NX=1024 never
escaped collapse in the epoch budgets tried; NX=256 does, ~4-5x faster per
epoch, with no solver-accuracy cost (still deeply spectrally-resolved at
L=100). All of this is a deliberate, documented deviation from the brief's
literal defaults (`NX=1024`, `mode="two_step"`, `dt_snap=0.25`), not a
silent change -- override with `--mode`/`--dt-snap` (and pass an explicit
`--dataset` generated at a different `NX` via
`ks_latent.solver.dataset.generate_trajectory_dataset`) to reproduce the
original recipe.

**Ported improvements, 2026-08-29** (CLAUDE_CODE_BRIEF.md §5.1 "Ported
improvements" addendum): a reference implementation at
`/Users/daltonjones/Documents/experiments/ks_latent/` trained to visibly
lower loss/higher accuracy than this project's original recipe. Two of its
fixes are now the default here rather than opt-in, because they are
straightforwardly correct regardless of any other config choice:
`weight_decay=1e-5` explicit on AdamW (previously silently 0.01, PyTorch's
default -- 1000x heavier than intended, decaying even the zero-initialized
propagator output head every step) and a linear LR warmup before the
cosine decay (previously the full peak LR hit a freshly-initialized,
non-identity encoder/decoder from epoch 0). `--noise-std`/`--lambda-z`/
`--lambda-decorr-band` are new, off-by-default *experiments* to try
(denoising reconstruction training; banded latent-index-smoothness +
off-band decorrelation), not yet validated as canonical-recipe changes.
**`--lambda-z` is not recommended -- see `RegConfig`'s docstring: the
reference project's own measurements already showed it collapsing the
latent.**

**Architecture/rollout comparison, 2026-08-29** (user-directed, following
the reference project's own L=94/d_z=45 benchmark reaching val recon MSE
~0.008 against this codebase's ~0.81 at the same operating point -- see
CLAUDE_CODE_BRIEF.md §5.1/5.2 addendum): two more variables from that
comparison, tested independently and in combination via `--encoder` and
`--multistep`.

- `--encoder mlp` swaps `KSAutoencoderPatched` (patched-transformer) for
  `KSAutoencoderMLP` (plain MLP, `MLPAutoencoderConfig` -- ported from the
  reference project's `MLPEncoder`/`MLPDecoder`). Default `transformer`
  (unchanged behavior).
- `--multistep` swaps the tiny, discarded-after-training `AuxPropagator`
  (brief §5.1: `hidden=64, n_blocks=2`, ~22k params, fixed `k_pred=2`) for
  one sized like the real Stage-2 propagator (`hidden=128, n_blocks=3`)
  trained with a rollout length ramped `2 -> 8` over the first 30% of
  epochs (`Stage1TrainingConfig.k_pred_max`/`k_pred_warmup_epochs` --
  starting at the full rollout length immediately is unstable, see that
  config's docstring). Off by default (fixed `k_pred=2`, small aux,
  unchanged behavior).

These are independent flags specifically so the two variables (encoder
architecture, rollout regime) can be tested in isolation or together --
four runs, `--tag` distinguishes their output artifacts.

**`--encoder vit`, 2026-08-29** (user-directed): a third architecture,
`KSAutoencoderViT`/`ViTAutoencoderConfig` -- also transformer-based, but
structurally different from the patched-transformer in the one respect
that matters most: it adds a periodicity-respecting positional encoding
(`CircularPositionalEncoding`) to every token before any attention layer,
on both encoder and decoder. `KSAutoencoderPatched` has **no positional
encoding anywhere** -- see `ViTAutoencoderConfig`'s docstring for the full
comparison. `patch_size=8` (same tokenization as the patched-transformer,
`n_tokens=32`) was chosen by the same tokens-per-characteristic-structure
reasoning the reference project used for its own default.

**`--full-propagator`, 2026-08-30** (user-directed, following persistent
Stage-2 fixed-point collapse -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md):
restructures the phased pipeline so Phase 1 trains the SAME, full-sized
propagator jointly with the AE (rather than a small aux discarded after
Stage 1), then saves it as its own checkpoint (`stage1_prop_{profile}{tag}.pt`,
`{"prop_state_dict", "prop_config"}` format) for
`train_stage2_patched.py --init-prop-checkpoint` to load and fine-tune for
longer rollouts, instead of Phase 2 starting a fresh propagator from
scratch. Implies Stage-2-standard sizing (`hidden=128, n_blocks=3`)
regardless of `--multistep`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch

from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    FourierMLPAutoencoderConfig,
    KSConfig,
    LocalFieldAutoencoderConfig,
    MaskedMLPAutoencoderConfig,
    MLPAutoencoderConfig,
    RegConfig,
    SpectralFieldAutoencoderConfig,
    Stage1TrainingConfig,
    ViTAutoencoderConfig,
    ViTFourierHybridAutoencoderConfig,
)
from ks_latent.models import load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.autoencoder_fourier_mlp import KSAutoencoderFourierMLP
from ks_latent.models.autoencoder_local_field import KSAutoencoderLocalField
from ks_latent.models.autoencoder_masked_mlp import KSAutoencoderMaskedMLP
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.autoencoder_spectral_field import KSAutoencoderSpectralField
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.autoencoder_vit_fourier_hybrid import KSAutoencoderViTFourierHybrid
from ks_latent.models.propagator import AuxPropagator, aux_cfg_to_propagator_cfg
from ks_latent.solver.dataset import generate_trajectory_dataset
from ks_latent.training.loops import train_stage1
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument(
        "--d-latent", type=int, default=None,
        help="Override the canonical d_latent=44 (--profile full only; --profile smoke keeps "
        "its own fixed d_latent=8). User-directed (2026-09-04): 'what if we increased the size "
        "of latent space? this might give us little more wiggle room for more smoothness but "
        "still having the same D_KY' -- w_spatial's local-coherence pressure operates over a "
        "FIXED ABSOLUTE bandwidth (spatial_bandwidth, index units), so a larger d_latent means "
        "the same absolute smoothing pressure eats a smaller fraction of the space, leaving "
        "more effectively-independent directions (participation ratio) for KS's own true "
        "positive-Lyapunov-exponent count to each get their own room, even under the same "
        "w_spatial. attn_window (encoder/--dec-attn-window/--prop-attn-window) stays in "
        "absolute d_latent-ring units too -- raise those explicitly if they need to track the "
        "new d_latent (e.g. to stay at the new 'effectively fully dense' saturation point of "
        "d_latent//2).",
    )
    parser.add_argument(
        "--nx", type=int, default=None,
        help="Override the canonical NX=256 physical/state dimension (--profile full only). "
        "Added 2026-09-22, user-directed: needed to train on non-KS datasets with a different "
        "state size (e.g. Lorenz-96 at N=64) -- every prior use of this script trained on KS "
        "data, always at NX=256, so this was hardcoded and never needed overriding until now. "
        "Must match --dataset's own trajectories.shape[-1] exactly, or the encoder's input "
        "layer will mismatch the data at the first training step.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["two_step", "markovian", "history"], default="markovian",
        help="Aux propagator's mode (brief §5.2 addendum). 'history' (added 2026-08-29, "
        "user-directed) requires --aux-backbone vit; see --n-history and PropagatorConfig's "
        "docstring.",
    )
    parser.add_argument(
        "--n-history", type=int, default=3,
        help="--mode history only. Total states including current (default 3 = 2 past + "
        "current). Must be >= 2.",
    )
    parser.add_argument(
        "--w-var", type=float, default=None,
        help="Override Stage1TrainingConfig.w_var (default 0.01): weight on L_var, the "
        "per-dimension unit-variance term (part of decorr_var_loss). User-directed "
        "(2026-08-29): raise this to push the encoder's raw latent harder toward Var(z_i)=1 "
        "per dimension.",
    )
    parser.add_argument(
        "--w-decorr", type=float, default=None,
        help="Override Stage1TrainingConfig.w_decorr (default 0.01): weight on L_decorr, the "
        "off-diagonal decorrelation term (part of decorr_var_loss) -- penalizes ANY nonzero "
        "off-diagonal correlation, uniformly, regardless of position. Added 2026-09-01, "
        "user-directed: found to be in direct tension with --w-spatial (which specifically "
        "REWARDS off-diagonal correlation concentrated near the diagonal) -- see Phase 2 "
        "architecture doc's terms-audit section. Set to 0 to eliminate L_decorr entirely.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dataset", type=str, default="artifacts/datasets/stage1_trajectories_dtsnap1.h5"
    )
    parser.add_argument(
        "--dt-snap", type=float, default=1.0,
        help="Only used if --dataset doesn't exist yet (triggers fresh generation).",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.0,
        help="Ported improvement (2026-08-29, off by default): Gaussian noise std "
        "added to the encoder input on the reconstruction path only. See "
        "Stage1TrainingConfig's docstring.",
    )
    parser.add_argument(
        "--lambda-z", type=float, default=0.0,
        help="Ported improvement (off by default): weight on the optional banded "
        "latent-index-smoothness penalty. See RegConfig's docstring.",
    )
    parser.add_argument(
        "--w-spatial", type=float, default=None,
        help="Override Stage1TrainingConfig.w_spatial (default 0.0, off). Weight on "
        "spatial_coherence_loss (added 2026-08-31, user-directed): a differentiable, "
        "training-time version of D7 (same-time channel correlation) that rewards the "
        "CURRENT latent index order for having circular-band correlation structure -- "
        "'push neighboring latent variables to vary together.' READ "
        "ks_latent.training.losses.spatial_coherence_loss's docstring before setting "
        "this above 0: RegConfig.lambda_z, a related mechanism, is documented as "
        "collapsing the latent even with an explicit anti-collapse counter-term, and "
        "this carries a plausibly similar (though mechanistically different) risk. "
        "Test at a small weight with full Gate 3/4 monitoring.",
    )
    parser.add_argument(
        "--spatial-bandwidth", type=float, default=None,
        help="--w-spatial only. Override Stage1TrainingConfig.spatial_bandwidth "
        "(default 3.0): Gaussian circular-band kernel width, same convention as D3/D6/"
        "D7's bandedness() and --attn-window elsewhere.",
    )
    parser.add_argument(
        "--spatial-signed", action="store_true",
        help="--w-spatial only. Sets Stage1TrainingConfig.spatial_signed (default False, "
        "off, unchanged behavior). Added 2026-09-01, user-directed (backs the new D8 "
        "diagnostic): uses SIGNED correlation instead of |correlation| in "
        "spatial_coherence_loss, rewarding SAME-SIGN local coherence specifically -- "
        "cheap to test since this training keeps running otherwise unchanged. See "
        "spatial_coherence_loss's 'signed' docstring section and "
        "ks_latent.analysis.diagnostics.signed_bandedness (D8).",
    )
    parser.add_argument(
        "--w-var-floor", type=float, default=None,
        help="Override Stage1TrainingConfig.w_var_floor (default 0.0, off). Weight on "
        "variance_floor_loss (added 2026-08-31, user-directed -- Phase 2 architecture "
        "doc Section 35/38 option 3a): a VICReg-style per-channel std FLOOR, "
        "relu(gamma - std_i)^2 averaged over channels -- zero once every channel's std "
        "is >= gamma, unlike the existing w_var (default on, weight 0.01) which pulls "
        "EVERY channel toward variance exactly 1 and so also fights naturally-strong "
        "channels. Targets the specific observed failure: one latent channel's variance "
        "collapsing many orders of magnitude below the rest over extended joint "
        "training. See ks_latent.training.losses.variance_floor_loss's docstring.",
    )
    parser.add_argument(
        "--var-floor-gamma", type=float, default=None,
        help="--w-var-floor only. Override Stage1TrainingConfig.var_floor_gamma "
        "(default 0.1): the per-channel std floor target. Choose well below 1 (e.g. "
        "0.05-0.2) -- it's a floor against collapse, not a target to pull every "
        "channel toward.",
    )
    parser.add_argument(
        "--w-logdet", type=float, default=None,
        help="Override Stage1TrainingConfig.w_logdet (default 0.0, off). Weight on "
        "logdet_barrier_loss (added 2026-08-31, user-directed -- Phase 2 architecture "
        "doc Section 35/38 option 3b): -logdet(Cov(z)+eps*I)/d, pushing the WHOLE "
        "covariance eigenspectrum away from zero (not just the diagonal) -- more "
        "principled than --w-var-floor (also catches correlation-driven joint "
        "eigenvalue collapse, not just marginal per-channel variance collapse), but "
        "more novel/untested in this codebase. See "
        "ks_latent.training.losses.logdet_barrier_loss's docstring.",
    )
    parser.add_argument(
        "--w-var-end", type=float, default=None,
        help="Override Stage1TrainingConfig.w_var_end (default None, off). When set, "
        "w_var linearly decays from --w-var (epoch 0) to this value (final epoch) "
        "instead of staying fixed -- added 2026-09-02, user-directed: 'initially "
        "having regularizer parameters high and then decaying them overtime to "
        "increase prediction performance,' since heavier w_var/w_logdet/w_spatial "
        "improves latent covariance conditioning but measurably worsens downstream "
        "DA/rollout metrics. Pass a NONZERO floor, not 0 -- see "
        "Stage1TrainingConfig's docstring for why (correlation-driven collapse risk).",
    )
    parser.add_argument(
        "--w-logdet-end", type=float, default=None,
        help="--w-logdet only. Override Stage1TrainingConfig.w_logdet_end (default "
        "None, off). Same linear-decay mechanism as --w-var-end, applied to w_logdet.",
    )
    parser.add_argument(
        "--w-spatial-end", type=float, default=None,
        help="--w-spatial only. Override Stage1TrainingConfig.w_spatial_end (default "
        "None, off). Same linear-decay mechanism as --w-var-end, applied to w_spatial.",
    )
    parser.add_argument(
        "--logdet-eps", type=float, default=None,
        help="--w-logdet only. Override Stage1TrainingConfig.logdet_eps (default "
        "1e-3): covariance regularization so slogdet stays finite/well-defined.",
    )
    parser.add_argument(
        "--w-logdet-physical", type=float, default=None,
        help="encoder='spectral_field' only. Override Stage1TrainingConfig."
        "w_logdet_physical (default 0.0, off). User-directed (2026-09-09, after "
        "Sections 112/113's polynomial-field_kind propagator both collapsed): same "
        "logdet_barrier_loss mechanism as --w-logdet, but applied to "
        "decode_from_spectrum(z) -- the exact irfft reconstruction of the PHYSICAL "
        "field w -- instead of z (the rFFT coefficients) directly.",
    )
    parser.add_argument(
        "--logdet-physical-eps", type=float, default=None,
        help="--w-logdet-physical only. Override Stage1TrainingConfig."
        "logdet_physical_eps (default 1e-3). Same role as --logdet-eps.",
    )
    parser.add_argument(
        "--w-logdet-physical-rollout", type=float, default=None,
        help="encoder='spectral_field', --full-propagator only. Override "
        "Stage1TrainingConfig.w_logdet_physical_rollout (default 0.0, off). "
        "User-directed (2026-09-09, after Section 114 still showed the collapse "
        "signature with --w-logdet-physical alone): Stage-1 analogue of "
        "--w-lowpass/--w-lowpass-rollout's own pairing -- applies the SAME "
        "logdet_barrier_loss(decode_from_spectrum(...)) mechanism as "
        "--w-logdet-physical, but to the JOINT aux propagator's own short "
        "rolled-out z_pred instead of the encoder's real z.",
    )
    parser.add_argument(
        "--logdet-physical-rollout-eps", type=float, default=None,
        help="--w-logdet-physical-rollout only. Override Stage1TrainingConfig."
        "logdet_physical_rollout_eps (default 1e-3).",
    )
    parser.add_argument(
        "--w-var-physical", type=float, default=None,
        help="encoder='spectral_field' only. Override Stage1TrainingConfig."
        "w_var_physical (default 0.0, off). User-directed (2026-09-09): 'add a w_var "
        "term for the irfft of the latent state z' -- decorr_var_loss's per-channel "
        "variance-vs-1 term (--w-var's own mechanism), applied to "
        "decode_from_spectrum(z) (the encoder's real physical-space field) instead of "
        "z directly. Same 'physical' scoping convention as --w-logdet-physical, but "
        "the marginal-variance-floor mechanism instead of the full-covariance-rank one.",
    )
    parser.add_argument(
        "--physics-prior-correction-warmup-epochs", type=int, default=None,
        help="--spectral-physics-prior only. Override Stage1TrainingConfig."
        "physics_prior_correction_warmup_epochs (default 0, off). User-directed "
        "(2026-09-09): 'how do we preserve the chaotic structure and nudge it in the "
        "direction we want? could we progressively add systems we know are chaotic? "
        "... sort of a dynamic system gradient descent' -- linearly ramps the learned "
        "correction's own contribution (aux.body.correction_scale) from 0.0 (exactly "
        "the true KS equation) to 1.0 (full learned correction) over this many epochs, "
        "then holds at 1.0. A homotopy/continuation schedule: gives the encoder a long "
        "runway to converge under REAL chaotic dynamics before the learned correction "
        "gets room to find the 'damp everything' shortcut short-horizon MSE otherwise "
        "rewards whenever the encoder's z is still imprecise.",
    )
    parser.add_argument(
        "--w-pred-warmup-epochs", type=int, default=None,
        help="Override Stage1TrainingConfig.w_pred_warmup_epochs (default 0, off). "
        "User-directed (2026-09-09, after correction_scale's own warmup alone still "
        "collapsed): 'why don't we just have a schedule that slowly ramps up w_pred "
        "loss' -- the same homotopy idea one level up: linearly ramps the EFFECTIVE "
        "w_pred weight from 0.0 (epoch 0, pure reconstruction) to --w-pred's own value "
        "(this many epochs in, then held there), instead of (or alongside) delaying "
        "just the learned correction's growth -- gives the encoder pressure-free time "
        "to become a faithful reconstruction of the real field before ANY prediction "
        "loss (which rewards contraction whenever z is imprecise) touches it at all.",
    )
    parser.add_argument(
        "--w-temporal-floor-z", type=float, default=None,
        help="encoder='spectral_field' only. Override Stage1TrainingConfig."
        "w_temporal_floor_z (default 0.0, off). User-directed (2026-09-09, after "
        "Section 124's diagnostic showed the collapse is an ENCODER phenomenon -- a "
        "propagator that is literally the exact analytic KS equation, zero learned "
        "dynamics at all, still collapsed once the encoder trained under real w_pred "
        "pressure -- and w_logdet_physical, a batch-level covariance-rank check, did "
        "NOT fix it): 'build and test that, but design it for w space and for z space "
        "where w = irfft(z)' -- a one-sided floor (ks_latent.training.losses."
        "temporal_expansion_floor_loss) on how close together, in z-space, real states "
        "--temporal-floor-lag real steps apart are allowed to become, calibrated from "
        "the TRUE system's own real-data separation at that lag (computed once, "
        "independent of the current encoder).",
    )
    parser.add_argument(
        "--w-temporal-floor-w", type=float, default=None,
        help="encoder='spectral_field' only. Override Stage1TrainingConfig."
        "w_temporal_floor_w (default 0.0, off). Same mechanism as --w-temporal-floor-z, "
        "applied in PHYSICAL space (decode_from_spectrum(z)) instead of z directly -- "
        "the same z-vs-physical scoping distinction used elsewhere (--w-logdet vs "
        "--w-logdet-physical, --w-var vs --w-var-physical). Either or both of "
        "--w-temporal-floor-z/-w may be set.",
    )
    parser.add_argument(
        "--temporal-floor-lag", type=int, default=None,
        help="--w-temporal-floor-z/-w only. Override Stage1TrainingConfig."
        "temporal_floor_lag (default 1, consecutive real snapshots). How many real "
        "dt_snap steps apart the compared states are.",
    )
    parser.add_argument(
        "--w-local-expansion-floor", type=float, default=None,
        help="--aux-backbone spectral_pde, mode=markovian only. Override "
        "Stage1TrainingConfig.w_local_expansion_floor (default 0.0, off). User-directed "
        "(2026-09-09, after 'we could try a rollout variant, but it's the propagator "
        "already collapsing in stage 1? so it's sort of irrelevant?'): a DIFFERENTIABLE "
        "one-sided floor on the propagator's own per-sample step-Jacobian spectral norm "
        "(largest singular value), evaluated at real encoded states -- the same quantity "
        "Gate 4's D9 diagnostic measures post-hoc, turned into a training-time "
        "regularizer via torch.func.vmap(jacrev(...)). Rewards the encoder for placing "
        "real states where the dynamics locally EXPAND rather than contract. EXPENSIVE "
        "(~0.5-1s per call at K=24) -- applied only once per epoch, on a small "
        "subsample, not every batch.",
    )
    parser.add_argument(
        "--local-expansion-floor-value", type=float, default=None,
        help="--w-local-expansion-floor only. Override Stage1TrainingConfig."
        "local_expansion_floor_value (default 1.0, the absolute expand-vs-contract "
        "boundary -- no real-data calibration needed, unlike --temporal-floor-lag's "
        "own reference).",
    )
    parser.add_argument(
        "--local-expansion-floor-n-samples", type=int, default=None,
        help="--w-local-expansion-floor only. Override Stage1TrainingConfig."
        "local_expansion_floor_n_samples (default 32). How many real states (per "
        "epoch, from that epoch's first batch) to evaluate the Jacobian at.",
    )
    parser.add_argument(
        "--w-spectrum-shape", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage1TrainingConfig."
        "w_spectrum_shape (default 0.0, off). User-directed (2026-09-10, Section 131, "
        "after --w-local-expansion-floor was found to only constrain the TOP singular "
        "value, letting the other d-1 collapse toward zero -- 'can we use the "
        "regularizer to force some singular vectors to have expansive values around "
        "1.5 and others to have contracting values?'): floors the top "
        "--spectrum-shape-n-expand singular values toward --spectrum-shape-expand-target "
        "(expansive) AND floors the rest toward --spectrum-shape-contract-floor (still "
        "permits net contraction -- KS is dissipative -- but prevents runaway collapse "
        "of the tail). Same expensive/once-per-epoch convention as "
        "--w-local-expansion-floor.",
    )
    parser.add_argument(
        "--spectrum-shape-n-expand", type=int, default=None,
        help="--w-spectrum-shape only. Override Stage1TrainingConfig.spectrum_shape_n_expand "
        "(default 11, this project's own L=100 replication target for the number of "
        "positive Lyapunov exponents -- CLAUDE.md section 18).",
    )
    parser.add_argument(
        "--spectrum-shape-expand-target", type=float, default=None,
        help="--w-spectrum-shape only. Override Stage1TrainingConfig."
        "spectrum_shape_expand_target (default 1.5, matching Section 85's own median "
        "propagator-Jacobian norm benchmark for a genuinely chaotic propagator).",
    )
    parser.add_argument(
        "--spectrum-shape-contract-floor", type=float, default=None,
        help="--w-spectrum-shape only. Override Stage1TrainingConfig."
        "spectrum_shape_contract_floor (default 0.7).",
    )
    parser.add_argument(
        "--spectrum-shape-n-samples", type=int, default=None,
        help="--w-spectrum-shape only. Override Stage1TrainingConfig.spectrum_shape_n_samples "
        "(default 32).",
    )
    parser.add_argument(
        "--w-jacobian-bandedness", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage1TrainingConfig."
        "w_jacobian_bandedness (default 0.0, off). User-directed (2026-09-24, Section 214: "
        "'let's make D3 into a loss, since this seems like it's the most important statistic "
        "to improve our chances at being able to localize as in 4.3 in the literature review "
        "document'). D3's differentiable analogue (ks_latent.training.losses.propagator_"
        "jacobian_bandedness_loss) -- pushes the propagator's own step-Jacobian toward "
        "concentrating coupling mass near the diagonal in the CURRENT latent index order "
        "(no seriation search -- meaningful for a local_field encoder, whose index IS "
        "physical site position by construction). Same expensive/once-per-epoch convention "
        "as --w-spectrum-shape.",
    )
    parser.add_argument(
        "--jacobian-bandedness-bandwidth", type=float, default=None,
        help="--w-jacobian-bandedness only. Override Stage1TrainingConfig."
        "jacobian_bandedness_bandwidth (default 3.0, in latent-index-ring units -- same "
        "convention as --spatial-bandwidth).",
    )
    parser.add_argument(
        "--jacobian-bandedness-n-samples", type=int, default=None,
        help="--w-jacobian-bandedness only. Override Stage1TrainingConfig."
        "jacobian_bandedness_n_samples (default 32).",
    )
    parser.add_argument(
        "--w-jacobian-diagonal-bound", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage1TrainingConfig."
        "w_jacobian_diagonal_bound (default 0.0, off). User-directed (2026-09-24, Section 216: "
        "'it looks like stage 2 is having trouble converging. If we combined the D3 regularizer "
        "with a term that bounded the magnitude of the diagonal of the jacobian, maybe that "
        "would help'). Companion to --w-jacobian-bandedness (ks_latent.training.losses."
        "propagator_jacobian_diagonal_bound_loss): caps each site's own self-coupling magnitude "
        "directly, orthogonal to bandedness -- Section 215 found bandedness ALONE made "
        "standalone divergence worse, since it constrains WHERE coupling concentrates, not HOW "
        "LARGE the surviving entries are. Same expensive/once-per-epoch convention.",
    )
    parser.add_argument(
        "--jacobian-diagonal-bound-ceiling", type=float, default=None,
        help="--w-jacobian-diagonal-bound only. Override Stage1TrainingConfig."
        "jacobian_diagonal_bound_ceiling (default 1.5, reusing --spectrum-shape-expand-target's "
        "own calibration point -- not independently tuned for the diagonal).",
    )
    parser.add_argument(
        "--jacobian-diagonal-bound-n-samples", type=int, default=None,
        help="--w-jacobian-diagonal-bound only. Override Stage1TrainingConfig."
        "jacobian_diagonal_bound_n_samples (default 32).",
    )
    parser.add_argument(
        "--w-spectrum-shape-graded", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage1TrainingConfig."
        "w_spectrum_shape_graded (default 0.0, off). User-directed (2026-09-10, Section 134, "
        "after a robust 200-sample per-rank measurement of Section 85's own real propagator "
        "showed the spectrum is a smooth graded decline, not two flat groups): a per-rank "
        "floor generalizing --w-spectrum-shape's two-group one. Requires "
        "--spectrum-shape-graded-reference-path. May be combined with --w-spectrum-shape.",
    )
    parser.add_argument(
        "--spectrum-shape-graded-reference-path", type=str, default=None,
        help="--w-spectrum-shape-graded only. Path to a .npy file of shape (d_latent,), "
        "descending -- the per-rank target spectrum, typically produced by "
        "scripts/compute_reference_spectrum.py from a trusted checkpoint's own real spectrum.",
    )
    parser.add_argument(
        "--spectrum-shape-graded-n-samples", type=int, default=None,
        help="--w-spectrum-shape-graded only. Override Stage1TrainingConfig."
        "spectrum_shape_graded_n_samples (default 32).",
    )
    parser.add_argument(
        "--spectrum-shape-two-sided", action="store_true",
        help="--w-spectrum-shape only. Section 186, user-directed (2026-09-18): "
        "switch from a one-sided floor (relu(target-sv)^2, never penalizes exceeding "
        "the target) to a plain squared-error match ((sv-target)^2) in both the "
        "expansive and contracting groups. Needed because Section 185's own trained "
        "checkpoint had top singular values 3-4x ABOVE expand_target, where the "
        "one-sided floor contributes exactly zero gradient.",
    )
    parser.add_argument(
        "--w-spectrum-shape-multistep", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage1TrainingConfig."
        "w_spectrum_shape_multistep (default 0.0, off). Section 186, user-directed: "
        "\"is there a way to force the pde's singular vectors ... to go from "
        "expansive to contractive and back again? ... if the expansive singular "
        "vectors all feed into other expansive singular vectors ... we will see "
        "runaway growth.\" Constrains the COMPOSED k-step Jacobian's singular-value "
        "spectrum (toward expand_target**k / contract_floor**k, always two-sided) "
        "instead of the one-step Jacobian's -- catches persistent expansive-subspace "
        "self-feeding that a one-step spectrum check cannot see. May be combined "
        "with --w-spectrum-shape/--w-spectrum-shape-graded.",
    )
    parser.add_argument(
        "--spectrum-shape-multistep-k", type=int, default=None,
        help="--w-spectrum-shape-multistep only. Override Stage1TrainingConfig."
        "spectrum_shape_multistep_k (default 10). Number of steps composed before "
        "the single Jacobian call -- recommend matching --k-pred-max.",
    )
    parser.add_argument(
        "--spectrum-shape-multistep-n-samples", type=int, default=None,
        help="--w-spectrum-shape-multistep only. Override Stage1TrainingConfig."
        "spectrum_shape_multistep_n_samples (default 16, smaller than the one-step "
        "version's default since composing k steps before differentiating is more "
        "expensive per sample).",
    )
    parser.add_argument(
        "--w-smooth", type=float, default=None,
        help="Override Stage1TrainingConfig.w_smooth (default 0.0, off). Weight on "
        "temporal_smoothness_loss (added 2026-09-04, user-directed: the "
        "differentiable training-time version of "
        "scripts/analyze_latent_smoothness.py's real-trajectory step-size/curvature "
        "diagnostic): mean(||z_{t+1}-z_t||^2)/d + smooth_curvature_weight * "
        "mean(||z_{t+1}-2z_t+z_{t-1}||^2)/d, computed on the SAME z_win already used "
        "for L_pred (no new data loading). Unlike --w-spatial (rewards a correlation "
        "structure, cannot trivially collapse), this term's global optimum in "
        "isolation genuinely IS latent collapse -- see "
        "ks_latent.training.losses.temporal_smoothness_loss's docstring before "
        "setting this above 0; test at a small weight with full Gate 3/4 monitoring.",
    )
    parser.add_argument(
        "--smooth-curvature-weight", type=float, default=None,
        help="--w-smooth only. Override Stage1TrainingConfig.smooth_curvature_weight "
        "(default 1.0): relative weight of the curvature (second-difference) term "
        "vs. the plain step-size term within temporal_smoothness_loss.",
    )
    parser.add_argument(
        "--w-lowpass", type=float, default=None,
        help="--encoder spectral_field only. Override Stage1TrainingConfig.w_lowpass "
        "(default 0.0, off). Weight on low_pass_spectral_loss (added 2026-09-06, see "
        "docs/sine_transform_pde_plan.md, user-directed: 'add a regularizer that acts "
        "as a low pass filter ... penalizes the higher frequency terms of z "
        "proportional to their frequency'): sum_k k^lowpass_power * |z_k|^2, k the "
        "physical angular wavenumber of mode k. See "
        "ks_latent.training.losses.low_pass_spectral_loss's docstring.",
    )
    parser.add_argument(
        "--lowpass-power", type=float, default=None,
        help="--w-lowpass only. Override Stage1TrainingConfig.lowpass_power (default "
        "1.0, the literal 'proportional to frequency' request). 2.0 is the classical "
        "H^1 Sobolev seminorm instead.",
    )
    parser.add_argument(
        "--w-lowpass-rollout", type=float, default=None,
        help="--encoder spectral_field only (any --aux-backbone; --full-propagator not "
        "required -- z_pred is always computed, just at whatever aux propagator size is "
        "active). Override Stage1TrainingConfig.w_lowpass_rollout (default 0.0, off). "
        "Stage-1 analogue of --w-lowpass (added 2026-09-08, see "
        "docs/sine_transform_pde_plan.md, user-directed after visualizing Section 104's "
        "D_KY=22 result), but applied to the joint-training AUX PROPAGATOR's own short "
        "rolled-out z_pred instead of the encoder's direct z -- lets the encoder and "
        "propagator co-adapt to the high-mode-energy constraint from the start of Phase 1, "
        "not just Phase 2.",
    )
    parser.add_argument(
        "--lowpass-rollout-power", type=float, default=None,
        help="--w-lowpass-rollout only. Override Stage1TrainingConfig.lowpass_rollout_power "
        "(default 1.0). Same convention as --lowpass-power.",
    )
    parser.add_argument(
        "--w-z-lowpass", type=float, default=None,
        help="Any --encoder (generalizes --w-lowpass, which requires --encoder "
        "spectral_field, to any raw z via its own self-FFT -- "
        "ks_latent.training.losses.latent_self_spectrum_lowpass_loss). Override "
        "Stage1TrainingConfig.w_z_lowpass (default 0.0, off). Added 2026-09-11, "
        "user-directed: 'I would like to penalize higher frequencies in the fourier "
        "transform of the latent space. This may help make the pde dynamics more easily "
        "fit and smooth.' Applied to the encoder's own z (reconstruction path).",
    )
    parser.add_argument(
        "--w-z-lowpass-rollout", type=float, default=None,
        help="Any --encoder. Override Stage1TrainingConfig.w_z_lowpass_rollout (default "
        "0.0, off). Same as --w-z-lowpass but applied to the joint-training AUX "
        "PROPAGATOR's own rolled-out z_pred, mirroring --w-lowpass-rollout's convention.",
    )
    parser.add_argument(
        "--z-lowpass-power", type=float, default=None,
        help="--w-z-lowpass only. Override Stage1TrainingConfig.z_lowpass_power (default "
        "1.0). Same convention as --lowpass-power.",
    )
    parser.add_argument(
        "--z-lowpass-rollout-power", type=float, default=None,
        help="--w-z-lowpass-rollout only. Override Stage1TrainingConfig."
        "z_lowpass_rollout_power (default 1.0).",
    )
    parser.add_argument(
        "--z-lowpass-K", type=int, default=None,
        help="--w-z-lowpass/--w-z-lowpass-rollout only. Override "
        "Stage1TrainingConfig.z_lowpass_K (default None = d_latent//2+1, i.e. no "
        "truncation of z's own self-spectrum -- matches a spectral_pde_raw pde_head's "
        "own default spectral_K).",
    )
    parser.add_argument(
        "--z-lowpass-L", type=float, default=None,
        help="--w-z-lowpass/--w-z-lowpass-rollout only. Override "
        "Stage1TrainingConfig.z_lowpass_L (default None = d_latent, matching a "
        "spectral_pde_raw pde_head's own default spectral_L).",
    )
    parser.add_argument(
        "--w-channel-mean", type=float, default=None,
        help="--encoder local_field only. Override Stage1TrainingConfig.w_channel_mean "
        "(default 0.0, off). Added 2026-09-11, user-directed: 'implement the root fix' -- "
        "penalizes each free residual channel (1..local_channels-1) for having a nonzero "
        "mean ACROSS SITES. See ks_latent.training.losses.local_field_channel_mean_loss's "
        "docstring for the full motivation (a measured period-local_channels flattening "
        "artifact that dominated a trained pde_head's own self-spectrum) and why this is "
        "a soft training-time penalty rather than an architectural fix.",
    )
    parser.add_argument(
        "--w-shape-floor", type=float, default=None,
        help="--encoder spectral_field only. Override Stage1TrainingConfig.w_shape_floor "
        "(default 0.0, off). Weight on spectral_shape_floor_loss (added 2026-09-07, see "
        "docs/sine_transform_pde_plan.md, user-directed: 'I really just want to come up "
        "with a regularizer to prevent the latent state from collapsing'): a ONE-SIDED "
        "floor precomputed from the REAL training data's own low-K-mode energy "
        "proportions -- penalizes z's own per-mode share falling BELOW that physically-"
        "grounded reference, never for exceeding it (so the encoder can still legitimately "
        "put extra 'compensation' energy into low modes for information folded in from "
        "u's discarded higher modes). See "
        "ks_latent.training.losses.spectral_shape_floor_loss's docstring.",
    )
    parser.add_argument(
        "--spectral-K", type=int, default=None,
        help="--encoder spectral_field only (REQUIRED for that encoder). "
        "SpectralFieldAutoencoderConfig.K: number of kept low rFFT modes -- z has "
        "dimension 2*K (K real + K imaginary parts). Must be in [1, N_w//2+1], "
        "N_w = --d-latent (reused as the intermediate spatial field w's physical "
        "resolution -- see SpectralFieldAutoencoderConfig's docstring for why).",
    )
    parser.add_argument(
        "--spectral-L", type=float, default=100.0,
        help="--encoder spectral_field only. SpectralFieldAutoencoderConfig.L: the "
        "physical domain length wavenumbers are computed against -- MUST match the "
        "training dataset's own KSConfig.L, or every synthesized derivative "
        "downstream (this backbone's propagator) will be silently scaled wrong.",
    )
    parser.add_argument(
        "--lambda-decorr-band", type=float, default=0.0,
        help="Ported improvement (off by default): weight on the optional "
        "off-band decorrelation penalty paired with --lambda-z.",
    )
    parser.add_argument(
        "--reg-start-epoch", type=int, default=0,
        help="Override RegConfig.start_epoch (default 0, active from the start). "
        "When > 0, --lambda-z/--lambda-decorr-band are skipped entirely for "
        "epochs before this one, then applied normally (uniform weight) from "
        "here on. Added 2026-09-02, user-directed.",
    )
    parser.add_argument(
        "--encoder", choices=[
            "transformer", "mlp", "vit", "masked_mlp", "fourier_mlp", "vit_fourier_hybrid",
            "spectral_field", "local_field",
        ],
        default="transformer",
        help="Autoencoder architecture. 'transformer' (default) = the brief's "
        "patched-transformer (no positional encoding). 'mlp' = the ported "
        "plain-MLP Track A. 'vit' = the ported ViT Track B (has an explicit "
        "circular positional encoding the patched-transformer lacks). "
        "'masked_mlp' (2026-08-31, user-directed) = 'mlp' with every Linear "
        "replaced by a circular-band-masked local version -- see "
        "MaskedMLPAutoencoderConfig's docstring and --ae-mask-window. "
        "'fourier_mlp' (2026-09-03, user-directed) = raw values PLUS a fixed "
        "Fourier featurization (real/imag rfft) fed through a plain residual "
        "MLP -- see FourierMLPAutoencoderConfig's docstring and "
        "--fourier-mlp-hidden/--fourier-mlp-blocks/--fourier-mlp-enc-fno-modes/"
        "--fourier-mlp-dec-fno-modes. 'vit_fourier_hybrid' (2026-09-04, "
        "user-directed) = a ViT (Section 52's structure, function-space) "
        "SUMMED with a ConservedFourierMLP (pure frequency-space, an L1- "
        "'conservation law' at every layer instead of a Lipschitz/spectral-"
        "norm bound) -- see ViTFourierHybridAutoencoderConfig's docstring "
        "and --vit-fourier-fourier-hidden/--vit-fourier-fourier-blocks. Uses "
        "--d-model/--vit-n-blocks/--attn-window/--pos-encoding/--pool/"
        "--token-window/etc for its ViT sub-config, same as --encoder vit.",
    )
    parser.add_argument(
        "--vit-fourier-fourier-hidden", type=int, default=None,
        help="--encoder vit_fourier_hybrid only. Override "
        "ViTFourierHybridAutoencoderConfig.fourier_hidden (default 128).",
    )
    parser.add_argument(
        "--vit-fourier-fourier-blocks", type=int, default=None,
        help="--encoder vit_fourier_hybrid only. Override "
        "ViTFourierHybridAutoencoderConfig.fourier_blocks (default 3).",
    )
    parser.add_argument(
        "--vit-fourier-enc-out-modes", type=int, default=None,
        help="--encoder vit_fourier_hybrid, --vit-fourier-kind ifft only. Override "
        "ViTFourierHybridAutoencoderConfig.enc_out_modes (default None = full spectrum, "
        "d_latent//2+1). Truncates how many frequency coefficients the ENCODER's Fourier "
        "branch (FourierIFFTBody) is allowed to predict before its irfft -- an exact "
        "low-pass truncation of that branch's own output. Added 2026-09-05, user-directed: "
        "'can we explicitly penalize higher frequency terms in the irfft matrix? or even "
        "truncate these completely? that way the embedding will be inherently smoother'. "
        "CAVEAT: the summed vit branch is unconstrained, so this alone does not guarantee "
        "a band-limited z -- see ViTFourierHybridAutoencoderConfig's docstring.",
    )
    parser.add_argument(
        "--vit-fourier-dec-out-modes", type=int, default=None,
        help="--encoder vit_fourier_hybrid, --vit-fourier-kind ifft only. Same as "
        "--vit-fourier-enc-out-modes, applied to the DECODER's Fourier branch instead "
        "(default None = full spectrum, NX//2+1).",
    )
    parser.add_argument(
        "--vit-fourier-kind", choices=["conserved", "ifft"], default=None,
        help="--encoder vit_fourier_hybrid only. Override "
        "ViTFourierHybridAutoencoderConfig.fourier_kind (default 'conserved'). "
        "'ifft' (2026-09-04, Section 81, user-directed): use "
        "ks_latent.models.propagator.FourierIFFTBody (Section 75's own "
        "Fourier-path mechanism) instead of ConservedFourierMLP, with no "
        "raw-value/masked path at all -- 'only using the frequency "
        "components'.",
    )
    parser.add_argument(
        "--fourier-mlp-hidden", type=int, default=None,
        help="--encoder fourier_mlp only. Override FourierMLPAutoencoderConfig.hidden "
        "(default 128). For --aux-backbone fourier_mlp's OWN size, use the existing "
        "generic --aux-hidden/--aux-blocks flags instead (already wired to any "
        "aux backbone, not fourier_mlp-specific).",
    )
    parser.add_argument(
        "--fourier-mlp-blocks", type=int, default=None,
        help="--encoder fourier_mlp only. Override FourierMLPAutoencoderConfig.n_blocks "
        "(default 3).",
    )
    parser.add_argument(
        "--fourier-mlp-enc-fno-modes", type=int, default=None,
        help="--encoder fourier_mlp only. Override FourierMLPAutoencoderConfig.enc_fno_modes "
        "(default None = full spectrum, NX//2+1).",
    )
    parser.add_argument(
        "--fourier-mlp-dec-fno-modes", type=int, default=None,
        help="--encoder fourier_mlp only. Override FourierMLPAutoencoderConfig.dec_fno_modes "
        "(default None = full spectrum, d_latent//2+1).",
    )
    parser.add_argument(
        "--ae-mask-window", type=int, default=4,
        help="--encoder masked_mlp only. MaskedMLPAutoencoderConfig.mask_window, "
        "in d_latent-ring units (same units as --attn-window elsewhere). "
        "Pass a Python None via omitting this flag's use entirely is not "
        "supported from the CLI; use a large value (e.g. >= d_latent) for "
        "an effectively-dense control.",
    )
    parser.add_argument(
        "--local-field-n-sites", type=int, default=32,
        help="--encoder local_field only. LocalFieldAutoencoderConfig.n_sites "
        "(default 32, CLAUDE.md section 12.2.1's own starting point). Must divide NX.",
    )
    parser.add_argument(
        "--local-field-channels", type=int, default=3,
        help="--encoder local_field only. LocalFieldAutoencoderConfig.local_channels "
        "(default 3). Channel 0 is always the gauge-anchored physical average, never "
        "learned; the rest are free learned residuals.",
    )
    parser.add_argument(
        "--local-field-mix-radius", type=int, default=2,
        help="--encoder local_field only. LocalFieldAutoencoderConfig.site_mix_radius "
        "(default 2) -- circular Conv1d kernel radius over SITES (not raw grid points).",
    )
    parser.add_argument(
        "--local-field-n-mix-layers", type=int, default=3,
        help="--encoder local_field only. LocalFieldAutoencoderConfig.n_site_mix_layers "
        "(default 3).",
    )
    parser.add_argument(
        "--local-field-hidden", type=int, default=32,
        help="--encoder local_field only. LocalFieldAutoencoderConfig.hidden "
        "(default 32) -- conv channel width in the site-mixing stack.",
    )
    parser.add_argument(
        "--aux-backbone",
        choices=[
            "mlp", "transformer", "vit", "fno_vit", "fourier_mlp", "local_mlp",
            "masked_mlp", "masked_mlp_wide", "masked_mlp_expand", "spectral_pde",
            "spectral_pde_raw",
        ],
        default="mlp",
        help="Backbone of the auxiliary propagator used only for Stage-1's "
        "L_pred term (brief §5.2 addendum; PropagatorConfig's docstring). "
        "Default 'mlp' matches the brief's original design. Independent of "
        "--encoder: e.g. --encoder vit --aux-backbone vit tests the ViT "
        "block in both roles at once; --encoder vit --aux-backbone mlp "
        "isolates the encoder's effect with the original aux propagator. "
        "'spectral_pde' (added 2026-09-07, user-directed: 'I want the encoder "
        "and decoder pair to be trained specifically to encode data that can "
        "be transformed accurately by the rk4/etdrk4/euler method' -- unlike "
        "'node'/'cnn', which remain Stage-2-only): requires --encoder "
        "spectral_field, so the AE co-adapts with the REAL propagator "
        "architecture from Stage 1 onward instead of a throwaway generic "
        "'mlp' aux -- see --spectral-integrator/--spectral-max-order and "
        "AuxPropagatorConfig's docstring.",
    )
    parser.add_argument(
        "--spectral-max-order", type=int, default=4,
        help="--aux-backbone spectral_pde only. AuxPropagatorConfig.spectral_max_order "
        "(default 4). See PropagatorConfig's 'spectral_pde' docstring.",
    )
    parser.add_argument(
        "--spectral-field-kind", choices=["mlp", "polynomial", "chebyshev", "forced_burgers"], default="mlp",
        help="--aux-backbone spectral_pde/spectral_pde_raw only. AuxPropagatorConfig."
        "spectral_field_kind (default 'mlp'). 'polynomial' (added 2026-09-09, "
        "user-directed: 'expand the pde as a polynomial (degree 1 or 2) in all the "
        "derivative terms, then directly learn the coefficients ... more interpretable, "
        "and potentially more stable') replaces the pointwise MLP with a single shared "
        "linear layer over the degree-<=--spectral-poly-degree monomial library built "
        "from the derivative stack -- a dense SINDy-style linear regression whose "
        "learned coefficients are directly readable after training. 'chebyshev' (added "
        "as a --spectral-field-kind choice 2026-09-18, Section 187 -- the underlying "
        "_SpectralPDEDeltaBody mechanism already existed and was exercised via "
        "--pde-field-kind chebyshev for the pde_head DISTILLATION path since Section "
        "152, just never exposed for the LIVE propagator driving training; identical "
        "library/term semantics to 'polynomial', each term built from a product of "
        "Chebyshev T_k's instead of monomials -- see --spectral-poly-* flags, all of "
        "which apply identically to this choice). 'forced_burgers' "
        "(added 2026-09-14, Section 175, user-directed: a revision of the Section 174 "
        "proposal) hardcodes w_t = -w*w_x + nu*w_xx + g_theta(w, w_x, ...) -- advection "
        "fixed at -1.0, nu LEARNABLE but architecturally forced positive (softplus, "
        "UNCONDITIONALLY stabilizing -- no 4th-order term needed at all, dramatically "
        "cheaper to integrate than --spectral-poly-stable-w-xx-w-xxxx's stiff w_xxxx), "
        "forcing g_theta a small shared pointwise MLP over the same derivative stack "
        "(same architecture as field_kind='mlp', just added as a residual on top of the "
        "two hardcoded physical terms instead of being the whole RHS). See "
        "PropagatorConfig.spectral_field_kind's docstring for the full motivation.",
    )
    parser.add_argument(
        "--spectral-burgers-nu-init", type=float, default=1.0,
        help="--spectral-field-kind forced_burgers only. AuxPropagatorConfig."
        "spectral_burgers_nu_init (default 1.0, matching true KS/Burgers' own unit-scale "
        "diffusion coefficient) -- initial value for nu (before the softplus "
        "reparametrization, i.e. softplus(raw_init) == this value exactly at construction).",
    )
    parser.add_argument(
        "--spectral-burgers-beta-max", type=float, default=1.0,
        help="--spectral-field-kind forced_burgers only. AuxPropagatorConfig."
        "spectral_burgers_beta_max (default 1.0). Added 2026-09-14, same section, user-"
        "directed: 'why don't we give -u*u_x a coefficient term too, if that's the cause "
        "of the instability' -- the advection term has its own CFL-type stability "
        "constraint, independent of nu. beta = beta_max*tanh(raw_beta), architecturally "
        "bounded to [-beta_max, +beta_max] regardless of training, raw_beta starts at "
        "exactly 0 (pure diffusion+forcing at init, advection strength discovered "
        "gradually via --stable-linear-lr-factor's low-LR group).",
    )
    parser.add_argument(
        "--spectral-burgers-nonlinear-nu", action="store_true",
        help="--spectral-field-kind forced_burgers only. Sets AuxPropagatorConfig."
        "spectral_burgers_nonlinear_nu (default False). Sketched 2026-09-14, user-directed: "
        "'should we consider creating a pde that isn't a polynomial? it could be a more "
        "generic nonlinear function of the derivatives' -- generalizes the constant nu to "
        "a state-dependent nu(w, w_x, ...) via its own small MLP trunk, softplus'd so it "
        "stays positive everywhere regardless of training (same guarantee as the constant "
        "case, not a bare unconstrained field_kind='mlp' correction). Zero-init gives "
        "nu(x)==--spectral-burgers-nu-init exactly at every point at construction. Not yet "
        "validated via a real training run -- implemented/unit-verified only as of "
        "2026-09-14; verify with a dry run before trusting it at scale.",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-instability", action="store_true",
        help="--spectral-field-kind forced_burgers only. Sets AuxPropagatorConfig."
        "spectral_burgers_kernel_instability (default False). Added 2026-09-14, Section 177, "
        "user-directed after reading Sakaguchi 2000 ('A Simple Model for Spatio-Temporal "
        "Chaos in an Unstable Burgers Equation') -- REPLACES the diffusion term with a "
        "diagonal Fourier multiplier Lhat(k) = -(A*exp(-k^2/width^2) + nu) * k^2. A "
        "(learnable, bounded to +-spectral-burgers-kernel-A-max via tanh, starts at 0) is "
        "the missing destabilizing ingredient Sections 175/176 lacked entirely (their fitted "
        "nu/beta barely moved, pure damping, hence their collapse); nu still guarantees "
        "UNCONDITIONAL high-k damping regardless of A. Bounded-magnitude (Gaussian envelope) "
        "unlike a k^4 term, so should stay numerically cheap (no ode_substeps~64 needed). "
        "Mutually exclusive with --spectral-burgers-nonlinear-nu.",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-A-max", type=float, default=1.0,
        help="--spectral-burgers-kernel-instability only. AuxPropagatorConfig."
        "spectral_burgers_kernel_A_max (default 1.0) -- architectural bound on the kernel "
        "amplitude A via tanh.",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-width-init", type=float, default=1.0,
        help="--spectral-burgers-kernel-instability only. AuxPropagatorConfig."
        "spectral_burgers_kernel_width_init (default 1.0) -- initial value for the Gaussian "
        "envelope's decay scale (before the softplus reparametrization).",
    )
    parser.add_argument(
        "--spectral-burgers-forcing-max", type=float, default=1.0,
        help="--spectral-field-kind forced_burgers only. AuxPropagatorConfig."
        "spectral_burgers_forcing_max (default 1.0). Added 2026-09-14, Section 178, "
        "user-directed after Sections 176/177 both diverged via an unconstrained forcing "
        "MLP -- bounds the forcing's raw output via tanh, then subtracts the CAPPED "
        "result's own spatial mean, giving an EXACTLY zero-mean forcing (architecturally "
        "guaranteed, matching true KS's own mass conservation) bounded by roughly 2x this "
        "value. See ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for the "
        "full derivation.",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-A-fixed", type=float, default=None,
        help="--spectral-burgers-kernel-instability only. Sets AuxPropagatorConfig."
        "spectral_burgers_kernel_A_fixed (default None, learnable). Added 2026-09-16, "
        "Section 180, user-directed ablation: 'is there any way to get kernel A to "
        "activate in 178 and 179?' -- 178/179 both measured A drifting toward MORE "
        "damping under gradient descent, never activating. FIXES A at this exact value "
        "(not learnable, no tanh cap) instead of leaving it learnable -- architecturally "
        "forces a genuine low-k unstable band. e.g. -2.0 (with default nu~1.0, width~1.0) "
        "gives Lhat(k)>0 for roughly the lowest third of kept modes. See "
        "ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for the derivation.",
    )
    parser.add_argument(
        "--spectral-burgers-beta-fixed", type=float, default=None,
        help="--spectral-field-kind forced_burgers only. Sets AuxPropagatorConfig."
        "spectral_burgers_beta_fixed (default None, learnable). Added 2026-09-16, "
        "Section 181, user-directed follow-up to --spectral-burgers-kernel-A-fixed: "
        "'if we remove delta_cap what would happen?' -> real KS-class saturation comes "
        "from the advection term cascading energy to damped high-k modes, but beta stayed "
        "near 0 in every prior section, so delta_cap alone was bounding growth. FIXES beta "
        "at this exact value (not learnable, no tanh cap) -- e.g. -1.0 matches true KS's "
        "own advection coefficient. See ks_latent.models.propagator._SpectralPDEDeltaBody's "
        "docstring for the derivation.",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-mu-init", type=float, default=None,
        help="--spectral-burgers-kernel-instability only. Sets AuxPropagatorConfig."
        "spectral_burgers_kernel_mu_init (default None, term absent). Added 2026-09-16, "
        "Section 182, user-directed: 'add a hyperviscosity term' -- extends Lhat(k) with "
        "a genuine -mu*k^4 term, mu=softplus(raw) always > 0, matching true KS's own "
        "-w_xxxx. e.g. 0.01 respects this backbone's ode_substeps=1 stability budget "
        "(mu*k_max^4 <~ 2 required; k_max=pi at K=49/L=96 gives k_max^4~97.4, so mu<~0.0205). "
        "See ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for the derivation.",
    )
    parser.add_argument(
        "--spectral-burgers-no-forcing", action="store_true",
        help="--spectral-field-kind forced_burgers only. Sets AuxPropagatorConfig."
        "spectral_burgers_no_forcing (default False). Added 2026-09-16, Section 182, "
        "user-directed: 'get rid of the forcing mlp term' -- removes the forcing MLP "
        "entirely (no trunk built at all, field() returns zero for it), leaving only the "
        "hardcoded/constrained physical terms, matching Sakaguchi's own equation "
        "structurally (no separate learned correction).",
    )
    parser.add_argument(
        "--spectral-burgers-kernel-A-init", type=float, default=0.0,
        help="--spectral-burgers-kernel-instability only, and only when "
        "--spectral-burgers-kernel-A-fixed is NOT set (learnable case). Sets "
        "AuxPropagatorConfig.spectral_burgers_kernel_A_init (default 0.0). Added "
        "2026-09-16, Section 183, user-directed: 'let's initialize with the same "
        "parameters but let A and beta be trainable' -- starts the LEARNABLE A at this "
        "value instead of 0 (e.g. -2.0, matching Section 180-182's own fixed value). "
        "Requires |value| < --spectral-burgers-kernel-A-max strictly.",
    )
    parser.add_argument(
        "--spectral-burgers-beta-init", type=float, default=0.0,
        help="Only when --spectral-burgers-beta-fixed is NOT set (learnable case). Sets "
        "AuxPropagatorConfig.spectral_burgers_beta_init (default 0.0). Same mechanism as "
        "--spectral-burgers-kernel-A-init, for beta (e.g. -1.0, matching Section 181-182's "
        "own fixed value). Requires |value| < --spectral-burgers-beta-max strictly.",
    )
    parser.add_argument(
        "--w-kernel-unstable-floor", type=float, default=None,
        help="--spectral-burgers-kernel-instability only. Override "
        "Stage1TrainingConfig.w_kernel_unstable_floor (default 0.0, off). Added "
        "2026-09-16, Section 183, user-directed: 'we should also increase the "
        "regularizer that tries to keep at least 13 unstable modes in the pde, or if "
        "that doesn't exist, implement it' -- hinge penalty (ks_latent.training.losses."
        "kernel_unstable_floor_loss) keeping the lowest --kernel-unstable-target-modes "
        "Fourier modes above --kernel-unstable-margin in Lhat(k), i.e. genuinely "
        "unstable/growing. A pure function of the model's own kernel parameters, no "
        "forward pass over data needed.",
    )
    parser.add_argument(
        "--kernel-unstable-target-modes", type=int, default=13,
        help="--w-kernel-unstable-floor only. Stage1TrainingConfig."
        "kernel_unstable_target_modes (default 13).",
    )
    parser.add_argument(
        "--kernel-unstable-margin", type=float, default=0.0,
        help="--w-kernel-unstable-floor only. Stage1TrainingConfig."
        "kernel_unstable_margin (default 0.0, i.e. Lhat(k) merely needs to be positive, "
        "not exceed some larger safety margin).",
    )
    parser.add_argument(
        "--spectral-poly-degree", type=int, choices=[1, 2, 3], default=2,
        help="--spectral-field-kind polynomial only. AuxPropagatorConfig."
        "spectral_poly_degree (default 2 -- sufficient to represent KS's own true "
        "nonlinearity -w*w_x exactly; 1 gives a purely linear PDE, no nonlinear terms "
        "at all; 3 adds cubic cross terms, added 2026-09-09, user-directed: 'higher "
        "degree polynomial for the pde').",
    )
    parser.add_argument(
        "--spectral-poly-max-term-order", type=int, default=None,
        help="--spectral-field-kind polynomial only. AuxPropagatorConfig."
        "spectral_poly_max_term_order (default None, unrestricted). Excludes any monomial "
        "whose derivative orders SUM to >= this value from the library entirely (added "
        "2026-09-09, user-directed: 'really only let the combined degree of the terms be "
        "less than 5 (so w_xxx * w_xxx or w_xxx*w_xxxx would have 0 coefficients since "
        "they have combined degree 6, 7 respectively)'). KS's own true equation has "
        "combined order <=4 on every term, so e.g. 5 keeps all of it while excluding "
        "physically-unmotivated high-combined-order cross terms.",
    )
    parser.add_argument(
        "--spectral-poly-norm-power", type=float, default=1.0,
        help="--spectral-field-kind polynomial only. AuxPropagatorConfig."
        "spectral_poly_norm_power (default 1.0, the original NaN-blowup-preventing "
        "strength). User-directed (2026-09-09): 'less normalization for the polynomial' "
        "-- a value <1.0 weakens the per-order rescaling (order-n's channel divided by "
        "char_k**(n*power) instead of char_k**n), 0.0 disables it entirely. Re-introduces "
        "real blowup risk at low values -- verify via a direct scale test, not just a "
        "short smoke run.",
    )
    parser.add_argument(
        "--spectral-poly-stable-leading", action="store_true",
        help="--spectral-field-kind polynomial only. Sets AuxPropagatorConfig."
        "spectral_poly_stable_leading (default False). User-directed (2026-09-09): 'is "
        "there a way to regularize or bound the eigenvalues of the differential operator "
        "induced by the pde?' -- forces the coefficient on the highest kept even-order "
        "linear derivative term to be <= 0, guaranteeing the induced linear operator's "
        "eigenvalue real part goes to -inf as wavenumber -> inf (bounded), without "
        "suppressing instability at lower wavenumbers. See "
        "ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for the full "
        "eigenvalue derivation.",
    )
    parser.add_argument(
        "--spectral-poly-no-constant", action="store_true",
        help="--spectral-field-kind polynomial only. Sets AuxPropagatorConfig."
        "spectral_poly_no_constant (default False). Main-aux analogue of pde_head's own "
        "--pde-poly-no-constant -- architecturally zeros the bias/intercept term (no "
        "counterpart in true KS). See ks_latent.models.propagator._SpectralPDEDeltaBody's "
        "docstring for the exact mechanism.",
    )
    parser.add_argument(
        "--spectral-poly-exclude-nonconservative", action="store_true",
        help="--spectral-field-kind polynomial only, requires --spectral-poly-degree<=2. "
        "Sets AuxPropagatorConfig.spectral_poly_exclude_nonconservative (default False). "
        "Main-aux analogue of pde_head's own --pde-poly-exclude-nonconservative -- "
        "architecturally zeros every even-combined-order two-factor term (true KS "
        "conserves int(u)dx exactly; a two-factor term d_i*d_j's spatial mean is exactly "
        "zero for any state when i+j is odd, generically nonzero when i+j is even). See "
        "Section 167's own docstring/postmortem for the full derivation.",
    )
    parser.add_argument(
        "--spectral-poly-stable-w-xx-w-xxxx", action="store_true",
        help="--spectral-field-kind polynomial only. Sets AuxPropagatorConfig."
        "spectral_poly_stable_linear_terms={2: -1.0, 4: -1.0} (default None, off). Added "
        "2026-09-14, Section 174, user-directed: parametrize the propagator as a KS-shaped "
        "template (u_t = -u*u_x + nu*u_xx + ...) with nu LEARNABLE but sign-guaranteed, "
        "instead of hoping a loss penalty discourages collapse after the fact (Sections "
        "170-173's energy floor / varmatch mechanisms ALL converged to the identical "
        "collapsed fixed point regardless of timing). UNLIKE --pde-fix-w-xx-w-xxxx (which "
        "FREEZES the coefficient, no gradient, no counterpart for the main aux propagator "
        "at all until now), this keeps w_xx's and w_xxxx's coefficients LEARNABLE -- "
        "reparametrized as -(raw)**2, architecturally guaranteed negative regardless of "
        "how raw moves -- initialized at true KS's own -1.0/-1.0. Pair with "
        "--stable-linear-lr-factor for a much smaller learning rate on just these two "
        "scalars ('can't make huge steps in nu'). See "
        "ks_latent.models.propagator._SpectralPDEDeltaBody's docstring for the full "
        "mechanism and root-cause diagnosis.",
    )
    parser.add_argument(
        "--stable-linear-lr-factor", type=float, default=None,
        help="--spectral-poly-stable-w-xx-w-xxxx only. Override Stage1TrainingConfig."
        "stable_linear_lr_factor (default 0.02, i.e. 1/50th of --lr) -- the LR multiplier "
        "applied to w_xx's/w_xxxx's raw coefficients specifically, via a separate optimizer "
        "param group (not a gradient clip).",
    )
    parser.add_argument(
        "--spectral-poly-time-deriv", action="store_true",
        help="--spectral-field-kind polynomial/chebyshev only. Sets AuxPropagatorConfig."
        "spectral_poly_time_deriv (default False). Added 2026-09-18, Section 189, "
        "user-directed: 'can we incorporate time derivatives into the pde polynomial? "
        "might give us a richer expression. We can approximate them using rollout terms "
        "potentially'. Appends one extra library variable, a finite-difference estimate "
        "w_t_fd = (w - w_prev) / dt_snap, so terms like w_t, w*w_t, w_t*w_xxxx, w_t^2 can "
        "appear in the fitted PDE subject to the same poly_degree/poly_max_term_order "
        "budget as every other term (exempted from the max_term_order ORDER-SUM check "
        "specifically, since it is not a spatial derivative -- see "
        "ks_latent.models.propagator._polynomial_term_indices's free_index docstring). "
        "w_prev is a genuine previous-step state, not a second implicit unknown: real data "
        "for the very first predicted step of a rollout, the model's OWN prior prediction "
        "for every step after that (LatentPropagator.rollout already threads consecutive "
        "z_prev/z_curr pairs through its loop; step() was updated to stop discarding "
        "z_prev for markovian mode when this flag is active). step_one(z)'s single-"
        "argument call path (used by every Jacobian-based regularizer: delta_cap, "
        "kernel_unstable_floor, both spectrum-shape losses) is unaffected -- it has no "
        "z_prev to give, so the new feature falls back to an architectural zero there, "
        "same as a cold rollout start. See ks_latent.models.propagator."
        "_SpectralPDEDeltaBody's poly_time_deriv docstring in __init__ for the full "
        "mechanism.",
    )
    parser.add_argument(
        "--spectral-poly-time-deriv-dt-snap", type=float, default=None,
        help="--spectral-poly-time-deriv only. Override AuxPropagatorConfig."
        "spectral_poly_time_deriv_dt_snap (default 1.0, matching this project's universal "
        "dt_snap convention).",
    )
    parser.add_argument(
        "--spectral-poly-time-deriv2", action="store_true",
        help="--spectral-field-kind polynomial/chebyshev only. Sets AuxPropagatorConfig."
        "spectral_poly_time_deriv2 (default False). Added 2026-09-18, Section 190, "
        "user-directed: 'that might be the next thing to try, incorporate u_tt'. Appends a "
        "SECOND finite-difference feature, w_tt_fd = (w - 2*w_prev + w_prev2) / "
        "spectral_poly_time_deriv_dt_snap**2 (the standard 3-point BACKWARD second "
        "difference -- only past states, matching 'approximate them using rollout terms', "
        "no future state needed), independent of --spectral-poly-time-deriv (either may be "
        "on without the other). Needs a genuine z_prev2 (state two steps back) -- "
        "LatentPropagator.rollout() now maintains a 3-wide sliding window internally, so "
        "this becomes available starting the SECOND predicted step of any rollout with no "
        "change needed to the training loop's existing window construction, same as "
        "--spectral-poly-time-deriv's own z_prev threading.",
    )
    parser.add_argument(
        "--spectral-integrator", choices=["euler", "rk4", "etdrk4"], default="euler",
        help="--aux-backbone spectral_pde only. AuxPropagatorConfig.spectral_integrator "
        "(default 'euler'). See PropagatorConfig's 'spectral_pde' docstring and "
        "docs/sine_transform_pde_plan.md Section 5a for the full euler/rk4/etdrk4 "
        "comparison.",
    )
    parser.add_argument(
        "--ode-substeps", type=int, default=1,
        help="--aux-backbone spectral_pde ('rk4'/'etdrk4') only. AuxPropagatorConfig."
        "ode_substeps (default 1).",
    )
    parser.add_argument(
        "--spectral-physics-prior", action="store_true",
        help="--aux-backbone spectral_pde only. AuxPropagatorConfig.spectral_physics_prior "
        "(default off). Bakes the EXACT true KS right-hand side into field()'s output as a "
        "fixed baseline (added 2026-09-08, user-directed: 'what if we had another variant "
        "that assumes that the latent space follows exactly the KS dynamics with some "
        "learned correction term ... That way we already have chaos in the formulation') -- "
        "the pointwise MLP now learns only a correction on top, rather than the whole "
        "dynamics from scratch. Integrator-aware: only -w*w_x is baked in under 'etdrk4' "
        "(the linear part is already handled separately there); the full -w*w_x-w_xx-w_xxxx "
        "is baked in under 'euler'/'rk4'. Requires --spectral-max-order >= 4. See "
        "PropagatorConfig.spectral_physics_prior's docstring for the full motivation "
        "(Lyapunov exponents are invariant under a smooth change of coordinates, so if the "
        "encoder is close to invertible this should inherit KS's own genuine chaos).",
    )
    parser.add_argument(
        "--pde-distill", action="store_true",
        help="Any --encoder (NOT combined with --aux-backbone spectral_pde/spectral_pde_raw "
        "-- a second head alongside an already-constrained aux is redundant, raises "
        "ValueError). Builds a SECOND propagator, always backbone='spectral_pde_raw' "
        "('pde_head', added 2026-09-08, generalizing the original spectral_field-only "
        "'spectral_pde' version -- user-directed: 'I would like to be able to use the "
        "encoder and decoder and propagator from 95 ... within the propagator, I want to "
        "take the fourier transform of the latent states, and train the spectral pde with "
        "this information'), trained jointly ALONGSIDE the real --aux-backbone (kept "
        "free/unconstrained, e.g. 'mlp' -- the one that actually drives l_pred) via a "
        "single-step distillation loss (--w-pde-distill) against aux's own realized "
        "one-step prediction. pde_head treats z's own d_latent indices as a periodic ring "
        "(same 'native index as space' convention as --w-spatial), takes z's OWN "
        "truncated self-FFT (--pde-K/--pde-L), then runs the ordinary derivative-"
        "synthesis + pointwise-MLP + integrator machinery on that -- see "
        "Stage1TrainingConfig.w_pde_distill's and train_stage1's own docstrings, and "
        "docs/sine_transform_pde_plan.md's pde_head sections, for the full mechanism. "
        "pde_head is never asked to integrate its own multi-step rollout, only to match a "
        "single-step regression target, sidestepping the stiffness/collapse issues that "
        "made backbone='spectral_pde' so hard to train as the PRIMARY propagator "
        "(Sections 101-106). Saved separately as stage1_pdehead_<profile><tag>.pt.",
    )
    parser.add_argument(
        "--w-pde-distill", type=float, default=None,
        help="--pde-distill only. Override Stage1TrainingConfig.w_pde_distill (default "
        "0.0, off -- must be set >0 together with --pde-distill for the term to have any "
        "effect).",
    )
    parser.add_argument(
        "--w-pde-coeff-l1", type=float, default=None,
        help="--pde-distill only, and only with --pde-field-kind polynomial/chebyshev. "
        "Override Stage1TrainingConfig.w_pde_coeff_l1 (default 0.0, off): weight on "
        "pde_head.body's own poly_coeffs.weight.abs().sum() -- an L1 sparsity prior on "
        "the distilled PDE's own coefficients, added 2026-09-10, user-directed: 'I "
        "wonder if we could impose some sparsity using l1 norm on the polynomial "
        "coefficients of the pde'. Raises ValueError if pde_head has no poly_coeffs "
        "(field_kind='mlp'). See --pde-coeff-l1-linear-only to restrict this to only "
        "the LINEAR terms.",
    )
    parser.add_argument(
        "--pde-coeff-l1-linear-only", action="store_true",
        help="--w-pde-coeff-l1 only. Sets Stage1TrainingConfig.pde_coeff_l1_linear_only=True "
        "(default False, penalizes every term). Restricts the L1 penalty to only the "
        "LINEAR (single-derivative-index) terms -- w, w_x, w_xx, etc. -- excluding the "
        "constant and every multi-factor product/power term. Added 2026-09-11, "
        "user-directed: 'please try using the w_pde_coeff_l1 machinery to penalize "
        "single index terms' -- motivated by the finding that every closure whose "
        "LARGEST coefficient was linear produced simple, near-periodic standalone "
        "rollouts (exact Fourier eigenmodes of a linear operator on a periodic domain), "
        "while closures dominated by a genuine nonlinear term (w*w_x, w_x*w_xx) looked "
        "visibly richer -- pressures the fit toward nonlinear-term dominance without "
        "directly penalizing (and thus suppressing) those nonlinear terms themselves.",
    )
    parser.add_argument(
        "--pde-mutual", action="store_true",
        help="--pde-distill only. Sets Stage1TrainingConfig.pde_distill_detach_target=False "
        "(default True, unchanged/protected behavior). Added 2026-09-08, user-directed: 'I "
        "think the loss should be mutual for stage 1 training too. We always want the "
        "propagator to have dynamics that can be easily modeled by the pde_head right?' -- "
        "lets gradient from the distillation loss ALSO reach aux directly (not just the "
        "encoder), genuinely pressuring the propagator itself toward PDE-describable "
        "dynamics. Real risk: this is the same 'pressure toward simplicity' mechanism "
        "behind this project's H-PROP fixed-point-collapse finding -- monitor l_pred/Gate "
        "3 D_KY closely when this is set. See train_stage1's own docstring.",
    )
    parser.add_argument(
        "--w-pde-distill-real", type=float, default=None,
        help="--pde-distill only. Override Stage1TrainingConfig.w_pde_distill_real (default "
        "0.0, off). A SEPARATE single-step distillation term, independent of "
        "--w-pde-distill -- instead of matching aux's own predicted next state, matches "
        "the REAL, actually-encoded next state from data (every consecutive pair already "
        "in the current batch's z_win, no new forward passes). Added 2026-09-11, "
        "user-directed: 'just train on one step at a time from the true latent dynamics "
        "derived from the encoder. but you can train on the whole batch of latent states "
        "from the encoder at a time'. NOT the same idea as the already-failed "
        "fit_latent_pde.py SINDy attempt (frozen closed-form regression on noisy finite-"
        "difference derivative estimates, R^2~0.005) -- this is gradient descent on the "
        "actual discrete step_one map, comparing real ENCODED VALUES directly (no "
        "derivative-estimation noise), and can run jointly with encoder/propagator "
        "training. See Stage1TrainingConfig.w_pde_distill_real's docstring.",
    )
    parser.add_argument(
        "--pde-distill-real-mutual", action="store_true",
        help="--w-pde-distill-real only. Sets Stage1TrainingConfig.pde_distill_real_detach"
        "=False (default True, protected). Same 'pressure toward simplicity' risk class as "
        "--pde-mutual -- monitor Gate 3 D_KY if set.",
    )
    parser.add_argument(
        "--w-pde-distill-real-rollout", type=float, default=None,
        help="--pde-distill only. Override Stage1TrainingConfig.w_pde_distill_real_rollout "
        "(default 0.0, off). iLED-style (arXiv:2309.05812) multi-step forecast distillation: "
        "pde_head.rollout for --pde-distill-real-rollout-k steps from a REAL starting state, "
        "compared at every step to REAL future states (never aux/propagator's own "
        "predictions). Added 2026-09-12, user-directed: 'directly replicate iLED's "
        "stabilization mechanism and incorporate it into a rollout term to pde_distill loss' "
        "-- supplies gradient pressure against exponential blowup under pde_head's own "
        "repeated integration, which --w-pde-distill-real's single-step target cannot. See "
        "Stage1TrainingConfig.w_pde_distill_real_rollout's docstring.",
    )
    parser.add_argument(
        "--pde-distill-real-rollout-k", type=int, default=4,
        help="--w-pde-distill-real-rollout only. Stage1TrainingConfig."
        "pde_distill_real_rollout_k (default 4). Must be <= max(--pde-k-pred, "
        "--pde-k-pred-max) -- raises otherwise.",
    )
    parser.add_argument(
        "--pde-distill-real-rollout-warmup-epochs", type=int, default=None,
        help="--w-pde-distill-real-rollout only. Override Stage1TrainingConfig."
        "pde_distill_real_rollout_warmup_epochs. Default (None) here computes "
        "max(1, round(0.3*epochs)), the SAME 30%%-of-training convention --multistep's "
        "own k_pred_max warmup uses. NOT safe to leave at Stage1TrainingConfig's own literal "
        "default of 0 (immediate full-horizon rollout) -- found empirically (2026-09-12) to "
        "diverge (inf/nan) on the very first batch against an undertrained pde_head, "
        "poisoning the whole loss and the optimizer's moment estimates permanently.",
    )
    parser.add_argument(
        "--w-pde-nonlinear-l2", type=float, default=None,
        help="--pde-distill only, and only with --pde-field-kind polynomial/chebyshev. "
        "Override Stage1TrainingConfig.w_pde_nonlinear_l2 (default 0.0, off). iLED-style "
        "(arXiv:2309.05812) L_non-linearity: penalizes the squared magnitude of pde_head's "
        "own NONLINEAR-only closure output (every term of length>=2 -- everything except "
        "the constant and single-derivative linear terms), evaluated on real detached "
        "states. The linear-part-stability half of iLED's mechanism is already covered by "
        "--pde-fix-w-xx-w-xxxx/--pde-poly-stable-leading; this is the other half -- keeping "
        "the nonlinear closure from growing large enough to dominate/destabilize it. Added "
        "2026-09-12, user-directed alongside --w-pde-distill-real-rollout.",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage1TrainingConfig."
        "w_pde_spectrum_shape (default 0.0, off). Added 2026-09-21, Section 192, "
        "user-directed: pde_head analogue of --w-spectrum-shape (Section 131), applied to "
        "pde_head.step_one specifically -- NEVER the main aux propagator. Motivation: "
        "Section 167's own pde_head (fit against real data only, propagator fully "
        "decoupled and already genuinely chaotic) never diverges standalone but decays "
        "toward a fixed point rather than sustaining chaos. Evaluated on real detached "
        "states (never mutual with the encoder), same once-per-epoch cost convention as "
        "the main-propagator version.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-n-expand", type=int, default=None,
        help="--w-pde-spectrum-shape only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_n_expand (default 13).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-expand-target", type=float, default=None,
        help="--w-pde-spectrum-shape only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_expand_target (default 1.1).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-contract-floor", type=float, default=None,
        help="--w-pde-spectrum-shape only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_contract_floor (default 0.6).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_n_samples (default 32).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-two-sided", action="store_true",
        help="--w-pde-spectrum-shape only. Sets Stage1TrainingConfig."
        "pde_spectrum_shape_two_sided (default False). See --spectrum-shape-two-sided's "
        "help text -- identical mechanism, applied to pde_head instead of the main "
        "propagator.",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-multistep", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage1TrainingConfig."
        "w_pde_spectrum_shape_multistep (default 0.0, off). pde_head analogue of "
        "--w-spectrum-shape-multistep (Section 186), applied to pde_head.step_one only.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-multistep-k", type=int, default=None,
        help="--w-pde-spectrum-shape-multistep only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_multistep_k (default 10).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-multistep-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape-multistep only. Override Stage1TrainingConfig."
        "pde_spectrum_shape_multistep_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-self", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage1TrainingConfig."
        "w_pde_spectrum_shape_self (default 0.0, off). Added 2026-09-21, Section 193, "
        "user-directed: SELF-rollout-sampled analogue of --w-pde-spectrum-shape -- evaluated "
        "at states pde_head's OWN free rollout actually visits (rolled --pde-spectrum-shape-"
        "self-rollout-k steps forward from real starting states under torch.no_grad(), then "
        "detached), not real on-attractor data. Independent weight from --w-pde-spectrum-"
        "shape (both may be on at once); shares that flag's n_expand/expand_target/"
        "contract_floor/two_sided targets.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-self-rollout-k", type=int, default=None,
        help="--w-pde-spectrum-shape-self/--w-pde-spectrum-shape-multistep-self only. "
        "Override Stage1TrainingConfig.pde_spectrum_shape_self_rollout_k (default 20) -- how "
        "many steps to roll pde_head forward (no_grad) before sampling the evaluation state.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-self-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape-self/--w-pde-spectrum-shape-multistep-self only. "
        "Override Stage1TrainingConfig.pde_spectrum_shape_self_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-multistep-self", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage1TrainingConfig."
        "w_pde_spectrum_shape_multistep_self (default 0.0, off). SELF-rollout-sampled "
        "analogue of --w-pde-spectrum-shape-multistep -- same self-rollout state pool as "
        "--w-pde-spectrum-shape-self (shares --pde-spectrum-shape-self-rollout-k/-n-samples), "
        "composed --pde-spectrum-shape-multistep-k further steps for the Jacobian check.",
    )
    parser.add_argument(
        "--w-pde-mean-conservation", type=float, default=None,
        help="--pde-distill only. Override Stage1TrainingConfig.w_pde_mean_conservation "
        "(default 0.0, off). Penalizes pde_head.body.field(z).mean(dim=-1).pow(2).mean() on "
        "real detached states -- true KS conserves int(u)dx exactly (every term in "
        "-u*u_x-u_xx-u_xxxx is a total x-derivative, integrating to zero over a periodic "
        "domain); a freely-fit polynomial closure has no such guarantee. Added 2026-09-12, "
        "user-directed: 'can you recommend a fix to prevent drift toward mode-0' -- Section "
        "164's standalone rollout was broadband at t=0 but drifted to 72-86%% concentration "
        "in the self-FFT DC/mean mode by t=54, and field(z).mean(dim=-1) measured "
        "-0.00031+-0.0045 on real states (small but systematically nonzero). See "
        "Stage1TrainingConfig.w_pde_mean_conservation's docstring for the full mechanism.",
    )
    parser.add_argument(
        "--w-pde-energy-floor", type=float, default=None,
        help="--pde-distill only. Override Stage1TrainingConfig.w_pde_energy_floor (default "
        "0.0, off). Penalizes ks_latent.training.losses.spatial_energy_floor_loss on "
        "pde_head's OWN unsupervised autoregressive rollout (no real target beyond the "
        "starting state). Added 2026-09-13, user-directed after viewing Section 167's own "
        "standalone smooth-field GIF: 'I also see the decay in the pde trajectory towards "
        "0... force the model to always have an energetic component' -- confirmed by direct "
        "measurement: pde_head-as-propagator's own Lyapunov spectrum gave D_KY=0/n_positive="
        "0/lambda1=-0.006 (converges to a fixed point, not genuine chaos). See "
        "Stage1TrainingConfig.w_pde_energy_floor's docstring for the full mechanism.",
    )
    parser.add_argument(
        "--pde-energy-floor-gamma", type=float, default=0.4,
        help="--w-pde-energy-floor only. Stage1TrainingConfig.pde_energy_floor_gamma "
        "(default 0.4). Real encoded z's own spatial std (across the d_latent/ring index) "
        "averages ~0.97, minimum ~0.53 (measured on Section 167's real data) -- keep this "
        "clearly below that range.",
    )
    parser.add_argument(
        "--pde-energy-floor-rollout-k", type=int, default=30,
        help="--w-pde-energy-floor only. Stage1TrainingConfig.pde_energy_floor_rollout_k "
        "(default 30). UNSUPERVISED (no real future states needed), so NOT constrained by "
        "the training window size -- can be much longer than --pde-distill-real-rollout-k.",
    )
    parser.add_argument(
        "--pde-energy-floor-warmup-epochs", type=int, default=None,
        help="--w-pde-energy-floor only. Override Stage1TrainingConfig."
        "pde_energy_floor_warmup_epochs. Default (None) here computes max(1, round(0.3*"
        "epochs)), the same convention --pde-distill-real-rollout-warmup-epochs uses. NOT "
        "safe to leave at the config's literal default of 0 (immediate full-horizon rollout "
        "against an undertrained pde_head risks the same inf/nan-and-permanently-corrupt-"
        "the-optimizer failure mode found empirically in Section 159's postmortem).",
    )
    parser.add_argument(
        "--w-prop-energy-floor", type=float, default=None,
        help="Override Stage1TrainingConfig.w_prop_energy_floor (default 0.0, off). "
        "Generalizes --w-pde-energy-floor to the PRIMARY propagator (aux) directly, for the "
        "case where aux itself IS the closure under test (e.g. --aux-backbone "
        "spectral_pde_raw, no separate pde_head). Added 2026-09-14, user-directed after "
        "Sections 169/171 both collapsed to a fixed point (Lyapunov lambda1=-0.04, D_KY=0) "
        "and Stage2TrainingConfig.w_varmatch (tried first) failed to prevent it -- w_varmatch "
        "is bounded by k_now/k_max and cannot see a collapse rate too slow to show up within "
        "that horizon. This term is UNSUPERVISED (no window-size constraint). See "
        "Stage1TrainingConfig.w_prop_energy_floor's docstring for the full mechanism.",
    )
    parser.add_argument(
        "--prop-energy-floor-gamma", type=float, default=0.4,
        help="--w-prop-energy-floor only. Stage1TrainingConfig.prop_energy_floor_gamma "
        "(default 0.4). See --pde-energy-floor-gamma's own help text.",
    )
    parser.add_argument(
        "--prop-energy-floor-rollout-k", type=int, default=30,
        help="--w-prop-energy-floor only. Stage1TrainingConfig.prop_energy_floor_rollout_k "
        "(default 30). UNSUPERVISED, no window-size constraint.",
    )
    parser.add_argument(
        "--prop-energy-floor-warmup-epochs", type=int, default=None,
        help="--w-prop-energy-floor only. Override Stage1TrainingConfig."
        "prop_energy_floor_warmup_epochs. Default (None) here computes max(1, round(0.3*"
        "epochs)). NOT safe to leave at the config's literal default of 0 -- see "
        "--pde-energy-floor-warmup-epochs's own help text for why.",
    )
    parser.add_argument(
        "--pde-K", type=int, default=None,
        help="--pde-distill only. pde_head's self-FFT truncation (AuxPropagatorConfig."
        "spectral_K under backbone='spectral_pde_raw' -- an internal parameter of "
        "pde_head's OWN transform, unrelated to any --spectral-K used elsewhere). "
        "Default (recommended, user-confirmed 2026-09-08): d_latent//2+1, i.e. NO "
        "truncation -- keeps the entire spectrum of z, discarding nothing.",
    )
    parser.add_argument(
        "--pde-L", type=float, default=None,
        help="--pde-distill only. pde_head's assumed ring circumference for treating "
        "z's own index as a spatial coordinate (AuxPropagatorConfig.spectral_L under "
        "backbone='spectral_pde_raw'). Genuinely arbitrary -- the latent index has no "
        "natural physical length scale. Default (recommended, user-confirmed "
        "2026-09-08): d_latent (unit index spacing).",
    )
    parser.add_argument(
        "--pde-hidden", type=int, default=128,
        help="--pde-distill only. pde_head's AuxPropagatorConfig.hidden (default 128, "
        "matching --full-propagator's own sizing convention).",
    )
    parser.add_argument(
        "--pde-n-blocks", type=int, default=3,
        help="--pde-distill only. pde_head's AuxPropagatorConfig.n_blocks (default 3).",
    )
    parser.add_argument(
        "--pde-integrator", choices=["euler", "rk4", "etdrk4"], default="euler",
        help="--pde-distill only. pde_head's AuxPropagatorConfig.spectral_integrator "
        "(default 'euler' -- NOT 'etdrk4', unlike backbone='spectral_pde's own default "
        "recommendation, and deliberately so: 'etdrk4' bakes in Lhat=k^2-k^4, KS's TRUE "
        "physical dispersion relation, EXACTLY -- a correct, load-bearing physical fact "
        "for a genuine spectral_field z, but an unjustified and empirically HARMFUL "
        "assumption for backbone='spectral_pde_raw''s self-FFT of an arbitrary, LEARNED "
        "latent ordering. Verified directly (2026-09-08): with --pde-integrator etdrk4 on "
        "a raw (non-spectral) encoder, training loss diverged (0.57 -> 268.0 over 5 "
        "epochs); --pde-integrator euler on the identical setup trained stably (0.51 -> "
        "0.48). 'euler'/'rk4' impose no dispersion-relation assumption at all -- the "
        "pointwise MLP learns the ENTIRE right-hand side from the closed-form "
        "derivatives, which is what this backbone's self-FFT is actually for (exact "
        "derivatives), not KS's own specific linear stiffness. 'etdrk4' remains "
        "selectable to deliberately test whether a k^2-k^4-like prior helps for a "
        "SPECIFIC learned ordering, but is not a safe default here.",
    )
    parser.add_argument(
        "--pde-ode-substeps", type=int, default=1,
        help="--pde-distill only. pde_head's AuxPropagatorConfig.ode_substeps (default "
        "1 -- pde_head matches ONE of aux's own steps, so 1 substep is the literal, "
        "natural default; raise only to test whether finer internal integration within "
        "that same single target interval fits better).",
    )
    parser.add_argument(
        "--pde-max-order", type=int, default=4,
        help="--pde-distill only. pde_head's AuxPropagatorConfig.spectral_max_order "
        "(default 4).",
    )
    parser.add_argument(
        "--pde-field-kind", choices=["mlp", "polynomial", "chebyshev"], default="mlp",
        help="--pde-distill only. pde_head's AuxPropagatorConfig.spectral_field_kind "
        "(default 'mlp'). See --spectral-field-kind's help text -- identical mechanism, "
        "applied to pde_head instead of a spectral_pde/spectral_pde_raw aux. 'chebyshev' "
        "(added 2026-09-11, user-directed: 'I want to consider whether or not another "
        "basis, e.g. chebyshev polynomials could be used to represent the pde') -- SAME "
        "term structure/count as 'polynomial' (--pde-poly-degree/--pde-poly-max-term-order "
        "apply identically), but each term is built from Chebyshev polynomials of the "
        "repeated derivative channels (ks_latent.models.propagator._chebyshev_library) "
        "instead of plain monomial powers -- potentially better-conditioned during "
        "training. Convert the trained poly_coeffs.weight back to ordinary, directly-"
        "interpretable monomial coefficients afterward via "
        "ks_latent.models.propagator._chebyshev_to_monomial_matrix (NOT done "
        "automatically -- this only changes the TRAINING-time basis).",
    )
    parser.add_argument(
        "--pde-poly-degree", type=int, choices=[1, 2, 3], default=2,
        help="--pde-field-kind polynomial only. pde_head's AuxPropagatorConfig."
        "spectral_poly_degree (default 2). 3 (added 2026-09-11, user-directed: 'limit "
        "the polynomial to have at most 3 values') allows up to 3-factor products "
        "(e.g. w*w_x*w_xx, w_x^3) -- combine with a low --pde-max-order and a modest "
        "--pde-poly-max-term-order to keep the library small: e.g. --pde-max-order 2 "
        "(only w/w_x/w_xx) --pde-poly-degree 3 --pde-poly-max-term-order 6 gives 19 "
        "terms total (allows e.g. w^2*w_xx or w*w_x^2*w_xx, EXCLUDES only w_xx^3, the "
        "single combined-order-6 term at this n_vars/degree).",
    )
    parser.add_argument(
        "--pde-poly-max-term-order", type=int, default=None,
        help="--pde-field-kind polynomial only. pde_head's AuxPropagatorConfig."
        "spectral_poly_max_term_order (default None, unrestricted). See "
        "--spectral-poly-max-term-order's help text -- identical mechanism.",
    )
    parser.add_argument(
        "--pde-poly-stable-leading", action="store_true",
        help="--pde-field-kind polynomial only. pde_head's AuxPropagatorConfig."
        "spectral_poly_stable_leading (default False). See "
        "--spectral-poly-stable-leading's help text -- identical mechanism, "
        "architecturally forces the leading kept even-order linear coefficient's "
        "sign to guarantee UV (high-wavenumber) stability. Added 2026-09-10, "
        "Section 143, alongside --pde-poly-max-term-order, after Sections 140-142 "
        "found pde_head's own free-running rollout diverges past a certain "
        "horizon regardless of term-order restriction alone.",
    )
    parser.add_argument(
        "--pde-poly-no-constant", action="store_true",
        help="--pde-field-kind polynomial/chebyshev only. pde_head's AuxPropagatorConfig."
        "spectral_poly_no_constant (default False). Added 2026-09-11, user-directed: "
        "'we should just force the constant to be 0 during training' -- removes the "
        "Linear layer's own bias AND the '()' term's own weight column (both "
        "architecturally, not just via regularization), so the closure can never learn "
        "a uniform additive offset. Motivated by a direct finding: a closure with "
        "--pde-coeff-l1-linear-only already zeroing every linear term still produced a "
        "near-static standalone rollout, with the constant term absorbing 61%% of the "
        "coefficient mass.",
    )
    parser.add_argument(
        "--pde-fix-w-xx-w-xxxx", action="store_true",
        help="--pde-field-kind polynomial/chebyshev only. Sets pde_head's AuxPropagatorConfig."
        "spectral_poly_fixed_linear_terms={2: -1.0, 4: -1.0} (default None, off). Added "
        "2026-09-11, user-directed: 'assume the pde always had -w_xx-w_xxxx, we'll just "
        "learn the rest of the terms around this' -- architecturally FIXES the physical "
        "coefficients on w_xx and w_xxxx to exactly -1.0 (matching true KS's own "
        "dissipation operator), never learned (a plain constant substituted every forward "
        "pass, zero gradient, frozen at that value for the whole run). Every OTHER term "
        "(w, w_x, w_xxx, w*w_x, and every higher cross/product term) stays fully learned. "
        "Requires max_order>=4 with both (2,) and (4,) surviving --pde-poly-max-term-order "
        "filtering, or raises ValueError.",
    )
    parser.add_argument(
        "--pde-poly-exclude-nonconservative", action="store_true",
        help="--pde-field-kind polynomial/chebyshev only, requires --pde-poly-degree<=2. Sets "
        "pde_head's AuxPropagatorConfig.spectral_poly_exclude_nonconservative=True (default "
        "False, off). Added 2026-09-12, user-directed: 'please try to exclude every "
        "even-combined-order two-factor term from the library' -- true KS conserves int(u)dx "
        "EXACTLY; for a two-factor term d_i*d_j, this integral is exactly zero for any state "
        "when i+j is odd (e.g. w_xx*w_xxx), and generically nonzero when i+j is even (e.g. "
        "w_x*w_x, w*w -- reduces to a perfect square (w^((i+j)/2))^2). Architecturally zeros "
        "every even-combined-order length-2 term's raw contribution -- an EXACT structural "
        "fix (no weight to tune, unlike --w-pde-mean-conservation), verified against Section "
        "164's own fitted coefficients (w_x*w_x measured as the largest violator, -0.043).",
    )
    parser.add_argument(
        "--multistep", action="store_true",
        help="Use a real-sized propagator (hidden=128, n_blocks=3) with a "
        "rollout length ramped 2->8 over the first 30%% of epochs, instead "
        "of the brief's tiny fixed-k_pred=2 auxiliary propagator.",
    )
    parser.add_argument(
        "--w-pred", type=float, default=None,
        help="Override Stage1TrainingConfig.w_pred (default 0.5): weight on L_pred, the "
        "joint dynamics-shaping term (aux propagator rolls z forward, decoded, compared "
        "against future ground truth -- flows gradients into the encoder). User-directed "
        "(2026-08-29): docs/PROJECT_HANDOFF.md attributes the reference project's genuinely "
        "chaotic latent (D_KY~21.4, 11 positive Lyapunov exponents) to this term specifically "
        "-- 'WHY Stage 2 works and Lyapunov is clean'. Our Stage-2 propagator instead collapsed "
        "to a fixed point (D_KY=0); raising this weight is the direct Stage-1-side lever to "
        "try recovering that result.",
    )
    parser.add_argument(
        "--k-pred-max", type=int, default=None,
        help="Override Stage1TrainingConfig.k_pred_max directly (independent of --multistep, "
        "which hardcodes 8). User-directed (2026-08-29): e.g. 4 for a 4-step-ahead rollout "
        "L_pred (ramped 2 -> k_pred_max over the first 30%% of epochs, same as --multistep). "
        "Default None preserves the existing --multistep-linked behavior (8 if --multistep "
        "else 0).",
    )
    parser.add_argument(
        "--masked-mlp-expand-factor", type=int, default=3,
        help="--aux-backbone masked_mlp_expand only. AuxPropagatorConfig."
        "masked_mlp_expand_factor (default 3): widens the backbone's middle "
        "layer to expand_factor*d_latent -- see PropagatorConfig's "
        "'masked_mlp_expand' docstring section and "
        "ks_latent.models.propagator._MaskedMLPExpandDeltaBody's docstring "
        "for the exact effective-neighbor-radius derivation (attn_window * "
        "expand_factor).",
    )
    parser.add_argument(
        "--attn-window", type=int, default=None,
        help="Restrict attention to +-attn_window ring-neighbours (circular "
        "distance, non-causal), instead of full attention. Applies to "
        "--encoder vit's encoder/decoder AND --aux-backbone in "
        "{transformer, vit}'s propagator (linear distance for 'transformer', "
        "ring distance for 'vit' -- see build_ring_local_attention_mask's "
        "docstring for why they differ). Default None = full attention "
        "(unchanged behavior). User-directed (2026-08-29): tests whether a "
        "hard locality constraint, not just the positional encoding alone, "
        "is needed to organize the latent's structure. Also reused (2026-09-03, "
        "user-directed) by --aux-backbone fourier_mlp AND --encoder fourier_mlp: "
        "instead of full attention, restricts their raw-value path(s) to a "
        "circular-band mask at this radius -- see PropagatorConfig's "
        "backbone='fourier_mlp' and FourierMLPAutoencoderConfig's docstrings.",
    )
    parser.add_argument(
        "--prop-dense", action="store_true",
        help="Force the propagator's attn_window to None (fully dense/full "
        "attention), overriding --attn-window for the propagator only -- "
        "the encoder/decoder still use --attn-window as given. User-directed "
        "(2026-09-03, Section 72): masked encoder/decoder + fully dense "
        "fourier_mlp propagator, decoupling the two after Section 71 shared "
        "one --attn-window value across both.",
    )
    parser.add_argument(
        "--dec-use-ifft", action="store_true",
        help="--encoder fourier_mlp only: force the DECODER to the fully-"
        "dense ifft-featurized structure (FourierMLPAutoencoderConfig."
        "dec_use_ifft), regardless of --attn-window (which then applies to "
        "the encoder only). User-directed (2026-09-03, Section 73): "
        "'a dense inverse fourier mlp for the decoder where the inverse "
        "fourier mlp applied the ifft not the fft' -- see that config "
        "field's docstring for the exact irfft-based feature computation.",
    )
    parser.add_argument(
        "--fourier-ifft-readout", action="store_true",
        help="--encoder fourier_mlp only: swap the encoder's/decoder's "
        "Fourier-features-only sub-network for a FourierIFFTBody "
        "(FourierMLPAutoencoderConfig.fourier_ifft_readout) -- predicts "
        "frequency-domain coefficients via an MLPDeltaBody, then explicitly "
        "inverse-transforms (irfft) back to state space, instead of a plain "
        "unconstrained linear readout. Highest priority of the decoder "
        "options (overrides --dec-use-ifft for the decoder). The decoder "
        "gets the SAME two-network structure as the encoder under this flag "
        "-- a masked raw-value path (at --dec-attn-window, falling back to "
        "--attn-window if not given) SUMMED with the FourierIFFTBody, still "
        "additive/non-interacting -- not a fully-dense/no-raw-path "
        "structure. User-directed (2026-09-03, Section 74, corrected "
        "same day): 'apply an inverse fft to the frequency component "
        "output of the fourier mlp to map back to state space in the "
        "encoder and propagator and decoder ... I want the propagator and "
        "the decoder to have the same structure as the encoder ... but I "
        "want the attn_window to be much larger'.",
    )
    parser.add_argument(
        "--dec-attn-window", type=int, default=None,
        help="--fourier-ifft-readout only: the decoder's masked raw-value "
        "path radius, independent of --attn-window (which is the "
        "encoder's). None (default) falls back to --attn-window's value.",
    )
    parser.add_argument(
        "--prop-fourier-ifft", action="store_true",
        help="--aux-backbone fourier_mlp only: propagator counterpart of "
        "--fourier-ifft-readout (PropagatorConfig.fourier_ifft_readout). "
        "Gets the same two-network structure (masked raw-value path SUMMED "
        "with the FourierIFFTBody) whenever the propagator's own window "
        "(--prop-attn-window, falling back to --attn-window, unless "
        "--prop-dense forces it to None) is not None; None means no "
        "raw-value path at all (pure Fourier+irfft). User-directed "
        "(2026-09-03, Section 74, corrected same day).",
    )
    parser.add_argument(
        "--prop-attn-window", type=int, default=None,
        help="The propagator's masked-path radius, independent of "
        "--attn-window (which is the encoder's/AE's). None (default) "
        "falls back to (None if --prop-dense else --attn-window)'s value -- "
        "i.e. this flag, when given, takes priority over both.",
    )
    parser.add_argument(
        "--nonexpansive", action="store_true",
        help="--encoder fourier_mlp only: constrain the AE's Linear layers "
        "to be non-expansive (spectral normalization + damped/averaged "
        "residuals + norm='ortho' FFTs -- FourierMLPAutoencoderConfig."
        "nonexpansive). User-directed (2026-09-04): 'would there be a way "
        "to constrain the fourier_mlp to be nonexpansive' -- ViT's "
        "attention is structurally non-expansive (softmax = convex "
        "combination); this gives fourier_mlp's plain linear layers an "
        "analogous guarantee. See ks_latent.models.propagator."
        "ResidualMLPBlock's docstring for the full mechanism.",
    )
    parser.add_argument(
        "--prop-nonexpansive", action="store_true",
        help="--aux-backbone fourier_mlp only: propagator counterpart of "
        "--nonexpansive (PropagatorConfig.nonexpansive). User-directed "
        "(2026-09-04).",
    )
    parser.add_argument(
        "--pos-encoding", choices=["circular", "linear"], default="circular",
        help="For --encoder vit and --aux-backbone vit only (2026-08-29, "
        "user-directed). 'circular' (default) = CircularPositionalEncoding "
        "+ ring-distance --attn-window masking (correct for the encoder's "
        "genuinely periodic domain). 'linear' = LinearPositionalEncoding "
        "(fixed sinusoidal, non-learned, but NOT periodic) + linear-distance "
        "masking -- same ViTBlock/mlp_ratio architecture either way, "
        "isolating exactly the periodic-vs-non-periodic assumption.",
    )
    parser.add_argument(
        "--aux-n-tokens", type=int, default=4,
        help="Number of tokens the aux propagator's transformer/vit backbone "
        "chunks z into (must divide d_latent). Default 4 matches the "
        "original transformer-backbone sizing, but on a 4-token ring the "
        "max possible ring distance is 2, so --attn-window 4 would impose "
        "no real restriction there. Raise this (e.g. 44 = one token per "
        "latent coordinate, at d_latent=44) to make --attn-window "
        "meaningfully restrictive for the propagator too.",
    )
    parser.add_argument(
        "--aux-token-d-model", type=int, default=32,
        help="d_model for the aux propagator's transformer/vit backbone. "
        "CircularPositionalEncoding (the 'vit' backbone) requires this to "
        "be >= the ring's independent harmonic count (~n_tokens for even "
        "n_tokens), so raise this alongside --aux-n-tokens (e.g. 64 for "
        "--aux-n-tokens 44) or construction will raise ValueError.",
    )
    parser.add_argument(
        "--aux-hidden", type=int, default=None,
        help="--aux-backbone mlp only. Override the aux propagator's hidden width "
        "(default 128 under --full-propagator/--multistep, 64 otherwise, both with "
        "n_blocks=3 -- see the aux_hidden/aux_blocks sizing logic). Added "
        "2026-09-02, user-directed ('make the models 20 percent bigger'): with "
        "n_blocks unchanged at 3 and d_latent=44, hidden=141 gives ~1.2x the "
        "hidden=128 default's total parameter count (111532 -> 133853), matching "
        "the precedent already used for Stage 2's --hidden 141.",
    )
    parser.add_argument(
        "--aux-blocks", type=int, default=None,
        help="--aux-backbone mlp only. Override the aux propagator's n_blocks "
        "(default 3 under --full-propagator/--multistep, 2 otherwise).",
    )
    parser.add_argument(
        "--token-window", type=int, default=None,
        help="--encoder vit only (2026-08-29, user-directed). Overlapping-patch "
        "tokenization: each encoder token's INPUT becomes a token_window-wide, "
        "circularly-padded slice of the field (>= patch_size), centered on its own "
        "patch_size-wide output slot -- the decoder is untouched (no overlap-add "
        "needed). Precedent: PVTv2/T2T-ViT overlapping patch embedding, PatchTST's "
        "overlapping-stride patching. Default None = non-overlapping (unchanged).",
    )
    parser.add_argument(
        "--aux-token-window", type=int, default=None,
        help="--aux-backbone vit only. Same idea as --token-window, applied to the aux "
        "propagator's tokenization of z (>= d_latent // --aux-n-tokens). Default None = "
        "non-overlapping (unchanged).",
    )
    parser.add_argument(
        "--pool",
        choices=[
            "mean", "cls", "none", "local", "local_attn", "banded", "gated",
            "token_mlp", "local_token_mlp",
        ],
        default="mean",
        help="--encoder vit only (2026-08-29, user-directed). 'mean' (default)/'cls' "
        "globally aggregate all tokens before a dense Linear(d_model -> d_latent) -- "
        "every z_k is a function of every point in physical space "
        "(LATENT_PDE_RESEARCH_NOTES.md's 'global support' obstruction). 'none' skips "
        "pooling: a shared per-token Linear(d_model -> c) replaces it, giving a local "
        "latent field (n_tokens, c) instead of a globally-mixed vector. 'local' is the "
        "compromise: mean-pool each --pool-window-token group into one site first, then "
        "the same shared per-site Linear -- bounded (not global) receptive field, with "
        "pool_window giving far more flexibility in matching a target d_latent than "
        "'none'. 'local_attn' is 'local' with a learned, content-adaptive attention "
        "pool (one query shared across sites) instead of a fixed uniform mean -- tests "
        "whether 'local''s reconstruction penalty comes from averaging away information "
        "or from the window size itself. 'banded' (2026-08-31, user-directed) replaces "
        "the dense Linear(d_model -> d_latent) with a LEARNED circular-band-masked map "
        "straight from the token axis to d_latent (--pool-bandwidth-nearby tokens only, "
        "in d_latent-ring units) -- see ViTAutoencoderConfig's docstring and "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 24. 'gated' (2026-09-05, "
        "user-directed) replaces 'mean''s fixed uniform average with a LEARNED "
        "weighted combination (one small Linear(d_model,1) gate + softmax over tokens) "
        "-- near-zero extra parameter cost; decode is identical to 'mean'/'cls'. "
        "'token_mlp' (2026-09-05, user-directed): a parameter-bounded alternative to "
        "naive flatten-then-dense pooling -- a per-token FFN compresses d_model -> "
        "d_model//--token-mlp-reduction BEFORE flattening, then a 2-layer MLP "
        "(--token-mlp-hidden wide) maps the flattened vector to d_latent; nothing is "
        "averaged away, only dimensionality-reduced per token first. See "
        "ViTAutoencoderConfig's docstring. Requires d_latent %% n_sites == 0 where "
        "n_sites = n_tokens // pool_window (for 'none'/'local'/'local_attn'). "
        "'local_token_mlp' (2026-09-06, Section 100, user-directed): the same "
        "per-token FFN as 'token_mlp' feeds a LEARNED circular-band-masked map "
        "(--attn-window-nearby tokens only, in n_tokens-ring units) instead of "
        "'token_mlp''s dense flatten+MLP -- requires --attn-window to be set. "
        "See ViTAutoencoderConfig's docstring.",
    )
    parser.add_argument(
        "--pool-window", type=int, default=1,
        help="--pool local only: number of contiguous tokens mean-pooled into one site "
        "before the per-site Linear head. Must divide n_tokens (= NX/patch_size). "
        "Default 1 (every token is its own site, i.e. --pool none's behavior).",
    )
    parser.add_argument(
        "--dec-pool",
        choices=[
            "mean", "cls", "none", "local", "local_attn", "banded", "gated",
            "token_mlp", "local_token_mlp",
        ],
        default=None,
        help="--encoder vit/vit_fourier_hybrid only. Override ViTAutoencoderConfig.dec_pool "
        "(default None = mirror --pool, unchanged behavior). Added 2026-09-05, user-directed: "
        "'for the decoder, I would also like to ... use global mean pooling for the vit' -- "
        "asked while --pool local for the ENCODER, i.e. genuinely asymmetric pooling. When set "
        "to a different value than --pool, the DECODER's expansion/un-pooling step uses this "
        "mode while the encoder keeps using --pool -- e.g. a hard-local encoder (bounded "
        "receptive field, intended to organize/smooth the latent) paired with a fully global "
        "mean-pooled decoder (so decode isn't ALSO bottlenecked by that same constraint). "
        "'gated'/'token_mlp' work here too, same semantics as --pool.",
    )
    parser.add_argument(
        "--token-mlp-reduction", type=int, default=None,
        help="--pool/--dec-pool token_mlp only. Override ViTAutoencoderConfig.token_mlp_reduction "
        "(default 8): the per-token FFN compresses d_model -> d_model // this value BEFORE "
        "flattening, keeping the flatten step's width (and the following MLP's parameter count) "
        "bounded regardless of d_model. Higher = fewer parameters.",
    )
    parser.add_argument(
        "--token-mlp-hidden", type=int, default=None,
        help="--pool/--dec-pool token_mlp only. Override ViTAutoencoderConfig.token_mlp_hidden "
        "(default 128): hidden width of the 2-layer MLP mapping the flattened "
        "(n_tokens*compressed_dim)-dim vector to/from d_latent.",
    )
    parser.add_argument(
        "--readout", choices=["linear", "mlp"], default=None,
        help="--encoder vit only, --pool in ('mean', 'cls') only (2026-09-03, "
        "user-directed). 'linear' (default) = the plain Linear(d_model -> d_latent) "
        "post-pool head, unchanged behavior. 'mlp' replaces it with "
        "Linear(d_model -> d_latent) -> ReLU -> Linear(d_latent -> d_latent). See "
        "ViTAutoencoderConfig.readout's docstring.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="Save a rolling mid-training checkpoint every N epochs (default 0 = off). "
        "Added 2026-08-31 after an unattended run was lost to an unexpected machine "
        "crash with no way to recover it -- overwrites a single "
        "stage1_ae_patched_{profile}{tag}_checkpoint.pt (+ the propagator equivalent, "
        "if --full-propagator) each time, not one file per checkpoint, to bound disk "
        "usage. Not a full resume mechanism (no optimizer/scheduler state saved) -- "
        "just enough to restart training from a recent point instead of from scratch.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Added 2026-09-01, user-directed ('we should be training in bfloat16 or "
        "float16, that will be way more efficient'): wraps each batch's forward pass + "
        "loss computation in torch.autocast(device_type=device.type, dtype=torch.bfloat16). "
        "bfloat16 specifically -- same exponent range as float32, so no GradScaler/loss-"
        "scaling needed, unlike float16. Confirmed working on this project's MPS device "
        "(Conv1d, scaled_dot_product_attention, Linear all run and backward correctly under "
        "MPS bfloat16 autocast). Backward/optimizer step stay float32 regardless -- only "
        "forward activations use the lower-precision dtype. Default off (unchanged "
        "behavior/numerics).",
    )
    parser.add_argument(
        "--pool-bandwidth", type=int, default=None,
        help="--pool banded only. ViTAutoencoderConfig.pool_bandwidth -- circular-band "
        "window (in d_latent-ring units, same convention as --attn-window) for the "
        "learned token-axis-to-d_latent map.",
    )
    parser.add_argument(
        "--init-ae-checkpoint", type=str, default=None,
        help="Continue Stage-1 joint training from an ALREADY-TRAINED autoencoder "
        "(added 2026-08-31, user-directed: fine-tune an existing AE jointly with an "
        "already-good propagator instead of a fresh random init). Loads the exact "
        "architecture from the checkpoint via load_autoencoder_checkpoint -- "
        "--encoder/--pool/--attn-window/etc CLI flags that would otherwise shape the "
        "AE are ignored when this is given.",
    )
    parser.add_argument(
        "--init-aux-checkpoint", type=str, default=None,
        help="Continue Stage-1 joint training using an ALREADY-TRAINED propagator (any "
        "Stage-2-format checkpoint, e.g. from train_stage2_patched.py) as the aux/full "
        "propagator, instead of building a fresh --aux-backbone one. Loaded via "
        "load_propagator_checkpoint -- AuxPropagator is architecturally identical to "
        "LatentPropagator for the same config (AuxPropagator.__init__ just converts its "
        "own config and delegates), so a Stage-2 checkpoint's state_dict loads directly, "
        "no conversion needed. --aux-backbone/--mode/--n-history/etc CLI flags are "
        "ignored when this is given (the loaded checkpoint's own config governs).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=8,
        help="--encoder vit only (2026-08-29). Token size; n_tokens = NX/patch_size "
        "(default 8 -> 32 tokens at NX=256, the canonical recipe). A smaller patch_size "
        "gives more, finer-grained tokens -- note this also shrinks attn_window's "
        "absolute physical receptive field (attn_window is measured in tokens, not "
        "physical length), unless attn_window is raised to compensate.",
    )
    parser.add_argument(
        "--n-channels", type=int, default=1,
        help="--encoder vit only (added 2026-09-23, Section 207). Override "
        "ViTAutoencoderConfig.n_channels (default 1, unchanged behavior). At n_channels=k>1, "
        "--nx is interpreted as (physical_sites * k), laid out SITE-MAJOR/CHANNEL-MINOR "
        "(e.g. k=2, x/x': [x_0, x'_0, x_1, x'_1, ...], NOT concatenated blocks) -- --dataset "
        "must already be stored this way. patch_size stays in PHYSICAL SITE units, so each "
        "token becomes patch_size*n_channels raw values (all channels of its patch_size "
        "sites) instead of patch_size -- see that field's docstring for why this keeps the "
        "positional encoding's 'adjacent token = adjacent physical site' meaning intact, "
        "unlike concatenating channels into one flat NX-dim vector (Sections 204-206's "
        "approach, found to likely be the cause of their badly-converging reconstruction).",
    )
    parser.add_argument(
        "--d-model", type=int, default=None,
        help="--encoder vit only. Override ViTAutoencoderConfig.d_model (default 96, "
        "n_heads=4/n_blocks=3/mlp_ratio=4 unchanged). Added 2026-09-02, user-directed "
        "('make the models 20 percent bigger' -- both encoder/decoder AND propagator, "
        "not just the propagator): d_model=108 gives 1011690 total encoder+decoder "
        "params vs. the d_model=96 default's 816342 (~1.24x, the closest achievable "
        "ratio to 1.2x with d_model divisible by n_heads=4, checked directly by "
        "instantiating both). Matches the precedent already used for the aux "
        "propagator's --aux-hidden 141 (also ~1.2x its own default).",
    )
    parser.add_argument(
        "--vit-n-blocks", type=int, default=None,
        help="--encoder vit only. Override ViTAutoencoderConfig.n_blocks (default 3). "
        "Added 2026-09-03, user-directed: testing more (narrower) blocks vs. fewer "
        "(wider) ones at a similar total param count -- attn_window restricts each "
        "block to local interactions, so more blocks directly extends the effective "
        "receptive field (~n_blocks * attn_window tokens) and adds more rounds of "
        "attention's non-expansive mixing, at the cost of less per-layer capacity.",
    )
    parser.add_argument(
        "--spectral-field-inner", choices=["vit", "mlp"], default="vit",
        help="--encoder spectral_field only. Selects SpectralFieldAutoencoderConfig's "
        "field-producing inner model (default 'vit', unchanged behavior). 'mlp' "
        "(added 2026-09-09, user-directed: 'we've artificially constrained the encoder "
        "quite a bit. Why don't we let the encoder be a general mlp and see if that "
        "measurably changes things. the vit might not be the right model for this') "
        "uses a plain, fully-connected KSAutoencoderMLP instead -- no attention, "
        "tokenization, or windowing at all, every output position can depend on every "
        "input position with no architectural locality bias.",
    )
    parser.add_argument(
        "--spectral-field-mlp-hidden", type=int, nargs="+", default=None,
        help="--spectral-field-inner mlp only. Override MLPAutoencoderConfig.hidden "
        "(default (512, 256, 128) at --profile full, (64, 32) at --profile smoke). "
        "Space-separated layer widths, e.g. --spectral-field-mlp-hidden 512 256 128.",
    )
    parser.add_argument(
        "--ae-fno", action="store_true",
        help="--encoder vit only. User-directed (2026-08-30): "
        "ViTAutoencoderConfig.use_fno -- inserts --ae-fno-n-layers FNO spectral-conv "
        "layers into both encoder and decoder, right after the positional encoding and "
        "before the ViT attention blocks. Unlike the propagator's --aux-backbone "
        "fno_vit (where the FFT's periodicity assumption is over the unverified latent "
        "channel index), here the token axis is genuine physical position on KS's "
        "periodic domain, so the periodicity assumption is actually correct -- see "
        "ViTAutoencoderConfig.use_fno's docstring.",
    )
    parser.add_argument(
        "--ae-fno-modes", type=int, default=None,
        help="--ae-fno only. ViTAutoencoderConfig.fno_modes (default None = every "
        "rfft mode, no truncation).",
    )
    parser.add_argument(
        "--ae-fno-n-layers", type=int, default=2,
        help="--ae-fno only. ViTAutoencoderConfig.fno_n_layers (default 2).",
    )
    parser.add_argument(
        "--aux-fno-modes", type=int, default=None,
        help="--aux-backbone fno_vit only. AuxPropagatorConfig.fno_modes (default "
        "None = every rfft mode, no truncation).",
    )
    parser.add_argument(
        "--aux-fno-n-layers", type=int, default=2,
        help="--aux-backbone fno_vit only. AuxPropagatorConfig.fno_n_layers (default 2).",
    )
    parser.add_argument(
        "--full-propagator", action="store_true",
        help="User-directed (2026-08-30): train the SAME, full-sized propagator "
        "(hidden=128, n_blocks=3, matching Stage 2's standard sizing) jointly with "
        "the AE in Phase 1, instead of the small discarded AuxPropagator. After "
        "training, the propagator is converted via aux_cfg_to_propagator_cfg and "
        "saved as its own checkpoint in the same {'prop_state_dict', 'prop_config'} "
        "format train_stage2_patched.py's --init-prop-checkpoint expects, so Phase 2 "
        "can fine-tune this same propagator for longer rollouts instead of starting "
        "from scratch. Overrides --multistep's sizing when both are given. See "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md §2.",
    )
    parser.add_argument(
        "--prop-delta-cap", type=float, default=None,
        help="--full-propagator only. AuxPropagatorConfig.delta_cap (see "
        "PropagatorConfig's docstring): tanh-saturated bound on the per-step "
        "residual, carried through to the saved propagator checkpoint's config.",
    )
    parser.add_argument(
        "--prop-delta-cap-relative", action="store_true",
        help="--full-propagator --prop-delta-cap only. Sets AuxPropagatorConfig."
        "delta_cap_relative (default False). Added 2026-09-17, Section 184, "
        "user-directed after Sections 180-183 all converged to the same sustained-"
        "growth-to-an-unphysical-scale failure regardless of the physics terms -- the "
        "one thing common to all four was delta_cap's FIXED ABSOLUTE per-step bound, "
        "not scaled to the state's own magnitude. When set, the cap becomes "
        "delta_cap * ||z_ref|| (the reference state's own norm) instead of a bare "
        "constant, so the allowed step size scales with the state's current scale. See "
        "ks_latent.models.propagator.LatentPropagator.capped_delta's docstring.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix appended to output artifact filenames, e.g. "
        "'mlp_multistep' -> stage1_ae_patched_full_mlp_multistep.pt. "
        "Use this to keep comparison runs' checkpoints from overwriting "
        "each other.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    suffix = f"_{args.tag}" if args.tag else ""

    if args.pde_distill:
        if args.aux_backbone in ("spectral_pde", "spectral_pde_raw"):
            raise ValueError(
                "--pde-distill is redundant with --aux-backbone spectral_pde/spectral_pde_raw "
                "-- the whole point of pde_head is a SECOND, distilled spectral_pde_raw head "
                "alongside a FREE/unconstrained aux; use --aux-backbone mlp (or another free "
                "backbone) with --pde-distill instead."
            )

    if args.profile == "smoke":
        NX, d_latent = 64, 8
        spectral_N_w_for_aux = None
        ks_cfg = KSConfig(L=22.0, NX=NX, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=args.seed)
        if args.encoder == "mlp":
            ae_cfg = MLPAutoencoderConfig(NX=NX, hidden=(64, 32), d_latent=d_latent)
        elif args.encoder == "masked_mlp":
            ae_cfg = MaskedMLPAutoencoderConfig(
                NX=NX, hidden=(64, 32), d_latent=d_latent, mask_window=args.ae_mask_window
            )
        elif args.encoder == "local_field":
            ae_cfg = LocalFieldAutoencoderConfig(
                NX=NX, n_sites=4, local_channels=2, site_mix_radius=1, n_site_mix_layers=1, hidden=8,
            )
            d_latent = ae_cfg.d_latent  # derived (n_sites*local_channels), not directly settable
        elif args.encoder == "vit":
            ae_cfg = ViTAutoencoderConfig(
                NX=NX, patch_size=8, d_model=16, n_heads=2, n_blocks=1, d_latent=d_latent,
                attn_window=args.attn_window, pos_encoding=args.pos_encoding,
                pool=args.pool, pool_window=args.pool_window, pool_bandwidth=args.pool_bandwidth,
                token_window=args.token_window, n_channels=args.n_channels,
                use_fno=args.ae_fno, fno_modes=args.ae_fno_modes, fno_n_layers=args.ae_fno_n_layers,
            )
        elif args.encoder == "spectral_field":
            spectral_K = args.spectral_K if args.spectral_K is not None else 4
            N_w = 16  # smoke-sized intermediate field resolution
            if args.spectral_field_inner == "mlp":
                mlp_hidden = (
                    tuple(args.spectral_field_mlp_hidden)
                    if args.spectral_field_mlp_hidden is not None
                    else (64, 32)
                )
                mlp_sub_cfg = MLPAutoencoderConfig(NX=NX, hidden=mlp_hidden, d_latent=N_w)
                ae_cfg = SpectralFieldAutoencoderConfig(mlp=mlp_sub_cfg, K=spectral_K, L=args.spectral_L)
            else:
                vit_sub_cfg = ViTAutoencoderConfig(
                    NX=NX, patch_size=NX // N_w, d_model=16, n_heads=2, n_blocks=1, d_latent=N_w,
                    pool="none", pos_encoding="linear",
                )
                ae_cfg = SpectralFieldAutoencoderConfig(vit=vit_sub_cfg, K=spectral_K, L=args.spectral_L)
            d_latent = ae_cfg.d_latent  # 2*K, overriding the generic smoke default of 8 above
            spectral_N_w_for_aux = N_w
        else:
            ae_cfg = AutoencoderConfig(
                NX=NX, patch_size=8, group_size=4, d_model=16, nhead=2, dim_ff=16,
                patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
                n_query_tokens=4, d_latent=d_latent,
            )
        aux_hidden, aux_blocks = (32, 2) if args.multistep else (16, 1)
        aux_spectral_kwargs = {}
        if args.aux_backbone == "spectral_pde":
            aux_spectral_kwargs = dict(
                spectral_K=args.spectral_K, spectral_N_w=spectral_N_w_for_aux, spectral_L=args.spectral_L,
                spectral_max_order=args.spectral_max_order, spectral_integrator=args.spectral_integrator,
                ode_substeps=args.ode_substeps, spectral_physics_prior=args.spectral_physics_prior,
                spectral_field_kind=args.spectral_field_kind, spectral_poly_degree=args.spectral_poly_degree,
                spectral_poly_max_term_order=args.spectral_poly_max_term_order,
                spectral_poly_norm_power=args.spectral_poly_norm_power,
                spectral_poly_stable_leading=args.spectral_poly_stable_leading,
                spectral_poly_no_constant=args.spectral_poly_no_constant,
                spectral_poly_exclude_nonconservative=args.spectral_poly_exclude_nonconservative,
                spectral_poly_stable_linear_terms=(
                    {2: -1.0, 4: -1.0} if args.spectral_poly_stable_w_xx_w_xxxx else None
                ),
                spectral_poly_time_deriv=args.spectral_poly_time_deriv,
                spectral_poly_time_deriv_dt_snap=(
                    args.spectral_poly_time_deriv_dt_snap
                    if args.spectral_poly_time_deriv_dt_snap is not None else 1.0
                ),
                spectral_poly_time_deriv2=args.spectral_poly_time_deriv2,
                spectral_burgers_nu_init=args.spectral_burgers_nu_init,
                spectral_burgers_beta_max=args.spectral_burgers_beta_max,
                spectral_burgers_nonlinear_nu=args.spectral_burgers_nonlinear_nu,
                spectral_burgers_kernel_instability=args.spectral_burgers_kernel_instability,
                spectral_burgers_kernel_A_max=args.spectral_burgers_kernel_A_max,
                spectral_burgers_kernel_width_init=args.spectral_burgers_kernel_width_init,
                spectral_burgers_forcing_max=args.spectral_burgers_forcing_max,
                spectral_burgers_kernel_A_fixed=args.spectral_burgers_kernel_A_fixed,
                spectral_burgers_beta_fixed=args.spectral_burgers_beta_fixed,
                spectral_burgers_kernel_mu_init=args.spectral_burgers_kernel_mu_init,
                spectral_burgers_no_forcing=args.spectral_burgers_no_forcing,
                spectral_burgers_kernel_A_init=args.spectral_burgers_kernel_A_init,
                spectral_burgers_beta_init=args.spectral_burgers_beta_init,
            )
        elif args.aux_backbone == "spectral_pde_raw":
            # Added 2026-09-10, Section 142, user-directed: "we could also
            # try to train a pde as a propagator using this current encoder
            # decoder setup right? that could work?" -- unlike "spectral_pde"
            # (requires --encoder spectral_field, d_latent==2*spectral_K),
            # "spectral_pde_raw" works on ANY encoder's raw z via its own
            # internal self-FFT truncation (see _SpectralPDERawDeltaBody's
            # docstring) -- no spectral_N_w (no target physical field
            # resolution to synthesize) and no spectral_physics_prior (raises
            # if set -- there is no "true governing equation" for an
            # arbitrary learned latent ordering the way there is for a
            # genuine w=irfft(z) field). Previously only ever constructed as
            # a SEPARATE pde_head (--pde-distill) alongside a free aux
            # propagator; this is the first time it's wired up as the
            # PRIMARY propagator itself.
            aux_spectral_kwargs = dict(
                spectral_K=args.spectral_K, spectral_L=args.spectral_L,
                spectral_max_order=args.spectral_max_order, spectral_integrator=args.spectral_integrator,
                ode_substeps=args.ode_substeps,
                spectral_field_kind=args.spectral_field_kind, spectral_poly_degree=args.spectral_poly_degree,
                spectral_poly_max_term_order=args.spectral_poly_max_term_order,
                spectral_poly_norm_power=args.spectral_poly_norm_power,
                spectral_poly_stable_leading=args.spectral_poly_stable_leading,
                spectral_poly_no_constant=args.spectral_poly_no_constant,
                spectral_poly_exclude_nonconservative=args.spectral_poly_exclude_nonconservative,
                spectral_poly_stable_linear_terms=(
                    {2: -1.0, 4: -1.0} if args.spectral_poly_stable_w_xx_w_xxxx else None
                ),
                spectral_poly_time_deriv=args.spectral_poly_time_deriv,
                spectral_poly_time_deriv_dt_snap=(
                    args.spectral_poly_time_deriv_dt_snap
                    if args.spectral_poly_time_deriv_dt_snap is not None else 1.0
                ),
                spectral_poly_time_deriv2=args.spectral_poly_time_deriv2,
                spectral_burgers_nu_init=args.spectral_burgers_nu_init,
                spectral_burgers_beta_max=args.spectral_burgers_beta_max,
                spectral_burgers_nonlinear_nu=args.spectral_burgers_nonlinear_nu,
                spectral_burgers_kernel_instability=args.spectral_burgers_kernel_instability,
                spectral_burgers_kernel_A_max=args.spectral_burgers_kernel_A_max,
                spectral_burgers_kernel_width_init=args.spectral_burgers_kernel_width_init,
                spectral_burgers_forcing_max=args.spectral_burgers_forcing_max,
                spectral_burgers_kernel_A_fixed=args.spectral_burgers_kernel_A_fixed,
                spectral_burgers_beta_fixed=args.spectral_burgers_beta_fixed,
                spectral_burgers_kernel_mu_init=args.spectral_burgers_kernel_mu_init,
                spectral_burgers_no_forcing=args.spectral_burgers_no_forcing,
                spectral_burgers_kernel_A_init=args.spectral_burgers_kernel_A_init,
                spectral_burgers_beta_init=args.spectral_burgers_beta_init,
            )
        aux_cfg = AuxPropagatorConfig(
            d_latent=d_latent, hidden=aux_hidden, n_blocks=aux_blocks, mode=args.mode,
            backbone=args.aux_backbone, n_tokens=args.aux_n_tokens, token_d_model=args.aux_token_d_model,
            token_nhead=2, token_n_layers=1, attn_window=(
                args.prop_attn_window if args.prop_attn_window is not None
                else (None if args.prop_dense else args.attn_window)
            ),
            pos_encoding=args.pos_encoding, token_window=args.aux_token_window,
            n_history=args.n_history, delta_cap=args.prop_delta_cap,
            delta_cap_relative=args.prop_delta_cap_relative,
            fno_modes=args.aux_fno_modes, fno_n_layers=args.aux_fno_n_layers,
            fourier_ifft_readout=args.prop_fourier_ifft, nonexpansive=args.prop_nonexpansive,
            masked_mlp_expand_factor=args.masked_mlp_expand_factor,
            **aux_spectral_kwargs,
        )
        smoke_kwargs = {
            "epochs": 2, "batch_size": 8, "noise_std": args.noise_std,
            "k_pred_max": 4 if args.multistep else 0,
            "k_pred_warmup_epochs": 1 if args.multistep else 0,
        }
        if args.w_spatial is not None:
            smoke_kwargs["w_spatial"] = args.w_spatial
        if args.spatial_bandwidth is not None:
            smoke_kwargs["spatial_bandwidth"] = args.spatial_bandwidth
        if args.spatial_signed:
            smoke_kwargs["spatial_signed"] = True
        if args.w_var_floor is not None:
            smoke_kwargs["w_var_floor"] = args.w_var_floor
        if args.var_floor_gamma is not None:
            smoke_kwargs["var_floor_gamma"] = args.var_floor_gamma
        if args.w_logdet is not None:
            smoke_kwargs["w_logdet"] = args.w_logdet
        if args.logdet_eps is not None:
            smoke_kwargs["logdet_eps"] = args.logdet_eps
        if args.w_logdet_physical is not None:
            smoke_kwargs["w_logdet_physical"] = args.w_logdet_physical
        if args.logdet_physical_eps is not None:
            smoke_kwargs["logdet_physical_eps"] = args.logdet_physical_eps
        if args.w_logdet_physical_rollout is not None:
            smoke_kwargs["w_logdet_physical_rollout"] = args.w_logdet_physical_rollout
        if args.logdet_physical_rollout_eps is not None:
            smoke_kwargs["logdet_physical_rollout_eps"] = args.logdet_physical_rollout_eps
        if args.w_var_physical is not None:
            smoke_kwargs["w_var_physical"] = args.w_var_physical
        if args.physics_prior_correction_warmup_epochs is not None:
            smoke_kwargs["physics_prior_correction_warmup_epochs"] = (
                args.physics_prior_correction_warmup_epochs
            )
        if args.w_pred_warmup_epochs is not None:
            smoke_kwargs["w_pred_warmup_epochs"] = args.w_pred_warmup_epochs
        if args.w_temporal_floor_z is not None:
            smoke_kwargs["w_temporal_floor_z"] = args.w_temporal_floor_z
        if args.w_temporal_floor_w is not None:
            smoke_kwargs["w_temporal_floor_w"] = args.w_temporal_floor_w
        if args.temporal_floor_lag is not None:
            smoke_kwargs["temporal_floor_lag"] = args.temporal_floor_lag
        if args.w_local_expansion_floor is not None:
            smoke_kwargs["w_local_expansion_floor"] = args.w_local_expansion_floor
        if args.local_expansion_floor_value is not None:
            smoke_kwargs["local_expansion_floor_value"] = args.local_expansion_floor_value
        if args.local_expansion_floor_n_samples is not None:
            smoke_kwargs["local_expansion_floor_n_samples"] = args.local_expansion_floor_n_samples
        if args.w_spectrum_shape is not None:
            smoke_kwargs["w_spectrum_shape"] = args.w_spectrum_shape
        if args.spectrum_shape_n_expand is not None:
            smoke_kwargs["spectrum_shape_n_expand"] = args.spectrum_shape_n_expand
        if args.spectrum_shape_expand_target is not None:
            smoke_kwargs["spectrum_shape_expand_target"] = args.spectrum_shape_expand_target
        if args.spectrum_shape_contract_floor is not None:
            smoke_kwargs["spectrum_shape_contract_floor"] = args.spectrum_shape_contract_floor
        if args.spectrum_shape_n_samples is not None:
            smoke_kwargs["spectrum_shape_n_samples"] = args.spectrum_shape_n_samples
        if args.w_jacobian_bandedness is not None:
            smoke_kwargs["w_jacobian_bandedness"] = args.w_jacobian_bandedness
        if args.jacobian_bandedness_bandwidth is not None:
            smoke_kwargs["jacobian_bandedness_bandwidth"] = args.jacobian_bandedness_bandwidth
        if args.jacobian_bandedness_n_samples is not None:
            smoke_kwargs["jacobian_bandedness_n_samples"] = args.jacobian_bandedness_n_samples
        if args.w_jacobian_diagonal_bound is not None:
            smoke_kwargs["w_jacobian_diagonal_bound"] = args.w_jacobian_diagonal_bound
        if args.jacobian_diagonal_bound_ceiling is not None:
            smoke_kwargs["jacobian_diagonal_bound_ceiling"] = args.jacobian_diagonal_bound_ceiling
        if args.jacobian_diagonal_bound_n_samples is not None:
            smoke_kwargs["jacobian_diagonal_bound_n_samples"] = args.jacobian_diagonal_bound_n_samples
        if args.w_spectrum_shape_graded is not None:
            smoke_kwargs["w_spectrum_shape_graded"] = args.w_spectrum_shape_graded
        if args.spectrum_shape_graded_reference_path is not None:
            smoke_kwargs["spectrum_shape_graded_reference_path"] = args.spectrum_shape_graded_reference_path
        if args.spectrum_shape_graded_n_samples is not None:
            smoke_kwargs["spectrum_shape_graded_n_samples"] = args.spectrum_shape_graded_n_samples
        if args.spectrum_shape_two_sided:
            smoke_kwargs["spectrum_shape_two_sided"] = True
        if args.w_spectrum_shape_multistep is not None:
            smoke_kwargs["w_spectrum_shape_multistep"] = args.w_spectrum_shape_multistep
        if args.spectrum_shape_multistep_k is not None:
            smoke_kwargs["spectrum_shape_multistep_k"] = args.spectrum_shape_multistep_k
        if args.spectrum_shape_multistep_n_samples is not None:
            smoke_kwargs["spectrum_shape_multistep_n_samples"] = args.spectrum_shape_multistep_n_samples
        if args.w_smooth is not None:
            smoke_kwargs["w_smooth"] = args.w_smooth
        if args.smooth_curvature_weight is not None:
            smoke_kwargs["smooth_curvature_weight"] = args.smooth_curvature_weight
        if args.w_lowpass is not None:
            smoke_kwargs["w_lowpass"] = args.w_lowpass
        if args.lowpass_power is not None:
            smoke_kwargs["lowpass_power"] = args.lowpass_power
        if args.w_lowpass_rollout is not None:
            smoke_kwargs["w_lowpass_rollout"] = args.w_lowpass_rollout
        if args.lowpass_rollout_power is not None:
            smoke_kwargs["lowpass_rollout_power"] = args.lowpass_rollout_power
        if args.w_z_lowpass is not None:
            smoke_kwargs["w_z_lowpass"] = args.w_z_lowpass
        if args.w_z_lowpass_rollout is not None:
            smoke_kwargs["w_z_lowpass_rollout"] = args.w_z_lowpass_rollout
        if args.z_lowpass_power is not None:
            smoke_kwargs["z_lowpass_power"] = args.z_lowpass_power
        if args.z_lowpass_rollout_power is not None:
            smoke_kwargs["z_lowpass_rollout_power"] = args.z_lowpass_rollout_power
        if args.z_lowpass_K is not None:
            smoke_kwargs["z_lowpass_K"] = args.z_lowpass_K
        if args.z_lowpass_L is not None:
            smoke_kwargs["z_lowpass_L"] = args.z_lowpass_L
        if args.w_channel_mean is not None:
            smoke_kwargs["w_channel_mean"] = args.w_channel_mean
        if args.w_shape_floor is not None:
            smoke_kwargs["w_shape_floor"] = args.w_shape_floor
        if args.w_pde_distill is not None:
            smoke_kwargs["w_pde_distill"] = args.w_pde_distill
        if args.w_pde_coeff_l1 is not None:
            smoke_kwargs["w_pde_coeff_l1"] = args.w_pde_coeff_l1
        if args.pde_coeff_l1_linear_only:
            smoke_kwargs["pde_coeff_l1_linear_only"] = True
        if args.w_pde_distill_real is not None:
            smoke_kwargs["w_pde_distill_real"] = args.w_pde_distill_real
        if args.w_pde_distill_real_rollout is not None:
            smoke_kwargs["w_pde_distill_real_rollout"] = args.w_pde_distill_real_rollout
            smoke_kwargs["pde_distill_real_rollout_k"] = args.pde_distill_real_rollout_k
            smoke_kwargs["pde_distill_real_rollout_warmup_epochs"] = (
                args.pde_distill_real_rollout_warmup_epochs
                if args.pde_distill_real_rollout_warmup_epochs is not None
                else max(1, round(0.3 * 2))
            )
        if args.w_pde_nonlinear_l2 is not None:
            smoke_kwargs["w_pde_nonlinear_l2"] = args.w_pde_nonlinear_l2
        if args.w_pde_spectrum_shape is not None:
            smoke_kwargs["w_pde_spectrum_shape"] = args.w_pde_spectrum_shape
        if args.pde_spectrum_shape_n_expand is not None:
            smoke_kwargs["pde_spectrum_shape_n_expand"] = args.pde_spectrum_shape_n_expand
        if args.pde_spectrum_shape_expand_target is not None:
            smoke_kwargs["pde_spectrum_shape_expand_target"] = args.pde_spectrum_shape_expand_target
        if args.pde_spectrum_shape_contract_floor is not None:
            smoke_kwargs["pde_spectrum_shape_contract_floor"] = args.pde_spectrum_shape_contract_floor
        if args.pde_spectrum_shape_n_samples is not None:
            smoke_kwargs["pde_spectrum_shape_n_samples"] = args.pde_spectrum_shape_n_samples
        if args.pde_spectrum_shape_two_sided:
            smoke_kwargs["pde_spectrum_shape_two_sided"] = True
        if args.w_pde_spectrum_shape_multistep is not None:
            smoke_kwargs["w_pde_spectrum_shape_multistep"] = args.w_pde_spectrum_shape_multistep
        if args.pde_spectrum_shape_multistep_k is not None:
            smoke_kwargs["pde_spectrum_shape_multistep_k"] = args.pde_spectrum_shape_multistep_k
        if args.pde_spectrum_shape_multistep_n_samples is not None:
            smoke_kwargs["pde_spectrum_shape_multistep_n_samples"] = args.pde_spectrum_shape_multistep_n_samples
        if args.w_pde_spectrum_shape_self is not None:
            smoke_kwargs["w_pde_spectrum_shape_self"] = args.w_pde_spectrum_shape_self
        if args.pde_spectrum_shape_self_rollout_k is not None:
            smoke_kwargs["pde_spectrum_shape_self_rollout_k"] = args.pde_spectrum_shape_self_rollout_k
        if args.pde_spectrum_shape_self_n_samples is not None:
            smoke_kwargs["pde_spectrum_shape_self_n_samples"] = args.pde_spectrum_shape_self_n_samples
        if args.w_pde_spectrum_shape_multistep_self is not None:
            smoke_kwargs["w_pde_spectrum_shape_multistep_self"] = args.w_pde_spectrum_shape_multistep_self
        if args.w_pde_mean_conservation is not None:
            smoke_kwargs["w_pde_mean_conservation"] = args.w_pde_mean_conservation
        if args.w_pde_energy_floor is not None:
            smoke_kwargs["w_pde_energy_floor"] = args.w_pde_energy_floor
            smoke_kwargs["pde_energy_floor_gamma"] = args.pde_energy_floor_gamma
            smoke_kwargs["pde_energy_floor_rollout_k"] = args.pde_energy_floor_rollout_k
            smoke_kwargs["pde_energy_floor_warmup_epochs"] = (
                args.pde_energy_floor_warmup_epochs
                if args.pde_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * 2))
            )
        if args.w_prop_energy_floor is not None:
            smoke_kwargs["w_prop_energy_floor"] = args.w_prop_energy_floor
            smoke_kwargs["prop_energy_floor_gamma"] = args.prop_energy_floor_gamma
            smoke_kwargs["prop_energy_floor_rollout_k"] = args.prop_energy_floor_rollout_k
            smoke_kwargs["prop_energy_floor_warmup_epochs"] = (
                args.prop_energy_floor_warmup_epochs
                if args.prop_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * 2))
            )
        if args.stable_linear_lr_factor is not None:
            smoke_kwargs["stable_linear_lr_factor"] = args.stable_linear_lr_factor
        if args.w_kernel_unstable_floor is not None:
            smoke_kwargs["w_kernel_unstable_floor"] = args.w_kernel_unstable_floor
            smoke_kwargs["kernel_unstable_target_modes"] = args.kernel_unstable_target_modes
            smoke_kwargs["kernel_unstable_margin"] = args.kernel_unstable_margin
        if args.pde_distill_real_mutual:
            smoke_kwargs["pde_distill_real_detach"] = False
        if args.pde_mutual:
            smoke_kwargs["pde_distill_detach_target"] = False
        train_cfg = Stage1TrainingConfig(**smoke_kwargs)
        device = torch.device("cpu")
        dataset_path = ARTIFACTS_DIR / "datasets" / "smoke_stage1_trajectories.h5"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate_trajectory_dataset(
            ks_cfg, dataset_path, n_train=4, n_val=2, trajectory_time=5.0
        )
    else:
        NX = args.nx if args.nx is not None else 256
        d_latent = args.d_latent if args.d_latent is not None else 44
        spectral_N_w_for_aux = None
        if args.encoder == "mlp":
            ae_cfg = MLPAutoencoderConfig(NX=NX, d_latent=d_latent)
        elif args.encoder == "masked_mlp":
            ae_cfg = MaskedMLPAutoencoderConfig(NX=NX, d_latent=d_latent, mask_window=args.ae_mask_window)
        elif args.encoder == "local_field":
            ae_cfg = LocalFieldAutoencoderConfig(
                NX=NX, n_sites=args.local_field_n_sites, local_channels=args.local_field_channels,
                site_mix_radius=args.local_field_mix_radius, n_site_mix_layers=args.local_field_n_mix_layers,
                hidden=args.local_field_hidden,
            )
            d_latent = ae_cfg.d_latent  # derived (n_sites*local_channels), not directly settable
        elif args.encoder == "vit":
            vit_kwargs = {}
            if args.d_model is not None:
                vit_kwargs["d_model"] = args.d_model
            if args.readout is not None:
                vit_kwargs["readout"] = args.readout
            if args.vit_n_blocks is not None:
                vit_kwargs["n_blocks"] = args.vit_n_blocks
            if args.token_mlp_reduction is not None:
                vit_kwargs["token_mlp_reduction"] = args.token_mlp_reduction
            if args.token_mlp_hidden is not None:
                vit_kwargs["token_mlp_hidden"] = args.token_mlp_hidden
            ae_cfg = ViTAutoencoderConfig(
                NX=NX, patch_size=args.patch_size, d_latent=d_latent, attn_window=args.attn_window,
                pos_encoding=args.pos_encoding, pool=args.pool, pool_window=args.pool_window,
                pool_bandwidth=args.pool_bandwidth, dec_pool=args.dec_pool,
                token_window=args.token_window, n_channels=args.n_channels,
                use_fno=args.ae_fno, fno_modes=args.ae_fno_modes, fno_n_layers=args.ae_fno_n_layers,
                **vit_kwargs,
            )
        elif args.encoder == "spectral_field":
            if args.spectral_K is None:
                raise ValueError("--encoder spectral_field requires --spectral-K to be set")
            if args.spectral_field_inner == "mlp":
                mlp_hidden = (
                    tuple(args.spectral_field_mlp_hidden)
                    if args.spectral_field_mlp_hidden is not None
                    else (512, 256, 128)
                )
                mlp_sub_cfg = MLPAutoencoderConfig(NX=NX, hidden=mlp_hidden, d_latent=d_latent)
                ae_cfg = SpectralFieldAutoencoderConfig(mlp=mlp_sub_cfg, K=args.spectral_K, L=args.spectral_L)
            else:
                if NX % d_latent != 0:
                    raise ValueError(
                        f"--encoder spectral_field requires NX={NX} divisible by --d-latent={d_latent} "
                        f"(reused as N_w, the intermediate field w's physical resolution, with "
                        f"patch_size = NX // N_w)"
                    )
                vit_sub_cfg = ViTAutoencoderConfig(
                    NX=NX, patch_size=NX // d_latent, d_model=(args.d_model if args.d_model is not None else 96),
                    n_heads=4, n_blocks=(args.vit_n_blocks if args.vit_n_blocks is not None else 3),
                    d_latent=d_latent, pool="none", pos_encoding=args.pos_encoding,
                )
                ae_cfg = SpectralFieldAutoencoderConfig(vit=vit_sub_cfg, K=args.spectral_K, L=args.spectral_L)
            spectral_N_w_for_aux = d_latent  # capture N_w before the reassignment just below
            d_latent = ae_cfg.d_latent  # 2*K, overriding N_w for the generic aux_cfg construction below
        elif args.encoder == "fourier_mlp":
            fourier_mlp_ae_kwargs = {}
            if args.fourier_mlp_hidden is not None:
                fourier_mlp_ae_kwargs["hidden"] = args.fourier_mlp_hidden
            if args.fourier_mlp_blocks is not None:
                fourier_mlp_ae_kwargs["n_blocks"] = args.fourier_mlp_blocks
            ae_cfg = FourierMLPAutoencoderConfig(
                NX=NX, d_latent=d_latent,
                enc_fno_modes=args.fourier_mlp_enc_fno_modes,
                dec_fno_modes=args.fourier_mlp_dec_fno_modes,
                attn_window=args.attn_window,
                dec_use_ifft=args.dec_use_ifft,
                fourier_ifft_readout=args.fourier_ifft_readout,
                dec_attn_window=args.dec_attn_window,
                nonexpansive=args.nonexpansive,
                **fourier_mlp_ae_kwargs,
            )
        elif args.encoder == "vit_fourier_hybrid":
            vit_kwargs = {}
            if args.d_model is not None:
                vit_kwargs["d_model"] = args.d_model
            if args.readout is not None:
                vit_kwargs["readout"] = args.readout
            if args.vit_n_blocks is not None:
                vit_kwargs["n_blocks"] = args.vit_n_blocks
            if args.token_mlp_reduction is not None:
                vit_kwargs["token_mlp_reduction"] = args.token_mlp_reduction
            if args.token_mlp_hidden is not None:
                vit_kwargs["token_mlp_hidden"] = args.token_mlp_hidden
            vit_sub_cfg = ViTAutoencoderConfig(
                NX=NX, patch_size=args.patch_size, d_latent=d_latent, attn_window=args.attn_window,
                pos_encoding=args.pos_encoding, pool=args.pool, pool_window=args.pool_window,
                pool_bandwidth=args.pool_bandwidth, dec_pool=args.dec_pool,
                token_window=args.token_window,
                use_fno=args.ae_fno, fno_modes=args.ae_fno_modes, fno_n_layers=args.ae_fno_n_layers,
                **vit_kwargs,
            )
            hybrid_kwargs = {}
            if args.vit_fourier_fourier_hidden is not None:
                hybrid_kwargs["fourier_hidden"] = args.vit_fourier_fourier_hidden
            if args.vit_fourier_fourier_blocks is not None:
                hybrid_kwargs["fourier_blocks"] = args.vit_fourier_fourier_blocks
            if args.vit_fourier_kind is not None:
                hybrid_kwargs["fourier_kind"] = args.vit_fourier_kind
            if args.vit_fourier_enc_out_modes is not None:
                hybrid_kwargs["enc_out_modes"] = args.vit_fourier_enc_out_modes
            if args.vit_fourier_dec_out_modes is not None:
                hybrid_kwargs["dec_out_modes"] = args.vit_fourier_dec_out_modes
            ae_cfg = ViTFourierHybridAutoencoderConfig(
                vit=vit_sub_cfg,
                enc_fno_modes=args.fourier_mlp_enc_fno_modes,
                dec_fno_modes=args.fourier_mlp_dec_fno_modes,
                **hybrid_kwargs,
            )
        else:
            ae_cfg = AutoencoderConfig()  # canonical NX=256 (changed 2026-08-29, see docs/RESULTS.md), d_latent=44
        # --multistep: size the auxiliary propagator like the real Stage-2
        # PropagatorConfig (hidden=128, n_blocks=3) instead of the brief's
        # tiny ~22k-param one -- see this script's docstring and
        # Stage1TrainingConfig.k_pred_max's docstring. --full-propagator
        # (2026-08-30, user-directed) forces the same Stage-2-standard sizing
        # regardless of --multistep, since the whole point is to save this
        # exact propagator for Phase 2 to fine-tune -- see --full-propagator's
        # help text and docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md §2.
        aux_hidden, aux_blocks = (128, 3) if (args.multistep or args.full_propagator) else (64, 2)
        if args.aux_hidden is not None:
            aux_hidden = args.aux_hidden
        if args.aux_blocks is not None:
            aux_blocks = args.aux_blocks
        aux_spectral_kwargs = {}
        if args.aux_backbone == "spectral_pde":
            aux_spectral_kwargs = dict(
                spectral_K=args.spectral_K, spectral_N_w=spectral_N_w_for_aux, spectral_L=args.spectral_L,
                spectral_max_order=args.spectral_max_order, spectral_integrator=args.spectral_integrator,
                ode_substeps=args.ode_substeps, spectral_physics_prior=args.spectral_physics_prior,
                spectral_field_kind=args.spectral_field_kind, spectral_poly_degree=args.spectral_poly_degree,
                spectral_poly_max_term_order=args.spectral_poly_max_term_order,
                spectral_poly_norm_power=args.spectral_poly_norm_power,
                spectral_poly_stable_leading=args.spectral_poly_stable_leading,
                spectral_poly_no_constant=args.spectral_poly_no_constant,
                spectral_poly_exclude_nonconservative=args.spectral_poly_exclude_nonconservative,
                spectral_poly_stable_linear_terms=(
                    {2: -1.0, 4: -1.0} if args.spectral_poly_stable_w_xx_w_xxxx else None
                ),
                spectral_poly_time_deriv=args.spectral_poly_time_deriv,
                spectral_poly_time_deriv_dt_snap=(
                    args.spectral_poly_time_deriv_dt_snap
                    if args.spectral_poly_time_deriv_dt_snap is not None else 1.0
                ),
                spectral_poly_time_deriv2=args.spectral_poly_time_deriv2,
                spectral_burgers_nu_init=args.spectral_burgers_nu_init,
                spectral_burgers_beta_max=args.spectral_burgers_beta_max,
                spectral_burgers_nonlinear_nu=args.spectral_burgers_nonlinear_nu,
                spectral_burgers_kernel_instability=args.spectral_burgers_kernel_instability,
                spectral_burgers_kernel_A_max=args.spectral_burgers_kernel_A_max,
                spectral_burgers_kernel_width_init=args.spectral_burgers_kernel_width_init,
                spectral_burgers_forcing_max=args.spectral_burgers_forcing_max,
                spectral_burgers_kernel_A_fixed=args.spectral_burgers_kernel_A_fixed,
                spectral_burgers_beta_fixed=args.spectral_burgers_beta_fixed,
                spectral_burgers_kernel_mu_init=args.spectral_burgers_kernel_mu_init,
                spectral_burgers_no_forcing=args.spectral_burgers_no_forcing,
                spectral_burgers_kernel_A_init=args.spectral_burgers_kernel_A_init,
                spectral_burgers_beta_init=args.spectral_burgers_beta_init,
            )
        elif args.aux_backbone == "spectral_pde_raw":
            # Added 2026-09-10, Section 142, user-directed: "we could also
            # try to train a pde as a propagator using this current encoder
            # decoder setup right? that could work?" -- unlike "spectral_pde"
            # (requires --encoder spectral_field, d_latent==2*spectral_K),
            # "spectral_pde_raw" works on ANY encoder's raw z via its own
            # internal self-FFT truncation (see _SpectralPDERawDeltaBody's
            # docstring) -- no spectral_N_w (no target physical field
            # resolution to synthesize) and no spectral_physics_prior (raises
            # if set -- there is no "true governing equation" for an
            # arbitrary learned latent ordering the way there is for a
            # genuine w=irfft(z) field). Previously only ever constructed as
            # a SEPARATE pde_head (--pde-distill) alongside a free aux
            # propagator; this is the first time it's wired up as the
            # PRIMARY propagator itself.
            aux_spectral_kwargs = dict(
                spectral_K=args.spectral_K, spectral_L=args.spectral_L,
                spectral_max_order=args.spectral_max_order, spectral_integrator=args.spectral_integrator,
                ode_substeps=args.ode_substeps,
                spectral_field_kind=args.spectral_field_kind, spectral_poly_degree=args.spectral_poly_degree,
                spectral_poly_max_term_order=args.spectral_poly_max_term_order,
                spectral_poly_norm_power=args.spectral_poly_norm_power,
                spectral_poly_stable_leading=args.spectral_poly_stable_leading,
                spectral_poly_no_constant=args.spectral_poly_no_constant,
                spectral_poly_exclude_nonconservative=args.spectral_poly_exclude_nonconservative,
                spectral_poly_stable_linear_terms=(
                    {2: -1.0, 4: -1.0} if args.spectral_poly_stable_w_xx_w_xxxx else None
                ),
                spectral_poly_time_deriv=args.spectral_poly_time_deriv,
                spectral_poly_time_deriv_dt_snap=(
                    args.spectral_poly_time_deriv_dt_snap
                    if args.spectral_poly_time_deriv_dt_snap is not None else 1.0
                ),
                spectral_poly_time_deriv2=args.spectral_poly_time_deriv2,
                spectral_burgers_nu_init=args.spectral_burgers_nu_init,
                spectral_burgers_beta_max=args.spectral_burgers_beta_max,
                spectral_burgers_nonlinear_nu=args.spectral_burgers_nonlinear_nu,
                spectral_burgers_kernel_instability=args.spectral_burgers_kernel_instability,
                spectral_burgers_kernel_A_max=args.spectral_burgers_kernel_A_max,
                spectral_burgers_kernel_width_init=args.spectral_burgers_kernel_width_init,
                spectral_burgers_forcing_max=args.spectral_burgers_forcing_max,
                spectral_burgers_kernel_A_fixed=args.spectral_burgers_kernel_A_fixed,
                spectral_burgers_beta_fixed=args.spectral_burgers_beta_fixed,
                spectral_burgers_kernel_mu_init=args.spectral_burgers_kernel_mu_init,
                spectral_burgers_no_forcing=args.spectral_burgers_no_forcing,
                spectral_burgers_kernel_A_init=args.spectral_burgers_kernel_A_init,
                spectral_burgers_beta_init=args.spectral_burgers_beta_init,
            )
        aux_cfg = AuxPropagatorConfig(
            d_latent=d_latent, hidden=aux_hidden, n_blocks=aux_blocks, mode=args.mode,
            backbone=args.aux_backbone, n_tokens=args.aux_n_tokens, token_d_model=args.aux_token_d_model,
            token_nhead=2, token_n_layers=2, attn_window=(
                args.prop_attn_window if args.prop_attn_window is not None
                else (None if args.prop_dense else args.attn_window)
            ),
            pos_encoding=args.pos_encoding, token_window=args.aux_token_window,
            n_history=args.n_history, delta_cap=args.prop_delta_cap,
            delta_cap_relative=args.prop_delta_cap_relative,
            fno_modes=args.aux_fno_modes, fno_n_layers=args.aux_fno_n_layers,
            fourier_ifft_readout=args.prop_fourier_ifft, nonexpansive=args.prop_nonexpansive,
            masked_mlp_expand_factor=args.masked_mlp_expand_factor,
            **aux_spectral_kwargs,
        )
        k_pred_max = args.k_pred_max if args.k_pred_max is not None else (8 if args.multistep else 0)
        stage1_kwargs = {
            "epochs": args.epochs,
            "noise_std": args.noise_std,
            "k_pred_max": k_pred_max,
            "k_pred_warmup_epochs": max(1, round(0.3 * args.epochs)) if k_pred_max > 0 else 0,
        }
        if args.w_pred is not None:
            stage1_kwargs["w_pred"] = args.w_pred
        if args.w_var is not None:
            stage1_kwargs["w_var"] = args.w_var
        if args.w_decorr is not None:
            stage1_kwargs["w_decorr"] = args.w_decorr
        if args.w_spatial is not None:
            stage1_kwargs["w_spatial"] = args.w_spatial
        if args.spatial_bandwidth is not None:
            stage1_kwargs["spatial_bandwidth"] = args.spatial_bandwidth
        if args.spatial_signed:
            stage1_kwargs["spatial_signed"] = True
        if args.w_var_floor is not None:
            stage1_kwargs["w_var_floor"] = args.w_var_floor
        if args.var_floor_gamma is not None:
            stage1_kwargs["var_floor_gamma"] = args.var_floor_gamma
        if args.w_logdet is not None:
            stage1_kwargs["w_logdet"] = args.w_logdet
        if args.logdet_eps is not None:
            stage1_kwargs["logdet_eps"] = args.logdet_eps
        if args.w_logdet_physical is not None:
            stage1_kwargs["w_logdet_physical"] = args.w_logdet_physical
        if args.logdet_physical_eps is not None:
            stage1_kwargs["logdet_physical_eps"] = args.logdet_physical_eps
        if args.w_logdet_physical_rollout is not None:
            stage1_kwargs["w_logdet_physical_rollout"] = args.w_logdet_physical_rollout
        if args.logdet_physical_rollout_eps is not None:
            stage1_kwargs["logdet_physical_rollout_eps"] = args.logdet_physical_rollout_eps
        if args.w_var_physical is not None:
            stage1_kwargs["w_var_physical"] = args.w_var_physical
        if args.physics_prior_correction_warmup_epochs is not None:
            stage1_kwargs["physics_prior_correction_warmup_epochs"] = (
                args.physics_prior_correction_warmup_epochs
            )
        if args.w_pred_warmup_epochs is not None:
            stage1_kwargs["w_pred_warmup_epochs"] = args.w_pred_warmup_epochs
        if args.w_temporal_floor_z is not None:
            stage1_kwargs["w_temporal_floor_z"] = args.w_temporal_floor_z
        if args.w_temporal_floor_w is not None:
            stage1_kwargs["w_temporal_floor_w"] = args.w_temporal_floor_w
        if args.temporal_floor_lag is not None:
            stage1_kwargs["temporal_floor_lag"] = args.temporal_floor_lag
        if args.w_local_expansion_floor is not None:
            stage1_kwargs["w_local_expansion_floor"] = args.w_local_expansion_floor
        if args.local_expansion_floor_value is not None:
            stage1_kwargs["local_expansion_floor_value"] = args.local_expansion_floor_value
        if args.local_expansion_floor_n_samples is not None:
            stage1_kwargs["local_expansion_floor_n_samples"] = args.local_expansion_floor_n_samples
        if args.w_spectrum_shape is not None:
            stage1_kwargs["w_spectrum_shape"] = args.w_spectrum_shape
        if args.spectrum_shape_n_expand is not None:
            stage1_kwargs["spectrum_shape_n_expand"] = args.spectrum_shape_n_expand
        if args.spectrum_shape_expand_target is not None:
            stage1_kwargs["spectrum_shape_expand_target"] = args.spectrum_shape_expand_target
        if args.spectrum_shape_contract_floor is not None:
            stage1_kwargs["spectrum_shape_contract_floor"] = args.spectrum_shape_contract_floor
        if args.spectrum_shape_n_samples is not None:
            stage1_kwargs["spectrum_shape_n_samples"] = args.spectrum_shape_n_samples
        if args.w_jacobian_bandedness is not None:
            stage1_kwargs["w_jacobian_bandedness"] = args.w_jacobian_bandedness
        if args.jacobian_bandedness_bandwidth is not None:
            stage1_kwargs["jacobian_bandedness_bandwidth"] = args.jacobian_bandedness_bandwidth
        if args.jacobian_bandedness_n_samples is not None:
            stage1_kwargs["jacobian_bandedness_n_samples"] = args.jacobian_bandedness_n_samples
        if args.w_jacobian_diagonal_bound is not None:
            stage1_kwargs["w_jacobian_diagonal_bound"] = args.w_jacobian_diagonal_bound
        if args.jacobian_diagonal_bound_ceiling is not None:
            stage1_kwargs["jacobian_diagonal_bound_ceiling"] = args.jacobian_diagonal_bound_ceiling
        if args.jacobian_diagonal_bound_n_samples is not None:
            stage1_kwargs["jacobian_diagonal_bound_n_samples"] = args.jacobian_diagonal_bound_n_samples
        if args.w_spectrum_shape_graded is not None:
            stage1_kwargs["w_spectrum_shape_graded"] = args.w_spectrum_shape_graded
        if args.spectrum_shape_graded_reference_path is not None:
            stage1_kwargs["spectrum_shape_graded_reference_path"] = args.spectrum_shape_graded_reference_path
        if args.spectrum_shape_graded_n_samples is not None:
            stage1_kwargs["spectrum_shape_graded_n_samples"] = args.spectrum_shape_graded_n_samples
        if args.spectrum_shape_two_sided:
            stage1_kwargs["spectrum_shape_two_sided"] = True
        if args.w_spectrum_shape_multistep is not None:
            stage1_kwargs["w_spectrum_shape_multistep"] = args.w_spectrum_shape_multistep
        if args.spectrum_shape_multistep_k is not None:
            stage1_kwargs["spectrum_shape_multistep_k"] = args.spectrum_shape_multistep_k
        if args.spectrum_shape_multistep_n_samples is not None:
            stage1_kwargs["spectrum_shape_multistep_n_samples"] = args.spectrum_shape_multistep_n_samples
        if args.w_smooth is not None:
            stage1_kwargs["w_smooth"] = args.w_smooth
        if args.smooth_curvature_weight is not None:
            stage1_kwargs["smooth_curvature_weight"] = args.smooth_curvature_weight
        if args.w_lowpass is not None:
            stage1_kwargs["w_lowpass"] = args.w_lowpass
        if args.lowpass_power is not None:
            stage1_kwargs["lowpass_power"] = args.lowpass_power
        if args.w_lowpass_rollout is not None:
            stage1_kwargs["w_lowpass_rollout"] = args.w_lowpass_rollout
        if args.lowpass_rollout_power is not None:
            stage1_kwargs["lowpass_rollout_power"] = args.lowpass_rollout_power
        if args.w_z_lowpass is not None:
            stage1_kwargs["w_z_lowpass"] = args.w_z_lowpass
        if args.w_z_lowpass_rollout is not None:
            stage1_kwargs["w_z_lowpass_rollout"] = args.w_z_lowpass_rollout
        if args.z_lowpass_power is not None:
            stage1_kwargs["z_lowpass_power"] = args.z_lowpass_power
        if args.z_lowpass_rollout_power is not None:
            stage1_kwargs["z_lowpass_rollout_power"] = args.z_lowpass_rollout_power
        if args.z_lowpass_K is not None:
            stage1_kwargs["z_lowpass_K"] = args.z_lowpass_K
        if args.z_lowpass_L is not None:
            stage1_kwargs["z_lowpass_L"] = args.z_lowpass_L
        if args.w_channel_mean is not None:
            stage1_kwargs["w_channel_mean"] = args.w_channel_mean
        if args.w_shape_floor is not None:
            stage1_kwargs["w_shape_floor"] = args.w_shape_floor
        if args.w_pde_distill is not None:
            stage1_kwargs["w_pde_distill"] = args.w_pde_distill
        if args.w_pde_coeff_l1 is not None:
            stage1_kwargs["w_pde_coeff_l1"] = args.w_pde_coeff_l1
        if args.pde_coeff_l1_linear_only:
            stage1_kwargs["pde_coeff_l1_linear_only"] = True
        if args.w_pde_distill_real is not None:
            stage1_kwargs["w_pde_distill_real"] = args.w_pde_distill_real
        if args.w_pde_distill_real_rollout is not None:
            stage1_kwargs["w_pde_distill_real_rollout"] = args.w_pde_distill_real_rollout
            stage1_kwargs["pde_distill_real_rollout_k"] = args.pde_distill_real_rollout_k
            stage1_kwargs["pde_distill_real_rollout_warmup_epochs"] = (
                args.pde_distill_real_rollout_warmup_epochs
                if args.pde_distill_real_rollout_warmup_epochs is not None
                else max(1, round(0.3 * args.epochs))
            )
        if args.w_pde_nonlinear_l2 is not None:
            stage1_kwargs["w_pde_nonlinear_l2"] = args.w_pde_nonlinear_l2
        if args.w_pde_spectrum_shape is not None:
            stage1_kwargs["w_pde_spectrum_shape"] = args.w_pde_spectrum_shape
        if args.pde_spectrum_shape_n_expand is not None:
            stage1_kwargs["pde_spectrum_shape_n_expand"] = args.pde_spectrum_shape_n_expand
        if args.pde_spectrum_shape_expand_target is not None:
            stage1_kwargs["pde_spectrum_shape_expand_target"] = args.pde_spectrum_shape_expand_target
        if args.pde_spectrum_shape_contract_floor is not None:
            stage1_kwargs["pde_spectrum_shape_contract_floor"] = args.pde_spectrum_shape_contract_floor
        if args.pde_spectrum_shape_n_samples is not None:
            stage1_kwargs["pde_spectrum_shape_n_samples"] = args.pde_spectrum_shape_n_samples
        if args.pde_spectrum_shape_two_sided:
            stage1_kwargs["pde_spectrum_shape_two_sided"] = True
        if args.w_pde_spectrum_shape_multistep is not None:
            stage1_kwargs["w_pde_spectrum_shape_multistep"] = args.w_pde_spectrum_shape_multistep
        if args.pde_spectrum_shape_multistep_k is not None:
            stage1_kwargs["pde_spectrum_shape_multistep_k"] = args.pde_spectrum_shape_multistep_k
        if args.pde_spectrum_shape_multistep_n_samples is not None:
            stage1_kwargs["pde_spectrum_shape_multistep_n_samples"] = args.pde_spectrum_shape_multistep_n_samples
        if args.w_pde_spectrum_shape_self is not None:
            stage1_kwargs["w_pde_spectrum_shape_self"] = args.w_pde_spectrum_shape_self
        if args.pde_spectrum_shape_self_rollout_k is not None:
            stage1_kwargs["pde_spectrum_shape_self_rollout_k"] = args.pde_spectrum_shape_self_rollout_k
        if args.pde_spectrum_shape_self_n_samples is not None:
            stage1_kwargs["pde_spectrum_shape_self_n_samples"] = args.pde_spectrum_shape_self_n_samples
        if args.w_pde_spectrum_shape_multistep_self is not None:
            stage1_kwargs["w_pde_spectrum_shape_multistep_self"] = args.w_pde_spectrum_shape_multistep_self
        if args.w_pde_mean_conservation is not None:
            stage1_kwargs["w_pde_mean_conservation"] = args.w_pde_mean_conservation
        if args.w_pde_energy_floor is not None:
            stage1_kwargs["w_pde_energy_floor"] = args.w_pde_energy_floor
            stage1_kwargs["pde_energy_floor_gamma"] = args.pde_energy_floor_gamma
            stage1_kwargs["pde_energy_floor_rollout_k"] = args.pde_energy_floor_rollout_k
            stage1_kwargs["pde_energy_floor_warmup_epochs"] = (
                args.pde_energy_floor_warmup_epochs
                if args.pde_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * args.epochs))
            )
        if args.w_prop_energy_floor is not None:
            stage1_kwargs["w_prop_energy_floor"] = args.w_prop_energy_floor
            stage1_kwargs["prop_energy_floor_gamma"] = args.prop_energy_floor_gamma
            stage1_kwargs["prop_energy_floor_rollout_k"] = args.prop_energy_floor_rollout_k
            stage1_kwargs["prop_energy_floor_warmup_epochs"] = (
                args.prop_energy_floor_warmup_epochs
                if args.prop_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * args.epochs))
            )
        if args.stable_linear_lr_factor is not None:
            stage1_kwargs["stable_linear_lr_factor"] = args.stable_linear_lr_factor
        if args.w_kernel_unstable_floor is not None:
            stage1_kwargs["w_kernel_unstable_floor"] = args.w_kernel_unstable_floor
            stage1_kwargs["kernel_unstable_target_modes"] = args.kernel_unstable_target_modes
            stage1_kwargs["kernel_unstable_margin"] = args.kernel_unstable_margin
        if args.pde_distill_real_mutual:
            stage1_kwargs["pde_distill_real_detach"] = False
        if args.pde_mutual:
            stage1_kwargs["pde_distill_detach_target"] = False
        if args.w_var_end is not None:
            stage1_kwargs["w_var_end"] = args.w_var_end
        if args.w_logdet_end is not None:
            stage1_kwargs["w_logdet_end"] = args.w_logdet_end
        if args.w_spatial_end is not None:
            stage1_kwargs["w_spatial_end"] = args.w_spatial_end
        train_cfg = Stage1TrainingConfig(**stage1_kwargs)
        device = get_device("mps")
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            snapshot_every = round(args.dt_snap / 0.05)
            ks_cfg = KSConfig(
                L=100.0, NX=NX, dt=0.05, snapshot_every=snapshot_every,
                spinup_time=500.0, seed=args.seed,
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            generate_trajectory_dataset(
                ks_cfg, dataset_path, n_train=50, n_val=10, trajectory_time=250.0
            )

    with h5py.File(dataset_path, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]

    if args.encoder == "mlp":
        ae = KSAutoencoderMLP(ae_cfg)
    elif args.encoder == "masked_mlp":
        ae = KSAutoencoderMaskedMLP(ae_cfg)
    elif args.encoder == "local_field":
        ae = KSAutoencoderLocalField(ae_cfg)
    elif args.encoder == "vit":
        ae = KSAutoencoderViT(ae_cfg)
    elif args.encoder == "spectral_field":
        ae = KSAutoencoderSpectralField(ae_cfg)
    elif args.encoder == "fourier_mlp":
        ae = KSAutoencoderFourierMLP(ae_cfg)
    elif args.encoder == "vit_fourier_hybrid":
        ae = KSAutoencoderViTFourierHybrid(ae_cfg)
    else:
        ae = KSAutoencoderPatched(ae_cfg)
    aux = AuxPropagator(aux_cfg)
    aux_prop_cfg_for_save = aux_cfg_to_propagator_cfg(aux_cfg)

    pde_head = None
    pde_head_prop_cfg_for_save = None
    if args.pde_distill:
        pde_K = args.pde_K if args.pde_K is not None else d_latent // 2 + 1
        pde_L = args.pde_L if args.pde_L is not None else float(d_latent)
        pde_head_cfg = AuxPropagatorConfig(
            d_latent=d_latent, hidden=args.pde_hidden, n_blocks=args.pde_n_blocks,
            mode="markovian", backbone="spectral_pde_raw",
            spectral_K=pde_K, spectral_L=pde_L,
            spectral_max_order=args.pde_max_order, spectral_integrator=args.pde_integrator,
            ode_substeps=args.pde_ode_substeps,
            spectral_field_kind=args.pde_field_kind, spectral_poly_degree=args.pde_poly_degree,
            spectral_poly_max_term_order=args.pde_poly_max_term_order,
            spectral_poly_stable_leading=args.pde_poly_stable_leading,
            spectral_poly_no_constant=args.pde_poly_no_constant,
            spectral_poly_fixed_linear_terms=(
                {2: -1.0, 4: -1.0} if args.pde_fix_w_xx_w_xxxx else None
            ),
            spectral_poly_exclude_nonconservative=args.pde_poly_exclude_nonconservative,
        )
        pde_head = AuxPropagator(pde_head_cfg)
        pde_head_prop_cfg_for_save = aux_cfg_to_propagator_cfg(pde_head_cfg)

    # --init-ae-checkpoint/--init-aux-checkpoint (added 2026-08-31,
    # user-directed): continue joint Stage-1 training from ALREADY-TRAINED
    # weights instead of a fresh random init -- e.g. fine-tuning an
    # existing AE jointly with an already Stage-2-converged propagator,
    # rather than the small/fresh aux this script would otherwise build.
    # Override AFTER the normal (fresh) construction above rather than
    # branching the whole config-building logic -- simpler, and the
    # discarded fresh ae/aux are harmless to have briefly constructed.
    encoder_kind_for_save = args.encoder
    if args.init_ae_checkpoint:
        ae, ae_cfg, loaded_ae_ckpt = load_autoencoder_checkpoint(args.init_ae_checkpoint, device=device)
        # Reuse the SOURCE checkpoint's own recorded encoder_kind rather
        # than re-deriving it from the loaded class -- avoids silently
        # saving the wrong dispatch tag if --encoder wasn't also passed to
        # match (a real correctness bug, not just cosmetic: a wrong
        # encoder_kind makes a later load_autoencoder_checkpoint build the
        # WRONG class for this state_dict).
        encoder_kind_for_save = loaded_ae_ckpt.get("encoder_kind", "transformer")
        print(f"[stage1] --init-ae-checkpoint: loaded AE from {args.init_ae_checkpoint} "
              f"({type(ae_cfg).__name__}, encoder_kind={encoder_kind_for_save!r})", flush=True)
    if args.init_aux_checkpoint:
        # AuxPropagator is architecturally identical to LatentPropagator for
        # the same effective config (AuxPropagator.__init__ just converts
        # its own config and delegates) -- a Stage-2 checkpoint's
        # state_dict loads directly as `aux`, no wrapping/conversion needed.
        aux, aux_prop_cfg_for_save, _ = load_propagator_checkpoint(args.init_aux_checkpoint, device=device)
        print(f"[stage1] --init-aux-checkpoint: loaded aux propagator from "
              f"{args.init_aux_checkpoint} ({type(aux_prop_cfg_for_save).__name__})", flush=True)
    save_full_propagator = args.full_propagator or bool(args.init_aux_checkpoint)

    reg_cfg = RegConfig(
        lambda_z=args.lambda_z, lambda_decorr=args.lambda_decorr_band,
        start_epoch=args.reg_start_epoch,
    )

    # Same cosmetic-bug class already fixed for train_stage2_patched.py's
    # --init-prop-checkpoint (2026-08-30): args.encoder is the CLI flag
    # (default "transformer"), not necessarily the actually-loaded
    # architecture when --init-ae-checkpoint overrode it.
    encoder_label = type(ae).__name__ if args.init_ae_checkpoint else args.encoder
    print(
        f"[stage1] training {encoder_label} AE ({train_cfg.epochs} epochs) on {device} -- "
        f"progress printed below every 10 epochs.",
        flush=True,
    )

    def _save_periodic_checkpoint(epoch: int) -> None:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        ckpt_path = ARTIFACTS_DIR / f"stage1_ae_patched_{args.profile}{suffix}_checkpoint.pt"
        torch.save(
            {
                "ae_state_dict": ae.state_dict(), "ae_config": ae_cfg, "aux_config": aux_cfg,
                "encoder_kind": encoder_kind_for_save, "epoch": epoch,
            },
            ckpt_path,
        )
        if save_full_propagator:
            prop_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_{args.profile}{suffix}_checkpoint.pt"
            torch.save(
                {"prop_state_dict": aux.state_dict(), "prop_config": aux_prop_cfg_for_save, "epoch": epoch},
                prop_ckpt_path,
            )
        if pde_head is not None:
            pdehead_ckpt_path = ARTIFACTS_DIR / f"stage1_pdehead_{args.profile}{suffix}_checkpoint.pt"
            torch.save(
                {
                    "prop_state_dict": pde_head.state_dict(), "prop_config": pde_head_prop_cfg_for_save,
                    "epoch": epoch,
                },
                pdehead_ckpt_path,
            )
        print(f"  [stage1] wrote mid-training checkpoint at epoch {epoch} -> {ckpt_path}", flush=True)

    t0 = time.time()
    result = train_stage1(
        ae, aux, train_traj, val_traj, train_cfg, device, reg_cfg=reg_cfg, verbose=True, log_every=10,
        checkpoint_every=args.checkpoint_every,
        on_epoch_end=_save_periodic_checkpoint if args.checkpoint_every > 0 else None,
        amp=args.amp,
        pde_head=pde_head,
    )
    wall_time = time.time() - t0

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    ckpt_path = ARTIFACTS_DIR / f"stage1_ae_patched_{args.profile}{suffix}.pt"
    torch.save(
        {
            "ae_state_dict": ae.state_dict(),
            "ae_config": ae_cfg,
            "aux_config": aux_cfg,
            "encoder_kind": encoder_kind_for_save,
            "val_recon_final": result.val_recon_final,
        },
        ckpt_path,
    )
    history_path = ARTIFACTS_DIR / f"stage1_history_{args.profile}{suffix}.json"
    history_path.write_text(json.dumps(result.train_history, indent=2))
    write_provenance(ckpt_path, config=ae_cfg, seed=args.seed, device=device.type, wall_time_s=wall_time)

    print(f"val_recon_final = {result.val_recon_final:.6f}")
    print(f"wrote {ckpt_path}, {history_path}")

    if save_full_propagator:
        prop_cfg = aux_prop_cfg_for_save
        prop_ckpt_path = ARTIFACTS_DIR / f"stage1_prop_{args.profile}{suffix}.pt"
        torch.save({"prop_state_dict": aux.state_dict(), "prop_config": prop_cfg}, prop_ckpt_path)
        write_provenance(prop_ckpt_path, config=prop_cfg, seed=args.seed, device=device.type, wall_time_s=wall_time)
        print(f"wrote {prop_ckpt_path} (Phase-1-trained full propagator, for --init-prop-checkpoint)")

    if pde_head is not None:
        pdehead_ckpt_path = ARTIFACTS_DIR / f"stage1_pdehead_{args.profile}{suffix}.pt"
        torch.save(
            {"prop_state_dict": pde_head.state_dict(), "prop_config": pde_head_prop_cfg_for_save},
            pdehead_ckpt_path,
        )
        write_provenance(
            pdehead_ckpt_path, config=pde_head_prop_cfg_for_save, seed=args.seed,
            device=device.type, wall_time_s=wall_time,
        )
        print(f"wrote {pdehead_ckpt_path} (distilled interpretable spectral_pde head, loadable via "
              f"load_propagator_checkpoint)")

    # --checkpoint-every's rolling mid-training checkpoint is only crash
    # insurance DURING a run (added 2026-08-31, see that flag's help text) --
    # once we've reached here, the real final checkpoint(s) above exist, so
    # delete it rather than leave a redundant, confusingly-similar file
    # behind. Only ever runs on a normal (non-crashed) exit, which is
    # exactly when it's safe to do so.
    if args.checkpoint_every > 0:
        for stale in (
            ARTIFACTS_DIR / f"stage1_ae_patched_{args.profile}{suffix}_checkpoint.pt",
            ARTIFACTS_DIR / f"stage1_prop_{args.profile}{suffix}_checkpoint.pt",
            ARTIFACTS_DIR / f"stage1_pdehead_{args.profile}{suffix}_checkpoint.pt",
        ):
            if stale.exists():
                stale.unlink()
                print(f"removed superseded mid-training checkpoint {stale}")


if __name__ == "__main__":
    main()
