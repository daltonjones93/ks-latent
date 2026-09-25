#!/usr/bin/env python
"""Phase 3 entry point: train the Stage-2 latent propagator on a frozen
Stage-1 AE's latents (brief §5.2).

    python scripts/train_stage2_patched.py                  # full run (needs a Stage-1 checkpoint)
    python scripts/train_stage2_patched.py --profile smoke   # <60s, trains its own tiny AE first

Device policy (brief §1.3.8, measured in docs/RESULTS.md): the original
tiny residual-MLP propagator is small enough that MPS does *not* win
(launch-overhead dominated, ~0.3s/epoch either way), so early runs pinned
CPU unconditionally. That benchmark does not hold for the attention-heavy
`--backbone vit`/`transformer` bodies (44 tokens, token_d_model=64,
k_max-step rollout) -- those are compute-bound and MPS wins substantially.
`--device {auto,cpu,mps,cuda}` (default `auto`, added 2026-08-29) now
routes through `ks_latent.utils.device.get_device`; `auto` picks MPS when
available.

**Ported improvements, 2026-08-29** (CLAUDE_CODE_BRIEF.md §5.1 addendum):
`Stage2TrainingConfig` now defaults to an explicit `weight_decay=1e-5` and a
linear LR warmup before the cosine decay (previously an implicit
`weight_decay=0.01` and no warmup); see `train_stage1_patched.py`'s
docstring for the shared rationale. `--latent-loss l1` is a new,
off-by-default experiment to try.

`--backbone {mlp,transformer,vit}` (2026-08-29): selects the propagator
architecture (brief §5.2 addendum). `vit` is a new, user-directed option
-- the ViT-style block (circular positional encoding, `mlp_ratio`-expansion
FFN) applied to the tokenized latent with no pooling/bottleneck step; see
`PropagatorConfig`'s docstring. Not yet benchmarked against the canonical
recipe.

**`--init-prop-checkpoint`, 2026-08-30** (user-directed, restructured phased
training -- see docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md §2): loads an
existing propagator checkpoint's architecture and weights (the same
`{"prop_state_dict", "prop_config"}` format this script always wrote, now
also written by `train_stage1_patched.py --full-propagator`) and fine-tunes
it with this run's schedule flags, instead of constructing and training a
fresh propagator from scratch. `--tag` (new) suffixes output artifact names
so a fine-tuning run's checkpoint doesn't overwrite the from-scratch one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import h5py
import torch

from ks_latent.config import (
    AutoencoderConfig,
    AuxPropagatorConfig,
    EnsemblePropagatorConfig,
    KSConfig,
    MLPAutoencoderConfig,
    PropagatorConfig,
    Stage1TrainingConfig,
    Stage2TrainingConfig,
    ViTAutoencoderConfig,
)
from ks_latent.models import build_propagator_from_config, load_autoencoder_checkpoint, load_propagator_checkpoint
from ks_latent.models.permuted_autoencoder import PermutedAutoencoder, load_latent_permutation
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.propagator import AuxPropagator, masked_mlp_warm_start
from ks_latent.solver.dataset import generate_trajectory_dataset
from ks_latent.training.loops import encode_dataset_with_shifts, train_stage1, train_stage2
from ks_latent.utils.device import get_device
from ks_latent.utils.io import write_provenance
from ks_latent.utils.seeding import set_seed

ARTIFACTS_DIR = Path("artifacts")
DEVICE = torch.device("cpu")  # overwritten in main() per --device before any training happens


def _load_or_train_ae(
    args,
) -> tuple[
    KSAutoencoderPatched | KSAutoencoderMLP | KSAutoencoderViT,
    AutoencoderConfig | MLPAutoencoderConfig | ViTAutoencoderConfig,
    torch.Tensor,
    torch.Tensor,
]:
    if args.profile == "smoke":
        ks_cfg = KSConfig(L=22.0, NX=64, dt=0.05, snapshot_every=5, spinup_time=10.0, seed=args.seed)
        ae_cfg = AutoencoderConfig(
            NX=64, patch_size=8, group_size=4, d_model=16, nhead=2, dim_ff=16,
            patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
            n_query_tokens=4, d_latent=8,
        )
        aux_cfg = AuxPropagatorConfig(d_latent=ae_cfg.d_latent, hidden=16, n_blocks=1)
        dataset_path = ARTIFACTS_DIR / "datasets" / "smoke_stage2_trajectories.h5"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate_trajectory_dataset(ks_cfg, dataset_path, n_train=6, n_val=2, trajectory_time=8.0)
        with h5py.File(dataset_path, "r") as f:
            n_train = int(f["metadata"].attrs["n_train"])
            trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
        train_traj, val_traj = trajectories[:n_train], trajectories[n_train:]
        ae = KSAutoencoderPatched(ae_cfg)
        aux = AuxPropagator(aux_cfg)
        train_stage1(ae, aux, train_traj, val_traj, Stage1TrainingConfig(epochs=2, batch_size=8), DEVICE)
        return ae, ae_cfg, train_traj, val_traj

    ckpt_path = Path(args.ae_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found; train Stage 1 first.")
    ae, ae_cfg, ckpt = load_autoencoder_checkpoint(ckpt_path, device=DEVICE)
    if args.latent_permutation:
        ae = PermutedAutoencoder(ae, load_latent_permutation(args.latent_permutation, args.latent_permutation_key)).to(DEVICE)

    dataset_path = Path(args.dataset)
    with h5py.File(dataset_path, "r") as f:
        n_train = int(f["metadata"].attrs["n_train"])
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    return ae, ae_cfg, trajectories[:n_train], trajectories[n_train:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["two_step", "markovian", "history"], default="markovian",
        help="'markovian' (default): z_n -> z_n+1. 'two_step': (z_n-1, z_n) -> z_n+1, MLP "
        "only. 'history' (added 2026-08-29, user-directed: 'the original model for this "
        "project used a propagator with information from the current step and a step in "
        "the past... design a ViT model that takes in several states from the past'): "
        "an arbitrary-length history z_hist (B, --n-history, d_latent) -> z_n+1, using "
        "joint spatio-temporal ViT attention -- requires --backbone vit. See "
        "PropagatorConfig's docstring.",
    )
    parser.add_argument(
        "--n-history", type=int, default=3,
        help="--mode history only. Total states including current (default 3 = 2 past + "
        "current, the case requested first). Must be >= 2.",
    )
    parser.add_argument(
        "--token-window", type=int, default=None,
        help="'transformer'/'vit'/'history' backbone-mode only. Overlapping-patch tokenization "
        "on the latent index (same construction as ViTAutoencoderConfig.token_window) -- each "
        "token's input becomes a token_window-wide, circularly-padded slice of z (>= "
        "d_latent // --prop-n-tokens), centered on its own output slot. Default None = "
        "non-overlapping (unchanged).",
    )
    parser.add_argument("--ae-checkpoint", type=str, default="artifacts/stage1_ae_patched_full.pt")
    parser.add_argument(
        "--dataset", type=str, default="artifacts/datasets/stage1_trajectories_dtsnap1.h5"
    )
    parser.add_argument(
        "--latent-loss", choices=["l2", "l1"], default="l2",
        help="Ported improvement (2026-08-29, default matches the brief's original "
        "L2 recipe): 'l1' is an untested-at-scale alternative found to give "
        "visibly better short-time tracking in a reference implementation "
        "(Stage2TrainingConfig's docstring).",
    )
    parser.add_argument(
        "--backbone",
        choices=[
            "mlp", "transformer", "vit", "fno_vit", "fno_mlp", "fourier_mlp",
            "local_mlp", "masked_mlp", "masked_mlp_wide", "masked_mlp_expand", "node", "cnn", "site_conv",
            "spectral_pde", "spectral_pde_raw",
        ], default="mlp",
        help="Propagator architecture (brief §5.2 addendum). 'mlp' (default) = "
        "the brief's residual-MLP body. 'transformer' = tokenized self-attention "
        "with a learned positional embedding. 'vit' (added 2026-08-29, "
        "user-directed): same tokenization, but the KSAutoencoderViT-style "
        "block (mlp_ratio-expansion FFN) with no pooling/bottleneck step -- "
        "see PropagatorConfig's docstring. 'fno_vit' (added 2026-08-30, "
        "user-directed 'Solution 2' for the fixed-point collapse -- see "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 5): same tokenization, "
        "but --fno-n-layers Fourier spectral-conv layers (global receptive field "
        "via a handful of low-frequency Fourier modes) run BEFORE the "
        "--prop-n-tokens ViT attention blocks -- see --fno-modes/--fno-n-layers. "
        "'node' (added 2026-08-31, user-directed -- Phase 2 architecture doc "
        "Section 42/43, the most literal 'model the latent as a PDE' option): "
        "dz/dt = f_theta(z) with a translation-equivariant local circular-conv "
        "vector field (--attn-window radius, --hidden/--n-blocks sized), "
        "integrated via fixed-step RK4 over --ode-substeps sub-steps -- see "
        "PropagatorConfig's 'node' docstring section. 'cnn' (added 2026-09-01, "
        "user-directed 'try a CNN for a propagator too... based on the best "
        "features of the current MLP'): the same input_proj -> n_blocks "
        "residual blocks -> final LayerNorm -> output_proj (zero-init) "
        "structure 'mlp' uses, with cross-index Linears replaced by circular "
        "Conv1d of kernel WIDTH --cnn-kernel-size (not a radius, unlike "
        "--attn-window) -- see PropagatorConfig's 'cnn' docstring section. "
        "'spectral_pde' (added 2026-09-06, see docs/sine_transform_pde_plan.md): "
        "requires an --ae-checkpoint trained with --encoder spectral_field. "
        "Synthesizes EXACT spatial derivatives (up to --spectral-max-order) from "
        "the truncated rFFT spectrum z via a diagonal (i*k)^n multiplier, then a "
        "shared pointwise MLP (--hidden/--n-blocks sized) maps that local "
        "derivative stack to w_t at every point -- the implicitly-learned PDE. "
        "Integrated via --spectral-integrator: 'euler' (single step, default), 'rk4' "
        "(--ode-substeps sub-steps, same RK4 loop as 'node'), or 'etdrk4' (exact "
        "exponential integration of KS's own true linear term, mirroring the real "
        "solver directly -- see --spectral-integrator's own help text). "
        "--spectral-K/--spectral-N-w/--spectral-L default to the paired "
        "checkpoint's own SpectralFieldAutoencoderConfig.K/.N_w/.L when not "
        "given explicitly -- see PropagatorConfig's 'spectral_pde' docstring.",
    )
    parser.add_argument(
        "--pos-encoding", choices=["circular", "linear"], default="circular",
        help="'vit' backbone only. 'circular' (default) = CircularPositionalEncoding "
        "+ ring-distance --attn-window masking. 'linear' (user-directed, "
        "2026-08-29) = LinearPositionalEncoding (fixed, non-periodic) + "
        "linear-distance masking -- validated at Stage-1 scale to perform "
        "indistinguishably from circular; chosen as the forward-looking "
        "default for future work since most real-world PDEs are not periodic "
        "(see docs/RESULTS.md's linear-vs-circular comparison).",
    )
    parser.add_argument(
        "--attn-window", type=int, default=None,
        help="'transformer'/'vit' backbone only. Restrict attention to "
        "+-attn_window neighbours (non-causal) instead of full attention. "
        "Default None = full attention. Also reused (2026-09-03, "
        "user-directed) by --backbone fourier_mlp: instead of full "
        "attention, restricts its raw-value path to a circular-band mask "
        "at this radius -- see PropagatorConfig's backbone='fourier_mlp' "
        "docstring section.",
    )
    parser.add_argument(
        "--fourier-ifft-readout", action="store_true",
        help="--backbone fourier_mlp only, fresh (non-warm-started) propagator: "
        "PropagatorConfig.fourier_ifft_readout -- see that field's docstring. "
        "User-directed (2026-09-04): testing whether a larger fresh propagator "
        "trained on an existing (frozen) AE checkpoint resolves a capacity "
        "bottleneck.",
    )
    parser.add_argument(
        "--nonexpansive", action="store_true",
        help="--backbone fourier_mlp only, fresh (non-warm-started) propagator: "
        "PropagatorConfig.nonexpansive -- see that field's docstring.",
    )
    parser.add_argument(
        "--prop-n-tokens", type=int, default=4,
        help="'transformer'/'vit'/'fno_vit' backbone only; must divide d_latent. Default 4. "
        "Raise this (e.g. 44 = one token per latent coordinate at d_latent=44) "
        "so a finite --attn-window is actually restrictive rather than "
        "covering the whole (small) token ring.",
    )
    parser.add_argument(
        "--prop-token-d-model", type=int, default=32,
        help="'transformer'/'vit'/'fno_vit' backbone only. CircularPositionalEncoding "
        "requires this to be >= the ring's independent harmonic count "
        "(~prop_n_tokens for even prop_n_tokens); LinearPositionalEncoding has "
        "no such constraint. Raise alongside --prop-n-tokens (e.g. 64 for "
        "--prop-n-tokens 44) when using pos_encoding='circular'.",
    )
    parser.add_argument(
        "--prop-token-nhead", type=int, default=2,
        help="'transformer'/'vit'/'fno_vit' backbone only. Override "
        "PropagatorConfig.token_nhead (previously hardcoded to 2 here). Must "
        "divide --prop-token-d-model. Added 2026-09-03 for precise param-count "
        "targeting.",
    )
    parser.add_argument(
        "--prop-token-n-layers", type=int, default=2,
        help="'transformer'/'vit'/'fno_vit' backbone only. Override "
        "PropagatorConfig.token_n_layers (previously hardcoded to 2 here). "
        "Added 2026-09-03 for precise param-count targeting.",
    )
    parser.add_argument(
        "--fno-modes", type=int, default=None,
        help="'fno_vit' backbone only. PropagatorConfig.fno_modes: number of "
        "low-frequency Fourier modes each spectral-conv layer keeps (default "
        "None = every rfft mode, i.e. prop_n_tokens // 2 + 1, no truncation).",
    )
    parser.add_argument(
        "--fno-n-layers", type=int, default=2,
        help="'fno_vit' backbone only. PropagatorConfig.fno_n_layers: number of "
        "stacked FNO spectral-conv layers before the ViT attention blocks (default 2).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
        help="Routed through ks_latent.utils.device.get_device (added "
        "2026-08-29). 'auto' (default) picks MPS when available. The old "
        "hardcoded CPU default was measured for the tiny residual-MLP "
        "propagator only -- --backbone vit/transformer are compute-bound "
        "attention stacks and train much faster on MPS.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print one progress line per --log-every epochs (default on, "
        "2026-08-29): epoch, k_now, lr, loss, val_kmax_mse, wall time.",
    )
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable --verbose.")
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Print a progress line every this many epochs (plus the final epoch). Default 1 "
        "(every epoch) -- only 20 epochs total by default, and each is expensive enough "
        "that per-epoch visibility matters more than log volume.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override Stage2TrainingConfig.epochs (default 20, brief §5.2's original recipe). "
        "Not applied to --profile smoke.",
    )
    parser.add_argument(
        "--hidden", type=int, default=None,
        help="--backbone mlp/node only. Override PropagatorConfig.hidden (default 128 -- "
        "the residual MLP body's width, or 'node''s vector field conv channel width). "
        "Added 2026-08-31, user-directed ('give the markovian mlp 1.2 times as many "
        "parameters'): with n_blocks=3, d_latent=44 unchanged, hidden=141 gives ~1.2x "
        "the default's total parameter count (111532 -> 133853).",
    )
    parser.add_argument(
        "--n-blocks", type=int, default=None,
        help="--backbone mlp/node only. Override PropagatorConfig.n_blocks (default 3 -- "
        "the number of residual MLP blocks, or 'node''s vector field residual conv blocks).",
    )
    parser.add_argument(
        "--ode-substeps", type=int, default=None,
        help="--backbone node only. Override PropagatorConfig.ode_substeps (default 4): "
        "number of fixed-step RK4 sub-steps per snapshot-index interval. See "
        "PropagatorConfig's 'node' docstring section.",
    )
    parser.add_argument(
        "--cnn-kernel-size", type=int, default=None,
        help="--backbone cnn only. Override PropagatorConfig.cnn_kernel_size (default 16): "
        "the circular Conv1d kernel WIDTH (not a radius, unlike --attn-window). See "
        "PropagatorConfig's 'cnn' docstring section.",
    )
    parser.add_argument(
        "--spectral-K", type=int, default=None,
        help="--backbone spectral_pde only. PropagatorConfig.spectral_K: number of kept "
        "rFFT modes (d_latent must equal 2*this). Default: the paired --ae-checkpoint's "
        "own SpectralFieldAutoencoderConfig.K.",
    )
    parser.add_argument(
        "--spectral-N-w", type=int, default=None,
        help="--backbone spectral_pde only. PropagatorConfig.spectral_N_w: physical grid "
        "length derivative fields are synthesized at. Default: the paired --ae-checkpoint's "
        "own SpectralFieldAutoencoderConfig.N_w.",
    )
    parser.add_argument(
        "--spectral-L", type=float, default=None,
        help="--backbone spectral_pde only. PropagatorConfig.spectral_L: physical domain "
        "length (must match the training dataset's KSConfig.L). Default: the paired "
        "--ae-checkpoint's own SpectralFieldAutoencoderConfig.L, else 100.0.",
    )
    parser.add_argument(
        "--spectral-max-order", type=int, default=4,
        help="--backbone spectral_pde only. PropagatorConfig.spectral_max_order (default "
        "4, matching true KS's own governing equation): highest spatial derivative order "
        "synthesized (0..this inclusive).",
    )
    parser.add_argument(
        "--spectral-integrator", choices=["euler", "rk4", "etdrk4"], default="euler",
        help="--backbone spectral_pde only. PropagatorConfig.spectral_integrator (default "
        "'euler'): 'rk4' reuses --ode-substeps sub-steps, same RK4 loop as 'node'. 'etdrk4' "
        "(added 2026-09-06, see docs/sine_transform_pde_plan.md) mirrors "
        "ks_latent/solver/ks.py's own ETDRK4 solver directly: the linear term Lhat(k)=k^2-k^4 "
        "(KS's own true, fixed, non-learned value) is integrated EXACTLY via exp(dt*Lhat) "
        "(unconditionally stable regardless of step size), and only the nonlinear residual is "
        "learned by the pointwise MLP -- unlike 'euler'/'rk4', which ask the MLP to learn the "
        "entire (stiff) right-hand side through a generic explicit integrator. Also reuses "
        "--ode-substeps. See PropagatorConfig's 'spectral_pde' docstring for the full "
        "correspondence with the real solver's step().",
    )
    parser.add_argument(
        "--spectral-physics-prior", action="store_true",
        help="--backbone spectral_pde only. PropagatorConfig.spectral_physics_prior (default "
        "off). Bakes the EXACT true KS right-hand side into field()'s output as a fixed "
        "baseline (added 2026-09-08, see docs/sine_transform_pde_plan.md) -- the pointwise "
        "MLP now learns only a correction on top, rather than the whole dynamics from "
        "scratch. Integrator-aware: only -w*w_x is baked in under 'etdrk4' (the linear part "
        "is already handled separately there); the full -w*w_x-w_xx-w_xxxx is baked in "
        "under 'euler'/'rk4'. Requires --spectral-max-order >= 4.",
    )
    parser.add_argument(
        "--spectral-field-kind", choices=["mlp", "polynomial"], default="mlp",
        help="--backbone spectral_pde only. PropagatorConfig.spectral_field_kind (default "
        "'mlp'). 'polynomial' (added 2026-09-09, see docs/sine_transform_pde_plan.md) "
        "replaces the pointwise MLP with a single shared linear layer over the degree-"
        "<=--spectral-poly-degree monomial library built from the derivative stack -- a "
        "dense SINDy-style linear regression whose learned coefficients are directly "
        "readable after training.",
    )
    parser.add_argument(
        "--spectral-poly-degree", type=int, choices=[1, 2, 3], default=2,
        help="--spectral-field-kind polynomial only. PropagatorConfig.spectral_poly_degree "
        "(default 2 -- sufficient to represent KS's own true nonlinearity -w*w_x exactly; "
        "3 adds cubic cross terms).",
    )
    parser.add_argument(
        "--spectral-poly-max-term-order", type=int, default=None,
        help="--spectral-field-kind polynomial only. PropagatorConfig."
        "spectral_poly_max_term_order (default None, unrestricted). Excludes any monomial "
        "whose derivative orders SUM to >= this value from the library entirely (e.g. "
        "w_xxx*w_xxxx has combined order 7, excluded at 5). See train_stage1_patched.py's "
        "own help text for the full motivation.",
    )
    parser.add_argument(
        "--spectral-poly-norm-power", type=float, default=1.0,
        help="--spectral-field-kind polynomial only. PropagatorConfig."
        "spectral_poly_norm_power (default 1.0). See train_stage1_patched.py's own help "
        "text for the full motivation ('less normalization for the polynomial').",
    )
    parser.add_argument(
        "--spectral-poly-stable-leading", action="store_true",
        help="--spectral-field-kind polynomial only. Sets PropagatorConfig."
        "spectral_poly_stable_leading (default False). See train_stage1_patched.py's own "
        "help text for the full motivation and eigenvalue derivation.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="Save a rolling mid-training checkpoint every N epochs (default 0 = off). "
        "See train_stage1_patched.py's --checkpoint-every help text for the motivation "
        "(added 2026-08-31 after an unattended run was lost to a machine crash) -- "
        "overwrites a single stage2_prop_patched_{profile}{tag}_checkpoint.pt each time, "
        "saving the CURRENT (not best-so-far) propagator state, not one file per "
        "checkpoint. Not a full resume mechanism (no optimizer/scheduler state saved).",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Added 2026-09-01, user-directed ('we should be training in bfloat16 or "
        "float16, that will be way more efficient'): wraps each batch's forward pass + "
        "loss computation in torch.autocast(device_type=device.type, dtype=torch.bfloat16). "
        "See train_stage1_patched.py's --amp help text for why bfloat16 (not float16) and "
        "confirmation it works on this project's MPS device. Default off (unchanged "
        "behavior/numerics).",
    )
    parser.add_argument(
        "--k-warmup-epochs", type=int, default=None,
        help="Override Stage2TrainingConfig.k_warmup_epochs (default 8): total epochs to ramp "
        "k from 2 to k_max, in either the single-segment or --k-mid two-segment schedule.",
    )
    parser.add_argument(
        "--k-mid", type=int, default=None,
        help="Split the K-curriculum into two linear segments, 2 -> k_mid over the first "
        "--k-mid-epochs epochs, then k_mid -> k_max over the rest of --k-warmup-epochs "
        "(user-directed, 2026-08-29: dwell longer on short rollout horizons after the vit/"
        "attn_window=4/linear propagator's val_kmax_mse plateaued once k hit 16). "
        "Default None = original single-segment linear ramp.",
    )
    parser.add_argument(
        "--k-mid-epochs", type=int, default=0,
        help="Epochs spent ramping 2 -> --k-mid before the second segment takes over to "
        "k_max. Only meaningful with --k-mid set.",
    )
    parser.add_argument(
        "--k-max", type=int, default=None,
        help="Override Stage2TrainingConfig.k_max (default 16). User-directed (2026-08-29): "
        "an autonomous free-running rollout of the vit/attn_window=4/linear propagator found "
        "it converges to a SINGLE fixed point (identical regardless of initial condition) "
        "within ~100-200 steps -- consistent with the Lyapunov spectrum (lambda1<0, D_KY=0) "
        "and the DA ensemble collapse (spread/rmse=0.066) from Gate 3. k_max=16 never sees far "
        "enough into a rollout for the training loss to penalize this. Raise substantially "
        "(e.g. 64) to push training into the regime where the collapse would show up as loss.",
    )
    parser.add_argument(
        "--compare-k", type=int, default=None,
        help="Override Stage2TrainingConfig.compare_k (default None, off). User-directed "
        "(2026-09-04): 'add a readout for val_kequals12_mse, so we can compare these runs "
        "more directly' -- val_kmax_mse pools its MSE over the WHOLE k_max-length rollout, so "
        "runs at different --k-max aren't directly comparable (a larger k_max's average "
        "includes extra, more-diverged late steps, inflating it for an equally good model -- "
        "see Stage2TrainingConfig.compare_k's docstring). Set to another run's k_max (e.g. 12) "
        "to also log/print val_k{compare_k}_mse, pooled over just that many steps of the SAME "
        "rollout, directly comparable to that other run's own val_kmax_mse. Must be <= --k-max.",
    )
    parser.add_argument(
        "--w-varmatch", type=float, default=0.0,
        help="Weight on a variance-matching term applied to the ROLLED-OUT predictions "
        "z_pred (user-directed, 2026-08-29, motivated by the fixed-point collapse above): "
        "penalizes (Var(z_pred_i) - 1)^2 per latent dim i, computed across the batch and all "
        "k_now rollout steps in the current epoch. A collapsing rollout has Var -> 0 as "
        "different initial conditions converge to the same point, so this directly opposes "
        "that failure mode. Reuses ks_latent.training.losses.decorr_var_loss (only its l_var "
        "term, target fixed at 1 -- NOT the off-diagonal l_decorr term, and NOT the separate "
        "banded RegConfig.lambda_z smoothness regularizer, which is unrelated and off by "
        "default). Default 0.0 = off (unchanged behavior).",
    )
    parser.add_argument(
        "--w-varmatch-adaptive", action="store_true",
        help="Only meaningful with --w-varmatch > 0 (user-directed, 2026-08-31, Phase 2 "
        "architecture doc Section 35 option 7): replace decorr_var_loss's hardcoded uniform "
        "target of 1 with the AE's own real per-channel latent variance (computed once from "
        "the encoded training data). Fixes a found conflict where the fixed target of 1 "
        "actively fights the primary loss on any channel whose true variance is far from 1 "
        "(e.g. a collapsed channel at ~2.4e-6), while preserving the original "
        "anti-fixed-point-collapse motivation. Default off (unchanged behavior).",
    )
    parser.add_argument(
        "--w-spatial", type=float, default=None,
        help="Override Stage2TrainingConfig.w_spatial (default 0.0, off). Stage-2 analogue of "
        "Stage1TrainingConfig.w_spatial (added 2026-09-01, user-directed): applies "
        "spatial_coherence_loss to the PROPAGATOR's rolled-out predictions z_pred, rewarding "
        "a propagator whose own predictions preserve the banded/local correlation structure "
        "Stage 1 induced in z, rather than relying on Stage 1 alone. Same collapse-risk "
        "precedent as Stage 1's own w_spatial -- see spatial_coherence_loss's docstring.",
    )
    parser.add_argument(
        "--spatial-bandwidth", type=float, default=None,
        help="--w-spatial (stage 2) only. Override Stage2TrainingConfig.spatial_bandwidth "
        "(default 3.0): Gaussian circular-band kernel width, same convention as Stage 1's "
        "--spatial-bandwidth.",
    )
    parser.add_argument(
        "--spatial-signed", action="store_true",
        help="--w-spatial (stage 2) only. Sets Stage2TrainingConfig.spatial_signed (default "
        "False, off, unchanged behavior). See train_stage1_patched.py's --spatial-signed "
        "help text -- identical mechanism, applied to Stage 2's own w_spatial term.",
    )
    parser.add_argument(
        "--w-lowpass-rollout", type=float, default=None,
        help="encoder_kind='spectral_field'/backbone='spectral_pde' only. Override "
        "Stage2TrainingConfig.w_lowpass_rollout (default 0.0, off). Stage-2 analogue of "
        "Stage1TrainingConfig.w_lowpass (added 2026-09-08, user-directed after visualizing "
        "Section 104's D_KY=22 result), but applied to the PROPAGATOR's own rolled-out "
        "z_pred rather than the encoder's direct z -- penalizes the propagator's own "
        "free-running rollout for carrying high-wavenumber energy the true attractor "
        "doesn't have, instead of only constraining the encoder's output.",
    )
    parser.add_argument(
        "--lowpass-rollout-power", type=float, default=None,
        help="--w-lowpass-rollout only. Override Stage2TrainingConfig.lowpass_rollout_power "
        "(default 1.0). Same convention as Stage 1's --lowpass-power.",
    )
    parser.add_argument(
        "--w-z-lowpass-rollout", type=float, default=None,
        help="Any encoder/backbone (generalizes --w-lowpass-rollout, which requires "
        "encoder_kind='spectral_field', to any raw z via its own self-FFT -- "
        "ks_latent.training.losses.latent_self_spectrum_lowpass_loss). Override "
        "Stage2TrainingConfig.w_z_lowpass_rollout (default 0.0, off). Added 2026-09-11, "
        "user-directed: 'I would like to penalize higher frequencies in the fourier "
        "transform of the latent space. This may help make the pde dynamics more easily "
        "fit and smooth.' Applied to the PROPAGATOR's own rolled-out z_pred (Stage 2 has "
        "no encoder to apply a non-rollout term to).",
    )
    parser.add_argument(
        "--z-lowpass-rollout-power", type=float, default=None,
        help="--w-z-lowpass-rollout only. Override Stage2TrainingConfig."
        "z_lowpass_rollout_power (default 1.0).",
    )
    parser.add_argument(
        "--z-lowpass-K", type=int, default=None,
        help="--w-z-lowpass-rollout only. Override Stage2TrainingConfig.z_lowpass_K "
        "(default None = d_latent//2+1, matching a spectral_pde_raw pde_head's own "
        "default spectral_K).",
    )
    parser.add_argument(
        "--z-lowpass-L", type=float, default=None,
        help="--w-z-lowpass-rollout only. Override Stage2TrainingConfig.z_lowpass_L "
        "(default None = d_latent).",
    )
    parser.add_argument(
        "--w-logdet-rollout", type=float, default=None,
        help="backbone='spectral_pde'/'spectral_pde_raw' only. Override "
        "Stage2TrainingConfig.w_logdet_rollout (default 0.0, off). User-directed (2026-09-09, "
        "after Section 113 -- w_varmatch/noise_step, applied directly to z_pred -- still "
        "collapsed): full-covariance log-det anti-collapse barrier (Stage 1's own w_logdet's "
        "mechanism), applied to the PROPAGATOR's own rolled-out z_pred mapped to PHYSICAL "
        "space (decode_from_spectrum/irfft) first, so it reacts to the field's own shape "
        "collapsing across initial conditions, not just its spectral-coefficient magnitudes.",
    )
    parser.add_argument(
        "--logdet-rollout-eps", type=float, default=None,
        help="--w-logdet-rollout only. Override Stage2TrainingConfig.logdet_rollout_eps "
        "(default 1e-3). Same role as Stage 1's --logdet-eps.",
    )
    parser.add_argument(
        "--w-logdet-rollout-latent", type=float, default=None,
        help="Any backbone. Override Stage2TrainingConfig.w_logdet_rollout_latent (default "
        "0.0, off). Added 2026-09-23, Section 203 -- general (non-spectral-backbone) "
        "Stage-2 analogue of Stage 1's own --w-logdet: applies logdet_barrier_loss directly "
        "to the propagator's rolled-out z_pred (no physical-space decode, unlike "
        "--w-logdet-rollout above, which requires backbone='spectral_pde'/'spectral_pde_raw'). "
        "Distinct from --w-logdet-rollout -- may be combined with it.",
    )
    parser.add_argument(
        "--logdet-rollout-latent-eps", type=float, default=None,
        help="--w-logdet-rollout-latent only. Override Stage2TrainingConfig."
        "logdet_rollout_latent_eps (default 1e-3).",
    )
    parser.add_argument(
        "--w-spectrum-shape", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage2TrainingConfig."
        "w_spectrum_shape (default 0.0, off). User-directed (2026-09-10, Section 132, "
        "after Section 131 found Stage 1's own copy of this term wasn't enough -- a "
        "checkpoint's propagator Jacobian was healthy right after Stage 1 but had fully "
        "re-collapsed by the end of Stage 2's own unregularized 300-epoch continuation): "
        "Stage-2 analogue of Stage 1's --w-spectrum-shape, evaluated directly on "
        "propagator.step_one (no rollout) using real states from the current batch.",
    )
    parser.add_argument(
        "--spectrum-shape-n-expand", type=int, default=None,
        help="--w-spectrum-shape only. Override Stage2TrainingConfig.spectrum_shape_n_expand "
        "(default 11, this project's own L=100 replication target for the number of "
        "positive Lyapunov exponents -- CLAUDE.md section 18).",
    )
    parser.add_argument(
        "--spectrum-shape-expand-target", type=float, default=None,
        help="--w-spectrum-shape only. Override Stage2TrainingConfig."
        "spectrum_shape_expand_target (default 1.5).",
    )
    parser.add_argument(
        "--spectrum-shape-contract-floor", type=float, default=None,
        help="--w-spectrum-shape only. Override Stage2TrainingConfig."
        "spectrum_shape_contract_floor (default 0.7).",
    )
    parser.add_argument(
        "--spectrum-shape-n-samples", type=int, default=None,
        help="--w-spectrum-shape only. Override Stage2TrainingConfig.spectrum_shape_n_samples "
        "(default 32).",
    )
    parser.add_argument(
        "--spectrum-shape-two-sided", action="store_true",
        help="--w-spectrum-shape only. Override Stage2TrainingConfig.spectrum_shape_two_sided "
        "(default False). Section 186 -- see Stage 1's own flag of the same name (and "
        "ks_latent.training.losses.propagator_spectrum_shape_loss's docstring) for the full "
        "mechanism: the one-sided floor has zero gradient once a singular value already "
        "exceeds its target, so it cannot correct an ALREADY-excessive value back down; "
        "two_sided replaces it with a plain squared-error match in both groups. Was missing "
        "from this script's CLI even though Stage2TrainingConfig.spectrum_shape_two_sided "
        "already existed (added here 2026-09-23, Section 202, needed to carry Stage 1's "
        "two_sided setting through into Stage 2 consistently).",
    )
    parser.add_argument(
        "--w-jacobian-bandedness", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage2TrainingConfig."
        "w_jacobian_bandedness (default 0.0, off). User-directed (2026-09-24, Section 214: "
        "'let's make D3 into a loss'). D3's differentiable analogue (ks_latent.training."
        "losses.propagator_jacobian_bandedness_loss), evaluated directly on propagator."
        "step_one (no rollout) using real, UNNOISED encoded states, same convention as "
        "--w-spectrum-shape.",
    )
    parser.add_argument(
        "--jacobian-bandedness-bandwidth", type=float, default=None,
        help="--w-jacobian-bandedness only. Override Stage2TrainingConfig."
        "jacobian_bandedness_bandwidth (default 3.0, in latent-index-ring units).",
    )
    parser.add_argument(
        "--jacobian-bandedness-n-samples", type=int, default=None,
        help="--w-jacobian-bandedness only. Override Stage2TrainingConfig."
        "jacobian_bandedness_n_samples (default 32).",
    )
    parser.add_argument(
        "--w-jacobian-diagonal-bound", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage2TrainingConfig."
        "w_jacobian_diagonal_bound (default 0.0, off). Section 216 companion to "
        "--w-jacobian-bandedness (ks_latent.training.losses.propagator_jacobian_diagonal_"
        "bound_loss) -- caps each site's own self-coupling magnitude, orthogonal to "
        "bandedness. Evaluated directly on propagator.step_one, same convention as "
        "--w-jacobian-bandedness.",
    )
    parser.add_argument(
        "--jacobian-diagonal-bound-ceiling", type=float, default=None,
        help="--w-jacobian-diagonal-bound only. Override Stage2TrainingConfig."
        "jacobian_diagonal_bound_ceiling (default 1.5).",
    )
    parser.add_argument(
        "--jacobian-diagonal-bound-n-samples", type=int, default=None,
        help="--w-jacobian-diagonal-bound only. Override Stage2TrainingConfig."
        "jacobian_diagonal_bound_n_samples (default 32).",
    )
    parser.add_argument(
        "--w-multistep-growth-ceiling", type=float, default=None,
        help="mode=markovian only. Override Stage2TrainingConfig.w_multistep_growth_ceiling "
        "(default 0.0, off). Section 219 companion to --w-jacobian-bandedness/--w-jacobian-"
        "diagonal-bound (ks_latent.training.losses.propagator_multistep_growth_ceiling_loss): "
        "one-sided ceiling on the composed k-step Jacobian's top singular value, evaluated "
        "directly on propagator.step_one, same convention as --w-jacobian-bandedness.",
    )
    parser.add_argument(
        "--multistep-growth-ceiling-k", type=int, default=None,
        help="--w-multistep-growth-ceiling only. Override Stage2TrainingConfig."
        "multistep_growth_ceiling_k (default 10).",
    )
    parser.add_argument(
        "--multistep-growth-ceiling-value", type=float, default=None,
        help="--w-multistep-growth-ceiling only. Override Stage2TrainingConfig."
        "multistep_growth_ceiling_value (default 150.0, Section 216-calibrated).",
    )
    parser.add_argument(
        "--multistep-growth-ceiling-n-samples", type=int, default=None,
        help="--w-multistep-growth-ceiling only. Override Stage2TrainingConfig."
        "multistep_growth_ceiling_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-multistep-growth-barrier", type=float, default=None,
        help="mode=markovian only. Override Stage2TrainingConfig.w_multistep_growth_barrier "
        "(default 0.0, off). Section 220 safeguarded log-barrier companion to "
        "--w-multistep-growth-ceiling (ks_latent.training.losses.propagator_multistep_"
        "growth_barrier_loss), evaluated directly on propagator.step_one.",
    )
    parser.add_argument(
        "--multistep-growth-barrier-k", type=int, default=None,
        help="--w-multistep-growth-barrier only. Override Stage2TrainingConfig."
        "multistep_growth_barrier_k (default 10).",
    )
    parser.add_argument(
        "--multistep-growth-barrier-ceiling", type=float, default=None,
        help="--w-multistep-growth-barrier only. Override Stage2TrainingConfig."
        "multistep_growth_barrier_ceiling (default 75.0).",
    )
    parser.add_argument(
        "--multistep-growth-barrier-epsilon", type=float, default=None,
        help="--w-multistep-growth-barrier only. Override Stage2TrainingConfig."
        "multistep_growth_barrier_epsilon (default None -> 0.05*ceiling).",
    )
    parser.add_argument(
        "--multistep-growth-barrier-n-samples", type=int, default=None,
        help="--w-multistep-growth-barrier only. Override Stage2TrainingConfig."
        "multistep_growth_barrier_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-spectrum-shape-self", type=float, default=None,
        help="mode=markovian or mode=history, any backbone. Override Stage2TrainingConfig."
        "w_spectrum_shape_self (default 0.0, off). Section 204, user-directed: 'I want to try "
        "the self rollout spectrum-shape regularizer' -- applies the SAME mechanism Section "
        "193 built for pde_head (w_pde_spectrum_shape_self) to the MAIN propagator instead: "
        "shapes propagator.step_one's (or, mode=history, step_history's) Jacobian spectrum at "
        "states drawn from the propagator's OWN self-generated rollout, not just real "
        "on-attractor states (--w-spectrum-shape). Shares --spectrum-shape-n-expand/"
        "expand-target/contract-floor/two-sided above (same target shape, different sampling "
        "distribution). May be combined with --w-spectrum-shape.",
    )
    parser.add_argument(
        "--spectrum-shape-self-rollout-k", type=int, default=None,
        help="--w-spectrum-shape-self only. Override Stage2TrainingConfig."
        "spectrum_shape_self_rollout_k (default 20): number of steps the propagator's own "
        "free rollout runs (torch.no_grad()) before its final state (or final n_hist-length "
        "history window) becomes the evaluation pool.",
    )
    parser.add_argument(
        "--spectrum-shape-self-n-samples", type=int, default=None,
        help="--w-spectrum-shape-self only. Override Stage2TrainingConfig."
        "spectrum_shape_self_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-spectrum-shape-graded", type=float, default=None,
        help="mode=markovian only (any backbone). Override Stage2TrainingConfig."
        "w_spectrum_shape_graded (default 0.0, off). Section 134 -- see Stage1's own flag of "
        "the same name for the full mechanism. Requires "
        "--spectrum-shape-graded-reference-path. May be combined with --w-spectrum-shape.",
    )
    parser.add_argument(
        "--spectrum-shape-graded-reference-path", type=str, default=None,
        help="--w-spectrum-shape-graded only. Path to a .npy file of shape (d_latent,), "
        "descending -- typically produced by scripts/compute_reference_spectrum.py.",
    )
    parser.add_argument(
        "--spectrum-shape-graded-n-samples", type=int, default=None,
        help="--w-spectrum-shape-graded only. Override Stage2TrainingConfig."
        "spectrum_shape_graded_n_samples (default 32).",
    )
    parser.add_argument(
        "--noise-in-start", type=float, default=None,
        help="Override Stage2TrainingConfig.noise_in_start (default 0.10): Gaussian noise std "
        "added to the (z_prev, z_curr) input pair at the start of training, annealed to "
        "--noise-in-end by the final epoch. User-directed (2026-08-29): raise this so training "
        "sees meaningfully off-manifold inputs throughout, not just near the true trajectory.",
    )
    parser.add_argument(
        "--noise-in-end", type=float, default=None,
        help="Override Stage2TrainingConfig.noise_in_end (default 0.02): the annealed-to floor "
        "for --noise-in-start.",
    )
    parser.add_argument(
        "--noise-step-start", type=float, default=None,
        help="Override Stage2TrainingConfig.noise_step_start (default 0.0 = off). Gaussian "
        "noise std injected at EVERY autoregressive rollout step during training, not just "
        "the starting pair (that's --noise-in-start). User-directed (2026-08-29): the direct "
        "fix tried after k_max/w_varmatch/noise_in all failed to prevent the fixed-point "
        "collapse -- forces repeated recovery-from-error within even a short rollout, "
        "ported from docs/ML_for_KS_writeup.md's sigma_step.",
    )
    parser.add_argument(
        "--noise-step-end", type=float, default=None,
        help="Override Stage2TrainingConfig.noise_step_end (default 0.0): the annealed-to "
        "floor for --noise-step-start.",
    )
    parser.add_argument(
        "--delta-cap", type=float, default=None,
        help="PropagatorConfig.delta_cap (default None = off). Architectural bound on the "
        "per-step residual: delta = delta_cap * tanh(raw_delta / delta_cap), applied at both "
        "train and eval/inference time (unlike --noise-in/--noise-step, which are training-"
        "only). User-directed (2026-08-29): MSE training on bounded rollouts has no way to "
        "keep predictions bounded except by biasing the map toward contraction everywhere, "
        "which was measured to collapse the propagator to a fixed point (D_KY=0) instead of "
        "chaos (the reference project's own D_KY~21.4). Real chaotic maps stay bounded via "
        "stretch-and-fold, not stretch-and-contract; this gives the model an architectural "
        "fold so training doesn't need to suppress local expansion. Preserves identity-at-"
        "init exactly (tanh(x/c)*c ~= x for small x).",
    )
    parser.add_argument(
        "--init-prop-checkpoint", type=str, default=None,
        help="User-directed (2026-08-30, restructured phased training -- see "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md §2): path to a propagator checkpoint "
        "(the {'prop_state_dict', 'prop_config'} format both this script and "
        "train_stage1_patched.py --full-propagator write) to load and FINE-TUNE, instead "
        "of constructing a fresh, randomly-initialized propagator from this script's own "
        "--backbone/--mode/--n-history/--attn-window/etc CLI flags. Those architecture "
        "flags are IGNORED when this is set -- fine-tuning requires the identical "
        "architecture the checkpoint was trained with. The --k-max/--epochs/--w-varmatch/"
        "--noise-*/--delta-cap schedule flags still apply, now as the fine-tuning "
        "schedule on top of the loaded weights. Default None = unchanged behavior (fresh "
        "propagator).",
    )
    parser.add_argument(
        "--init-pdehead-checkpoint", type=str, default=None,
        help="Path to a pde_head checkpoint (the {'prop_state_dict', 'prop_config'} format "
        "train_stage1_patched.py --pde-distill writes, stage1_pdehead_<profile><tag>.pt) to "
        "load and CONTINUE training alongside the propagator here (added 2026-09-08, "
        "user-directed: 'is the pde_head trained during phase 2 as well. we should try to "
        "get pde rollout to have decent performance'). See train_stage2's own docstring for "
        "the full mechanism -- this Stage-2 continuation is DELIBERATELY MUTUAL (not "
        "detached, unlike Stage 1's own pde_head loss), since Stage 2 never touches the "
        "encoder for a detached target to meaningfully regularize. Default None = no "
        "pde_head (--w-pde-distill/--w-pde-rollout have no effect without this).",
    )
    parser.add_argument(
        "--w-pde-distill", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig.w_pde_distill "
        "(default 0.0, off). 'Option A': single-step match between pde_head.step_one and "
        "the propagator's own realized next state, at EVERY position along its rollout -- "
        "numerically safe (pde_head's own gradient is always single-step).",
    )
    parser.add_argument(
        "--w-pde-rollout", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig.w_pde_rollout "
        "(default 0.0, off). 'Option B': pde_head's OWN autoregressive k_now-step rollout "
        "vs. the propagator's -- the riskier direct approach (see train_stage2's docstring, "
        "Sections 101-106). Either or both of this and --w-pde-distill may be active.",
    )
    parser.add_argument(
        "--w-pde-coeff-l1", type=float, default=None,
        help="--init-pdehead-checkpoint only, and only with a polynomial/chebyshev-field-"
        "kind pde_head. Override Stage2TrainingConfig.w_pde_coeff_l1 (default 0.0, off): "
        "weight on pde_head.body's own poly_coeffs.weight.abs().sum(), an L1 sparsity prior "
        "on the distilled PDE's own coefficients, added 2026-09-10, user-directed: 'I wonder "
        "if we could impose some sparsity using l1 norm on the polynomial coefficients of "
        "the pde'. Raises ValueError if pde_head has no poly_coeffs (field_kind='mlp'). See "
        "--pde-coeff-l1-linear-only to restrict this to only the LINEAR terms.",
    )
    parser.add_argument(
        "--pde-coeff-l1-linear-only", action="store_true",
        help="--w-pde-coeff-l1 only. Sets Stage2TrainingConfig.pde_coeff_l1_linear_only=True "
        "(default False). See train_stage1_patched.py's own --pde-coeff-l1-linear-only help "
        "text -- identical mechanism, restricts the L1 penalty to only w/w_x/w_xx/etc.",
    )
    parser.add_argument(
        "--w-pde-distill-real", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig."
        "w_pde_distill_real (default 0.0, off). A SEPARATE single-step distillation term "
        "against the REAL, FROZEN encoded z sequence (train_sequences/windows thereof) -- "
        "completely independent of the propagator (no z_pred involved at all). Added "
        "2026-09-11, user-directed: 'just train on one step at a time from the true latent "
        "dynamics derived from the encoder. but you can train on the whole batch of latent "
        "states from the encoder at a time'. See Stage1TrainingConfig.w_pde_distill_real's "
        "docstring for the full mechanism and the NOT-the-same-as-fit_latent_pde.py "
        "distinction.",
    )
    parser.add_argument(
        "--pde-distill-real-mutual", action="store_true",
        help="--w-pde-distill-real only. Sets Stage2TrainingConfig.pde_distill_real_detach"
        "=False. A no-op in Stage 2 (the encoder is already frozen, nothing else depends on "
        "the real z sequence's graph) -- kept for CLI/config symmetry with Stage 1.",
    )
    parser.add_argument(
        "--w-pde-distill-real-rollout", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig."
        "w_pde_distill_real_rollout (default 0.0, off). iLED-style (arXiv:2309.05812) "
        "multi-step forecast distillation: pde_head.rollout for "
        "--pde-distill-real-rollout-k steps from a REAL starting state (windows), compared "
        "at every step to REAL future states -- never the propagator's own predictions. "
        "Added 2026-09-12, user-directed: 'directly replicate iLED's stabilization "
        "mechanism and incorporate it into a rollout term to pde_distill loss'. See "
        "Stage1TrainingConfig.w_pde_distill_real_rollout's docstring for the full mechanism.",
    )
    parser.add_argument(
        "--pde-distill-real-rollout-k", type=int, default=4,
        help="--w-pde-distill-real-rollout only. Stage2TrainingConfig."
        "pde_distill_real_rollout_k (default 4). Must be <= --k-max -- raises otherwise.",
    )
    parser.add_argument(
        "--pde-distill-real-rollout-warmup-epochs", type=int, default=None,
        help="--w-pde-distill-real-rollout only. Override Stage2TrainingConfig."
        "pde_distill_real_rollout_warmup_epochs. Default (None) here computes "
        "max(1, round(0.3*epochs)). See train_stage1_patched.py's own flag help text for "
        "why this is not safe to leave at the config's literal default of 0 (found "
        "empirically, 2026-09-12, to diverge on the very first batch and permanently "
        "corrupt the optimizer state).",
    )
    parser.add_argument(
        "--w-pde-nonlinear-l2", type=float, default=None,
        help="--init-pdehead-checkpoint only, and only with pde_head built with "
        "field_kind polynomial/chebyshev. Override Stage2TrainingConfig.w_pde_nonlinear_l2 "
        "(default 0.0, off). iLED-style (arXiv:2309.05812) L_non-linearity: penalizes the "
        "squared magnitude of pde_head's own NONLINEAR-only closure output, evaluated on "
        "real detached states. See Stage1TrainingConfig.w_pde_nonlinear_l2's docstring for "
        "the full mechanism.",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape", type=float, default=None,
        help="--init-pdehead-checkpoint only, mode=markovian only. Override "
        "Stage2TrainingConfig.w_pde_spectrum_shape (default 0.0, off). Added 2026-09-21, "
        "Section 192, user-directed: pde_head analogue of --w-spectrum-shape, applied to "
        "pde_head.step_one specifically -- NEVER the main propagator. See Stage1TrainingConfig."
        "w_pde_spectrum_shape's docstring for the full mechanism and motivation.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-n-expand", type=int, default=None,
        help="--w-pde-spectrum-shape only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_n_expand (default 13).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-expand-target", type=float, default=None,
        help="--w-pde-spectrum-shape only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_expand_target (default 1.1).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-contract-floor", type=float, default=None,
        help="--w-pde-spectrum-shape only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_contract_floor (default 0.6).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_n_samples (default 32).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-two-sided", action="store_true",
        help="--w-pde-spectrum-shape only. Sets Stage2TrainingConfig."
        "pde_spectrum_shape_two_sided (default False).",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-multistep", type=float, default=None,
        help="--init-pdehead-checkpoint only, mode=markovian only. Override "
        "Stage2TrainingConfig.w_pde_spectrum_shape_multistep (default 0.0, off). pde_head "
        "analogue of --w-spectrum-shape-multistep, applied to pde_head.step_one only.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-multistep-k", type=int, default=None,
        help="--w-pde-spectrum-shape-multistep only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_multistep_k (default 10).",
    )
    parser.add_argument(
        "--pde-spectrum-shape-multistep-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape-multistep only. Override Stage2TrainingConfig."
        "pde_spectrum_shape_multistep_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-self", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage2TrainingConfig."
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
        "Override Stage2TrainingConfig.pde_spectrum_shape_self_rollout_k (default 20) -- how "
        "many steps to roll pde_head forward (no_grad) before sampling the evaluation state.",
    )
    parser.add_argument(
        "--pde-spectrum-shape-self-n-samples", type=int, default=None,
        help="--w-pde-spectrum-shape-self/--w-pde-spectrum-shape-multistep-self only. "
        "Override Stage2TrainingConfig.pde_spectrum_shape_self_n_samples (default 16).",
    )
    parser.add_argument(
        "--w-pde-spectrum-shape-multistep-self", type=float, default=None,
        help="--pde-distill only, mode=markovian only. Override Stage2TrainingConfig."
        "w_pde_spectrum_shape_multistep_self (default 0.0, off). SELF-rollout-sampled "
        "analogue of --w-pde-spectrum-shape-multistep -- same self-rollout state pool as "
        "--w-pde-spectrum-shape-self (shares --pde-spectrum-shape-self-rollout-k/-n-samples), "
        "composed --pde-spectrum-shape-multistep-k further steps for the Jacobian check.",
    )
    parser.add_argument(
        "--w-pde-mean-conservation", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig."
        "w_pde_mean_conservation (default 0.0, off). Penalizes pde_head.body.field(z)."
        "mean(dim=-1).pow(2).mean() on real detached states -- true KS conserves int(u)dx "
        "exactly. See Stage1TrainingConfig.w_pde_mean_conservation's docstring for the full "
        "mechanism (Section 164's mode-0 drift finding).",
    )
    parser.add_argument(
        "--w-pde-energy-floor", type=float, default=None,
        help="--init-pdehead-checkpoint only. Override Stage2TrainingConfig."
        "w_pde_energy_floor (default 0.0, off). Penalizes ks_latent.training.losses."
        "spatial_energy_floor_loss on pde_head's OWN unsupervised autoregressive rollout. "
        "See Stage1TrainingConfig.w_pde_energy_floor's docstring for the full mechanism "
        "(Section 167's pde_head-as-propagator Lyapunov spectrum: D_KY=0/lambda1=-0.006, "
        "converges to a fixed point rather than genuine chaos).",
    )
    parser.add_argument(
        "--pde-energy-floor-gamma", type=float, default=0.4,
        help="--w-pde-energy-floor only. Stage2TrainingConfig.pde_energy_floor_gamma "
        "(default 0.4). See train_stage1_patched.py's own flag help text.",
    )
    parser.add_argument(
        "--pde-energy-floor-rollout-k", type=int, default=30,
        help="--w-pde-energy-floor only. Stage2TrainingConfig.pde_energy_floor_rollout_k "
        "(default 30). UNSUPERVISED, not constrained by --k-max.",
    )
    parser.add_argument(
        "--pde-energy-floor-warmup-epochs", type=int, default=None,
        help="--w-pde-energy-floor only. Override Stage2TrainingConfig."
        "pde_energy_floor_warmup_epochs. Default (None) here computes max(1, round(0.3*"
        "epochs)). See train_stage1_patched.py's own flag help text for why this is not "
        "safe to leave at the config's literal default of 0.",
    )
    parser.add_argument(
        "--w-prop-energy-floor", type=float, default=None,
        help="Override Stage2TrainingConfig.w_prop_energy_floor (default 0.0, off). Applies "
        "to the PRIMARY propagator directly (no --init-pdehead-checkpoint needed). See "
        "train_stage1_patched.py's own --w-prop-energy-floor help text for the full "
        "mechanism (Sections 169/171's fixed-point collapse, w_varmatch unable to see far "
        "enough ahead).",
    )
    parser.add_argument(
        "--prop-energy-floor-gamma", type=float, default=0.4,
        help="--w-prop-energy-floor only. Stage2TrainingConfig.prop_energy_floor_gamma "
        "(default 0.4).",
    )
    parser.add_argument(
        "--prop-energy-floor-rollout-k", type=int, default=30,
        help="--w-prop-energy-floor only. Stage2TrainingConfig.prop_energy_floor_rollout_k "
        "(default 30). UNSUPERVISED, not constrained by --k-max.",
    )
    parser.add_argument(
        "--prop-energy-floor-warmup-epochs", type=int, default=None,
        help="--w-prop-energy-floor only. Override Stage2TrainingConfig."
        "prop_energy_floor_warmup_epochs. Default (None) here computes max(1, round(0.3*"
        "epochs)). NOT safe to leave at the config's literal default of 0.",
    )
    parser.add_argument(
        "--stable-linear-lr-factor", type=float, default=None,
        help="Override Stage2TrainingConfig.stable_linear_lr_factor (default 0.02, i.e. "
        "1/50th of --lr). Only has an effect if the loaded propagator checkpoint's "
        "PropagatorConfig.spectral_poly_stable_linear_terms is set (inherited from Stage "
        "1's --spectral-poly-stable-w-xx-w-xxxx -- there is no separate Stage-2-only way "
        "to turn the mechanism itself on, since --init-prop-checkpoint's architecture "
        "always wins over CLI flags here). See train_stage1_patched.py's own "
        "--spectral-poly-stable-w-xx-w-xxxx help text for the full mechanism.",
    )
    parser.add_argument(
        "--w-kernel-unstable-floor", type=float, default=None,
        help="Override Stage2TrainingConfig.w_kernel_unstable_floor (default 0.0, off). "
        "Only has an effect if the loaded propagator exposes kernel_Lhat() (inherited "
        "from Stage 1's --spectral-burgers-kernel-instability -- architecture always wins "
        "over CLI flags here via --init-prop-checkpoint). See train_stage1_patched.py's "
        "own --w-kernel-unstable-floor help text for the full mechanism.",
    )
    parser.add_argument(
        "--kernel-unstable-target-modes", type=int, default=13,
        help="--w-kernel-unstable-floor only. Stage2TrainingConfig.kernel_unstable_target_modes "
        "(default 13).",
    )
    parser.add_argument(
        "--kernel-unstable-margin", type=float, default=0.0,
        help="--w-kernel-unstable-floor only. Stage2TrainingConfig.kernel_unstable_margin "
        "(default 0.0).",
    )
    parser.add_argument(
        "--freeze-propagator", action="store_true",
        help="Requires --init-pdehead-checkpoint (and --init-prop-checkpoint, an "
        "already-trained propagator to distill from). 'Phase 3' (added 2026-09-08, "
        "user-directed: 'should we add a phase 3 that refines the pde with the propagator "
        "fixed? basically how should we extract the pde?') -- the propagator is kept in "
        ".eval() and EXCLUDED from the optimizer entirely; only pde_head trains, with no "
        "competing pressure on the propagator at all. See train_stage2's own docstring for "
        "the full rationale -- this is the actual answer to 'how should we extract the "
        "pde': ordinary teacher-frozen distillation once the propagator is done training. "
        "--w-pde-distill/--w-pde-rollout can be set more aggressively here than in Phases "
        "1-2, since there's no more collapse risk to protect the propagator from.",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="User-directed (2026-08-30), 'Solution 1' for the fixed-point collapse -- see "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 4 and EnsemblePropagatorConfig's "
        "docstring. Wraps the propagator architecture selected by --backbone/--mode/etc into "
        "an EnsemblePropagatorConfig: --ensemble-n-members independently-initialized copies of "
        "that SAME architecture, each perturbed by diagonal Gaussian noise (std proportional to "
        "each latent dimension's own variance, measured from the encoded training set) injected "
        "into its input at every step, combined via a (optionally learned) weighted mean. "
        "Ignored (has no effect) when --init-prop-checkpoint is given -- that loads whatever "
        "architecture (ensemble or not) the checkpoint already has.",
    )
    parser.add_argument(
        "--ensemble-n-members", type=int, default=5,
        help="--ensemble only. EnsemblePropagatorConfig.n_members (default 5).",
    )
    parser.add_argument(
        "--ensemble-noise-std-frac", type=float, default=0.05,
        help="--ensemble only. EnsemblePropagatorConfig.noise_std_frac (default 0.05): each "
        "member's injected per-step noise std, as a fraction of sqrt(Var(z_i)) -- i.e. "
        "noise_std_i = this * sqrt(Var(z_i)), diagonal (no cross-dimension covariance), "
        "measured from the encoded training set right before training starts.",
    )
    parser.add_argument(
        "--ensemble-learned-weights", action="store_true", default=True,
        help="--ensemble only (default on). Combination weights across members are a learned, "
        "z-dependent softmax gate (EnsemblePropagatorConfig.learned_weights=True) rather than a "
        "fixed uniform 1/n_members mean.",
    )
    parser.add_argument(
        "--ensemble-uniform-weights", dest="ensemble_learned_weights", action="store_false",
        help="--ensemble only. Use a fixed uniform 1/n_members mean instead of a learned gate "
        "(EnsemblePropagatorConfig.learned_weights=False).",
    )
    parser.add_argument(
        "--ensemble-gate-hidden", type=int, default=32,
        help="--ensemble --ensemble-learned-weights only. EnsemblePropagatorConfig.gate_hidden "
        "(default 32): hidden width of the 2-layer softmax gating MLP.",
    )
    parser.add_argument(
        "--masked-mlp-warm-start", type=str, default=None,
        help="backbone='masked_mlp' only. Path to a LOCAL (finite --attn-window) masked_mlp "
        "propagator checkpoint (e.g. from a --full-propagator Phase 1 run). Builds a fresh "
        "masked_mlp propagator with attn_window=None (fully dense), copies every weight "
        "directly across (identical shapes by construction -- see MaskedLinear's docstring), "
        "then perturbs the off-band entries (exactly zero in the local checkpoint, since they "
        "never received gradient under its mask) with small Gaussian noise to 'unlock' them "
        "for further training. User-directed 2026-08-30: 'use the local mlp as the auxiliary "
        "propagator to create structure... then a full mlp for the full propagator... "
        "initialize the full mlp with a slightly perturbed version of the local mlp.' Mutually "
        "exclusive with --init-prop-checkpoint (which preserves the SAME architecture; this "
        "deliberately changes it, local -> full).",
    )
    parser.add_argument(
        "--masked-mlp-perturb-std", type=float, default=0.01,
        help="--masked-mlp-warm-start only. Std of the Gaussian noise added to the "
        "previously-exactly-zero off-band entries when unlocking them (default 0.01).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Suffix appended to output artifact filenames (matches "
        "train_stage1_patched.py's --tag), e.g. 'ensemble' -> "
        "stage2_prop_patched_full_ensemble.pt. Keeps comparison runs from "
        "overwriting each other's checkpoints.",
    )
    parser.add_argument(
        "--latent-permutation", type=str, default=None,
        help="See run_diagnostics.py's --latent-permutation help text and "
        "docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 20. Wraps "
        "--ae-checkpoint before it is used to encode training sequences, "
        "so the propagator is trained directly in the permuted latent "
        "coordinate system.",
    )
    parser.add_argument(
        "--latent-permutation-key", type=str, default="d3_permutation",
        help="See run_diagnostics.py's --latent-permutation-key help text.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    global DEVICE
    DEVICE = get_device(args.device)
    print(f"[stage2] device = {DEVICE}", flush=True)

    ae, ae_cfg, train_traj, val_traj = _load_or_train_ae(args)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    shifts = (
        [0, ae_cfg.NX // 4, ae_cfg.NX // 2, 3 * ae_cfg.NX // 4]
        if args.profile == "full"
        else [0, ae_cfg.NX // 2]
    )
    train_seq = encode_dataset_with_shifts(ae, train_traj, shifts, DEVICE)
    val_seq = encode_dataset_with_shifts(ae, val_traj, shifts, DEVICE)

    prop = None
    if args.init_prop_checkpoint is not None and args.masked_mlp_warm_start is not None:
        raise ValueError("--init-prop-checkpoint and --masked-mlp-warm-start are mutually exclusive")
    if args.init_prop_checkpoint is not None:
        prop, prop_cfg, _ = load_propagator_checkpoint(args.init_prop_checkpoint, device=DEVICE)
        print(
            f"[stage2] --init-prop-checkpoint: loaded architecture from "
            f"{args.init_prop_checkpoint} ({type(prop_cfg).__name__}); --backbone/--mode/"
            f"--n-history/--attn-window/--ensemble*/etc CLI flags ignored.",
            flush=True,
        )
    elif args.masked_mlp_warm_start is not None:
        local_prop, local_cfg, _ = load_propagator_checkpoint(args.masked_mlp_warm_start, device="cpu")
        full_cfg = dataclasses.replace(local_cfg, attn_window=None)
        prop = masked_mlp_warm_start(
            local_prop, full_cfg, perturb_std=args.masked_mlp_perturb_std, seed=args.seed
        ).to(DEVICE)
        prop_cfg = full_cfg
        print(
            f"[stage2] --masked-mlp-warm-start: loaded LOCAL architecture from "
            f"{args.masked_mlp_warm_start} (attn_window={local_cfg.attn_window}), built a fresh "
            f"FULL (attn_window=None) masked_mlp propagator, copied all weights, and perturbed "
            f"the off-band entries with std={args.masked_mlp_perturb_std}; --backbone/--mode/"
            f"--attn-window/etc CLI flags ignored.",
            flush=True,
        )
    else:
        if args.backbone in ("transformer", "vit", "fno_vit", "fno_mlp", "local_mlp"):
            n_tokens = args.prop_n_tokens
            if ae_cfg.d_latent % n_tokens != 0:
                raise ValueError(
                    f"d_latent={ae_cfg.d_latent} must be divisible by n_tokens={n_tokens} "
                    f"for --backbone {args.backbone}; pass a checkpoint with a compatible d_latent."
                )
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone=args.backbone,
                n_tokens=n_tokens, token_d_model=args.prop_token_d_model,
                token_nhead=args.prop_token_nhead, token_n_layers=args.prop_token_n_layers,
                attn_window=args.attn_window, pos_encoding=args.pos_encoding, delta_cap=args.delta_cap,
                n_history=args.n_history, token_window=args.token_window,
                fno_modes=args.fno_modes, fno_n_layers=args.fno_n_layers,
            )
        elif args.backbone == "masked_mlp":
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="masked_mlp",
                attn_window=args.attn_window, delta_cap=args.delta_cap,
            )
        elif args.backbone == "node":
            node_kwargs = {}
            if args.hidden is not None:
                node_kwargs["hidden"] = args.hidden
            if args.n_blocks is not None:
                node_kwargs["n_blocks"] = args.n_blocks
            if args.ode_substeps is not None:
                node_kwargs["ode_substeps"] = args.ode_substeps
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="node",
                attn_window=args.attn_window, delta_cap=args.delta_cap,
                **node_kwargs,
            )
        elif args.backbone == "cnn":
            cnn_kwargs = {}
            if args.hidden is not None:
                cnn_kwargs["hidden"] = args.hidden
            if args.n_blocks is not None:
                cnn_kwargs["n_blocks"] = args.n_blocks
            if args.cnn_kernel_size is not None:
                cnn_kwargs["cnn_kernel_size"] = args.cnn_kernel_size
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="cnn",
                delta_cap=args.delta_cap,
                **cnn_kwargs,
            )
        elif args.backbone == "spectral_pde":
            spectral_K = args.spectral_K if args.spectral_K is not None else getattr(ae_cfg, "K", None)
            spectral_N_w = args.spectral_N_w if args.spectral_N_w is not None else getattr(ae_cfg, "N_w", None)
            spectral_L = args.spectral_L if args.spectral_L is not None else getattr(ae_cfg, "L", 100.0)
            if spectral_K is None or spectral_N_w is None:
                raise ValueError(
                    "--backbone spectral_pde requires --spectral-K/--spectral-N-w (or an "
                    "--ae-checkpoint trained with --encoder spectral_field, whose "
                    "SpectralFieldAutoencoderConfig.K/.N_w are used as defaults)"
                )
            spectral_kwargs = {}
            if args.hidden is not None:
                spectral_kwargs["hidden"] = args.hidden
            if args.n_blocks is not None:
                spectral_kwargs["n_blocks"] = args.n_blocks
            if args.ode_substeps is not None:
                spectral_kwargs["ode_substeps"] = args.ode_substeps
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="spectral_pde",
                spectral_K=spectral_K, spectral_N_w=spectral_N_w, spectral_L=spectral_L,
                spectral_max_order=args.spectral_max_order,
                spectral_integrator=args.spectral_integrator,
                spectral_physics_prior=args.spectral_physics_prior,
                spectral_field_kind=args.spectral_field_kind,
                spectral_poly_degree=args.spectral_poly_degree,
                spectral_poly_max_term_order=args.spectral_poly_max_term_order,
                spectral_poly_norm_power=args.spectral_poly_norm_power,
                spectral_poly_stable_leading=args.spectral_poly_stable_leading,
                delta_cap=args.delta_cap,
                **spectral_kwargs,
            )
        elif args.backbone == "fourier_mlp":
            fourier_mlp_kwargs = {}
            if args.hidden is not None:
                fourier_mlp_kwargs["hidden"] = args.hidden
            if args.n_blocks is not None:
                fourier_mlp_kwargs["n_blocks"] = args.n_blocks
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="fourier_mlp",
                delta_cap=args.delta_cap, n_history=args.n_history, fno_modes=args.fno_modes,
                attn_window=args.attn_window,
                fourier_ifft_readout=args.fourier_ifft_readout, nonexpansive=args.nonexpansive,
                **fourier_mlp_kwargs,
            )
        else:
            mlp_kwargs = {}
            if args.hidden is not None:
                mlp_kwargs["hidden"] = args.hidden
            if args.n_blocks is not None:
                mlp_kwargs["n_blocks"] = args.n_blocks
            member_cfg = PropagatorConfig(
                d_latent=ae_cfg.d_latent, mode=args.mode, backbone="mlp", delta_cap=args.delta_cap,
                n_history=args.n_history,
                **mlp_kwargs,
            )
        if args.ensemble:
            prop_cfg = EnsemblePropagatorConfig(
                member=member_cfg, n_members=args.ensemble_n_members,
                noise_std_frac=args.ensemble_noise_std_frac,
                learned_weights=args.ensemble_learned_weights, gate_hidden=args.ensemble_gate_hidden,
            )
        else:
            prop_cfg = member_cfg
    if args.profile == "full":
        full_kwargs = {"latent_loss": args.latent_loss}
        if args.epochs is not None:
            full_kwargs["epochs"] = args.epochs
        if args.k_warmup_epochs is not None:
            full_kwargs["k_warmup_epochs"] = args.k_warmup_epochs
        if args.k_mid is not None:
            full_kwargs["k_mid"] = args.k_mid
            full_kwargs["k_mid_epochs"] = args.k_mid_epochs
        if args.k_max is not None:
            full_kwargs["k_max"] = args.k_max
        if args.compare_k is not None:
            full_kwargs["compare_k"] = args.compare_k
        if args.w_varmatch != 0.0:
            full_kwargs["w_varmatch"] = args.w_varmatch
        if args.w_varmatch_adaptive:
            full_kwargs["w_varmatch_adaptive"] = True
        if args.w_spatial is not None:
            full_kwargs["w_spatial"] = args.w_spatial
        if args.spatial_bandwidth is not None:
            full_kwargs["spatial_bandwidth"] = args.spatial_bandwidth
        if args.spatial_signed:
            full_kwargs["spatial_signed"] = True
        if args.w_lowpass_rollout is not None:
            full_kwargs["w_lowpass_rollout"] = args.w_lowpass_rollout
        if args.lowpass_rollout_power is not None:
            full_kwargs["lowpass_rollout_power"] = args.lowpass_rollout_power
        if args.w_z_lowpass_rollout is not None:
            full_kwargs["w_z_lowpass_rollout"] = args.w_z_lowpass_rollout
        if args.z_lowpass_rollout_power is not None:
            full_kwargs["z_lowpass_rollout_power"] = args.z_lowpass_rollout_power
        if args.z_lowpass_K is not None:
            full_kwargs["z_lowpass_K"] = args.z_lowpass_K
        if args.z_lowpass_L is not None:
            full_kwargs["z_lowpass_L"] = args.z_lowpass_L
        if args.w_logdet_rollout is not None:
            full_kwargs["w_logdet_rollout"] = args.w_logdet_rollout
        if args.logdet_rollout_eps is not None:
            full_kwargs["logdet_rollout_eps"] = args.logdet_rollout_eps
        if args.w_logdet_rollout_latent is not None:
            full_kwargs["w_logdet_rollout_latent"] = args.w_logdet_rollout_latent
        if args.logdet_rollout_latent_eps is not None:
            full_kwargs["logdet_rollout_latent_eps"] = args.logdet_rollout_latent_eps
        if args.w_spectrum_shape is not None:
            full_kwargs["w_spectrum_shape"] = args.w_spectrum_shape
        if args.spectrum_shape_n_expand is not None:
            full_kwargs["spectrum_shape_n_expand"] = args.spectrum_shape_n_expand
        if args.spectrum_shape_expand_target is not None:
            full_kwargs["spectrum_shape_expand_target"] = args.spectrum_shape_expand_target
        if args.spectrum_shape_contract_floor is not None:
            full_kwargs["spectrum_shape_contract_floor"] = args.spectrum_shape_contract_floor
        if args.spectrum_shape_n_samples is not None:
            full_kwargs["spectrum_shape_n_samples"] = args.spectrum_shape_n_samples
        if args.spectrum_shape_two_sided:
            full_kwargs["spectrum_shape_two_sided"] = True
        if args.w_jacobian_bandedness is not None:
            full_kwargs["w_jacobian_bandedness"] = args.w_jacobian_bandedness
        if args.jacobian_bandedness_bandwidth is not None:
            full_kwargs["jacobian_bandedness_bandwidth"] = args.jacobian_bandedness_bandwidth
        if args.jacobian_bandedness_n_samples is not None:
            full_kwargs["jacobian_bandedness_n_samples"] = args.jacobian_bandedness_n_samples
        if args.w_jacobian_diagonal_bound is not None:
            full_kwargs["w_jacobian_diagonal_bound"] = args.w_jacobian_diagonal_bound
        if args.jacobian_diagonal_bound_ceiling is not None:
            full_kwargs["jacobian_diagonal_bound_ceiling"] = args.jacobian_diagonal_bound_ceiling
        if args.jacobian_diagonal_bound_n_samples is not None:
            full_kwargs["jacobian_diagonal_bound_n_samples"] = args.jacobian_diagonal_bound_n_samples
        if args.w_multistep_growth_ceiling is not None:
            full_kwargs["w_multistep_growth_ceiling"] = args.w_multistep_growth_ceiling
        if args.multistep_growth_ceiling_k is not None:
            full_kwargs["multistep_growth_ceiling_k"] = args.multistep_growth_ceiling_k
        if args.multistep_growth_ceiling_value is not None:
            full_kwargs["multistep_growth_ceiling_value"] = args.multistep_growth_ceiling_value
        if args.multistep_growth_ceiling_n_samples is not None:
            full_kwargs["multistep_growth_ceiling_n_samples"] = args.multistep_growth_ceiling_n_samples
        if args.w_multistep_growth_barrier is not None:
            full_kwargs["w_multistep_growth_barrier"] = args.w_multistep_growth_barrier
        if args.multistep_growth_barrier_k is not None:
            full_kwargs["multistep_growth_barrier_k"] = args.multistep_growth_barrier_k
        if args.multistep_growth_barrier_ceiling is not None:
            full_kwargs["multistep_growth_barrier_ceiling"] = args.multistep_growth_barrier_ceiling
        if args.multistep_growth_barrier_epsilon is not None:
            full_kwargs["multistep_growth_barrier_epsilon"] = args.multistep_growth_barrier_epsilon
        if args.multistep_growth_barrier_n_samples is not None:
            full_kwargs["multistep_growth_barrier_n_samples"] = args.multistep_growth_barrier_n_samples
        if args.w_spectrum_shape_self is not None:
            full_kwargs["w_spectrum_shape_self"] = args.w_spectrum_shape_self
        if args.spectrum_shape_self_rollout_k is not None:
            full_kwargs["spectrum_shape_self_rollout_k"] = args.spectrum_shape_self_rollout_k
        if args.spectrum_shape_self_n_samples is not None:
            full_kwargs["spectrum_shape_self_n_samples"] = args.spectrum_shape_self_n_samples
        if args.w_spectrum_shape_graded is not None:
            full_kwargs["w_spectrum_shape_graded"] = args.w_spectrum_shape_graded
        if args.spectrum_shape_graded_reference_path is not None:
            full_kwargs["spectrum_shape_graded_reference_path"] = args.spectrum_shape_graded_reference_path
        if args.spectrum_shape_graded_n_samples is not None:
            full_kwargs["spectrum_shape_graded_n_samples"] = args.spectrum_shape_graded_n_samples
        if args.w_pde_distill is not None:
            full_kwargs["w_pde_distill"] = args.w_pde_distill
        if args.w_pde_rollout is not None:
            full_kwargs["w_pde_rollout"] = args.w_pde_rollout
        if args.w_pde_coeff_l1 is not None:
            full_kwargs["w_pde_coeff_l1"] = args.w_pde_coeff_l1
        if args.pde_coeff_l1_linear_only:
            full_kwargs["pde_coeff_l1_linear_only"] = True
        if args.w_pde_distill_real is not None:
            full_kwargs["w_pde_distill_real"] = args.w_pde_distill_real
        if args.w_pde_distill_real_rollout is not None:
            full_kwargs["w_pde_distill_real_rollout"] = args.w_pde_distill_real_rollout
            full_kwargs["pde_distill_real_rollout_k"] = args.pde_distill_real_rollout_k
            _effective_epochs = args.epochs if args.epochs is not None else 20
            full_kwargs["pde_distill_real_rollout_warmup_epochs"] = (
                args.pde_distill_real_rollout_warmup_epochs
                if args.pde_distill_real_rollout_warmup_epochs is not None
                else max(1, round(0.3 * _effective_epochs))
            )
        if args.w_pde_nonlinear_l2 is not None:
            full_kwargs["w_pde_nonlinear_l2"] = args.w_pde_nonlinear_l2
        if args.w_pde_spectrum_shape is not None:
            full_kwargs["w_pde_spectrum_shape"] = args.w_pde_spectrum_shape
        if args.pde_spectrum_shape_n_expand is not None:
            full_kwargs["pde_spectrum_shape_n_expand"] = args.pde_spectrum_shape_n_expand
        if args.pde_spectrum_shape_expand_target is not None:
            full_kwargs["pde_spectrum_shape_expand_target"] = args.pde_spectrum_shape_expand_target
        if args.pde_spectrum_shape_contract_floor is not None:
            full_kwargs["pde_spectrum_shape_contract_floor"] = args.pde_spectrum_shape_contract_floor
        if args.pde_spectrum_shape_n_samples is not None:
            full_kwargs["pde_spectrum_shape_n_samples"] = args.pde_spectrum_shape_n_samples
        if args.pde_spectrum_shape_two_sided:
            full_kwargs["pde_spectrum_shape_two_sided"] = True
        if args.w_pde_spectrum_shape_multistep is not None:
            full_kwargs["w_pde_spectrum_shape_multistep"] = args.w_pde_spectrum_shape_multistep
        if args.pde_spectrum_shape_multistep_k is not None:
            full_kwargs["pde_spectrum_shape_multistep_k"] = args.pde_spectrum_shape_multistep_k
        if args.pde_spectrum_shape_multistep_n_samples is not None:
            full_kwargs["pde_spectrum_shape_multistep_n_samples"] = args.pde_spectrum_shape_multistep_n_samples
        if args.w_pde_spectrum_shape_self is not None:
            full_kwargs["w_pde_spectrum_shape_self"] = args.w_pde_spectrum_shape_self
        if args.pde_spectrum_shape_self_rollout_k is not None:
            full_kwargs["pde_spectrum_shape_self_rollout_k"] = args.pde_spectrum_shape_self_rollout_k
        if args.pde_spectrum_shape_self_n_samples is not None:
            full_kwargs["pde_spectrum_shape_self_n_samples"] = args.pde_spectrum_shape_self_n_samples
        if args.w_pde_spectrum_shape_multistep_self is not None:
            full_kwargs["w_pde_spectrum_shape_multistep_self"] = args.w_pde_spectrum_shape_multistep_self
        if args.w_pde_mean_conservation is not None:
            full_kwargs["w_pde_mean_conservation"] = args.w_pde_mean_conservation
        if args.w_pde_energy_floor is not None:
            full_kwargs["w_pde_energy_floor"] = args.w_pde_energy_floor
            full_kwargs["pde_energy_floor_gamma"] = args.pde_energy_floor_gamma
            full_kwargs["pde_energy_floor_rollout_k"] = args.pde_energy_floor_rollout_k
            _effective_epochs_energy = args.epochs if args.epochs is not None else 20
            full_kwargs["pde_energy_floor_warmup_epochs"] = (
                args.pde_energy_floor_warmup_epochs
                if args.pde_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * _effective_epochs_energy))
            )
        if args.w_prop_energy_floor is not None:
            full_kwargs["w_prop_energy_floor"] = args.w_prop_energy_floor
            full_kwargs["prop_energy_floor_gamma"] = args.prop_energy_floor_gamma
            full_kwargs["prop_energy_floor_rollout_k"] = args.prop_energy_floor_rollout_k
            _effective_epochs_prop_energy = args.epochs if args.epochs is not None else 20
            full_kwargs["prop_energy_floor_warmup_epochs"] = (
                args.prop_energy_floor_warmup_epochs
                if args.prop_energy_floor_warmup_epochs is not None
                else max(1, round(0.3 * _effective_epochs_prop_energy))
            )
        if args.stable_linear_lr_factor is not None:
            full_kwargs["stable_linear_lr_factor"] = args.stable_linear_lr_factor
        if args.w_kernel_unstable_floor is not None:
            full_kwargs["w_kernel_unstable_floor"] = args.w_kernel_unstable_floor
            full_kwargs["kernel_unstable_target_modes"] = args.kernel_unstable_target_modes
            full_kwargs["kernel_unstable_margin"] = args.kernel_unstable_margin
        if args.pde_distill_real_mutual:
            full_kwargs["pde_distill_real_detach"] = False
        if args.noise_in_start is not None:
            full_kwargs["noise_in_start"] = args.noise_in_start
        if args.noise_in_end is not None:
            full_kwargs["noise_in_end"] = args.noise_in_end
        if args.noise_step_start is not None:
            full_kwargs["noise_step_start"] = args.noise_step_start
        if args.noise_step_end is not None:
            full_kwargs["noise_step_end"] = args.noise_step_end
        train_cfg = Stage2TrainingConfig(**full_kwargs)
    else:
        train_cfg = Stage2TrainingConfig(
            epochs=2, batch_size=8, k_max=4, k_warmup_epochs=1, latent_loss=args.latent_loss
        )
    if prop is None:
        prop = build_propagator_from_config(prop_cfg)
        prop.to(DEVICE)
        if isinstance(prop_cfg, EnsemblePropagatorConfig):
            # Var(z_i) measured from the encoded training set (all shift
            # augmentations, all timesteps) -- see EnsemblePropagatorConfig's
            # docstring: noise must be scaled to each latent dimension's OWN
            # variance, not isotropic unit noise.
            latent_var = train_seq.reshape(-1, prop_cfg.d_latent).var(dim=0)
            prop.set_latent_var(latent_var)
            print(
                f"[stage2] --ensemble: {prop_cfg.n_members} members, "
                f"noise_std_frac={prop_cfg.noise_std_frac}, "
                f"learned_weights={prop_cfg.learned_weights}; latent_var range "
                f"[{latent_var.min().item():.4g}, {latent_var.max().item():.4g}] set from "
                f"the encoded training set.",
                flush=True,
            )

    if args.freeze_propagator and args.init_pdehead_checkpoint is None:
        raise ValueError("--freeze-propagator requires --init-pdehead-checkpoint")

    pde_head = None
    pde_head_cfg = None
    if args.init_pdehead_checkpoint is not None:
        pde_head, pde_head_cfg, _ = load_propagator_checkpoint(args.init_pdehead_checkpoint, device=DEVICE)
        role = (
            "REFINE against the propagator (Phase 3, propagator FROZEN)" if args.freeze_propagator
            else "continue training alongside the propagator"
        )
        print(
            f"[stage2] --init-pdehead-checkpoint: loaded pde_head from "
            f"{args.init_pdehead_checkpoint} ({type(pde_head_cfg).__name__}, "
            f"backbone={pde_head_cfg.backbone!r}); will {role} "
            f"(w_pde_distill={train_cfg.w_pde_distill}, "
            f"w_pde_rollout={train_cfg.w_pde_rollout}).",
            flush=True,
        )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_path = ARTIFACTS_DIR / f"stage2_prop_patched_{args.profile}{suffix}.pt"
    history_path = ARTIFACTS_DIR / f"stage2_history_{args.profile}{suffix}.json"
    print(
        f"[stage2] training {type(prop_cfg).__name__} ({train_cfg.epochs} epochs) "
        f"on {DEVICE} -- per-epoch loss log: {history_path} (written at the end); "
        f"live progress printed below every {args.log_every} epochs.",
        flush=True,
    )

    def _save_periodic_checkpoint(epoch: int) -> None:
        periodic_ckpt_path = ARTIFACTS_DIR / f"stage2_prop_patched_{args.profile}{suffix}_checkpoint.pt"
        torch.save(
            {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg, "epoch": epoch},
            periodic_ckpt_path,
        )
        if pde_head is not None:
            pdehead_periodic_ckpt_path = (
                ARTIFACTS_DIR / f"stage2_pdehead_patched_{args.profile}{suffix}_checkpoint.pt"
            )
            torch.save(
                {"prop_state_dict": pde_head.state_dict(), "prop_config": pde_head_cfg, "epoch": epoch},
                pdehead_periodic_ckpt_path,
            )
        print(f"  [stage2] wrote mid-training checkpoint at epoch {epoch} -> {periodic_ckpt_path}", flush=True)

    t0 = time.time()
    result = train_stage2(
        prop, train_seq, val_seq, train_cfg, DEVICE, verbose=args.verbose, log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        on_epoch_end=_save_periodic_checkpoint if args.checkpoint_every > 0 else None,
        amp=args.amp,
        pde_head=pde_head,
        freeze_propagator=args.freeze_propagator,
    )
    wall_time = time.time() - t0

    torch.save(
        {"prop_state_dict": prop.state_dict(), "prop_config": prop_cfg,
         "best_val_kmax_mse": result.best_val_kmax_mse},
        ckpt_path,
    )
    history_path.write_text(json.dumps(result.train_history, indent=2))
    write_provenance(ckpt_path, config=prop_cfg, seed=args.seed, device=DEVICE.type, wall_time_s=wall_time)

    if pde_head is not None:
        pdehead_ckpt_path = ARTIFACTS_DIR / f"stage2_pdehead_patched_{args.profile}{suffix}.pt"
        torch.save({"prop_state_dict": pde_head.state_dict(), "prop_config": pde_head_cfg}, pdehead_ckpt_path)
        write_provenance(
            pdehead_ckpt_path, config=pde_head_cfg, seed=args.seed, device=DEVICE.type, wall_time_s=wall_time,
        )
        print(f"wrote {pdehead_ckpt_path} (pde_head, continued through Stage 2)")

    print(f"best_val_kmax_mse = {result.best_val_kmax_mse:.6f}")
    if train_cfg.compare_k is not None:
        best_entry = min(result.train_history, key=lambda h: h["val_kmax_mse"])
        print(f"  (at best epoch) val_k{train_cfg.compare_k}_mse = {best_entry[f'val_k{train_cfg.compare_k}_mse']:.6f}")
    print(f"wrote {ckpt_path}, {history_path}")

    # --checkpoint-every's rolling mid-training checkpoint is only crash
    # insurance DURING a run (added 2026-08-31, see that flag's help text) --
    # once we've reached here, ckpt_path above already holds the real final
    # (best-val) propagator, so delete it rather than leave a redundant,
    # confusingly-similar file behind. Only ever runs on a normal
    # (non-crashed) exit, which is exactly when it's safe to do so.
    if args.checkpoint_every > 0:
        stale = ARTIFACTS_DIR / f"stage2_prop_patched_{args.profile}{suffix}_checkpoint.pt"
        if stale.exists():
            stale.unlink()
            print(f"removed superseded mid-training checkpoint {stale}")
        if pde_head is not None:
            stale_pdehead = ARTIFACTS_DIR / f"stage2_pdehead_patched_{args.profile}{suffix}_checkpoint.pt"
            if stale_pdehead.exists():
                stale_pdehead.unlink()
                print(f"removed superseded mid-training checkpoint {stale_pdehead}")


if __name__ == "__main__":
    main()
