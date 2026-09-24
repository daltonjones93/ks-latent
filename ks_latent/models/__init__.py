"""Shared helpers for constructing/loading Stage-1 autoencoders regardless
of architecture (`encoder_kind` in {"transformer", "mlp", "vit"}).

Added 2026-08-29 after catching a real bug: three Gate-3/4 scripts
(`run_analysis_suite.py`, `run_da_pff.py`, `run_diagnostics.py`) hardcoded
`KSAutoencoderPatched` when loading a Stage-1 checkpoint, regardless of
which architecture it was actually trained with -- silently wrong (mismatched
state_dict keys, likely a crash or worse a silent shape-coincidence bug) for
any checkpoint trained with `--encoder mlp` or `--encoder vit`. Every script
that loads a Stage-1 checkpoint should use `load_autoencoder_checkpoint`
below rather than re-deriving this dispatch itself.
"""

from __future__ import annotations

import torch

from ks_latent.config import AuxPropagatorConfig, EnsemblePropagatorConfig, PropagatorConfig
from ks_latent.models.autoencoder_fourier_mlp import KSAutoencoderFourierMLP
from ks_latent.models.autoencoder_local_field import KSAutoencoderLocalField
from ks_latent.models.autoencoder_masked_mlp import KSAutoencoderMaskedMLP
from ks_latent.models.autoencoder_mlp import KSAutoencoderMLP
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched
from ks_latent.models.autoencoder_spectral_field import KSAutoencoderSpectralField
from ks_latent.models.autoencoder_vit import KSAutoencoderViT
from ks_latent.models.autoencoder_vit_fourier_hybrid import KSAutoencoderViTFourierHybrid
from ks_latent.models.ensemble_propagator import EnsemblePropagator
from ks_latent.models.propagator import AuxPropagator, LatentPropagator

_ENCODER_CLASSES = {
    "mlp": KSAutoencoderMLP,
    "vit": KSAutoencoderViT,
    "transformer": KSAutoencoderPatched,
    "masked_mlp": KSAutoencoderMaskedMLP,
    "fourier_mlp": KSAutoencoderFourierMLP,
    "vit_fourier_hybrid": KSAutoencoderViTFourierHybrid,
    "spectral_field": KSAutoencoderSpectralField,
    "local_field": KSAutoencoderLocalField,
}


def build_autoencoder(encoder_kind: str, ae_cfg):
    """Construct the autoencoder class matching `encoder_kind` (as stored
    in a Stage-1 checkpoint's `"encoder_kind"` field). Unknown/missing
    defaults to `"transformer"` (`KSAutoencoderPatched`), matching
    checkpoints written before `encoder_kind` existed."""
    cls = _ENCODER_CLASSES.get(encoder_kind, KSAutoencoderPatched)
    return cls(ae_cfg)


def load_autoencoder_checkpoint(path, device: str | torch.device = "cpu"):
    """Load a Stage-1 checkpoint written by `scripts/train_stage1_patched.py`,
    dispatching on its `encoder_kind`. Returns `(ae, ae_cfg, ckpt)` --
    `ae` has `load_state_dict` already applied; `ckpt` is the full raw dict
    (callers may want `ckpt["val_recon_final"]`, etc.)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ae_cfg = ckpt["ae_config"]
    encoder_kind = ckpt.get("encoder_kind", "transformer")
    ae = build_autoencoder(encoder_kind, ae_cfg)
    sd = ckpt["ae_state_dict"]
    # Legacy compatibility (added 2026-09-23, Section 207): local_field
    # checkpoints trained before Section 153's own "root fix" (see
    # KSAutoencoderLocalField's docstring -- a learnable enc_out bias let
    # the residual channels drift to a large, arbitrary constant, aliasing
    # onto one self-FFT mode) were saved with an enc_out.bias parameter
    # that the CURRENT architecture (bias=False) no longer has. Restore it
    # dynamically ONLY for this exact legacy shape (an unexpected
    # 'enc_out.bias' key on an otherwise-matching local_field model) so a
    # pre-fix checkpoint's actually-trained bias loads faithfully instead
    # of the whole load failing outright or the value being silently
    # dropped via strict=False.
    if (
        encoder_kind == "local_field"
        and "enc_out.bias" in sd
        and getattr(ae, "enc_out", None) is not None
        and ae.enc_out.bias is None
    ):
        ae.enc_out.bias = torch.nn.Parameter(torch.zeros(ae.enc_out.out_channels))
    ae.load_state_dict(sd)
    # `build_autoencoder` constructs fresh (CPU) parameters, and
    # `load_state_dict` copies values in place without moving device -- so
    # without this, `ae` silently stays on CPU even when `device` is mps/cuda
    # (caught 2026-08-29 relaunching Stage-2 training on MPS: "Tensor for
    # argument weight is on cpu but expected on mps").
    ae = ae.to(device)
    return ae, ae_cfg, ckpt


def build_propagator_from_config(prop_cfg):
    """Construct the propagator class matching `prop_cfg`'s type (added
    2026-08-30, same bug class `load_autoencoder_checkpoint` above was
    added to fix: a propagator checkpoint's config fully determines which
    class it belongs to, so no caller should hardcode `LatentPropagator`
    directly). `EnsemblePropagatorConfig` -> `EnsemblePropagator`;
    `AuxPropagatorConfig` -> `AuxPropagator`; `PropagatorConfig` (or
    anything else) -> `LatentPropagator`."""
    if isinstance(prop_cfg, EnsemblePropagatorConfig):
        return EnsemblePropagator(prop_cfg)
    if isinstance(prop_cfg, AuxPropagatorConfig):
        return AuxPropagator(prop_cfg)
    return LatentPropagator(prop_cfg)


def load_propagator_checkpoint(path, device: str | torch.device = "cpu"):
    """Load a propagator checkpoint (the `{"prop_state_dict", "prop_config"}`
    format written by `train_stage2_patched.py` and
    `train_stage1_patched.py --full-propagator`), dispatching on
    `prop_config`'s type via `build_propagator_from_config`. Returns `(prop,
    prop_cfg, ckpt)` -- `prop` has `load_state_dict` already applied and has
    been moved to `device` (see `load_autoencoder_checkpoint`'s docstring
    for why that move is not automatic)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    prop_cfg = ckpt["prop_config"]
    prop = build_propagator_from_config(prop_cfg)
    prop.load_state_dict(ckpt["prop_state_dict"])
    prop = prop.to(device)
    return prop, prop_cfg, ckpt
