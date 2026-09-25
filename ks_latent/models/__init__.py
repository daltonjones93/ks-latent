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


def load_autoencoder_checkpoint_resized(
    path, new_n_sites: int, new_NX: int, device: str | torch.device = "cpu",
):
    """Phase F2/F3 L-transfer (`docs/steps_4-3.md`, added 2026-09-25,
    user-directed: "run Phase F and run a larger L without retraining").
    Loads a `local_field` checkpoint's TRAINED WEIGHTS into a freshly
    constructed model at a DIFFERENT `n_sites`/`NX` -- this is the literal
    mechanism the L-transfer claim rests on, not a new architecture.

    Only valid for `encoder_kind == "local_field"` (raises otherwise --
    no other architecture in this codebase has size-independent weight
    shapes). Every other `LocalFieldAutoencoderConfig` field
    (`local_channels`, `site_mix_radius`, `n_site_mix_layers`, `hidden`)
    is carried over UNCHANGED from the checkpoint -- only `n_sites`/`NX`
    differ.

    **Why this works with a plain `load_state_dict(strict=True)` and no
    reshaping/interpolation of any parameter**: every learned layer in
    `KSAutoencoderLocalField` is either a `Conv1d`/`ConvTranspose1d`
    whose kernel width is `patch_size` (`enc_patchify`/`dec_unpatchify`)
    or `2*site_mix_radius+1` (`enc_mix`/`dec_mix`), or a `1x1` conv
    (`enc_out`/`dec_in`) -- NONE of these shapes depend on `n_sites` or
    `NX` at all, only on `patch_size` (`= NX/n_sites`, held fixed by
    construction here -- see below), `site_mix_radius`, and `hidden`.
    Verified directly before this function was written: loading Section
    224's exact trained `state_dict` into a `n_sites=32, NX=512` model
    (double the trained `n_sites=16, NX=256`) succeeds with zero missing
    or unexpected keys.

    **Caller's responsibility, not checked here beyond the assertion
    below**: `new_NX / new_n_sites` MUST equal the checkpoint's own
    trained `patch_size` (`NX/n_sites`) for the loaded conv weights to
    mean the same thing physically -- e.g. doubling `L` while holding
    `dx = L/NX` fixed doubles `NX` and (since `patch_size` stays fixed)
    doubles `n_sites` too, both by the SAME factor. Passing an
    `new_NX`/`new_n_sites` pair with a different `patch_size` would still
    load (shapes still match) but would silently change what the
    patchify convolution sees per site -- not a genuine L-transfer test.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder_kind = ckpt.get("encoder_kind", "transformer")
    if encoder_kind != "local_field":
        raise ValueError(
            f"load_autoencoder_checkpoint_resized only supports encoder_kind='local_field' "
            f"(size-independent weight shapes) -- got {encoder_kind!r}"
        )
    old_cfg = ckpt["ae_config"]
    old_patch_size = old_cfg.NX // old_cfg.n_sites
    new_patch_size = new_NX // new_n_sites
    if new_patch_size != old_patch_size:
        raise ValueError(
            f"new_NX/new_n_sites={new_NX}/{new_n_sites}={new_patch_size} must equal the "
            f"checkpoint's own trained patch_size={old_patch_size} ({old_cfg.NX}/{old_cfg.n_sites}) "
            "for this to be a genuine L-transfer (same physical content per patchify site) -- "
            "got a different patch_size, which would silently change what each site sees."
        )
    from ks_latent.config import LocalFieldAutoencoderConfig
    new_cfg = LocalFieldAutoencoderConfig(
        NX=new_NX, n_sites=new_n_sites, local_channels=old_cfg.local_channels,
        site_mix_radius=old_cfg.site_mix_radius, n_site_mix_layers=old_cfg.n_site_mix_layers,
        hidden=old_cfg.hidden,
    )
    ae = KSAutoencoderLocalField(new_cfg)
    ae.load_state_dict(ckpt["ae_state_dict"], strict=True)
    ae = ae.to(device)
    return ae, new_cfg, ckpt


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


def load_propagator_checkpoint_resized(
    path, new_d_latent: int, new_n_tokens: int, device: str | torch.device = "cpu",
):
    """Phase F2/F3 L-transfer companion to `load_autoencoder_checkpoint_
    resized` (added 2026-09-25) -- loads a `backbone="local_mlp"`
    propagator's trained weights into a freshly constructed model at a
    different `d_latent`/`n_tokens`.

    Only `backbone="local_mlp"` is supported (raises otherwise): its
    layers (`token_embed`/`token_unembed`: `Linear(chunk_size,
    token_d_model)`/`Linear(token_d_model, chunk_size)`; each
    `_LocalMixerBlock`'s `Conv1d(token_d_model, token_d_model,
    kernel=2*attn_window+1)`) all have shapes depending on `chunk_size
    (= d_latent/n_tokens)`, `token_d_model`, and `attn_window` -- NONE on
    `n_tokens` itself. Verified directly before this function was
    written: Section 224's exact trained `state_dict` loads with zero
    missing/unexpected keys into a `d_latent=96, n_tokens=32` model
    (double the trained `d_latent=48, n_tokens=16`). `masked_mlp`/
    `masked_mlp_expand`/`mlp` do NOT have this property (their layers are
    masked or plain DENSE matrices tied to `d_latent` -- see
    `docs/RESULTS.md`'s Section 221 writeup for the direct parameter-
    count measurement that ruled them out) and are explicitly rejected
    here rather than silently producing a shape-mismatch crash deeper in
    `load_state_dict`.

    **Caller's responsibility**: `new_d_latent/new_n_tokens` MUST equal
    the checkpoint's own trained `chunk_size` (`d_latent/n_tokens`), same
    reasoning as `load_autoencoder_checkpoint_resized`'s `patch_size`
    check -- `d_latent` should scale with `n_sites` (hence with the
    autoencoder's own `new_n_sites`) at a FIXED `local_channels`, so
    `new_n_tokens` should equal the autoencoder's own `new_n_sites`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    old_cfg = ckpt["prop_config"]
    if getattr(old_cfg, "backbone", None) != "local_mlp":
        raise ValueError(
            f"load_propagator_checkpoint_resized only supports backbone='local_mlp' "
            f"(verified size-independent weight shapes) -- got {getattr(old_cfg, 'backbone', None)!r}. "
            "masked_mlp/masked_mlp_expand/mlp have layers tied to d_latent and cannot be resized."
        )
    old_chunk_size = old_cfg.d_latent // old_cfg.n_tokens
    new_chunk_size = new_d_latent // new_n_tokens
    if new_chunk_size != old_chunk_size:
        raise ValueError(
            f"new_d_latent/new_n_tokens={new_d_latent}/{new_n_tokens}={new_chunk_size} must equal "
            f"the checkpoint's own trained chunk_size={old_chunk_size} "
            f"({old_cfg.d_latent}/{old_cfg.n_tokens}) for this to be a genuine L-transfer."
        )
    import dataclasses
    new_cfg = dataclasses.replace(old_cfg, d_latent=new_d_latent, n_tokens=new_n_tokens)
    prop = build_propagator_from_config(new_cfg)
    prop.load_state_dict(ckpt["prop_state_dict"], strict=True)
    prop = prop.to(device)
    return prop, new_cfg, ckpt
