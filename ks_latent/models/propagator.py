"""Latent propagators (brief §5.2, PROJECT_HANDOFF.md; markovian mode added
per brief §5.2 addendum, 2026-08-29).

Two modes, both exposed through the same `LatentPropagator` class via
`cfg.mode` (a single flag both the model and the training loops read, so
they can never disagree about which map is in use):

- `"two_step"` (original design): `(z_{n-1}, z_n) -> z_{n+1}`, giving the
  model a discrete "velocity proxy" via the two-step history.
- `"markovian"` (addendum): `M(z_n) -> z_{n+1}`, using only the current
  state. Motivation: the KS PDE is first order in time
  (`du/dt = -u*u_x - u_xx - u_xxxx`, no second time derivative anywhere),
  so a sufficiently informative encoding of `u(x,t)` alone should in
  principle already be Markovian, making the two-step history an
  unnecessary (and possibly confounding) crutch. `mode="markovian"` tests
  that directly. Three backbones for `M` (`"two_step"` only ever uses
  `"mlp"`):
    - `"mlp"`: the same residual-MLP pattern as `two_step`, just with a
      `d`-dim rather than `2d`-dim input.
    - `"transformer"`: tokenizes `z` into `n_tokens` chunks, adds a
      *learned* absolute positional embedding, runs a couple of
      `nn.TransformerEncoderLayer` blocks (optionally with a banded/local
      attention mask), then projects back down. "Local" here means
      *adjacent latent channel indices*, which is an arbitrary ordering
      for the current flat global latent (there is no physical-space
      meaning to being "near" in index) -- it becomes a genuinely local
      *spatial* window only once Phase 10's spatially-organized latent
      field exists. Included as an architecture option to try, not a
      claim of physical locality.
    - `"vit"` (added 2026-08-29, user-directed): tokenizes `z` the same
      way, but reuses `ks_latent.models.autoencoder_vit`'s
      `CircularPositionalEncoding` (fixed Fourier ring embedding, not
      learned) and `ViTBlock` (pre-norm attention + `mlp_ratio`-expansion
      MLP) -- the same architecture as `KSAutoencoderViT`'s encoder/decoder
      -- applied directly to the tokenized latent **with no
      pooling/bottleneck step**: the token grid keeps its full
      `(n_tokens, token_d_model)` shape through every block, unlike
      `KSAutoencoderViT` which collapses all tokens to one vector and
      re-expands. It is a plain same-shape seq2seq transformer over the
      tokens, not an autoencoder. See `_ViTDeltaBody`.

`AuxPropagator` is the same `LatentPropagator` class, just built from
`AuxPropagatorConfig`'s (smaller) sizing -- both share this module so the
"which map" flag can never drift out of sync between Stage-1's auxiliary
propagator and Stage-2's real one.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.config import AuxPropagatorConfig, PropagatorConfig
from ks_latent.models.autoencoder_masked_mlp import MaskedLinearRect
from ks_latent.models.spectral_field import (
    decode_from_spectrum, encode_to_spectrum, rfft_wavenumbers, synthesize_derivatives,
)
from ks_latent.models.autoencoder_vit import (
    CircularPositionalEncoding,
    FNOLayer,
    LinearPositionalEncoding,
    TokenMLPBlock,
    ViTBlock,
    apply_fno_layers,
    build_local_attention_mask,
    build_ring_local_attention_mask,
    circular_overlap_tokenize,
)
from ks_latent.models.spectral_norm import _sn


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual block: `x + Linear(GELU(Linear(LayerNorm(x))))`.

    `nonexpansive` (added 2026-09-04, user-directed: "would there be a way
    to constrain the fourier_mlp to be nonexpansive" -- ViT's attention is
    structurally non-expansive, softmax outputs being convex combinations
    of value vectors; this gives `fourier_mlp`'s plain linear layers an
    analogous guarantee). `False` (default, unchanged behavior): as
    above. `True`: `fc1`/`fc2` are spectral-normalized (`_sn`, operator
    norm <=1 each) and `LayerNorm` is DROPPED (not uniformly 1-Lipschitz
    -- dividing by a small per-sample norm can amplify near-zero inputs,
    so it would silently break the guarantee this option exists to
    provide). Critically, the residual itself changes from `x + h` to
    `0.5*(x + h)`: a PLAIN residual is NOT non-expansive even when `h`'s
    own branch is -- Lipschitz constants add under a sum, so `x+h` can
    have Lipschitz constant up to `1+Lip(h)` -- while `0.5*(x+h)` is a
    genuine convex combination of `x` and `h`, giving `Lip <= max(Lip(x),
    Lip(h)) <= 1` by the triangle inequality whenever `Lip(h)<=1`. This is
    the same mechanism (convex combination, not addition) that makes ViT
    attention's own residual-free mixing bounded."""

    def __init__(self, dim: int, dropout: float = 0.0, nonexpansive: bool = False):
        super().__init__()
        self.nonexpansive = nonexpansive
        if nonexpansive:
            self.ln = None
            self.fc1 = _sn(dim, dim)
            self.fc2 = _sn(dim, dim)
        else:
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x) if self.ln is not None else x
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        if self.nonexpansive:
            return 0.5 * (x + h)
        return x + h


class MLPDeltaBody(nn.Module):
    """`Linear(in_dim->H) -> n_blocks x ResidualMLPBlock(H) -> LayerNorm -> Linear(H->d)`.

    Shared by `two_step` (`in_dim=2*d`) and `markovian`+`mlp` (`in_dim=d`).

    Public (renamed from `_MLPDeltaBody`, 2026-09-03), same cross-module
    reuse reason as `autoencoder_vit.py`'s `FNOLayer`/`TokenMLPBlock`/etc:
    `ks_latent.models.autoencoder_fourier_mlp.KSAutoencoderFourierMLP`
    reuses this exact class as its encoder's and decoder's body (despite
    "delta" in the name, it's a generic `in_dim -> hidden... -> out_dim`
    residual MLP with no built-in assumption of a residual/delta
    semantics -- that convention lives in the CALLER, e.g.
    `LatentPropagator.step_one`'s `z + capped_delta(self.body(z))`).

    `nonexpansive` (added 2026-09-04, see `ResidualMLPBlock`'s docstring):
    `input_proj` is spectral-normalized and every block uses the damped
    residual, so the `input_proj -> blocks` portion is genuinely <=1-
    Lipschitz end to end (composition of non-expansive maps is non-
    expansive). `final_ln` is dropped (same reasoning as `ResidualMLPBlock`).
    `output_proj` is deliberately left as an ORDINARY (non-spectral-
    normalized) `Linear`, NOT wrapped -- spectral_norm divides by the
    weight's estimated largest singular value, which is exactly zero for
    an all-zero weight matrix (0/0 = NaN), so it is fundamentally
    incompatible with `zero_init`'s exact-zero initialization, which this
    codebase relies on throughout for the propagator's identity-at-init
    property. This is an accepted, documented compromise -- the DEPTH
    (the part where repeated composition could compound amplification)
    is fully constrained; only the single final linear readout isn't,
    which contributes at most a constant (non-compounding) factor. The
    actual empirical effect is verified directly via `analyze_75_76_77_
    latent_geometry.py`'s propagator-independent local-sensitivity
    diagnostic, not claimed as a formal proof."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden: int, n_blocks: int, dropout: float, zero_init: bool,
        nonexpansive: bool = False,
    ):
        super().__init__()
        self.input_proj = _sn(in_dim, hidden) if nonexpansive else nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden, dropout, nonexpansive=nonexpansive) for _ in range(n_blocks)]
        )
        self.final_ln = nn.Identity() if nonexpansive else nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, out_dim)
        if zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.output_proj(x)


class FourierIFFTBody(nn.Module):
    """Fourier-feature `MLPDeltaBody` whose OUTPUT is interpreted as raw
    frequency-domain coefficients and explicitly inverse-transformed
    (`irfft`) back to state space, instead of a generic dense linear
    readout directly to state space -- added 2026-09-03, user-directed:
    "apply an inverse fft to the frequency component output of the
    fourier mlp to map back to state space in the encoder and propagator
    and decoder". Public, reused across `ks_latent.models.propagator`'s
    own `_FourierMLPHistoryDeltaBody` (`backbone="fourier_mlp"`'s
    "fourier path") AND `ks_latent.models.autoencoder_fourier_mlp`'s
    `KSAutoencoderFourierMLP` encoder/decoder Fourier paths -- same
    cross-module reuse pattern as `MLPDeltaBody`/`MaskedMLPResidualBlock`.

    Differs from `FourierMLPAutoencoderConfig.dec_use_ifft` (2026-09-03,
    Section 73, superseded by this for the decoder role): that scheme fed
    the RAW LATENT directly through `irfft` with no MLP in frequency
    space at all. This class instead runs an `MLPDeltaBody` ON the input's
    `rfft` FEATURES, predicts `2*out_modes` numbers (real then imaginary
    parts of `out_modes` output-spectrum coefficients), and inverse-
    transforms THAT prediction -- i.e. the MLP does real work IN frequency
    space (a learned per-mode operator) before the explicit, mathematically
    exact `irfft` maps back to physical/state space, rather than an
    arbitrary unconstrained linear layer learning that mapping itself.

    Tried and reverted same-day (2026-09-03): a version where this body's
    INTERNALS were made genuinely complex-valued (`nn.Linear(...,
    dtype=torch.cfloat)` + a `ModReLU` nonlinearity, user-directed: "make
    the parameters and inputs to the first part of the mlp ... imaginary
    ... mod relu nonlinearity"). Even after adding a stabilizing
    magnitude-based pre-norm (plain complex `Linear`+`ModReLU` residual
    blocks turned out not to be norm-preserving, ~1.2x RMS growth per
    block measured directly), that version still diverged progressively
    over a realistic training horizon (30-epoch smoke test: loss
    2.18 -> 84,830 -> 845,080 -> 646,646, `val_recon_final~=1145`, with
    gradient clipping confirmed active and correctly handling complex
    gradients throughout) -- reverted back to this real-valued form,
    user-directed: "what if we just expanded each imaginary number into
    an element in R^2 so x+iy gets mapped to <x,y>. then we can use the
    real valued fourier body but still have the benefit of complex
    numbers. we can then collapse the <x,y> vector to x+yi before the
    irfft" -- which is exactly this class's existing real/imag-concat-
    then-split design below (mathematically: concatenating real/imag
    halves vs. interleaving `<x,y>` pairs are the same information handed
    to a fully-connected `MLPDeltaBody` up to a fixed input-column
    permutation, which a dense `Linear` layer can represent identically
    either way).

    `out_modes` defaults to the full available spectrum of `out_len`
    (`out_len // 2 + 1`) if not given, matching this file's existing
    "`None` = full spectrum" convention elsewhere. `zero_init` (passed
    through to the internal `MLPDeltaBody`) zeros the predicted
    frequency coefficients at init, and `irfft` of an all-zero spectrum is
    exactly zero -- so this preserves the same identity-at-init property
    the plain `MLPDeltaBody`-based Fourier path had for the propagator's
    residual/delta role.

    `nonexpansive` (added 2026-09-04, see `ResidualMLPBlock`'s docstring):
    threaded into the internal `MLPDeltaBody`, AND switches this class's
    own `irfft` to `norm="ortho"` (an isometry by Parseval's theorem --
    `torch.fft.irfft`'s default "backward" normalization is NOT norm-
    preserving, so leaving it unnormalized would undermine the whole
    guarantee at exactly the last step). Callers are responsible for
    ALSO using `norm="ortho"` on the matching forward `rfft` that computed
    `feats` in the first place (see `_fourier_features`'s `nonexpansive`
    parameter) -- this class only controls its own `irfft`."""

    def __init__(
        self, in_modes: int, out_len: int, hidden: int, n_blocks: int, dropout: float,
        zero_init: bool, out_modes: int | None = None, nonexpansive: bool = False,
    ):
        super().__init__()
        self.out_len = out_len
        self.out_modes = out_modes if out_modes is not None else out_len // 2 + 1
        self.nonexpansive = nonexpansive
        self.body = MLPDeltaBody(
            2 * in_modes, 2 * self.out_modes, hidden, n_blocks, dropout, zero_init, nonexpansive=nonexpansive,
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """`feats`: `(B, 2*in_modes)` -- rfft features of some input.
        Returns `(B, out_len)`: the `irfft`-reconstructed physical/state-
        space signal. Explicit `float()` upcast before `irfft` (same
        bfloat16-under-autocast bug class as `_fourier_features`/
        `SpectralConv1d`/etc -- `torch.fft.irfft` doesn't support
        bfloat16 and requires a complex input)."""
        orig_dtype = feats.dtype
        raw = self.body(feats).float()
        real, imag = raw[:, : self.out_modes], raw[:, self.out_modes :]
        coeffs = torch.complex(real, imag)
        norm = "ortho" if self.nonexpansive else None
        out = torch.fft.irfft(coeffs, n=self.out_len, dim=-1, norm=norm)
        return out.to(orig_dtype)


def _circular_pad_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Pad `x` (`B, C, d_latent`) circularly for a SAME-length `Conv1d`
    with `padding=0`, for arbitrary (odd or even) `kernel_size` -- unlike
    `nn.Conv1d(..., padding_mode="circular")`'s own built-in padding
    (which only gives exact same-length output for ODD kernel sizes: a
    single scalar `padding` pads both sides equally, so an even
    `kernel_size` -- e.g. this backbone's requested width `16` -- would
    return `d_latent + 1` outputs, silently misaligned). Splits the
    required `kernel_size - 1` total padding asymmetrically
    (`(kernel_size-1)//2` left, `kernel_size//2` right) so the output
    length always exactly matches the input."""
    pad_left = (kernel_size - 1) // 2
    pad_right = kernel_size // 2
    return F.pad(x, (pad_left, pad_right), mode="circular")


class _CNNResidualBlock1D(nn.Module):
    """CNN analogue of `ResidualMLPBlock`, porting the same "best features"
    (pre-norm, `Conv -> GELU -> Dropout -> Conv`, residual, no bias-shift
    surprises) but with both `Linear`s replaced by circular `Conv1d`s
    along the latent-INDEX axis -- see `_CNNDeltaBody`'s docstring.
    `LayerNorm(hidden)` normalizes over the CHANNEL axis only (`x` is
    channels-last, `(B, d_latent, hidden)`, so `LayerNorm`'s last-dim
    normalization is per spatial position) -- unlike
    `MaskedMLPResidualBlock`'s deliberate avoidance of `LayerNorm`
    (there, `dim` WAS the spatial axis, so normalizing over it would
    reintroduce global coupling); here the spatial axis is untouched by
    the norm, so it stays local/correct."""

    def __init__(self, hidden: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, d_latent, hidden)`, channels-last."""
        h = self.ln(x).transpose(1, 2)  # (B, hidden, d_latent) for Conv1d
        h = self.conv1(_circular_pad_1d(h, self.kernel_size))
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(_circular_pad_1d(h, self.kernel_size))
        return x + h.transpose(1, 2)


class _CNNDeltaBody(nn.Module):
    """`backbone="cnn"` (added 2026-09-01, user-directed: "try a CNN for a
    propagator too... implement this based on the best features of the
    current MLP... 1d in the spatial dimension, with width 16"). Mirrors
    `MLPDeltaBody`'s exact structure -- `input_proj -> n_blocks residual
    blocks -> final LayerNorm -> output_proj (zero-init)` -- with every
    `Linear` that mixes across the latent-INDEX axis replaced by a
    circular `Conv1d` of kernel size `cnn_kernel_size` (default `16`, a
    literal kernel WIDTH, unlike `local_mlp`/`masked_mlp`/`node`'s
    `attn_window`, which is a RADIUS -- `2*window+1`). `input_proj`/
    `output_proj` stay per-position `Linear(1, hidden)`/`Linear(hidden,
    1)` (a "1x1 conv" in effect), so only the residual blocks' `Conv1d`s
    do any cross-index mixing.

    Genuinely translation-equivariant (weight-shared `Conv1d` kernels,
    same construction as `node`'s `_LocalVectorField`) -- unlike
    `masked_mlp`'s per-position-independent masked weights. Unlike
    `node`, this is a single discrete-jump map (`z_n -> z_n+1` directly,
    `n_blocks` sequential conv blocks, no RK4 sub-stepping), so its op
    count per forward pass is `O(n_blocks)` rather than `node`'s
    `O(ode_substeps * 4 * n_blocks)` -- deliberately cheap on MPS after
    `node`'s wide-kernel (`attn_window=6`) circular-padding op-dispatch
    cost proved catastrophic (Section 43's `node`/`window=6` run: ~3900s/
    epoch, vs. `window=2`'s ~94s/epoch -- a 42x jump from what should have
    been a ~2.6x compute increase, implicating something pathological in
    MPS's handling of wider circular padding at that op count). This
    backbone's much lower total op count is deliberately chosen to be
    more robust to that risk, but was NOT proven safe until directly
    timed -- see the smoke-timing check before any real launch.

    Zero-init (`zero_init=True`, default): `output_proj`'s weight/bias
    are zeroed, so this backbone returns delta=0 at init, matching every
    other backbone's identity-at-init property."""

    def __init__(
        self, d_latent: int, hidden: int, n_blocks: int, kernel_size: int, dropout: float, zero_init: bool
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden)
        self.blocks = nn.ModuleList(
            [_CNNResidualBlock1D(hidden, kernel_size, dropout) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, 1)
        if zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `(B, d_latent)`."""
        x = self.input_proj(z.unsqueeze(-1))  # (B, d_latent, hidden)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.output_proj(x).squeeze(-1)  # (B, d_latent)


class _SiteConvDeltaBody(nn.Module):
    """`backbone="site_conv"` (added 2026-09-24, Section 222, user-
    directed: "could we use a model like the encoder from 221 as a
    propagator? I've never thought of trying that"). Reuses
    `KSAutoencoderLocalField`'s own circular-`Conv1d` site-mixing design
    (`ks_latent/models/autoencoder_local_field.py`) almost verbatim, but
    as a DIMENSION-PRESERVING `z -> z` map instead of that encoder's
    `u (NX) -> z (n_sites*local_channels)` compression: `z` is reshaped
    to its native `(n_sites, local_channels)` field (site-major, matching
    `KSAutoencoderLocalField`'s own flattening convention exactly), a
    `1x1 Conv1d` projects `local_channels -> hidden` (mirroring the
    encoder's own `dec_in`), `n_layers` circular `Conv1d(hidden, hidden,
    kernel_size=2*radius+1, padding=radius, padding_mode='circular')` +
    GELU mix ACROSS SITES (mirroring `enc_mix`/`dec_mix` exactly -- plain
    sequential GELU convs, no residual skip, matching that architecture's
    own already-validated design rather than adding an unrequested
    change), and a final `1x1 Conv1d(hidden, local_channels, bias=False)`
    projects back down, zero-initialized for this backbone's
    identity-at-init property.

    Unlike the existing `backbone="cnn"` (`_CNNDeltaBody`), which
    convolves over the RAW `d_latent` index as one flat sequence with an
    embedded `hidden` width unrelated to any site structure, this
    backbone is built specifically for a `local_field`-encoded latent: it
    reshapes `z` into its TRUE `(n_sites, local_channels)` grid first and
    only mixes across the SITE axis, treating `local_channels` the same
    way the encoder treats its own per-site channel count -- i.e. this is
    the encoder's own architecture, reused as a dynamics model on the
    space it already knows how to represent, not a generic conv over an
    arbitrary flat vector.

    `bias=False` on the final projection (added deliberately, not an
    oversight): `KSAutoencoderLocalField.enc_out` had exactly this same
    `bias=False` fix applied for a documented reason (see that class's
    own docstring) -- an unconstrained per-channel additive bias in a
    convolutional output layer is otherwise free to drift to an arbitrary
    constant offset during training, since none of this project's usual
    anti-collapse losses (`w_var`/`w_logdet`/`w_decorr`) constrain the
    MEAN of `z`, and a site-major-periodic offset pattern aliases onto a
    single spurious self-FFT mode (`KSAutoencoderLocalField`'s own
    docstring has the full measured finding). Applying the same fix here
    preemptively avoids re-discovering that failure mode a second time in
    a structurally identical architecture.

    `radius` (reuses `attn_window`, same convention as `local_mlp`/
    `masked_mlp`/`node`) and `n_layers`/`hidden` (`site_conv_n_layers`/
    `site_conv_hidden`) default to `LocalFieldAutoencoderConfig`'s OWN
    defaults (`site_mix_radius=2`, `n_site_mix_layers=3`, `hidden=32`) --
    not independently tuned, deliberately reusing parameters already
    validated (by the encoder's own reconstruction quality) rather than
    guessing fresh ones for an architecture being tried as a propagator
    for the first time. `n_sites`/`local_channels` (`site_conv_n_sites`/
    `site_conv_local_channels`) MUST match the `local_field` encoder this
    propagator is paired with exactly (`d_latent == n_sites *
    local_channels`, checked in `PropagatorConfig.__post_init__`) -- this
    backbone has no meaning for a non-spatially-organized latent."""

    def __init__(
        self, n_sites: int, local_channels: int, hidden: int, n_layers: int, radius: int, zero_init: bool,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.local_channels = local_channels
        self.proj_in = nn.Conv1d(local_channels, hidden, kernel_size=1)
        self.mix = nn.ModuleList(
            [
                nn.Conv1d(hidden, hidden, kernel_size=2 * radius + 1, padding=radius, padding_mode="circular")
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Conv1d(hidden, local_channels, kernel_size=1, bias=False)
        if zero_init:
            nn.init.zeros_(self.proj_out.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, n_sites*local_channels)`, site-major -> same shape."""
        B = z.shape[0]
        x = z.reshape(B, self.n_sites, self.local_channels).transpose(1, 2)  # (B, local_channels, n_sites)
        h = F.gelu(self.proj_in(x))
        for conv in self.mix:
            h = F.gelu(conv(h))
        delta = self.proj_out(h)  # (B, local_channels, n_sites)
        return delta.transpose(1, 2).reshape(B, self.n_sites * self.local_channels)


def _circular_band_mask(dim: int, window: int | None) -> torch.Tensor | None:
    """`(dim, dim)` 0/1 mask, `1` where circular index distance `<= window`,
    else `0`. `None` (fully dense, no restriction) if `window is None`."""
    if window is None:
        return None
    idx = torch.arange(dim)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    dist = torch.minimum(diff, dim - diff)
    return (dist <= window).float()


class MaskedLinear(nn.Module):
    """`Linear(dim, dim)` with an optional fixed circular-band mask on its
    weight matrix (added 2026-08-30, user-directed -- see
    `_MaskedMLPDeltaBody`'s docstring). `window=None`: fully dense (no
    mask, ordinary `Linear`). `window` set: entries beyond circular
    distance `window` are zeroed ONCE at construction and then excluded
    from every forward pass (`weight * mask`) -- since a zeroed entry
    always has exactly zero gradient (`d loss/d weight[i,j] = d loss/d
    (weight*mask)[i,j] * mask[i,j] = 0` when `mask[i,j]=0`), those entries
    stay EXACTLY zero for the entire time this module is trained with a
    finite `window`, not just at initialization.

    This is the whole point: `window=None` (dense) and any finite `window`
    (local) share the exact same parameter shapes, so a "local" module's
    trained weights can be copied directly into a fresh "dense" module
    (`load_state_dict`, since `mask` is a non-persistent buffer excluded
    from `state_dict()`) with the previously off-band, exactly-zero
    entries then perturbed to "unlock" them for further training -- see
    `masked_mlp_warm_start`.

    `nonexpansive` (added 2026-09-04, see `ResidualMLPBlock`'s docstring):
    spectral-normalizes the inner linear (`_sn`/`_SpectralNormLinear`)
    BEFORE the mask is applied -- `forward` calls `effective_weight()`
    (the spectral-normalized value) instead of reading `.weight` directly
    when `nonexpansive` is set. Masking a spectral-norm-1 matrix is not a
    *mathematically guaranteed* <=1 operator norm in the fully general
    case (unlike, say, the Frobenius norm, zeroing entries can in
    adversarial cases raise the operator/spectral norm), but is the same
    practical, empirically-effective heuristic spectral normalization
    already is even in the unmasked case (power iteration only
    approximates the true top singular value) -- verified empirically via
    `analyze_75_76_77_latent_geometry.py`'s local-sensitivity diagnostic,
    not claimed as a formal certificate.

    Also: the "exactly zero gradient outside the band" guarantee above
    (which holds for the DEFAULT, non-`nonexpansive` case) does NOT
    extend to `nonexpansive=True`'s RAW underlying weight -- confirmed
    directly: spectral norm's estimated top singular value is a function
    of the WHOLE raw matrix, so gradient flows into every raw entry via
    that normalization constant, including off-band ones, even though
    they still never affect the EFFECTIVE (mask-multiplied) forward
    output. The functional guarantee (masked positions never influence
    the model's output) holds regardless; only the "raw parameter
    literally frozen at zero off-band" cosmetic property (what
    `masked_mlp_warm_start` relies on) does not carry over.
    """

    def __init__(self, dim: int, window: int | None, nonexpansive: bool = False):
        super().__init__()
        self.nonexpansive = nonexpansive
        self.linear = _sn(dim, dim) if nonexpansive else nn.Linear(dim, dim)
        mask = _circular_band_mask(dim, window)
        if mask is not None and not nonexpansive:
            with torch.no_grad():
                self.linear.weight.mul_(mask)
        self.register_buffer("mask", mask if mask is not None else torch.ones(dim, dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.effective_weight() if self.nonexpansive else self.linear.weight
        return F.linear(x, weight * self.mask, self.linear.bias)


class MaskedMLPResidualBlock(nn.Module):
    """Residual block, structurally like `ResidualMLPBlock` but with both
    `Linear`s replaced by `MaskedLinear` at a shared `window` -- see
    `_MaskedMLPDeltaBody`'s docstring. Deliberately NO `LayerNorm`
    (unlike `ResidualMLPBlock`): `nn.LayerNorm(dim)` normalizes across
    the ENTIRE `dim` axis (computing one mean/variance over all
    positions), which here is the physical latent-index axis -- it would
    silently reintroduce a fully global coupling between every position
    regardless of any masking, defeating the entire point of this
    backbone (caught via the receptive-field-boundedness test).

    Public (renamed from `_MaskedMLPResidualBlock`, 2026-09-03), same
    cross-module reuse reason as `MLPDeltaBody`:
    `ks_latent.models.autoencoder_fourier_mlp`'s masked encoder/decoder
    path reuses this exact class (dimension-preserving, `d_latent ->
    d_latent`) for the square part of its rectangular `NX <-> d_latent`
    masked stack -- see `MaskedLinearRect`'s docstring for the rectangular
    half.

    `nonexpansive` (added 2026-09-04, see `ResidualMLPBlock`'s docstring):
    both `MaskedLinear`s spectral-normalized, and the residual changes
    from `x + h` to `0.5*(x + h)` -- same reasoning as `ResidualMLPBlock`
    (a plain residual is not non-expansive even when its branch is)."""

    def __init__(self, dim: int, window: int | None, dropout: float, nonexpansive: bool = False):
        super().__init__()
        self.nonexpansive = nonexpansive
        self.fc1 = MaskedLinear(dim, window, nonexpansive=nonexpansive)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = MaskedLinear(dim, window, nonexpansive=nonexpansive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        if self.nonexpansive:
            return 0.5 * (x + h)
        return x + h


class _MaskedMLPDeltaBody(nn.Module):
    """`backbone="masked_mlp"` (added 2026-08-30, user-directed): a residual
    MLP, structurally like `MLPDeltaBody`, but DIMENSION-PRESERVING
    throughout (every layer is `d_latent -> d_latent`, ignoring `cfg.hidden`
    -- every layer must keep the same "position i = latent index i" meaning
    for a circular-band mask on it to mean anything) and every `Linear`
    replaced by `MaskedLinear` at a shared `window` (reusing `attn_window`;
    `None` = fully dense = "full mlp").

    Motivation (user, 2026-08-30, following `local_mlp`'s (Section 10)
    confirmed collapse -- a DIFFERENT architecture, tokenized with a
    `Conv1d` mixer, chosen there specifically to test receptive field
    width in isolation): "use the local mlp as the auxiliary propagator to
    create structure in the latent variable... then a full mlp for the
    full propagator. We can initialize the full mlp with a slightly
    perturbed version of the local mlp (since this should just have zeros
    off diagonal)." This backbone is built specifically to make that warm
    start literal: local (masked) and full (dense) share identical
    parameter shapes, so `masked_mlp_warm_start` can copy a trained local
    network's weights directly into a fresh full network and perturb only
    the entries the mask had excluded (which, per `MaskedLinear`'s
    docstring, are exactly zero after local training, not arbitrary
    leftover noise).
    """

    def __init__(
        self, d_latent: int, n_blocks: int, window: int | None, dropout: float, zero_init: bool,
        nonexpansive: bool = False,
    ):
        super().__init__()
        self.input_proj = MaskedLinear(d_latent, window, nonexpansive=nonexpansive)
        self.blocks = nn.ModuleList(
            [MaskedMLPResidualBlock(d_latent, window, dropout, nonexpansive=nonexpansive) for _ in range(n_blocks)]
        )
        # No final LayerNorm -- see MaskedMLPResidualBlock's docstring for
        # why any nn.LayerNorm(d_latent) here would silently reintroduce a
        # fully global coupling between every latent position.
        # output_proj deliberately NEVER spectral-normalized (nonexpansive
        # or not) -- see MLPDeltaBody's docstring for why zero_init and
        # spectral_norm are fundamentally incompatible (0/0 on an all-zero
        # weight's estimated singular value).
        self.output_proj = MaskedLinear(d_latent, window, nonexpansive=False)
        if zero_init:
            nn.init.zeros_(self.output_proj.linear.weight)
            nn.init.zeros_(self.output_proj.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


def masked_mlp_warm_start(
    local_prop: "LatentPropagator", full_cfg: PropagatorConfig, perturb_std: float = 0.01,
    seed: int | None = None,
) -> "LatentPropagator":
    """Build a fresh `backbone="masked_mlp"` propagator from `full_cfg`
    (must have `attn_window=None`, i.e. fully dense), copy every weight
    from `local_prop` (same backbone, a finite `attn_window`) directly
    across -- identical shapes throughout by construction, see
    `MaskedLinear`'s docstring -- then perturb the entries that were
    exactly zero in the local model (the off-band region its mask
    excluded, which never received gradient -- see `MaskedLinear`'s
    docstring for why those are exactly zero, not arbitrary) with small
    Gaussian noise (`perturb_std`) to "unlock" them for further training.
    User-directed 2026-08-30 -- see `_MaskedMLPDeltaBody`'s docstring."""
    if local_prop.cfg.backbone != "masked_mlp" or full_cfg.backbone != "masked_mlp":
        raise ValueError("masked_mlp_warm_start requires backbone='masked_mlp' on both configs")
    if local_prop.cfg.attn_window is None:
        raise ValueError("local_prop must have a finite attn_window (it should be the LOCAL/masked model)")
    if full_cfg.attn_window is not None:
        raise ValueError("full_cfg must have attn_window=None (it should be the FULL/dense model)")
    full_prop = LatentPropagator(full_cfg)
    full_prop.load_state_dict(local_prop.state_dict())
    if seed is not None:
        torch.manual_seed(seed)
    local_modules = dict(local_prop.body.named_modules())
    with torch.no_grad():
        for name, module in full_prop.body.named_modules():
            if isinstance(module, MaskedLinear):
                zero_mask = local_modules[name].mask == 0  # entries the LOCAL model's mask excluded
                noise = torch.randn_like(module.linear.weight) * perturb_std
                module.linear.weight[zero_mask] += noise[zero_mask]
    return full_prop


class _MaskedMLPWideDeltaBody(nn.Module):
    """`backbone="masked_mlp_wide"` (added 2026-09-06, Section 100) -- see
    `PropagatorConfig`'s docstring for the full motivation and sizing
    discussion. A SINGLE hidden layer: `MaskedLinearRect(d_latent, hidden,
    window, d_latent) -> GELU -> MaskedLinearRect(hidden, d_latent, window,
    d_latent)`. No `LayerNorm` (same reasoning as `MaskedMLPResidualBlock`:
    it would normalize across the latent-index axis a circular mask is
    trying to respect, reintroducing full global coupling). `window` is
    referenced against `d_latent`'s own ring in both directions -- an
    output "hidden slot" `h` is treated as sitting at ring position
    `h/hidden`, so a wide `hidden` still respects the same physical
    locality radius as `d_latent`-width layers do (see
    `_circular_band_mask_rect`'s docstring)."""

    def __init__(self, d_latent: int, hidden: int, window: int, zero_init: bool):
        super().__init__()
        self.input_proj = MaskedLinearRect(d_latent, hidden, window, d_latent)
        self.act = nn.GELU()
        self.output_proj = MaskedLinearRect(hidden, d_latent, window, d_latent)
        if zero_init:
            nn.init.zeros_(self.output_proj.linear.weight)
            nn.init.zeros_(self.output_proj.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        return self.output_proj(h)


class _MaskedMLPExpandDeltaBody(nn.Module):
    """`backbone="masked_mlp_expand"` (added 2026-09-10, Section 128,
    user-directed): "I want to try the encoder and decoder pair from 98,
    along with the regularizers from 98. but I want to use a masked mlp
    with three layers, attention_window=3 for the propagator. the thing I
    would like to do though is in the middle layer of the mlp, expand the
    dimension 3x, but in this expanded dimensions, still respect the
    attention window (I guess it would be 9 in that case, so only a very
    limited number of neighboring values interact.) make sure that the
    whole mlp I've described only lets a small number of neighbors
    interact."

    THREE `MaskedLinearRect` layers, all sharing `window` referenced
    against `d_latent`'s own ring via `ref_dim=d_latent` -- exactly
    `_MaskedMLPWideDeltaBody`'s (Section 100) own convention, just with an
    extra layer that stays at the WIDENED dimension instead of contracting
    immediately after one hidden layer:

        input_proj: MaskedLinearRect(d_latent -> hidden, window, d_latent)  # expand
        mid_proj:   MaskedLinearRect(hidden -> hidden,   window, d_latent)  # "the middle layer"
        output_proj:MaskedLinearRect(hidden -> d_latent, window, d_latent)  # contract

    `hidden = expand_factor * d_latent` -- "the middle layer... expand the
    dimension 3x" is literally `hidden`'s width at `expand_factor=3`.

    The user's own "I guess it would be 9" arithmetic is exactly right,
    and follows directly from `_circular_band_mask_rect`'s existing
    convention (window is always expressed in `ref_dim=d_latent` ring
    units, then converted to whatever discrete radius applies at the
    layer's OWN width -- the same mechanism `_MaskedMLPWideDeltaBody`
    already relies on): for `mid_proj` (a `hidden -> hidden` transition,
    `ref_dim=d_latent` still), the fractional threshold is `window /
    d_latent` on a ring of `hidden = expand_factor * d_latent` points, so
    the DISCRETE one-sided radius in hidden-index units works out to
    `window * expand_factor` exactly (derivation: `|i-j| <= hidden *
    window / d_latent = expand_factor * d_latent * window / d_latent =
    expand_factor * window`). At `window=3, expand_factor=3` that is `3 *
    3 = 9` -- so `mid_proj` connects each widened output slot to only
    `2*9+1=19` of the `hidden` widened input slots, a small, bounded
    neighborhood exactly as requested, not the full `hidden`-wide dense
    matrix. `input_proj`/`output_proj` (the `d_latent <-> hidden`
    transitions) use the same `window`/`ref_dim` pair and are
    correspondingly local too (each `d_latent`-width position connects to
    roughly `2*(expand_factor*window)+1` of the `hidden` positions
    clustered around its own image on the shared ring) -- see
    `_circular_band_mask_rect`'s own docstring for the general rectangular
    case. No `nn.LayerNorm` anywhere (same reasoning as every other masked
    body in this module: it would normalize across the entire
    latent-index/hidden axis a circular mask is trying to respect,
    silently reintroducing full global coupling regardless of masking).
    `zero_init=True` (default) zeros `output_proj`'s weight/bias, matching
    every other backbone's identity-at-init property."""

    def __init__(self, d_latent: int, window: int, expand_factor: int, zero_init: bool):
        super().__init__()
        hidden = expand_factor * d_latent
        self.input_proj = MaskedLinearRect(d_latent, hidden, window, d_latent)
        self.act = nn.GELU()
        self.mid_proj = MaskedLinearRect(hidden, hidden, window, d_latent)
        self.output_proj = MaskedLinearRect(hidden, d_latent, window, d_latent)
        if zero_init:
            nn.init.zeros_(self.output_proj.linear.weight)
            nn.init.zeros_(self.output_proj.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        h = self.act(self.mid_proj(h))
        return self.output_proj(h)


def _build_local_attention_mask(n_tokens: int, window: int | None) -> torch.Tensor | None:
    """Kept as a private alias for backward compatibility (existing imports
    and tests reference this name) -- delegates to the shared, public
    `ks_latent.models.autoencoder_vit.build_local_attention_mask` (linear,
    not circular, index distance -- the latent channel ordering has no
    periodic topology to wrap around, unlike the `vit` backbone's
    `pos_encoding="circular"` case)."""
    return build_local_attention_mask(n_tokens, window)


class _LocalMixerBlock(nn.Module):
    """One block of `backbone="local_mlp"`: a LOCAL, UNCONSTRAINED linear
    token-mixer (circular `nn.Conv1d`, kernel width `2*window+1`, no
    softmax/normalization anywhere) in place of self-attention's
    token-mixing, followed by the same per-token feedforward MLP
    (`Linear -> GELU -> Linear`) `ViTBlock` uses -- pre-norm residual
    connections around each, mirroring `ViTBlock`'s structure exactly so
    the only thing that changes is the token-mixing mechanism itself. See
    `_LocalMLPDeltaBody`'s docstring for the motivation."""

    def __init__(self, d_model: int, window: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mix = nn.Conv1d(
            d_model, d_model, kernel_size=2 * window + 1, padding=window, padding_mode="circular"
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, n_tokens, d_model)`."""
        h = self.norm1(x).transpose(1, 2)  # (B, d_model, n_tokens) for Conv1d
        h = self.mix(h).transpose(1, 2)
        x = x + self.dropout(h)
        return x + self.dropout(self.mlp(self.norm2(x)))


class _LocalMLPDeltaBody(nn.Module):
    """`backbone="local_mlp"` (added 2026-08-30, user-directed): tokenizes
    `z` the same way `_ViTDeltaBody` does, then runs `n_layers`
    `_LocalMixerBlock`s -- a LOCAL (bounded receptive field, `attn_window`
    radius) but UNCONSTRAINED (no softmax) linear token-mixer -- then
    projects back to `chunk_size` per token. No pooling/bottleneck, same
    design as `_ViTDeltaBody`.

    Motivation: by this point in the investigation (see
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Sections 5/9), every
    self-attention-based propagator tried -- LOCAL window (`vit`,
    `attn_window=4`) or GLOBAL (`vit`, full attention) -- collapsed to a
    fixed point (`D_KY=0`); every architecture WITHOUT self-attention
    (`mlp`, and `fno_vit`, whose FNO layer precedes its own internal
    attention) recovered rich chaos (`D_KY~21`), with plain `mlp` winning
    outright on every metric. This leaves ambiguous whether receptive
    field width or the softmax normalization itself is the deciding
    factor -- both "no attention" architectures tried so far also happen
    to have a GLOBAL receptive field. This backbone is the missing cell:
    local receptive field, but no softmax anywhere -- a plain circular
    convolution's weights are exactly as free to amplify as `MLPDeltaBody`
    or `FNOLayer`'s, just restricted to nearby tokens. If this ALSO
    recovers chaos, softmax specifically is implicated; if it collapses
    like the attention-based propagators, receptive field width (not
    softmax) is the real variable.

    `attn_window` (reused, not a new field -- see `PropagatorConfig`'s
    docstring) sets the conv kernel radius and is REQUIRED (no "global
    local_mlp": that degenerates to a dense/circulant mixer, already
    covered by `mlp`/`fno_vit`). No positional encoding: a circular
    convolution is already translation-equivariant by construction
    (matching `fno_vit`'s equivariance, unlike `vit`'s explicit
    `CircularPositionalEncoding`), which is a clean, deliberate property
    for this backbone to have on KS's genuinely periodic domain, not an
    oversight.
    """

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        n_layers: int,
        mlp_ratio: int,
        window: int,
        dropout: float,
        zero_init: bool,
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        self.n_tokens = n_tokens
        self.chunk_size = d_latent // n_tokens
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        self.blocks = nn.ModuleList(
            [_LocalMixerBlock(d_model, window, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tokens = circular_overlap_tokenize(z, self.chunk_size, self.token_window, self.n_tokens)
        h = self.token_embed(tokens)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        delta = self.token_unembed(h)
        return delta.reshape(B, self.n_tokens * self.chunk_size)


class _LocalVectorField(nn.Module):
    """The vector field `f_theta` for `backbone="node"` -- see
    `_NeuralODEDeltaBody`'s docstring for the motivation. A circular 1D
    convolutional residual network operating DIRECTLY on the raw
    latent-index axis (one scalar per index, unlike `_LocalMLPDeltaBody`'s
    token-chunk tokenization) -- `window` is a raw latent-index radius,
    the same convention `_MaskedMLPDeltaBody`'s masking uses (`dim` passed
    straight through as `d_latent`), not `local_mlp`'s token-space radius.
    This matters: `spatial_coherence_loss`/D7 measure and reward
    correlation structure at the raw per-INDEX level, so a vector field
    whose own locality is defined in that same raw index space is the
    direct match for the structure actually being induced, not an
    approximation of it via coarser tokens.

    No `LayerNorm`/normalization anywhere, same reasoning as
    `MaskedMLPResidualBlock`: a norm across the spatial (index) axis
    would reintroduce global coupling and defeat the point of this
    backbone. Weight-shared across every index by construction (an
    `nn.Conv1d` kernel is applied identically at every position) --
    genuinely translation-equivariant, unlike `masked_mlp`."""

    def __init__(self, d_latent: int, window: int, hidden: int, n_blocks: int, zero_init: bool):
        super().__init__()
        k = 2 * window + 1
        self.in_conv = nn.Conv1d(1, hidden, k, padding=window, padding_mode="circular")
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv1": nn.Conv1d(hidden, hidden, k, padding=window, padding_mode="circular"),
                        "conv2": nn.Conv1d(hidden, hidden, k, padding=window, padding_mode="circular"),
                    }
                )
                for _ in range(n_blocks)
            ]
        )
        self.out_conv = nn.Conv1d(hidden, 1, 1)
        if zero_init:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `(B, d_latent)`, `f_theta(z)`."""
        h = self.act(self.in_conv(z.unsqueeze(1)))  # (B, hidden, d_latent)
        for block in self.blocks:
            residual = h
            h2 = self.act(block["conv1"](h))
            h2 = block["conv2"](h2)
            h = residual + h2
        return self.out_conv(h).squeeze(1)  # (B, d_latent)


class _NeuralODEDeltaBody(nn.Module):
    """`backbone="node"` (added 2026-08-31, user-directed -- Phase 2
    architecture doc Section 42/43, the most literal reading of "model
    the latent variable with a PDE"): parameterizes `dz/dt = f_theta(z)`
    with `_LocalVectorField` (translation-equivariant, weight-shared,
    local, no softmax) and integrates it via fixed-step RK4 over
    `ode_substeps` sub-steps spanning ONE unit of the dataset's own
    snapshot-index interval -- every dataset this project uses is built
    at `dt_snap=1.0`, and every OTHER backbone already implicitly learns
    `z_n -> z_n+1` in that same unit, so no literal physical `dt` needs
    threading through here.

    Distinct from `local_mlp`/`masked_mlp` in kind, not just degree: those
    learn ONE discrete jump `z_n -> z_n+1` directly; this learns a
    continuous-time vector field and gets the jump by integrating it
    `ode_substeps` times, which is the actual mathematical object a PDE's
    semi-discretization (e.g. KS's own spectral/finite-difference
    time-stepping) is -- a `dt`-independent right-hand side integrated by
    a fixed numerical scheme, rather than a single learned map baked in at
    one specific step size.

    Zero-init (`zero_init=True`, default): `_LocalVectorField`'s final
    projection is zero at init, so `f_theta(z)=0` everywhere, every RK4
    stage (`k1..k4`) is exactly `0`, and this body's returned delta
    (`z_T - z_0`) is exactly `0` -- identity-at-init, matching every other
    backbone."""

    def __init__(
        self, d_latent: int, window: int, hidden: int, n_blocks: int, ode_substeps: int, zero_init: bool
    ):
        super().__init__()
        self.field = _LocalVectorField(d_latent, window, hidden, n_blocks, zero_init)
        self.ode_substeps = ode_substeps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = 1.0 / self.ode_substeps
        z_t = z
        for _ in range(self.ode_substeps):
            k1 = self.field(z_t)
            k2 = self.field(z_t + 0.5 * h * k1)
            k3 = self.field(z_t + 0.5 * h * k2)
            k4 = self.field(z_t + h * k3)
            z_t = z_t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z_t - z


def _polynomial_term_indices(
    n_vars: int, degree: int, max_term_order: int | None = None, free_indices: frozenset[int] | None = None
) -> list[tuple[int, ...]]:
    """The exact monomial index tuples making up the library, in FIXED
    order -- `()` for the constant term, `(n,)` for the linear term `d_n`,
    `(i, j)` (`i<=j`) for the quadratic term `d_i*d_j`. SINGLE SOURCE OF
    TRUTH for both `_polynomial_library_size` and `_polynomial_library`
    (so the two can never disagree on term count/order).

    `max_term_order` (added 2026-09-09, user-directed: "really only let
    the combined degree of the terms be less than 5 (so w_xxx * w_xxx or
    w_xxx*w_xxxx would have 0 coefficients since they have combined degree
    6, 7 respectively)"): excludes any monomial whose derivative orders
    SUM to `>= max_term_order` -- e.g. `w_xxx*w_xxx` is `(3, 3)`, combined
    order `6`; `w_xxx*w_xxxx` is `(3, 4)`, combined order `7`; both
    excluded at `max_term_order=5` (keeps combined order `0..4`). A single
    linear term `(n,)` has combined order `n` itself. The constant term
    `()` always has combined order `0` and is never excluded. `None`
    (default) is unrestricted -- every monomial up to `degree` is kept,
    the original (pre-2026-09-09) behavior. Physically motivated: KS's own
    true equation `-w*w_x - w_xx - w_xxxx` has combined order `1`, `2`,
    and `4` respectively on its three terms -- all comfortably kept by any
    `max_term_order >= 5` -- while a high-combined-order cross term like
    `w_xxx*w_xxxx` has no obvious physical justification and is exactly
    the kind of term the raw (degree-2, no truncation) library made
    available for the optimizer to (mis)use.

    `degree` up to 3 (added 2026-09-09, user-directed: "higher degree
    polynomial for the pde" -- the original design only supported degree
    1/2): for `degree=d`, generates every non-decreasing index tuple of
    length `d` (`itertools.combinations_with_replacement`, e.g. `(i, j, k)`
    with `i<=j<=k` for a cubic term `d_i*d_j*d_k`) for every length `1..d`
    -- this is the SAME construction the original degree<=2 code used by
    hand (linear terms are length-1 tuples, quadratic length-2), just
    generalized so it doesn't need a new hand-written loop per degree.

    `free_indices` (added 2026-09-18, Section 189, alongside `poly_time_deriv`;
    generalized from a single `free_index` to a set, Section 190, alongside
    `poly_time_deriv2`): variable indices (when given) whose contribution
    to the `max_term_order` SUM is treated as `0` instead of their literal
    value -- for every OTHER variable, index and spatial derivative order
    coincide (`d_n` IS the order-`n` derivative, so summing raw indices is
    exactly summing derivative orders), but the appended time-derivative
    variables (`w_t`, `w_tt`) are not spatial derivatives at all and have
    no natural "order" in that sum -- treating them as free (contribute 0)
    lets `w_t`, `w_tt`, `w*w_t`, `w_t*w_xxxx`, `w_t*w_tt`, etc. appear at
    whatever `max_term_order` the SPATIAL terms alone would already allow,
    rather than being excluded outright merely because their raw indices
    happen to be numerically large (they are placed at the END, right
    after every real spatial-derivative order). Still subject to `degree`
    (the factor-COUNT cap) exactly like every other variable -- only the
    `max_term_order` order-SUM check is exempted."""
    indices: list[tuple[int, ...]] = [()]
    for term_degree in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), term_degree):
            order = sum(0 if (free_indices and v in free_indices) else v for v in combo)
            if max_term_order is None or order < max_term_order:
                indices.append(combo)
    return indices


def _polynomial_library_size(
    n_vars: int, degree: int, max_term_order: int | None = None, free_indices: frozenset[int] | None = None
) -> int:
    """Number of monomials `_polynomial_library` actually produces for
    this `n_vars`/`degree`/`max_term_order` combination, INCLUDING the
    constant term -- see `_polynomial_term_indices`'s own docstring for
    the exact filtering (including `free_indices`)."""
    if degree not in (0, 1, 2, 3):
        raise ValueError(f"degree must be 0, 1, 2, or 3, got {degree!r}")
    return len(_polynomial_term_indices(n_vars, degree, max_term_order, free_indices))


def _polynomial_library(
    derivs: torch.Tensor, degree: int, max_term_order: int | None = None, free_indices: frozenset[int] | None = None
) -> torch.Tensor:
    """`derivs`: `(..., n_vars)` (the `max_order+1` synthesized spatial
    derivatives `w, w_x, w_xx, ...` at one point) -> `(..., n_terms)`, the
    degree-`<=degree` (optionally combined-order-truncated, see
    `max_term_order`) monomial library ready for a single, SHARED linear
    layer (`nn.Linear(n_terms, 1, bias=True)`) whose weights/bias literally
    ARE the PDE's own learned coefficients -- see
    `_SpectralPDEDeltaBody.field_kind="polynomial"`'s docstring for the
    full motivation (added 2026-09-09, user-directed: "expand the pde as a
    polynomial (degree 1 or 2) in all the derivative terms, then directly
    learn the coefficients").

    Term order (fixed via `_polynomial_term_indices`, so a saved model's
    coefficients are always interpretable the same way): `[1, d_0, d_1,
    ..., d_m]` for `degree=1`; `[1, d_0, ..., d_m, d_0*d_0, d_0*d_1, ...,
    d_0*d_m, d_1*d_1, ..., d_m*d_m]` for `degree=2` (upper-triangular
    `i<=j` pairs, i.e. squares included, each cross term appearing ONCE,
    not twice -- matches how a human would write out a polynomial's own
    distinct terms, e.g. the true KS nonlinearity `-w*w_x` is exactly the
    single term `d_0*d_1`, not `0.5*(d_0*d_1 + d_1*d_0)`); `degree=3` adds
    every non-decreasing cubic index triple `d_i*d_j*d_k` (`i<=j<=k`) the
    same way. Any term excluded by `max_term_order` is simply absent
    (equivalent to, but more efficient than, keeping it with a
    permanently-zero coefficient)."""
    if degree not in (0, 1, 2, 3):
        raise ValueError(f"degree must be 0, 1, 2, or 3, got {degree!r}")
    n_vars = derivs.shape[-1]
    ones = torch.ones_like(derivs[..., :1])
    cols = []
    for idx in _polynomial_term_indices(n_vars, degree, max_term_order, free_indices):
        if len(idx) == 0:
            cols.append(ones)
        else:
            term = derivs[..., idx[0]]
            for extra in idx[1:]:
                term = term * derivs[..., extra]
            cols.append(term.unsqueeze(-1))
    return torch.cat(cols, dim=-1)


def _chebyshev_eval(x: torch.Tensor, k: int) -> torch.Tensor:
    """`T_k(x)`, the degree-`k` Chebyshev polynomial of the first kind,
    elementwise, via the standard three-term recurrence `T_0=1, T_1=x,
    T_{n+1}=2x*T_n - T_{n-1}`. Well-defined for ANY real `x` (the
    recurrence itself has no domain restriction) -- the classical
    orthogonality/conditioning properties Chebyshev polynomials are prized
    for only hold on `x in [-1,1]`, so outside that range this is simply
    "the polynomial you get by continuing the recurrence," a perfectly
    valid alternate basis for the SAME space of ordinary polynomials up to
    degree `k`, just without the numerical-conditioning guarantee. See
    `field_kind="chebyshev"` on `_SpectralPDEDeltaBody` for why this is
    used anyway (added 2026-09-11, user-directed)."""
    if k == 0:
        return torch.ones_like(x)
    t_prev, t_curr = torch.ones_like(x), x
    for _ in range(k - 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def _chebyshev_library(
    derivs: torch.Tensor, degree: int, max_term_order: int | None = None, free_indices: frozenset[int] | None = None
) -> torch.Tensor:
    """Chebyshev-basis analogue of `_polynomial_library`: EXACT SAME term
    enumeration (`_polynomial_term_indices`, so term COUNT/order/filtering
    is identical and `_chebyshev_to_monomial_matrix` below can convert
    between the two term-for-term), but each term is built from Chebyshev
    polynomials of the repeated derivative channels instead of plain
    powers -- e.g. the monomial term `(0, 0, 2)` (`w^2 * w_xx`, `w` picked
    twice, `w_xx` once) becomes `T_2(w) * T_1(w_xx)` here (`T_1(x)=x`, so
    any variable appearing with multiplicity 1 is UNCHANGED -- only
    repeated variables, i.e. genuine powers, differ from the monomial
    library). Added 2026-09-11, user-directed: 'I want to consider whether
    or not another basis, e.g. chebyshev polynomials could be used to
    represent the pde' -- motivation and the interpretability tradeoff are
    in `_SpectralPDEDeltaBody`'s own `field_kind="chebyshev"` docstring."""
    if degree not in (0, 1, 2, 3):
        raise ValueError(f"degree must be 0, 1, 2, or 3, got {degree!r}")
    n_vars = derivs.shape[-1]
    ones = torch.ones_like(derivs[..., :1])
    cols = []
    for idx in _polynomial_term_indices(n_vars, degree, max_term_order, free_indices):
        if len(idx) == 0:
            cols.append(ones)
            continue
        mult: dict[int, int] = {}
        for v in idx:
            mult[v] = mult.get(v, 0) + 1
        term = None
        for v in sorted(mult):
            factor = _chebyshev_eval(derivs[..., v], mult[v])
            term = factor if term is None else term * factor
        cols.append(term.unsqueeze(-1))
    return torch.cat(cols, dim=-1)


def _chebyshev_to_monomial_matrix(
    n_vars: int, degree: int, max_term_order: int | None = None, free_indices: frozenset[int] | None = None
) -> torch.Tensor:
    """Returns a `(n_terms, n_terms)` matrix `B` such that, for a
    `field_kind="chebyshev"` `poly_coeffs.weight` (shape `(1, n_terms)`,
    ordered by the SAME `_polynomial_term_indices` enumeration
    `_chebyshev_library`/`_polynomial_library` both use),
    `monomial_weight = chebyshev_weight @ B.T` (or equivalently `(B @
    chebyshev_weight.squeeze(0)).unsqueeze(0)`) gives the coefficient
    vector of the ORDINARY monomial polynomial (same term order) that
    computes the EXACT SAME function -- i.e. a genuine, exact change of
    basis, not an approximation. Added 2026-09-11, user-directed: 'train
    in chebyshev space, then transform back' -- the whole point being
    that training happens in the (potentially better-conditioned)
    Chebyshev basis, while every downstream interpretation/reporting step
    (comparing against true KS's own `-w*w_x-w_xx-w_xxxx`, as this project
    has done for every prior `field_kind="polynomial"` extraction) still
    reads off ordinary, directly-comparable monomial coefficients.

    Derivation: each single-variable factor `T_k(x)` expands into ordinary
    powers `x^0..x^k` via `numpy.polynomial.chebyshev.cheb2poly` (the
    standard, exact Chebyshev-to-power-basis conversion). A multi-variable
    Chebyshev term (one distinct `T_k` per repeated-variable-group) expands
    into the OUTER PRODUCT of each variable's own single-variable
    expansion -- e.g. `T_2(w)*T_1(w_xx) = (2w^2-1)*w_xx = 2*(w^2*w_xx) -
    1*(w_xx)`, contributing to TWO different monomial terms (importantly,
    Chebyshev expansions only ever produce monomials of the SAME OR LOWER
    combined derivative-order than the original term -- `T_k`'s own
    expansion only has powers `k, k-2, k-4, ...`, never higher -- so every
    resulting monomial is guaranteed to already be present in the SAME
    `_polynomial_term_indices(n_vars, degree, max_term_order)` enumeration;
    this matrix is never singular for representing that guarantee, no
    extra terms are ever needed)."""
    import numpy as np
    from numpy.polynomial.chebyshev import cheb2poly

    term_indices = _polynomial_term_indices(n_vars, degree, max_term_order, free_indices)
    n_terms = len(term_indices)
    term_to_row = {t: i for i, t in enumerate(term_indices)}

    single_var_expand = []  # single_var_expand[k]: monomial coeffs (length k+1) of T_k
    for k in range(degree + 1):
        cheb_coeffs = np.zeros(k + 1)
        cheb_coeffs[k] = 1.0
        single_var_expand.append(cheb2poly(cheb_coeffs))

    B = np.zeros((n_terms, n_terms))
    for col, combo in enumerate(term_indices):
        if len(combo) == 0:
            B[term_to_row[()], col] = 1.0
            continue
        mult: dict[int, int] = {}
        for v in combo:
            mult[v] = mult.get(v, 0) + 1
        # accumulate as {monomial_index_tuple: coefficient}, building up the
        # outer product across variables one at a time.
        partial: dict[tuple[int, ...], float] = {(): 1.0}
        for v in sorted(mult):
            expand = single_var_expand[mult[v]]
            new_partial: dict[tuple[int, ...], float] = {}
            for existing_tuple, existing_coeff in partial.items():
                for power, coeff in enumerate(expand):
                    if coeff == 0.0:
                        continue
                    new_tuple = tuple(sorted(existing_tuple + (v,) * power))
                    new_partial[new_tuple] = new_partial.get(new_tuple, 0.0) + existing_coeff * coeff
            partial = new_partial
        for monomial_tuple, coeff in partial.items():
            B[term_to_row[monomial_tuple], col] = coeff
    return torch.tensor(B, dtype=torch.float32)


class _SpectralPDEDeltaBody(nn.Module):
    """`backbone="spectral_pde"` (added 2026-09-06, see
    docs/sine_transform_pde_plan.md for the full design and motivation).

    Given `z` (a truncated rFFT spectrum, `PropagatorConfig.spectral_K`
    complex modes, same `SpectralFieldAutoencoderConfig` convention),
    synthesizes the EXACT physical-space fields `w, w_x, ..., w^(max_order)`
    at `spectral_N_w` grid points (`ks_latent.models.spectral_field.
    synthesize_derivatives` -- a diagonal `(i*k)^n` multiplier per order,
    no discretization error, unlike a finite-difference or learned-
    convolution derivative estimate), stacks them into a `(max_order+1)`-dim
    feature vector at every point, and runs a SHARED (identical weights at
    every point, translation-equivariant, no positional encoding -- same
    design principle as `TokenMLPBlock`/`backbone="fno_mlp"`) small MLP
    (`ResidualMLPBlock`-based) mapping that local derivative stack to
    `w_t` at that point. This pointwise MLP **is** the implicitly-learned
    PDE: `w_t = f_theta(w, w_x, w_xx, ..., w^(max_order))`, matching real
    KS's own governing-equation *form* (a local, translation-invariant
    function of the field and its own spatial derivatives at that point) --
    see the plan doc's §4.1 for why this is a real, uncertain test of
    whether this project's H-PROP finding (local propagators collapse to a
    fixed point regardless of mechanism) reflects locality itself or
    (as this design bets) locality without genuine derivative information.

    `field(z)` is the right-hand-side estimator (`w_t`, in physical space);
    `forward(z)` integrates it forward exactly one unit of `dt_snap`, via
    one of three `integrator` options:

    - `"euler"`: forward Euler, sub-stepped over `ode_substeps` steps of size
      `h=1/ode_substeps` each (`w_t = w_t + h*field(encode(w_t))`, re-encoding
      to `z` between sub-steps exactly like `"rk4"`/`"etdrk4"` do -- added
      2026-09-07, user-directed: "couldn't we also integrate euler over
      multiple steps?" -- previously hardcoded to a single full step
      regardless of `ode_substeps`, inconsistent with the other two
      integrators and silently ignoring whatever `ode_substeps` a config
      claimed). `ode_substeps=1` (the default) is exactly the original
      single-step form (`w_next = w + field(z)`), matching the proposal's
      literal `(w(t+1)-w(t))/dt` finite-difference training target.
    - `"rk4"`: fixed-step RK4 (`ode_substeps` sub-steps, mirroring
      `_NeuralODEDeltaBody`'s existing loop structure -- the only difference
      is that the "state" advanced here is `w` internally, re-encoded to `z`
      via the SAME fixed transform between RK stages, since `field` needs a
      spectral input to synthesize derivatives from). Both `"euler"` and
      `"rk4"` ask `field` (the MLP) to reproduce the ENTIRE right-hand side,
      including KS's own stiff linear diffusion term (`-u_xx - u_xxxx`) --
      no different, numerically, than asking a generic black-box ODE to
      integrate a stiff system with an explicit scheme (which is exactly
      why the real solver does NOT do this -- see below).
    - `"etdrk4"` (added 2026-09-06, user-directed: "how closely does this
      method mirror the actual method we use to integrate the KS system
      ... being as close as possible to that methodology will give us the
      best chance for success"): mirrors `ks_latent/solver/ks.py`'s own
      ETDRK4 solver (Kassam & Trefethen 2005) as directly as possible,
      reusing its `linear_operator`/`etdrk4_coefficients` functions.
      Splits the dynamics the SAME way the true PDE does: `z_t = Lhat*z +
      N(z)`, `Lhat(k) = k^2 - k^4` (the exactly-known Fourier symbol of
      `-u_xx - u_xxxx`, fixed at KS's own true value for the `spectral_K`
      kept modes, never learned -- a real, zero-parameter physical prior,
      not a black box), integrated EXACTLY via the integrating factor
      `exp(dt*Lhat)` (unconditionally stable for the linear part regardless
      of step size -- the entire point of exponential time differencing),
      while `N(z)` -- the genuinely nonlinear residual `field` needs to
      learn -- is evaluated at 4 RK-like stages and combined via the SAME
      `E/E2/Q/f1/f2/f3` coefficients (`ks_latent.solver.ks.
      etdrk4_coefficients`, computed ONCE at construction from the fixed
      `Lhat` and `ode_substeps`' implied step size -- not re-derived per
      forward pass, since `Lhat` is not trainable here) the real solver
      uses in its own `step()` function -- see `_etdrk4_step` below for the
      literal correspondence. `N(z)` itself is computed the same
      pseudo-spectral way the true solver computes its own nonlinear term
      (`ks_latent.solver.ks._nonlinear`): synthesize the exact derivative
      stack in PHYSICAL space, evaluate the pointwise MLP there (`field`),
      then transform the result back to the SAME truncated frequency
      representation (`encode_to_spectrum`) the ETD combination needs.
      Splitting the equation this way does NOT make the overall system
      linear -- `N(z)` is exactly as nonlinear as it ever was (see the
      class-level user Q&A in docs/sine_transform_pde_plan.md) -- it only
      removes the ALREADY-SOLVED linear sub-problem from what the MLP has
      to discover, and removes the stiffness that sub-problem would
      otherwise impose on an explicit scheme.

    `zero_init=True` (default): `output_proj`'s weight/bias are zeroed, so
    `field(z) = 0` everywhere at init. For `"euler"`/`"rk4"` this trivially
    gives `w_next = w`, and `forward` returns `_encode(w) - z`, which is
    EXACTLY zero by the exact rFFT/irFFT round-trip invariant (zero-padding
    and truncating are exact linear inverses on the kept low modes,
    unaffected by whatever is in the discarded high modes, since no
    nonlinear step intervenes) -- matching every other backbone's
    identity-at-init property, verified directly in this module's tests
    rather than assumed. (This exact round-trip holds for any `z` that
    actually came from a real field via `encode_to_spectrum` -- see that
    function's docstring for the one caveat, an always-zero-in-practice
    DC-imaginary slot that a synthetic all-random test `z` could trip up.)

    For `"etdrk4"`, `zero_init=True` does NOT give identity: with `N(z)=0`
    identically, every RK stage's nonlinear evaluation is zero too, so the
    combination collapses to `z_next = exp(Lhat)*z` -- the EXACT linearized-
    KS map (real growth at low wavenumbers below the `k=1/sqrt(2)`
    instability threshold, real decay above it), not a no-op. This is a
    deliberate, arguably better-motivated departure from the project's
    usual "start as pure identity" convention: this backbone starts
    already implementing an exact linear KS integrator, with the MLP's
    nonlinear correction the only thing left to learn -- see this module's
    tests for the exact invariant this reduces to (composes correctly
    across `ode_substeps`, since `exp(a)*exp(a)*...*exp(a)` (`n` times)
    equals `exp(n*a)` for a diagonal/scalar exponent, regardless of how
    finely the unit interval is subdivided)."""

    def __init__(
        self, K: int, N_w: int, L: float, max_order: int, hidden: int, n_blocks: int,
        dropout: float, zero_init: bool, integrator: str = "euler", ode_substeps: int = 1,
        physics_prior: bool = False, field_kind: str = "mlp", poly_degree: int = 2,
        poly_max_term_order: int | None = None, poly_norm_power: float = 1.0,
        poly_stable_leading: bool = False, poly_no_constant: bool = False,
        poly_fixed_linear_terms: dict[int, float] | None = None,
        poly_exclude_nonconservative: bool = False,
        poly_stable_linear_terms: dict[int, float] | None = None,
        poly_time_deriv: bool = False,
        poly_time_deriv_dt_snap: float = 1.0,
        poly_time_deriv2: bool = False,
        burgers_nu_init: float = 1.0,
        burgers_beta_max: float = 1.0,
        burgers_nonlinear_nu: bool = False,
        burgers_kernel_instability: bool = False,
        burgers_kernel_A_max: float = 1.0,
        burgers_kernel_width_init: float = 1.0,
        burgers_forcing_max: float = 1.0,
        burgers_kernel_A_fixed: float | None = None,
        burgers_beta_fixed: float | None = None,
        burgers_kernel_mu_init: float | None = None,
        burgers_no_forcing: bool = False,
        burgers_kernel_A_init: float = 0.0,
        burgers_beta_init: float = 0.0,
    ):
        super().__init__()
        self.K = K
        self.N_w = N_w
        self.L = L
        self.max_order = max_order
        self.integrator = integrator
        self.ode_substeps = ode_substeps
        self.physics_prior = physics_prior
        # `correction_scale` (added 2026-09-09, user-directed: "how do we
        # preserve the chaotic structure and nudge it in the direction we
        # want? could we progressively add systems we know are chaotic?"):
        # a plain, mutable Python float (NOT a learned parameter or
        # buffer -- the training loop sets it directly, once per epoch,
        # like a homotopy/continuation parameter), scaling the LEARNED
        # correction's contribution in `field()`'s physics_prior branch:
        # `prior + correction_scale * correction`. At `0.0` the propagator
        # is EXACTLY the true KS equation (genuinely chaotic, completely
        # untouched by the learned correction); ramping it 0->1 over
        # training lets the encoder converge to an accurate representation
        # under REAL chaotic dynamics before the learned correction gets
        # enough room to damp anything -- directly targets the mechanism
        # diagnosed the same day: short-horizon MSE prediction loss
        # rewards contraction whenever the encoder's z is imprecise
        # (always true early in training), since a damped system's errors
        # shrink regardless of input quality while a genuinely chaotic
        # one's errors grow regardless of model quality. Default `1.0`
        # (unchanged behavior -- the full learned correction is always
        # active unless a training loop explicitly schedules this down).
        self.correction_scale = 1.0
        self.field_kind = field_kind
        self.poly_degree = poly_degree
        self.poly_max_term_order = poly_max_term_order
        self.poly_norm_power = poly_norm_power
        self.poly_stable_leading = poly_stable_leading
        self.poly_no_constant = poly_no_constant
        self.poly_fixed_linear_terms = poly_fixed_linear_terms
        self.poly_exclude_nonconservative = poly_exclude_nonconservative
        self.poly_stable_linear_terms = poly_stable_linear_terms
        self.burgers_nu_init = burgers_nu_init
        self.burgers_beta_max = burgers_beta_max
        self.burgers_nonlinear_nu = burgers_nonlinear_nu
        self.burgers_kernel_instability = burgers_kernel_instability
        self.burgers_kernel_A_max = burgers_kernel_A_max
        self.burgers_forcing_max = burgers_forcing_max
        self.burgers_kernel_A_fixed = burgers_kernel_A_fixed
        self.burgers_beta_fixed = burgers_beta_fixed
        self.burgers_no_forcing = burgers_no_forcing
        if burgers_kernel_instability and burgers_nonlinear_nu:
            raise ValueError(
                "burgers_kernel_instability and burgers_nonlinear_nu are mutually exclusive -- "
                "both replace the diffusion term's own computation with a different mechanism; "
                "combining them is not supported."
            )
        if poly_exclude_nonconservative and poly_degree > 2:
            raise ValueError(
                f"poly_exclude_nonconservative=True requires poly_degree<=2, got "
                f"poly_degree={poly_degree!r} -- the even/odd-combined-order conservation "
                "rule is only derived and verified for two-factor (length<=2) product terms; "
                "degree=3 introduces length-3 terms this flag does not know how to classify."
            )
        if poly_time_deriv and field_kind not in ("polynomial", "chebyshev"):
            raise ValueError(
                f"poly_time_deriv=True requires field_kind in ('polynomial', 'chebyshev'), got "
                f"{field_kind!r} -- the mechanism appends a finite-difference time-derivative "
                "feature to the SINDy-style library those two field_kinds build; an MLP/"
                "forced_burgers field has no such library to extend."
            )
        if poly_time_deriv2 and field_kind not in ("polynomial", "chebyshev"):
            raise ValueError(
                f"poly_time_deriv2=True requires field_kind in ('polynomial', 'chebyshev'), got "
                f"{field_kind!r} -- same reasoning as poly_time_deriv."
            )
        self.poly_time_deriv = poly_time_deriv
        self.poly_time_deriv_dt_snap = poly_time_deriv_dt_snap
        self.poly_time_deriv2 = poly_time_deriv2
        if field_kind in ("polynomial", "chebyshev"):
            # Both share the EXACT SAME term_indices/poly_coeffs/zero_init/
            # poly_stable_leading setup -- only field()'s forward
            # evaluation differs (monomial vs Chebyshev product per term).
            # poly_stable_leading's own mechanism only ever touches a
            # single-index linear term (highest_even,) -- multiplicity 1,
            # where T_1(x)=x is IDENTICAL to the monomial case -- so it is
            # basis-invariant and needs no special-casing here.
            #
            # `poly_time_deriv` (added 2026-09-18, Section 189, user-
            # directed: "can we incorporate time derivatives into the pde
            # polynomial? might give us a richer expression. We can
            # approximate them using rollout terms potentially"): appends
            # ONE extra library variable, a finite-difference estimate
            # `w_t_fd = (w - w_prev) / dt_snap`, at index `max_order+1`
            # (right after the spatial derivative stack `w..w^(max_order)`
            # at indices `0..max_order`) -- so it can appear alone or in
            # any product term (`w_t`, `w*w_t`, `w_t*w_x`, `w_t^2`, ...)
            # subject to the SAME poly_degree/poly_max_term_order budget as
            # every other variable, not a separate mechanism bolted on
            # after the fact. `w_prev` (`= self._decode(z_prev)`) is a
            # genuine PREVIOUS-STEP decoded physical field; `field()`'s new
            # `z_prev` kwarg takes the previous state in `z`'s OWN
            # (spectral) representation and decodes it internally -- NOT a
            # second learned/implicit unknown (which would make this an
            # implicit/delay equation requiring iterative solving); see
            # `field()`'s own docstring for exactly where `z_prev` comes
            # from at training vs. rollout time (the user's
            # own "approximate them using rollout terms" framing: real
            # data for the very first predicted step, the model's OWN
            # prior prediction for every step after that -- no change
            # needed to the training loop's existing window construction,
            # since `LatentPropagator.rollout` already threads consecutive
            # `(z_prev, z_curr)` pairs through its loop; only `step()`
            # needed to stop discarding `z_prev` for markovian mode, see
            # `LatentPropagator.step`'s docstring for the exact dispatch).
            # `step_one(z)` (single-argument, used by EVERY Jacobian-based
            # regularizer built in Sections 183-186 -- delta_cap,
            # kernel_unstable_floor, both spectrum-shape losses) has no
            # `z_prev` to give, so it defaults to `None` there and
            # `w_t_fd` falls back to an architectural zero -- this keeps
            # `step_one` a well-defined, differentiable, PURE
            # single-argument function, so nothing built this arc needs to
            # change to keep working (it simply never sees a nonzero
            # `w_t_fd`, the same "no history available yet" semantics a
            # cold rollout start already has).
            n_vars = max_order + 1 + (1 if poly_time_deriv else 0) + (1 if poly_time_deriv2 else 0)
            # `_time_deriv_var_index`/`_time_deriv2_var_index` (Section 189,
            # extended Section 190 for `poly_time_deriv2`/`w_tt`): the new
            # w_t_fd/w_tt_fd variables' own indices, appended in that order
            # right after the spatial stack (`w_t` at `max_order+1` if
            # active, `w_tt` right after it -- at `max_order+1` if `w_t` is
            # OFF, else `max_order+2`). Both passed as `_polynomial_term_
            # indices`'s `free_indices` so both are exempt from the
            # `max_term_order` combined-SPATIAL-order budget -- see that
            # parameter's own docstring for why a raw
            # index-sum check would otherwise make them nearly unreachable
            # (their indices are numerically as large as a high-order
            # spatial derivative, despite not being spatial derivatives at
            # all). `None` when the corresponding flag is off.
            self._time_deriv_var_index = (max_order + 1) if poly_time_deriv else None
            self._time_deriv2_var_index = (
                (max_order + 1 + (1 if poly_time_deriv else 0)) if poly_time_deriv2 else None
            )
            self._time_deriv_free_indices = frozenset(
                i for i in (self._time_deriv_var_index, self._time_deriv2_var_index) if i is not None
            ) or None
            term_indices = _polynomial_term_indices(
                n_vars, poly_degree, poly_max_term_order, self._time_deriv_free_indices
            )
            n_terms = len(term_indices)
            # `_nonlinear_term_mask` (added 2026-09-12, user-directed:
            # "directly replicate iLED's stabilization mechanism" --
            # arXiv:2309.05812): 1.0 at every genuinely NONLINEAR (product,
            # length>=2) term, 0.0 at the constant (length 0) and every
            # single-derivative linear term (length 1) -- used by
            # `nonlinear_field` below to isolate iLED's `Psi_1` analogue
            # (the nonlinear closure alone) from the already-stability-
            # constrained linear part (`poly_fixed_linear_terms`/
            # `poly_stable_leading`, iLED's own `A_theta` analogue -- see
            # those params' docstrings; already a stronger guarantee than
            # iLED's `A = W - W^T - diag(|w|)` reparametrization, since
            # here the leading term's PHYSICAL coefficient can be fixed
            # exactly rather than merely sign-constrained).
            nonlinear_mask = torch.tensor(
                [len(idx) >= 2 for idx in term_indices], dtype=torch.float32
            )
            self.register_buffer("_nonlinear_term_mask", nonlinear_mask, persistent=False)
            # `poly_exclude_nonconservative` (added 2026-09-12, user-
            # directed: "please try to exclude every even-combined-order
            # two-factor term from the library"): true KS conserves
            # int(u)dx EXACTLY because every term in -u*u_x-u_xx-u_xxxx is
            # a total x-derivative. For a TWO-FACTOR product term d_i*d_j
            # (derivative orders i,j), repeated integration by parts (push
            # derivatives from one factor to the other, each step costing
            # exactly one total-derivative correction term) lets i and j
            # be balanced toward the middle of their sum: if i+j=2m is
            # EVEN, this lands on (w^(m))^2 -- a perfect square, manifestly
            # NOT a total derivative (its integral is generically nonzero
            # and sign-definite); if i+j=2m+1 is ODD, this lands on
            # w^(m)*w^(m+1) = (1/2)*d/dx((w^(m))^2) -- an EXACT total
            # derivative, integral always exactly zero, for ANY state.
            # This is a general, provable, exact rule for length-2 terms
            # (verified directly 2026-09-12 against every term in Section
            # 160-166's own fitted coefficient tables: w*w_x (0+1=1, odd)
            # -- the true KS nonlinearity itself -- and w_xx*w_xxx (2+3=5,
            # odd) both measured contributing exactly zero to the mean;
            # w_x*w_x (1+1=2, even, the LARGEST measured violator at
            # -0.043) and w*w (0+0=0, even) both measured contributing a
            # real, systematic, non-oscillating drift). Single-derivative
            # (length<=1) terms are always safe regardless of parity --
            # they are literal derivatives, integrating to zero for n>=1,
            # or (for the length-0 constant) already handled separately by
            # `poly_no_constant`.
            #
            # Rather than a soft penalty (`w_pde_mean_conservation`,
            # Section 165 -- which needs a heuristically-tuned weight and
            # only matches the constraint on the TRAINING batch, not
            # architecturally for every state), this ARCHITECTURALLY
            # zeros the raw contribution of every length-2 term whose
            # combined order is even, in `_poly_weight()` -- the same
            # "always exactly zero, no gradient, no weight to tune"
            # mechanism `poly_no_constant`/`poly_fixed_linear_terms`
            # already use. Requires `poly_degree<=2` (validated above) --
            # the rule is not derived for length>=3 terms.
            nonconservative_mask = torch.tensor(
                [len(idx) == 2 and sum(idx) % 2 == 0 for idx in term_indices], dtype=torch.float32
            )
            self.register_buffer("_nonconservative_term_mask", nonconservative_mask, persistent=False)
            # `poly_no_constant` (added 2026-09-11, user-directed: "we
            # should just force the constant to be 0 during training"):
            # removes BOTH additive-constant avenues at once -- the
            # Linear layer's own `bias` (bias=False, no parameter at all)
            # and the `()` term's own weight column (architecturally
            # zeroed every forward pass in `field()`, so it never
            # influences the output and therefore never receives
            # nonzero gradient -- same "zero contribution -> stays at
            # its init value forever" mechanism `poly_stable_leading`
            # already relies on for its own reparametrized column).
            # `()` is ALWAYS term_indices[0] by `_polynomial_term_indices`'s
            # own fixed construction order, so no lookup is needed.
            self.poly_coeffs = nn.Linear(n_terms, 1, bias=not poly_no_constant)
            if zero_init:
                nn.init.zeros_(self.poly_coeffs.weight)
                if self.poly_coeffs.bias is not None:
                    nn.init.zeros_(self.poly_coeffs.bias)
            if poly_stable_leading:
                # `poly_stable_leading` (added 2026-09-09, user-directed:
                # "is there a way to regularize or bound the eigenvalues of
                # the differential operator induced by the pde?"):
                # architecturally forces the coefficient on the HIGHEST
                # kept EVEN-order linear derivative term to be <= 0.
                #
                # Why this specific term: the polynomial's linear (degree-1)
                # part is a genuine constant-coefficient differential
                # operator, so each Fourier mode e^{ikx} is an eigenfunction
                # with eigenvalue lambda(k) = sum_n c_n*(ik)^n. Only EVEN n
                # contributes to Re(lambda(k)) (odd-order terms are purely
                # imaginary -- dispersive, not growth/decay). For the system
                # to stay BOUNDED as energy reaches high wavenumber, we need
                # Re(lambda(k)) -> -inf as k -> inf, which reduces to: the
                # coefficient on the highest even order must be negative --
                # exactly the sign KS's own true `-w_xxxx` term has. A
                # positive coefficient there makes Re(lambda(k)) grow like
                # +c*k^max_order, unbounded amplification at high k -- a
                # direct, closed-form explanation for the exact
                # "trains fine short-horizon, diverges to NaN over a longer
                # free rollout" failure mode found empirically the same day
                # (Section 117, poly_norm_power<1.0 relaxing the original
                # normalization). This only bounds the WORST case at high k
                # -- it does NOT suppress genuine instability at low/mid
                # wavenumber (still needed for real chaos), so it's a
                # targeted fix for runaway blowup, not another collapse-
                # inducing constraint.
                #
                # Reparametrization: c_leading = -(raw)^2 (always <= 0,
                # exactly 0 at raw=0) -- NOT softplus, which can only
                # approach 0 in the limit and would break the existing
                # zero_init=True "exact identity at init" invariant.
                # `self.poly_coeffs.weight`'s own entry at the leading
                # term's column IS `raw` -- no separate parameter needed,
                # the transform is applied inline in `field()` every
                # forward pass.
                highest_even = None
                for n in range(max_order, -1, -1):
                    if n % 2 == 0 and (n,) in term_indices:
                        highest_even = n
                        break
                if highest_even is None:
                    raise ValueError(
                        "poly_stable_leading=True requires at least one even-order linear "
                        f"term to survive poly_max_term_order={poly_max_term_order!r} filtering "
                        f"at max_order={max_order!r} -- none found, nothing to constrain."
                    )
                self._stable_leading_idx = term_indices.index((highest_even,))
                self._stable_leading_order = highest_even
                # Re((ik)^n) cycles with period 4: +1 when n%4==0, -1 when
                # n%4==2 (only even n reach here at all). Re(lambda(k))'s
                # leading (highest-order) term is `c_n * Re((ik)^n) * k^n`;
                # for this to go to -inf as k->inf we need `c_n *
                # Re((ik)^n) < 0`, i.e. c_n's REQUIRED SIGN is the opposite
                # of Re((ik)^n)'s -- NEGATIVE when the leading order is
                # 0 mod 4 (matches KS's own w_xxxx, order 4), but POSITIVE
                # when the leading order is 2 mod 4 (ordinary diffusion,
                # e.g. if max_order were ever lowered to 2, dropping
                # w_xxxx entirely and leaving w_xx as the leading term
                # instead) -- getting this backwards would ACTIVELY
                # destabilize rather than bound the operator.
                self._stable_leading_sign = -1.0 if highest_even % 4 == 0 else 1.0
            # FIXED (not learned, not batch-dependent) per-order rescaling
            # (added 2026-09-09, user-directed after a real Stage-2 NaN
            # blowup: "yes please implement that normalization"): raw
            # derivative orders differ by many orders of magnitude
            # (differentiation multiplies by k^n), so an UNnormalized
            # linear regression over them is badly conditioned -- confirmed
            # directly, degree=2's output already reached ~1.3e7 at nominal
            # coefficient scale vs. degree=1's ~550. Dividing order-n's
            # channel by `char_k**n` (char_k = 2*pi*K/L, the characteristic
            # kept wavenumber) brings every term into roughly the same
            # scale before the shared linear layer sees it, restoring good
            # conditioning -- WITHOUT losing interpretability: a learned
            # coefficient on the normalized order-n channel, divided back
            # by `char_k**n`, recovers the coefficient on the TRUE
            # (physical, unnormalized) derivative -- see `field`'s
            # docstring for the exact recovery formula.
            #
            # `poly_norm_power` (added 2026-09-09, user-directed: "less
            # normalization for the polynomial"): generalizes the exponent
            # from a fixed `n` to `n*poly_norm_power` -- `1.0` (default)
            # is the original, NaN-blowup-preventing strength; `<1.0`
            # weakens it (e.g. `0.5` means order-`n`'s channel is divided
            # by `char_k**(n/2)` instead of `char_k**n`, so high-order
            # derivative channels keep more of their raw dynamic range
            # relative to low-order ones), `0.0` disables normalization
            # entirely (every scale becomes `char_k**0=1`). Weakening this
            # is a real re-introduction of the original blowup risk this
            # mechanism exists to prevent -- always re-verify via a direct
            # scale test (not just a short smoke run) before trusting a
            # value below `1.0`, same as the original fix was verified.
            char_k = 2.0 * math.pi * K / L
            norm_scale_list = [char_k ** (n * poly_norm_power) for n in range(max_order + 1)]
            if poly_time_deriv:
                # The new w_t_fd channel is a TEMPORAL, not spatial, quantity
                # -- it has no (i*k)^n Fourier-order interpretation, so the
                # spatial norm_scales' derivation doesn't apply. Reuses
                # index 0's scale (char_k**0=1.0, i.e. no rescaling): w and
                # w_t are expected to share the same O(1) magnitude under
                # this project's unit-variance field normalization and
                # dt_snap=1.0 convention, not a precisely-derived quantity
                # the way the spatial scales are -- flagged here rather
                # than silently assumed.
                norm_scale_list.append(1.0)
            if poly_time_deriv2:
                # Same reasoning/caveat as poly_time_deriv's own scale
                # above, extended to w_tt_fd -- also not precisely derived
                # (a second finite difference of unit-variance data could
                # plausibly have a different characteristic scale than a
                # first difference, but no measurement was made to pin
                # that down; reusing 1.0 keeps the same "flag, don't
                # silently assume" discipline rather than guessing a
                # different number with no more justification).
                norm_scale_list.append(1.0)
            norm_scales = torch.tensor(norm_scale_list, dtype=torch.float32)
            self.register_buffer("_deriv_norm_scales", norm_scales, persistent=False)
            # `poly_fixed_linear_terms` (added 2026-09-11, user-directed:
            # "assume the pde always had -w_xx-w_xxxx, we'll just learn the
            # rest of the terms around this"): architecturally FIXES the
            # PHYSICAL (denormalized) coefficient on specific single-index
            # linear terms (e.g. `{2: -1.0, 4: -1.0}` for w_xx and w_xxxx)
            # to a given constant, matching true KS's own dissipation
            # operator `-w_xx-w_xxxx` exactly -- those columns are NEVER
            # learned (a plain constant substituted in every forward pass
            # in `field()`, completely disconnected from the underlying
            # `poly_coeffs.weight` entry at that position, which therefore
            # receives exactly zero gradient and stays at its init value
            # forever -- the same "zero contribution -> frozen" mechanism
            # `poly_no_constant` already relies on). Every OTHER term
            # (including w*w_x, the classical nonlinearity, and every
            # higher cross/product term the library contains) stays fully
            # learned. Since `correction`'s normalized library entry for
            # order `n` is `derivs[n]/char_k**(n*poly_norm_power)`, fixing
            # the PHYSICAL coefficient to `target` requires the RAW
            # (pre-division) weight entry to be `target *
            # char_k**(n*poly_norm_power)` -- computed once here (matches
            # `_deriv_norm_scales`'s own scale exactly, so this is
            # consistent with whatever `poly_norm_power` is set to).
            if poly_fixed_linear_terms:
                self._fixed_term_raw_values: dict[int, float] = {}
                for order, target_physical in poly_fixed_linear_terms.items():
                    if (order,) not in term_indices:
                        raise ValueError(
                            f"poly_fixed_linear_terms order {order!r} not present in the term "
                            f"library (check max_order={max_order!r}/poly_degree={poly_degree!r}/"
                            f"poly_max_term_order={poly_max_term_order!r}) -- nothing to fix."
                        )
                    pos = term_indices.index((order,))
                    self._fixed_term_raw_values[pos] = target_physical * (char_k ** (order * poly_norm_power))
            # `poly_stable_linear_terms` (added 2026-09-14, Section 174,
            # user-directed): user proposed parametrizing the propagator
            # directly as a KS-shaped template (`u_t = -u*u_x + nu*u_xx +
            # forcing`) with `nu` LEARNABLE but slow-moving, arguing this
            # guarantees bounded dynamics by construction rather than
            # hoping a loss penalty (energy floor, varmatch -- Sections
            # 170-173, ALL FOUR independently converged to the identical
            # collapsed fixed point: D_KY=0, lambda1~-0.04 to -0.05,
            # standalone rollout decaying to ~1e-4 by step 200) discourages
            # collapse after the fact.
            #
            # Diagnosis of *why* the proposed forcing-term form wouldn't
            # have worked (Section 174 discussion): a fixed function of
            # PHYSICAL POSITION (not of the field itself) cannot provide
            # the state-dependent high-wavenumber damping that actually
            # bounds KS -- that mechanism is the `-w_xxxx` term's Fourier
            # symbol dominating `-w_xx`'s at high k (Re(lambda(k)) =
            # c2*(-k^2) + c4*(k^4) -> -inf as k->inf iff c4<0). Sections
            # 169-173 NEVER constrained this: `poly_stable_leading` (if
            # used at all) only pins ONE auto-detected leading-order
            # column's sign; nothing ever stopped w_xx's (order=2)
            # coefficient from drifting positive, which makes
            # Re(lambda(k)) = -|c2|*k^2 - |c4|*k^4 -- damped at EVERY
            # wavenumber, i.e. the exact global-fixed-point collapse
            # measured every single time. This is very likely the actual
            # root cause of the whole 169-173 saga, not a timing/loss-
            # weight issue.
            #
            # This mechanism generalizes `poly_stable_leading` (which only
            # ever targets the single highest-even-order column, with an
            # auto-detected sign meant purely for asymptotic
            # boundedness) to an explicit SET of linear orders, each
            # reparametrized as `c_n = -(raw_n)**2` (architecturally
            # negative, ALWAYS, regardless of raw_n -- exact same
            # "guaranteed sign, never flips" trick `poly_stable_leading`
            # uses, just not restricted to "whichever order happens to be
            # highest"). Used as `{2: -1.0, 4: -1.0}`, this pins BOTH
            # w_xx's and w_xxxx's coefficient signs to match true KS's own
            # `-w_xx-w_xxxx` (order=2 negative supplies the LOW-wavenumber
            # instability that makes chaos possible at all; order=4
            # negative supplies the HIGH-wavenumber damping that bounds
            # it) while leaving their MAGNITUDES fully learnable --
            # `raw_n` is a genuine `nn.Parameter`, receives real gradient,
            # and is exposed via `stable_linear_raw_parameters()` below so
            # the training loop can put it in its own low-LR optimizer
            # param group (`Stage{1,2}TrainingConfig.stable_linear_lr_factor`)
            # -- directly implementing the user's "can't make huge steps
            # in nu" requirement without fighting Adam's own per-parameter
            # adaptive scaling via manual clipping.
            #
            # Distinguish from `poly_fixed_linear_terms`: that mechanism
            # FREEZES a column to a constant (zero gradient, never moves);
            # this one CONSTRAINS the column's sign while leaving its
            # magnitude free to learn (nonzero gradient, moves slowly).
            # Mutually exclusive in practice -- do not name the same order
            # in both (whichever is applied later in `_poly_weight()` -- see
            # that method -- silently wins; not tested/supported as a
            # deliberate combination).
            if poly_stable_linear_terms:
                self._stable_linear_positions: dict[int, int] = {}
                stable_raw_params: list[nn.Parameter] = []
                for order, init_physical in poly_stable_linear_terms.items():
                    if (order,) not in term_indices:
                        raise ValueError(
                            f"poly_stable_linear_terms order {order!r} not present in the term "
                            f"library (check max_order={max_order!r}/poly_degree={poly_degree!r}/"
                            f"poly_max_term_order={poly_max_term_order!r}) -- nothing to constrain."
                        )
                    if init_physical >= 0:
                        raise ValueError(
                            f"poly_stable_linear_terms order {order!r} init_physical={init_physical!r} "
                            "must be < 0 -- this mechanism architecturally guarantees a NEGATIVE "
                            "coefficient (c_n = -(raw)**2), so a non-negative initial target is "
                            "unreachable by construction."
                        )
                    pos = term_indices.index((order,))
                    raw_init = math.sqrt(-init_physical * (char_k ** (order * poly_norm_power)))
                    raw_param = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
                    self.register_parameter(f"_stable_linear_raw_{order}", raw_param)
                    self._stable_linear_positions[pos] = order
                    stable_raw_params.append(raw_param)
                self._stable_linear_raw_params = stable_raw_params
        elif field_kind == "forced_burgers":
            # `field_kind="forced_burgers"` (added 2026-09-14, Section 175,
            # user-directed, refining their own Section 174 proposal after
            # the "learnable KS template" mechanism above turned out
            # numerically expensive -- the stiff `-w_xxxx` term (even
            # sign-guaranteed, not frozen) needs ode_substeps~64 for a
            # stable forward-Euler step here, ~10-15h for a full Stage-2
            # run): user's REVISED idea -- forced Burgers with REAL
            # (positive, unconditionally stabilizing) viscosity, where the
            # forcing is a LEARNED function of the local latent state
            # (not a fixed function of physical position, which couldn't
            # have sustained chaos -- see the Section 174 discussion for
            # why):
            #
            #   w_t = beta*w*w_x + nu*w_xx + g_theta(w, w_x, ..., w^(max_order))
            #
            # `beta` (added 2026-09-14, same section, user-directed after
            # the FIRST version of this mechanism -- hardcoded advection at
            # exactly -1.0, nu_init=1.0, ode_substeps=1 -- diverged to nan
            # by epoch 4/k=6 even under a genuinely gradual k-curriculum
            # ramp: "why don't we give -u*u_x a coefficient term too, if
            # that's the cause of the instability"): the nonlinear
            # advection term has its OWN amplitude-dependent CFL-type
            # explicit-Euler stability constraint, completely independent
            # of nu -- lowering nu alone can never fix an advection-driven
            # blowup. `beta = burgers_beta_max * tanh(raw_beta)`, a
            # genuine nn.Parameter but architecturally bounded to
            # `[-burgers_beta_max, +burgers_beta_max]` regardless of how
            # raw_beta moves (same "hard cap, not just hope" philosophy as
            # `delta_cap`'s own tanh reparametrization elsewhere in this
            # file) -- `raw_beta` initialized to exactly `0` (`beta=0` at
            # construction, matching `zero_init`: the propagator starts as
            # PURE diffusion+forcing, no advection at all, and has to
            # slowly discover -- via the same low-LR `stable_linear_
            # raw_parameters()` param group nu uses -- how much advection
            # strength is safe rather than being handed the full,
            # destabilizing magnitude on step 1). NOTE: in the actual
            # Section 175 launch this was validated jointly with
            # `--prop-delta-cap 0.5` (found, independently, to ALREADY
            # fully resolve the nan by itself, gradual ramp k=2->16 clean)
            # -- `beta` is an additional, complementary safeguard on top of
            # that, not the sole fix.
            # Two structural guarantees, both architectural (not learned
            # correctness, same "can't be wrong" philosophy as
            # `poly_stable_linear_terms`/`poly_stable_leading` above):
            # - Advection coefficient fixed at exactly -1.0 (no learned
            #   parameter) -- matches the user's literal `-u*u_x`, and this
            #   term's own overall scale is already redundant with the
            #   encoder/decoder's free choice of latent scale, so nothing
            #   is lost by not learning it.
            # - `nu = softplus(raw_nu)`, ALWAYS > 0 regardless of raw_nu --
            #   ordinary (not anti-) diffusion, giving Re(lambda(k)) =
            #   -nu*k^2 -> -inf as k -> inf UNCONDITIONALLY, for ANY
            #   positive nu. This is a strictly SIMPLER and more robust
            #   boundedness mechanism than true KS's own `-w_xx-w_xxxx`
            #   (which relies on a delicate low-k/high-k sign cancellation
            #   between two terms) -- and, critically, it removes the
            #   `-w_xxxx` term ENTIRELY, so the stiffest eigenvalue in the
            #   whole operator drops from ~char_k^4 (~100+ at K=49/L=96) to
            #   ~nu*char_k^2 (~10 at nu~1) -- roughly a 10x relaxation of
            #   the forward-Euler stability constraint, i.e. ode_substeps
            #   can come back down from ~64 toward single digits.
            #
            # `g_theta` (the "forcing... learned output from an mlp
            # depending on the latent state", the user's own words) reuses
            # the EXACT SAME shared-pointwise-MLP architecture the
            # existing `field_kind="mlp"` branch below already has
            # (input_proj/blocks/final_ln/output_proj over the same local
            # derivative stack `[w, w_x, ..., w^(max_order)]`) -- no new
            # architecture, just relabeled as the residual forcing term
            # added on top of the two hardcoded/constrained physical
            # terms, rather than being the WHOLE right-hand side.
            # `zero_init=True` zeros `output_proj` so `g_theta(z)=0` at
            # init (forcing starts silent; the propagator starts as pure
            # forced-free viscous Burgers with nu=burgers_nu_init).
            # `burgers_no_forcing=True` (added 2026-09-16, Section 182,
            # user-directed: "let's keep the sakaguchi setup, add a
            # hyperviscosity term and get rid of the forcing mlp term" --
            # Section 181 (kernel A AND beta both forced active) made
            # things WORSE, not better (RMSE climbing monotonically to
            # ~32 over 200 steps, never saturating) -- hypothesis: real
            # KS-class saturation needs 4th-order hyperviscosity to
            # absorb energy the advection term cascades to high k
            # (forced_burgers only ever had 2nd-order diffusion), AND the
            # still-active forcing MLP (magnitude ~0.7-1.0, not
            # negligible even mean-zero) could itself be contributing to
            # or masking the dynamics rather than the pure physical terms
            # alone. Skips building the forcing trunk ENTIRELY (no
            # input_proj/blocks/final_ln/output_proj) -- field() returns
            # a plain zero for `forcing` in this mode -- so the ONLY
            # terms left are the hardcoded/constrained physical ones
            # (advection, diffusion, kernel instability, hyperviscosity),
            # exactly matching Sakaguchi's own equation (1) structurally
            # (nonlocal-kernel-modified diffusion + advection + regular
            # diffusion, no separate learned correction term at all).
            if not burgers_no_forcing:
                self.input_proj = nn.Linear(max_order + 1, hidden)
                self.blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(n_blocks)])
                self.final_ln = nn.LayerNorm(hidden)
                self.output_proj = nn.Linear(hidden, 1)
                if zero_init:
                    nn.init.zeros_(self.output_proj.weight)
                    nn.init.zeros_(self.output_proj.bias)
            # `burgers_beta_fixed` (added 2026-09-16, Section 181,
            # user-directed follow-up to the kernel-A ablation: "if we
            # remove delta_cap what would happen?" -> "[explained: real
            # KS-class saturation comes from the ADVECTION term
            # cascading energy from unstable low-k modes to damped
            # high-k modes -- beta has ALSO stayed near 0 in every one
            # of 175-180's fitted values, so with only A forced
            # unstable (Section 180), delta_cap was the ONLY thing
            # bounding growth, producing an artificial 'freeze at a
            # larger scale' instead of genuine nonlinear saturation]" ->
            # "sure try that"). Exactly analogous to
            # `burgers_kernel_A_fixed`: when set, `beta` is a FIXED
            # buffer (not learnable, no tanh reparametrization) instead
            # of a slow-LR-learnable parameter -- forces the nonlinear
            # advection term to genuinely engage regardless of what
            # gradient descent would have chosen, testing whether real
            # energy-cascade saturation (bounded chaos at the correct
            # physical scale) emerges once BOTH the destabilizing (A)
            # and the saturating-nonlinearity (beta) mechanisms are
            # active simultaneously, rather than delta_cap alone doing
            # all the (artificial) bounding work.
            if burgers_beta_fixed is not None:
                self.register_buffer(
                    "_beta_fixed", torch.tensor(burgers_beta_fixed, dtype=torch.float32),
                    persistent=False,
                )
            else:
                # `burgers_beta_init` (added 2026-09-16, Section 183,
                # user-directed: "let's initialize with the same
                # parameters but let A and beta be trainable" -- Section
                # 182 confirmed FIXING both A and beta didn't produce
                # genuine saturation (identical unbounded-linear-growth
                # signature to 180/181); this tests whether letting them
                # remain LEARNABLE, but starting from the SAME physically-
                # motivated values (-2.0/-1.0) rather than 0, lets
                # gradient descent find a genuine balance the hard-coded
                # values couldn't. Inverse-tanh so
                # burgers_beta_max*tanh(raw_init) == burgers_beta_init
                # EXACTLY at construction (requires
                # |burgers_beta_init| < burgers_beta_max, strictly, since
                # atanh diverges at +-1 -- e.g. beta_max=1.5 for a
                # beta_init of -1.0, leaving real headroom either side).
                raw_beta_init = (
                    math.atanh(burgers_beta_init / burgers_beta_max) if burgers_beta_init != 0.0 else 0.0
                )
                self.raw_beta = nn.Parameter(torch.tensor(raw_beta_init, dtype=torch.float32))
            # Inverse-softplus so softplus(x_init) == burgers_nu_init
            # EXACTLY at construction (softplus(x) = log(1+e^x), so
            # x = log(e^target - 1) = log(expm1(target))) -- shared by
            # both the constant-nu and nonlinear-nu cases below.
            raw_nu_init = math.log(math.expm1(burgers_nu_init))
            if burgers_kernel_instability:
                # `burgers_kernel_instability=True` (added 2026-09-14,
                # Section 177, user-directed after reading Sakaguchi,
                # "A Simple Model for Spatio-Temporal Chaos in an Unstable
                # Burgers Equation," Prog. Theor. Phys. 103 (2000) 703):
                # that paper's own unstable Burgers equation gets its
                # chaos-sustaining instability from a NONLOCAL kernel
                # acting on w_xx instead of a literal -w_xx/-w_xxxx term:
                # `w_t = integral(g(x-x') w_xx(x') dx') + w*w_x + nu*w_xx`,
                # which in Fourier space is exactly diagonal:
                # `dw_hat_k/dt = -[g(k)+nu]*k^2 * w_hat_k`. Their own
                # requirement for genuine bounded chaos: g(k)+nu < 0 for
                # small k (instability -- MISSING from Sections 175/176,
                # whose fitted nu/beta barely moved from init, i.e. pure
                # damping, hence their collapse) while g(k) -> 0 RAPIDLY
                # for large k (so nu alone dominates and damps high
                # wavenumbers -- unlike a k^4 term, whose magnitude GROWS
                # without bound, forcing Section 174's ode_substeps~64).
                #
                # Reuses the EXACT same K-mode spectral representation
                # `z`/`field()` already receives (no extra transform
                # needed) to build a genuinely diagonal Fourier multiplier
                # `Lhat(k) = -(A*exp(-k^2/width^2) + nu) * k^2`:
                # - `A = kernel_A_max*tanh(raw_A)`: learnable, architecturally
                #   bounded to [-A_max, A_max] regardless of training (same
                #   hard-cap philosophy as `beta`) -- this is Sakaguchi's
                #   `g(k)`'s AMPLITUDE, any sign (the destabilizing
                #   ingredient missing from 175/176). raw_A starts at
                #   EXACTLY 0 (A=0 at init).
                # - `width = softplus(raw_width)`: learnable, always > 0 --
                #   the Gaussian envelope's decay scale (Sakaguchi's own
                #   numerical example used a Gaussian bump too:
                #   g(k)=0.4*exp(-16k^2)).
                # - `nu = softplus(raw_nu)`: SAME mechanism/init as the
                #   constant-nu case -- guarantees Lhat(k) -> -nu*k^2 < 0
                #   as k -> infinity UNCONDITIONALLY (the envelope vanishes
                #   regardless of A/width), matching Sakaguchi's own
                #   stability requirement exactly.
                # At A=0 (init), Lhat(k) = -nu*k^2 exactly -- IDENTICAL to
                # the constant-nu case's own initial behavior, so this is
                # a strict generalization, not a different starting point.
                # Note Lhat(0) = 0 identically (the overall k^2 prefactor
                # vanishes at the DC mode) regardless of A/width/nu -- the
                # mean/DC mode is architecturally UNTOUCHED by this term,
                # matching true KS's own exact mass conservation for free.
                # `burgers_kernel_A_fixed` (added 2026-09-16, Section 180,
                # user-directed ablation: "is there any way to get kernel
                # A to activate in 178 and 179?" -- 178/179 both showed A
                # drifting slightly NEGATIVE under gradient descent (more
                # damping, not less), meaning the local k-step MSE
                # gradient actively opposes instability -- a warm-start
                # init alone would likely just get walked back to ~0 over
                # many epochs under the slow-LR group. This is the clean
                # causal test instead: FIX A entirely (not learnable, no
                # tanh reparametrization -- the raw target value itself,
                # since there is no training dynamic to guard against
                # here), architecturally forcing a genuine low-k unstable
                # band regardless of what gradient descent would have
                # chosen. Isolates whether the forcing MLP simply
                # compensates for a forced-unstable A (reproducing the
                # same frozen-band collapse, meaning the kernel was never
                # the bottleneck) or whether real chaos emerges once the
                # model can no longer avoid a destabilizing linear term.
                if burgers_kernel_A_fixed is not None:
                    self.register_buffer(
                        "_kernel_A_fixed", torch.tensor(burgers_kernel_A_fixed, dtype=torch.float32),
                        persistent=False,
                    )
                else:
                    # `burgers_kernel_A_init` (Section 183, same user
                    # direction as `burgers_beta_init` above): starts A
                    # learnable but at a genuinely destabilizing value
                    # instead of 0, testing whether gradient descent can
                    # find a real balance from an already-unstable start
                    # rather than needing to discover instability is
                    # worthwhile from scratch (which it never did in
                    # 175-179). Same inverse-tanh construction, requires
                    # |burgers_kernel_A_init| < burgers_kernel_A_max.
                    raw_A_init = (
                        math.atanh(burgers_kernel_A_init / burgers_kernel_A_max)
                        if burgers_kernel_A_init != 0.0 else 0.0
                    )
                    self.raw_kernel_A = nn.Parameter(torch.tensor(raw_A_init, dtype=torch.float32))
                raw_width_init = math.log(math.expm1(burgers_kernel_width_init))
                self.raw_kernel_width = nn.Parameter(torch.tensor(raw_width_init, dtype=torch.float32))
                self.raw_nu = nn.Parameter(torch.tensor(raw_nu_init, dtype=torch.float32))
                # `burgers_kernel_mu_init` (added 2026-09-16, Section 182,
                # user-directed: "add a hyperviscosity term"): extends
                # Lhat(k) with a genuine 4th-order damping term,
                # `-mu*k^4`, matching true KS's own `-w_xxxx` -- the
                # mechanism the "forced_burgers" family (Section 175
                # onward) deliberately dropped for cost reasons (a k^4
                # term's magnitude grows unboundedly with k, forcing
                # Section 174's ode_substeps~64). `mu = softplus(raw_mu)`,
                # architecturally ALWAYS > 0 (genuine damping, never
                # anti-damping), same mechanism/init-recovery formula as
                # `nu`. Default init (0.01) chosen to respect this
                # backbone's ode_substeps=1 forward-Euler stability
                # budget: at the highest kept wavenumber (k_max=pi for
                # K=49/L=96, k_max^4~97.4), stability requires roughly
                # mu*k_max^4 <~ 2, i.e. mu <~ 0.0205 -- 0.01 leaves
                # genuine headroom even if mu grows somewhat under
                # training (same slow-LR group as nu/A/width, so it
                # cannot grow FAST regardless). `None` disables this term
                # entirely (no mu parameter created, Lhat unchanged from
                # Section 177/180/181's own kernel-only formula).
                if burgers_kernel_mu_init is not None:
                    raw_mu_init = math.log(math.expm1(burgers_kernel_mu_init))
                    self.raw_kernel_mu = nn.Parameter(torch.tensor(raw_mu_init, dtype=torch.float32))
                k_phys = rfft_wavenumbers(K, L)
                self.register_buffer("_kernel_k", k_phys, persistent=False)
                self.register_buffer("_kernel_k_dup", torch.cat([k_phys, k_phys]), persistent=False)
            elif burgers_nonlinear_nu:
                # `burgers_nonlinear_nu=True` (sketched 2026-09-14, same
                # section, user-directed follow-up: "should we consider
                # creating a pde that isn't a polynomial? it could be a
                # more generic nonlinear function of the derivatives" --
                # generalizes the CONSTANT `nu` above to a genuinely
                # nonlinear, STATE-DEPENDENT diffusion coefficient
                # `nu(w, w_x, ..., w^(max_order))`, analogous to real
                # nonlinear-diffusion PDEs (e.g. the porous medium
                # equation `u_t = (D(u) u_x)_x`, D(u)>0 -- or Perona-Malik
                # edge-preserving diffusion), rather than a bare
                # `field_kind="mlp"` unconstrained correction (which would
                # reintroduce exactly the collapse risk Sections 169-173
                # spent this whole arc escaping -- nothing would stop an
                # unconstrained MLP from learning a globally-contractive
                # map).
                #
                # Own SEPARATE small trunk (input_proj_nu/blocks_nu/
                # final_ln_nu/output_proj_nu), independent of the forcing
                # MLP's trunk above -- deliberate, not an efficiency
                # shortcut: keeps `nu`'s entire computation path isolable
                # into its own low-LR `stable_linear_raw_parameters()`
                # group (the "can't make huge steps in nu" requirement),
                # without also slowing down the forcing MLP's own
                # learning by sharing hidden features with it.
                #
                # `nu(x) = softplus(output_proj_nu(trunk_nu(x)))` --
                # architecturally POSITIVE AT EVERY POINT, for every
                # input, regardless of training (softplus applied
                # pointwise) -- the SAME unconditional per-point diffusion
                # guarantee the constant-nu case has, just spatially/
                # state-varying instead of uniform. `output_proj_nu`'s
                # WEIGHT is zero-initialized (so at construction `nu(x)`
                # does not yet depend on the input at all) but its BIAS is
                # set to `raw_nu_init` (not zero) -- giving `nu(x) ==
                # burgers_nu_init` EXACTLY, for every point, matching the
                # constant-nu case's own init value one-for-one. Only the
                # bias needs the inverse-softplus value since the weight
                # contributes nothing at init; as training proceeds the
                # weight (in the slow-LR group) gradually lets `nu` depend
                # on the local derivative stack.
                self.nu_input_proj = nn.Linear(max_order + 1, hidden)
                self.nu_blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(n_blocks)])
                self.nu_final_ln = nn.LayerNorm(hidden)
                self.nu_output_proj = nn.Linear(hidden, 1)
                nn.init.zeros_(self.nu_output_proj.weight)
                nn.init.constant_(self.nu_output_proj.bias, raw_nu_init)
            else:
                self.raw_nu = nn.Parameter(torch.tensor(raw_nu_init, dtype=torch.float32))
        else:
            self.input_proj = nn.Linear(max_order + 1, hidden)
            self.blocks = nn.ModuleList([ResidualMLPBlock(hidden, dropout) for _ in range(n_blocks)])
            self.final_ln = nn.LayerNorm(hidden)
            self.output_proj = nn.Linear(hidden, 1)
            if zero_init:
                nn.init.zeros_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
        if integrator == "etdrk4":
            self._build_etdrk4_coeffs()

    def _build_etdrk4_coeffs(self) -> None:
        """Computes `Lhat(k) = k^2 - k^4` (KS's own true linear symbol, NOT
        learned) for the `K` kept modes, then the ETDRK4 combination
        coefficients (`ks_latent.solver.ks.etdrk4_coefficients`, the SAME
        contour-integral computation the real solver uses) for one
        `1/ode_substeps`-sized sub-step. Each length-`K` coefficient array
        is duplicated to length `2*K` (`Lhat` is real, so it multiplies a
        mode's real and imaginary parts identically) to act directly on
        `z`'s `concat(real, imag)` layout. Computed once here (numpy, at
        construction) rather than per-forward-pass -- `Lhat` is fixed, not
        trainable, so nothing here needs to be differentiable."""
        import numpy as np

        from ks_latent.solver.ks import etdrk4_coefficients

        k = 2.0 * math.pi * np.arange(self.K) / self.L
        Lhat = k**2 - k**4
        h = 1.0 / self.ode_substeps
        coeffs = etdrk4_coefficients(Lhat, dt=h, M=32)

        def _dup(arr: "np.ndarray") -> torch.Tensor:
            # `coeffs.E`/`.E2` come back complex128 (Lhat is cast to complex128
            # internally for the contour-integral averaging), with EXACTLY zero
            # imaginary part since Lhat itself is real -- `.real` makes that
            # explicit instead of relying on torch's (correct, but warning-
            # producing) implicit truncation.
            arr = np.real(arr)
            return torch.tensor(np.concatenate([arr, arr]), dtype=torch.float32)

        self.register_buffer("_etd_E", _dup(coeffs.E), persistent=False)
        self.register_buffer("_etd_E2", _dup(coeffs.E2), persistent=False)
        self.register_buffer("_etd_Q", _dup(coeffs.Q), persistent=False)
        self.register_buffer("_etd_f1", _dup(coeffs.f1), persistent=False)
        self.register_buffer("_etd_f2", _dup(coeffs.f2), persistent=False)
        self.register_buffer("_etd_f3", _dup(coeffs.f3), persistent=False)

    def _encode(self, w: torch.Tensor) -> torch.Tensor:
        return encode_to_spectrum(w, self.K)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return decode_from_spectrum(z, self.K, self.N_w)

    def _poly_weight(self) -> torch.Tensor:
        """`self.poly_coeffs.weight` after applying every active
        reparametrization/substitution (`poly_stable_leading`,
        `poly_no_constant`, `poly_fixed_linear_terms`,
        `poly_stable_linear_terms`, `poly_exclude_nonconservative`, in that
        order) -- factored out of `field()` so `nonlinear_field()` (below)
        can reuse the EXACT same effective weight without duplicating the
        substitution logic. Only called when at least one of those five is
        active (`field()`/`nonlinear_field()` both guard this)."""
        weight = self.poly_coeffs.weight
        if self.poly_stable_leading:
            # See __init__'s docstring for the full derivation -- forces
            # the leading (highest even-order) linear coefficient to
            # `self._stable_leading_sign * (raw)^2` (always on the
            # required side of zero for THAT order, per the (ik)^n
            # period-4 sign cycle -- negative when the leading order is 0
            # mod 4, positive when 2 mod 4) so the induced linear
            # operator's eigenvalue real part is guaranteed to go to -inf
            # as wavenumber -> inf (bounded/well-posed), without
            # constraining any other coefficient.
            raw = weight[:, self._stable_leading_idx]
            stable_col = self._stable_leading_sign * (raw**2)
            weight = torch.cat(
                [weight[:, : self._stable_leading_idx], stable_col.unsqueeze(1),
                 weight[:, self._stable_leading_idx + 1 :]],
                dim=1,
            )
        if self.poly_no_constant:
            # See __init__'s docstring -- `()` is always term_indices[0],
            # zeroed every forward pass so it never contributes and never
            # receives gradient.
            weight = torch.cat([torch.zeros_like(weight[:, :1]), weight[:, 1:]], dim=1)
        if self.poly_fixed_linear_terms:
            # See __init__'s docstring -- each specified column is
            # replaced by a plain constant (the precomputed raw value
            # matching the target PHYSICAL coefficient), disconnected
            # from `poly_coeffs.weight` entirely, so it can never move and
            # never receives gradient. Applied LAST (after
            # stable_leading/no_constant) so it always wins if a position
            # happens to overlap.
            cols = []
            for i in range(weight.shape[1]):
                if i in self._fixed_term_raw_values:
                    cols.append(torch.full_like(weight[:, i : i + 1], self._fixed_term_raw_values[i]))
                else:
                    cols.append(weight[:, i : i + 1])
            weight = torch.cat(cols, dim=1)
        if self.poly_stable_linear_terms:
            # See __init__'s docstring -- each specified column is
            # replaced by `-(raw_n)**2` (a DIFFERENTIABLE expression, not
            # a frozen constant like `poly_fixed_linear_terms` above --
            # `raw_n` is a genuine nn.Parameter and receives real
            # gradient), architecturally guaranteeing a negative sign
            # regardless of how `raw_n` moves under training. Applied
            # after `poly_fixed_linear_terms` so it wins if a position
            # happens to overlap (not a supported/tested combination in
            # practice -- see the docstring above).
            cols = []
            for i in range(weight.shape[1]):
                if i in self._stable_linear_positions:
                    order = self._stable_linear_positions[i]
                    raw = getattr(self, f"_stable_linear_raw_{order}")
                    cols.append((-(raw**2)).reshape(1, 1).expand(weight.shape[0], 1))
                else:
                    cols.append(weight[:, i : i + 1])
            weight = torch.cat(cols, dim=1)
        if self.poly_exclude_nonconservative:
            # See __init__'s docstring for the full derivation -- zeros
            # every length-2 (two-factor) term whose combined derivative
            # order is even (e.g. w_x*w_x, w*w), the terms proven to
            # generically violate true KS's exact mean-conservation law.
            # Multiplicative masking (not `poly_fixed_linear_terms`'s
            # `torch.full_like` substitution) since the target is always
            # exactly 0 -- gradient through multiplying by a constant 0 is
            # exactly zero at those positions, same "permanently zero,
            # never receives gradient" guarantee.
            weight = weight * (1.0 - self._nonconservative_term_mask)
        return weight

    def kernel_Lhat(self) -> torch.Tensor:
        """`(K,)` diagonal Fourier multiplier Lhat(k) for the CURRENT
        parameter values -- `field_kind == "forced_burgers"` with
        `burgers_kernel_instability=True` only. A pure function of
        A/width/nu/mu (and the fixed wavenumbers `_kernel_k`), completely
        INDEPENDENT of any input `z` or batch of real data -- unlike
        every other diagnostic/loss quantity in this file, this can be
        (and is, by `kernel_unstable_floor_loss` in
        `ks_latent.training.losses`) computed with zero forward passes,
        directly from the model's own parameters.

        Added 2026-09-16, Section 183, user-directed: "we should also
        increase the regularizer that tries to keep at least 13 unstable
        modes in the pde, or if that doesn't exist, implement it" -- no
        such regularizer existed (the only lever was the initial value
        of A, which Sections 175-179 showed gradient descent drifting
        AWAY from instability under the ordinary k-step MSE loss, and
        Sections 180-182 showed FIXING A/beta doesn't produce genuine
        saturation either) -- exposing Lhat(k) directly here is what lets
        the training loop penalize losing unstable modes without needing
        a real batch to compute it from."""
        if not (self.field_kind == "forced_burgers" and self.burgers_kernel_instability):
            raise ValueError(
                "kernel_Lhat() requires field_kind='forced_burgers' with "
                f"burgers_kernel_instability=True, got field_kind={self.field_kind!r}, "
                f"burgers_kernel_instability={self.burgers_kernel_instability!r}."
            )
        A = (
            self._kernel_A_fixed if self.burgers_kernel_A_fixed is not None
            else self.burgers_kernel_A_max * torch.tanh(self.raw_kernel_A)
        )
        width = F.softplus(self.raw_kernel_width)
        nu = F.softplus(self.raw_nu)
        Lhat = -(A * torch.exp(-(self._kernel_k**2) / (width**2)) + nu) * (self._kernel_k**2)
        if hasattr(self, "raw_kernel_mu"):
            mu = F.softplus(self.raw_kernel_mu)
            Lhat = Lhat - mu * (self._kernel_k**4)
        return Lhat

    def stable_linear_raw_parameters(self) -> list[nn.Parameter]:
        """The raw scalar `nn.Parameter`s underlying `poly_stable_linear_terms`
        (`field_kind in ("polynomial", "chebyshev")`) or `raw_nu`/`raw_beta`
        (`field_kind == "forced_burgers"`) -- empty list if neither
        mechanism is active. Exposed so a training loop can put these in
        their own low-LR optimizer param group --
        `Stage{1,2}TrainingConfig.stable_linear_lr_factor` -- implementing
        "can't make huge steps in nu" without fighting Adam's per-parameter
        adaptive scaling via manual gradient clipping.

        `burgers_nonlinear_nu=True`: `raw_nu` doesn't exist (nu is a
        function, not a scalar) -- the ENTIRE independent nu-trunk
        (`nu_input_proj`/`nu_blocks`/`nu_final_ln`/`nu_output_proj`) goes
        into the slow-LR group instead, so "can't make huge steps"
        applies to the whole `nu(x)` computation, not just a final
        linear layer on top of fast-moving shared features.

        `burgers_kernel_instability=True` (Section 177, Sakaguchi-style
        nonlocal kernel instability): `raw_kernel_A`, `raw_kernel_width`,
        and `raw_nu` (all three scalars controlling the diagonal Fourier
        multiplier Lhat(k)) plus `raw_beta` -- EXCEPT when
        `burgers_kernel_A_fixed` is set (Section 180), in which case `A`
        is a fixed buffer, not a parameter, and is excluded here."""
        if self.field_kind in ("polynomial", "chebyshev") and self.poly_stable_linear_terms:
            return list(self._stable_linear_raw_params)
        if self.field_kind == "forced_burgers":
            beta_params = [] if self.burgers_beta_fixed is not None else [self.raw_beta]
            if self.burgers_kernel_instability:
                kernel_params = (
                    [] if self.burgers_kernel_A_fixed is not None else [self.raw_kernel_A]
                )
                mu_params = [self.raw_kernel_mu] if hasattr(self, "raw_kernel_mu") else []
                return kernel_params + [self.raw_kernel_width, self.raw_nu] + mu_params + beta_params
            if self.burgers_nonlinear_nu:
                nu_params = (
                    list(self.nu_input_proj.parameters())
                    + list(self.nu_blocks.parameters())
                    + list(self.nu_final_ln.parameters())
                    + list(self.nu_output_proj.parameters())
                )
                return nu_params + beta_params
            return [self.raw_nu] + beta_params
        return []

    def _augment_derivs_with_time_derivs(
        self,
        derivs: torch.Tensor,
        z_prev: torch.Tensor | None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Appends the `w_t_fd = (w - w_prev) / dt_snap` channel (when
        `self.poly_time_deriv`) and/or the `w_tt_fd = (w - 2*w_prev +
        w_prev2) / dt_snap**2` channel (when `self.poly_time_deriv2`,
        Section 190 -- the standard 3-point BACKWARD second difference,
        using only PAST states, matching the "approximate them using
        rollout terms" framing: no future state is available or needed)
        to `derivs`, in that order -- shared by `field()` and
        `nonlinear_field()` so the two never disagree about which channel
        is which.

        `z_prev`/`z_prev2`: the previous / previous-previous step's state
        in the SAME (spectral) representation `z` itself uses -- decoded
        to physical fields via `self._decode` right here, the one place
        this class needs it. Either being `None` (no history available,
        e.g. `step_one`'s single-argument call path, or the first one or
        two steps of a fresh rollout) falls back to an architectural zero
        for the channel(s) that need it -- see `poly_time_deriv`/
        `poly_time_deriv2`'s docstrings in `__init__`. `w_tt_fd` needs
        BOTH `z_prev` and `z_prev2`; missing either one gives a zero
        `w_tt_fd`, even if the other is available. No-op (returns `derivs`
        unchanged) when both flags are off, so call sites can invoke this
        unconditionally."""
        if not self.poly_time_deriv and not self.poly_time_deriv2:
            return derivs
        w = derivs[..., 0]
        w_prev = self._decode(z_prev) if z_prev is not None else None
        cols = [derivs]
        if self.poly_time_deriv:
            if w_prev is None:
                w_t_fd = torch.zeros_like(w)
            else:
                w_t_fd = (w - w_prev) / self.poly_time_deriv_dt_snap
            cols.append(w_t_fd.unsqueeze(-1))
        if self.poly_time_deriv2:
            if w_prev is None or z_prev2 is None:
                w_tt_fd = torch.zeros_like(w)
            else:
                w_prev2 = self._decode(z_prev2)
                w_tt_fd = (w - 2.0 * w_prev + w_prev2) / (self.poly_time_deriv_dt_snap**2)
            cols.append(w_tt_fd.unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    def nonlinear_field(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """iLED-style (arXiv:2309.05812, "Interpretable Learning of
        Effective Dynamics for Multiscale Systems") nonlinear-closure-only
        output. Added 2026-09-12, user-directed: "directly replicate
        iLED's stabilization mechanism". iLED's own effective dynamics are
        `dz/dt = A_theta*z + Psi_1(z,h)`: `A_theta` is REPARAMETRIZED as
        `W - W^T - diag(|w|)`, guaranteeing its eigenvalues have
        non-positive real part (a provably stable linear part), and
        `Psi_1` (the nonlinear closure) is separately penalized via
        `||Psi_1||^2` in their loss, keeping the nonlinear correction from
        growing large enough to dominate/destabilize the (already-stable)
        linear part.

        This project's `_SpectralPDEDeltaBody` already has a linear-part
        stability mechanism -- `poly_fixed_linear_terms` (fixes specific
        single-derivative coefficients, e.g. w_xx/w_xxxx, to their TRUE
        physical KS values) and/or `poly_stable_leading` (architecturally
        forces the highest-order even-derivative coefficient's sign so
        Re(lambda(k)) -> -inf as k -> inf) -- which is a STRONGER
        guarantee than iLED's own sign-only reparametrization wherever
        active. What was MISSING is iLED's other half: a penalty on the
        nonlinear closure's own magnitude. `nonlinear_field` returns
        exactly that closure's output alone -- the same `field()`
        computation, but with every LINEAR-or-constant term (length <=1 in
        `_polynomial_term_indices`' enumeration: the constant `()` and
        every single-derivative `(n,)` term) masked to zero via
        `self._nonlinear_term_mask`, isolating the genuinely nonlinear
        (product, length>=2) terms' contribution. `w_pde_nonlinear_l2`
        (Stage1TrainingConfig/Stage2TrainingConfig) penalizes this
        method's squared output directly, the same role iLED's
        `L_non-linearity` plays for `Psi_1`.

        Only defined for `field_kind in ("polynomial", "chebyshev")` -- an
        MLP field has no clean linear/nonlinear split to isolate."""
        if self.field_kind not in ("polynomial", "chebyshev"):
            raise ValueError(
                f"nonlinear_field requires field_kind in ('polynomial', 'chebyshev'), got "
                f"{self.field_kind!r} -- an MLP field has no linear/nonlinear split to isolate."
            )
        derivs = synthesize_derivatives(z, self.K, self.N_w, self.L, self.max_order)
        derivs = self._augment_derivs_with_time_derivs(derivs, z_prev, z_prev2)
        derivs_norm = derivs / self._deriv_norm_scales
        if self.field_kind == "chebyshev":
            library = _chebyshev_library(derivs_norm, self.poly_degree, self.poly_max_term_order, self._time_deriv_free_indices)
        else:
            library = _polynomial_library(derivs_norm, self.poly_degree, self.poly_max_term_order, self._time_deriv_free_indices)
        if self.poly_stable_leading or self.poly_no_constant or self.poly_fixed_linear_terms or self.poly_stable_linear_terms or self.poly_exclude_nonconservative:
            weight = self._poly_weight()
        else:
            weight = self.poly_coeffs.weight
        weight = weight * self._nonlinear_term_mask
        return F.linear(library, weight, None).squeeze(-1)  # (B, N_w)

    def field(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z`: `(B, 2*K)` -> `w_t`: `(B, N_w)`, the shared pointwise
        model's CORRECTION to the field's time derivative at every physical
        point (or, if `physics_prior=False`, the entire estimate -- see
        below), given the EXACT spatial derivative stack synthesized from
        `z`. `field_kind` (constructor arg) selects what "shared pointwise
        model" means:

        - `"mlp"` (default): a small nonlinear MLP over the derivative
          stack -- expressive, but its learned function has no closed
          form a human can read off.
        - `"polynomial"` (added 2026-09-09, user-directed: "expand the pde
          as a polynomial (degree 1 or 2) in all the derivative terms,
          then directly learn the coefficients... more interpretable, and
          potentially more stable"): `correction = poly_coeffs @
          _polynomial_library(derivs, poly_degree)` -- a SINGLE shared
          linear layer over the degree-`<=poly_degree` monomial library
          built from `[w, w_x, w_xx, ...]` (see `_polynomial_library`'s
          own docstring for the exact term order). `poly_coeffs.weight`
          IS the learned PDE's own coefficient vector, directly readable
          after training (e.g. `degree=2` should in principle be able to
          recover something close to KS's own `w_t = -w*w_x -w_xx -w_xxxx`
          if the encoder cooperates -- the coefficient on the `w*w_x`
          monomial, the `w_xx` monomial, and the `w_xxxx` monomial should
          each land near -1, everything else near 0). Also potentially
          more stable than the MLP: it can't represent arbitrarily sharp
          nonlinear responses to a single noisy high-derivative channel
          the way a multi-layer MLP can, since every term is literally
          just a monomial times a constant.

          **Normalized before the library is built** (added 2026-09-09,
          fixing a real Stage-2 NaN blowup found empirically): each order-
          `n` channel is divided by `self._deriv_norm_scales[n]`
          (`= (2*pi*K/L)**n`, a FIXED constant, not learned/batch-
          dependent) before `_polynomial_library` sees it -- raw
          derivative orders differ by many orders of magnitude (k^n
          amplification), so an unnormalized regression is badly
          conditioned (confirmed directly: degree=2's raw output reached
          ~1.3e7 at nominal coefficient scale vs. degree=1's ~550, and a
          real Stage-2 continuation diverged to NaN within 2 epochs
          without this). To recover the coefficient on the TRUE
          (unnormalized, physical) derivative from a trained
          `poly_coeffs.weight`, divide the learned coefficient on a
          degree-1 term at order `n` by `self._deriv_norm_scales[n]`, or
          for a degree-2 cross term at orders `(i, j)`, by
          `self._deriv_norm_scales[i] * self._deriv_norm_scales[j]`.

        `physics_prior=True` (added 2026-09-08, see `PropagatorConfig.
        spectral_physics_prior`'s docstring): adds a FIXED (not learned)
        baseline computed directly from the exact true KS equation,
        `w_t = -w*w_x - w_xx - w_xxxx`, before the learned correction --
        integrator-aware to avoid double-counting `"etdrk4"`'s own separate
        exact treatment of the linear part:
        - `"etdrk4"`: only `-w*w_x` (the nonlinear term) is baked in here --
          `-w_xx-w_xxxx` is already handled exactly via `exp(dt*Lhat)`
          elsewhere (`_etdrk4_step`); adding it here too would apply it
          twice.
        - `"euler"`/`"rk4"`: the FULL `-w*w_x - w_xx - w_xxxx` is baked in
          (these integrators have no separate linear treatment).

        The learned `correction` is scaled by `self.correction_scale`
        (default `1.0`) before being added to `prior` -- see
        `correction_scale`'s own docstring in `__init__` for the homotopy/
        continuation motivation.

        `z_prev`/`z_prev2` (added 2026-09-18, Section 189/190): the
        PREVIOUS / PREVIOUS-PREVIOUS step's state in the SAME `(B, 2*K)`
        spectral representation `z` itself uses -- decoded to physical
        fields internally (`_augment_derivs_with_time_derivs`), used only
        when `self.poly_time_deriv`/`self.poly_time_deriv2` are set
        (`field_kind in ("polynomial", "chebyshev")` only -- see those
        flags' docstrings in `__init__` for the full mechanism and where
        `z_prev`/`z_prev2` come from at training vs. rollout time).
        Ignored entirely otherwise; `None` (default) is the "no history
        available" case for each."""
        derivs = synthesize_derivatives(z, self.K, self.N_w, self.L, self.max_order)  # (B, N_w, max_order+1)
        if self.field_kind in ("polynomial", "chebyshev"):
            derivs = self._augment_derivs_with_time_derivs(derivs, z_prev, z_prev2)
            derivs_norm = derivs / self._deriv_norm_scales
            if self.field_kind == "chebyshev":
                library = _chebyshev_library(derivs_norm, self.poly_degree, self.poly_max_term_order, self._time_deriv_free_indices)
            else:
                library = _polynomial_library(derivs_norm, self.poly_degree, self.poly_max_term_order, self._time_deriv_free_indices)  # (B, N_w, n_terms)
            if self.poly_stable_leading or self.poly_no_constant or self.poly_fixed_linear_terms or self.poly_stable_linear_terms or self.poly_exclude_nonconservative:
                weight = self._poly_weight()
                correction = F.linear(library, weight, self.poly_coeffs.bias).squeeze(-1)  # (B, N_w)
            else:
                correction = self.poly_coeffs(library).squeeze(-1)  # (B, N_w)
        elif self.field_kind == "forced_burgers":
            # See __init__'s docstring for the full derivation. `correction`
            # here is the ENTIRE right-hand side (advection + real
            # diffusion + learned forcing), not a residual added to a
            # separate `physics_prior` baseline -- the two constrained
            # physical terms ARE this field_kind's prior, computed inline.
            w = derivs[..., 0]
            w_x = derivs[..., 1]
            w_xx = derivs[..., 2]
            if self.burgers_kernel_instability:
                # See __init__'s docstring for the full derivation
                # (Sakaguchi 2000). Lhat(k) = -(A*exp(-k^2/width^2)+nu)*k^2,
                # a genuinely diagonal Fourier multiplier applied directly
                # to `z` (the K-mode spectral input `field()` already
                # receives, real+imag concatenated, `_kernel_k_dup` gives
                # the matching per-entry physical wavenumber) -- converted
                # back to physical space via `self._decode` (the same
                # fixed transform used everywhere else in this class) to
                # combine with the other (already physical-space) terms.
                A = (
                    self._kernel_A_fixed if self.burgers_kernel_A_fixed is not None
                    else self.burgers_kernel_A_max * torch.tanh(self.raw_kernel_A)
                )
                width = F.softplus(self.raw_kernel_width)
                nu_scalar = F.softplus(self.raw_nu)
                Lhat = -(A * torch.exp(-(self._kernel_k_dup**2) / (width**2)) + nu_scalar) * (
                    self._kernel_k_dup**2
                )
                if hasattr(self, "raw_kernel_mu"):
                    # Hyperviscosity (Section 182): genuine 4th-order
                    # damping, -mu*k^4, mu=softplus(raw_mu) always > 0 --
                    # see __init__'s docstring for the derivation and the
                    # ode_substeps=1 stability budget this init respects.
                    mu = F.softplus(self.raw_kernel_mu)
                    Lhat = Lhat - mu * (self._kernel_k_dup**4)
                linear_term = self._decode(Lhat * z)  # (B, N_w)
            elif self.burgers_nonlinear_nu:
                # nu(x) = softplus(nu_trunk(derivs)(x)) -- POSITIVE
                # everywhere, own independent trunk (see __init__'s
                # docstring for why not shared with the forcing trunk).
                h_nu = self.nu_input_proj(derivs)
                for block in self.nu_blocks:
                    h_nu = block(h_nu)
                h_nu = self.nu_final_ln(h_nu)
                nu = F.softplus(self.nu_output_proj(h_nu).squeeze(-1))  # (B, N_w)
                linear_term = nu * w_xx
            else:
                nu = F.softplus(self.raw_nu)  # scalar, broadcasts
                linear_term = nu * w_xx
            beta = (
                self._beta_fixed if self.burgers_beta_fixed is not None
                else self.burgers_beta_max * torch.tanh(self.raw_beta)
            )
            if self.burgers_no_forcing:
                forcing = torch.zeros_like(w)
            else:
                h = self.input_proj(derivs)
                for block in self.blocks:
                    h = block(h)
                h = self.final_ln(h)
                raw_forcing = self.output_proj(h).squeeze(-1)  # (B, N_w)
                # Two guarantees added 2026-09-14, Section 178, user-directed
                # after Sections 176/177 both diverged (standalone rollout
                # max|z| growing ~linearly, 2.5->51 over 100 steps) while
                # their physically-constrained terms (nu/beta/kernel A) stayed
                # near their safe init values -- the forcing MLP was the
                # actual, unconstrained point of failure. Unlike beta/nu/A,
                # `raw_forcing` had NO structural guarantee at all: it is
                # never trained on states outside Stage 1's own on-attractor
                # k=2 window, so its behavior off-distribution (which is
                # exactly where an autoregressive rollout ends up once
                # anything drifts even slightly) is unconstrained extrapolation.
                # 1. Bounded magnitude: `capped = burgers_forcing_max *
                #    tanh(raw_forcing / burgers_forcing_max)` -- same
                #    architectural-cap philosophy as `delta_cap`/`beta`. Alone
                #    this does NOT stop drift (a bounded-but-biased forcing
                #    still integrates to unbounded growth over many
                #    autoregressive steps), so it's a rate-limit, not a fix by
                #    itself -- paired with guarantee 2 below.
                # 2. EXACT zero spatial mean: `forcing = capped -
                #    capped.mean(dim=-1, keepdim=True)`. This architecturally
                #    guarantees the forcing term can NEVER shift `w`'s total
                #    mass (int(w)dx), matching true KS's own exact mean
                #    conservation -- same principle as `poly_exclude_
                #    nonconservative` above, applied to this field_kind's own
                #    otherwise-unconstrained MLP term. This is the primary fix
                #    for the specific near-uniform amplitude DRIFT observed in
                #    176/177 (a systematic, same-sign forcing every step is
                #    exactly what produces that kind of steady growth; an
                #    exactly-zero-mean forcing cannot have such a bias).
                # Order matters: capping first, then mean-subtracting, means
                # the FINAL result is no longer bounded by exactly
                # burgers_forcing_max (subtracting a mean itself bounded by
                # burgers_forcing_max from values bounded by
                # burgers_forcing_max gives at most 2*burgers_forcing_max) --
                # but the zero-mean property is EXACT regardless, which is
                # what actually matters here. zero_init preserved: at
                # raw_forcing=0, capped=0, mean=0, forcing=0.
                capped_forcing = self.burgers_forcing_max * torch.tanh(raw_forcing / self.burgers_forcing_max)
                forcing = capped_forcing - capped_forcing.mean(dim=-1, keepdim=True)
            correction = beta * w * w_x + linear_term + forcing
        else:
            h = self.input_proj(derivs)
            for block in self.blocks:
                h = block(h)
            h = self.final_ln(h)
            correction = self.output_proj(h).squeeze(-1)  # (B, N_w)
        if not self.physics_prior:
            return correction
        w = derivs[..., 0]
        w_x = derivs[..., 1]
        prior = -w * w_x
        if self.integrator != "etdrk4":
            w_xx = derivs[..., 2]
            w_xxxx = derivs[..., 4]
            prior = prior - w_xx - w_xxxx
        return prior + self.correction_scale * correction

    def _nonlinear_freq(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`N(z)` in the SAME truncated-frequency `(B, 2*K)` representation
        `z` itself uses -- the pseudo-spectral pattern the real solver's
        own `_nonlinear` follows (physical-space pointwise evaluation,
        transformed back to frequency space), with the hand-coded
        `-(ik/2)*FFT(IFFT(v)^2)` formula replaced by `field`'s learned
        pointwise response to the exact derivative stack. `z_prev`/
        `z_prev2`: see `field`'s docstring -- forwarded unchanged."""
        return self._encode(self.field(z, z_prev=z_prev, z_prev2=z_prev2))

    def _etdrk4_step(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One ETDRK4 sub-step, literally the same combination
        `ks_latent.solver.ks.step` uses (Kassam & Trefethen 2005, eq. 20),
        with `_nonlinear_freq` (learned) standing in for that function's
        hand-coded nonlinear term. `z_prev`/`z_prev2`: forwarded, unchanged,
        to every one of the four internal `_nonlinear_freq` evaluations --
        an approximation when `poly_time_deriv`/`poly_time_deriv2` is
        active (the true previous physical field(s) only exist at the
        START of this whole sub-step, not at each RK stage's intermediate
        state; held fixed across stages rather than re-estimated, see
        `poly_time_deriv`'s docstring in `__init__`)."""
        Nv = self._nonlinear_freq(z, z_prev=z_prev, z_prev2=z_prev2)
        a = self._etd_E2 * z + self._etd_Q * Nv
        Na = self._nonlinear_freq(a, z_prev=z_prev, z_prev2=z_prev2)
        b = self._etd_E2 * z + self._etd_Q * Na
        Nb = self._nonlinear_freq(b, z_prev=z_prev, z_prev2=z_prev2)
        cc = self._etd_E2 * a + self._etd_Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear_freq(cc, z_prev=z_prev, z_prev2=z_prev2)
        return self._etd_E * z + Nv * self._etd_f1 + 2.0 * (Na + Nb) * self._etd_f2 + Nc * self._etd_f3

    def forward(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z_prev`/`z_prev2`: see `field`'s docstring -- the previous /
        previous-previous step's state in the same spectral representation
        `z` uses, only meaningful when `self.poly_time_deriv`/
        `self.poly_time_deriv2` are set. Held fixed across every internal
        `field()` call this single
        `forward()` invocation makes (relevant only for `ode_substeps>1`
        or `integrator in ("rk4","etdrk4")` -- this project's actual
        recipes all use `ode_substeps=1`/`"euler"`, where there is exactly
        one `field()` call and no approximation is introduced)."""
        if self.integrator == "etdrk4":
            z_t = z
            for _ in range(self.ode_substeps):
                z_t = self._etdrk4_step(z_t, z_prev=z_prev, z_prev2=z_prev2)
            return z_t - z
        w = self._decode(z)
        if self.integrator == "rk4":
            h = 1.0 / self.ode_substeps
            w_t = w
            for _ in range(self.ode_substeps):
                k1 = self.field(self._encode(w_t), z_prev=z_prev, z_prev2=z_prev2)
                k2 = self.field(self._encode(w_t + 0.5 * h * k1), z_prev=z_prev, z_prev2=z_prev2)
                k3 = self.field(self._encode(w_t + 0.5 * h * k2), z_prev=z_prev, z_prev2=z_prev2)
                k4 = self.field(self._encode(w_t + h * k3), z_prev=z_prev, z_prev2=z_prev2)
                w_t = w_t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            w_next = w_t
        else:
            h = 1.0 / self.ode_substeps
            w_t = w
            for _ in range(self.ode_substeps):
                w_t = w_t + h * self.field(self._encode(w_t), z_prev=z_prev, z_prev2=z_prev2)
            w_next = w_t
        return self._encode(w_next) - z


class _SpectralPDERawDeltaBody(nn.Module):
    """`backbone="spectral_pde_raw"` (added 2026-09-08, see
    docs/sine_transform_pde_plan.md §20, user-directed: "I would like to be
    able to use the encoder and decoder and propagator from 95 ... within
    the propagator, I want to take the fourier transform of the latent
    states, and train the spectral pde with this information (since this
    gives us closed form derivatives.) This then becomes part of the loss").

    Generalizes `_SpectralPDEDeltaBody` to work on the raw latent `z` of
    ANY encoder (`(B, d_latent)`, no assumed spectral structure -- unlike
    `backbone="spectral_pde"`, which requires `z` to already BE a truncated
    rFFT spectrum from a dedicated `spectral_field` encoder). Treats `z`'s
    own `d_latent` indices as if they were a spatial coordinate on a
    periodic ring of circumference `L` (the same "native index as space"
    convention already used elsewhere in this project --
    `spatial_coherence_loss`/`circular_weighted_stats` -- extended here to
    actually synthesize closed-form derivatives along it, not just measure
    correlation banding): a FIXED, non-learned self-FFT
    (`encode_to_spectrum(z, K)`) produces a truncated spectrum `z_hat`,
    fed through the ordinary `_SpectralPDEDeltaBody` machinery UNCHANGED
    (exact derivative synthesis + shared pointwise MLP + integrator), then
    transformed back (`decode_from_spectrum`) to the SAME raw `d_latent`
    space the encoder/`aux`/everything else actually operates in -- so
    `pde_head`'s own delta is directly comparable to `aux`'s.

    No `spectral_physics_prior` option: there is no "true governing
    equation" for an arbitrary, LEARNED latent ordering the way there is
    for a genuine `w=irfft(z)` physical field -- see
    `AuxPropagatorConfig`'s validation, which rejects the combination.

    **`integrator="etdrk4"` caveat, found empirically (2026-09-08):** unlike
    `_SpectralPDEDeltaBody` (where `etdrk4`'s `Lhat=k^2-k^4` is KS's TRUE
    physical dispersion relation -- a correct, load-bearing fact), here it
    bakes that SAME formula onto an arbitrary self-FFT of a LEARNED latent
    ordering, where it has no justification at all. Verified directly: a
    real `train_stage1(..., pde_head=...)` run with `integrator="etdrk4"`
    diverged (loss `0.57 -> 268.0` over 5 epochs); the identical setup
    with `integrator="euler"` trained stably (`0.51 -> 0.48`). `euler`/
    `rk4` impose no dispersion-relation assumption -- the pointwise MLP
    learns the entire right-hand side from the closed-form derivatives,
    which is what this backbone's self-FFT is actually for. `etdrk4`
    remains selectable (to test whether a `k^2-k^4`-like prior happens to
    help a specific learned ordering) but is NOT the default here (see
    `train_stage1_patched.py`'s `--pde-integrator`, default `"euler"`,
    unlike `--spectral-integrator`'s own `"etdrk4"`-favoring guidance for
    genuine `backbone="spectral_pde"`)."""

    def __init__(
        self, d_latent: int, K: int, L: float, max_order: int, hidden: int, n_blocks: int,
        dropout: float, zero_init: bool, integrator: str = "euler", ode_substeps: int = 1,
        field_kind: str = "mlp", poly_degree: int = 2, poly_max_term_order: int | None = None,
        poly_norm_power: float = 1.0, poly_stable_leading: bool = False,
        poly_no_constant: bool = False, poly_fixed_linear_terms: dict[int, float] | None = None,
        poly_exclude_nonconservative: bool = False,
        poly_stable_linear_terms: dict[int, float] | None = None,
        poly_time_deriv: bool = False,
        poly_time_deriv_dt_snap: float = 1.0,
        poly_time_deriv2: bool = False,
        burgers_nu_init: float = 1.0,
        burgers_beta_max: float = 1.0,
        burgers_nonlinear_nu: bool = False,
        burgers_kernel_instability: bool = False,
        burgers_kernel_A_max: float = 1.0,
        burgers_kernel_width_init: float = 1.0,
        burgers_forcing_max: float = 1.0,
        burgers_kernel_A_fixed: float | None = None,
        burgers_beta_fixed: float | None = None,
        burgers_kernel_mu_init: float | None = None,
        burgers_no_forcing: bool = False,
        burgers_kernel_A_init: float = 0.0,
        burgers_beta_init: float = 0.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.K = K
        self.poly_time_deriv = poly_time_deriv
        self.poly_time_deriv2 = poly_time_deriv2
        self.inner = _SpectralPDEDeltaBody(
            K, d_latent, L, max_order, hidden, n_blocks, dropout, zero_init,
            integrator=integrator, ode_substeps=ode_substeps, physics_prior=False,
            field_kind=field_kind, poly_degree=poly_degree, poly_max_term_order=poly_max_term_order,
            poly_norm_power=poly_norm_power, poly_stable_leading=poly_stable_leading,
            poly_no_constant=poly_no_constant, poly_fixed_linear_terms=poly_fixed_linear_terms,
            poly_exclude_nonconservative=poly_exclude_nonconservative,
            poly_stable_linear_terms=poly_stable_linear_terms,
            poly_time_deriv=poly_time_deriv, poly_time_deriv_dt_snap=poly_time_deriv_dt_snap,
            poly_time_deriv2=poly_time_deriv2,
            burgers_nu_init=burgers_nu_init,
            burgers_beta_max=burgers_beta_max,
            burgers_nonlinear_nu=burgers_nonlinear_nu,
            burgers_kernel_instability=burgers_kernel_instability,
            burgers_kernel_A_max=burgers_kernel_A_max,
            burgers_kernel_A_fixed=burgers_kernel_A_fixed,
            burgers_beta_fixed=burgers_beta_fixed,
            burgers_kernel_mu_init=burgers_kernel_mu_init,
            burgers_no_forcing=burgers_no_forcing,
            burgers_kernel_A_init=burgers_kernel_A_init,
            burgers_beta_init=burgers_beta_init,
            burgers_forcing_max=burgers_forcing_max,
            burgers_kernel_width_init=burgers_kernel_width_init,
        )

    def stable_linear_raw_parameters(self) -> list[nn.Parameter]:
        """Delegates to `self.inner.stable_linear_raw_parameters()` -- see
        that method's docstring."""
        return self.inner.stable_linear_raw_parameters()

    def kernel_Lhat(self) -> torch.Tensor:
        """Delegates to `self.inner.kernel_Lhat()` -- see that method's
        docstring."""
        return self.inner.kernel_Lhat()

    @staticmethod
    def _encode_or_none(z: torch.Tensor | None, K: int) -> torch.Tensor | None:
        """`encode_to_spectrum(z, K)` if `z` is given, else `None` --
        shared by `forward`/`nonlinear_field`/`field` for `z_prev`/
        `z_prev2`."""
        return encode_to_spectrum(z, K) if z is not None else None

    def forward(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z_prev` (added 2026-09-18, Section 189)/`z_prev2` (Section 190):
        raw `(B, d_latent)`, the previous / previous-previous step's
        latent state -- only meaningful when `self.poly_time_deriv`/
        `self.poly_time_deriv2` are set (see `_SpectralPDEDeltaBody.
        poly_time_deriv`'s docstring in `__init__`). Encoded to the SAME
        self-FFT spectrum `z_hat` uses, then forwarded to `self.inner`'s
        own `z_prev`/`z_prev2` kwargs -- `self.inner` (a
        `_SpectralPDEDeltaBody`) does its own physical decode internally,
        right where it's needed (see that class's
        `_augment_derivs_with_time_derivs`)."""
        z_hat = encode_to_spectrum(z, self.K)
        z_hat_prev = self._encode_or_none(z_prev, self.K)
        z_hat_prev2 = self._encode_or_none(z_prev2, self.K)
        z_hat_next = z_hat + self.inner(z_hat, z_prev=z_hat_prev, z_prev2=z_hat_prev2)
        z_next = decode_from_spectrum(z_hat_next, self.K, self.d_latent)
        return z_next - z

    def nonlinear_field(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z`: raw `(B, d_latent)` -> nonlinear-only physical-space
        correction, `(B, d_latent)` -- delegates to `self.inner.
        nonlinear_field` on the self-FFT spectrum, exactly mirroring
        `forward()`'s own `z_hat = encode_to_spectrum(z, self.K)` step.
        See `_SpectralPDEDeltaBody.nonlinear_field`'s docstring for the
        full iLED (arXiv:2309.05812) motivation. `z_prev`/`z_prev2`: see
        `forward()`'s docstring."""
        z_hat = encode_to_spectrum(z, self.K)
        z_hat_prev = self._encode_or_none(z_prev, self.K)
        z_hat_prev2 = self._encode_or_none(z_prev2, self.K)
        return self.inner.nonlinear_field(z_hat, z_prev=z_hat_prev, z_prev2=z_hat_prev2)

    def field(
        self,
        z: torch.Tensor,
        z_prev: torch.Tensor | None = None,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`z`: raw `(B, d_latent)` -> FULL physical-space correction
        (every term, linear and nonlinear), `(B, d_latent)` -- delegates
        to `self.inner.field` on the self-FFT spectrum, exactly mirroring
        `forward()`'s own `z_hat = encode_to_spectrum(z, self.K)` step.
        Added 2026-09-12 alongside `mean_conservation_loss` (see
        `ks_latent.training.loops.pde_head_field_output`'s docstring) --
        exposes the same quantity `nonlinear_field` isolates a SUBSET of,
        uniformly across both `_SpectralPDEDeltaBody` (which already has
        its own native `field()`) and this raw-body wrapper, so callers
        can use `pde_head.body.field(z)` regardless of backbone. `z_prev`/
        `z_prev2`: see `forward()`'s docstring."""
        z_hat = encode_to_spectrum(z, self.K)
        z_hat_prev = self._encode_or_none(z_prev, self.K)
        z_hat_prev2 = self._encode_or_none(z_prev2, self.K)
        return self.inner.field(z_hat, z_prev=z_hat_prev, z_prev2=z_hat_prev2)


class _TransformerDeltaBody(nn.Module):
    """Tokenize `z` (d,) into `n_tokens` chunks, run a couple of
    (optionally banded/local) self-attention blocks, project back to `d`.
    See module docstring for the "local = adjacent index" caveat."""

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        dropout: float,
        zero_init: bool,
        attn_window: int | None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        self.n_tokens = n_tokens
        self.chunk_size = d_latent // n_tokens
        self.token_embed = nn.Linear(self.chunk_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model, dropout=dropout,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)
        mask = _build_local_attention_mask(n_tokens, attn_window)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tokens = z.view(B, self.n_tokens, self.chunk_size)
        tokens = self.token_embed(tokens) + self.pos_embed
        tokens = self.transformer(tokens, mask=self.attn_mask)
        delta = self.token_unembed(tokens)
        return delta.reshape(B, self.n_tokens * self.chunk_size)


class _ViTDeltaBody(nn.Module):
    """Tokenize `z` into `n_tokens` chunks, add a *circular* (not learned)
    positional encoding, run `n_layers` ViT-style pre-norm attention+MLP
    blocks (`ks_latent.models.autoencoder_vit.ViTBlock`, the same design
    `KSAutoencoderViT` uses), project back to `chunk_size` per token.

    Unlike `KSAutoencoderViT`, there is **no pooling or bottleneck step**:
    the token grid keeps its full `(n_tokens, d_model)` shape through every
    block, so nothing is compressed to a single vector and re-expanded --
    this is a plain same-shape seq2seq transformer over the tokenized
    latent, not an autoencoder. Requested explicitly by the user as "an
    architecture like the ViT architecture ... without dimension
    reduction/expansion" (2026-08-29), to test whether the ViT autoencoder's
    win (see CLAUDE_CODE_BRIEF.md §5.1/5.2) carries over to the propagator.

    Note on `CircularPositionalEncoding` here: it assumes ring adjacency
    among the `n_tokens` chunks of the latent index, which -- exactly as
    for `_TransformerDeltaBody`'s banded attention mask -- is an *assumed*
    ordering, not a physically meaningful one, until Phase 10's
    spatially-organized latent field exists. Using the circular (rather
    than the transformer backbone's learned) encoding here is specifically
    the thing being tested, not a claim that the ring assumption is known
    to be correct.

    `pos_encoding` (added 2026-08-29, user-directed): `"circular"` (default)
    pairs `CircularPositionalEncoding` with `attn_window`-if-given measured
    by **ring** distance (`build_ring_local_attention_mask`) -- consistency
    matters, since token `0` and token `n_tokens-1` are ring-adjacent to
    the positional encoding, so the attention restriction has to agree
    with that or the two would be testing incompatible assumptions at
    once. `"linear"` swaps in `LinearPositionalEncoding` (fixed,
    non-learned, but *not* periodic) paired with `build_local_attention_mask`
    (linear distance, the same one `_TransformerDeltaBody` uses) -- same
    `ViTBlock`/`mlp_ratio` architecture either way, isolating exactly the
    periodic-vs-non-periodic assumption from every other difference between
    this backbone and `"transformer"`. Non-causal either way (symmetric by
    distance, not direction).

    `token_window` (added 2026-08-29, user-directed): overlapping-patch
    tokenization on the latent index -- `None` (default) tiles `z` into
    `n_tokens` non-overlapping `chunk_size`-wide slices as before; a value
    `> chunk_size` gives each token a wider, circularly-padded input slice
    while its *output* delta slot stays exactly `chunk_size` wide (no
    overlap-add needed). See `KSAutoencoderViT`'s docstring for the same
    idea applied to physical-space patches, and
    `circular_overlap_tokenize`'s docstring for the construction.
    """

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        mlp_ratio: int,
        dropout: float,
        zero_init: bool,
        attn_window: int | None,
        pos_encoding: str = "circular",
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        if pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {pos_encoding!r}")
        self.n_tokens = n_tokens
        self.chunk_size = d_latent // n_tokens
        # token_window (added 2026-08-29, user-directed): overlapping-patch
        # tokenization on the latent index, same construction as
        # KSAutoencoderViT's -- see ViTAutoencoderConfig's docstring and
        # circular_overlap_tokenize.
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        self.pos = (
            CircularPositionalEncoding(n_tokens, d_model)
            if pos_encoding == "circular"
            else LinearPositionalEncoding(n_tokens, d_model)
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)
        mask_fn = build_ring_local_attention_mask if pos_encoding == "circular" else build_local_attention_mask
        mask = mask_fn(n_tokens, attn_window)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tokens = circular_overlap_tokenize(z, self.chunk_size, self.token_window, self.n_tokens)
        h = self.pos(self.token_embed(tokens))
        for block in self.blocks:
            h = block(h, attn_mask=self.attn_mask)
        h = self.norm(h)
        delta = self.token_unembed(h)
        return delta.reshape(B, self.n_tokens * self.chunk_size)


class _ViTHistoryDeltaBody(nn.Module):
    """`mode="history"`'s body (added 2026-08-29, user-directed): tokenizes
    *every* state in a length-`n_history` history `z_hist` (oldest to
    newest) the same way `_ViTDeltaBody` tokenizes a single state, adds a
    shared spatial positional encoding plus a learned per-time-step offset,
    and lets all `n_history * n_tokens` tokens attend **jointly across
    both space and time** -- a token can draw on any spatial position at
    any point in the history, not just its own position at the current
    step. Only the *most recent* time-slice's tokens are projected back
    out to a `chunk_size`-wide delta; the older states are context the
    attention can use, never additional outputs (so the output shape
    matches every other backbone: `(B, d_latent)`, one delta for the
    current state only).

    Motivated directly by the user's framing: "the original model for this
    project used a propagator with information from the current step and
    a step in the past" (`mode="two_step"`, fixed at exactly 2 states, MLP
    only) -- this generalizes that to an arbitrary, configurable history
    length using genuine spatio-temporal attention (not just wider
    per-token input channels, which would not be specific to a ViT at
    all).

    The attention mask (if `attn_window` is set) restricts *spatial*
    distance only, tiled identically across every pair of time-slices --
    two tokens at the same or different times can attend if their spatial
    positions are within `attn_window`, regardless of *how far apart in
    time* they are (temporal reach is always full). `pos_encoding`/
    `token_window` behave exactly as in `_ViTDeltaBody` (see its and
    `ViTAutoencoderConfig`'s docstrings), applied identically to every
    time-slice.
    """

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        mlp_ratio: int,
        dropout: float,
        zero_init: bool,
        attn_window: int | None,
        pos_encoding: str,
        n_history: int,
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        if pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {pos_encoding!r}")
        if n_history < 2:
            raise ValueError(f"n_history must be >= 2, got {n_history!r}")
        self.n_tokens = n_tokens
        self.n_history = n_history
        self.chunk_size = d_latent // n_tokens
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        self.pos = (
            CircularPositionalEncoding(n_tokens, d_model)
            if pos_encoding == "circular"
            else LinearPositionalEncoding(n_tokens, d_model)
        )
        # Learned per-time-step offset, shared across spatial tokens; zero-init
        # so at init every time-slice carries only the (nonzero) spatial
        # encoding -- the model must learn to tell time-slices apart from data.
        self.time_embed = nn.Parameter(torch.zeros(n_history, 1, d_model))
        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)
        mask_fn = build_ring_local_attention_mask if pos_encoding == "circular" else build_local_attention_mask
        spatial_mask = mask_fn(n_tokens, attn_window)
        # Tile the (n_tokens, n_tokens) spatial mask into an (n_history*n_tokens,
        # n_history*n_tokens) block matrix: restricts spatial distance only,
        # identically regardless of which pair of time-slices is attending.
        mask = spatial_mask.repeat(n_history, n_history) if spatial_mask is not None else None
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """`z_hist`: `(B, n_history, d_latent)`, oldest to newest. Returns
        the delta for the *current* (most recent) state only: `(B, d_latent)`."""
        B, H, d = z_hist.shape
        if H != self.n_history:
            raise ValueError(f"z_hist has {H} states, expected n_history={self.n_history}")
        flat = z_hist.reshape(B * H, d)
        tokens = circular_overlap_tokenize(flat, self.chunk_size, self.token_window, self.n_tokens)
        h = self.pos(self.token_embed(tokens))  # (B*H, n_tokens, d_model)
        h = h.view(B, H, self.n_tokens, -1) + self.time_embed
        h = h.reshape(B, H * self.n_tokens, -1)
        for block in self.blocks:
            h = block(h, attn_mask=self.attn_mask)
        h = self.norm(h)
        h = h.view(B, H, self.n_tokens, -1)
        delta = self.token_unembed(h[:, -1])  # only the current time-slice
        return delta.reshape(B, d)


class _FNOViTDeltaBody(nn.Module):
    """`backbone="fno_vit"`, `mode="markovian"` (added 2026-08-30,
    user-directed "Solution 2" for the fixed-point collapse -- see
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 5): tokenizes `z` the
    same way `_ViTDeltaBody` does, runs `fno_n_layers` `FNOLayer` (from
    `ks_latent.models.autoencoder_vit`, shared with `KSAutoencoderViT`'s
    optional FNO encoder/decoder) spectral-conv blocks, THEN `token_n_layers`
    `ViTBlock` attention blocks on the same token grid, then projects back
    to `chunk_size` per token. No pooling/bottleneck (same design choice as
    `_ViTDeltaBody`): the token grid keeps its full `(n_tokens, d_model)`
    shape throughout.

    Motivation (user, 2026-08-30): "a combination of a Fourier neural
    operator and a vision transformer... in Fourier space we preserve
    spatial relationships and may therefore more easily learn structure in
    latent space." The FNO half gives cheap, exact global mixing through a
    handful of learned Fourier modes; the ViT half adds content-adaptive,
    higher-order interactions on top of that global context. See
    `_FNOViTHistoryDeltaBody` for `mode="history"`.

    `pos_encoding`/`attn_window`/`token_window` govern the ViT half exactly
    as in `_ViTDeltaBody` (see its docstring) -- the FNO half's own
    circularity comes unconditionally from `rfft`'s periodicity assumption,
    independent of `pos_encoding`.
    """

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        mlp_ratio: int,
        dropout: float,
        zero_init: bool,
        attn_window: int | None,
        pos_encoding: str,
        fno_modes: int | None,
        fno_n_layers: int,
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        if pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {pos_encoding!r}")
        self.n_tokens = n_tokens
        self.chunk_size = d_latent // n_tokens
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        self.pos = (
            CircularPositionalEncoding(n_tokens, d_model)
            if pos_encoding == "circular"
            else LinearPositionalEncoding(n_tokens, d_model)
        )
        modes = fno_modes if fno_modes is not None else n_tokens // 2 + 1
        self.fno_layers = nn.ModuleList([FNOLayer(d_model, modes) for _ in range(fno_n_layers)])
        self.vit_blocks = nn.ModuleList(
            [ViTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)
        mask_fn = build_ring_local_attention_mask if pos_encoding == "circular" else build_local_attention_mask
        mask = mask_fn(n_tokens, attn_window)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tokens = circular_overlap_tokenize(z, self.chunk_size, self.token_window, self.n_tokens)
        h = self.pos(self.token_embed(tokens))  # (B, n_tokens, d_model)
        h = apply_fno_layers(h, self.fno_layers)
        for block in self.vit_blocks:
            h = block(h, attn_mask=self.attn_mask)
        h = self.norm(h)
        delta = self.token_unembed(h)
        return delta.reshape(B, self.n_tokens * self.chunk_size)


class _FNOMLPDeltaBody(nn.Module):
    """`backbone="fno_mlp"`, `mode="markovian"` -- see `PropagatorConfig`'s
    `backbone="fno_mlp"` docstring section for the full motivation
    (Section 66's discovered latent-index periodicity) and the
    translation-equivariance argument. Structurally `_FNOViTDeltaBody`
    with its `vit_blocks` (`ViTBlock`, attention) replaced by
    `mlp_blocks` (`TokenMLPBlock`, no attention) and NO positional
    encoding anywhere -- both are necessary for the whole body to stay
    translation-equivariant end to end (a positional encoding, or an
    attention layer that could learn position-dependent behavior, would
    each independently break it)."""

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        mlp_ratio: int,
        n_mlp_layers: int,
        dropout: float,
        zero_init: bool,
        fno_modes: int | None,
        fno_n_layers: int,
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        self.n_tokens = n_tokens
        self.chunk_size = d_latent // n_tokens
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        modes = fno_modes if fno_modes is not None else n_tokens // 2 + 1
        self.fno_layers = nn.ModuleList([FNOLayer(d_model, modes) for _ in range(fno_n_layers)])
        self.mlp_blocks = nn.ModuleList(
            [TokenMLPBlock(d_model, mlp_ratio, dropout) for _ in range(n_mlp_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tokens = circular_overlap_tokenize(z, self.chunk_size, self.token_window, self.n_tokens)
        h = self.token_embed(tokens)  # (B, n_tokens, d_model) -- NO positional encoding
        h = apply_fno_layers(h, self.fno_layers)
        for block in self.mlp_blocks:
            h = block(h)
        h = self.norm(h)
        delta = self.token_unembed(h)
        return delta.reshape(B, self.n_tokens * self.chunk_size)


class _FNOViTHistoryDeltaBody(nn.Module):
    """`backbone="fno_vit"`, `mode="history"` (added 2026-08-30,
    user-directed: combine the FNO+ViT hybrid with the multi-step history
    propagator -- "each propagator should train on history from the current
    step and the last two steps"). The `_ViTHistoryDeltaBody` counterpart of
    `_FNOViTDeltaBody`: tokenizes every state in the length-`n_history`
    history the same way, runs `fno_n_layers` `FNOLayer` spectral-conv
    blocks INDEPENDENTLY PER TIME-SLICE (spectral conv over the spatial
    `n_tokens` axis only -- time is not part of the FFT, since "the FNO
    preserves SPATIAL relationships" is the whole motivation, not temporal
    ones, which joint attention already handles), THEN lets all
    `n_history * n_tokens` tokens attend jointly across space and time
    exactly as `_ViTHistoryDeltaBody` does, then projects only the
    most-recent time-slice back to a `chunk_size`-wide delta.
    """

    def __init__(
        self,
        d_latent: int,
        n_tokens: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        mlp_ratio: int,
        dropout: float,
        zero_init: bool,
        attn_window: int | None,
        pos_encoding: str,
        fno_modes: int | None,
        fno_n_layers: int,
        n_history: int,
        token_window: int | None = None,
    ):
        super().__init__()
        if d_latent % n_tokens != 0:
            raise ValueError(f"d_latent={d_latent} must be divisible by n_tokens={n_tokens}")
        if pos_encoding not in ("circular", "linear"):
            raise ValueError(f"pos_encoding must be 'circular' or 'linear', got {pos_encoding!r}")
        if n_history < 2:
            raise ValueError(f"n_history must be >= 2, got {n_history!r}")
        self.n_tokens = n_tokens
        self.n_history = n_history
        self.chunk_size = d_latent // n_tokens
        self.token_window = token_window if token_window is not None else self.chunk_size
        self.token_embed = nn.Linear(self.token_window, d_model)
        self.pos = (
            CircularPositionalEncoding(n_tokens, d_model)
            if pos_encoding == "circular"
            else LinearPositionalEncoding(n_tokens, d_model)
        )
        # Zero-init, same reasoning as _ViTHistoryDeltaBody's time_embed: at
        # init every time-slice carries only the (nonzero) spatial encoding.
        self.time_embed = nn.Parameter(torch.zeros(n_history, 1, d_model))
        modes = fno_modes if fno_modes is not None else n_tokens // 2 + 1
        self.fno_layers = nn.ModuleList([FNOLayer(d_model, modes) for _ in range(fno_n_layers)])
        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.token_unembed = nn.Linear(d_model, self.chunk_size)
        if zero_init:
            nn.init.zeros_(self.token_unembed.weight)
            nn.init.zeros_(self.token_unembed.bias)
        mask_fn = build_ring_local_attention_mask if pos_encoding == "circular" else build_local_attention_mask
        spatial_mask = mask_fn(n_tokens, attn_window)
        mask = spatial_mask.repeat(n_history, n_history) if spatial_mask is not None else None
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """`z_hist`: `(B, n_history, d_latent)`, oldest to newest. Returns
        the delta for the *current* (most recent) state only: `(B, d_latent)`."""
        B, H, d = z_hist.shape
        if H != self.n_history:
            raise ValueError(f"z_hist has {H} states, expected n_history={self.n_history}")
        flat = z_hist.reshape(B * H, d)
        tokens = circular_overlap_tokenize(flat, self.chunk_size, self.token_window, self.n_tokens)
        h = self.pos(self.token_embed(tokens))  # (B*H, n_tokens, d_model)
        h = apply_fno_layers(h, self.fno_layers)  # per-(batch*time)-slice spatial FNO
        h = h.view(B, H, self.n_tokens, -1) + self.time_embed
        h = h.reshape(B, H * self.n_tokens, -1)
        for block in self.blocks:
            h = block(h, attn_mask=self.attn_mask)
        h = self.norm(h)
        h = h.view(B, H, self.n_tokens, -1)
        delta = self.token_unembed(h[:, -1])  # only the current time-slice
        return delta.reshape(B, d)


class _FourierMLPHistoryDeltaBody(nn.Module):
    """`backbone="fourier_mlp"`, `mode="history"` (`n_history>=2`) OR
    `mode="markovian"` (added 2026-09-04, user-directed: "make the
    fourier mlp markovian" -- Section 84; constructed with `n_history=1`
    by `LatentPropagator.__init__`'s markovian branch, and called via a
    single `(B, 1, d_latent)`-reshaped current state rather than a real
    multi-state history -- see `LatentPropagator.step_one`) -- see
    `PropagatorConfig`'s `backbone="fourier_mlp"` docstring section for
    the full motivation (Section 66's discovered latent-index
    periodicity) and how this differs from `"fno_mlp"` (a LEARNED
    spectral-conv operator replacing the raw state) vs. this backbone (a
    FIXED Fourier featurization concatenated alongside the raw state).
    At `n_history=1` the "concatenate every history state's features"
    step below is trivially a no-op (there is only one state to
    concatenate), so the class needs no special-casing for the markovian
    case beyond accepting `n_history=1` at all -- the masked raw-value
    path (`masked_body`, when `mask_window` is set) already only ever
    reads the CURRENT (`z_hist[:, -1]`) state regardless of `n_history`.

    For each of the `n_history` states: raw `d_latent` values, plus the
    real and imaginary parts of its first `n_modes` `rfft` frequencies
    (`2*n_modes` numbers). No tokenization, no attention, no positional
    encoding.

    `mask_window=None` (default, unchanged behavior): every history
    state's [raw || Fourier] features are flattened together into one
    vector and fed through a single `MLPDeltaBody` (the same plain
    residual-MLP body `backbone="mlp"` uses) -- fully dense, every raw
    value and every Fourier coefficient free to interact with every
    other.

    `mask_window` set (added 2026-09-03, user-directed: "use all the
    fourier coefficients, but only let real variables interact with their
    neighbors"): splits into two independent, SUMMED contributions
    instead of one dense body:
      - a MASKED path on the CURRENT state's raw values only
        (`_MaskedMLPDeltaBody`, the same circular-band-masked
        dimension-preserving residual MLP `backbone="masked_mlp"` uses,
        at radius `mask_window`) -- raw value `i` can only ever influence
        (or be influenced through this path by) raw values within
        `mask_window` of it.
      - a DENSE path on EVERY history state's Fourier features, ALL of
        them, fully free to interact (`MLPDeltaBody`) -- "use all the
        fourier coefficients" un-restricted, since a single Fourier
        coefficient is already a global summary of the whole state, so
        masking it "by index" the way raw values are masked has no
        analogous physical meaning.
    Past states' raw values are not fed to the masked path (only the
    current state's are) -- their information is still available, just
    exclusively through their Fourier features, which are a redundant,
    information-preserving representation of the same state at the
    default `fno_modes=None` (full spectrum). Both paths' output layers
    are zero-initialized when `zero_init=True`, so their sum is
    identity-at-init exactly like the unmasked path.

    `fourier_ifft_readout` (added 2026-09-03, user-directed: "apply an
    inverse fft to the frequency component output of the fourier mlp to
    map back to state space ... in the propagator"): replaces the
    Fourier-features-only sub-network (`fourier_body` above) with
    `FourierIFFTBody` (see that class's docstring): an `MLPDeltaBody` that
    predicts frequency-domain coefficients, explicitly inverse-transformed
    (`irfft`) back to `d_latent` instead of read out by an arbitrary
    linear layer. When `mask_window` is set, this is SUMMED with the
    unchanged masked raw-value path (`masked_body`) exactly as before --
    the same two-network structure `FourierMLPAutoencoderConfig`'s encoder
    uses (corrected 2026-09-03, user-directed: "I want the propagator and
    the decoder to have the same structure as the encoder... but I want
    the attn_window to be much larger" -- an earlier version of this
    option wrongly dropped the raw-value path entirely when `mask_window`
    was `None` ("dense"), instead of the intended meaning: keep the
    two-network structure with a much larger `mask_window` than the
    encoder's). `mask_window=None` still means no raw-value path at all
    (pure Fourier+`irfft`, since a plain concatenated raw+Fourier `body`
    has no distinct frequency-only output to invert) -- to get the
    two-network structure, set `mask_window`/`attn_window` to a large but
    finite radius.

    `nonexpansive` (added 2026-09-04, see `ResidualMLPBlock`'s docstring):
    threaded into `masked_body`/`fourier_body`/`body`, AND switches
    `_fourier_features`'s `rfft` to `norm="ortho"` (an isometry, matching
    `FourierIFFTBody`'s own `irfft` doing the same -- see that class's
    `nonexpansive` docstring). The two-network SUM (`masked_body(...) +
    fourier_body(...)`) becomes an AVERAGE (`0.5*(...)`) when
    `nonexpansive=True`: even if both branches are individually <=1-
    Lipschitz w.r.t. their (different) inputs, their SUM is only
    guaranteed <=2-Lipschitz -- averaging keeps the combined map a
    genuine convex combination, <=1-Lipschitz overall (same triangle-
    inequality argument as the residual blocks)."""

    def __init__(
        self,
        d_latent: int,
        n_history: int,
        fno_modes: int | None,
        hidden: int,
        n_blocks: int,
        dropout: float,
        zero_init: bool,
        mask_window: int | None = None,
        fourier_ifft_readout: bool = False,
        nonexpansive: bool = False,
    ):
        super().__init__()
        if n_history < 1:
            raise ValueError(f"n_history must be >= 1, got {n_history!r}")
        self.d_latent = d_latent
        self.n_history = n_history
        self.mask_window = mask_window
        self.fourier_ifft_readout = fourier_ifft_readout
        self.nonexpansive = nonexpansive
        max_modes = d_latent // 2 + 1
        self.n_modes = min(fno_modes, max_modes) if fno_modes is not None else max_modes
        if mask_window is None:
            self.masked_body = None
            if fourier_ifft_readout:
                self.body = None
                self.fourier_body = FourierIFFTBody(
                    n_history * self.n_modes, d_latent, hidden, n_blocks, dropout, zero_init,
                    nonexpansive=nonexpansive,
                )
            else:
                in_dim = n_history * (d_latent + 2 * self.n_modes)
                self.body = MLPDeltaBody(in_dim, d_latent, hidden, n_blocks, dropout, zero_init, nonexpansive=nonexpansive)
                self.fourier_body = None
        else:
            self.body = None
            self.masked_body = _MaskedMLPDeltaBody(
                d_latent, n_blocks, mask_window, dropout, zero_init, nonexpansive=nonexpansive,
            )
            if fourier_ifft_readout:
                self.fourier_body = FourierIFFTBody(
                    n_history * self.n_modes, d_latent, hidden, n_blocks, dropout, zero_init,
                    nonexpansive=nonexpansive,
                )
            else:
                fourier_in_dim = n_history * 2 * self.n_modes
                self.fourier_body = MLPDeltaBody(
                    fourier_in_dim, d_latent, hidden, n_blocks, dropout, zero_init, nonexpansive=nonexpansive,
                )

    def _fourier_features(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)`. Returns `(B, 2*n_modes)`: real then
        imaginary parts of the first `n_modes` `rfft` frequencies. Explicit
        `float()` upcast -- same bfloat16-under-autocast bug class as
        `SpectralConv1d`/`logdet_barrier_loss` (`torch.fft.rfft` doesn't
        support bfloat16)."""
        orig_dtype = z.dtype
        norm = "ortho" if self.nonexpansive else None
        z_ft = torch.fft.rfft(z.float(), dim=-1, norm=norm)[:, : self.n_modes]
        return torch.cat([z_ft.real, z_ft.imag], dim=-1).to(orig_dtype)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """`z_hist`: `(B, n_history, d_latent)`, oldest to newest. Returns
        the delta for the *current* (most recent) state only: `(B, d_latent)`."""
        B, H, d = z_hist.shape
        if H != self.n_history:
            raise ValueError(f"z_hist has {H} states, expected n_history={self.n_history}")
        if self.mask_window is None:
            if self.fourier_ifft_readout:
                fourier_parts = [self._fourier_features(z_hist[:, h]) for h in range(H)]
                return self.fourier_body(torch.cat(fourier_parts, dim=-1))
            parts = []
            for h in range(H):
                z_h = z_hist[:, h]
                parts.append(z_h)
                parts.append(self._fourier_features(z_h))
            return self.body(torch.cat(parts, dim=-1))
        if self.fourier_ifft_readout:
            fourier_parts = [self._fourier_features(z_hist[:, h]) for h in range(H)]
            masked_out = self.masked_body(z_hist[:, -1])
            fourier_out = self.fourier_body(torch.cat(fourier_parts, dim=-1))
            return 0.5 * (masked_out + fourier_out) if self.nonexpansive else masked_out + fourier_out
        fourier_parts = [self._fourier_features(z_hist[:, h]) for h in range(H)]
        masked_out = self.masked_body(z_hist[:, -1])
        fourier_out = self.fourier_body(torch.cat(fourier_parts, dim=-1))
        return 0.5 * (masked_out + fourier_out) if self.nonexpansive else masked_out + fourier_out


class LatentPropagator(nn.Module):
    """`cfg.mode == "two_step"`: `(z_{n-1}, z_n) -> z_{n+1} = z_n + delta`.
    `cfg.mode == "markovian"`: `z_n -> z_{n+1} = z_n + M(z_n)`.

    Delta (residual) parameterization plus zero-initializing the output
    head make the model exactly the identity at init in *either* mode
    (brief §5.2): an initial multi-step rollout is constant instead of
    exploding. See `test_propagator_is_identity_at_init`.

    `.step(z_prev, z_curr)` is the uniform external interface for `"two_step"`
    and `"markovian"` (in `"markovian"` mode `z_prev` is accepted and
    ignored) so every other consumer in this codebase (DA cycling, the
    Lyapunov adapters, the D3 coupling diagnostic) works unchanged
    regardless of which of *those two* modes is active -- `.step_one(z)`
    is the direct single-argument form, clearer at call sites that only
    ever have a single current state in hand.

    `mode="history"` (added 2026-08-29, user-directed) cannot fit through
    that fixed-arity interface -- it needs `.step_history(z_hist)`/
    `.rollout_history(z_hist, k)` instead (see `PropagatorConfig`'s
    docstring); calling `.step`/`.rollout`/`.step_one` on a
    `mode="history"` propagator raises immediately rather than silently
    truncating or misinterpreting the history (fail loudly, brief ground
    rule 2).
    """

    def __init__(self, cfg: PropagatorConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode
        d = cfg.d_latent
        if cfg.mode == "two_step":
            self.body = MLPDeltaBody(2 * d, d, cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init)
        elif cfg.mode == "markovian":
            if cfg.backbone == "mlp":
                self.body = MLPDeltaBody(d, d, cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init)
            elif cfg.backbone == "transformer":
                self.body = _TransformerDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_nhead, cfg.token_n_layers,
                    cfg.dropout, cfg.zero_init, cfg.attn_window,
                )
            elif cfg.backbone == "vit":
                self.body = _ViTDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_nhead, cfg.token_n_layers,
                    cfg.token_mlp_ratio, cfg.dropout, cfg.zero_init, cfg.attn_window,
                    cfg.pos_encoding, cfg.token_window,
                )
            elif cfg.backbone == "fno_vit":
                self.body = _FNOViTDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_nhead, cfg.token_n_layers,
                    cfg.token_mlp_ratio, cfg.dropout, cfg.zero_init, cfg.attn_window,
                    cfg.pos_encoding, cfg.fno_modes, cfg.fno_n_layers, cfg.token_window,
                )
            elif cfg.backbone == "fno_mlp":
                self.body = _FNOMLPDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_mlp_ratio, cfg.token_n_layers,
                    cfg.dropout, cfg.zero_init, cfg.fno_modes, cfg.fno_n_layers, cfg.token_window,
                )
            elif cfg.backbone == "local_mlp":
                # _validate_propagator_mode_backbone/__post_init__ already
                # enforce attn_window is not None for this backbone.
                self.body = _LocalMLPDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_n_layers, cfg.token_mlp_ratio,
                    cfg.attn_window, cfg.dropout, cfg.zero_init, cfg.token_window,
                )
            elif cfg.backbone == "masked_mlp":
                # attn_window doubles as the mask's circular-band radius here
                # too (None = fully dense "full mlp"); unlike "local_mlp", None
                # is explicitly ALLOWED (it's the warm-start target).
                self.body = _MaskedMLPDeltaBody(d, cfg.n_blocks, cfg.attn_window, cfg.dropout, cfg.zero_init)
            elif cfg.backbone == "masked_mlp_wide":
                # _validate_propagator_mode_backbone/__post_init__ already
                # enforce attn_window is not None for this backbone. cfg.hidden
                # is the (typically very wide, see PropagatorConfig's docstring)
                # single hidden layer's width; cfg.n_blocks is unused (single
                # layer, not a configurable depth).
                self.body = _MaskedMLPWideDeltaBody(d, cfg.hidden, cfg.attn_window, cfg.zero_init)
            elif cfg.backbone == "masked_mlp_expand":
                # _validate_propagator_mode_backbone/__post_init__ already
                # enforce attn_window is not None for this backbone.
                # cfg.masked_mlp_expand_factor sizes the middle layer's
                # width (hidden = expand_factor * d); cfg.hidden/cfg.n_blocks
                # are unused (three fixed layers, not a configurable depth).
                self.body = _MaskedMLPExpandDeltaBody(
                    d, cfg.attn_window, cfg.masked_mlp_expand_factor, cfg.zero_init
                )
            elif cfg.backbone == "spectral_pde":
                # __post_init__ already enforces spectral_K/spectral_N_w set
                # and d_latent == 2*spectral_K.
                self.body = _SpectralPDEDeltaBody(
                    cfg.spectral_K, cfg.spectral_N_w, cfg.spectral_L, cfg.spectral_max_order,
                    cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init,
                    integrator=cfg.spectral_integrator, ode_substeps=cfg.ode_substeps,
                    physics_prior=cfg.spectral_physics_prior,
                    field_kind=cfg.spectral_field_kind, poly_degree=cfg.spectral_poly_degree,
                    poly_max_term_order=cfg.spectral_poly_max_term_order,
                    poly_norm_power=cfg.spectral_poly_norm_power,
                    poly_stable_leading=cfg.spectral_poly_stable_leading,
                    poly_no_constant=cfg.spectral_poly_no_constant,
                    poly_fixed_linear_terms=cfg.spectral_poly_fixed_linear_terms,
                    poly_exclude_nonconservative=cfg.spectral_poly_exclude_nonconservative,
                    poly_stable_linear_terms=cfg.spectral_poly_stable_linear_terms,
                    poly_time_deriv=cfg.spectral_poly_time_deriv,
                    poly_time_deriv_dt_snap=cfg.spectral_poly_time_deriv_dt_snap,
                    poly_time_deriv2=cfg.spectral_poly_time_deriv2,
                    burgers_nu_init=cfg.spectral_burgers_nu_init,
                    burgers_beta_max=cfg.spectral_burgers_beta_max,
                    burgers_nonlinear_nu=cfg.spectral_burgers_nonlinear_nu,
                    burgers_kernel_instability=cfg.spectral_burgers_kernel_instability,
                    burgers_kernel_A_max=cfg.spectral_burgers_kernel_A_max,
                    burgers_kernel_width_init=cfg.spectral_burgers_kernel_width_init,
                    burgers_forcing_max=cfg.spectral_burgers_forcing_max,
                    burgers_kernel_A_fixed=cfg.spectral_burgers_kernel_A_fixed,
                    burgers_beta_fixed=cfg.spectral_burgers_beta_fixed,
                    burgers_kernel_mu_init=cfg.spectral_burgers_kernel_mu_init,
                    burgers_no_forcing=cfg.spectral_burgers_no_forcing,
                    burgers_kernel_A_init=cfg.spectral_burgers_kernel_A_init,
                    burgers_beta_init=cfg.spectral_burgers_beta_init,
                )
            elif cfg.backbone == "spectral_pde_raw":
                # __post_init__ already enforces spectral_K set and in range.
                self.body = _SpectralPDERawDeltaBody(
                    d, cfg.spectral_K, cfg.spectral_L, cfg.spectral_max_order,
                    cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init,
                    integrator=cfg.spectral_integrator, ode_substeps=cfg.ode_substeps,
                    field_kind=cfg.spectral_field_kind, poly_degree=cfg.spectral_poly_degree,
                    poly_max_term_order=cfg.spectral_poly_max_term_order,
                    poly_norm_power=cfg.spectral_poly_norm_power,
                    poly_stable_leading=cfg.spectral_poly_stable_leading,
                    poly_no_constant=cfg.spectral_poly_no_constant,
                    poly_fixed_linear_terms=cfg.spectral_poly_fixed_linear_terms,
                    poly_exclude_nonconservative=cfg.spectral_poly_exclude_nonconservative,
                    poly_stable_linear_terms=cfg.spectral_poly_stable_linear_terms,
                    poly_time_deriv=cfg.spectral_poly_time_deriv,
                    poly_time_deriv_dt_snap=cfg.spectral_poly_time_deriv_dt_snap,
                    poly_time_deriv2=cfg.spectral_poly_time_deriv2,
                    burgers_nu_init=cfg.spectral_burgers_nu_init,
                    burgers_beta_max=cfg.spectral_burgers_beta_max,
                    burgers_nonlinear_nu=cfg.spectral_burgers_nonlinear_nu,
                    burgers_kernel_instability=cfg.spectral_burgers_kernel_instability,
                    burgers_kernel_A_max=cfg.spectral_burgers_kernel_A_max,
                    burgers_kernel_width_init=cfg.spectral_burgers_kernel_width_init,
                    burgers_forcing_max=cfg.spectral_burgers_forcing_max,
                    burgers_kernel_A_fixed=cfg.spectral_burgers_kernel_A_fixed,
                    burgers_beta_fixed=cfg.spectral_burgers_beta_fixed,
                    burgers_kernel_mu_init=cfg.spectral_burgers_kernel_mu_init,
                    burgers_no_forcing=cfg.spectral_burgers_no_forcing,
                    burgers_kernel_A_init=cfg.spectral_burgers_kernel_A_init,
                    burgers_beta_init=cfg.spectral_burgers_beta_init,
                )
            elif cfg.backbone == "node":
                # _validate_propagator_mode_backbone/__post_init__ already
                # enforce attn_window is not None for this backbone.
                self.body = _NeuralODEDeltaBody(
                    d, cfg.attn_window, cfg.hidden, cfg.n_blocks, cfg.ode_substeps, cfg.zero_init
                )
            elif cfg.backbone == "cnn":
                self.body = _CNNDeltaBody(
                    d, cfg.hidden, cfg.n_blocks, cfg.cnn_kernel_size, cfg.dropout, cfg.zero_init
                )
            elif cfg.backbone == "site_conv":
                # _validate_propagator_mode_backbone/__post_init__ already
                # enforce attn_window is not None and d == site_conv_n_sites
                # * site_conv_local_channels for this backbone.
                self.body = _SiteConvDeltaBody(
                    cfg.site_conv_n_sites, cfg.site_conv_local_channels, cfg.site_conv_hidden,
                    cfg.site_conv_n_layers, cfg.attn_window, cfg.zero_init,
                )
            elif cfg.backbone == "fourier_mlp":
                # n_history=1 (not cfg.n_history, which is meaningless here --
                # see PropagatorConfig's docstring): a single current state,
                # no separate history states to featurize. step_one reshapes
                # the (B, d) input to (B, 1, d) before calling this body -- see
                # step_one's docstring.
                self.body = _FourierMLPHistoryDeltaBody(
                    d, 1, cfg.fno_modes, cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init,
                    mask_window=cfg.attn_window, fourier_ifft_readout=cfg.fourier_ifft_readout,
                    nonexpansive=cfg.nonexpansive,
                )
            else:
                raise ValueError(f"Unknown backbone {cfg.backbone!r}")
        elif cfg.mode == "history":
            # _validate_propagator_mode_backbone already enforces backbone in
            # ('mlp', 'vit', 'fno_vit', 'fourier_mlp').
            if cfg.backbone == "mlp":
                # Generalizes "two_step" (fixed n_history=2, in_dim=2*d) to an
                # arbitrary history length: flatten (n_history, d) -> a single
                # n_history*d vector and reuse the same MLPDeltaBody, no new
                # class needed. Added 2026-08-30, user-directed: after a plain
                # MLP markovian propagator was found to recover chaos (D_KY~21)
                # where every attention-based propagator collapsed (see
                # docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 9), the user
                # asked whether the same holds for the history-mode recipe
                # they'd originally requested.
                self.body = MLPDeltaBody(
                    cfg.n_history * d, d, cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init
                )
            elif cfg.backbone == "vit":
                self.body = _ViTHistoryDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_nhead, cfg.token_n_layers,
                    cfg.token_mlp_ratio, cfg.dropout, cfg.zero_init, cfg.attn_window,
                    cfg.pos_encoding, cfg.n_history, cfg.token_window,
                )
            elif cfg.backbone == "fno_vit":
                self.body = _FNOViTHistoryDeltaBody(
                    d, cfg.n_tokens, cfg.token_d_model, cfg.token_nhead, cfg.token_n_layers,
                    cfg.token_mlp_ratio, cfg.dropout, cfg.zero_init, cfg.attn_window,
                    cfg.pos_encoding, cfg.fno_modes, cfg.fno_n_layers, cfg.n_history, cfg.token_window,
                )
            elif cfg.backbone == "fourier_mlp":
                self.body = _FourierMLPHistoryDeltaBody(
                    d, cfg.n_history, cfg.fno_modes, cfg.hidden, cfg.n_blocks, cfg.dropout, cfg.zero_init,
                    mask_window=cfg.attn_window, fourier_ifft_readout=cfg.fourier_ifft_readout,
                    nonexpansive=cfg.nonexpansive,
                )
            else:
                raise ValueError(f"Unknown backbone {cfg.backbone!r} for mode='history'")
        else:
            raise ValueError(f"Unknown mode {cfg.mode!r}")

    def stable_linear_raw_parameters(self) -> list[nn.Parameter]:
        """Delegates to `self.body.stable_linear_raw_parameters()` when the
        body supports it (`backbone in ("spectral_pde", "spectral_pde_raw")`
        with `poly_stable_linear_terms` set), else `[]`. See
        `_SpectralPDEDeltaBody.stable_linear_raw_parameters`'s docstring."""
        fn = getattr(self.body, "stable_linear_raw_parameters", None)
        return fn() if fn is not None else []

    def kernel_Lhat(self) -> torch.Tensor | None:
        """Delegates to `self.body.kernel_Lhat()` when the body supports
        it (`backbone in ("spectral_pde", "spectral_pde_raw")`,
        `field_kind="forced_burgers"` with `burgers_kernel_instability=
        True`), else `None`. See `_SpectralPDEDeltaBody.kernel_Lhat`'s
        docstring."""
        fn = getattr(self.body, "kernel_Lhat", None)
        return fn() if fn is not None else None

    def capped_delta(self, delta: torch.Tensor, z_ref: torch.Tensor | None = None) -> torch.Tensor:
        """`cap * tanh(delta / cap)` if `cfg.delta_cap` is set, else `delta`
        unchanged -- see `PropagatorConfig.delta_cap`'s docstring. Identity
        for small `delta` (preserves zero-init), a hard per-step bound for
        large `delta`. Public (renamed from `_capped_delta`, 2026-08-30)
        since `EnsemblePropagator` calls this directly on each member to
        compose `z + capped_delta(body(z_noised))` -- applying the member's
        own delta cap to a delta computed from a noised input, but adding
        it to the CLEAN `z` -- see `EnsemblePropagatorConfig`'s docstring
        for why.

        `cfg.delta_cap_relative=True` (added 2026-09-17, Section 184, see
        `PropagatorConfig.delta_cap_relative`'s docstring for the full
        motivation): `cap = delta_cap * z_ref.norm(dim=-1, keepdim=True)`
        instead of the bare `delta_cap` constant -- the allowed step size
        scales with the CURRENT state's own magnitude rather than staying
        fixed, so a genuine instability isn't forced to grow the state's
        scale just to make a fixed absolute cap negligible. Requires
        `z_ref` (the state the delta will be added to) whenever this is
        active. `.clamp_min(1e-6)` guards the degenerate `z_ref=0` case
        (division by a near-zero cap would otherwise blow up)."""
        if self.cfg.delta_cap is None:
            return delta
        if self.cfg.delta_cap_relative:
            if z_ref is None:
                raise ValueError(
                    "capped_delta() with cfg.delta_cap_relative=True requires z_ref "
                    "(the state the delta will be added to) -- got None."
                )
            cap = self.cfg.delta_cap * z_ref.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            cap = self.cfg.delta_cap
        return cap * torch.tanh(delta / cap)

    def _require_not_history(self, called: str) -> None:
        if self.mode == "history":
            raise ValueError(
                f"{called}() is not defined for mode='history' -- there is no well-defined "
                f"way to fit an arbitrary-length (n_history={self.cfg.n_history}) history into "
                f"a fixed 2-argument signature. Use step_history(z_hist)/rollout_history(z_hist, "
                f"k) instead, with z_hist of shape (B, n_history, d_latent), oldest to newest."
            )

    def step_one(self, z: torch.Tensor) -> torch.Tensor:
        """`markovian`-style single-argument step: `z_next = z + M(z)`.

        `backbone="fourier_mlp"` (added 2026-09-04, see
        `_FourierMLPHistoryDeltaBody`'s docstring) is the one markovian
        backbone whose body expects a `(B, n_history, d_latent)` tensor
        (it's the same class `mode="history"` uses, built with
        `n_history=1`) rather than a plain `(B, d_latent)` one -- reshape
        via `unsqueeze(1)` before calling it, matching `step_history`'s
        existing per-backbone `body_input` convention."""
        self._require_not_history("step_one")
        body_input = z.unsqueeze(1) if self.cfg.backbone == "fourier_mlp" else z
        return z + self.capped_delta(self.body(body_input), z_ref=z)

    def step(
        self, z_prev: torch.Tensor, z_curr: torch.Tensor, z_prev2: torch.Tensor | None = None
    ) -> torch.Tensor:
        """See `step_one`'s docstring for the single-argument case.
        `z_prev` is used for `mode="two_step"` as always; for
        `mode="markovian"` it is normally ignored EXCEPT when the body
        supports `poly_time_deriv`/`poly_time_deriv2` (Sections 189/190,
        `_SpectralPDEDeltaBody`/`_SpectralPDERawDeltaBody` with
        `field_kind in ("polynomial", "chebyshev")`), in which case
        `z_prev`/`z_prev2` are threaded through to the body's own
        `z_prev`/`z_prev2` kwargs to build genuine finite-difference
        time-derivative library features -- see `poly_time_deriv`'s
        docstring in `_SpectralPDEDeltaBody.__init__` for the full
        mechanism, including why `step_one` (no `z_prev`/`z_prev2`
        available) stays a well-defined pure function via a zero fallback
        instead of also needing this dispatch. `z_prev2` (added Section
        190, `w_tt`): only meaningful alongside `poly_time_deriv2`; ignored
        (defaults to `None`, the "no second-order history yet" case) for
        every other combination, including plain `mode="two_step"`, which
        never looks at it."""
        self._require_not_history("step")
        if self.mode == "two_step":
            x = torch.cat([z_prev, z_curr], dim=-1)
            return z_curr + self.capped_delta(self.body(x), z_ref=z_curr)
        if getattr(self.body, "poly_time_deriv", False) or getattr(self.body, "poly_time_deriv2", False):
            return z_curr + self.capped_delta(
                self.body(z_curr, z_prev=z_prev, z_prev2=z_prev2), z_ref=z_curr
            )
        return self.step_one(z_curr)  # markovian: z_prev accepted, ignored

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> torch.Tensor:
        return self.step(z_prev, z_curr)

    def rollout(
        self,
        z_prev: torch.Tensor,
        z_curr: torch.Tensor,
        k: int,
        step_noise: float = 0.0,
        z_prev2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for `k` steps; returns `(B, k, d_latent)`,
        `z_{n+1}..z_{n+k}`. Uniform for `"two_step"`/`"markovian"` (`z_prev`
        unused if `mode == "markovian"`, EXCEPT when the body supports
        `poly_time_deriv`/`poly_time_deriv2` -- see `step`'s docstring);
        raises for `mode == "history"` (use `rollout_history` instead).

        `z_prev2` (added 2026-09-18, Section 190, alongside
        `poly_time_deriv2`/`w_tt`): the state TWO steps before `z_curr`,
        if the caller happens to have one (most callers won't -- default
        `None`). Internally maintains a 3-wide sliding window
        (`z_prev2, z_prev, z_curr = z_prev, z_curr, z_next` each
        iteration, mirroring the EXISTING `z_prev, z_curr = z_curr, z_next`
        pattern below), so a genuine two-step-back state becomes available
        starting the SECOND predicted step even when the caller never
        supplies one explicitly -- the user's own 'approximate them using
        rollout terms' framing, extended one order further: only the
        first step (where neither `z_prev` nor `z_prev2` reflects real
        history yet) sees the architectural zero fallback.

        `step_noise` (added 2026-08-29, user-directed): Gaussian noise std
        added to *every* step's output during the rollout (`z_{n+k+1} <-
        z_{n+k+1} + step_noise * eps`), not just the initial `(z_prev,
        z_curr)` pair -- ported from `docs/ML_for_KS_writeup.md` §4.2's
        `sigma_step`, which the reference project used to train recovery
        from the model's own errors. Distinct from (and complementary to)
        `train_stage2`'s existing `noise_in`, which only perturbs the
        rollout's starting pair once; a model can still see purely
        on-manifold states for every step *after* that single perturbed
        start. Default `0.0` (off, unchanged behavior) -- training-only,
        callers must pass `0.0` (or omit) at evaluation time, matching the
        reference project's convention."""
        self._require_not_history("rollout")
        outputs = []
        for _ in range(k):
            z_next = self.step(z_prev, z_curr, z_prev2=z_prev2)
            if step_noise > 0:
                z_next = z_next + step_noise * torch.randn_like(z_next)
            outputs.append(z_next)
            z_prev2, z_prev, z_curr = z_prev, z_curr, z_next
        return torch.stack(outputs, dim=1)

    def step_history(self, z_hist: torch.Tensor) -> torch.Tensor:
        """`mode="history"`'s step: `z_hist`: `(B, n_history, d_latent)`,
        oldest to newest. Returns `z_next = z_hist[:, -1] + delta`."""
        if self.mode != "history":
            raise ValueError(f"step_history() requires mode='history', got mode={self.mode!r}")
        # backbone="mlp"'s MLPDeltaBody expects a flat (B, n_history*d) vector
        # (same convention as "two_step"'s in_dim=2*d); the vit/fno_vit bodies
        # tokenize the (B, n_history, d) tensor themselves.
        body_input = z_hist.reshape(z_hist.shape[0], -1) if self.cfg.backbone == "mlp" else z_hist
        return z_hist[:, -1] + self.capped_delta(self.body(body_input), z_ref=z_hist[:, -1])

    def rollout_history(self, z_hist: torch.Tensor, k: int, step_noise: float = 0.0) -> torch.Tensor:
        """Autoregressive rollout for `k` steps under `mode="history"`,
        sliding the `n_history`-length window forward by one state each
        step (drop the oldest, append the new prediction). Returns `(B, k,
        d_latent)`, `z_{n+1}..z_{n+k}`. `step_noise`: see `rollout`'s
        docstring -- same semantics, applied to the appended state before
        it becomes part of the next step's history window."""
        if self.mode != "history":
            raise ValueError(f"rollout_history() requires mode='history', got mode={self.mode!r}")
        outputs = []
        hist = z_hist
        for _ in range(k):
            z_next = self.step_history(hist)
            if step_noise > 0:
                z_next = z_next + step_noise * torch.randn_like(z_next)
            outputs.append(z_next)
            hist = torch.cat([hist[:, 1:], z_next.unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)


def aux_cfg_to_propagator_cfg(cfg: AuxPropagatorConfig) -> PropagatorConfig:
    return PropagatorConfig(
        d_latent=cfg.d_latent, hidden=cfg.hidden, n_blocks=cfg.n_blocks,
        dropout=cfg.dropout, zero_init=cfg.zero_init, mode=cfg.mode, backbone=cfg.backbone,
        n_tokens=cfg.n_tokens, token_d_model=cfg.token_d_model, token_nhead=cfg.token_nhead,
        token_n_layers=cfg.token_n_layers, token_mlp_ratio=cfg.token_mlp_ratio,
        attn_window=cfg.attn_window, pos_encoding=cfg.pos_encoding, token_window=cfg.token_window,
        n_history=cfg.n_history, delta_cap=cfg.delta_cap, delta_cap_relative=cfg.delta_cap_relative,
        fno_modes=cfg.fno_modes, fno_n_layers=cfg.fno_n_layers,
        fourier_ifft_readout=cfg.fourier_ifft_readout, nonexpansive=cfg.nonexpansive,
        spectral_K=cfg.spectral_K, spectral_N_w=cfg.spectral_N_w, spectral_L=cfg.spectral_L,
        spectral_max_order=cfg.spectral_max_order, spectral_integrator=cfg.spectral_integrator,
        ode_substeps=cfg.ode_substeps, spectral_physics_prior=cfg.spectral_physics_prior,
        spectral_field_kind=cfg.spectral_field_kind, spectral_poly_degree=cfg.spectral_poly_degree,
        spectral_poly_max_term_order=cfg.spectral_poly_max_term_order,
        spectral_poly_norm_power=cfg.spectral_poly_norm_power,
        spectral_poly_stable_leading=cfg.spectral_poly_stable_leading,
        spectral_poly_no_constant=cfg.spectral_poly_no_constant,
        spectral_poly_fixed_linear_terms=cfg.spectral_poly_fixed_linear_terms,
        spectral_poly_exclude_nonconservative=cfg.spectral_poly_exclude_nonconservative,
        spectral_poly_stable_linear_terms=cfg.spectral_poly_stable_linear_terms,
        spectral_poly_time_deriv=cfg.spectral_poly_time_deriv,
        spectral_poly_time_deriv_dt_snap=cfg.spectral_poly_time_deriv_dt_snap,
        spectral_poly_time_deriv2=cfg.spectral_poly_time_deriv2,
        spectral_burgers_nu_init=cfg.spectral_burgers_nu_init,
        spectral_burgers_beta_max=cfg.spectral_burgers_beta_max,
        spectral_burgers_nonlinear_nu=cfg.spectral_burgers_nonlinear_nu,
        spectral_burgers_kernel_instability=cfg.spectral_burgers_kernel_instability,
        spectral_burgers_kernel_A_max=cfg.spectral_burgers_kernel_A_max,
        spectral_burgers_kernel_width_init=cfg.spectral_burgers_kernel_width_init,
        spectral_burgers_forcing_max=cfg.spectral_burgers_forcing_max,
        spectral_burgers_kernel_A_fixed=cfg.spectral_burgers_kernel_A_fixed,
        spectral_burgers_beta_fixed=cfg.spectral_burgers_beta_fixed,
        spectral_burgers_kernel_mu_init=cfg.spectral_burgers_kernel_mu_init,
        spectral_burgers_no_forcing=cfg.spectral_burgers_no_forcing,
        spectral_burgers_kernel_A_init=cfg.spectral_burgers_kernel_A_init,
        spectral_burgers_beta_init=cfg.spectral_burgers_beta_init,
        masked_mlp_expand_factor=cfg.masked_mlp_expand_factor,
        site_conv_n_sites=cfg.site_conv_n_sites, site_conv_local_channels=cfg.site_conv_local_channels,
        site_conv_hidden=cfg.site_conv_hidden, site_conv_n_layers=cfg.site_conv_n_layers,
    )


def build_propagator(cfg: PropagatorConfig | AuxPropagatorConfig) -> LatentPropagator:
    """Single entry point so scripts/tests never have to branch on mode
    themselves -- pass either config type, get a correctly-built propagator."""
    if isinstance(cfg, AuxPropagatorConfig):
        cfg = aux_cfg_to_propagator_cfg(cfg)
    return LatentPropagator(cfg)


class AuxPropagator(LatentPropagator):
    """Same architecture family as `LatentPropagator`, sized (and, since
    `AuxPropagatorConfig` carries its own `mode`/`backbone`, potentially
    shaped) per `AuxPropagatorConfig`."""

    def __init__(self, cfg: AuxPropagatorConfig):
        super().__init__(aux_cfg_to_propagator_cfg(cfg))
