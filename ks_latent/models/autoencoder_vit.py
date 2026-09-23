r"""ViT-style encoder/decoder ("Track B"), added 2026-08-29.

Ported from a reference implementation
(`/Users/daltonjones/Documents/experiments/ks_latent/models.py`,
`ViTEncoder`/`ViTDecoder`/`CircularPositionalEncoding`/`TransformerBlock`)
as a second transformer-based alternative to
`ks_latent/models/autoencoder_patched.py`'s patched-transformer -- see
`ViTAutoencoderConfig`'s docstring for the structural comparison (in short:
this architecture has an explicit, periodicity-respecting positional
encoding on every token before any attention layer; the existing
patched-transformer has none). Exposes the same `encode`/`decode`/
`forward` interface as the other autoencoders in this package, so it is a
drop-in replacement everywhere one is passed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ks_latent.config import ViTAutoencoderConfig
from ks_latent.models.autoencoder_masked_mlp import MaskedLinearRect


def build_ring_local_attention_mask(n_tokens: int, window: int | None) -> torch.Tensor | None:
    """Additive attention bias mask, `-inf` outside a `+-window` band around
    the diagonal, measured by **ring (circular) distance**
    `min(|i-j|, n_tokens-|i-j|)` -- not linear index distance. `None` (full
    attention) if `window` is `None`.

    Ring, not linear, distance matters here specifically because this
    module's tokens sit on a genuinely periodic domain (KS has periodic
    boundary conditions) and `CircularPositionalEncoding` already encodes
    that: token `0` and token `n_tokens-1` are real spatial neighbours. A
    linear-distance mask would treat them as maximally far apart, which
    would contradict the positional encoding it is paired with. Non-causal:
    the mask is symmetric in `i`/`j` (by distance, not by `i-j`), so every
    token attends to `window` neighbours on *both* sides, not just the
    past. Reused by `ks_latent.models.propagator._ViTDeltaBody` for the
    same reason -- see that class's docstring.
    """
    if window is None:
        return None
    idx = torch.arange(n_tokens)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    dist = torch.minimum(diff, n_tokens - diff)
    mask = torch.zeros(n_tokens, n_tokens)
    mask.masked_fill_(dist > window, float("-inf"))
    return mask


def build_local_attention_mask(n_tokens: int, window: int | None) -> torch.Tensor | None:
    """The direct non-periodic counterpart of `build_ring_local_attention_mask`:
    additive attention bias mask, `-inf` outside a `+-window` band around
    the diagonal, measured by **linear** index distance `|i-j|` (no
    wraparound -- token `0` and token `n_tokens-1` are treated as maximally
    far apart). `None` (full attention) if `window` is `None`. Pairs with
    `LinearPositionalEncoding`, for the same consistency reason
    `build_ring_local_attention_mask` pairs with `CircularPositionalEncoding`
    -- see `pos_encoding="linear"` in `ViTAutoencoderConfig`/
    `PropagatorConfig`'s docstrings. Non-causal (symmetric in `i`/`j`).
    """
    if window is None:
        return None
    idx = torch.arange(n_tokens)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = torch.zeros(n_tokens, n_tokens)
    mask.masked_fill_(dist > window, float("-inf"))
    return mask


def circular_overlap_tokenize(x: torch.Tensor, chunk_size: int, window: int, n_tokens: int) -> torch.Tensor:
    """Slice the last dim of `x` (length `n_tokens * chunk_size`, assumed
    periodic) into `n_tokens` possibly-overlapping windows of width
    `window` (>= `chunk_size`), each centered on its own non-overlapping
    `chunk_size`-wide output slot (stride = `chunk_size`), wrapping
    circularly at the domain boundary. `window == chunk_size` is exactly
    the original non-overlapping tiling (`x.view(..., n_tokens,
    chunk_size)`, no padding).

    Overlapping-patch tokenization (added 2026-08-29, user-directed,
    opt-in via `token_window`/`ViTAutoencoderConfig`/`PropagatorConfig`):
    precedent in PVTv2's/T2T-ViT's overlapping patch embedding and
    PatchTST's overlapping-stride patching for 1D time series. Each
    token's *input* sees a wider slice than the `chunk_size`-wide slot it
    is responsible for on the *output* side, smoothing chunk-boundary
    artifacts at the tokenization level itself, with **no change needed
    on the output/decoder side** (which still tiles at exactly
    `chunk_size` spacing) -- deliberately asymmetric to avoid the
    overlap-add reconstruction an overlapping *output* would require.
    """
    if window < chunk_size:
        raise ValueError(f"window={window} must be >= chunk_size={chunk_size}")
    if window == chunk_size:
        return x.view(*x.shape[:-1], n_tokens, chunk_size)
    pad = window - chunk_size
    left_pad, right_pad = pad // 2, pad - pad // 2
    left = x[..., x.shape[-1] - left_pad :] if left_pad > 0 else x[..., :0]
    right = x[..., :right_pad] if right_pad > 0 else x[..., :0]
    x_padded = torch.cat([left, x, right], dim=-1)
    return x_padded.unfold(-1, window, chunk_size)


class LinearPositionalEncoding(nn.Module):
    r"""Fixed (non-learned) sinusoidal positional encoding on a *line* of
    `n_tokens` positions -- the classic Vaswani et al. (2017) formula:

        PE[pos, 2k]   = sin(pos / 10000^(2k/d_model))
        PE[pos, 2k+1] = cos(pos / 10000^(2k/d_model))

    with a single learnable scalar gain (matching `CircularPositionalEncoding`'s
    convention) so the model can trade the encoding off against token content.

    This is the direct non-periodic counterpart of `CircularPositionalEncoding`:
    it is fixed rather than learned (so it is directly comparable -- both
    supply position information from a closed-form construction, not from
    gradient descent), but unlike the circular encoding it makes **no
    periodicity assumption** -- position `0` and position `n_tokens-1` are
    not treated as neighbours. Added 2026-08-29, user-directed, to isolate
    exactly the periodic-vs-non-periodic question from everything else that
    differs between the `vit` and `transformer` backbones (block design,
    feedforward width, learned-vs-fixed): `pos_encoding="linear"` keeps the
    same `ViTBlock`/`mlp_ratio` architecture as `pos_encoding="circular"`
    and swaps only the positional encoding and its paired mask
    (`build_local_attention_mask`, linear distance, instead of
    `build_ring_local_attention_mask`).
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        pos = torch.arange(n_tokens, dtype=torch.float32)[:, None]
        k = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :]
        div = torch.exp(-k * (math.log(10000.0) / d_model))
        ang = pos * div
        pe = torch.zeros(n_tokens, d_model)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang[:, : pe[:, 1::2].shape[1]])
        self.register_buffer("pe_fixed", pe)
        self.gain = nn.Parameter(torch.ones(()))

    @property
    def pe(self) -> torch.Tensor:
        return self.gain * self.pe_fixed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe


class CircularPositionalEncoding(nn.Module):
    r"""Exactly periodic positional encoding on a ring of `n_tokens` tokens.

    KS has periodic boundary conditions, so token 0 and token `n_tokens-1`
    are neighbours. A plain linear or learned positional encoding does not
    know that and has to discover it. This instead builds the encoding
    from the Fourier basis *on the ring*: for harmonics
    `m = 1 .. floor(n_tokens/2)`,

        F[i, :] = (a_m sin(2*pi*m*i/n_tokens), a_m cos(2*pi*m*i/n_tokens))_m,
        a_m = 1/m,

    followed by a **fixed orthogonal** linear map `F @ W` to `d_model`
    channels, with a single learnable scalar gain so the model can trade
    the encoding off against token content.

    Three properties, all of which matter (ported verbatim from the
    reference project's derivation, which measured each one directly):

    * **Exactly periodic.** `F` depends on `i` only through
      `2*pi*m*i/n_tokens`, so index `n_tokens` is index `0`; a whole-field
      cyclic shift permutes the encodings cyclically instead of running
      off an end.
    * **Distance depends only on ring separation.**
      `||F_i - F_j||^2 = sum_m 2*a_m^2*(1 - cos(2*pi*m*(i-j)/n_tokens))`, a
      function of `(i-j) mod n_tokens` alone. This survives the projection
      only if `W @ W.T` is proportional to the identity, which is why `W`
      is orthogonal and fixed rather than learned: a learned (or randomly
      initialized dense) `W` is not an isometry and turns the exact ring
      metric into an asymmetric one.
    * **Nearby tokens are actually near.** That needs the amplitudes to
      *decay*. With flat amplitudes, harmonics above `n_tokens/2` alias and
      every distinct pair of positions ends up nearly equidistant,
      destroying the ring geometry (the reference project measured the
      antipodal token becoming *closer* than the immediate neighbour on a
      24-token ring with flat amplitudes). With `a_m = 1/m` the `m=1` term
      dominates and the distance is monotone in ring separation.
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.n_tokens = n_tokens
        m = torch.arange(1, n_tokens // 2 + 1, dtype=torch.float32)
        i = torch.arange(n_tokens, dtype=torch.float32)[:, None]
        ang = 2 * torch.pi * m[None, :] * i / n_tokens
        amp = 1.0 / m[None, :]
        basis = torch.cat([amp * torch.sin(ang), amp * torch.cos(ang)], dim=1)
        n_basis = basis.shape[1]
        if n_basis > d_model:
            raise ValueError(
                f"d_model={d_model} is smaller than the {n_basis} independent "
                f"ring harmonics of a {n_tokens}-token ring"
            )
        # Orthonormal columns => W @ W.T is proportional to the identity on
        # the basis subspace, so the projection is an isometry and the ring
        # metric is preserved exactly.
        W = torch.empty(n_basis, d_model)
        nn.init.orthogonal_(W)
        self.register_buffer("pe_fixed", basis @ W)  # (n_tokens, d_model)
        self.gain = nn.Parameter(torch.ones(()))

    @property
    def pe(self) -> torch.Tensor:
        return self.gain * self.pe_fixed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe


class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution (Li et al. 2020, Fourier Neural
    Operator). Moved here (2026-08-30, from `ks_latent/models/propagator.py`)
    so both the propagator's `"fno_vit"` backbone AND `KSAutoencoderViT`'s
    encoder/decoder (`ViTAutoencoderConfig.use_fno`, user-directed) can share
    it without a circular import (`propagator.py` already imports `ViTBlock`/
    positional encodings from this module).

    FFT along the token axis (implicitly periodic -- an FFT always treats
    its input as one period of a periodic signal), multiply the lowest
    `modes` frequency components by a learned, per-mode `(in_channels,
    out_channels)` complex weight (independent weights per frequency, no
    sharing across modes), zero every higher frequency, inverse FFT back to
    token space. Gives an `O(modes)` (not `O(n_tokens)`) global receptive
    field: even the DC (mode 0) component alone already mixes every token
    together in one matrix multiply. This is the "in Fourier space we
    preserve spatial relationships" property motivating both uses (see
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 5 for the propagator
    case, Section 6 for the encoder/decoder case) -- note that for the
    encoder/decoder specifically, the token axis IS genuine physical
    position on KS's periodic domain, so the periodicity assumption is
    actually correct here, unlike the propagator's latent-index case where
    it is an unverified ordering assumption (see `PropagatorConfig`'s
    docstring).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(modes, in_channels, out_channels))
        self.weight_imag = nn.Parameter(scale * torch.randn(modes, in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, in_channels, n_tokens)`. Returns `(B, out_channels, n_tokens)`.

        Explicit `float()` upcast (added 2026-09-03, same bfloat16 bug
        class as `logdet_barrier_loss`'s `slogdet` fix): `torch.fft.rfft`/
        `irfft` don't support bfloat16, so this crashes under `--amp`
        autocast otherwise (`RuntimeError: Unsupported dtype BFloat16`).
        Cast the result back to `x`'s original dtype so autocast still
        governs everything downstream as usual.
        """
        B, C, N = x.shape
        orig_dtype = x.dtype
        x = x.float()
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        weight = torch.complex(self.weight_real[:n_modes].float(), self.weight_imag[:n_modes].float())
        out_ft = x_ft.new_zeros(B, self.out_channels, x_ft.shape[-1])
        out_ft[:, :, :n_modes] = torch.einsum("bim,mio->bom", x_ft[:, :, :n_modes], weight)
        return torch.fft.irfft(out_ft, n=N, dim=-1).to(orig_dtype)


class FNOLayer(nn.Module):
    """One FNO block: a spectral-conv path (global, low-frequency) summed
    with a pointwise (kernel-1 conv, i.e. a per-token `Linear`) local skip
    path, then `GELU` -- the standard FNO layer design. The pointwise path
    exists so purely local/high-frequency information isn't thrown away by
    the spectral path's mode truncation. Public (moved from propagator.py's
    `_FNOLayer`, 2026-08-30) for the same cross-module reuse reason as
    `SpectralConv1d` above."""

    def __init__(self, d_model: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(d_model, d_model, modes)
        self.skip = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, d_model, n_tokens)`."""
        return self.act(self.spectral(x) + self.skip(x))


def apply_fno_layers(h: torch.Tensor, layers) -> torch.Tensor:
    """`h`: `(B, n_tokens, d_model)` -> same shape, after running `layers`
    (a sequence of `FNOLayer`, possibly empty) -- handles the transpose to/
    from `FNOLayer`/`SpectralConv1d`'s channels-first `(B, d_model,
    n_tokens)` convention. A no-op if `layers` is empty, so callers can pass
    an empty `nn.ModuleList()` unconditionally instead of branching on
    whether FNO is enabled."""
    if len(layers) == 0:
        return h
    h = h.transpose(1, 2)
    for layer in layers:
        h = layer(h)
    return h.transpose(1, 2)


class TokenMLPBlock(nn.Module):
    """Pre-norm residual MLP, no attention: `x + MLP(LN(x))`. Public
    (added 2026-09-03), the attention-free half of `ViTBlock`, extracted
    for `ks_latent.models.propagator._FNOMLPDeltaBody`'s `backbone=
    "fno_mlp"` (see `PropagatorConfig`'s docstring): applied identically
    (shared weights) at every token position, so a stack of these adds no
    way to distinguish token position from data content -- combined with
    `FNOLayer`'s exact translation-equivariance, the whole body stays
    translation-equivariant end to end."""

    def __init__(self, d_model: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class ViTBlock(nn.Module):
    """Pre-norm block: `x + Attn(LN(x))`, then `x + MLP(LN(x))`.

    Public (not `_`-prefixed): reused as-is by
    `ks_latent/models/propagator.py`'s `"vit"` backbone, which applies the
    same block design directly to the tokenized latent with no
    pooling/bottleneck step -- see that module for why. `attn_mask` is
    optional and unused by `KSAutoencoderViT` (full attention only); the
    propagator's optional local/banded mask is what it exists for.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class KSAutoencoderViT(nn.Module):
    """See `ViTAutoencoderConfig`'s docstring for the full architecture and
    the comparison against `KSAutoencoderPatched`. `decoder_mode="delta"`
    (the reference project's residual physical-space decoding option) is
    not ported here -- out of scope for this comparison, see
    CLAUDE_CODE_BRIEF.md's addendum for why.

    `cfg.attn_window` (added 2026-08-29, default `None` = full attention):
    restricts both the encoder's and decoder's attention to `+-attn_window`
    neighbours (measured according to `cfg.pos_encoding`) instead of the
    full token sequence. Non-causal (symmetric by distance, not direction).

    `cfg.pos_encoding` (added 2026-08-29, user-directed): `"circular"`
    (default) pairs `CircularPositionalEncoding` with ring-distance
    masking -- correct for these tokens, which sit on KS's genuinely
    periodic domain. `"linear"` swaps in `LinearPositionalEncoding` (fixed,
    non-learned, but *not* periodic) with linear-distance masking --
    deliberately tests a non-periodic assumption against a domain that
    actually is periodic, to isolate whether periodicity specifically
    matters. See `ViTAutoencoderConfig`'s docstring.

    `cfg.pool = "none"` (added 2026-08-29, user-directed): removes the
    global mean/CLS pooling step entirely. `enc_out`/`dec_expand` become
    shared per-token Linear layers instead of a pool + dense Linear /
    dense Linear + reshape, so `z` is a genuine local latent field
    `(n_tokens, c)` (flattened only for interface compatibility) instead of
    a vector every one of whose entries is a dense function of every point
    in physical space. See `ViTAutoencoderConfig`'s docstring for why this
    matters (LATENT_PDE_RESEARCH_NOTES.md's "global support" obstruction).

    `cfg.pool = "local"` (added 2026-08-29, user-directed): the compromise
    between global pooling and `"none"` -- mean-pools each contiguous group
    of `cfg.pool_window` tokens into one site before the same shared
    per-site Linear head, so each `z_k` has a bounded (not global) receptive
    field while `pool_window` gives far more flexibility in matching a
    target `d_latent` than `"none"`'s rigid `d_latent == n_tokens * c`.
    Decoding un-pools by broadcasting each site back to all `pool_window`
    tokens in its group (`repeat_interleave`) before the decoder blocks run.
    `pool="none"` is exactly `pool="local"` with `pool_window=1`. See
    `ViTAutoencoderConfig`'s docstring.

    `cfg.pool = "local_attn"` (added 2026-08-29, user-directed): same
    windowing as `"local"`, but the aggregation is a learned, content-
    adaptive local attention pool (one query shared across sites,
    cross-attending over just its own site's tokens) instead of a fixed
    uniform mean -- isolates whether `"local"`'s reconstruction penalty vs
    global pooling came from averaging away information, or from the
    window size itself. Decoding is identical to `"local"`. See
    `ViTAutoencoderConfig`'s docstring.

    `cfg.pool = "banded"` (added 2026-08-31, user-directed): replaces
    `"mean"`'s global-average + dense `Linear(d_model, d_latent)` with a
    LEARNED, circular-band-MASKED linear map straight from the `n_tokens`
    axis to `d_latent` (`MaskedLinearRect`, the same class the `masked_mlp`
    encoder, Section 17, uses) -- each `z_k` is a learned combination of
    only its `cfg.pool_bandwidth`-nearby tokens, not a dense function of
    every token the way `"mean"` is. Applied per-`d_model`-channel (the
    banded map acts on the token axis; `d_model` is untouched, broadcast
    like an ordinary `nn.Linear`), followed by a shared `Linear(d_model, 1)`
    readout to collapse to a scalar per latent channel. Decoding mirrors
    this with its own independently-learned banded map back out to
    `n_tokens`. Unlike `pool="local"` (fixed uniform mean over a
    contiguous, non-overlapping token window), this is a LEARNED,
    OVERLAPPING-band weighting with no averaging constraint. See
    docs/PHASE2_ARCHITECTURE_EXPERIMENTS.md Section 24.

    `cfg.use_fno` (added 2026-08-30, user-directed): inserts `fno_n_layers`
    `FNOLayer` spectral-conv blocks right after the positional encoding, on
    both the encoder (before any CLS-token concatenation or pooling) and
    the decoder (before the `dec_blocks` ViT attention) -- see
    `ViTAutoencoderConfig.use_fno`'s docstring for why the physical-space
    token axis makes the periodicity assumption behind this actually
    correct, unlike the propagator's `backbone="fno_vit"` case.
    """

    def __init__(self, cfg: ViTAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        n_tokens = cfg.n_tokens
        pos_cls = CircularPositionalEncoding if cfg.pos_encoding == "circular" else LinearPositionalEncoding
        mask_fn = build_ring_local_attention_mask if cfg.pos_encoding == "circular" else build_local_attention_mask

        # token_window (added 2026-08-29, user-directed): overlapping-patch
        # tokenization -- enc_proj's input width becomes the wider window
        # instead of patch_size when set. See ViTAutoencoderConfig's docstring.
        self._token_window = cfg.token_window if cfg.token_window is not None else cfg.patch_size
        self.enc_proj = nn.Linear(self._token_window, cfg.d_model)
        # A CLS aggregation token has no position on the ring; folding it
        # into the circular encoding would make the ring n_tokens+1 long
        # and break the periodicity the encoding exists to express, so it
        # gets its own free embedding instead (only built when pool="cls").
        self.enc_pos = pos_cls(n_tokens, cfg.d_model)
        self.enc_cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.pool == "cls" else None
        # FNO+ViT hybrid encoder/decoder (added 2026-08-30, user-directed --
        # see ViTAutoencoderConfig.use_fno's docstring). Runs on the
        # positionally-encoded token sequence, BEFORE the CLS token (if any)
        # is concatenated and BEFORE the ViT attention blocks -- the CLS
        # token has no ring position (see enc_cls's comment above), so it is
        # excluded from the FFT for the same reason it is excluded from the
        # positional encoding and gets special-cased in the attention mask
        # below. An empty ModuleList when use_fno=False, so encode()/decode()
        # can call apply_fno_layers unconditionally.
        fno_modes = cfg.fno_modes if cfg.fno_modes is not None else n_tokens // 2 + 1
        self.enc_fno = (
            nn.ModuleList([FNOLayer(cfg.d_model, fno_modes) for _ in range(cfg.fno_n_layers)])
            if cfg.use_fno
            else nn.ModuleList()
        )
        self.enc_blocks = nn.ModuleList(
            [ViTBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        # pool in ("none", "local") (added 2026-08-29, user-directed): a
        # shared per-site Linear(d_model -> c) instead of pool + dense
        # Linear(d_model -> d_latent) -- see ViTAutoencoderConfig's
        # docstring. nn.Linear applied to a (B, n_sites, d_model) tensor
        # already broadcasts the same weights over the site dimension.
        if cfg.pool == "banded":
            self.enc_out = nn.Identity()
        elif cfg.pool in ("none", "local", "local_attn"):
            self.enc_out = nn.Linear(cfg.d_model, cfg.local_channels)
        elif cfg.pool == "token_mlp":
            # Bypassed entirely -- see enc_token_ffn/enc_flatten_mlp below,
            # which produce d_latent directly with no separate readout step.
            self.enc_out = None
        elif cfg.readout == "mlp":
            # `ViTAutoencoderConfig.readout` docstring: replaces the plain
            # Linear(d_model, d_latent) readout with a one-hidden-layer MLP.
            # Also used by pool="gated" (readout applies to whatever vector
            # the aggregation step produces, gated or plain mean alike).
            self.enc_out = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_latent), nn.ReLU(), nn.Linear(cfg.d_latent, cfg.d_latent),
            )
        else:
            self.enc_out = nn.Linear(cfg.d_model, cfg.d_latent)
        # pool="gated" (added 2026-09-05, user-directed: "have a ffn that
        # maps ... option 2 is the best to try first" -- soft/gated
        # attention pooling, "A Structured Self-Attentive Sentence
        # Embedding," Lin et al. 2017): ONE small Linear(d_model, 1) scores
        # each token, softmax over the token axis gives a LEARNED weighted
        # combination instead of `pool="mean"`'s fixed uniform average --
        # strictly more expressive at near-zero extra parameter cost (the
        # existing enc_out above is reused unchanged for the post-pool
        # readout). Decode reuses the exact same dense expand mean/cls use
        # (see dec_expand below) -- gated pooling only changes the ENCODE
        # aggregation step, the (B, d_latent) -> (B, n_tokens, d_model)
        # expansion problem it faces on the way back out is identical to
        # mean/cls's.
        self.enc_gate = nn.Linear(cfg.d_model, 1) if cfg.pool == "gated" else None
        # pool="token_mlp" (added 2026-09-05, user-directed: "a ffn that
        # maps (B, n_tokens, d_model) to (B, n_tokens, d_model // 8) ...
        # and then another mlp that takes the flattened vector and maps to
        # d_latent" -- a parameter-BOUNDED alternative to naive
        # flatten-then-dense pooling, whose Linear(n_tokens*d_model,
        # d_latent) would otherwise dominate total AE parameter count
        # (~163K-962K at this project's typical n_tokens=32/d_model=92-136
        # scale, verified by direct computation -- comparable to or
        # larger than the entire Fourier branch in the vit_fourier_hybrid
        # architecture). Two stages: (1) a per-token FFN (shared weights,
        # nn.Linear broadcasts over the token axis same as enc_out) reduces
        # d_model -> compressed_dim = d_model // token_mlp_reduction BEFORE
        # flattening, keeping the flatten step's width bounded regardless
        # of d_model; (2) a 2-layer MLP maps the flattened
        # (n_tokens*compressed_dim)-dim vector to d_latent. Unlike
        # "gated"/"mean"/"cls", every token's full (compressed) content
        # reaches the readout -- nothing is averaged away, only
        # dimensionality-reduced per token first.
        if cfg.pool in ("token_mlp", "local_token_mlp"):
            compressed_dim = max(1, cfg.d_model // cfg.token_mlp_reduction)
            self.enc_token_ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, compressed_dim),
            )
        else:
            self.enc_token_ffn = None
        if cfg.pool == "token_mlp":
            compressed_dim = max(1, cfg.d_model // cfg.token_mlp_reduction)
            self.enc_flatten_mlp = nn.Sequential(
                nn.Linear(n_tokens * compressed_dim, cfg.token_mlp_hidden), nn.ReLU(),
                nn.Linear(cfg.token_mlp_hidden, cfg.d_latent),
            )
        else:
            self.enc_flatten_mlp = None
        # pool="local_token_mlp" (added 2026-09-06, Section 100, user-directed:
        # "Let's include a token mlp in that encoder that respects the local
        # attention window" -- see ViTAutoencoderConfig's docstring): the same
        # per-token FFN above feeds a LEARNED, circular-band-masked map straight
        # from the n_tokens axis to d_latent (MaskedLinearRect, window=
        # attn_window in n_tokens-ring units -- the SAME window restricting this
        # encoder's own attention) instead of token_mlp's dense flatten+MLP.
        # Structurally mirrors pool="banded" (see enc_pool/enc_readout above)
        # with the compressed per-token representation substituted for the raw
        # d_model one.
        if cfg.pool == "local_token_mlp":
            compressed_dim = max(1, cfg.d_model // cfg.token_mlp_reduction)
            self.enc_local_pool = MaskedLinearRect(n_tokens, cfg.d_latent, cfg.attn_window, n_tokens)
            self.enc_local_readout = nn.Linear(compressed_dim, 1)
        else:
            self.enc_local_pool = None
            self.enc_local_readout = None
        # pool="banded" (added 2026-08-31, user-directed): see
        # ViTAutoencoderConfig's docstring and this class's own. The
        # banded map operates on the TOKEN axis (n_tokens -> d_latent),
        # applied per-d_model-channel via MaskedLinearRect's ordinary
        # nn.Linear broadcasting; enc_readout then collapses the d_model
        # axis to a scalar per latent channel.
        if cfg.pool == "banded":
            self.enc_pool = MaskedLinearRect(n_tokens, cfg.d_latent, cfg.pool_bandwidth, cfg.d_latent)
            self.enc_readout = nn.Linear(cfg.d_model, 1)
        else:
            self.enc_pool = None
            self.enc_readout = None
        # pool="local_attn" (added 2026-08-29, user-directed): one learned query,
        # shared across all sites, cross-attends over just its own site's tokens --
        # see ViTAutoencoderConfig's docstring. Degenerate (a no-op single-key
        # softmax) at pool_window=1, so skip allocating it there.
        if cfg.pool == "local_attn" and cfg.pool_window > 1:
            self.site_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            self.pool_attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
            )
        else:
            self.site_query = None
            self.pool_attn = None

        enc_mask = mask_fn(n_tokens, cfg.attn_window)
        if enc_mask is not None and self.enc_cls is not None:
            # The CLS token has no ring position, so it is left fully
            # visible (and visible to everyone) -- an all-zero border
            # around the restricted (n_tokens, n_tokens) block.
            padded = torch.zeros(n_tokens + 1, n_tokens + 1)
            padded[1:, 1:] = enc_mask
            enc_mask = padded
        self.register_buffer("enc_attn_mask", enc_mask, persistent=False)

        # pool in ("none", "local"): shared per-site Linear(c -> d_model),
        # mirroring enc_out, instead of one dense Linear(d_latent ->
        # n_tokens*d_model) that mixes every latent coordinate into every
        # token.
        #
        # `dec_pool_mode` (added 2026-09-05, see ViTAutoencoderConfig.dec_pool's
        # docstring): the decoder's OWN effective pool mode, which can differ
        # from the encoder's `cfg.pool` -- every module built in this block is
        # keyed off `dec_pool_mode`, not `cfg.pool` directly, so an asymmetric
        # (e.g. encoder="local", decoder="mean") config gets the right shapes.
        dec_pool_mode = cfg.dec_pool_mode
        if dec_pool_mode == "banded":
            self.dec_expand = nn.Identity()
            # Independently-learned banded map back out (d_latent -> n_tokens),
            # NOT simply enc_pool's transpose -- see this class's docstring.
            self.dec_expand_readout = nn.Linear(1, cfg.d_model)
            self.dec_pool = MaskedLinearRect(cfg.d_latent, n_tokens, cfg.pool_bandwidth, cfg.d_latent)
            self.dec_flatten_mlp = None
            self.dec_token_ffn = None
        elif dec_pool_mode == "token_mlp":
            # Mirrors enc_token_ffn/enc_flatten_mlp in reverse (added
            # 2026-09-05, see ViTAutoencoderConfig.token_mlp_reduction's
            # docstring): a 2-layer MLP expands d_latent -> the flattened
            # (n_tokens*compressed_dim)-dim space, then a per-token FFN
            # (shared weights) expands compressed_dim back to d_model.
            # Bounded parameter count for the same reason the encode side
            # is -- the expensive step is never a dense
            # Linear(d_latent, n_tokens*d_model).
            self.dec_expand = None
            self.dec_expand_readout = None
            self.dec_pool = None
            compressed_dim = max(1, cfg.d_model // cfg.token_mlp_reduction)
            self.dec_flatten_mlp = nn.Sequential(
                nn.Linear(cfg.d_latent, cfg.token_mlp_hidden), nn.ReLU(),
                nn.Linear(cfg.token_mlp_hidden, n_tokens * compressed_dim),
            )
            self.dec_token_ffn = nn.Sequential(
                nn.Linear(compressed_dim, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model),
            )
        elif dec_pool_mode == "local_token_mlp":
            # Mirrors enc_local_pool/enc_local_readout in reverse (added
            # 2026-09-06, Section 100, see ViTAutoencoderConfig's docstring):
            # dec_local_expand_readout broadcasts each latent scalar to a
            # compressed_dim-wide vector (Linear(1, compressed_dim), same
            # trick pool="banded"'s dec_expand_readout uses), dec_local_pool
            # is an INDEPENDENTLY-learned banded map back out to n_tokens
            # (d_latent -> n_tokens, same window), then dec_token_ffn (shared
            # with token_mlp's own mirror) expands compressed_dim -> d_model.
            self.dec_expand = None
            self.dec_expand_readout = None
            self.dec_pool = None
            compressed_dim = max(1, cfg.d_model // cfg.token_mlp_reduction)
            self.dec_local_expand_readout = nn.Linear(1, compressed_dim)
            self.dec_local_pool = MaskedLinearRect(cfg.d_latent, n_tokens, cfg.attn_window, n_tokens)
            self.dec_token_ffn = nn.Sequential(
                nn.Linear(compressed_dim, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.dec_flatten_mlp = None
        else:
            self.dec_expand = (
                nn.Linear(cfg.local_channels, cfg.d_model)
                if dec_pool_mode in ("none", "local", "local_attn")
                else nn.Linear(cfg.d_latent, n_tokens * cfg.d_model)
            )
            self.dec_expand_readout = None
            self.dec_pool = None
            self.dec_flatten_mlp = None
            self.dec_token_ffn = None
        if dec_pool_mode != "local_token_mlp":
            self.dec_local_expand_readout = None
            self.dec_local_pool = None
        self.dec_pos = pos_cls(n_tokens, cfg.d_model)
        self.dec_fno = (
            nn.ModuleList([FNOLayer(cfg.d_model, fno_modes) for _ in range(cfg.fno_n_layers)])
            if cfg.use_fno
            else nn.ModuleList()
        )
        self.dec_blocks = nn.ModuleList(
            [ViTBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.dec_out = nn.Linear(cfg.d_model, cfg.patch_size)
        # No CLS token on the decoder side, so no padding needed here.
        self.register_buffer("dec_attn_mask", mask_fn(n_tokens, cfg.attn_window), persistent=False)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """`u`: `(B, NX)` -> `z`: `(B, d_latent)`."""
        cfg = self.cfg
        B = u.shape[0]
        tokens = circular_overlap_tokenize(u, cfg.patch_size, self._token_window, cfg.n_tokens)
        h = self.enc_pos(self.enc_proj(tokens))
        h = apply_fno_layers(h, self.enc_fno)
        if self.enc_cls is not None:
            h = torch.cat([self.enc_cls.expand(B, -1, -1), h], dim=1)
        for block in self.enc_blocks:
            h = block(h, attn_mask=self.enc_attn_mask)
        h = self.enc_norm(h)
        if cfg.pool == "token_mlp":
            compressed = self.enc_token_ffn(h)  # (B, n_tokens, compressed_dim), shared per-token weights
            flat = compressed.reshape(B, -1)
            return self.enc_flatten_mlp(flat)
        if cfg.pool == "local_token_mlp":
            compressed = self.enc_token_ffn(h)  # (B, n_tokens, compressed_dim), shared per-token weights
            # (B, n_tokens, compressed_dim) -> (B, compressed_dim, n_tokens) -> MaskedLinearRect
            # (broadcasts over compressed_dim) -> (B, compressed_dim, d_latent) -> (B, d_latent,
            # compressed_dim) -> shared Linear(compressed_dim, 1) -> (B, d_latent).
            pooled = self.enc_local_pool(compressed.transpose(1, 2)).transpose(1, 2)
            return self.enc_local_readout(pooled).squeeze(-1)
        if cfg.pool == "gated":
            # Soft/gated attention pooling ("A Structured Self-Attentive
            # Sentence Embedding," Lin et al. 2017): a LEARNED weighted
            # combination of tokens instead of pool="mean"'s fixed uniform
            # average -- see enc_gate's docstring above.
            gate_logits = self.enc_gate(h)  # (B, n_tokens, 1)
            weights = torch.softmax(gate_logits, dim=1)
            pooled = (weights * h).sum(dim=1)  # (B, d_model)
            return self.enc_out(pooled)
        if cfg.pool == "local_attn" and cfg.pool_window > 1:
            # One learned query, shared across all sites (translation-equivariant),
            # does cross-attention over just its own site's pool_window tokens --
            # (B, n_tokens, d_model) -> (B*n_sites, pool_window, d_model) keys/values,
            # (B*n_sites, 1, d_model) query -> (B, n_sites, d_model).
            kv = h.reshape(B * cfg.n_sites, cfg.pool_window, cfg.d_model)
            q = self.site_query.expand(B * cfg.n_sites, 1, cfg.d_model)
            pooled_sites, _ = self.pool_attn(q, kv, kv, need_weights=False)
            h = pooled_sites.reshape(B, cfg.n_sites, cfg.d_model)
            return self.enc_out(h).reshape(B, cfg.d_latent)
        if cfg.pool in ("none", "local", "local_attn"):
            if cfg.pool_window > 1:
                # (B, n_tokens, d_model) -> (B, n_sites, pool_window, d_model) -> mean over
                # each contiguous window -> (B, n_sites, d_model).
                h = h.view(B, cfg.n_sites, cfg.pool_window, cfg.d_model).mean(dim=2)
            return self.enc_out(h).reshape(B, cfg.d_latent)  # (B, n_sites, c) -> (B, d_latent)
        if cfg.pool == "banded":
            # (B, n_tokens, d_model) -> (B, d_model, n_tokens) -> MaskedLinearRect
            # (broadcasts over d_model) -> (B, d_model, d_latent) -> (B, d_latent, d_model)
            # -> shared Linear(d_model, 1) -> (B, d_latent).
            pooled = self.enc_pool(h.transpose(1, 2)).transpose(1, 2)
            return self.enc_readout(pooled).squeeze(-1)
        pooled = h[:, 0] if self.enc_cls is not None else h.mean(dim=1)
        return self.enc_out(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: `(B, d_latent)` -> `u_hat`: `(B, NX)`."""
        cfg = self.cfg
        B = z.shape[0]
        dec_pool_mode = cfg.dec_pool_mode
        if dec_pool_mode == "token_mlp":
            flat = self.dec_flatten_mlp(z)  # (B, n_tokens*compressed_dim)
            compressed = flat.view(B, cfg.n_tokens, -1)
            h = self.dec_token_ffn(compressed)  # (B, n_tokens, d_model), shared per-token weights
        elif dec_pool_mode == "local_token_mlp":
            # (B, d_latent) -> (B, d_latent, 1) -> Linear(1, compressed_dim) -> (B, d_latent,
            # compressed_dim) -> (B, compressed_dim, d_latent) -> MaskedLinearRect ->
            # (B, compressed_dim, n_tokens) -> (B, n_tokens, compressed_dim) -> dec_token_ffn ->
            # (B, n_tokens, d_model).
            expanded = self.dec_local_expand_readout(z.unsqueeze(-1))
            compressed = self.dec_local_pool(expanded.transpose(1, 2)).transpose(1, 2)
            h = self.dec_token_ffn(compressed)
        elif dec_pool_mode in ("none", "local", "local_attn"):
            h = self.dec_expand(z.view(B, cfg.n_sites, cfg.local_channels))  # (B, n_sites, d_model)
            if cfg.pool_window > 1:
                # Un-pool: broadcast each site's vector to all pool_window tokens in its
                # group (inverse of encode's windowed mean) -> (B, n_tokens, d_model).
                h = h.repeat_interleave(cfg.pool_window, dim=1)
        elif dec_pool_mode == "banded":
            # (B, d_latent) -> (B, d_latent, 1) -> Linear(1, d_model) -> (B, d_latent, d_model)
            # -> (B, d_model, d_latent) -> MaskedLinearRect -> (B, d_model, n_tokens)
            # -> (B, n_tokens, d_model).
            expanded = self.dec_expand_readout(z.unsqueeze(-1))
            h = self.dec_pool(expanded.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.dec_expand(z).view(B, cfg.n_tokens, cfg.d_model)
        h = self.dec_pos(h)
        h = apply_fno_layers(h, self.dec_fno)
        for block in self.dec_blocks:
            h = block(h, attn_mask=self.dec_attn_mask)
        h = self.dec_norm(h)
        patches = self.dec_out(h)  # (B, n_tokens, patch_size)
        return patches.reshape(B, cfg.NX)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(u)
        u_hat = self.decode(z)
        return u_hat, z
