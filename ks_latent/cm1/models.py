"""Column local-field autoencoder and local stencil propagator for CM1.

Autoencoder. A column of `C = V*Z` channels at each of `NX` grid cells is
mapped to a latent field `(P, c)` with `P = NX / patch` sites:
  encoder  Conv1d(C -> H, kernel = stride = patch)   (exact patchify)
           n_mix x [Conv1d(H -> H, 2r+1) + GELU], residual
           Conv1d(H -> c, 1, bias=False)
  decoder  the mirror, ConvTranspose1d(H -> C, kernel = stride = patch)
Locality is horizontal only: each site sees its own patch plus `n_mix * r`
sites either side (encoder reach), and the decoder adds the same reach.
Shifting the input by a multiple of `patch` shifts the latent by whole
sites, exactly, at sites farther than the reach from the domain edges
(tested). Padding is never circular: the lateral boundaries are open.

Encoder biases. The final 1x1 layer has no bias, for the reason found in
the 1D KS local field (`autoencoder_local_field.py`): a free offset per
channel drifts during training and nothing in the loss resists it.

Per-variable ablation (`groups`). With `groups = V` every layer is a grouped convolution, so
each variable's Z levels are encoded into their own c/V latent channels and decoded from them
alone: V independent autoencoders side by side. Variables then interact only through the
propagator. `groups = 1` (default) is the joint column encoder. Requires C, hidden and
latent_channels divisible by `groups`; channels are variable-major, so the groups are variables.

Vertical front end (`vertical_channels` > 0, D14b): before the patch embedding, a convolution along
height only (kernel `vertical_kernel` x 1) maps the V variables at each level to `vertical_channels`
features, so the encoder starts from local vertical structure rather than one flat C-vector per
cell. Width 1 in x: locality and shift equivariance in x are unchanged.

Non-negative outputs. The condensate channels (sqrt of mixing ratio) pass
through a sharp softplus, `log(1 + exp(beta x)) / beta`, so decoded
condensate is >= 0 and reaches ~0 (5e-6 at x = -1 for beta = 10) without
killing the gradient as a ReLU would.

Propagator. `z_{n+1} = z_n + f(z_n)` on the latent sites, `f` a stack of
`layers` convolutions of radius `r` (reach `layers * r` sites per step),
with the last layer zero-initialized so the map is exactly the identity at
initialization. Markovian: the PDE is first order in time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ColumnAEConfig:
    n_channels: int = 7 * 62  # C = V*Z
    n_levels: int = 62
    patch: int = 8  # cells per site (8 x 250 m = 2 km)
    hidden: int = 256
    latent_channels: int = 16
    n_mix: int = 2
    mix_radius: int = 1
    padding_mode: str = "replicate"
    positive_beta: float = 10.0
    groups: int = 1  # 1: joint column encoder; V: one encoder/decoder per variable (D7 ablation)
    vertical_channels: int = 0  # > 0: per-cell vertical convolution front end with this many channels (D14b)
    vertical_kernel: int = 5
    env_raw_dim: int = 0        # > 0: environment features per member (env.py, 3*Z) given to the encoder (D11)
    env_dim: int = 16           # their learned projection, broadcast along x as extra input channels
    kind: str = "conv"          # "conv": ColumnFieldAE; "pca": PCAColumnAE (fixed linear baseline, I-PCA); "vit": ViTColumnAE
    prefix_m: int = 0           # > 0: latent channels >= prefix_m are a constant (`prefix_fill`): a pruned latent (5s)
    # kind = "vit" (8a, 2026-11-13): site tokens (one 2 km patch of all columns each), pre-RMSNorm transformer blocks with
    # SwiGLU MLPs (hidden = vit_mlp_ratio x vit_dim) and a learned relative-position bias; vit_layers blocks in the encoder
    # and as many in the decoder. vit_radius > 0: attention within vit_radius sites (local field, reach layers x radius);
    # 0: attention over every site of the window (global), with vit_global_reach as the nominal edge margin.
    vit_dim: int = 1216
    vit_layers: int = 4
    vit_heads: int = 19
    vit_mlp_ratio: float = 4.0
    vit_radius: int = 2
    vit_global_reach: int = 4

    @property
    def reach_sites(self) -> int:
        if self.kind == "pca":
            return 0
        if self.kind == "vit":
            return self.vit_layers * self.vit_radius if self.vit_radius > 0 else self.vit_global_reach
        return self.n_mix * self.mix_radius


class _Mixer(nn.Module):
    def __init__(self, H: int, r: int, n: int, padding_mode: str, groups: int = 1):
        super().__init__()
        if padding_mode == "circular":
            raise ValueError("circular padding is wrong for CM1 (open lateral boundaries)")
        self.convs = nn.ModuleList(nn.Conv1d(H, H, 2 * r + 1, padding=r, padding_mode=padding_mode, groups=groups)
                                   for _ in range(n))

    def forward(self, h):
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        return h


class ColumnFieldAE(nn.Module):
    """`u`: `(B, C, NX)` normalized columns -> `z`: `(B, P, c)` -> `u_hat`: `(B, C, NX)`."""

    def __init__(self, cfg: ColumnAEConfig, positive_mask: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        C, H, c, p, g = cfg.n_channels, cfg.hidden, cfg.latent_channels, cfg.patch, cfg.groups
        if C % g or H % g or c % g:
            raise ValueError(f"groups={g} must divide n_channels={C}, hidden={H} and latent_channels={c}")
        self.env_proj = nn.Linear(cfg.env_raw_dim, cfg.env_dim) if cfg.env_raw_dim else None
        E = cfg.env_dim if cfg.env_raw_dim else 0
        self.vert = None
        if cfg.vertical_channels:
            if g != 1:
                raise ValueError("vertical_channels and groups > 1 are not combined")
            V = C // cfg.n_levels
            if V * cfg.n_levels != C:
                raise ValueError("n_channels must be variables x n_levels for the vertical front end")
            k = cfg.vertical_kernel
            self.vert = nn.Conv2d(V, cfg.vertical_channels, (k, 1), padding=(k // 2, 0))   # along z only
            self.enc_in = nn.Conv1d(cfg.vertical_channels * cfg.n_levels + E, H, p, stride=p)
        else:
            if E and g != 1:
                raise ValueError("environment conditioning and groups > 1 are not combined")
            self.enc_in = nn.Conv1d(C + E, H, p, stride=p, groups=g)
        self.enc_mix = _Mixer(H, cfg.mix_radius, cfg.n_mix, cfg.padding_mode, g)
        self.enc_out = nn.Conv1d(H, c, 1, bias=False, groups=g)
        self.dec_in = nn.Conv1d(c, H, 1, groups=g)
        self.dec_mix = _Mixer(H, cfg.mix_radius, cfg.n_mix, cfg.padding_mode, g)
        self.dec_out = nn.ConvTranspose1d(H, C, p, stride=p, groups=g)
        pm = torch.zeros(C, dtype=torch.bool) if positive_mask is None else positive_mask.bool()
        if pm.shape != (C,):
            raise ValueError(f"positive_mask shape {tuple(pm.shape)} != ({C},)")
        self.register_buffer("positive_mask", pm)
        if cfg.prefix_m:
            self.register_buffer("prefix_fill", torch.zeros(c))

    def set_prefix(self, m: int, fill: torch.Tensor) -> None:
        """Prune the latent to its first `m` channels; the others become the constant `fill` (c,)."""
        self.cfg.prefix_m = m
        self.register_buffer("prefix_fill", fill.detach().clone().to(self.positive_mask.device, torch.float32))

    def encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        z = self._encode(u, env)
        if self.cfg.prefix_m:
            m = self.cfg.prefix_m
            z = torch.cat([z[..., :m], self.prefix_fill[m:].expand(*z.shape[:-1], -1)], -1)
        return z

    def _encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        """`env` (B, env_raw_dim): required when the model is environment-conditioned (D11)."""
        if (env is None) != (self.env_proj is None):
            raise ValueError("pass env exactly when the autoencoder is environment-conditioned")
        if u.shape[-1] % self.cfg.patch:
            raise ValueError(f"width {u.shape[-1]} is not a multiple of patch {self.cfg.patch}")
        if self.vert is not None:
            B, C, NX = u.shape
            Zl = self.cfg.n_levels
            u = F.gelu(self.vert(u.reshape(B, C // Zl, Zl, NX))).reshape(B, -1, NX)
        if env is not None:
            e = self.env_proj(env.to(u.dtype))[:, :, None].expand(-1, -1, u.shape[-1])
            u = torch.cat([u, e], dim=1)
        h = F.gelu(self.enc_in(u))
        h = self.enc_mix(h)
        return self.enc_out(h).transpose(1, 2)  # (B, P, c)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec_in(z.transpose(1, 2)))
        h = self.dec_mix(h)
        out = self.dec_out(h)
        if self.positive_mask.any():
            pos = F.softplus(out, beta=self.cfg.positive_beta)
            out = torch.where(self.positive_mask[None, :, None], pos, out)
        return out

    def forward(self, u, env: torch.Tensor | None = None):
        z = self.encode(u, env)
        return self.decode(z), z


def _patchify(u: torch.Tensor, p: int) -> torch.Tensor:
    """`(B, C, NX)` -> `(B, P, C*p)`: one vector per site, channel-major (all `p` cells of channel 0 first)."""
    B, C, NX = u.shape
    if NX % p:
        raise ValueError(f"width {NX} is not a multiple of patch {p}")
    return u.reshape(B, C, NX // p, p).permute(0, 2, 1, 3).reshape(B, NX // p, C * p)


def _unpatchify(x: torch.Tensor, C: int, p: int) -> torch.Tensor:
    B, P, _ = x.shape
    return x.reshape(B, P, C, p).permute(0, 2, 1, 3).reshape(B, C, P * p)


class PCAColumnAE(nn.Module):
    """Fixed linear local latent (plan item I-PCA): each 2 km site's `C x patch` column block is projected
    on the leading `latent_channels` principal components of training site blocks.

        z = (x - mean) V^T / s,      x_hat = z s V + mean   (then condensate clamped at 0)

    `V` (c, C*patch) orthonormal rows, `s` one global scale making the mean variance of the c coordinates
    1, so the propagator sees O(1) inputs. A single scale (not per-component whitening) keeps squared
    latent distance proportional to squared error on the reconstructed block, as in the autoencoder's
    training. Same interface as `ColumnFieldAE`; no trainable parameters (Stage C then trains only the
    propagator). Strictly local: a site's latent depends on its own block only (reach 0). The clamp of
    condensate at 0 is the only nonlinearity, matching the autoencoder's positivity guarantee.
    """

    def __init__(self, cfg: ColumnAEConfig, positive_mask: torch.Tensor | None = None):
        super().__init__()
        if cfg.kind != "pca":
            raise ValueError("PCAColumnAE needs cfg.kind == 'pca'")
        if cfg.env_raw_dim:
            raise ValueError("the PCA baseline is not environment-conditioned")
        self.cfg = cfg
        D = cfg.n_channels * cfg.patch
        self.register_buffer("mean", torch.zeros(D))
        self.register_buffer("components", torch.zeros(cfg.latent_channels, D))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("explained", torch.zeros(cfg.latent_channels))   # variance fraction per component
        pm = torch.zeros(cfg.n_channels, dtype=torch.bool) if positive_mask is None else positive_mask.bool()
        self.register_buffer("positive_mask", pm)
        self._device_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)   # next(parameters()) -> device

    @torch.no_grad()
    def fit(self, blocks_iter, active_weight: float = 0.5) -> dict:
        """Fit from an iterable of `(x, active)`: `x` (N, C*patch) site blocks, `active` (N,) bool (block
        contains storm columns). The second moment is `(1 - w) E_all + w E_active`, mirroring the 50%
        storm-centred window sampling the autoencoder is trained with (`active_frac` 0.5)."""
        D = self.mean.numel()
        dev = self.mean.device
        S = {k: torch.zeros(D, dtype=torch.float64, device=dev) for k in ("all", "act")}
        Q = {k: torch.zeros(D, D, dtype=torch.float64, device=dev) for k in ("all", "act")}
        n = {"all": 0, "act": 0}
        for x, act in blocks_iter:
            x = x.to(dev, self.mean.dtype)
            for k, xs in (("all", x), ("act", x[act.to(dev)])):
                if len(xs):
                    S[k] += xs.sum(0).double(); Q[k] += (xs.T @ xs).double(); n[k] += len(xs)
        if not n["all"]:
            raise ValueError("no site blocks to fit the PCA on")
        w = active_weight if n["act"] else 0.0
        mean = (1 - w) * S["all"] / n["all"] + (w * S["act"] / n["act"] if w else 0)
        second = (1 - w) * Q["all"] / n["all"] + (w * Q["act"] / n["act"] if w else 0)
        cov = second - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        ev, vec = torch.linalg.eigh(cov)
        ev, vec = ev.flip(0), vec.flip(1)
        c = self.cfg.latent_channels
        if ev[c - 1] <= 0:
            raise ValueError(f"covariance has only {int((ev > 0).sum())} positive eigenvalues (< {c})")
        self.mean.copy_(mean)
        self.components.copy_(vec[:, :c].T)
        self.scale.copy_(ev[:c].mean().sqrt())
        self.explained.copy_(ev[:c] / ev.clamp(min=0).sum())
        return {"n_blocks": n["all"], "n_active_blocks": n["act"], "explained_total": float(self.explained.sum()),
                "eigenvalues": ev[: 2 * c].tolist()}

    def encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if env is not None:
            raise ValueError("the PCA baseline is not environment-conditioned")
        x = _patchify(u, self.cfg.patch)
        return (x - self.mean) @ self.components.T / self.scale                     # (B, P, c)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = (z * self.scale) @ self.components + self.mean
        out = _unpatchify(x, self.cfg.n_channels, self.cfg.patch)
        if self.positive_mask.any():
            out = torch.where(self.positive_mask[None, :, None], out.clamp(min=0), out)
        return out

    def forward(self, u, env: torch.Tensor | None = None):
        z = self.encode(u, env)
        return self.decode(z), z


class SwiGLU(nn.Module):
    """x -> W3 (silu(W1 x) * (W2 x)), hidden width `h` (the up-projection)."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class _ViTBlock(nn.Module):
    """Pre-RMSNorm block over site tokens: multi-head attention with a learned relative-position bias per head (radius > 0:
    one entry per offset within the window, -inf outside; radius 0: global, signed log-binned offsets), then SwiGLU.
    No per-site parameters, so the block is translation-equivariant away from the edges and runs on any width."""

    def __init__(self, d: int, heads: int, mlp_hidden: int, radius: int):
        super().__init__()
        if d % heads:
            raise ValueError(f"vit_dim {d} not divisible by {heads} heads")
        self.h, self.r = heads, radius
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * radius + 1 if radius > 0 else 2 * len(_REL_EDGES) - 1))
        self.mlp = SwiGLU(d, mlp_hidden)

    def bias(self, P: int, device) -> torch.Tensor:
        i = torch.arange(P, device=device)
        off = i[None, :] - i[:, None]                                    # key - query
        if self.r > 0:
            b = self.rel[:, off.clamp(-self.r, self.r) + self.r]
            return b.masked_fill((off.abs() > self.r)[None], float("-inf"))
        return self.rel[:, _rel_bin(off)]

    def forward(self, x):                                                # (B, P, d)
        B, P, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, P, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias(P, x.device)[None].to(q.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(B, P, d))
        return x + self.mlp(self.n2(x))


class ViTColumnAE(nn.Module):
    """8a: `u` (B, C, NX) -> site tokens (B, P, C*patch) -> Linear -> vit_layers blocks -> RMSNorm -> Linear (no bias) -> `z`
    (B, P, c); the decoder mirrors it (Linear c -> d, blocks, RMSNorm, Linear d -> C*patch, unpatchify, positivity).
    Same interface as ColumnFieldAE (encode / decode / forward / set_prefix, prefix_m), so propagators, rollouts and DA
    run unchanged. Environment conditioning is not supported."""

    def __init__(self, cfg: ColumnAEConfig, positive_mask: torch.Tensor | None = None):
        super().__init__()
        if cfg.env_raw_dim or cfg.groups != 1 or cfg.vertical_channels:
            raise ValueError("the ViT autoencoder does not support env conditioning, groups or the vertical front end")
        self.cfg = cfg
        C, p, c, d = cfg.n_channels, cfg.patch, cfg.latent_channels, cfg.vit_dim
        hm = int(round(cfg.vit_mlp_ratio * d))
        mk = lambda: nn.ModuleList(_ViTBlock(d, cfg.vit_heads, hm, cfg.vit_radius) for _ in range(cfg.vit_layers))
        self.enc_in, self.enc_blocks, self.enc_norm = nn.Linear(C * p, d), mk(), nn.RMSNorm(d)
        self.enc_out = nn.Linear(d, c, bias=False)
        self.dec_in, self.dec_blocks, self.dec_norm = nn.Linear(c, d), mk(), nn.RMSNorm(d)
        self.dec_out = nn.Linear(d, C * p)
        self.apply(self._init)
        for blk in list(self.enc_blocks) + list(self.dec_blocks):     # residual branches scaled by depth
            for w in (blk.proj.weight, blk.mlp.w3.weight):
                nn.init.normal_(w, std=0.02 / (2 * cfg.vit_layers) ** 0.5)
        pm = torch.zeros(C, dtype=torch.bool) if positive_mask is None else positive_mask.bool()
        if pm.shape != (C,):
            raise ValueError(f"positive_mask shape {tuple(pm.shape)} != ({C},)")
        self.register_buffer("positive_mask", pm)
        if cfg.prefix_m:
            self.register_buffer("prefix_fill", torch.zeros(c))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def set_prefix(self, m: int, fill: torch.Tensor) -> None:
        self.cfg.prefix_m = m
        self.register_buffer("prefix_fill", fill.detach().clone().to(self.positive_mask.device, torch.float32))

    def encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if env is not None:
            raise ValueError("the ViT autoencoder is not environment-conditioned")
        h = self.enc_in(_patchify(u, self.cfg.patch))
        for b in self.enc_blocks:
            h = b(h)
        z = self.enc_out(self.enc_norm(h))
        if self.cfg.prefix_m:
            m = self.cfg.prefix_m
            z = torch.cat([z[..., :m], self.prefix_fill[m:].expand(*z.shape[:-1], -1)], -1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        for b in self.dec_blocks:
            h = b(h)
        out = _unpatchify(self.dec_out(self.dec_norm(h)), self.cfg.n_channels, self.cfg.patch)
        if self.positive_mask.any():
            pos = F.softplus(out, beta=self.cfg.positive_beta)
            out = torch.where(self.positive_mask[None, :, None], pos, out)
        return out

    def forward(self, u, env: torch.Tensor | None = None):
        z = self.encode(u, env)
        return self.decode(z), z


def build_autoencoder(cfg: ColumnAEConfig, positive_mask: torch.Tensor | None = None) -> nn.Module:
    """The autoencoder named by `cfg.kind` (every loader goes through here)."""
    if cfg.kind == "conv":
        return ColumnFieldAE(cfg, positive_mask)
    if cfg.kind == "vit":
        return ViTColumnAE(cfg, positive_mask)
    if cfg.kind == "pca":
        if cfg.prefix_m:
            raise ValueError("prefix_m is not supported for the PCA autoencoder")
        return PCAColumnAE(cfg, positive_mask)
    raise ValueError(f"unknown autoencoder kind {cfg.kind!r}")


@dataclass
class StencilPropConfig:
    latent_channels: int = 16
    hidden: int = 256
    layers: int = 3
    radius: int = 1
    padding_mode: str = "replicate"
    n_hist: int = 1          # frames of history the step sees (D13); 1 = Markovian
    noise_channels: int = 0  # random input channels per site (D14c stochastic propagator); 0 = deterministic
    body: str = "conv"       # "conv" (stencil convolutions), "attention" (M1a), "law" (FPc: structured latent PDE) or "mlp"
                             # (8a: a dense SwiGLU MLP per site on the flattened neighbourhood of layers x radius sites)
    mlp_ratio: float = 4.0   # body "mlp": SwiGLU hidden = mlp_ratio x hidden
    attn_swiglu: bool = False  # body "attention": SwiGLU FFN (hidden = mlp_ratio x hidden) instead of the 2x GELU MLP
    heads: int = 4           # attention heads (body = "attention")
    # body = "law" (FPc): dz/dt = A0 z + A1 d1z + A2 d2z + b + Q [z_i z_j, z_i d1z_j] + R(z, d1z, d2z), integrated
    # over one frame; see LatentLaw. Penalties are applied by `StencilPropagator.penalty()` in Stages B / C.
    law_quadratic: bool = True        # include the quadratic library
    law_residual_hidden: int = 64     # pointwise MLP residual width (0: none)
    law_integrator: str = "heun"      # "euler" or "heun"
    law_substeps: int = 1
    law_l1: float = 0.0               # L1 weight on A0 (off-diagonal), A1, A2 and Q (sparsity and sparse coupling)
    law_res_weight: float = 0.0       # weight of the residual's share of the tendency
    law_global: bool = False          # + G <z>: each channel's mean over all sites the step sees (hybrid local/global)
    law_cap: float = 0.0              # > 0: dz/dt = cap tanh(F / cap), a saturating bound so rollouts stay finite
    # FPf (2026-11-09): constrain channel 0's equation to a short closed law in z0 alone,
    #   dz0/dt = b0 + a0 z0 + a1 d1z0 + nu d2z0 + q z0^2 + a2 z0 d1z0     (no other channel, no residual),
    # so training must find a leading coordinate that obeys it; channels 1.. keep the full law. "" = off.
    law_lead: str = ""                # "" or "z0"
    law_lead_nu_pos: bool = False     # nu = softplus(.) >= 0 in channel 0's law (no anti-diffusion)
    # FPg (2026-11-09): a separate closed PDE g on the first `law_head_n` channels (the most important, by the nested
    # dropout's ordering), dz_i/dt = g_i(z_{0..n-1}, d1z, d2z) for i < n (same library as the law, no residual), trained
    # with a representation loss  law_head_weight * |F_{<n}(z) - g(z_{<n})|^2 / |F_{<n}|^2  against the dense law's
    # tendency on those channels. The dense law F is untouched structurally (stays accurate); the loss pulls F's
    # leading-channel tendency toward closure in the leading channels. 0 = off.
    law_head_n: int = 0
    law_head_weight: float = 0.0
    prefix_m: int = 0                 # > 0: channels >= prefix_m are a constant (`prefix_fill`) in input and output (5s)
    # 7a (2026-11-13): global coupling of the leading channels (body "conv" only). The first `global_k` channels of every
    # site (the most important ones under nested dropout) feed a domain-wide module whose `global_feat` per-site outputs
    # are extra inputs of the local stencil. "attn": full self-attention over all sites with a signed, log-binned relative
    # position bias (no per-site parameters, any domain width); "mean": an MLP of the domain mean broadcast to every
    # site. The stencil's weights on these inputs start at zero, so at init the model is the local one. "" = off.
    global_kind: str = ""
    global_k: int = 8
    global_hidden: int = 64
    global_layers: int = 2
    global_heads: int = 4
    global_feat: int = 16

    @property
    def reach_sites(self) -> int:
        if self.body == "law":
            return (2 if self.law_integrator == "heun" else 1) * self.law_substeps
        return self.layers * self.radius


class _WindowAttention(nn.Module):
    """Pre-norm block: multi-head self-attention over sites within `radius` of each other (open domain: the
    window is truncated at the edges), with a learned relative-position bias per head and offset, shared by
    every site (so the block is translation-equivariant away from the edges), then a 2x MLP."""

    def __init__(self, H: int, heads: int, radius: int, swiglu_ratio: float = 0.0):
        super().__init__()
        if H % heads:
            raise ValueError(f"hidden {H} not divisible by {heads} heads")
        self.h, self.r = heads, radius
        self.n1, self.n2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.qkv, self.proj = nn.Linear(H, 3 * H), nn.Linear(H, H)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * radius + 1))
        self.mlp = (SwiGLU(H, int(round(swiglu_ratio * H))) if swiglu_ratio > 0      # 8a: SwiGLU FFN
                    else nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B, P, H)
        B, P, H = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, P, 3, self.h, H // self.h).permute(2, 0, 3, 1, 4)
        i = torch.arange(P, device=x.device)
        off = i[None, :] - i[:, None]                                  # (P, P): key - query
        inside = off.abs() <= self.r
        bias = self.rel[:, (off.clamp(-self.r, self.r) + self.r)]       # (heads, P, P)
        bias = bias.masked_fill(~inside[None], float("-inf"))
        a = torch.softmax(q @ k.transpose(-1, -2) / (H // self.h) ** 0.5 + bias[None], dim=-1)
        x = x + self.proj((a @ v).transpose(1, 2).reshape(B, P, H))
        return x + self.mlp(self.n2(x))


_REL_EDGES = (0, 1, 2, 3, 5, 9, 17, 33, 65, 129)          # |offset| bins (sites): 0, 1, 2, 3-4, 5-8, ..., 65-128, >= 129


def _rel_bin(off: torch.Tensor) -> torch.Tensor:
    """Signed log-binned relative offset -> index in [0, 2 * len(_REL_EDGES) - 1)."""
    edges = torch.tensor(_REL_EDGES, device=off.device)
    b = torch.bucketize(off.abs(), edges, right=True) - 1                        # 0 .. n-1
    n = len(_REL_EDGES)
    return torch.where(off >= 0, n - 1 + b, n - 1 - b)


class _GlobalAttention(nn.Module):
    """Pre-norm full self-attention over all sites (open domain, any width) with a learned signed relative-position bias
    per head and log-spaced distance bin, shared by all sites, then a 2x MLP. `record` keeps the last attention weights."""

    def __init__(self, H: int, heads: int):
        super().__init__()
        if H % heads:
            raise ValueError(f"hidden {H} not divisible by {heads} heads")
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(H), nn.LayerNorm(H)
        self.qkv, self.proj = nn.Linear(H, 3 * H), nn.Linear(H, H)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * len(_REL_EDGES) - 1))
        self.mlp = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))
        self.record, self.last_attn = False, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B, P, H)
        B, P, H = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, P, 3, self.h, H // self.h).permute(2, 0, 3, 1, 4)
        i = torch.arange(P, device=x.device)
        bias = self.rel[:, _rel_bin(i[None, :] - i[:, None])]          # (heads, P, P)
        a = torch.softmax(q @ k.transpose(-1, -2) / (H // self.h) ** 0.5 + bias[None], dim=-1)
        if self.record:
            self.last_attn = a.detach()
        x = x + self.proj((a @ v).transpose(1, 2).reshape(B, P, H))
        return x + self.mlp(self.n2(x))


class GlobalCoupling(nn.Module):
    """7a: domain-wide features from the first k latent channels of every site, (B, P, c) -> (B, P, G)."""

    def __init__(self, kind: str, k: int, H: int, layers: int, heads: int, G: int):
        super().__init__()
        self.kind, self.k = kind, k
        if kind == "attn":
            self.embed = nn.Linear(k, H)
            self.blocks = nn.ModuleList(_GlobalAttention(H, heads) for _ in range(layers))
            self.out = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, G))
        elif kind == "mean":
            self.out = nn.Sequential(nn.Linear(k, H), nn.GELU(), nn.Linear(H, G))
        else:
            raise ValueError(f"unknown global coupling {kind!r}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z[..., :self.k]
        if self.kind == "mean":
            return self.out(x.mean(-2, keepdim=True)).expand(*x.shape[:-1], -1)
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def _d1_d2(z: torch.Tensor) -> tuple:
    """Centred first and second differences along the sites of `z` (B, P, c), replicate padding (open edges)."""
    zp = torch.cat([z[:, :1], z, z[:, -1:]], dim=1)
    return 0.5 * (zp[:, 2:] - zp[:, :-2]), zp[:, 2:] - 2 * z + zp[:, :-2]


class LatentLaw(nn.Module):
    """FPc: an explicit latent PDE in x with one frame as the time unit,

        dz/dt = A0 z + A1 d1z + A2 d2z + b + Q1 [z_i z_j]_{i<=j} + Q2 [z_i d1z_j] + R(z, d1z, d2z),

    d1, d2 centred differences over sites (radius 1). The linear part gives each latent mode's dispersion
    relation directly (A1: propagation, A2: diffusion / anti-diffusion, A0: growth / decay and channel coupling);
    Q1 / Q2 are the quadratic interactions (Q2 includes advection-like terms z_i d1z_j); R is a pointwise MLP
    residual (zero-initialized output) whose share of the tendency measures how far the law is from closed form.
    Every coefficient is zero at initialization, so the step is exactly the identity. Integrated over one frame
    with `substeps` explicit Euler or Heun steps; reach = stages x substeps sites per frame. With `cap` > 0 the
    tendency is cap tanh(F / cap): identical to F for small F, bounded for large."""

    def __init__(self, c: int, quadratic: bool = True, residual_hidden: int = 64, integrator: str = "heun",
                 substeps: int = 1, global_mean: bool = False, cap: float = 0.0, lead: str = "",
                 lead_nu_pos: bool = False, head_n: int = 0):
        super().__init__()
        if integrator not in ("euler", "heun") or substeps < 1 or cap < 0:
            raise ValueError("integrator must be 'euler' or 'heun', substeps >= 1, cap >= 0")
        if lead not in ("", "z0") or (lead_nu_pos and not lead):
            raise ValueError("lead must be '' or 'z0'; lead_nu_pos needs lead 'z0'")
        self.c, self.integrator, self.substeps, self.cap = c, integrator, substeps, cap
        self.A = nn.Parameter(torch.zeros(3, c, c))               # A0, A1, A2 (output, input)
        self.b = nn.Parameter(torch.zeros(c))
        iu = torch.triu_indices(c, c)
        self.register_buffer("iu", iu)
        self.quadratic = quadratic
        if quadratic:
            self.Q1 = nn.Parameter(torch.zeros(c, iu.shape[1]))
            self.Q2 = nn.Parameter(torch.zeros(c, c * c))
        self.R = None
        if residual_hidden:
            self.R = nn.Sequential(nn.Linear(3 * c, residual_hidden), nn.GELU(), nn.Linear(residual_hidden, c))
            nn.init.zeros_(self.R[-1].weight)
            nn.init.zeros_(self.R[-1].bias)
        self.global_mean = global_mean
        if global_mean:   # hybrid: G <z>, <z> the mean over every site the step sees (the whole domain in evaluation;
            # train on whole-domain windows so training and evaluation see the same mean). Not local: the reach
            # below is the local part's.
            self.G = nn.Parameter(torch.zeros(c, c))
        self.register_buffer("masks", torch.ones(0))              # filled by `prune`: keep-masks of A, Q1, Q2, G
        self.head_n = head_n
        self.head = None
        if head_n:
            if not 1 <= head_n <= c:
                raise ValueError("head_n must be in 1..c")
            self.head = LatentLaw(head_n, quadratic, 0, integrator, 1, False, 0.0)
        self._hd, self._hr = [], []
        self.lead, self.lead_nu_pos = lead, lead_nu_pos
        if lead:   # fixed structural masks: row 0 keeps only its z0-alone terms (FPf)
            mA = torch.ones(3, c, c); mA[:, 0, :] = 0; mA[:, 0, 0] = 1
            self.register_buffer("lead_A", mA)
            if quadratic:
                m1 = torch.ones(c, iu.shape[1]); m1[0] = 0; m1[0, 0] = 1        # iu column 0 is (0, 0): z0^2
                m2 = torch.ones(c, c * c); m2[0] = 0; m2[0, 0] = 1              # column 0 is z0 d1z0
                self.register_buffer("lead_Q1", m1); self.register_buffer("lead_Q2", m2)
            if global_mean:
                mG = torch.ones(c, c); mG[0] = 0
                self.register_buffer("lead_G", mG)
            if lead_nu_pos:   # softplus(-6) = 2.5e-3 sites^2/frame: nearly the identity at initialization
                self.nu0_raw = nn.Parameter(torch.tensor(-6.0))
                e = torch.zeros(3, c, c); e[2, 0, 0] = 1
                self.register_buffer("lead_nu_e", e)
        self._res, self._tot = [], []

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        m = state_dict.get(prefix + "masks")
        if m is not None and m.shape != self.masks.shape:     # a pruned checkpoint: take its keep-masks
            self.masks = torch.empty_like(m, device=self.A.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def coefficient_tensors(self) -> dict:
        out = {"A": self.A}
        if self.quadratic:
            out.update(Q1=self.Q1, Q2=self.Q2)
        if self.global_mean:
            out["G"] = self.G
        return out

    def _lead(self, name: str, v: torch.Tensor) -> torch.Tensor:
        """Apply the FPf structural constraint on channel 0's row (identity when `lead` is off)."""
        if not self.lead:
            return v
        v = v * getattr(self, "lead_" + name)
        if name == "A" and self.lead_nu_pos:
            v = v * (1 - self.lead_nu_e) + self.lead_nu_e * F.softplus(self.nu0_raw)
        return v

    def tendency(self, z: torch.Tensor) -> torch.Tensor:
        d1, d2 = _d1_d2(z)
        A = self.A
        if self.masks.numel():
            A = A * self._mask("A")
        A = self._lead("A", A)
        f = z @ A[0].T + d1 @ A[1].T + d2 @ A[2].T + self.b
        if self.quadratic:
            Q1, Q2 = (self.Q1 * self._mask("Q1"), self.Q2 * self._mask("Q2")) if self.masks.numel() else (self.Q1, self.Q2)
            Q1, Q2 = self._lead("Q1", Q1), self._lead("Q2", Q2)
            zz = (z[..., :, None] * z[..., None, :])[..., self.iu[0], self.iu[1]]
            zd = (z[..., :, None] * d1[..., None, :]).flatten(-2)
            f = f + zz @ Q1.T + zd @ Q2.T
        if self.global_mean:
            G = self.G * self._mask("G") if self.masks.numel() else self.G
            f = f + z.mean(dim=-2, keepdim=True) @ self._lead("G", G).T
        if self.R is not None:
            r = self.R(torch.cat([z, d1, d2], dim=-1))
            if self.lead:   # channel 0's law is closed: no residual
                r = torch.cat([torch.zeros_like(r[..., :1]), r[..., 1:]], dim=-1)
            if self.training:
                self._res.append((r ** 2).mean())
                self._tot.append(((f + r) ** 2).mean().detach())
            f = f + r
        if self.head is not None and self.training:
            g = self.head.tendency(z[..., :self.head_n])
            self._hd.append(((f[..., :self.head_n] - g) ** 2).mean())
            self._hr.append((f[..., :self.head_n] ** 2).mean().detach())
        if self.cap:
            # Saturating bound (FPc fix, 2026-11-07): an explicit quadratic law rolled out 45 steps blew up to inf
            # in Stage B (Saidie 8999). cap tanh(F / cap) equals F within 3% for |F| < 0.3 cap, so the law reads
            # as written in the normal regime, and every step is bounded by cap.
            f = self.cap * torch.tanh(f / self.cap)
        return f

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = 1.0 / self.substeps
        for _ in range(self.substeps):
            k1 = self.tendency(z)
            if self.integrator == "euler":
                z = z + h * k1
            else:
                k2 = self.tendency(z + h * k1)
                z = z + 0.5 * h * (k1 + k2)
        return z

    def penalty(self, l1: float, res_weight: float, head_weight: float = 0.0) -> torch.Tensor:
        """L1 on the law's coefficients (A0's diagonal exempt: each mode's own growth or decay is not a coupling)
        plus `res_weight` x the residual's mean share of the squared tendency over the calls since the last
        penalty (training mode only). Clears the accumulated residual statistics."""
        p = self.A.new_zeros(())
        if l1:
            off = self.A[0] - torch.diag_embed(torch.diagonal(self.A[0]))
            p = p + l1 * (off.abs().sum() + self.A[1:].abs().sum())
            if self.quadratic:
                p = p + l1 * (self.Q1.abs().sum() + self.Q2.abs().sum())
            if self.global_mean:
                p = p + l1 * self.G.abs().sum()
        if res_weight and self._res:
            p = p + res_weight * torch.stack(self._res).mean() / (torch.stack(self._tot).mean() + 1e-8)
        if self.head is not None:
            if head_weight and self._hd:
                p = p + head_weight * torch.stack(self._hd).mean() / (torch.stack(self._hr).mean() + 1e-8)
            if l1:
                p = p + self.head.penalty(l1, 0.0)
        self._res, self._tot, self._hd, self._hr = [], [], [], []
        return p

    def _mask(self, name: str) -> torch.Tensor:
        sizes = {k: v.numel() for k, v in self.coefficient_tensors().items()}
        off = 0
        for k, n in sizes.items():
            if k == name:
                return self.masks[off:off + n].view_as(self.coefficient_tensors()[k])
            off += n
        raise KeyError(name)

    def effective_coefficients(self) -> dict:
        """The coefficients the law actually uses: raw parameters times their keep-masks. After pruning, a
        masked parameter can drift away from zero (Adam momentum, weight decay) without any effect on the law,
        so reports and term counts must read these, not the raw parameters."""
        co = self.coefficient_tensors()
        if self.masks.numel():
            co = {k: v * self._mask(k) for k, v in co.items()}
        return {k: self._lead(k, v) for k, v in co.items()}

    @torch.no_grad()
    def prune(self, threshold: float) -> int:
        """Zero (and keep zero) every coefficient of A, Q1, Q2, G with |value| < threshold; returns the number
        kept. A0's diagonal is always kept. Repeated pruning only removes terms: a term pruned earlier stays
        pruned (FPd's sequential thresholding during training)."""
        keep = []
        for k, v in self.coefficient_tensors().items():
            m = (v.abs() >= threshold).float()
            if self.masks.numel():
                m = m * self._mask(k)
            if k == "A":
                m[0] = torch.maximum(m[0], torch.eye(self.c, device=m.device))
            v.mul_(m)
            keep.append(m.flatten())
        self.masks = torch.cat(keep)
        return int(self.masks.sum().item())

    def n_terms(self, threshold: float = 0.0) -> int:
        return int(sum((v.abs() > threshold).sum().item() for v in self.effective_coefficients().values()))

    def max_coupling(self) -> float:
        """Largest |coefficient| of the law excluding A0's diagonal (the scale for relative thresholds)."""
        mx = 0.0
        for k, v in self.effective_coefficients().items():
            v = v.detach()
            if k == "A":
                v = torch.cat([(v[0] - torch.diag_embed(torch.diagonal(v[0]))).flatten(), v[1:].flatten()])
            mx = max(mx, float(v.abs().max()))
        return mx


class StencilPropagator(nn.Module):
    """One step `z_{n+1} = z_n + f(...)`, `f` local with reach `layers * radius` sites.

    Input: `(B, P, c)` for `n_hist = 1`, or `(B, n_hist, P, c)` oldest-to-newest history; the update is
    added to the newest frame. With `noise_channels > 0`, `f` also sees standard-normal channels drawn
    per call (pass `eps` to fix them), so repeated calls give samples of a stochastic one-step map.
    Defaults (`n_hist = 1`, no noise) are the original Markovian deterministic propagator, with the same
    parameter shapes, so earlier checkpoints load unchanged."""

    def __init__(self, cfg: StencilPropConfig):
        super().__init__()
        if cfg.padding_mode == "circular":
            raise ValueError("circular padding is wrong for CM1 (open lateral boundaries)")
        if cfg.n_hist < 1 or cfg.noise_channels < 0:
            raise ValueError("n_hist >= 1 and noise_channels >= 0 required")
        self.cfg = cfg
        c, H, r = cfg.latent_channels, cfg.hidden, cfg.radius
        if cfg.prefix_m:
            self.register_buffer("prefix_fill", torch.zeros(c))
        n_in = c * cfg.n_hist + cfg.noise_channels
        self.global_off = False                       # 7a ablation: True feeds zeros in place of the global features
        if cfg.global_kind and cfg.body != "conv":
            raise ValueError("global coupling (7a) is implemented for body 'conv' only")
        if cfg.global_kind:
            if not 1 <= cfg.global_k <= c:
                raise ValueError(f"global_k must be in 1..{c}")
            self.glob = GlobalCoupling(cfg.global_kind, cfg.global_k, cfg.global_hidden, cfg.global_layers,
                                       cfg.global_heads, cfg.global_feat)
            n_in += cfg.global_feat
        if cfg.body == "conv":
            dims = [n_in] + [H] * (cfg.layers - 1)
            self.convs = nn.ModuleList(nn.Conv1d(a, H, 2 * r + 1, padding=r, padding_mode=cfg.padding_mode)
                                       for a in dims)
            if cfg.global_kind:                       # the stencil starts blind to the global features
                with torch.no_grad():
                    self.convs[0].weight[:, n_in - cfg.global_feat:].zero_()
            self.head = nn.Conv1d(H, c, 1)
        elif cfg.body == "law":         # FPc: explicit latent PDE (Markovian, deterministic)
            if cfg.n_hist != 1 or cfg.noise_channels:
                raise ValueError("body 'law' is Markovian and deterministic: n_hist = 1, noise_channels = 0")
            self.law = LatentLaw(c, cfg.law_quadratic, cfg.law_residual_hidden, cfg.law_integrator, cfg.law_substeps,
                                 cfg.law_global, cfg.law_cap, cfg.law_lead, cfg.law_lead_nu_pos,
                                 cfg.law_head_n)
            return
        elif cfg.body == "mlp":         # 8a: dense MLP on the whole neighbourhood (same reach as the conv body)
            R = cfg.layers * r
            self.mlp_in = nn.Linear(n_in * (2 * R + 1), H)
            self.mlp_blocks = nn.ModuleList(nn.ModuleDict({"n": nn.RMSNorm(H), "f": SwiGLU(H, int(round(cfg.mlp_ratio * H)))})
                                            for _ in range(max(1, cfg.layers - 1)))
            self.norm = nn.RMSNorm(H)
            self.head = nn.Linear(H, c)
        elif cfg.body == "attention":   # M1a: same reach (layers x radius), input-dependent neighbour weights
            self.embed = nn.Linear(n_in, H)
            self.blocks = nn.ModuleList(_WindowAttention(H, cfg.heads, r, cfg.mlp_ratio if cfg.attn_swiglu else 0.0)
                                        for _ in range(cfg.layers))
            self.norm = nn.LayerNorm(H)
            self.head = nn.Linear(H, c)
        else:
            raise ValueError(f"unknown propagator body {cfg.body!r}")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def penalty(self) -> torch.Tensor:
        """Regularization of the step to add to the training loss (zero except for body 'law')."""
        if self.cfg.body == "law":
            return self.law.penalty(self.cfg.law_l1, self.cfg.law_res_weight, self.cfg.law_head_weight)
        return next(self.parameters()).new_zeros(())

    def set_prefix(self, m: int, fill: torch.Tensor) -> None:
        self.cfg.prefix_m = m
        self.register_buffer("prefix_fill", fill.detach().clone().to(next(self.parameters()).device, torch.float32))

    def forward(self, z: torch.Tensor, eps: torch.Tensor | None = None,
                gen: torch.Generator | None = None) -> torch.Tensor:
        m = self.cfg.prefix_m
        if not m:
            return self._forward(z, eps, gen)
        z = torch.cat([z[..., :m], self.prefix_fill[m:].expand(*z.shape[:-1], -1)], -1)       # the dropped channels are the fill
        out = self._forward(z, eps, gen)
        return torch.cat([out[..., :m], self.prefix_fill[m:].expand(*out.shape[:-1], -1)], -1)

    def _forward(self, z: torch.Tensor, eps: torch.Tensor | None = None,
                 gen: torch.Generator | None = None) -> torch.Tensor:
        if self.cfg.body == "law":
            return self.law(z[:, -1] if z.dim() == 4 else z)
        if self.cfg.n_hist == 1:
            if z.dim() == 4:
                z = z[:, -1]
            last, x = z, z.transpose(1, 2)                                   # (B, c, P)
        else:
            if z.dim() != 4 or z.shape[1] != self.cfg.n_hist:
                raise ValueError(f"expected (B, {self.cfg.n_hist}, P, c) history, got {tuple(z.shape)}")
            last = z[:, -1]
            B, Hh, P, c = z.shape
            x = z.permute(0, 1, 3, 2).reshape(B, Hh * c, P)
        if self.cfg.noise_channels:
            if eps is None:
                eps = torch.randn((x.shape[0], self.cfg.noise_channels, x.shape[2]), generator=gen,
                                  device="cpu").to(x.device, x.dtype)
            x = torch.cat([x, eps], dim=1)
        if self.cfg.global_kind:
            g = self.glob(last)                                                # (B, P, G)
            if self.global_off:
                g = torch.zeros_like(g)
            x = torch.cat([x, g.transpose(1, 2)], dim=1)
        if self.cfg.body == "mlp":
            R = self.cfg.layers * self.cfg.radius
            xp = F.pad(x, (R, R), mode=self.cfg.padding_mode)                  # (B, n_in, P + 2R)
            nb = xp.unfold(2, 2 * R + 1, 1)                                    # (B, n_in, P, 2R+1)
            h = self.mlp_in(nb.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1))
            for blk in self.mlp_blocks:
                h = h + blk["f"](blk["n"](h))
            return last + self.head(self.norm(h))
        if self.cfg.body == "attention":
            h = self.embed(x.transpose(1, 2))                                  # (B, P, H)
            for blk in self.blocks:
                h = blk(h)
            return last + self.head(self.norm(h))
        h = x
        for conv in self.convs:
            h = F.gelu(conv(h))
        return last + self.head(h).transpose(1, 2)


def _with_true_halo(state: torch.Tensor, truth: torch.Tensor, halo: int) -> torch.Tensor:
    if halo <= 0:
        return state
    P = state.shape[-2]
    return torch.cat([truth[..., :halo, :], state[..., halo:P - halo, :], truth[..., P - halo:, :]], dim=-2)


def forced_halo_rollout(prop: StencilPropagator, z_true: torch.Tensor, k: int, halo: int,
                        noise: float = 0.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """`z_true`: `(B, H-1+k+1, P, c)` with H = `prop.cfg.n_hist` (H-1 extra history frames first). Rolls
    `k` steps from the true history ending at frame H-1 (+ input noise on the history), setting the
    `halo` sites at each end of every frame the step sees to the true state (the boundary condition a
    limited-area model is given). Returns the `k` predicted states, `(B, k, P, c)`, halo true."""
    H = prop.cfg.n_hist
    if z_true.shape[1] < H + k:
        raise ValueError(f"need {H + k} frames for history {H} and k = {k}, got {z_true.shape[1]}")
    hist = [z_true[:, j] for j in range(H)]
    if noise > 0:
        hist = [h + noise * torch.randn_like(h) for h in hist]
    out = []
    for j in range(k):
        t = H - 1 + j                                                       # index of the newest frame
        seen = [_with_true_halo(hist[-H + i], z_true[:, t - H + 1 + i], halo) for i in range(H)]
        inp = seen[0] if H == 1 else torch.stack(seen, dim=1)
        state = _with_true_halo(prop(inp, gen=gen), z_true[:, t + 1], halo)
        out.append(state)
        hist.append(state)
    return torch.stack(out, dim=1)


def rollout_from(prop: StencilPropagator, z_traj: torch.Tensor, t0: int, n: int, halo: int,
                 n_samples: int = 1, gen: torch.Generator | None = None, boundary: str = "truth") -> torch.Tensor:
    """Free forecast from frame `t0` of one latent trajectory `z_traj` (T, P, c), history from the true
    trajectory. Returns `(n, n_samples, P, c)`. The `halo` sites at each end are the boundary condition:
    `boundary="truth"` sets them to the true state at every step (a limited-area model with perfect
    lateral boundaries); `"persistence"` holds them at their state at t0, so nothing after t0 is known -
    the check that the truth at the edges is not what makes the forecasts good."""
    H = prop.cfg.n_hist
    if t0 - H + 1 < 0 or t0 + n >= z_traj.shape[0]:
        raise ValueError("forecast window outside the trajectory")
    if boundary == "truth":
        win = z_traj[t0 - H + 1:t0 + n + 1]
    elif boundary == "persistence":
        win = torch.cat([z_traj[t0 - H + 1:t0 + 1], z_traj[t0:t0 + 1].expand(n, -1, -1)], dim=0)
    else:
        raise ValueError(f"unknown boundary {boundary!r}")
    win = win[None].expand(n_samples, -1, -1, -1)
    return forced_halo_rollout(prop, win, n, halo, gen=gen).transpose(0, 1)

