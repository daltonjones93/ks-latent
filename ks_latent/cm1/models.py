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

    @property
    def reach_sites(self) -> int:
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

    def encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
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


@dataclass
class StencilPropConfig:
    latent_channels: int = 16
    hidden: int = 256
    layers: int = 3
    radius: int = 1
    padding_mode: str = "replicate"
    n_hist: int = 1          # frames of history the step sees (D13); 1 = Markovian
    noise_channels: int = 0  # random input channels per site (D14c stochastic propagator); 0 = deterministic

    @property
    def reach_sites(self) -> int:
        return self.layers * self.radius


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
        dims = [c * cfg.n_hist + cfg.noise_channels] + [H] * (cfg.layers - 1)
        self.convs = nn.ModuleList(nn.Conv1d(a, H, 2 * r + 1, padding=r, padding_mode=cfg.padding_mode)
                                   for a in dims)
        self.head = nn.Conv1d(H, c, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, eps: torch.Tensor | None = None,
                gen: torch.Generator | None = None) -> torch.Tensor:
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

