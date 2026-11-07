"""SM4: a generative (conditional flow-matching) propagator for the CM1 latent field.

Instead of regressing the next latent field (whose MSE optimum is the smooth conditional mean), the model
learns to SAMPLE it. With delta = (z_{n+1} - z_n) / scale (per-channel scale of the training increments),
x0 ~ N(0, I) per site and channel, and x_s = (1 - s) x0 + s delta for flow time s in [0, 1], a velocity
network v(x_s, s | z_n) is trained to match x1 - x0 (rectified flow / conditional flow matching). Sampling
integrates dx/ds = v from s = 0 to 1 with `n_flow` Euler steps; z_{n+1} = z_n + scale * x(1).

Locality: the condition z_n enters through a stencil context network (reach `layers * radius` sites), and
the velocity network mixes neighbouring sites of x (reach `flow_layers * flow_radius` per Euler step), so a
sample's spatial structure is built locally; the total reach per physical step is
`layers * radius + n_flow * flow_layers * flow_radius` sites (reported as `reach_sites`).

With `flow_radius` 0 the velocity network is pointwise in x, so the reach is the context network's alone
(`layers * radius`, 3 sites as for the stencil propagator) and the flow model fits Stage C's forced halo
(SM4b): neighbouring sites are coupled through the condition, and over successive physical steps.

Interface: like a stochastic `StencilPropagator` (`forward(z, eps=None, gen=None)`, `cfg.noise_channels > 0`,
`cfg.n_hist == 1`), so `rollout_from`, `forecast_report` and the DA twin use it unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowPropConfig:
    latent_channels: int
    hidden: int = 256
    layers: int = 3                 # context (condition) network
    radius: int = 1
    flow_hidden: int = 256
    flow_layers: int = 2            # velocity network
    flow_radius: int = 1
    n_flow: int = 8                 # Euler steps per sample
    time_features: int = 16
    padding_mode: str = "replicate"
    n_hist: int = 1
    delta_scale: list = field(default_factory=list)   # per-channel std of the training increments

    @property
    def noise_channels(self) -> int:        # tells rollout/eval/twin code this model is stochastic
        return self.latent_channels

    @property
    def reach_sites(self) -> int:
        return self.layers * self.radius + self.n_flow * self.flow_layers * self.flow_radius


class FlowPropagator(nn.Module):
    def __init__(self, cfg: FlowPropConfig):
        super().__init__()
        if cfg.padding_mode == "circular":
            raise ValueError("circular padding is wrong for CM1 (open lateral boundaries)")
        self.cfg = cfg
        c, H, r = cfg.latent_channels, cfg.hidden, cfg.radius
        self.ctx = nn.ModuleList(nn.Conv1d(c if i == 0 else H, H, 2 * r + 1, padding=r, padding_mode=cfg.padding_mode)
                                 for i in range(cfg.layers))
        Hf, rf = cfg.flow_hidden, cfg.flow_radius
        dims = [c + H + cfg.time_features] + [Hf] * (cfg.flow_layers - 1)
        self.vel = nn.ModuleList(nn.Conv1d(a, Hf, 2 * rf + 1, padding=rf, padding_mode=cfg.padding_mode) for a in dims)
        self.out = nn.Conv1d(Hf, c, 1)
        scale = torch.as_tensor(cfg.delta_scale if cfg.delta_scale else [1.0] * c, dtype=torch.float32)
        if scale.numel() != c or (scale <= 0).any():
            raise ValueError("delta_scale needs one positive value per latent channel")
        self.register_buffer("scale", scale)
        self.temperature = 1.0      # sampling only: the source noise is drawn with this standard deviation

    def context(self, z: torch.Tensor) -> torch.Tensor:
        h = z.transpose(1, 2)                                                  # (B, c, P)
        for conv in self.ctx:
            h = F.gelu(conv(h))
        return h

    def _time(self, s: torch.Tensor, P: int) -> torch.Tensor:
        """(B,) flow times -> (B, time_features, P) sinusoidal features."""
        k = self.cfg.time_features // 2
        f = torch.exp(torch.linspace(0, math.log(100.0), k, device=s.device, dtype=s.dtype))
        a = s[:, None] * f[None] * math.pi
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)[:, :, None].expand(-1, -1, P)

    def velocity(self, x: torch.Tensor, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x (B, c, P), s (B,), h (B, H, P) -> (B, c, P)."""
        y = torch.cat([x, h, self._time(s, x.shape[-1])], dim=1)
        for conv in self.vel:
            y = F.gelu(conv(y))
        return self.out(y)

    def loss(self, z0: torch.Tensor, z1: torch.Tensor, gen: torch.Generator | None = None,
             margin: int = 0) -> torch.Tensor:
        """Flow-matching loss for latent pairs z0 -> z1, (B, P, c) each; `margin` sites at each end left out."""
        B, P, c = z0.shape
        x1 = ((z1 - z0) / self.scale).transpose(1, 2)                          # (B, c, P)
        x0 = torch.randn(x1.shape, generator=gen, device="cpu").to(x1.device, x1.dtype)
        s = torch.rand(B, generator=gen, device="cpu").to(x1.device, x1.dtype)
        xs = (1 - s)[:, None, None] * x0 + s[:, None, None] * x1
        v = self.velocity(xs, s, self.context(z0))
        e = (v - (x1 - x0)) ** 2
        return e[..., margin:P - margin].mean() if margin else e.mean()

    def forward(self, z: torch.Tensor, eps: torch.Tensor | None = None, gen: torch.Generator | None = None) -> torch.Tensor:
        if z.dim() == 4:
            z = z[:, -1]
        B, P, c = z.shape
        h = self.context(z)
        x = eps if eps is not None else self.temperature * torch.randn((B, c, P), generator=gen, device="cpu").to(z.device, z.dtype)
        n = self.cfg.n_flow
        for i in range(n):
            s = torch.full((B,), i / n, device=z.device, dtype=z.dtype)
            x = x + self.velocity(x, s, h) / n
        return z + (x * self.scale[:, None]).transpose(1, 2)
