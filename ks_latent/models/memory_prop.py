"""Memory channels for the 2D local latent field (Z10a, user idea 2026-10-13; docs/split_paper_tasks.md Z10a).

Per site the state is `(z, m)`: the `c` encoder channels (a function of the current field only; the decoder reads
only these) plus `m < c` memory channels that the encoder never produces. Encoding a field gives `m = 0`; memory
fills only as the propagator runs over a history of states. One step maps `(z_t, m_t) -> (z_{t+1}, m_{t+1})`:

  z_{t+1} = z_t + U_z( Mix_z( E_z([z_t, m_t]) ) )                  (reads neighbours' z AND m within the reach)
  m_{t+1} = lam * m_t + (1 - lam) * tanh( U_m([ Mix_m(E_m(z_t)), m_t ]) )   (option (i), default)

Option (i): the memory update reads only its OWN site's memory plus its neighbours' encoder channels, so memory never
relays memory sideways. Information in memory still spreads sideways over time through z (memory_j -> z_j ->
neighbours' z -> their memory), at most one reach per step. In forced-halo training the halo's z is overwritten by
data, which cuts that route at the window edge, so a window rollout equals the large-domain forced reference when the
window carries one extra reach of forced data beyond the halo (or IS the periodic training box). `relay=True`
(option (ii)) lets the memory update read neighbours' memory too; then memory relays memory and the window rollout is
no longer exact (tests/unit/test_memory_prop.py checks both). `lam` is a learned per-channel leak in (0, 1), so memory
stays bounded (|m| < 1 for |m_0| < 1).

Every weight depends on `c`, `m`, `token_d_model`, `window`, `n_layers`, never on the number of sites, so the model
resizes like `local_mlp_2d`. The z readout is zero-initialized (identity at init for z, as KC2); with `m = 0` channels
the model reduces to KC2's `local_mlp_2d` body exactly (same parameterization of the z path).

External interface (so `lyapunov_spectrum_latent_propagator(mode="single_state")` and the eval scripts work on the
extended, Markovian state): `cfg.d_latent = n_sites_x * n_sites_y * (c + m)`, site-major with the `c + m` channels of a
site contiguous (z first); `step(z_prev, z)` (ignores `z_prev`), `step_one(z)`, `rollout(z_prev, z, k)`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from ks_latent.models.propagator import _LocalMixerBlock2D


@dataclass
class MemoryPropConfig:
    n_sites_x: int
    n_sites_y: int
    local_channels: int  # c, encoder channels
    memory_channels: int  # m
    token_d_model: int = 32
    n_layers: int = 1
    window: int = 3
    mlp_ratio: int = 4
    relay: bool = False  # option (ii)
    dropout: float = 0.0
    zero_init: bool = True

    @property
    def d_latent(self) -> int:
        return self.n_sites_x * self.n_sites_y * (self.local_channels + self.memory_channels)

    @property
    def d_encoder(self) -> int:
        return self.n_sites_x * self.n_sites_y * self.local_channels


class MemoryLocalProp2D(nn.Module):
    mode = "markovian"

    def __init__(self, cfg: MemoryPropConfig):
        super().__init__()
        self.cfg = cfg
        c, m, D = cfg.local_channels, cfg.memory_channels, cfg.token_d_model
        # z path: KC2's local_mlp_2d body on [z, m]
        self.z_embed = nn.Linear(c + m, D)
        self.z_blocks = nn.ModuleList([_LocalMixerBlock2D(D, cfg.window, cfg.mlp_ratio, cfg.dropout)
                                       for _ in range(cfg.n_layers)])
        self.z_norm = nn.LayerNorm(D)
        self.z_unembed = nn.Linear(D, c)
        if cfg.zero_init:
            nn.init.zeros_(self.z_unembed.weight)
            nn.init.zeros_(self.z_unembed.bias)
        if m > 0:
            self.m_embed = nn.Linear(c + m if cfg.relay else c, D)
            self.m_blocks = nn.ModuleList([_LocalMixerBlock2D(D, cfg.window, cfg.mlp_ratio, cfg.dropout)
                                           for _ in range(cfg.n_layers)])
            self.m_norm = nn.LayerNorm(D)
            self.m_out = nn.Sequential(nn.Linear(D + m, D), nn.GELU(), nn.Linear(D, m))
            self.m_leak = nn.Parameter(torch.full((m,), 2.0))  # lam = sigmoid(2) ~ 0.88 at init

    # -- core map on (B, Px, Py, c) / (B, Px, Py, m) ------------------------------------------------------------
    def step_zm(self, z: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, Px, Py, c = z.shape
        m = self.cfg.memory_channels
        zm = torch.cat([z, mem], dim=-1) if m > 0 else z
        h = self.z_embed(zm.reshape(B, Px * Py, -1))
        for blk in self.z_blocks:
            h = blk(h, Px, Py)
        z_next = z + self.z_unembed(self.z_norm(h)).reshape(B, Px, Py, c)
        if m == 0:
            return z_next, mem
        src = zm if self.cfg.relay else z
        g = self.m_embed(src.reshape(B, Px * Py, -1))
        for blk in self.m_blocks:
            g = blk(g, Px, Py)
        cand = torch.tanh(self.m_out(torch.cat([self.m_norm(g), mem.reshape(B, Px * Py, m)], dim=-1)))
        lam = torch.sigmoid(self.m_leak)
        m_next = lam * mem + (1 - lam) * cand.reshape(B, Px, Py, m)
        return z_next, m_next

    # -- flat extended-state interface -------------------------------------------------------------------------
    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        x = x.reshape(x.shape[0], cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels + cfg.memory_channels)
        return x[..., :cfg.local_channels], x[..., cfg.local_channels:]

    @staticmethod
    def join(z: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, mem], dim=-1).reshape(z.shape[0], -1)

    def extend(self, z_enc: torch.Tensor) -> torch.Tensor:
        """Encoder output `(B, n_sites*c)` -> extended state with memory 0."""
        cfg = self.cfg
        z = z_enc.reshape(z_enc.shape[0], cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
        return self.join(z, z.new_zeros(*z.shape[:3], cfg.memory_channels))

    def encoder_part(self, x: torch.Tensor) -> torch.Tensor:
        """Extended state(s) `(..., d_latent)` -> encoder channels `(..., n_sites*c)` (what the decoder reads)."""
        lead = x.shape[:-1]
        z, _ = self.split(x.reshape(-1, x.shape[-1]))
        return z.reshape(*lead, -1)

    def step_one(self, x: torch.Tensor) -> torch.Tensor:
        z, mem = self.split(x)
        return self.join(*self.step_zm(z, mem))

    def step(self, x_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.step_one(x)

    def rollout(self, x_prev: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
        out = []
        for _ in range(k):
            x = self.step_one(x)
            out.append(x)
        return torch.stack(out, dim=1)

    def warmup(self, z_hist: torch.Tensor) -> torch.Tensor:
        """`z_hist`: `(B, W+1, n_sites*c)` encoded truth states, oldest first. Runs W steps with z forced to the
        data (memory evolves from 0) and returns the extended state at the last history time (z = data)."""
        cfg = self.cfg
        B, W1, _ = z_hist.shape
        zs = z_hist.reshape(B, W1, cfg.n_sites_x, cfg.n_sites_y, cfg.local_channels)
        mem = zs.new_zeros(B, cfg.n_sites_x, cfg.n_sites_y, cfg.memory_channels)
        for t in range(W1 - 1):
            _, mem = self.step_zm(zs[:, t], mem)
        return self.join(zs[:, -1], mem)


def resized_memory_config(cfg: MemoryPropConfig, n_sites_x: int, n_sites_y: int) -> MemoryPropConfig:
    return dataclasses.replace(cfg, n_sites_x=n_sites_x, n_sites_y=n_sites_y)


def load_memory_prop_resized(path, new_n_sites_x: int, new_n_sites_y: int, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = resized_memory_config(ckpt["memory_prop_config"], new_n_sites_x, new_n_sites_y)
    prop = MemoryLocalProp2D(cfg)
    prop.load_state_dict(ckpt["prop_state_dict"], strict=True)
    return prop.to(device), cfg, ckpt


def is_memory_checkpoint(path) -> bool:
    return "memory_prop_config" in torch.load(path, map_location="cpu", weights_only=False)
