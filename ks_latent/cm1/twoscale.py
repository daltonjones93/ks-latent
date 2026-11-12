"""Two-scale latent model (plan 6a): a deterministic large-scale model plus a stochastic small-scale model conditioned on it.

Motivation (docs/cm1_notes.md, 6a pre-registration). Small perturbations in CM1 grow at 0.1-0.25 / min after ~15 min, so convective-scale
detail is predictable only for 10-20 min; a deterministic model trained on MSE regresses it to its conditional mean (smooth, not
chaotic). Here the large scales are carried by an existing deterministic model (the pruned, nested-dropout `fpc_nested_p8_dft`, whose
first 8 channels are ~40-100 km, slow fields) and what it misses,

    r = u - D_L(E_L(u))                      (residual split: u = D_L(z_L) + r exactly)

is carried by a separate small-scale autoencoder (z_S = E_S(r), r_hat = D_S(z_S)) and a STOCHASTIC small-scale propagator

    z_S(t+1) = z_S(t) + g(z_S(t), z_L(t), z_L(t+1), noise)                (one-way: large drives small)

trained on a distributional score (fair CRPS + energy score + log-spectrum) of the decoded residual, so its samples are meant to be
plausible small scales consistent with the large-scale state, not their mean.

`TwoScaleAE` and `TwoScaleProp` expose the same interface as `ColumnFieldAE` / `StencilPropagator` on the concatenated latent
z = [z_L (all c_L channels of the large model; its pruned channels stay at their fill), z_S], so every existing rollout, evaluation
and DA code path runs on the two-scale model unchanged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ks_latent.cm1.models import ColumnAEConfig, build_autoencoder


@dataclass
class SmallPropConfig:
    small_channels: int = 16
    cond_channels: int = 8          # leading large-scale channels used as conditioning (the pruned model's kept ones)
    noise_channels: int = 8
    hidden: int = 256
    layers: int = 2
    radius: int = 1
    padding_mode: str = "replicate"

    @property
    def reach_sites(self) -> int:
        return self.layers * self.radius


class SmallProp(nn.Module):
    """g: stencil convolutions over sites of [z_S(t), z_L(t), z_L(t+1) - z_L(t), noise] -> increment of z_S (zero-initialized head:
    the identity at start)."""

    def __init__(self, cfg: SmallPropConfig):
        super().__init__()
        self.cfg = cfg
        n_in = cfg.small_channels + 2 * cfg.cond_channels + cfg.noise_channels
        dims = [n_in] + [cfg.hidden] * (cfg.layers - 1)
        r = cfg.radius
        self.convs = nn.ModuleList(nn.Conv1d(a, cfg.hidden, 2 * r + 1, padding=r, padding_mode=cfg.padding_mode) for a in dims)
        self.head = nn.Conv1d(cfg.hidden, cfg.small_channels, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, zs: torch.Tensor, zl0: torch.Tensor, zl1: torch.Tensor, gen: torch.Generator | None = None,
                eps: torch.Tensor | None = None) -> torch.Tensor:
        """zs (B, P, cs); zl0, zl1 (B, P, cond_channels). Returns zs(t+1)."""
        x = torch.cat([zs, zl0, zl1 - zl0], -1).transpose(1, 2)               # (B, n, P)
        if self.cfg.noise_channels:
            if eps is None:
                eps = torch.randn((x.shape[0], self.cfg.noise_channels, x.shape[2]), generator=gen, device="cpu").to(x.device, x.dtype)
            x = torch.cat([x, eps], 1)
        h = x
        for conv in self.convs:
            h = F.gelu(conv(h))
        return zs + self.head(h).transpose(1, 2)


class TwoScaleAE(nn.Module):
    """encode: z = [E_L(u), E_S(u - D_L(E_L(u)))]; decode: D_L(z_L) + D_S(z_S), condensate clamped at 0."""

    def __init__(self, ae_l: nn.Module, ae_s: nn.Module):
        super().__init__()
        self.ae_l, self.ae_s = ae_l, ae_s
        self.cl = ae_l.cfg.latent_channels
        self.cfg = SimpleNamespace(latent_channels=self.cl + ae_s.cfg.latent_channels, patch=ae_l.cfg.patch,
                                   reach_sites=max(ae_l.cfg.reach_sites, ae_s.cfg.reach_sites), prefix_m=0,
                                   n_levels=ae_l.cfg.n_levels, n_channels=ae_l.cfg.n_channels, kind="twoscale")
        self.positive_mask = ae_l.positive_mask

    def encode(self, u: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        zl = self.ae_l.encode(u, env)
        return torch.cat([zl, self.ae_s.encode(u - self.ae_l.decode(zl))], -1)

    def decode_parts(self, z: torch.Tensor) -> tuple:
        return self.ae_l.decode(z[..., :self.cl]), self.ae_s.decode(z[..., self.cl:])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        a, b = self.decode_parts(z)
        u = a + b
        return torch.where(self.positive_mask[None, :, None], u.clamp(min=0), u)

    def forward(self, u, env=None):
        z = self.encode(u, env)
        return self.decode(z), z


class TwoScaleProp(nn.Module):
    """One step of the coupled model on z = [z_L, z_S]: z_L <- P_L(z_L) (deterministic, frozen large model), then
    z_S <- g(z_S, z_L(t), z_L(t+1), noise)."""

    def __init__(self, prop_l: nn.Module, small: SmallProp):
        super().__init__()
        self.prop_l, self.small = prop_l, small
        self.cl = prop_l.cfg.latent_channels
        self.ck = small.cfg.cond_channels
        self.cfg = SimpleNamespace(latent_channels=self.cl + small.cfg.small_channels, n_hist=1,
                                   noise_channels=small.cfg.noise_channels, prefix_m=0, body="twoscale",
                                   reach_sites=prop_l.cfg.reach_sites + small.cfg.reach_sites)

    def forward(self, z: torch.Tensor, eps: torch.Tensor | None = None, gen: torch.Generator | None = None) -> torch.Tensor:
        if z.dim() == 4:
            z = z[:, -1]
        zl, zs = z[..., :self.cl], z[..., self.cl:]
        zl1 = self.prop_l(zl)
        return torch.cat([zl1, self.small(zs, zl[..., :self.ck], zl1[..., :self.ck], gen=gen, eps=eps)], -1)

    def project_latent(self, Z: torch.Tensor) -> torch.Tensor:
        """After additive noise in the twin: the large model's pruned channels back to their fill."""
        pm = getattr(self.prop_l.cfg, "prefix_m", 0)
        if not pm:
            return Z
        fill = self.prop_l.prefix_fill[pm:].to(Z)
        return torch.cat([Z[..., :pm], fill.expand(*Z.shape[:-1], -1), Z[..., self.cl:]], -1)

    def penalty(self) -> torch.Tensor:
        return next(self.small.parameters()).new_zeros(())


def save_twoscale(out: Path, large_run: Path, ae_s: nn.Module, small: SmallProp, extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ae_s.state_dict(), "cfg": asdict(ae_s.cfg)}, out / "ae_s.pt")
    torch.save({"state_dict": small.state_dict(), "cfg": asdict(small.cfg)}, out / "small_prop.pt")
    (out / "twoscale.json").write_text(json.dumps({"large_run": str(large_run), **(extra or {})}, indent=1))


def load_twoscale(run: Path, dev, load_large):
    """`load_large(path, dev)` -> (ae, prop, norm, cfg, split) of the large run (da_twin.load_run). Returns the same tuple for the
    two-scale model; cfg is the large run's with the halo widened to the coupled model's reach."""
    meta = json.loads((run / "twoscale.json").read_text())
    lr = Path(meta["large_run"])
    if not lr.exists():
        lr = Path.home() / "cm1_runs" / lr.name
    ae_l, prop_l, norm, cfg, split = load_large(lr, dev)
    cka, ckp = torch.load(run / "ae_s.pt", map_location=dev), torch.load(run / "small_prop.pt", map_location=dev)
    ae_s = build_autoencoder(ColumnAEConfig(**cka["cfg"]), None).to(dev)
    ae_s.load_state_dict(cka["state_dict"])
    small = SmallProp(SmallPropConfig(**ckp["cfg"])).to(dev)
    small.load_state_dict(ckp["state_dict"])
    ae, prop = TwoScaleAE(ae_l, ae_s).eval(), TwoScaleProp(prop_l, small).eval()
    cfg = json.loads(json.dumps(cfg))
    cfg["stage_b"]["halo_sites"] = max(cfg["stage_b"]["halo_sites"], prop.cfg.reach_sites)
    return ae, prop, norm, cfg, split
