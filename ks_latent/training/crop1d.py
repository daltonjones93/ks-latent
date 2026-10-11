"""Window (crop) training for the 1D local latent field (Paper II, option B of 2026-10-11).

The 1D analogue of `ks_latent.training.crop`: a 1D `KSAutoencoderLocalField` and a `local_mlp`
propagator are trained on random x-windows of ONE wide periodic 1D simulation instead of on whole
L = 100 periodic boxes, exactly as the 2D models of Paper I are trained on windows of the 264^2 domain.

- Stage A (`train_crop_ae_1d`): reconstruction on windows of `crop_sites` sites; sites within
  `mask_sites` (= encoder + decoder reach) of a window edge are excluded from the loss and from the
  channel-covariance regularizers, so circular padding at the window edge never reaches the loss.
  For a per-patch autoencoder (reach 0) the mask is 0.
- Stage B: windows are encoded once by the frozen encoder (`encode_crop_bank_1d`, the outer
  `enc_sites` dropped) into a bank of shape `(n_crops, T, Pu, 1, c)`, i.e. the 2D bank layout with a
  single site in y, and the propagator is trained by the 2D trainer `crop.train_crop_prop` with
  `halo_sites_y = 0`. A 1D `local_mlp` propagator flattens site-major, so `(B, Pu, 1, c)` reshaped to
  `(B, Pu*c)` is exactly its input; the forced-halo construction (Paper I, Prop. 2) is unchanged.

`resize_*` save the trained weights in the L = 100 checkpoint format (16 sites, NX = 256), which is
what `scripts/eval_1d_transfer.py` and the 1D data-assimilation scripts load and resize from.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ks_latent.training.crop import warmup_cosine
from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss


def gather_crops_1d(u: torch.Tensor, traj: torch.Tensor, t: torch.Tensor, x0: torch.Tensor,
                    width: int) -> torch.Tensor:
    """`u`: `(N, T, NX)`; `traj`, `x0`: `(B,)`; `t`: `(B,)`. Returns `(B, width)` windows, wrapping
    mod NX (the wide domain is periodic, so a window across its seam is physically valid)."""
    NX = u.shape[2]
    idx = (x0[:, None] + torch.arange(width, device=u.device)[None, :]) % NX
    return u[traj[:, None], t[:, None], idx]


@dataclass
class CropAE1DConfig:
    crop_sites: int = 40
    mask_sites: int = 0  # encoder + decoder reach in sites; 0 for a per-patch autoencoder
    batch: int = 64
    steps: int = 8000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.02
    grad_clip: float = 1.0
    w_var: float = 0.02
    w_logdet: float = 0.0035
    logdet_eps: float = 1e-3
    val_every: int = 250
    seed: int = 0


def train_crop_ae_1d(ae, u_train: torch.Tensor, u_val: torch.Tensor, cfg: CropAE1DConfig, log=print) -> dict:
    """`ae`: a 1D `KSAutoencoderLocalField` built at the window size (`NX = crop_sites * patch`).
    `u_*`: normalized wide fields `(N, T, NX_wide)` on the training device."""
    dev = u_train.device
    acfg = ae.cfg
    patch = acfg.NX // acfg.n_sites
    Wc = cfg.crop_sites * patch
    assert acfg.n_sites == cfg.crop_sites and acfg.NX == Wc, "AE must be built at the window size"
    m, ms, c = cfg.mask_sites * patch, cfg.mask_sites, acfg.local_channels
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))

    def sample(u, B, gen):
        N, T, NX = u.shape
        tr = torch.randint(0, N, (B,), generator=gen).to(dev)
        t = torch.randint(0, T, (B,), generator=gen).to(dev)
        x0 = torch.randint(0, NX, (B,), generator=gen).to(dev)
        return gather_crops_1d(u, tr, t, x0, Wc)

    vg = torch.Generator(device="cpu").manual_seed(12345)
    u_val_crops = torch.cat([sample(u_val, cfg.batch, vg) for _ in range(8)])
    hist, best, best_state = [], float("inf"), None
    t_start = time.time()
    for step in range(cfg.steps):
        ae.train()
        uc = sample(u_train, cfg.batch, g)
        z = ae.encode(uc)
        uh = ae.decode(z)
        l_rec = F.mse_loss(uh[:, m:Wc - m], uc[:, m:Wc - m])
        zs = z.reshape(cfg.batch, acfg.n_sites, c)[:, ms:acfg.n_sites - ms].reshape(-1, c)
        _, l_var = decorr_var_loss(zs)
        l_ld = logdet_barrier_loss(zs, eps=cfg.logdet_eps)
        loss = l_rec + cfg.w_var * l_var + cfg.w_logdet * l_ld
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            ae.eval()
            with torch.no_grad():
                vr = [F.mse_loss(ae.decode(ae.encode(v))[:, m:Wc - m], v[:, m:Wc - m]).item()
                      for v in u_val_crops.split(cfg.batch)]
                val = sum(vr) / len(vr)
            hist.append({"step": step + 1, "train_recon": l_rec.item(), "l_var": l_var.item(),
                         "l_logdet": l_ld.item(), "val_recon_interior": val, "wall_s": time.time() - t_start})
            log(f"  [cropAE1d] step {step + 1}/{cfg.steps} recon {l_rec.item():.5f} val {val:.5f} "
                f"var {l_var.item():.4f} [{time.time() - t_start:.0f}s]")
            if val < best:
                best = val
                best_state = {k: v.detach().clone() for k, v in ae.state_dict().items()}
    ae.load_state_dict(best_state)
    return {"history": hist, "best_val_recon_interior": best}


@torch.no_grad()
def encode_crop_bank_1d(ae, u: torch.Tensor, n_offsets: int, crop_sites: int, enc_sites: int, seed: int,
                        batch: int = 64) -> torch.Tensor:
    """Encode `n_offsets` random windows of every trajectory at every time. Returns
    `(N*n_offsets, T, crop_sites - 2*enc_sites, 1, c)`: the 2D bank layout of
    `crop.encode_crop_bank` with one site in y, so `crop.train_crop_prop` trains on it unchanged."""
    ae.eval()
    acfg = ae.cfg
    patch = acfg.NX // acfg.n_sites
    N, T, NX = u.shape
    Wc = crop_sites * patch
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for n in range(N):
        for x0 in torch.randint(0, NX, (n_offsets,), generator=g).tolist():
            zs = []
            for t0 in range(0, T, batch):
                tt = torch.arange(t0, min(T, t0 + batch), device=u.device)
                B = len(tt)
                crops = gather_crops_1d(u, torch.full((B,), n, device=u.device), tt,
                                        torch.full((B,), x0, device=u.device), Wc)
                z = ae.encode(crops).reshape(B, acfg.n_sites, acfg.local_channels)
                zs.append(z[:, enc_sites:acfg.n_sites - enc_sites])
            out.append(torch.cat(zs))
    return torch.stack(out).unsqueeze(3)


def resized_ae_config(cfg, n_sites: int):
    """The same autoencoder at `n_sites` sites (patch size held fixed): every weight shape depends
    only on the patch size, channels, reach and width, never on the number of sites."""
    patch = cfg.NX // cfg.n_sites
    return dataclasses.replace(cfg, NX=n_sites * patch, n_sites=n_sites)


def resized_prop_config(cfg, n_sites: int, channels: int):
    """The same `local_mlp` propagator on `n_sites` sites (one token per site)."""
    if getattr(cfg, "backbone", None) != "local_mlp":
        raise ValueError(f"only backbone='local_mlp' is size-independent here, got {cfg.backbone!r}")
    return dataclasses.replace(cfg, d_latent=n_sites * channels, n_tokens=n_sites)
