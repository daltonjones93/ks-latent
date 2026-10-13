"""Forced-halo training of the memory-channel propagator (Z10a; ks_latent/models/memory_prop.py).

A training window holds `W + 1 + k` encoded states. Warm-up: W steps with every site's encoder channels forced to the
data while memory evolves from 0 (gradients flow through it: truncated BPTT over W + k steps). Rollout: from the data
state at time W (+ input noise), k steps with the halo's encoder channels overwritten by data before every step, the
free interior predicted; memory is never forced (it has no data). Loss on the interior encoder channels only. With
option (i) memory, a window that is the whole periodic training box (the KC2 geometry: 8-site box, halo 3) or that has
one extra reach of forced data beyond the halo reproduces the large-domain forced reference exactly.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ks_latent.training.crop import CropPropConfig, _k_now, warmup_cosine


def forced_halo_rollout_memory(prop, z_win: torch.Tensor, W: int, k: int, halo: int, noise: float = 0.0,
                               halo_y: int = 0) -> torch.Tensor:
    """`z_win`: `(B, W + 1 + k, Pu, Pv, c)`. Returns the predicted free interior `(B, k, Pu-2h, Pv-2h_y, c)`
    for targets `z_win[:, W+1:]`."""
    B, T, Pu, Pv, c = z_win.shape
    if T != W + 1 + k:
        raise ValueError(f"z_win has {T} states, expected W + 1 + k = {W + 1 + k}")
    mem = z_win.new_zeros(B, Pu, Pv, prop.cfg.memory_channels)
    for t in range(W):
        _, mem = prop.step_zm(z_win[:, t], mem)
    state = z_win[:, W].clone()
    if noise > 0:
        state = state + noise * torch.randn_like(state)
    ix, iy = slice(halo, Pu - halo), slice(halo_y, Pv - halo_y)
    preds = []
    for j in range(k):
        inp = z_win[:, W + j].clone()
        inp[:, ix, iy] = state[:, ix, iy]
        state, mem = prop.step_zm(inp, mem)
        preds.append(state[:, ix, iy])
    return torch.stack(preds, 1)


def train_crop_prop_memory(prop, bank_train: torch.Tensor, bank_val: torch.Tensor, cfg: CropPropConfig, W: int,
                           log=print) -> dict:
    """`train_crop_prop`'s schedule (AdamW, warmup-cosine LR, k curriculum 2 -> k_mid -> k_max, input noise
    annealed, best-val checkpointing) for the memory propagator."""
    dev = bank_train.device
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    opt = torch.optim.AdamW(prop.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    h, hy = cfg.halo_sites, cfg.halo_sites_y

    def windows(bank, B, k, gen):
        ci = torch.randint(0, bank.shape[0], (B,), generator=gen).to(dev)
        t0 = torch.randint(0, bank.shape[1] - (W + 1 + k) + 1, (B,), generator=gen).to(dev)
        tt = t0[:, None] + torch.arange(W + 1 + k, device=dev)[None, :]
        return bank[ci[:, None], tt]

    val_win = windows(bank_val, 128, cfg.k_max, torch.Generator(device="cpu").manual_seed(54321))
    tgt = lambda w: w[:, W + 1:, h:w.shape[2] - h, hy:w.shape[3] - hy]
    hist, best, best_state, t_start = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        frac = step / cfg.steps
        k = _k_now(frac, cfg)
        noise = cfg.noise_in_start + (cfg.noise_in_end - cfg.noise_in_start) * frac
        prop.train()
        win = windows(bank_train, cfg.batch, k, g)
        loss = F.mse_loss(forced_halo_rollout_memory(prop, win, W, k, h, noise, hy), tgt(win))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            prop.eval()
            with torch.no_grad():
                val = F.mse_loss(forced_halo_rollout_memory(prop, val_win, W, cfg.k_max, h, halo_y=hy),
                                 tgt(val_win)).item()
                lam = torch.sigmoid(prop.m_leak).tolist() if prop.cfg.memory_channels else []
            hist.append({"step": step + 1, "k": k, "train_loss": loss.item(), "val_kmax_mse": val,
                         "memory_leak": lam, "wall_s": time.time() - t_start})
            log(f"  [memProp] step {step + 1}/{cfg.steps} k {k} loss {loss.item():.5f} val_kmax {val:.5f} "
                f"leak {[round(x, 3) for x in lam]} [{time.time() - t_start:.0f}s]")
            if val < best:
                best = val
                best_state = {kk: v.detach().clone() for kk, v in prop.state_dict().items()}
    prop.load_state_dict(best_state)
    return {"history": hist, "best_val_kmax_mse": best}
