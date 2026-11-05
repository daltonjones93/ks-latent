"""Training for the CM1 column local field: Stage A (autoencoder) then Stage B (propagator).

Stage A trains on random x-windows of single frames. Sites within the
autoencoder's reach (encoder + decoder) of a window edge see padding rather
than real neighbours, so they are excluded from the reconstruction loss and
from the latent regularizers; only the interior counts, as in the 1D/2D crop
training of Papers I-II (`training/crop1d.py`). Anti-collapse: the per-site
channel covariance is pushed toward unit variance (`decorr_var_loss`) and
away from singularity (`logdet_barrier_loss`, the regularizer that resolved
channel collapse in the KS local field).

Stage B: the frozen encoder encodes every member's whole domain once, giving
latent trajectories `(T, P, c)`. The propagator is trained on random windows
of these with a forced halo (`models.forced_halo_rollout`): the halo sites
are reset to the true latent state before every step, so the window edges
behave like the lateral boundary conditions of a limited-area model. Rollout
length follows a curriculum 2 -> k_max, input noise is annealed, the loss is
latent MSE on the free interior. Both stages always run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ks_latent.cm1.data import ColumnData
from ks_latent.cm1.models import ColumnFieldAE, StencilPropagator, forced_halo_rollout
from ks_latent.training.crop import warmup_cosine
from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss


@dataclass
class StageAConfig:
    window_sites: int = 32  # window width in latent sites (x patch cells)
    batch: int = 32
    steps: int = 6000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.02
    grad_clip: float = 1.0
    w_var: float = 0.02
    w_logdet: float = 0.0035
    logdet_eps: float = 1e-3
    active_frac: float = 0.5
    val_every: int = 250
    val_batches: int = 8
    seed: int = 0


@dataclass
class StageBConfig:
    window_sites: int = 48
    halo_sites: int = 6  # >= propagator reach per step; reset to the truth every step
    batch: int = 32
    steps: int = 6000
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.02
    grad_clip: float = 1.0
    k_max: int = 8
    k_ramp_frac: float = 0.6  # k grows 2 -> k_max over this fraction of training
    noise_start: float = 0.10
    noise_end: float = 0.02
    active_frac: float = 0.5
    val_every: int = 250
    seed: int = 0


def _interior(x: torch.Tensor, m: int, dim: int = -1) -> torch.Tensor:
    return x.narrow(dim, m, x.shape[dim] - 2 * m) if m > 0 else x


def train_stage_a(ae: ColumnFieldAE, data_tr: ColumnData, data_va: ColumnData, cfg: StageAConfig,
                  log=print) -> dict:
    dev = data_tr.device
    p, reach = ae.cfg.patch, ae.cfg.reach_sites
    mask_sites = 2 * reach  # encoder + decoder reach
    W = cfg.window_sites * p
    if cfg.window_sites <= 2 * mask_sites:
        raise ValueError(f"window of {cfg.window_sites} sites leaves no interior after masking {mask_sites} per side")
    g = torch.Generator().manual_seed(cfg.seed)
    vg = torch.Generator().manual_seed(cfg.seed + 12345)
    val = [data_va.windows(cfg.batch, W, 1, vg, cfg.active_frac)[0][:, 0].to(dev) for _ in range(cfg.val_batches)]
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    m = mask_sites * p
    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        ae.train()
        u = data_tr.windows(cfg.batch, W, 1, g, cfg.active_frac)[0][:, 0].to(dev)  # (B, C, W)
        z = ae.encode(u)
        uh = ae.decode(z)
        l_rec = F.mse_loss(_interior(uh, m), _interior(u, m))
        zs = _interior(z, mask_sites, dim=1).reshape(-1, z.shape[-1])
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
                v = sum(F.mse_loss(_interior(ae(x)[0], m), _interior(x, m)).item() for x in val) / len(val)
            rec = {"step": step + 1, "train_recon": l_rec.item(), "l_var": l_var.item(), "l_logdet": l_ld.item(),
                   "val_recon": v, "wall_s": time.time() - t0}
            hist.append(rec)
            log(f"  [A] step {step + 1}/{cfg.steps} recon {rec['train_recon']:.5f} val {v:.5f} "
                f"var {rec['l_var']:.3f} logdet {rec['l_logdet']:.3f} [{rec['wall_s']:.0f}s]")
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in ae.state_dict().items()}
    ae.load_state_dict(best_state)
    return {"history": hist, "best_val_recon": best}


@torch.no_grad()
def encode_members(ae: ColumnFieldAE, data: ColumnData, batch: int = 16) -> list[torch.Tensor]:
    """Whole-domain latent trajectories, one `(T, P, c)` tensor per member."""
    ae.eval()
    out = []
    for m in range(len(data.u)):
        zs = [ae.encode(data.full(m, slice(t, t + batch))) for t in range(0, data.T[m], batch)]
        out.append(torch.cat(zs))
    return out


def _k_now(frac: float, cfg: StageBConfig) -> int:
    return min(cfg.k_max, int(round(2 + (cfg.k_max - 2) * frac / cfg.k_ramp_frac)))


def latent_windows(zs: list[torch.Tensor], active: list[torch.Tensor], patch: int, B: int, width: int,
                   n_frames: int, gen: torch.Generator, active_frac: float) -> torch.Tensor:
    """Random `(B, n_frames, width, c)` windows of latent trajectories; storm-centred with prob. active_frac."""
    out = []
    for _ in range(B):
        m = int(torch.randint(0, len(zs), (1,), generator=gen))
        T, P = zs[m].shape[:2]
        act = active[m]
        act = act[act[:, 0] < T - n_frames + 1] if len(act) else act
        if len(act) and float(torch.rand(1, generator=gen)) < active_frac:
            t, xc = act[torch.randint(0, len(act), (1,), generator=gen)][0].tolist()
            x0 = min(max(xc // patch - width // 2, 0), P - width)
        else:
            t = int(torch.randint(0, T - n_frames + 1, (1,), generator=gen))
            x0 = int(torch.randint(0, P - width + 1, (1,), generator=gen))
        out.append(zs[m][t:t + n_frames, x0:x0 + width])
    return torch.stack(out)


def train_stage_b(prop: StencilPropagator, z_tr: list[torch.Tensor], z_va: list[torch.Tensor],
                  act_tr: list[torch.Tensor], act_va: list[torch.Tensor], patch: int, cfg: StageBConfig,
                  log=print) -> dict:
    if cfg.halo_sites < prop.cfg.reach_sites:
        raise ValueError(f"halo {cfg.halo_sites} < propagator reach {prop.cfg.reach_sites} sites per step")
    dev = z_tr[0].device
    g = torch.Generator().manual_seed(cfg.seed)
    vg = torch.Generator().manual_seed(cfg.seed + 54321)
    val = latent_windows(z_va, act_va, patch, 64, cfg.window_sites, cfg.k_max + 1, vg, cfg.active_frac).to(dev)
    opt = torch.optim.AdamW(prop.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    h = cfg.halo_sites
    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        frac = step / cfg.steps
        k = _k_now(frac, cfg)
        noise = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * frac
        prop.train()
        win = latent_windows(z_tr, act_tr, patch, cfg.batch, cfg.window_sites, k + 1, g, cfg.active_frac).to(dev)
        pred = forced_halo_rollout(prop, win, k, h, noise)
        loss = F.mse_loss(_interior(pred, h, dim=2), _interior(win[:, 1:], h, dim=2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            prop.eval()
            with torch.no_grad():
                pv = forced_halo_rollout(prop, val, cfg.k_max, h)
                err = ((_interior(pv, h, dim=2) - _interior(val[:, 1:], h, dim=2)) ** 2).mean(dim=(0, 2, 3))
                pers = ((_interior(val[:, :1], h, dim=2) - _interior(val[:, 1:], h, dim=2)) ** 2).mean(dim=(0, 2, 3))
            v = err.mean().item()
            rec = {"step": step + 1, "k": k, "train_loss": loss.item(), "val_mse_kmax": v,
                   "val_mse_by_lead": err.tolist(), "persistence_by_lead": pers.tolist(), "wall_s": time.time() - t0}
            hist.append(rec)
            log(f"  [B] step {step + 1}/{cfg.steps} k {k} loss {loss.item():.5f} val {v:.5f} "
                f"(persistence {pers.mean().item():.5f}) [{rec['wall_s']:.0f}s]")
            if v < best:
                best, best_state = v, {kk: t.detach().clone() for kk, t in prop.state_dict().items()}
    prop.load_state_dict(best_state)
    return {"history": hist, "best_val_mse": best}
