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

import numpy as np
import torch
import torch.nn.functional as F

from ks_latent.cm1.data import VARS, ColumnData
from ks_latent.cm1.models import ColumnFieldAE, PCAColumnAE, StencilPropagator, _patchify, forced_halo_rollout
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
    nested_frac: float = 0.0   # FPc: share of samples decoded from only their first k channels (k uniform)


def _penalty(prop) -> torch.Tensor | float:
    """The propagator's own regularization (FPc law body), or 0 for propagators without one (e.g. the flow model)."""
    f = getattr(prop, "penalty", None)
    return f() if f is not None else 0.0


def nested_truncate(z: torch.Tensor, frac: float, gen: torch.Generator) -> torch.Tensor:
    """FPc nested dropout: for a share `frac` of the samples, keep only the first k latent channels (k uniform in
    1..c, zero the rest) before decoding, so channel 1 alone is the best 1-channel description, 1-2 the best pair,
    and so on: an importance-ordered latent. `z` (B, P, c); frac 0 returns z unchanged."""
    if frac <= 0:
        return z
    B, c = z.shape[0], z.shape[-1]
    k = torch.randint(1, c + 1, (B,), generator=gen)
    k = torch.where(torch.rand(B, generator=gen) < frac, k, torch.full_like(k, c))
    keep = (torch.arange(c)[None, :] < k[:, None]).to(z.device, z.dtype)
    return z * keep[:, None, :]


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
    # "max": every batch rolls out the current curriculum length; "uniform": each batch draws its
    # length uniformly in [2, current length], so short leads keep being trained (D3)
    k_sampling: str = "max"
    # > 0: a stochastic propagator (noise_channels > 0) is trained with the fair CRPS of this many samples
    # instead of MSE (D14c); 0 = MSE
    crps_samples: int = 0
    noise_start: float = 0.10
    noise_end: float = 0.02
    active_frac: float = 0.5
    val_every: int = 250
    seed: int = 0


def _interior(x: torch.Tensor, m: int, dim: int = -1) -> torch.Tensor:
    return x.narrow(dim, m, x.shape[dim] - 2 * m) if m > 0 else x


def site_blocks(data: ColumnData, patch: int, device, stride_t: int = 2, frames: int = 16, active_dbz: float = 10.0):
    """Yield `(x, active)`: site blocks `(N, C*patch)` of every member (every `stride_t`-th frame) and whether
    each block holds a storm column (composite reflectivity > `active_dbz`)."""
    for m in range(len(data.u)):
        ts = list(range(0, data.T[m], stride_t))
        for i in range(0, len(ts), frames):
            tt = ts[i:i + frames]
            u = data.full(m, torch.tensor(tt), device=device)                       # (F, C, NX)
            x = _patchify(u, patch)
            act = (data.cref[m][tt] > active_dbz).reshape(len(tt), x.shape[1], patch).any(-1)
            yield x.reshape(-1, x.shape[-1]), act.reshape(-1)


@torch.no_grad()
def fit_pca_stage(ae: PCAColumnAE, data_tr: ColumnData, data_va: ColumnData, log=print) -> dict:
    """Stage A for the PCA baseline: fit on training site blocks; report validation reconstruction MSE
    (normalized units, all columns and storm columns)."""
    dev = next(ae.parameters()).device
    t0 = time.time()
    info = ae.fit(site_blocks(data_tr, ae.cfg.patch, dev))
    se = se_a = 0.0; n = n_a = 0
    for x, act in site_blocks(data_va, ae.cfg.patch, dev):
        z = (x - ae.mean) @ ae.components.T / ae.scale
        xr = (z * ae.scale) @ ae.components + ae.mean
        C = ae.cfg.n_channels
        xr = xr.reshape(len(x), C, -1)
        xr = torch.where(ae.positive_mask[None, :, None], xr.clamp(min=0), xr).reshape(len(x), -1)
        e = ((xr - x) ** 2).mean(1)
        se += float(e.sum()); n += len(e)
        se_a += float(e[act.to(dev)].sum()); n_a += int(act.sum())
    out = {**info, "best_val_recon": se / n, "val_recon_storm": se_a / max(n_a, 1), "wall_s": time.time() - t0}
    log(f"[A] PCA-{ae.cfg.latent_channels} on {info['n_blocks']} site blocks ({info['n_active_blocks']} with storm "
        f"columns): explains {100 * info['explained_total']:.1f}% of the (storm-weighted) variance; validation "
        f"reconstruction MSE {out['best_val_recon']:.4f}, storm blocks {out['val_recon_storm']:.4f} [{out['wall_s']:.0f}s]")
    return out


def train_stage_a(ae: ColumnFieldAE, data_tr: ColumnData, data_va: ColumnData, cfg: StageAConfig,
                  log=print) -> dict:
    dev = next(ae.parameters()).device  # the model's device; data may live elsewhere
    p, reach = ae.cfg.patch, ae.cfg.reach_sites
    mask_sites = 2 * reach  # encoder + decoder reach
    W = cfg.window_sites * p
    if cfg.window_sites <= 2 * mask_sites:
        raise ValueError(f"window of {cfg.window_sites} sites leaves no interior after masking {mask_sites} per side")
    g = torch.Generator().manual_seed(cfg.seed)
    vg = torch.Generator().manual_seed(cfg.seed + 12345)
    def draw(data, gen):
        u, mem, _, _ = data.windows(cfg.batch, W, 1, gen, cfg.active_frac)
        e = data.env_of(mem)
        return u[:, 0].to(dev), (None if e is None else e.to(dev))
    val = [draw(data_va, vg) for _ in range(cfg.val_batches)]
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    m = mask_sites * p
    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        ae.train()
        u, env = draw(data_tr, g)                                                  # (B, C, W)
        z = ae.encode(u, env)
        uh = ae.decode(nested_truncate(z, cfg.nested_frac, g))
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
                v = sum(F.mse_loss(_interior(ae(x, e)[0], m), _interior(x, m)).item() for x, e in val) / len(val)
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
    dev = next(ae.parameters()).device
    out = []
    for m in range(len(data.u)):
        e = data.env_of([m])
        zs = []
        for t in range(0, data.T[m], batch):
            x = data.full(m, slice(t, t + batch), device=dev)
            zs.append(ae.encode(x, None if e is None else e.to(dev).expand(x.shape[0], -1)))
        out.append(torch.cat(zs))
    return out


def _k_now(frac: float, cfg: StageBConfig, gen: torch.Generator | None = None) -> int:
    k = min(cfg.k_max, int(round(2 + (cfg.k_max - 2) * frac / cfg.k_ramp_frac)))
    if cfg.k_sampling == "uniform":
        if gen is None:
            raise ValueError("k_sampling='uniform' needs a generator")
        return int(torch.randint(2, k + 1, (1,), generator=gen))
    if cfg.k_sampling != "max":
        raise ValueError(f"unknown k_sampling {cfg.k_sampling!r}")
    return k


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


def crps_loss(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fair (unbiased) ensemble CRPS, averaged over elements: E|X - y| - E|X - X'| / 2 with the
    M(M-1) off-diagonal pairs. `samples` (M, ...), `target` (...)."""
    M = samples.shape[0]
    if M < 2:
        raise ValueError("CRPS needs at least 2 samples")
    skill = (samples - target[None]).abs().mean(0)
    spread = (samples[:, None] - samples[None]).abs().sum(dim=(0, 1)) / (M * (M - 1))
    return (skill - 0.5 * spread).mean()


def crps_elements(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fair ensemble CRPS per element, `samples` (M, ...), `target` (...) -> (...). The spread term uses
    the sorted-sample identity sum_ij |x_i - x_j| = 2 sum_i (2i - M - 1) x_(i), so memory is O(M) rather
    than the O(M^2) of the pairwise form (`crps_loss`); the two agree exactly."""
    M = samples.shape[0]
    if M < 2:
        raise ValueError("CRPS needs at least 2 samples")
    skill = (samples - target[None]).abs().mean(0)
    xs = torch.sort(samples, dim=0).values
    w = (2 * torch.arange(1, M + 1, device=samples.device, dtype=samples.dtype) - M - 1)
    spread = 2 * (w.view(-1, *([1] * (samples.dim() - 1))) * xs).sum(0) / (M * (M - 1))
    return skill - 0.5 * spread


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fair energy score of a sample ensemble. `samples` (S, ..., C, W), `target` (..., C, W); each field is
    the last two dims (channels x columns), everything in between is batch.
    ES = mean_s ||X_s - y|| - 1/(2 S (S-1)) sum_{s != s'} ||X_s - X_s'||, with RMS-normalized field norms,
    averaged over the batch. Unlike the pointwise CRPS it rewards the right joint (spatial) structure."""
    S = samples.shape[0]
    if S < 2:
        raise ValueError("the energy score needs at least 2 samples")
    rms = lambda x: x.pow(2).mean(dim=(-2, -1)).add(1e-12).sqrt()
    skill = torch.stack([rms(samples[s] - target) for s in range(S)]).mean(0)
    spread = sum(rms(samples[s] - samples[r]) for s in range(S) for r in range(s + 1, S)) * 2 / (S * (S - 1))
    return (skill - 0.5 * spread).mean()


def spectral_loss(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared difference of log horizontal power spectra (rfft over the last dim, mean removed) between
    every sample and the target, per channel; averaged. Only meaningful on samples (on a deterministic
    forecast it trades accuracy for sharpness)."""
    ps = lambda x: torch.fft.rfft(x - x.mean(dim=-1, keepdim=True), dim=-1).abs().pow(2)
    lt = torch.log(ps(target) + 1e-6)
    return torch.stack([(torch.log(ps(samples[s]) + 1e-6) - lt).pow(2).mean() for s in range(samples.shape[0])]).mean()


def train_stage_b(prop: StencilPropagator, z_tr: list[torch.Tensor], z_va: list[torch.Tensor],
                  act_tr: list[torch.Tensor], act_va: list[torch.Tensor], patch: int, cfg: StageBConfig,
                  log=print) -> dict:
    if cfg.halo_sites < prop.cfg.reach_sites:
        raise ValueError(f"halo {cfg.halo_sites} < propagator reach {prop.cfg.reach_sites} sites per step")
    dev = z_tr[0].device
    Hh = prop.cfg.n_hist
    M = cfg.crps_samples
    if M and not prop.cfg.noise_channels:
        raise ValueError("crps_samples > 0 needs a stochastic propagator (noise_channels > 0)")
    g = torch.Generator().manual_seed(cfg.seed)
    vg = torch.Generator().manual_seed(cfg.seed + 54321)
    val = latent_windows(z_va, act_va, patch, 64, cfg.window_sites, cfg.k_max + Hh, vg, cfg.active_frac).to(dev)
    opt = torch.optim.AdamW(prop.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    h = cfg.halo_sites
    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        frac = step / cfg.steps
        k = _k_now(frac, cfg, g)
        noise = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * frac
        prop.train()
        win = latent_windows(z_tr, act_tr, patch, cfg.batch, cfg.window_sites, k + Hh, g, cfg.active_frac).to(dev)
        target = _interior(win[:, Hh:], h, dim=2)
        if M:
            pred = forced_halo_rollout(prop, win.repeat(M, 1, 1, 1), k, h, noise, gen=g)
            loss = crps_loss(_interior(pred, h, dim=2).reshape(M, *target.shape), target)
        else:
            pred = forced_halo_rollout(prop, win, k, h, noise, gen=g)
            loss = F.mse_loss(_interior(pred, h, dim=2), target)
        loss = loss + _penalty(prop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            prop.eval()
            with torch.no_grad():
                vgen = torch.Generator().manual_seed(7)
                if M:   # score the sample mean
                    pv = forced_halo_rollout(prop, val.repeat(M, 1, 1, 1), cfg.k_max, h, gen=vgen)
                    pv = pv.reshape(M, *val[:, Hh:].shape).mean(0)
                else:
                    pv = forced_halo_rollout(prop, val, cfg.k_max, h, gen=vgen)
                tv = _interior(val[:, Hh:], h, dim=2)
                err = ((_interior(pv, h, dim=2) - tv) ** 2).mean(dim=(0, 2, 3))
                pers = ((_interior(val[:, Hh - 1:Hh], h, dim=2) - tv) ** 2).mean(dim=(0, 2, 3))
            v = err.mean().item()
            rec = {"step": step + 1, "k": k, "train_loss": loss.item(), "val_mse_kmax": v,
                   "val_mse_by_lead": err.tolist(), "persistence_by_lead": pers.tolist(), "wall_s": time.time() - t0}
            hist.append(rec)
            log(f"  [B] step {step + 1}/{cfg.steps} k {k} loss {loss.item():.5f} val {v:.5f} "
                f"(persistence {pers.mean().item():.5f}) [{rec['wall_s']:.0f}s]")
            if not (np.isfinite(v) and np.isfinite(loss.item())):
                raise FloatingPointError(f"Stage B diverged at step {step + 1} (k {k}): train loss {loss.item()}, "
                                         f"val (k={cfg.k_max}) {v}; for a law propagator try --law-cap")
            if v < best:
                best, best_state = v, {kk: t.detach().clone() for kk, t in prop.state_dict().items()}
    prop.load_state_dict(best_state)
    return {"history": hist, "best_val_mse": best}


@dataclass
class StageCConfig:
    """Stage C (D12): joint fine-tuning of autoencoder and propagator on a decoded rollout loss."""
    window_sites: int = 48
    halo_sites: int = 3
    batch: int = 12
    steps: int = 4000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.05
    grad_clip: float = 1.0
    k_max: int = 20
    k_ramp_frac: float = 0.3
    k_sampling: str = "uniform"
    w_rec: float = 1.0          # reconstruction of the first frame
    w_var: float = 0.02
    w_logdet: float = 0.0035
    logdet_eps: float = 1e-3
    w_sfc_th: float = 0.0       # extra weight on theta' below `sfc_levels` levels (cold pool, G10)
    sfc_levels: int = 10        # ~ the lowest 1 km of the 62-level grid
    crps_samples: int = 0       # > 1: stochastic propagator, decoded-space fair CRPS over this many rollouts
    nested_frac: float = 0.0    # FPc: nested dropout on the reconstruction term (keeps the channel ordering)
    w_energy: float = 0.0       # SM2: + w * energy score of the decoded samples (spatial structure); needs samples
    w_spectral: float = 0.0     # SM3: + w * log-spectrum mismatch of each sample; needs samples
    # FPd (law propagator only): shape the latent so the explicit law closes
    law_l1: float = -1.0              # >= 0: override the law's L1 weight for this stage
    law_res_weight_end: float = -1.0  # >= 0: ramp the residual-share weight linearly to this value ...
    law_res_ramp_frac: float = 0.5    # ... over this fraction of the steps
    law_prune_every: int = 0          # > 0: sequential thresholding every this many steps ...
    law_prune_start_frac: float = 0.4 # ... from this fraction of the steps on ...
    law_prune_rel: float = 0.02       # ... zeroing terms below this fraction of the largest coupling
    w_site_smooth: float = 0.0        # + w * mean (second difference of z over sites)^2 / var(z), first frame
    freeze_ae: bool = False           # M3: train the propagator only (the autoencoder, hence the latent, stays fixed)
    active_frac: float = 0.5
    val_every: int = 250
    val_batches: int = 4
    seed: int = 0


def site_smoothness(z: torch.Tensor) -> torch.Tensor:
    """FPd: mean squared second difference of the latent over neighbouring sites, relative to the latent's
    variance (scale-free, so shrinking z does not lower it). `z` (B, P, c). Over physical space (sites), not
    over the channel index."""
    d2 = z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2]
    return (d2 ** 2).mean() / (z.var() + 1e-8)


def train_stage_c(ae: ColumnFieldAE, prop: StencilPropagator, data_tr: ColumnData, data_va: ColumnData,
                  cfg: StageCConfig, log=print, teacher: tuple | None = None) -> dict:
    """Windows of true columns over H + k frames are encoded, rolled k steps with the true (encoded) halo,
    decoded, and scored against the true columns on the window interior (halo + decoder reach excluded),
    plus reconstruction of the first frame and the latent anti-collapse terms. Both models are updated.
    With `crps_samples` S > 1 the (stochastic) propagator is rolled S times per window with independent
    noise and the decoded samples are scored with the fair CRPS instead of the MSE: an MSE would train
    the noise away, CRPS rewards a sharp, calibrated sample ensemble. Validation uses the same loss.
    With `teacher = (ae_t, prop_t)` (frozen, deterministic, not environment-conditioned) the targets are the
    teacher's: the decoded forced-halo rollout of the teacher from the same true windows, and the teacher's
    reconstruction of the first frame, instead of the true columns (5t: distillation of a pruned student)."""
    dev = next(ae.parameters()).device
    p, Hh = ae.cfg.patch, prop.cfg.n_hist
    h, Pw = cfg.halo_sites, cfg.window_sites
    if h < prop.cfg.reach_sites:
        raise ValueError(f"halo {h} < propagator reach {prop.cfg.reach_sites}")
    m_sites = h + ae.cfg.reach_sites
    if Pw <= 2 * m_sites + 2:
        raise ValueError("window too narrow for the halo and decoder reach")
    W, m = Pw * p, m_sites * p
    C, Zl = data_tr.C, data_tr.Z
    Ms = cfg.crps_samples
    if (cfg.w_energy or cfg.w_spectral) and Ms < 2:
        raise ValueError("w_energy / w_spectral need crps_samples >= 2 (they score samples)")
    if Ms and (Ms < 2 or not prop.cfg.noise_channels):
        raise ValueError("Stage C CRPS needs crps_samples >= 2 and a stochastic propagator (noise_channels > 0)")
    if teacher is not None:
        if Ms or prop.cfg.noise_channels:
            raise ValueError("distillation is for deterministic propagators")
        ae_t, prop_t = teacher
        ae_t.eval(); prop_t.eval()
        for q in list(ae_t.parameters()) + list(prop_t.parameters()):
            q.requires_grad_(False)
    wch = torch.ones(C, device=dev)
    if cfg.w_sfc_th > 0:
        ith = VARS.index("th") * Zl
        wch[ith:ith + cfg.sfc_levels] += cfg.w_sfc_th
    wch = wch / wch.mean()
    g = torch.Generator().manual_seed(cfg.seed)
    vg = torch.Generator().manual_seed(cfg.seed + 999)
    def draw(data, F_, gen):
        u, mem, _, _ = data.windows(cfg.batch, W, F_, gen, cfg.active_frac)
        e = data.env_of(mem)
        return u.to(dev), (None if e is None else e.to(dev))
    val = [draw(data_va, Hh + cfg.k_max, vg) for _ in range(cfg.val_batches)]
    if cfg.freeze_ae:
        ae.requires_grad_(False)
    params = ([] if cfg.freeze_ae else list(ae.parameters())) + list(prop.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    kcfg = StageBConfig(k_max=cfg.k_max, k_ramp_frac=cfg.k_ramp_frac, k_sampling=cfg.k_sampling)
    is_law = getattr(prop.cfg, "body", None) == "law"
    fpd = cfg.law_l1 >= 0 or cfg.law_res_weight_end >= 0 or cfg.law_prune_every > 0
    if fpd and not is_law:
        raise ValueError("law_l1 / law_res_weight_end / law_prune_every need a law propagator (body 'law')")
    if cfg.law_l1 >= 0:
        prop.cfg.law_l1 = cfg.law_l1
    res0 = prop.cfg.law_res_weight if is_law else 0.0
    prune_start = int(cfg.law_prune_start_frac * cfg.steps)

    def rollout_err(u, env, k, weighted, gen):
        B, F_ = u.shape[:2]
        e = None if env is None else env.repeat_interleave(F_, dim=0)
        z = ae.encode(u.reshape(B * F_, C, W), e).reshape(B, F_, Pw, -1)
        S = Ms or 1
        zr = z.repeat(S, 1, 1, 1) if S > 1 else z                                  # (S*B, F, Pw, c), sample-major
        pred = forced_halo_rollout(prop, zr, k, h, gen=gen)                      # (S*B, k, Pw, c)
        dec = ae.decode(pred.reshape(S * B * k, Pw, -1)).reshape(S, B, k, C, W)[..., m:W - m]
        tgt = u[:, Hh:Hh + k][..., m:W - m]
        if teacher is not None:
            if env is not None:
                raise ValueError("distillation does not support environment conditioning")
            with torch.no_grad():
                zt = ae_t.encode(u.reshape(B * F_, C, W)).reshape(B, F_, Pw, -1)
                pt = forced_halo_rollout(prop_t, zt, k, h)
                tgt = ae_t.decode(pt.reshape(B * k, Pw, -1)).reshape(B, k, C, W)[..., m:W - m]
        e = crps_elements(dec, tgt) if S > 1 else (dec[0] - tgt) ** 2             # CRPS (stochastic) or MSE
        if weighted:
            e = e * wch[None, None, :, None]
        loss = e.mean()
        if S > 1 and cfg.w_energy:
            loss = loss + cfg.w_energy * energy_score(dec, tgt)
        if S > 1 and cfg.w_spectral:
            loss = loss + cfg.w_spectral * spectral_loss(dec, tgt)
        return loss, z

    hist, best, best_state, t0 = [], float("inf"), None, time.time()
    for step in range(cfg.steps):
        ae.train(); prop.train()
        k = _k_now(step / cfg.steps, kcfg, g)
        if cfg.law_res_weight_end >= 0:
            fr = min(1.0, step / max(1, cfg.law_res_ramp_frac * cfg.steps))
            prop.cfg.law_res_weight = res0 + (cfg.law_res_weight_end - res0) * fr
        u, env = draw(data_tr, Hh + k, g)                                          # (B, H+k, C, W)
        l_roll, z = rollout_err(u, env, k, True, g)
        u0 = u[:, Hh - 1]
        if teacher is not None:
            with torch.no_grad():
                u0 = ae_t.decode(ae_t.encode(u0))
        l_rec = F.mse_loss(_interior(ae.decode(nested_truncate(z[:, Hh - 1], cfg.nested_frac, g)), m), _interior(u0, m))
        zs = _interior(z[:, Hh - 1], m_sites, dim=1).reshape(-1, z.shape[-1])
        if ae.cfg.prefix_m:
            zs = zs[:, :ae.cfg.prefix_m]                                          # the pruned (constant) channels carry no variance
        _, l_var = decorr_var_loss(zs)
        l_ld = logdet_barrier_loss(zs, eps=cfg.logdet_eps)
        loss = l_roll + cfg.w_rec * l_rec + cfg.w_var * l_var + cfg.w_logdet * l_ld + _penalty(prop)
        if cfg.w_site_smooth:
            loss = loss + cfg.w_site_smooth * site_smoothness(_interior(z[:, Hh - 1], m_sites, dim=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sched.step()
        if cfg.law_prune_every and step + 1 >= prune_start and (step + 1 - prune_start) % cfg.law_prune_every == 0:
            thr = cfg.law_prune_rel * prop.law.max_coupling()
            kept = prop.law.prune(thr)
            best, best_state = float("inf"), None     # checkpoints from before a pruning are not eligible
            log(f"  [C] step {step + 1}: pruned below {thr:.2e} ({cfg.law_prune_rel:g} x largest coupling): "
                f"{kept} terms kept, residual weight {prop.cfg.law_res_weight:.3g}")
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            ae.eval(); prop.eval()
            with torch.no_grad():
                v = sum(rollout_err(x, e, cfg.k_max, False, torch.Generator().manual_seed(3))[0].item()
                        for x, e in val) / len(val)
            rec = {"step": step + 1, "k": k, "l_roll": l_roll.item(), "l_rec": l_rec.item(), "l_logdet": l_ld.item(),
                   "val_decoded_rollout_mse": v, "wall_s": time.time() - t0}
            if not (np.isfinite(v) and np.isfinite(rec["l_roll"])):
                raise FloatingPointError(f"Stage C diverged at step {step + 1} (k {k}): rollout loss {rec['l_roll']}, "
                                         f"val (k={cfg.k_max}) {v}")
            if is_law:
                rec.update(law_terms=prop.law.n_terms(), law_res_weight=prop.cfg.law_res_weight)
            hist.append(rec)
            log(f"  [C] step {step + 1}/{cfg.steps} k {k} roll {rec['l_roll']:.4f} rec {rec['l_rec']:.4f} "
                f"val (decoded, k={cfg.k_max}) {v:.4f} [{rec['wall_s']:.0f}s]")
            if v < best:
                best = v
                best_state = ({kk: t.detach().clone() for kk, t in ae.state_dict().items()},
                              {kk: t.detach().clone() for kk, t in prop.state_dict().items()})
    ae.load_state_dict(best_state[0]); prop.load_state_dict(best_state[1])
    return {"history": hist, "best_val_decoded_rollout_mse": best}

