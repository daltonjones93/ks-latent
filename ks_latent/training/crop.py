"""Crop training on a wide periodic domain (docs/ks_local_field_ltransfer.tex
Sec. crop_training, variant B).

Every learned layer in `KSAutoencoderLocalField2D` and the `local_mlp_2d`
propagator is a weight-shared local operator, so a model can be fit on
x-crops of a wide domain and deployed at any size via the existing resize
loaders. The crop's own circular padding is wrong physics at the crop edge;
instead of changing the architecture, the edge region is excluded:

- Stage A (autoencoder): reconstruction loss and latent regularizers are
  computed only on the crop interior, `mask_sites` sites in from each x-edge
  (the encoder+decoder receptive field), so padding at the crop edge cannot
  reach the loss.
- Stage B (propagator, forced lateral boundaries): crops are encoded once by
  the frozen autoencoder; the outer `enc_sites` sites (corrupted by padding)
  are dropped. In the remaining strip, a halo of `halo_sites` (>= the
  propagator's per-step receptive field) on each x-side is overwritten with
  the true encoded state at every rollout step, the interior runs free, and
  only the interior is scored -- the limited-area-model training setup.

y is the full periodic Ly throughout (Kolmogorov forcing depends on y).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ks_latent.training.losses import decorr_var_loss, logdet_barrier_loss


def gather_x_crops(u: torch.Tensor, traj: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, width: int) -> torch.Tensor:
    """`u`: `(N, T, NX, NY)`; `traj`, `x0`: `(B,)`; `t`: `(B,)` or `(B, K)`.
    Returns `(B, NXc, NY)` or `(B, K, NXc, NY)` x-windows, wrapping mod NX
    (the wide domain itself is periodic, so a crop across its seam is a
    physically valid window)."""
    NX = u.shape[2]
    xi = (x0[:, None] + torch.arange(width, device=u.device)[None, :]) % NX  # (B, W)
    if t.dim() == 1:
        frames = u[traj, t]  # (B, NX, NY)
        return torch.gather(frames, 1, xi[:, :, None].expand(-1, -1, u.shape[3]))
    K = t.shape[1]
    frames = u[traj[:, None], t]  # (B, K, NX, NY)
    idx = xi[:, None, :, None].expand(-1, K, -1, u.shape[3])
    return torch.gather(frames, 2, idx)


def gather_xy_crops(u: torch.Tensor, traj: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor,
                    wx: int, wy: int) -> torch.Tensor:
    """2D windows (P7: isotropic systems). `u`: `(N, T, NX, NY)`; `traj`, `x0`,
    `y0`: `(B,)`; `t`: `(B,)` or `(B, K)`. Returns `(B, wx, wy)` or
    `(B, K, wx, wy)`, wrapping both axes of the periodic wide domain."""
    NX, NY = u.shape[2], u.shape[3]
    B = traj.shape[0]
    xi = (x0[:, None] + torch.arange(wx, device=u.device)[None, :]) % NX
    yi = (y0[:, None] + torch.arange(wy, device=u.device)[None, :]) % NY
    b = torch.arange(B, device=u.device)
    if t.dim() == 1:
        frames = u[traj, t]  # (B, NX, NY)
        return frames[b[:, None, None], xi[:, :, None], yi[:, None, :]]
    K = t.shape[1]
    frames = u[traj[:, None], t]  # (B, K, NX, NY)
    k = torch.arange(K, device=u.device)
    return frames[b[:, None, None, None], k[None, :, None, None], xi[:, None, :, None], yi[:, None, None, :]]


def warmup_cosine(step: int, total: int, warmup: int, min_factor: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * prog))


@dataclass
class CropAEConfig:
    crop_sites: int = 48
    crop_sites_y: int | None = None  # P7: also crop (and mask) in y; None = full periodic y
    y_align: int = 1  # P9: y-offsets restricted to multiples of this many grid points (forcing period)
    mask_sites: int = 12  # encoder (6) + decoder (6) site receptive field
    enc_sites: int = 6
    batch: int = 32
    steps: int = 8000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.02
    grad_clip: float = 1.0
    w_var: float = 0.02
    w_logdet: float = 0.0035
    # logdet_barrier_loss's own eps (default 1e-3, additive/safe). User-directed
    # 2026-10-07: tested eps=0 (bare logdet, singularity exactly at zero, not
    # additively stabilized) after confirming empirically that Cov(z)'s smallest
    # eigenvalue at initialization for this architecture is positive but small
    # (0.0008-0.0044 across 10 seeds, every seed tested) -- eps=0 is safe at init
    # (no seed had a non-positive eigenvalue), but a MEANINGFULLY negative eps
    # (a genuine hard floor, e.g. -0.01) would be infeasible at step 1 for every
    # seed tested, since the natural initial scale is well below 0.01.
    logdet_eps: float = 1e-3
    # Gradient-matching (H1) reconstruction term, 2026-10-08 accuracy study: MSE between grid
    # finite differences of the reconstruction and of the target, in x and y, on the interior.
    # Weights error by ~|k|^2, so patch-edge seams (jumps between independently decoded patches)
    # cost far more than smooth error. 0 = off (original recipe).
    w_grad: float = 0.0
    val_every: int = 250
    seed: int = 0


def train_crop_ae(ae, u_train: torch.Tensor, u_val: torch.Tensor, cfg: CropAEConfig, log=print) -> dict:
    """`u_*`: normalized wide fields `(N, T, NX, NY)` on the training device.
    With `cfg.crop_sites_y` set, crops and masks are 2D; otherwise y is the
    full periodic width (Kolmogorov)."""
    dev = u_train.device
    acfg = ae.cfg
    patch, patch_y = acfg.patch_x, acfg.patch_y
    Wc = cfg.crop_sites * patch
    crop2d = cfg.crop_sites_y is not None
    Wcy = cfg.crop_sites_y * patch_y if crop2d else u_train.shape[3]
    assert acfg.NX == Wc and acfg.n_sites_x == cfg.crop_sites, "AE must be built at crop size"
    assert acfg.NY == Wcy, "AE must be built at crop size in y"
    m = cfg.mask_sites * patch
    my = cfg.mask_sites * patch_y if crop2d else 0
    ey = cfg.enc_sites if crop2d else 0
    c = acfg.local_channels
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    opt = torch.optim.AdamW(ae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))

    def sample(u, B, gen):
        N, T, NX, NY = u.shape
        tr = torch.randint(0, N, (B,), generator=gen).to(dev)
        t = torch.randint(0, T, (B,), generator=gen).to(dev)
        x0 = torch.randint(0, NX, (B,), generator=gen).to(dev)
        if crop2d:
            y0 = (torch.randint(0, NY // cfg.y_align, (B,), generator=gen) * cfg.y_align).to(dev)
            return gather_xy_crops(u, tr, t, x0, y0, Wc, Wcy)
        return gather_x_crops(u, tr, t, x0, Wc)

    def interior(x):
        return x[:, m:Wc - m, my:Wcy - my]

    vg = torch.Generator(device="cpu").manual_seed(12345)
    u_val_crops = torch.cat([sample(u_val, cfg.batch, vg) for _ in range(8)])
    hist, best, best_state = [], float("inf"), None
    t_start = time.time()
    for step in range(cfg.steps):
        ae.train()
        uc = sample(u_train, cfg.batch, g)  # (B, Wc, Wcy)
        z = ae.encode(uc.reshape(cfg.batch, -1))
        uh = ae.decode(z).reshape(uc.shape)
        l_rec = F.mse_loss(interior(uh), interior(uc))
        if cfg.w_grad > 0:
            eh, ec = interior(uh), interior(uc)
            l_grad = (F.mse_loss(torch.diff(eh, dim=1), torch.diff(ec, dim=1))
                      + F.mse_loss(torch.diff(eh, dim=2), torch.diff(ec, dim=2)))
        else:
            l_grad = torch.zeros((), device=dev)
        zs = z.reshape(cfg.batch, acfg.n_sites_x, acfg.n_sites_y, c)
        zs = zs[:, cfg.enc_sites:acfg.n_sites_x - cfg.enc_sites, ey:acfg.n_sites_y - ey].reshape(-1, c)
        _, l_var = decorr_var_loss(zs)
        l_ld = logdet_barrier_loss(zs, eps=cfg.logdet_eps)
        loss = l_rec + cfg.w_var * l_var + cfg.w_logdet * l_ld + cfg.w_grad * l_grad
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            ae.eval()
            with torch.no_grad():
                vr = []
                for i in range(0, len(u_val_crops), cfg.batch):
                    vc = u_val_crops[i:i + cfg.batch]
                    vh = ae.decode(ae.encode(vc.reshape(len(vc), -1))).reshape(vc.shape)
                    vr.append(F.mse_loss(interior(vh), interior(vc)).item())
                val = sum(vr) / len(vr)
            hist.append({"step": step + 1, "train_recon": l_rec.item(), "l_var": l_var.item(),
                         "l_logdet": l_ld.item(), "val_recon_interior": val,
                         "wall_s": time.time() - t_start})
            log(f"  [cropAE] step {step + 1}/{cfg.steps} recon {l_rec.item():.5f} val {val:.5f} "
                f"var {l_var.item():.4f} [{time.time() - t_start:.0f}s]")
            if val < best:
                best = val
                best_state = {k: v.detach().clone() for k, v in ae.state_dict().items()}
    ae.load_state_dict(best_state)
    return {"history": hist, "best_val_recon_interior": best}


@torch.no_grad()
def encode_crop_bank(ae, u: torch.Tensor, n_offsets: int, cfg_crop_sites: int, enc_sites: int,
                     seed: int, batch: int = 64, crop_sites_y: int | None = None,
                     y_align: int = 1) -> torch.Tensor:
    """Encode `n_offsets` random crops of every trajectory at every time.
    Returns `(N*n_offsets, T, P_usable_x, P_usable_y, c)` with the outer
    `enc_sites` (padding-corrupted) sites dropped in x, and in y too when
    `crop_sites_y` is set (2D crops)."""
    ae.eval()
    acfg = ae.cfg
    N, T, NX, NY = u.shape
    Wc = cfg_crop_sites * acfg.patch_x
    crop2d = crop_sites_y is not None
    Wcy = crop_sites_y * acfg.patch_y if crop2d else NY
    ey = enc_sites if crop2d else 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for n in range(N):
        offs = torch.randint(0, NX, (n_offsets,), generator=g).tolist()
        offs_y = ((torch.randint(0, NY // y_align, (n_offsets,), generator=g) * y_align).tolist()
                  if crop2d else [0] * n_offsets)
        for x0, y0 in zip(offs, offs_y):
            zs = []
            for t0 in range(0, T, batch):
                tt = torch.arange(t0, min(T, t0 + batch), device=u.device)
                B = len(tt)
                nn_ = torch.full((B,), n, device=u.device)
                xx = torch.full((B,), x0, device=u.device)
                if crop2d:
                    crops = gather_xy_crops(u, nn_, tt, xx, torch.full((B,), y0, device=u.device), Wc, Wcy)
                else:
                    crops = gather_x_crops(u, nn_, tt, xx, Wc)
                z = ae.encode(crops.reshape(B, -1)).reshape(B, acfg.n_sites_x, acfg.n_sites_y, acfg.local_channels)
                zs.append(z[:, enc_sites:acfg.n_sites_x - enc_sites, ey:acfg.n_sites_y - ey])
            out.append(torch.cat(zs))
    return torch.stack(out)


@dataclass
class CropPropConfig:
    halo_sites: int = 6  # >= propagator per-step receptive field (n_layers * window)
    halo_sites_y: int = 0  # P7 (2D crops): forced halo in y as well
    batch: int = 64
    steps: int = 6000
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup: int = 200
    lr_min_factor: float = 0.02
    grad_clip: float = 1.0
    k_max: int = 12
    k_mid: int = 8
    k_mid_frac: float = 175 / 300  # matches Stage 2's k_mid_epochs/epochs
    k_warmup_frac: float = 210 / 300  # matches Stage 2's k_warmup_epochs/epochs
    noise_in_start: float = 0.10
    noise_in_end: float = 0.02
    val_every: int = 250
    seed: int = 0
    w_rank: float = 0.0  # C7: one-sided per-site log-det penalty against rank collapse
    # Spectral-shares regularizer (2026-10-05, docs/ks_local_field_ltransfer.tex
    # sec:propagator_lessons item 3): penalizes the ROLLED-OUT field's kinetic-energy
    # distribution across |k| bands for drifting from the true data's own distribution
    # -- a direct, differentiable version of the diagnostic
    # (exp_rollout_stats_kolmogorov.py) that caught P9's and P8's free-rollout failures,
    # rather than relying on k_max alone (a blunt proxy) to prevent that drift. 0 = off
    # (all existing recipes unaffected). Requires `ae` and `site_len` at call time; 2D
    # crops only (needs a real (NX,NY) physical field, decode()'s 1D-flat convention is
    # reshaped internally using ae.cfg).
    w_spectral: float = 0.0
    spectral_bands: tuple = ((0.0, 2.0), (3.0, 5.0))  # |k| band edges, physical units
    spectral_every: int = 1  # apply every N steps (cost control; the decode+FFT adds work)


def spectral_band_shares(field: torch.Tensor, Lx: float, Ly: float,
                         bands: "tuple[tuple[float, float], ...]") -> torch.Tensor:
    """`field`: `(..., NX, NY)` real physical field (any number of leading
    batch/time dims). Returns `(len(bands),)`: the share of total
    Poisson-solve kinetic energy (`|FFT(field)|^2 / k^2`, the same
    quantity `exp_rollout_stats_kolmogorov.py`'s `field_stats` computes)
    falling in each `|k|` band, averaged over every leading dim. Fully
    differentiable (plain `torch.fft`, no `.item()`/numpy), so it can be
    used as a training-time loss term, not just a post-hoc diagnostic.
    The `k=0` (mean) mode is excluded from both the band sums and the
    total, matching the diagnostic script."""
    *lead, NX, NY = field.shape
    kx = torch.fft.fftfreq(NX, d=Lx / NX, device=field.device, dtype=field.dtype) * (2 * math.pi)
    ky = torch.fft.fftfreq(NY, d=Ly / NY, device=field.device, dtype=field.dtype) * (2 * math.pi)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    K = torch.sqrt(K2)
    nonzero = K2 > 0
    K2safe = torch.where(nonzero, K2, torch.ones_like(K2))
    P = (torch.fft.fft2(field).abs() ** 2 / K2safe) * nonzero
    P = P.reshape(-1, NX, NY)
    tot = P.sum(dim=(-1, -2)).clamp_min(1e-12)
    shares = [((P * ((K >= lo) & (K < hi))).sum(dim=(-1, -2)) / tot).mean() for lo, hi in bands]
    return torch.stack(shares)


def _resize_decoder_for_spectral_reg(ae, n_sites_x: int, n_sites_y: int):
    """Build a weight-sharing copy of `ae`'s decoder, resized so it can
    decode a `(n_sites_x, n_sites_y)` latent grid directly (same
    mechanism as `ks_latent.models.load_autoencoder_checkpoint_resized_2d`,
    from the in-memory model rather than a checkpoint). Used only by the
    spectral regularizer (`CropPropConfig.w_spectral`) to turn a training
    window's latent into a physical field for the band-share loss.

    **Exact only when the decoder has zero reach** (`dec_site_mix_radius`
    0 or `dec_n_site_mix_layers` 0): with any decoder reach, the circular
    padding inside the resized copy would treat this grid's own edges as
    periodic, which is wrong here (its true neighbours are the
    propagator's forced-halo sites, not a wraparound of itself) -- the
    same boundary problem the crop-training edge masking elsewhere in
    this module exists to avoid. Asserts zero reach rather than silently
    producing a corrupted target/prediction."""
    from ks_latent.config import LocalFieldAutoencoderConfig2D
    old = ae.cfg
    dr = old.site_mix_radius if getattr(old, "dec_site_mix_radius", None) is None else old.dec_site_mix_radius
    dl = old.n_site_mix_layers if getattr(old, "dec_n_site_mix_layers", None) is None else old.dec_n_site_mix_layers
    assert dr == 0 or dl == 0, (
        f"spectral regularizer needs decoder reach 0 (dec_site_mix_radius={dr}, "
        f"dec_n_site_mix_layers={dl}); resizing a nonzero-reach decoder would "
        "introduce circular-wraparound artifacts at this grid's own edges."
    )
    px, py = old.NX // old.n_sites_x, old.NY // old.n_sites_y
    new_cfg = LocalFieldAutoencoderConfig2D(
        NX=n_sites_x * px, NY=n_sites_y * py, n_sites_x=n_sites_x, n_sites_y=n_sites_y,
        local_channels=old.local_channels, site_mix_radius=old.site_mix_radius,
        n_site_mix_layers=old.n_site_mix_layers, hidden=old.hidden,
        dec_site_mix_radius=getattr(old, "dec_site_mix_radius", None),
        dec_n_site_mix_layers=getattr(old, "dec_n_site_mix_layers", None),
    )
    ref = next(ae.parameters())
    new_ae = type(ae)(new_cfg).to(device=ref.device, dtype=ref.dtype)
    new_ae.load_state_dict(ae.state_dict())
    new_ae.eval()
    for p in new_ae.parameters():
        p.requires_grad_(False)
    return new_ae


def per_site_logdet(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """log det of the c x c channel covariance pooled over all leading dims."""
    c = z.shape[-1]
    x = z.reshape(-1, c).float()
    x = x - x.mean(0, keepdim=True)
    cov = x.T @ x / (x.shape[0] - 1) + eps * torch.eye(c, device=z.device)
    return torch.linalg.slogdet(cov)[1]


def _k_now(frac: float, cfg: CropPropConfig) -> int:
    if frac < cfg.k_mid_frac:
        return int(round(2 + (cfg.k_mid - 2) * frac / cfg.k_mid_frac))
    if frac < cfg.k_warmup_frac:
        f2 = (frac - cfg.k_mid_frac) / (cfg.k_warmup_frac - cfg.k_mid_frac)
        return int(round(cfg.k_mid + (cfg.k_max - cfg.k_mid) * f2))
    return cfg.k_max


def site_anchor(u: torch.Tensor, patch_x: int, patch_y: int) -> torch.Tensor:
    """Exact gauge-anchor field (the encoder's channel 0): patch means of `u`
    `(..., NX, NY)` -> `(..., NX/patch_x, NY/patch_y)`."""
    *lead, NX, NY = u.shape
    return u.reshape(*lead, NX // patch_x, patch_x, NY // patch_y, patch_y).mean(dim=(-3, -1))


@torch.no_grad()
def encode_crop_bank_aligned(ae, u: torch.Tensor, n_offsets: int, crop_sites: int, enc_sites: int,
                             seed: int, batch: int = 64):
    """Like `encode_crop_bank`, but crop offsets are whole patches, so crop
    sites coincide with wide-domain sites (needed to place the model's own
    anchor predictions back into the wide-domain field for C9's global
    Biot-Savart solve). Returns `(bank, crop_traj, strip_x0)`: `strip_x0` is
    the wide-domain site index of the usable strip's first site."""
    ae.eval()
    acfg = ae.cfg
    N, T, NX, NY = u.shape
    px = acfg.patch_x
    Pw = NX // px
    Wc = crop_sites * px
    g = torch.Generator(device="cpu").manual_seed(seed)
    bank, ctraj, cx0 = [], [], []
    for n in range(N):
        for s0 in torch.randint(0, Pw, (n_offsets,), generator=g).tolist():
            zs = []
            for t0 in range(0, T, batch):
                tt = torch.arange(t0, min(T, t0 + batch), device=u.device)
                B = len(tt)
                crops = gather_x_crops(u, torch.full((B,), n, device=u.device), tt,
                                       torch.full((B,), s0 * px, device=u.device), Wc)
                z = ae.encode(crops.reshape(B, -1)).reshape(B, acfg.n_sites_x, acfg.n_sites_y, acfg.local_channels)
                zs.append(z[:, enc_sites:acfg.n_sites_x - enc_sites])
            bank.append(torch.cat(zs))
            ctraj.append(n)
            cx0.append((s0 + enc_sites) % Pw)
    return torch.stack(bank), torch.tensor(ctraj, device=u.device), torch.tensor(cx0, device=u.device)


@dataclass
class BSContext:
    """C9: everything needed to compute the global Biot-Savart velocity during
    forced-halo crop rollouts. `anchors`: truth anchor fields `(N, T, Pw, Py)`
    of the wide domain; `Lx`, `Ly`: wide-domain physical size."""
    anchors: torch.Tensor
    Lx: float
    Ly: float


def _strip_index(x0: torch.Tensor, width: int, Pw: int) -> torch.Tensor:
    return (x0[:, None] + torch.arange(width, device=x0.device)[None, :]) % Pw


def global_velocity_for_strip(anchor_wide: torch.Tensor, state: torch.Tensor, x0: torch.Tensor, halo: int,
                              Lx: float, Ly: float) -> torch.Tensor:
    """`anchor_wide`: truth anchors `(B, Pw, Py)`; `state`: current strip latents
    `(B, Pu, Py, c)` whose free interior `[halo, Pu-halo)` replaces the truth
    anchor at the matching wide-domain sites. Returns the exact Biot-Savart
    velocity on the strip, `(B, Pu, Py, 2)`."""
    from ks_latent.models.propagator import _biot_savart_velocity
    B, Pw, Py = anchor_wide.shape
    Pu = state.shape[1]
    interior = _strip_index(x0 + halo, Pu - 2 * halo, Pw)  # (B, Pu-2h)
    A = anchor_wide.clone()
    A.scatter_(1, interior[:, :, None].expand(-1, -1, Py), state[:, halo:Pu - halo, :, 0])
    vel = _biot_savart_velocity(A, Lx, Ly)  # (B, Pw, Py, 2)
    strip = _strip_index(x0, Pu, Pw)
    return torch.gather(vel, 1, strip[:, :, None, None].expand(-1, -1, Py, 2))


def forced_halo_rollout(prop, z_win: torch.Tensor, k: int, halo: int, noise: float = 0.0,
                        bs: "tuple | None" = None, halo_y: int = 0) -> torch.Tensor:
    """`z_win`: `(B, k+1, Pu, Pv, c)` true encoded window. Rolls `k` steps from
    `z_win[:, 0]` (+ noise), overwriting the forced halo (`halo` sites on each
    x-side, `halo_y` on each y-side) with the true state before every step.
    Returns the predicted free interior `(B, k, Pu-2h, Pv-2h_y, c)`."""
    B, _, Pu, Pv, c = z_win.shape
    state = z_win[:, 0].clone()
    if noise > 0:
        state = state + noise * torch.randn_like(state)
    preds = []
    body = getattr(prop, "body", None)
    ix, iy = slice(halo, Pu - halo), slice(halo_y, Pv - halo_y)
    try:
        for j in range(k):
            inp = z_win[:, j].clone()
            inp[:, ix, iy] = state[:, ix, iy]
            if bs is not None:  # C9: (anchor_win (B, k+1, Pw, Py), strip_x0 (B,), Lx, Ly)
                anchor_win, x0, Lx, Ly = bs
                vel = global_velocity_for_strip(anchor_win[:, j], inp, x0, halo, Lx, Ly)
                body.velocity_override = vel.reshape(B, Pu * Pv, 2)
            state = prop.step_one(inp.reshape(B, -1)).reshape(B, Pu, Pv, c)
            preds.append(state[:, ix, iy])
    finally:
        if bs is not None:
            body.velocity_override = None
    return torch.stack(preds, dim=1)


def train_crop_prop(prop, bank_train: torch.Tensor, bank_val: torch.Tensor, cfg: CropPropConfig, log=print,
                    bs_train: "tuple | None" = None, bs_val: "tuple | None" = None,
                    ae=None, site_len: float = 1.0) -> dict:
    """C9: `bs_train`/`bs_val` = `(crop_traj, strip_x0, BSContext)` enable the
    global Biot-Savart velocity input (see `forced_halo_rollout`).

    `ae`/`site_len` (physical length per site, assumed square/isotropic
    patches): required only when `cfg.w_spectral > 0`
    (`spectral_band_shares`'s training-time regularizer); ignored
    otherwise, so every existing call site (all `w_spectral=0`) is
    unaffected."""
    """`bank_*`: `(n_crops, T, Pu, Py, c)` from `encode_crop_bank`."""
    dev = bank_train.device
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    opt = torch.optim.AdamW(prop.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, cfg.steps, cfg.warmup, cfg.lr_min_factor))
    h, hy = cfg.halo_sites, cfg.halo_sites_y
    nC, T = bank_train.shape[:2]

    def windows(bank, B, k, gen, bsx=None):
        ci = torch.randint(0, bank.shape[0], (B,), generator=gen).to(dev)
        t0 = torch.randint(0, bank.shape[1] - k, (B,), generator=gen).to(dev)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None, :]
        if bsx is None:
            return bank[ci[:, None], tt], None
        ctraj, cx0, ctx = bsx
        return bank[ci[:, None], tt], (ctx.anchors[ctraj[ci][:, None], tt], cx0[ci], ctx.Lx, ctx.Ly)

    vg = torch.Generator(device="cpu").manual_seed(54321)
    val_win, val_bs = windows(bank_val, 128, cfg.k_max, vg, bs_val)
    # Rank floor = the TRUE encoded data's per-site log-det (interior), so the
    # penalty only resists collapse below what the data support.
    with torch.no_grad():
        ld_target = per_site_logdet(bank_train[:, :, h:bank_train.shape[2] - h, hy:bank_train.shape[3] - hy]).item()
    log(f"  [cropProp] per-site logdet target (true interior) = {ld_target:.3f}; w_rank = {cfg.w_rank}")
    spec_dec = spec_target = None
    if cfg.w_spectral > 0:
        assert ae is not None, "cfg.w_spectral > 0 needs the frozen autoencoder (train_crop_prop(..., ae=...))"
        Pu_full, Pv_full = bank_train.shape[2], bank_train.shape[3]
        spec_dec = _resize_decoder_for_spectral_reg(ae, Pu_full, Pv_full)
        Lx_spec, Ly_spec = Pu_full * site_len, Pv_full * site_len
        with torch.no_grad():
            tg = torch.Generator(device="cpu").manual_seed(999)
            true_sample = bank_train[torch.randint(0, nC, (256,), generator=tg),
                                     torch.randint(0, T, (256,), generator=tg)]
            NXs, NYs = spec_dec.cfg.NX, spec_dec.cfg.NY
            true_field = spec_dec.decode(true_sample.reshape(256, -1)).reshape(256, NXs, NYs)
            spec_target = spectral_band_shares(true_field, Lx_spec, Ly_spec, cfg.spectral_bands)
        log(f"  [cropProp] spectral target shares {cfg.spectral_bands} = "
            f"{[round(v, 4) for v in spec_target.tolist()]}; w_spectral = {cfg.w_spectral}")
    hist, best, best_state = [], float("inf"), None
    t_start = time.time()
    for step in range(cfg.steps):
        frac = step / cfg.steps
        k = _k_now(frac, cfg)
        noise = cfg.noise_in_start + (cfg.noise_in_end - cfg.noise_in_start) * frac
        prop.train()
        win, wbs = windows(bank_train, cfg.batch, k, g, bs_train)
        pred = forced_halo_rollout(prop, win, k, h, noise, bs=wbs, halo_y=hy)
        loss = F.mse_loss(pred, win[:, 1:, h:win.shape[2] - h, hy:win.shape[3] - hy])
        if cfg.w_rank > 0:
            loss = loss + cfg.w_rank * F.relu(ld_target - per_site_logdet(pred))
        if cfg.w_spectral > 0 and step % cfg.spectral_every == 0:
            # Only the terminal step of the rollout, and only the full `win`
            # grid (NOT `pred`'s narrower free interior) -- the spectral
            # regularizer needs a full Pu x Pv grid to decode exactly (see
            # `_resize_decoder_for_spectral_reg`), and penalizing just the
            # rollout's own final prediction, forced-halo sites included,
            # gives the model credit for the halo it was given, matching how
            # free-rollout evaluation actually sees the field.
            pred_full = win[:, -1].clone()
            pred_full[:, h:win.shape[2] - h, hy:win.shape[3] - hy] = pred[:, -1]
            field = spec_dec.decode(pred_full.reshape(pred_full.shape[0], -1)).reshape(
                pred_full.shape[0], spec_dec.cfg.NX, spec_dec.cfg.NY)
            shares = spectral_band_shares(field, Lx_spec, Ly_spec, cfg.spectral_bands)
            loss = loss + cfg.w_spectral * F.mse_loss(shares, spec_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.steps - 1:
            prop.eval()
            with torch.no_grad():
                vp = forced_halo_rollout(prop, val_win, cfg.k_max, h, bs=val_bs, halo_y=hy)
                val = F.mse_loss(vp, val_win[:, 1:, h:val_win.shape[2] - h, hy:val_win.shape[3] - hy]).item()
            hist.append({"step": step + 1, "k": k, "train_loss": loss.item(), "val_kmax_mse": val,
                         "wall_s": time.time() - t_start})
            log(f"  [cropProp] step {step + 1}/{cfg.steps} k {k} loss {loss.item():.5f} "
                f"val_kmax {val:.5f} [{time.time() - t_start:.0f}s]")
            if val < best:
                best = val
                best_state = {kk: v.detach().clone() for kk, v in prop.state_dict().items()}
    prop.load_state_dict(best_state)
    return {"history": hist, "best_val_kmax_mse": best}
