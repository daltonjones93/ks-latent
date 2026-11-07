"""G2b: a local analog forecast model, the equation-free baseline that "looks things up" instead of learning.

The archive holds, for every training storm, frame and latent site, the local latent patch (the site and its
`radius` neighbours on each side, edges padded by replication) and the site's latent change over the next
frame. A forecast step finds, for every site of every member, the `k` archive patches nearest to the current
patch and applies their increment to the site: the mean of the k increments (`stochastic=False`), or the
increment of one of the k drawn at random (`stochastic=True`, so members differ without added noise).

Same latent space, same data and the same one-minute step as the surrogate, and local like it, so the
comparison isolates "learned map" against "nearest neighbours". Not matched by time since initiation (a patch
already carries the storm's phase locally). Distances are computed after a PCA of the patches to `pca_dim`
dimensions (fitted on the archive), which keeps the search cheap; the increments are applied in the full latent.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AnalogConfig:
    radius: int = 1          # patch = 2 radius + 1 sites
    k: int = 10
    stochastic: bool = True
    frame_stride: int = 2    # archive every this many frames
    pca_dim: int = 32        # 0: no projection
    chunk: int = 512         # queries per distance block (512 x 2.2 M keys x 4 B = 4.5 GB)
    n_hist: int = 1          # interface of the twin (Markovian, no learned noise)
    noise_channels: int = 0

    @property
    def reach_sites(self) -> int:
        return self.radius


def _patches(z: torch.Tensor, r: int) -> torch.Tensor:
    """(B, P, c) -> (B, P, (2r+1) c): each site with its r neighbours on each side, edges replicated."""
    if r == 0:
        return z
    zp = torch.cat([z[:, :1].expand(-1, r, -1), z, z[:, -1:].expand(-1, r, -1)], dim=1)
    return torch.cat([zp[:, i:i + z.shape[1]] for i in range(2 * r + 1)], dim=-1)


class AnalogPropagator:
    """Callable like a propagator: `step(Z, gen=None)` with Z (N, P, c) -> next frame."""

    def __init__(self, z_list: list[torch.Tensor], cfg: AnalogConfig, device: torch.device | str = "cpu"):
        if cfg.k < 1 or cfg.radius < 0 or cfg.frame_stride < 1:
            raise ValueError("k >= 1, radius >= 0, frame_stride >= 1 required")
        self.cfg = cfg
        keys, vals = [], []
        for z in z_list:                                    # (T, P, c)
            z = z.float()
            t = torch.arange(0, z.shape[0] - 1, cfg.frame_stride)
            keys.append(_patches(z[t], cfg.radius).reshape(-1, (2 * cfg.radius + 1) * z.shape[-1]).cpu())
            vals.append((z[t + 1] - z[t]).reshape(-1, z.shape[-1]).cpu())
        K, self.vals = torch.cat(keys), torch.cat(vals).to(device)
        if len(K) < cfg.k:
            raise ValueError(f"archive has {len(K)} patches, fewer than k = {cfg.k}")
        self.mu = K.mean(0, keepdim=True)
        if cfg.pca_dim and cfg.pca_dim < K.shape[1]:
            Kc = K - self.mu
            sub = Kc[torch.randperm(len(Kc), generator=torch.Generator().manual_seed(0))[:200_000]].double()
            _, _, Vh = torch.linalg.svd(sub, full_matrices=False)
            self.proj = Vh[: cfg.pca_dim].T.float()          # (D, pca_dim)
        else:
            self.proj = torch.eye(K.shape[1])
        self.keys = ((K - self.mu) @ self.proj).to(device)
        self.knorm = (self.keys ** 2).sum(1)
        self.mu, self.proj = self.mu.to(device), self.proj.to(device)
        self.device = torch.device(device)

    def eval(self):          # the twin calls .eval() on its propagators
        return self

    def neighbours(self, q: torch.Tensor) -> torch.Tensor:
        """Indices (M, k) of the k nearest archive patches to projected queries q (M, d)."""
        out = []
        for i in range(0, len(q), self.cfg.chunk):
            qc = q[i:i + self.cfg.chunk]
            d = self.knorm[None, :] - 2.0 * qc @ self.keys.T + (qc ** 2).sum(1, keepdim=True)
            out.append(torch.topk(d, self.cfg.k, dim=1, largest=False).indices)
        return torch.cat(out)

    @torch.no_grad()
    def __call__(self, Z: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        N, P, c = Z.shape
        q = (_patches(Z.float().to(self.device), self.cfg.radius).reshape(N * P, -1) - self.mu) @ self.proj
        idx = self.neighbours(q)                                            # (N P, k)
        if self.cfg.stochastic and self.cfg.k > 1:
            pick = torch.randint(0, self.cfg.k, (N * P,), generator=gen).to(self.device)
            inc = self.vals[idx[torch.arange(N * P, device=self.device), pick]]
        else:
            inc = self.vals[idx].mean(1)
        return Z + inc.reshape(N, P, c).to(Z.device, Z.dtype)
