"""T1 experiment (docs/ks_local_field_ltransfer.tex sec:c19_da_hypotheses,
user-directed 2026-10-06: "set up and start the T1 and T2 experiments").

T1's own test list asked for more than the unit-test equality check
already in `tests/unit/test_jacobian_coloring.py`: "time both [jacrev
and the coloured Jacobian] at 44^2/66^2/88^2". This script does that,
on the real `p7b_ks2d` checkpoint (not a small synthetic system), at
native size and a zero-retraining L-transfer size, confirming (a) exact
agreement between the two Jacobian methods on real data and (b) the
actual wall-clock cost ratio -- H3's "flat cost" claim is about the
NUMBER of forward/reverse passes (channels=6, fixed, vs m, which grows
with the observation grid), not an assumption that wall time itself is
flat, so this reports real numbers rather than re-asserting the
unit-tested claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.func import jacrev

from ks_latent.da.jacobian_coloring import (
    assert_decoder_zero_reach,
    coloured_obs_jacobian_2d,
    obs_site_map_2d,
)
from ks_latent.models import load_autoencoder_checkpoint_resized_2d

AE_CKPT = "artifacts/stage1_ae_kolmogorov_crop_p7b_ks2d.pt"
OLD_L = 44.0
OBS_STRIDE_AT_OLD_L = 32


def _sparse_obs_operator_2d(NX, NY, stride_x, stride_y):
    def op(u_phys: torch.Tensor) -> torch.Tensor:
        return u_phys.reshape(NX, NY)[::stride_x, ::stride_y].reshape(-1)
    return op


def time_one_size(new_L: float, n_repeats: int = 3) -> dict:
    scale = new_L / OLD_L
    old_ckpt = torch.load(AE_CKPT, map_location="cpu", weights_only=False)
    old_cfg = old_ckpt["ae_config"]
    new_n_sites = int(round(old_cfg.n_sites_x * scale))
    new_NX = int(round(old_cfg.NX * scale))
    obs_stride = int(round(OBS_STRIDE_AT_OLD_L * scale))
    assert new_NX % new_n_sites == 0

    ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
        AE_CKPT, new_n_sites_x=new_n_sites, new_n_sites_y=new_n_sites, new_NX=new_NX, new_NY=new_NX,
    )
    assert_decoder_zero_reach(ae_cfg)
    ae = ae.double().eval()
    d = new_n_sites * new_n_sites * ae_cfg.local_channels
    obs_op = _sparse_obs_operator_2d(new_NX, new_NX, obs_stride, obs_stride)

    def h(z_single):
        return obs_op(ae.decode(z_single.unsqueeze(0)).squeeze(0))

    torch.manual_seed(0)
    z_bar = torch.randn(d, dtype=torch.float64)
    site_of_obs = obs_site_map_2d(new_NX, new_NX, new_n_sites, new_n_sites, obs_stride, obs_stride)
    m = site_of_obs.shape[0]

    # Warm up (first call pays for any lazy compilation/tracing setup).
    with torch.no_grad():
        _ = h(z_bar)
    _ = jacrev(h)(z_bar)
    _ = coloured_obs_jacobian_2d(h, z_bar, site_of_obs, ae_cfg.local_channels)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        J_true = jacrev(h)(z_bar)
    t_jacrev = (time.perf_counter() - t0) / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        J_coloured = coloured_obs_jacobian_2d(h, z_bar, site_of_obs, ae_cfg.local_channels)
    t_coloured = (time.perf_counter() - t0) / n_repeats

    max_err = (J_coloured - J_true).abs().max().item()

    return {
        "new_L": new_L, "scale": scale, "new_NX": new_NX, "new_n_sites": new_n_sites,
        "d_latent": d, "m_obs": m, "channels": ae_cfg.local_channels,
        "t_jacrev_s": t_jacrev, "t_coloured_s": t_coloured,
        "speedup": t_jacrev / t_coloured if t_coloured > 0 else float("inf"),
        "max_abs_err": max_err,
    }


def main() -> None:
    results = []
    for new_L in [44.0, 66.0, 88.0]:
        print(f"[t1] timing at L={new_L} ...")
        r = time_one_size(new_L)
        print(f"[t1]   d={r['d_latent']} m={r['m_obs']}  jacrev={r['t_jacrev_s']:.4f}s  "
              f"coloured={r['t_coloured_s']:.4f}s  speedup={r['speedup']:.1f}x  "
              f"max_err={r['max_abs_err']:.2e}")
        results.append(r)

    out_path = Path("artifacts/experiments/t1_jacobian_timing.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
