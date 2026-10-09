"""DA cycling experiment driver (brief §7): repeatedly forecast the ensemble
through the latent propagator, assimilate a noisy physical-space observation
via NAT-PFF, and record the exact spacetime/RMSE conventions the brief
specifies.

Spacetime convention (brief §7): per cycle, the recorded segment is
`n_prop_steps` rows of the *forecast* trajectory, with only the **last**
row overwritten by the analysis (`u_da_history` uses `decode(mean(z))`, not
`mean(decode(z))` -- the two differ for a nonlinear decoder, and the brief
is explicit about which one). Truth is `u_truth_full[1:]` (the truth at the
same time indices as the recorded forecast/analysis rows, since row 0 of
`u_truth_full` is the initial condition rather than a forecast target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ks_latent.da.forecast import (
    MODEL_NOISE_STD,
    OBS_NOISE_STD,
    forecast_ensemble,
    forecast_ensemble_history,
)
from ks_latent.da.letkf import LETKF
from ks_latent.da.letkf_local import LETKFLocal2D, LETKFLocal2DETKF
from ks_latent.da.serial_enkf import SerialSECEnKF
from ks_latent.da.pff import ParticleFlowFilter, PFFConfig
from ks_latent.da.rmse import per_dim_rmse
from ks_latent.models.propagator import LatentPropagator

PhysObsOperator = Callable[[torch.Tensor], torch.Tensor]  # (NX,) -> (m,)


@dataclass(frozen=True)
class CycleConfig:
    n_prop_steps: int
    n_cycles: int
    n_ensemble: int = 64
    model_noise_std: float = MODEL_NOISE_STD
    obs_noise_std: float = OBS_NOISE_STD
    # Observation-error std the FILTER assumes (R), if different from the noise actually added
    # to the synthetic observations (obs_noise_std). None = the same. Used to account for
    # representation error: the decoder cannot reproduce the truth exactly, so the error of an
    # observation relative to anything the model can represent is larger than the instrument
    # noise (2026-10-08, accuracy study).
    filter_obs_std: float | None = None
    # Also record mean(decode(z)) over members at each analysis (diagnostic only; the filter and
    # the u_da history are unchanged, which use decode(mean(z)) as the brief prescribes).
    record_decoded_member_mean: bool = False
    # Variational recentring of the analysis mean (ks_latent.da.var_recenter); None = off.
    var_recenter: object | None = None
    # Physical-space additive model noise (accuracy study, 2026-10-09): once per cycle, before the
    # analysis, each member is perturbed by a Gaussian random field xi added to its decoded field
    # and re-encoded, z <- z + E(D(z) + xi) - E(D(z)). xi has std `phys_noise_std` (standardized
    # units) and correlation length `phys_noise_len` grid points (Gaussian kernel), so the ensemble
    # spread acquires smooth, field-shaped directions instead of per-coordinate latent noise.
    phys_noise_std: float = 0.0
    phys_noise_len: float = 4.0
    phys_noise_shape: tuple[int, int] | None = None
    pff_config: PFFConfig = field(default_factory=PFFConfig)
    seed: int = 0
    localize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None  # brief §9
    # `method` (added 2026-09-28, user-directed: "implement LETKF"):
    # "pff" (default, unchanged behavior) or "letkf" -- see
    # `ks_latent.da.letkf`'s module docstring for the exact algorithm and
    # why it's a clean, single-variable comparator against NAT-PFF (same
    # forecast, same ensemble-mean-Jacobian linearization, same
    # `localize_fn` taper; only the analysis update rule differs).
    method: str = "pff"
    letkf_inflation: float = 1.0
    # `method="letkf_local2d"` (added 2026-10-06, docs/ks_local_field_ltransfer.tex
    # sec:c19_da_hypotheses test T2): the per-site local LETKF
    # (`ks_latent.da.letkf_local.LETKFLocal2D`), valid only for a
    # zero-decoder-reach `local_field_2d` autoencoder
    # (`ks_latent.da.jacobian_coloring.assert_decoder_zero_reach` --
    # callers must check this themselves before setting this method).
    # These five fields are REQUIRED when `method == "letkf_local2d"`
    # and ignored otherwise.
    site_of_obs: torch.Tensor | None = None  # (m,) int64
    n_sites_x: int | None = None
    n_sites_y: int | None = None
    local_channels: int | None = None
    gc_c_local: float | None = None
    # `method="serial_sec_enkf"` (task D3(ii), 2026-10-08): serial perturbed-observation EnKF
    # with the SEC correction on each regression coefficient (`ks_latent.da.serial_enkf`).
    # `sec_table` None = plain serial EnKF.
    sec_table: object | None = None


@dataclass(frozen=True)
class DAExperimentResult:
    u_truth: torch.Tensor  # (T, NX), T = n_cycles * n_prop_steps
    u_da: torch.Tensor  # (T, NX) -- decode(mean(z)), analysis overwrites last row/cycle
    u_free: torch.Tensor  # (T, NX) -- free run, no assimilation
    rmse_da: torch.Tensor  # (T,) per-dim RMSE **in latent space** (brief/handoff convention)
    rmse_free: torch.Tensor  # (T,)
    spread: torch.Tensor  # (T,) per-dim ensemble std at each step, latent space
    z_da_mean: torch.Tensor  # (T, d)
    z_truth: torch.Tensor  # (T, d) -- encode(u_truth), the RMSE comparison target
    u_da_member_mean: torch.Tensor | None = None  # (n_cycles, NX) mean(decode(z)) at analyses, if recorded


def _decode_mean(decoder: Callable[[torch.Tensor], torch.Tensor], Z: torch.Tensor) -> torch.Tensor:
    """`decode(mean(z))`, brief §7: "not mean(decode(z))"."""
    with torch.no_grad():
        return decoder(Z.mean(dim=0, keepdim=True)).squeeze(0)


def run_da_experiment(
    propagator: LatentPropagator,
    decoder: Callable[[torch.Tensor], torch.Tensor],
    encoder: Callable[[torch.Tensor], torch.Tensor],
    obs_operator_phys: PhysObsOperator,
    truth_traj: torch.Tensor,  # (T_full+1, NX): index 0 is the shared IC for both DA and free ensembles
    z0_ensemble: torch.Tensor,  # (N, d): initial ensemble at truth_traj[0]'s encoded pair start
    z_minus1_ensemble: torch.Tensor,  # (N, d): the step *before* z0 (propagator needs 2-step history)
    cfg: CycleConfig,
    z_hist_ensemble: torch.Tensor | None = None,
) -> DAExperimentResult:
    """`z_hist_ensemble` (added 2026-08-29, user-directed): `(N, n_history,
    d)`, oldest to newest -- required instead of `z0_ensemble`/
    `z_minus1_ensemble` when `propagator.mode == "history"` (a propagator
    without a `.mode` attribute at all, e.g. a test double, is treated as
    the original `markovian`/`two_step` 2-state case). `z0_ensemble`/
    `z_minus1_ensemble` are ignored in that case."""
    T = cfg.n_cycles * cfg.n_prop_steps
    if truth_traj.shape[0] < T + 1:
        raise ValueError(f"truth_traj needs >= {T + 1} rows for {cfg.n_cycles} cycles, got {truth_traj.shape[0]}")

    is_history = getattr(propagator, "mode", None) == "history"
    if is_history and z_hist_ensemble is None:
        raise ValueError("propagator.mode == 'history' requires z_hist_ensemble, (N, n_history, d)")

    r_std = cfg.obs_noise_std if cfg.filter_obs_std is None else cfg.filter_obs_std
    R = r_std**2 * torch.eye(obs_operator_phys(truth_traj[0]).shape[0], dtype=torch.float64)

    if is_history:
        Z_da_hist, Z_free_hist = z_hist_ensemble.clone(), z_hist_ensemble.clone()
    else:
        Z_da_prev, Z_da_curr = z_minus1_ensemble.clone(), z0_ensemble.clone()
        Z_free_prev, Z_free_curr = z_minus1_ensemble.clone(), z0_ensemble.clone()

    u_da_list, u_free_list = [], []
    u_da_member_mean_list: list[torch.Tensor] = []
    z_da_mean_list, z_free_mean_list, spread_list = [], [], []

    rng = torch.Generator().manual_seed(cfg.seed)
    t = 0
    for cycle in range(cfg.n_cycles):
        if is_history:
            fc_da = forecast_ensemble_history(propagator, Z_da_hist, cfg.n_prop_steps, cfg.model_noise_std)
            fc_free = forecast_ensemble_history(propagator, Z_free_hist, cfg.n_prop_steps, model_noise_std=0.0)
            Z_da_hist, Z_free_hist = fc_da.z_hist_final, fc_free.z_hist_final
            Z_da_curr = Z_da_hist[:, -1]
        else:
            fc_da = forecast_ensemble(propagator, Z_da_prev, Z_da_curr, cfg.n_prop_steps, cfg.model_noise_std)
            fc_free = forecast_ensemble(propagator, Z_free_prev, Z_free_curr, cfg.n_prop_steps, model_noise_std=0.0)
            Z_da_prev, Z_da_curr = fc_da.z_prev_final, fc_da.z_curr_final
            Z_free_prev, Z_free_curr = fc_free.z_prev_final, fc_free.z_curr_final

        da_segment = fc_da.history.clone()  # (n_prop_steps, N, d)
        if cfg.phys_noise_std > 0:
            nx, ny = cfg.phys_noise_shape
            with torch.no_grad():
                U = torch.cat([decoder(Z_da_curr[i:i + 16]) for i in range(0, Z_da_curr.shape[0], 16)])
                xi = torch.randn(U.shape[0], nx, ny, generator=rng)
                fx = torch.fft.fftfreq(nx)[:, None]; fy = torch.fft.fftfreq(ny)[None, :]
                kern = torch.exp(-2 * (torch.pi * cfg.phys_noise_len) ** 2 * (fx ** 2 + fy ** 2))
                xi = torch.fft.ifft2(torch.fft.fft2(xi) * kern).real
                xi = xi / xi.std(dim=(1, 2), keepdim=True) * cfg.phys_noise_std
                Zn = torch.cat([encoder((U + xi.reshape(U.shape[0], -1))[i:i + 16]) for i in range(0, U.shape[0], 16)])
                Z0 = torch.cat([encoder(U[i:i + 16]) for i in range(0, U.shape[0], 16)])
            Z_da_curr = Z_da_curr + (Zn - Z0).to(Z_da_curr.dtype)
            da_segment[-1] = Z_da_curr

        # Assimilate at the last row of this cycle's segment.
        u_true_at_analysis = truth_traj[t + cfg.n_prop_steps]
        y_true = obs_operator_phys(u_true_at_analysis)
        y_obs = y_true + cfg.obs_noise_std * torch.randn(y_true.shape[0], generator=rng)

        model_dtype = Z_da_curr.dtype

        def h(z_single):
            # PFF's Jacobian/linalg run in float64; the trained decoder is
            # float32 (MPS-trained) -- cast at this one boundary.
            u_phys = decoder(z_single.to(model_dtype).unsqueeze(0)).squeeze(0)
            return obs_operator_phys(u_phys).to(torch.float64)

        if cfg.method == "pff":
            analyzer = ParticleFlowFilter(h, R, decoder_phys=decoder, config=cfg.pff_config, localize_fn=cfg.localize_fn)
        elif cfg.method == "letkf":
            analyzer = LETKF(h, R, localize_fn=cfg.localize_fn, inflation=cfg.letkf_inflation, seed=cfg.seed + cycle)
        elif cfg.method in ("letkf_local2d", "letkf_local2d_etkf"):
            missing = [
                name for name in ("site_of_obs", "n_sites_x", "n_sites_y", "local_channels", "gc_c_local")
                if getattr(cfg, name) is None
            ]
            if missing:
                raise ValueError(f"CycleConfig.method={cfg.method!r} requires {missing} to be set")
            # "_etkf" (task D4): deterministic per-site ensemble transform, same localization.
            local_cls = LETKFLocal2DETKF if cfg.method == "letkf_local2d_etkf" else LETKFLocal2D
            analyzer = local_cls(
                h, torch.diagonal(R).clone(), cfg.site_of_obs, cfg.n_sites_x, cfg.n_sites_y,
                cfg.local_channels, cfg.gc_c_local, inflation=cfg.letkf_inflation, seed=cfg.seed + cycle,
            )
        elif cfg.method == "serial_sec_enkf":
            analyzer = SerialSECEnKF(h, torch.diagonal(R).clone(), table=cfg.sec_table, seed=cfg.seed + cycle)
        else:
            raise ValueError(f"CycleConfig.method must be 'pff', 'letkf', 'letkf_local2d', 'letkf_local2d_etkf' "
                             f"or 'serial_sec_enkf', got {cfg.method!r}")
        Z_forecast = Z_da_curr
        analysis = analyzer.analyze(Z_da_curr.to(torch.float64), y_obs.to(torch.float64))
        Z_da_curr = analysis.Z_analysis.detach().to(Z_da_curr.dtype)
        if cfg.var_recenter is not None:
            from ks_latent.da.var_recenter import var_recenter_mean
            r_std_v = cfg.obs_noise_std if cfg.filter_obs_std is None else cfg.filter_obs_std
            z_star = var_recenter_mean(decoder, obs_operator_phys, y_obs, r_std_v, Z_da_curr.mean(0),
                                       Z_forecast, cfg.var_recenter)
            Z_da_curr = Z_da_curr - Z_da_curr.mean(0, keepdim=True) + z_star.to(Z_da_curr.dtype)
        da_segment[-1] = Z_da_curr  # "last forecast row overwritten by the analysis"
        if cfg.record_decoded_member_mean:
            with torch.no_grad():
                dec = torch.cat([decoder(Z_da_curr[i:i + 16]) for i in range(0, Z_da_curr.shape[0], 16)])
            u_da_member_mean_list.append(dec.mean(dim=0))
        if is_history:
            # Fold the analysis-corrected state back into the sliding history
            # window so the *next* cycle's forecast starts from it, not the
            # raw (pre-analysis) forecast.
            Z_da_hist = torch.cat([Z_da_hist[:, 1:], Z_da_curr.unsqueeze(1)], dim=1)

        for k in range(cfg.n_prop_steps):
            z_slice_da = da_segment[k]
            z_slice_free = fc_free.history[k]
            u_da_list.append(_decode_mean(decoder, z_slice_da))
            u_free_list.append(_decode_mean(decoder, z_slice_free))
            z_da_mean_list.append(z_slice_da.mean(dim=0))
            z_free_mean_list.append(z_slice_free.mean(dim=0))
            spread_list.append(per_dim_rmse(z_slice_da.std(dim=0)))

        t += cfg.n_prop_steps

    u_da = torch.stack(u_da_list, dim=0)
    u_free = torch.stack(u_free_list, dim=0)
    u_truth = truth_traj[1 : T + 1]
    with torch.no_grad():
        z_truth = encoder(u_truth)  # (T, d)

    z_da_mean = torch.stack(z_da_mean_list, dim=0)
    z_free_mean = torch.stack(z_free_mean_list, dim=0)

    rmse_da = torch.tensor([per_dim_rmse(z_da_mean[i] - z_truth[i]) for i in range(T)])
    rmse_free = torch.tensor([per_dim_rmse(z_free_mean[i] - z_truth[i]) for i in range(T)])

    return DAExperimentResult(
        u_truth=u_truth,
        u_da=u_da,
        u_free=u_free,
        rmse_da=rmse_da,
        rmse_free=rmse_free,
        spread=torch.tensor(spread_list),
        z_da_mean=z_da_mean,
        z_truth=z_truth,
        u_da_member_mean=torch.stack(u_da_member_mean_list) if u_da_member_mean_list else None,
    )
