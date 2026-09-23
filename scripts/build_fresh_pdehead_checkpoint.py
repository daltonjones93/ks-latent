#!/usr/bin/env python
"""Builds a FRESH (untrained), `backbone="spectral_pde_raw"` `pde_head`
checkpoint in the exact `{"prop_state_dict", "prop_config"}` format
`train_stage1_patched.py --pde-distill` writes -- so it can be fed to
`train_stage2_patched.py --init-pdehead-checkpoint ... --freeze-propagator`
(Phase 3 distillation, `train_stage2`'s own docstring) WITHOUT first
running an entire Stage-1 joint-training job just to seed it.

Added 2026-09-10, Section 138, user-directed: "let's set up the test so
that we distill from both 136 and 137 so we can compare the results" --
distilling FROM an already-trained, already-validated, FROZEN free
propagator (Section 136/137's own `mlp` propagator) needs a pde_head that
starts fresh and is trained ONLY against that frozen target in Phase 3 --
there is no reason to pre-train it jointly with anything first (Phase 3's
whole point is training pde_head with zero competing pressure on the
frozen main propagator).

`backbone="spectral_pde_raw"` (not plain "spectral_pde"): works on ANY
encoder's raw z via its own internal self-FFT truncation -- no
`d_latent==2*spectral_K` constraint, unlike "spectral_pde", which requires
z to literally BE a truncated rFFT spectrum (the OLD spectral_field
convention). `local_field`'s z is a genuine (site, channel) field,
structurally nothing like a Fourier spectrum, so "spectral_pde_raw" is the
only one of the two that makes sense here. `spectral_physics_prior` is
NOT available for this backbone (`PropagatorConfig.__post_init__` raises
if set) for the same reason -- there is no "true governing equation" for
an arbitrary learned latent ordering the way there is for a genuine
w=irfft(z) physical field.

`field_kind="polynomial"` (not "mlp"): the whole point of distillation is
an INTERPRETABLE local closure with named coefficients (Sections 99-127's
own "interpretable polynomial-coefficient PDE" effort), not another
opaque MLP. `integrator="euler"` (not "etdrk4"): matches
train_stage1_patched.py's own `--pde-integrator` default and its stated
reasoning -- "etdrk4" bakes in KS's TRUE physical dispersion relation
(Lhat=k^2-k^4), a fact that does NOT hold for an arbitrary learned index
ordering the way it does for a genuine truncated Fourier spectrum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ks_latent.config import AuxPropagatorConfig
from ks_latent.models.propagator import AuxPropagator, aux_cfg_to_propagator_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-latent", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pde-K", type=int, default=None, help="Default: d_latent//2+1.")
    parser.add_argument("--pde-L", type=float, default=None, help="Default: float(d_latent).")
    parser.add_argument("--pde-hidden", type=int, default=128)
    parser.add_argument("--pde-n-blocks", type=int, default=3)
    parser.add_argument("--pde-max-order", type=int, default=4)
    parser.add_argument("--pde-integrator", choices=["euler", "rk4", "etdrk4"], default="euler")
    parser.add_argument("--pde-ode-substeps", type=int, default=1)
    parser.add_argument("--pde-field-kind", choices=["mlp", "polynomial"], default="polynomial")
    parser.add_argument("--pde-poly-degree", type=int, default=2)
    parser.add_argument("--pde-poly-max-term-order", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    d_latent = args.d_latent
    pde_K = args.pde_K if args.pde_K is not None else d_latent // 2 + 1
    pde_L = args.pde_L if args.pde_L is not None else float(d_latent)

    pde_head_cfg = AuxPropagatorConfig(
        d_latent=d_latent, hidden=args.pde_hidden, n_blocks=args.pde_n_blocks,
        mode="markovian", backbone="spectral_pde_raw",
        spectral_K=pde_K, spectral_L=pde_L,
        spectral_max_order=args.pde_max_order, spectral_integrator=args.pde_integrator,
        ode_substeps=args.pde_ode_substeps,
        spectral_field_kind=args.pde_field_kind, spectral_poly_degree=args.pde_poly_degree,
        spectral_poly_max_term_order=args.pde_poly_max_term_order,
    )
    pde_head = AuxPropagator(pde_head_cfg)
    pde_head_prop_cfg_for_save = aux_cfg_to_propagator_cfg(pde_head_cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prop_state_dict": pde_head.state_dict(), "prop_config": pde_head_prop_cfg_for_save}, out_path
    )
    n_params = sum(p.numel() for p in pde_head.parameters())
    print(f"wrote {out_path} (fresh spectral_pde_raw pde_head, d_latent={d_latent}, "
          f"K={pde_K}, L={pde_L}, field_kind={args.pde_field_kind!r}, {n_params} params)")


if __name__ == "__main__":
    main()
