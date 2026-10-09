"""Why did model seed 2 (m2) collapse under zero-shot transfer while m0/m1
did not, despite being indistinguishable at native size?

The propagator's site-mixing layers (`local_mlp_2d`'s `_LocalMixerBlock2D.mix`)
are CIRCULAR Conv2d kernels -- genuinely linear, shift-invariant filters
over the site grid (the token-mixing step has no softmax/attention). For a
circulant linear operator, the per-channel gain at Fourier mode (kx,ky) is
exactly the DFT of the kernel evaluated at that frequency, and the relevant
stability quantity is the spectral radius (largest eigenvalue magnitude) of
the resulting (d_model x d_model) complex gain matrix.

Training only ever exposes the kernel to the discrete frequency grid of the
TRAINED site count (n_sites_x=8). Resizing to a larger domain changes the
site count and therefore introduces NEW frequencies the kernel was never
constrained at during training. This script checks whether m2's kernels
have a dangerously large gain specifically at those new, never-trained
frequencies, while m0/m1 do not.

Also: (b) a weight-norm summary, and (c) an empirical latent-norm rollout
ramp across intermediate resize factors to locate exactly where collapse
sets in.
"""
from __future__ import annotations

import json
import numpy as np
import torch

from ks_latent.models import load_autoencoder_checkpoint_resized_2d, load_propagator_checkpoint_resized_2d
from ks_latent.config import KolmogorovConfig
from ks_latent.solver.kolmogorov import spinup_kolmogorov

REPO = "/Users/daltonjones/Documents/latent_DA"
TAGS = {
    "m0": "section233_kolmogorov_alpha0p1_localmlp2d_window3",
    "m1": "section233_kolmogorov_alpha0p1_localmlp2d_window3_seed1",
    "m2": "section233_kolmogorov_alpha0p1_localmlp2d_window3_seed2",
}


def prop_path(tag):
    return f"{REPO}/artifacts/stage2_prop_kolmogorov_full_{tag}_warmstart_k12_300ep.pt"


def ae_path(tag):
    return f"{REPO}/artifacts/stage1_ae_kolmogorov_full_{tag}.pt"


# ---------------------------------------------------------------------
# (a) Spectral-radius-at-each-frequency analysis of the circular mixer
# ---------------------------------------------------------------------

def gain_matrix_at_freq(kernel: torch.Tensor, kx: float, ky: float) -> np.ndarray:
    """kernel: (out_ch, in_ch, K, K) circular conv weight, padding=K//2
    (odd K, 'same' circular conv). Returns the (out_ch,in_ch) complex gain
    matrix G(kx,ky) = sum_{dx,dy} W[:,:,dx,dy] * exp(-2pi i (kx*dx+ky*dy))
    where (dx,dy) range over the centered kernel offsets, kx,ky in cycles
    per site (i.e. frequency/n_sites, continuous -- this is the whole
    point: we can evaluate it at frequencies that only exist for a LARGER
    site count than the kernel was ever trained at)."""
    K = kernel.shape[-1]
    r = K // 2
    offsets = torch.arange(-r, r + 1, dtype=torch.float64)
    phase_x = torch.exp(-2j * np.pi * kx * offsets)
    phase_y = torch.exp(-2j * np.pi * ky * offsets)
    W = kernel.detach().to(torch.float64).numpy()  # (out,in,K,K)
    phase = np.outer(phase_x.numpy(), phase_y.numpy())  # (K,K) complex
    G = np.tensordot(W, phase, axes=([2, 3], [0, 1]))  # (out,in) complex
    return G


def spectral_radius(G: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def analyze_frequency_response(tag: str, n_sites_trained=8, transfer_sizes=(8, 16, 24)):
    prop, cfg, ckpt = load_propagator_checkpoint_resized_2d(prop_path(tag), new_n_sites_x=n_sites_trained, new_n_sites_y=8)
    body = prop.body
    n_blocks = len(body.blocks)

    results = {"tag": tag, "blocks": []}
    for bi, block in enumerate(body.blocks):
        kernel = block.mix.weight  # (d_model, d_model, K, K)
        trained_freqs_x = [k / n_sites_trained for k in range(n_sites_trained)]
        per_size = {}
        for n_sites in transfer_sizes:
            freqs_x = [k / n_sites for k in range(n_sites)]
            new_only = [f for f in freqs_x if not any(abs(f - tf) < 1e-9 for tf in trained_freqs_x)]
            all_radii = []
            new_radii = []
            for fx in freqs_x:
                for fy in trained_freqs_x:  # y-grid unchanged (n_sites_y fixed at 8)
                    G = gain_matrix_at_freq(kernel, fx, fy)
                    rho = spectral_radius(G)
                    all_radii.append(rho)
                    if fx in new_only:
                        new_radii.append(rho)
            per_size[n_sites] = {
                "max_gain_all_freqs": max(all_radii),
                "max_gain_new_freqs_only": max(new_radii) if new_radii else None,
            }
        results["blocks"].append({"block": bi, "by_size": per_size})
    return results


# ---------------------------------------------------------------------
# (b) Weight-norm summary
# ---------------------------------------------------------------------

def weight_norm_summary(tag: str):
    prop, cfg, ckpt = load_propagator_checkpoint_resized_2d(prop_path(tag), new_n_sites_x=8, new_n_sites_y=8)
    out = {}
    for name, p in prop.named_parameters():
        out[name] = {"norm": float(p.detach().norm()), "max_abs": float(p.detach().abs().max())}
    return out


# ---------------------------------------------------------------------
# (c) Empirical latent-norm rollout ramp across intermediate resize factors
# ---------------------------------------------------------------------

def rollout_norm_ramp(tag: str, scales=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0), n_steps=300, seed=0):
    trace = {}
    for scale in scales:
        new_n_sites_x = int(round(8 * scale))
        new_NX = int(round(128 * scale))
        if new_NX % new_n_sites_x != 0:
            continue  # keep patch size exact, same convention as exp_g3
        ae, ae_cfg, _ = load_autoencoder_checkpoint_resized_2d(
            ae_path(tag), new_n_sites_x=new_n_sites_x, new_n_sites_y=8, new_NX=new_NX, new_NY=128,
        )
        prop, prop_cfg, _ = load_propagator_checkpoint_resized_2d(
            prop_path(tag), new_n_sites_x=new_n_sites_x, new_n_sites_y=8,
        )
        ae.eval(); prop.eval()
        cfg = KolmogorovConfig(Lx=2 * np.pi * scale, Ly=2 * np.pi, NX=new_NX, NY=128,
                               nu=0.01, alpha=0.1, forcing_n=4, dt=0.01, spinup_time=100.0, seed=seed)
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            u0 = spinup_kolmogorov(cfg, rng)
            z = ae.encode(torch.tensor(u0.reshape(-1), dtype=torch.float32).unsqueeze(0))
            norms = [float(z.norm())]
            for _ in range(n_steps):
                z = prop.step_one(z)
                norms.append(float(z.norm()))
        trace[scale] = {"n_sites_x": new_n_sites_x, "norms": norms,
                        "initial": norms[0], "final": norms[-1],
                        "min_after_50": min(norms[50:]) if len(norms) > 50 else None}
        print(f"  {tag} scale={scale}: n0={norms[0]:.2f} n50={norms[50]:.2f} "
              f"n150={norms[150]:.2f} n_final={norms[-1]:.4f}", flush=True)
    return trace


def main():
    out = {"frequency_response": {}, "weight_norms": {}, "rollout_ramp": {}}

    print("=== (a) Frequency-response / spectral-radius analysis ===", flush=True)
    for tag in TAGS.values():
        print(f"-- {tag} --", flush=True)
        res = analyze_frequency_response(tag)
        out["frequency_response"][tag] = res
        for b in res["blocks"]:
            print(f"  block {b['block']}:", {k: v for k, v in b["by_size"].items()}, flush=True)

    print("\n=== (b) Weight norm summary ===", flush=True)
    for tag in TAGS.values():
        out["weight_norms"][tag] = weight_norm_summary(tag)

    print("\n=== (c) Rollout norm ramp (300 steps each) ===", flush=True)
    for key, tag in TAGS.items():
        print(f"-- {tag} ({key}) --", flush=True)
        out["rollout_ramp"][tag] = rollout_norm_ramp(tag)

    with open(f"{REPO}/artifacts/experiments/t1/m2_autopsy.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote artifacts/experiments/t1/m2_autopsy.json")


if __name__ == "__main__":
    main()
