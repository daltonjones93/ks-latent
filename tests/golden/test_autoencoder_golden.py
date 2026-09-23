"""Golden forward-pass test (brief §5.3, §19: "regenerating requires an
explicit --update-golden flag producing a reviewable diff").

Pins the exact numerical output of a fixed-seed, fixed-architecture AE
forward pass. A silent change here (without deliberately regenerating and
reviewing the diff) means something about the model's structure or
initialization changed unintentionally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ks_latent.config import AutoencoderConfig
from ks_latent.models.autoencoder_patched import KSAutoencoderPatched

GOLDEN_PATH = Path(__file__).parent / "autoencoder_forward.npz"


def _build_and_run():
    torch.manual_seed(1234)
    cfg = AutoencoderConfig(
        NX=32, patch_size=4, group_size=4, d_model=16, nhead=2, dim_ff=16,
        patch_embed_hidden=32, n_local_layers=1, n_global_layers=1,
        n_query_tokens=4, d_latent=6, dropout=0.0,
    )
    ae = KSAutoencoderPatched(cfg)
    ae.eval()
    u = torch.linspace(-1.0, 1.0, steps=3 * cfg.NX).reshape(3, cfg.NX)
    with torch.no_grad():
        u_hat, z = ae(u)
    return u_hat.numpy(), z.numpy()


def test_autoencoder_golden_forward(request):
    u_hat, z = _build_and_run()

    if request.config.getoption("--update-golden"):
        np.savez(GOLDEN_PATH, u_hat=u_hat, z=z)
        return

    assert GOLDEN_PATH.exists(), "No golden reference found; run with --update-golden first."
    ref = np.load(GOLDEN_PATH)
    np.testing.assert_allclose(u_hat, ref["u_hat"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(z, ref["z"], rtol=1e-5, atol=1e-6)
