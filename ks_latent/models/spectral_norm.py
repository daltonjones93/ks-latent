"""Standalone leaf module (added 2026-09-04, user-directed: "would there
be a way to constrain the fourier_mlp to be nonexpansive") for
`SpectralNormLinear`/`_sn` -- shared by `ks_latent.models.propagator` AND
`ks_latent.models.autoencoder_masked_mlp`, which cannot import from each
other directly: `propagator.py` imports from `autoencoder_vit.py`, which
itself imports `MaskedLinearRect` from `autoencoder_masked_mlp.py` -- so
defining this class in either of those two files and importing it into
the other creates a circular import. This module has no `ks_latent.models`
dependencies of its own, breaking the cycle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormLinear(nn.Module):
    """Manual spectral-normalized `Linear` (power iteration), NOT using
    `torch.nn.utils.parametrizations.spectral_norm` -- that built-in was
    found to crash under `--amp` on MPS (`RuntimeError: Failed to create
    function state object for: div_true_strided_float_bfloat`), from its
    internal float32 power-iteration buffers getting divided against a
    bfloat16-autocast-dispatched tensor mid-forward (`torch.mv`/`matmul`
    are autocast-registered, so computing against the weight inside an
    active `bfloat16` autocast region silently produces a `bfloat16`
    result even though the buffer itself is `float32`) -- same bfloat16-
    incompatibility bug class as `torch.fft.rfft`/`torch.linalg.slogdet`
    elsewhere in this codebase; same fix here: the power iteration itself
    runs inside an explicit `torch.autocast(..., enabled=False)` block,
    forcing genuine `float32` regardless of the ambient autocast context,
    while the actual `F.linear(x, ...)` call against the caller's input
    stays OUTSIDE that block so it still gets the surrounding autocast's
    speed benefit normally.

    Exposes the RAW `.weight`/`.bias` parameters directly (so `state_dict`
    stays a plain weight/bias, no `parametrizations.*` key renaming) plus
    `effective_weight()` for the spectral-normalized value --
    `ks_latent.models.propagator.MaskedLinear`/`ks_latent.models.
    autoencoder_masked_mlp.MaskedLinearRect` call `effective_weight()`
    instead of reading `.weight` directly when wrapping this class (see
    their own `nonexpansive` docstring sections)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.register_buffer("_u", F.normalize(torch.randn(out_features), dim=0, eps=eps))
        self.eps = eps

    def effective_weight(self) -> torch.Tensor:
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            w = self.weight.float()
            with torch.no_grad():
                v = F.normalize(torch.mv(w.t(), self._u), dim=0, eps=self.eps)
                u = F.normalize(torch.mv(w, v), dim=0, eps=self.eps)
                if self.training:
                    self._u.copy_(u)
                else:
                    u = self._u
            sigma = torch.dot(u, torch.mv(w, v))
            return (w / sigma).to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


def _sn(in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    """Construct a spectral-normalized `Linear` (operator norm <=1, i.e.
    this single layer can never amplify any input perturbation) -- shared
    helper for every `nonexpansive=True` path in `propagator.py`/
    `autoencoder_masked_mlp.py`. See `ks_latent.models.propagator.
    ResidualMLPBlock`'s docstring for why this alone is not sufficient
    for a whole non-expansive NETWORK (residual connections and multi-
    path sums need their own handling), and `SpectralNormLinear`'s
    docstring for why this doesn't use PyTorch's built-in `spectral_norm`
    parametrization."""
    return SpectralNormLinear(in_features, out_features, bias=bias)
