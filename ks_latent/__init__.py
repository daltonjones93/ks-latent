"""ks_latent: latent-space data assimilation and manifold analysis for KS."""

import logging
import os
import warnings

# Must be set before torch is imported anywhere in the process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_mps_logger = logging.getLogger("ks_latent.mps_fallback")
_mps_logger.setLevel(logging.WARNING)
if not _mps_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[ks_latent.mps_fallback] %(message)s"))
    _mps_logger.addHandler(_handler)

_seen_fallback_ops: set[str] = set()
_original_showwarning = warnings.showwarning


def _mps_fallback_showwarning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if "not currently supported on the MPS backend" in text or "will fall back to run on the CPU" in text:
        if text not in _seen_fallback_ops:
            _seen_fallback_ops.add(text)
            _mps_logger.warning(
                "op fell back to CPU (this is a 50x-slowdown risk in an inner loop): %s", text
            )
        return
    _original_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _mps_fallback_showwarning

import torch  # noqa: E402  (must follow env var + warning-hook setup above)

__all__ = ["torch"]
