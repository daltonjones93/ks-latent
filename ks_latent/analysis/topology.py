"""Persistent homology: plain Rips and DTM (distance-to-measure) filtration
(brief §6.3).

Two filtrations, both via giotto-tda's simplicial-homology transformers:
- Plain Vietoris-Rips (`VietorisRipsPersistence`): sensitive to outliers,
  since a single far point can inflate the filtration scale at which real
  structure appears.
- DTM filtration (`WeightedRipsPersistence(weights="DTM")`, Anai, Chazal,
  Glisse, Ike, Inakoshi, Tinarrage & Umeda 2019, "DTM-based filtrations"):
  reweights the filtration by local density, so outliers (which have a low
  local density / high DTM value) are pushed to appear late rather than
  distorting the whole diagram.

The brief's "mass" parameter (the standard Chazal et al. notion, the
fraction of the point cloud used to estimate local density at each point)
is converted here to giotto-tda's `n_neighbors` via `k = round(mass * N)`.

Docstring note carried over from the brief: birth/death *scales* differ
between different probes (e.g. a raw high-dim cloud vs. a low-dim diffusion
embedding) purely because each space's natural distance scale differs; only
the *shape* of the diagram -- distance from the diagonal, i.e. lifetime --
is comparable across probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence


@dataclass(frozen=True)
class PersistenceResult:
    diagrams: dict[int, np.ndarray]  # homology_dim -> (n_features, 2) birth/death


def _split_diagram(diagram: np.ndarray, homology_dimensions: tuple[int, ...]) -> dict[int, np.ndarray]:
    out = {}
    for d in homology_dimensions:
        mask = diagram[:, 2] == d
        out[d] = diagram[mask][:, :2]
    return out


def rips_persistence(
    X: np.ndarray, homology_dimensions: tuple[int, ...] = (0, 1, 2)
) -> PersistenceResult:
    """Plain Vietoris-Rips persistent homology."""
    vr = VietorisRipsPersistence(homology_dimensions=list(homology_dimensions))
    diagram = vr.fit_transform([X])[0]
    return PersistenceResult(diagrams=_split_diagram(diagram, homology_dimensions))


def dtm_persistence(
    X: np.ndarray, mass: float, homology_dimensions: tuple[int, ...] = (0, 1, 2), r: float = 2.0
) -> PersistenceResult:
    """DTM-filtration persistent homology at the given mass parameter `mass`
    (fraction of the point cloud used for the local density estimate)."""
    N = X.shape[0]
    k = max(2, round(mass * N))
    wr = WeightedRipsPersistence(
        homology_dimensions=list(homology_dimensions),
        weights="DTM",
        weight_params={"n_neighbors": k, "r": r},
    )
    diagram = wr.fit_transform([X])[0]
    return PersistenceResult(diagrams=_split_diagram(diagram, homology_dimensions))


def lifetimes(diagram: np.ndarray) -> np.ndarray:
    """`death - birth` for each feature, dropping any essential (infinite
    death) class -- significance is distance from the diagonal."""
    finite = diagram[np.isfinite(diagram[:, 1])]
    return finite[:, 1] - finite[:, 0]


def max_lifetime(diagram: np.ndarray) -> float:
    lt = lifetimes(diagram)
    return float(lt.max()) if len(lt) > 0 else 0.0


def n_significant_features(diagram: np.ndarray, threshold: float) -> int:
    """Count features with lifetime > `threshold` -- a simple, explicit
    "off the diagonal" criterion rather than an eyeballed picture."""
    return int(np.sum(lifetimes(diagram) > threshold))
