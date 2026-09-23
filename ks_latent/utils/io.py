"""Provenance sidecars and dataset I/O (brief ground rule 7, §3.3)."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ks_latent.config import config_hash


def get_git_sha(repo_dir: str | Path = ".") -> str:
    """Current git SHA, or 'unknown' outside a repo (never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_provenance(
    artifact_path: str | Path,
    *,
    config: Any,
    seed: int,
    device: str,
    wall_time_s: float,
    extra: dict | None = None,
) -> Path:
    """Write `<artifact_path>.meta.json` alongside an artifact.

    Every artifact this codebase produces (dataset, checkpoint, figure, JSON
    result) gets one of these: git SHA, config hash, seed, device, timestamp,
    wall time, and the fully resolved config, so any result can be traced
    back to exactly what produced it.
    """
    artifact_path = Path(artifact_path)
    meta = {
        "artifact": str(artifact_path.name),
        "git_sha": get_git_sha(),
        "config_hash": config_hash(config),
        "config": dataclasses.asdict(config),
        "seed": seed,
        "device": device,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_s": wall_time_s,
    }
    if extra:
        meta["extra"] = extra
    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta_path
