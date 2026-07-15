#!/usr/bin/env python3
"""
Configuration loading for the AutoAntibiotic Discovery Pipeline.

This module isolates the YAML/env parsing of the pipeline run-mode so the
orchestrator (``discovery_pipeline``) does not need to know *how* the mode is
resolved — only *what* mode it ended up with.
"""

import os
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger("AutoAntibiotic")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load pipeline configuration from *config_path* (YAML) or environment.

    The configuration exposes a ``mode`` key, either ``"ci"`` (CI/mock runs,
    no physical redocking) or ``"science"`` (real scientific validation).

    Resolution order (first match wins):
        1. ``AUTOANTIBIOTIC_MODE`` environment variable — explicit override,
            takes precedence over everything on disk. Accepted values:
            ``"ci"`` or ``"science"``.
        2. ``config.yaml`` on disk — the preferred, version-controlled source
            of truth. The ``mode`` key must be exactly ``"ci"`` or
            ``"science"``; anything else is ignored and the warning path runs.
        3. Default fallback — if no file exists (or it is unreadable / missing
            a valid ``mode``), the pipeline defaults to ``mode: ci`` (a fast,
            offline run that proves the install works) and emits a warning.

    To perform heavy scientific computations, create a ``config.yaml`` with
    ``mode: science`` (or set ``AUTOANTIBIOTIC_MODE=science``).

    Returns:
        dict with at least a ``mode`` key.
    """
    cfg: Dict[str, str] = {"mode": "ci"}

    # ── 1: Environment override (explicit is preferred over implicit) ──
    env_mode = os.environ.get("AUTOANTIBIOTIC_MODE")
    if env_mode in ("ci", "science"):
        cfg["mode"] = env_mode
        return cfg

    # ── 2: config.yaml on disk ──
    config_file = Path(config_path)
    if config_file.exists():
        try:
            import yaml

            with open(config_file) as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict) and data.get("mode") in ("ci", "science"):
                cfg["mode"] = data["mode"]
            else:
                log.warning(
                    f"  ⚠  {config_path} missing a valid 'mode' (ci/science); "
                    "defaulting to mode='ci'."
                )
        except ImportError:
            log.warning(
                "  ⚠  pyyaml is not installed; cannot parse config.yaml. "
                "Defaulting to mode='ci'. Install pyyaml for config support."
            )
        except Exception as exc:
            log.warning(
                f"  ⚠  Failed to read {config_path} ({exc}); "
                "defaulting to mode='ci'."
            )
    else:
        log.warning(
            f"  ⚠  {config_path} not found; defaulting to mode='ci'. "
            "Create a config.yaml (mode: ci|science) to set the run mode explicitly."
        )

    # ── 4: default fallback already set above (mode: ci) ──
    return cfg
