"""
OpenMM platform selection for the AutoAntibiotic MD pipeline.

Provides a single, well-tested entry point for choosing which OpenMM platform
should run a simulation on the host machine, with clear logging of the choice
and the reason. Preference order is:

    Metal → CUDA → OpenCL → CPU

Rationale:

* ``Metal`` — fastest on Apple Silicon (M-series), when a Metal-enabled OpenMM
  build is available. OpenMM does **not** ship a native Metal platform in any
  prebuilt conda/pip package (8.5.x included); the literal ``Metal`` platform
  only exists via the third-party ``openmm-metal`` plugin built against a
  matching OpenMM source tree. This module therefore *tries* Metal first and
  transparently falls back when it is absent. See ``docs/metal_acceleration.md``
  for how to build/obtain a Metal-enabled OpenMM.

* ``CUDA`` — NVIDIA GPUs.

* ``OpenCL`` — the practical GPU path on Apple Silicon with stock OpenMM.
  Apple's OpenCL runtime compiles kernels to Metal under the hood (the same
  software path the ``openmm-metal`` plugin used via ``cl2Metal``), so it is
  Metal-backed in effect. On the reference M5 Pro machine it runs the real
  419k-atom PBP2a solvated system at ~7–9 ns/day versus ~1.06 ns/day for CPU.

* ``CPU`` — always correct; last resort.

Two important correctness rules are enforced/document here:

1. **Periodic restraint expressions.** ``CustomExternalForce`` expressions that
   reference absolute coordinates (``k*(x-x0)^2 + ...``) are *wrong* on GPU
   platforms for periodic systems: GPU platforms wrap coordinates into the
   primary cell, so ``x - x0`` can differ by a lattice vector and blow up to
   NaN. Any position restraint on a periodic system must use the built-in
   ``periodicdistance(x, y, z, x0, y0, z0)`` function instead. This module
   exposes :func:`position_restraint_force` that builds a correct restraint.

2. **Precision.** On macOS Apple's OpenCL runtime exposes single precision
   only; ``mixed``/``double`` OpenCL precision is unavailable there. The
   selectors below therefore never force mixed/double precision on OpenCL.
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Dict, List, Optional

log = logging.getLogger("AutoAntibiotic.openmm_platform")

# Preferred platform order (best accelerator first).
PREFERENCE_ORDER: List[str] = ["Metal", "CUDA", "OpenCL", "CPU"]

# Platforms that are known to require the periodicdistance() restraint form.
# Reference/CPU store absolute coordinates in double precision and tolerate the
# naive (x-x0)^2 form, but every GPU backend uses the wrapped-cell convention.
GPU_PLATFORMS = {"Metal", "CUDA", "OpenCL", "HIP"}

# OpenMM platform properties that improve throughput / correctness per backend.
_PROPERTY_HINTS: Dict[str, Dict[str, str]] = {
    "CPU": {"Threads": "default"},
    "OpenCL": {"DeviceIndex": "0"},
    "CUDA": {"DeviceIndex": "0", "Precision": "mixed"},
}


def _platform_names() -> List[str]:
    """Return the names of platforms registered with the loaded OpenMM build."""
    try:
        import openmm
        return [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
    except ImportError:
        return []


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def available_platforms() -> List[str]:
    """Return registered OpenMM platform names, preferred order first."""
    registered = set(_platform_names())
    return [p for p in PREFERENCE_ORDER if p in registered] + [
        p for p in registered if p not in PREFERENCE_ORDER
    ]


def has_platform(name: str) -> bool:
    """True if OpenMM exposes a platform called *name*."""
    return name in _platform_names()


def _platform_properties(platform_name: str, threads: Optional[int]) -> Dict[str, str]:
    props: Dict[str, str] = {}
    hints = _PROPERTY_HINTS.get(platform_name, {})
    if platform_name == "CPU":
        if threads is None:
            # OpenMM uses all cores by default; only pin threads when asked.
            return props
        props["Threads"] = str(threads)
    else:
        props.update({k: v for k, v in hints.items() if k != "Threads"})
    return props


def select_platform(
    preference: Optional[str] = None,
    allow: Optional[List[str]] = None,
    threads: Optional[int] = None,
) -> Dict[str, object]:
    """Select an OpenMM platform for a simulation.

    Args:
        preference: Explicit platform name to force (``--platform`` /
            ``OPENMM_PLATFORM`` override). ``"auto"`` or ``None`` chooses the
            best available automatically.
        allow: Optional allow-list of platform names. When given, only these
            (in preference order) are considered. Used internally to skip
            platforms that are unsuitable for a given phase.
        threads: CPU thread count to pin when falling back to CPU (default:
            OpenMM's own default).

    Returns a dict::

        {
            "name": str,                 # chosen platform name
            "platform": openmm.Platform, # the resolved platform object
            "properties": dict,          # platformProperties to pass to Simulation
            "fallback_chain": list[str], # names tried before the choice
            "reason": str,               # human-readable explanation
        }

    Raises:
        RuntimeError if no platform in the candidate set is available.
    """
    import openmm

    order = PREFERENCE_ORDER
    if preference and preference.lower() not in ("auto", "none"):
        # Honour an explicit user override: force that exact platform.
        if has_platform(preference):
            props = _platform_properties(preference, threads)
            log.info(
                "  OpenMM platform: %s (explicit override, properties=%s)",
                preference, props or "{}",
            )
            return {
                "name": preference,
                "platform": openmm.Platform.getPlatformByName(preference),
                "properties": props,
                "fallback_chain": [],
                "reason": f"explicit --platform override requested {preference}",
            }
        log.warning(
            "  Requested OpenMM platform '%s' is not available; falling back "
            "to auto-detection. Registered: %s", preference, _platform_names(),
        )

    registered = _platform_names()
    if allow:
        order = [p for p in PREFERENCE_ORDER if p in allow] + [
            p for p in order if p in allow and p not in PREFERENCE_ORDER
        ]
    candidates = [p for p in order if p in registered]
    fallback_chain: List[str] = []
    for name in candidates:
        props = _platform_properties(name, threads)
        if name == "CPU":
            log.info(
                "  OpenMM platform: CPU (threads=%s)", props.get("Threads", "default"),
            )
        else:
            log.info("  OpenMM platform: %s (properties=%s)", name, props or "{}")
        return {
            "name": name,
            "platform": openmm.Platform.getPlatformByName(name),
            "properties": props,
            "fallback_chain": fallback_chain,
            "reason": f"auto-detected best available platform ({name})",
        }
    raise RuntimeError(
        "No usable OpenMM platform found. Registered platforms: "
        f"{registered}. Install a GPU-enabled OpenMM build or use CPU."
    )


def position_restraint_force(
    force_constant_kj_per_mole_nm2: float,
    periodic: bool,
) -> "openmm.CustomExternalForce":
    """Build a correct harmonic position-restraint ``CustomExternalForce``.

    The expression uses ``periodicdistance(...)`` so the restraint is correct
    on GPU platforms (wrapped-cell convention) as well as CPU/Reference. This
    is the fix for the historic pipeline bug where the naive ``(x-x0)^2 + ...``
    form produced NaN energies on OpenCL for periodic systems.

    Returns a force with per-particle parameters ``k, x0, y0, z0`` (in internal
    OpenMM units, kJ/mol/nm² and nm). Callers add particles with
    ``force.addParticle(index, [k, x0, y0, z0])``.
    """
    import openmm

    if periodic:
        expr = "k * periodicdistance(x, y, z, x0, y0, z0)^2"
    else:
        expr = "k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    force = openmm.CustomExternalForce(expr)
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    # Set the PBC flag when the build exposes the setter; the current conda/pip
    # 8.5.x builds only expose the getter, in which case the force stays
    # non-periodic, which is the behaviour we want for position restraints.
    setter = getattr(force, "setUsesPeriodicBoundaryConditions", None)
    if callable(setter):
        setter(periodic)
    _ = force_constant_kj_per_mole_nm2  # constant is passed per-particle by callers
    return force


def note_metal_status() -> str:
    """Return a one-line summary of Metal availability for logging/reporting."""
    if has_platform("Metal"):
        return "Metal platform available (Metal-enabled OpenMM build)"
    if _is_apple_silicon():
        return (
            "No Metal platform in this OpenMM build; using OpenCL (Apple's "
            "Metal-backed runtime). See docs/metal_acceleration.md for a "
            "Metal-enabled build."
        )
    return "Not Apple Silicon; Metal not applicable"
