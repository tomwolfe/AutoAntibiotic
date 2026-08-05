"""Unit tests for the OpenMM platform-selection module (utils/openmm_platform.py).

Pure-function tests; they load OpenMM (available in the science env) but do
not launch simulations.
"""
import pytest

from utils.openmm_platform import (
    GPU_PLATFORMS,
    PREFERENCE_ORDER,
    available_platforms,
    has_platform,
    note_metal_status,
    position_restraint_force,
    select_platform,
)


def test_preference_order_starts_accelerator_first():
    # Accelerators must be preferred over the always-available CPU fallback.
    assert PREFERENCE_ORDER[0] in ("Metal", "CUDA", "OpenCL")
    assert PREFERENCE_ORDER[-1] == "CPU"


def test_available_platforms_returns_preferred_order_first():
    avail = available_platforms()
    assert avail, "at least the CPU platform must be registered"
    ranked = [p for p in PREFERENCE_ORDER if p in avail]
    # The registered subset must appear in preference order at the front.
    assert avail[: len(ranked)] == ranked


def test_has_platform_cpu():
    # CPU is always registered by every OpenMM build.
    assert has_platform("CPU")


def test_select_platform_returns_valid_spec():
    spec = select_platform()
    assert spec["name"] in available_platforms()
    assert spec["platform"].getName() == spec["name"]
    assert isinstance(spec["properties"], dict)
    assert isinstance(spec["reason"], str)
    assert spec["fallback_chain"] == []


def test_select_platform_explicit_override():
    if not has_platform("OpenCL"):
        pytest.skip("OpenCL platform not registered in this OpenMM build")
    spec = select_platform(preference="OpenCL")
    assert spec["name"] == "OpenCL"
    assert spec["reason"].startswith("explicit")


def test_select_platform_unknown_preference_falls_back():
    spec = select_platform(preference="NotARealPlatform")
    assert spec["name"] in available_platforms()


def test_position_restraint_force_periodic_uses_periodicdistance():
    force = position_restraint_force(4184.0, periodic=True)
    assert "periodicdistance" in force.getEnergyFunction()
    params = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
    assert params == ["k", "x0", "y0", "z0"]


def test_position_restraint_force_nonperiodic_uses_absolute_form():
    force = position_restraint_force(4184.0, periodic=False)
    assert "periodicdistance" not in force.getEnergyFunction()
    assert "(x - x0)^2" in force.getEnergyFunction()


def test_gpu_platforms_require_periodic_restraints():
    # Every GPU backend uses the wrapped-cell convention, so they must be
    # flagged as needing the periodicdistance() restraint expression.
    assert "OpenCL" in GPU_PLATFORMS
    assert "Metal" in GPU_PLATFORMS
    assert "CUDA" in GPU_PLATFORMS


def test_note_metal_status_returns_string():
    msg = note_metal_status()
    assert isinstance(msg, str) and msg
