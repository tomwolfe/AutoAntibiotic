"""Unit tests for the trajectory binding-stability analysis helpers.

Pure-math tests for block averaging, running means, and integrated
autocorrelation time; they do not read MD output or launch simulations.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ma():
    path = REPO / "scripts" / "md_analysis.py"
    spec = spec_from_file_location("md_analysis_mod", str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compute_block_stats_contiguous_and_counts(ma):
    rmsd = np.arange(10, dtype=float)
    blocks = ma.compute_block_stats(rmsd, n_blocks=5)
    assert len(blocks) == 5
    # Each block is mean of its 2-element segment, in order.
    for b in blocks:
        assert b["frames"] == 2
    assert blocks[0]["mean_A"] == pytest.approx(np.mean([0.0, 1.0]))
    assert blocks[-1]["mean_A"] == pytest.approx(np.mean([8.0, 9.0]))
    # Blocks tile the whole series without overlap.
    starts = [b["start_frame"] for b in blocks]
    ends = [b["end_frame"] for b in blocks]
    assert starts[0] == 0 and ends[-1] == 9


def test_compute_block_stats_exceeds_length(ma):
    # More blocks than frames -> fall back to 1 block (never empty).
    blocks = ma.compute_block_stats(np.array([1.0, 2.0]), n_blocks=10)
    assert len(blocks) == 2
    assert blocks[0]["mean_A"] == pytest.approx(1.0)


def test_compute_block_stats_empty(ma):
    assert ma.compute_block_stats(np.array([]), n_blocks=5) == []


def test_running_mean_pads_and_smooths(ma):
    x = np.array([0.0, 10.0, 0.0, 10.0], dtype=float)
    win, run = ma.running_mean(x, window_ns=1.0, dt_ps=10.0)  # win == 100? clamped to n=4
    assert win == 4  # clamped to series length
    assert run.shape == x.shape
    assert run[-1] == pytest.approx(np.mean(x))
    assert np.isnan(run[0])


def test_autocorrelation_time_constant_series(ma):
    # Constant series -> undefined (no variance); returns None without raising.
    res = ma.autocorrelation_time(np.ones(50), dt_ps=10.0)
    assert res["tau_ps"] is None


def test_autocorrelation_time_dimensionality(ma):
    # Clean exponential decay -> finite, positive tau in the right units.
    n = 100
    t = np.arange(n)
    rmsd = 2.0 + np.exp(-t / 10.0) * 1.0
    res = ma.autocorrelation_time(rmsd, dt_ps=10.0)
    assert res["tau_ps"] is not None and res["tau_ps"] > 0
    # tau scales linearly with the step when the underlying series is identical.
    res2 = ma.autocorrelation_time(rmsd, dt_ps=20.0)
    assert res2["tau_ps"] == pytest.approx(2 * res["tau_ps"])


def test_frame_dt_from_summary(ma):
    summary = {"production": {"npt_duration_ns": 100.0, "n_frames": 10000}}
    assert ma._frame_dt_ps(10000, summary) == pytest.approx(10.0)
    assert ma._frame_dt_ps(None, summary) == pytest.approx(10.0)
    assert ma._frame_dt_ps(5, None) == pytest.approx(2.0)  # default fallback (2 ps/frame)