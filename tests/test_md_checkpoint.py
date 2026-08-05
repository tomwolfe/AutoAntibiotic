"""Tests for the checkpointed-production helpers in scripts/explicit_solvent_md.py.

Exercises the pure helpers (``_flush_production``, ``_report_dcd``) and the
frame-list contract that the analysis phase relies on, without launching a full
419k-atom simulation.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


class _Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _FakeSim:
    def __init__(self, ckpt):
        self.ckpt = ckpt

    def saveCheckpoint(self, path):
        Path(path).write_text("ckpt")


def _load():
    path = REPO / "scripts" / "explicit_solvent_md.py"
    spec = importlib.util.spec_from_file_location("explicit_md_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# _flush_production
# ─────────────────────────────────────────────────────────────────────────
def test_flush_production_writes_header_frames_ckpt(tmp_path):
    em = _load()
    fb = tmp_path / "frames.dat"
    eb = tmp_path / "e.npy"
    rb = tmp_path / "r.npy"
    cp = tmp_path / "c.cpt"
    cj = tmp_path / "c.json"
    sim = _FakeSim(cp)
    frame = [_Vec(1, 2, 3), _Vec(4, 5, 6)]
    em._flush_production(fb, eb, rb, cp, cj, [], [], [],
                         [frame], [1.5], [0.2], sim, 25000)
    raw = np.fromfile(fb, dtype=np.float64)
    # int64 n_atoms header + n_frame * n_atom * 3 coords
    assert raw.shape[0] == 1 + 1 * 2 * 3
    assert np.array_equal(raw[1:], [1, 2, 3, 4, 5, 6])
    assert np.load(eb).tolist() == [1.5]
    assert np.load(rb).tolist() == [0.2]
    assert cp.read_text() == "ckpt"
    assert json.loads(cj.read_text()) == {"step_done": 25000}


def test_flush_production_appends_chunks(tmp_path):
    em = _load()
    fb = tmp_path / "frames.dat"
    eb = tmp_path / "e.npy"
    rb = tmp_path / "r.npy"
    cp = tmp_path / "c.cpt"
    cj = tmp_path / "c.json"
    sim = _FakeSim(cp)
    frame = [_Vec(1, 2, 3), _Vec(4, 5, 6)]
    em._flush_production(fb, eb, rb, cp, cj, [], [], [], [frame], [1.5], [0.2], sim, 1)
    em._flush_production(fb, eb, rb, cp, cj, [], [], [], [frame], [2.5], [0.3], sim, 2)
    assert np.load(eb).tolist() == [1.5, 2.5]
    assert np.load(rb).tolist() == [0.2, 0.3]
    raw = np.fromfile(fb, dtype=np.float64)
    assert raw.shape[0] == 1 + 2 * 2 * 3


# ─────────────────────────────────────────────────────────────────────────
# frames rolling-file round-trip (the contract the production/analysis uses)
# ─────────────────────────────────────────────────────────────────────────
def test_frames_file_roundtrip_preserves_frame_structure(tmp_path):
    em = _load()
    fb = tmp_path / "frames.dat"
    # header (int64 n_atoms) + one frame of n_atom=2
    with open(fb, "wb") as fh:
        fh.write(np.int64(2).tobytes())
        fh.write(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64).tobytes())
    # Replicate the rebuild used by explicit_solvent_md (frames as list-of-frames).
    with open(fb, "rb") as fh:
        nat = int(np.frombuffer(fh.read(8), dtype=np.int64)[0])
        nfr = int(os.path.getsize(fb) - 8) // (nat * 3 * 8)
        raw = np.fromfile(fh, dtype=np.float64).reshape(nfr, nat, 3)
    frames = [[_Vec(*row) for row in frame] for frame in raw]
    assert len(frames) == 1
    assert isinstance(frames[0], list)
    assert [frames[0][0].x, frames[0][0].y, frames[0][0].z] == [1, 2, 3]
    assert [frames[0][1].x, frames[0][1].y, frames[0][1].z] == [4, 5, 6]


def test_quantity_frames_matches_state_getpositions_contract():
    """Resumed rebuild frames must be Quantity-wrapped so analysis calling
    ``pos[i][j].value_in_unit(unit.angstrom)`` works (plain Vec3 floats don't)."""
    em = _load()
    from openmm import unit
    raw = np.array([[[1.0, 2.0, 3.0], [4, 5, 6]]])  # 1 frame, 2 atoms
    frames = em._quantity_frames(raw)
    assert isinstance(frames, list) and len(frames) == 1
    # 1 nm == 10 Å for the Quantity contract used by the RMSF/RMSD analysis.
    assert frames[0][0][0].value_in_unit(unit.angstrom) == 10.0
    diff = frames[0][0][0] - frames[0][1][0]  # Quantity - Quantity
    assert diff.value_in_unit(unit.angstrom) == -30.0


# ─────────────────────────────────────────────────────────────────────────
# _report_dcd (frame-list input writes an mdtraj-loadable DCD)
# ─────────────────────────────────────────────────────────────────────────
def test_report_dcd_writes_frames(tmp_path):
    import mdtraj as md
    from openmm import app
    from openmm.app import element
    em = _load()
    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("SOL", ch)
    top.addAtom("O", element.oxygen, res)
    top.addAtom("H", element.hydrogen, res)
    f0 = [_Vec(0, 0, 0), _Vec(0.1, 0, 0)]
    f1 = [_Vec(0.2, 0, 0), _Vec(0.1, 0.1, 0)]
    out = em._report_dcd(str(tmp_path / "t.dcd"), top, [f0, f1])
    traj = md.load(out, top=md.Topology.from_openmm(top))
    assert traj.xyz.shape == (2, 2, 3)