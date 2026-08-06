"""Tests for the checkpointed-production helpers in scripts/explicit_solvent_md.py.

Exercises the pure helpers (``_flush_production``, ``_report_dcd``) and the
frame-list contract that the analysis phase relies on, without launching a full
419k-atom simulation. Also tests Hydrogen Mass Repartitioning (HMR).
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
    data = json.loads(cj.read_text())
    assert data["step_done"] == 25000
    assert data["n_particles"] == 2  # falls back to the recorded frame size


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


# ─────────────────────────────────────────────────────────────────────────
# HMR: Hydrogen Mass Repartitioning
# ─────────────────────────────────────────────────────────────────────────
def test_hmr_function_exists():
    em = _load()
    assert hasattr(em, "apply_hydrogen_mass_repartitioning")
    assert hasattr(em, "HMR_TIMESTEP_PS")
    assert em.HMR_TIMESTEP_PS == 0.004


def test_hmr_constants():
    em = _load()
    assert hasattr(em, "HMR_FACTOR")
    assert em.HMR_FACTOR == 3.0
    assert hasattr(em, "HMR_MAX_NAN_RETRIES")
    assert em.HMR_MAX_NAN_RETRIES == 3


def test_apply_hmr_transfers_mass(tmp_path):
    """HMR increases hydrogen mass by factor and decreases heavy atom mass
    by the same total amount (total mass preserved)."""
    try:
        import openmm
        from openmm import app, unit
        from openmm.app import element
    except ImportError:
        pytest.skip("OpenMM not installed")
    em = _load()

    # Build a minimal topology: CH4 (C surrounded by 4 H)
    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("CH4", ch)
    c_atom = top.addAtom("C", element.carbon, res)
    h_atoms = []
    for i in range(4):
        h = top.addAtom("H", element.hydrogen, res)
        h_atoms.append(h)
    # Add bonds: C-H
    for h in h_atoms:
        top.addBond(c_atom, h)

    c = c_atom.index
    h_indices = [h.index for h in h_atoms]

    # Build a system with standard masses
    system = openmm.System()
    # 1 carbon (12 amu) + 4 hydrogens (1 amu each)
    system.addParticle(12.0)
    for _ in range(4):
        system.addParticle(1.0)

    # Apply HMR with factor=3.0
    system = em.apply_hydrogen_mass_repartitioning(system, top, factor=3.0)

    # After HMR: each H should be 3.0 amu, carbon should be 12 - 4*2 = 4.0 amu
    h_masses = [float(system.getParticleMass(i).value_in_unit(openmm.unit.dalton)) for i in h_indices]
    c_mass = float(system.getParticleMass(c).value_in_unit(openmm.unit.dalton))
    
    for h_m in h_masses:
        assert h_m == pytest.approx(3.0, abs=0.01)
    assert c_mass == pytest.approx(4.0, abs=0.01)
    
    # Total mass should be preserved: 12 + 4*1 = 16 before, 4 + 4*3 = 16 after
    total = c_mass + sum(h_masses)
    assert total == pytest.approx(16.0, abs=0.01)
    
    for h_m in h_masses:
        assert h_m == pytest.approx(3.0, abs=0.01)
    assert c_mass == pytest.approx(4.0, abs=0.01)
    
    # Total mass should be preserved: 12 + 4*1 = 16 before, 4 + 4*3 = 16 after
    total = c_mass + sum(h_masses)
    assert total == pytest.approx(16.0, abs=0.01)


def test_apply_hmr_preserves_total_mass(tmp_path):
    """Total mass of the system is preserved by HMR."""
    try:
        import openmm
        from openmm import app
        from openmm.app import element
    except ImportError:
        pytest.skip("OpenMM not installed")
    em = _load()

    top = app.Topology()
    ch = top.addChain()
    res = top.addResidue("ALA", ch)
    atoms = []
    # Simple: 1 N, 1 Cα, 1 H (attached to N), 1 H (attached to Cα)
    atoms.append(top.addAtom("N", element.nitrogen, res))
    atoms.append(top.addAtom("H", element.hydrogen, res))
    atoms.append(top.addAtom("C", element.carbon, res))
    atoms.append(top.addAtom("H", element.hydrogen, res))
    top.addBond(atoms[0], atoms[1])  # N-H
    top.addBond(atoms[2], atoms[3])  # C-H

    system = openmm.System()
    # Standard masses (approx): N=14.0, H=1.0, C=12.0, H=1.0
    system.addParticle(14.0)
    system.addParticle(1.0)
    system.addParticle(12.0)
    system.addParticle(1.0)

    initial_total = sum(float(system.getParticleMass(i).value_in_unit(openmm.unit.dalton)) for i in range(system.getNumParticles()))
    
    system = em.apply_hydrogen_mass_repartitioning(system, top, factor=3.0)
    
    final_total = sum(float(system.getParticleMass(i).value_in_unit(openmm.unit.dalton)) for i in range(system.getNumParticles()))
    
    assert final_total == pytest.approx(initial_total, rel=1e-5)
    
    # Hydrogens should be heavier
    h0_mass = float(system.getParticleMass(atoms[1].index).value_in_unit(openmm.unit.dalton))
    h1_mass = float(system.getParticleMass(atoms[3].index).value_in_unit(openmm.unit.dalton))
    assert h0_mass > 1.0
    assert h1_mass > 1.0


def test_hmr_checkpoint_json_includes_hmr_flag(tmp_path):
    """_flush_production writes the hmr flag into the checkpoint JSON."""
    em = _load()
    fb = tmp_path / "frames.dat"
    eb = tmp_path / "e.npy"
    rb = tmp_path / "r.npy"
    cp = tmp_path / "c.cpt"
    cj = tmp_path / "c.json"
    sim = _FakeSim(cp)
    frame = [_Vec(1, 2, 3), _Vec(4, 5, 6)]
    em._flush_production(fb, eb, rb, cp, cj, [], [], [],
                         [frame], [1.5], [0.2], sim, 25000, hmr=True)
    data = json.loads(cj.read_text())
    assert data["hmr"] is True
    assert data["step_done"] == 25000

    # Without HMR
    cj2 = tmp_path / "c2.json"
    em._flush_production(fb, eb, rb, cp, cj2, [], [], [],
                         [frame], [1.5], [0.2], sim, 50000, hmr=False)
    data2 = json.loads(cj2.read_text())
    assert data2["hmr"] is False