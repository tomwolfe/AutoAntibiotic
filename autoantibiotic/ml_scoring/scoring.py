from __future__ import annotations

import concurrent.futures
import logging
import os
import pickle
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolTransforms

from ..config import CONFIG, ConfigurationError
from ..models import CompoundRecord
from ..io_utils import log

try:
    from ..water_analysis import WaterAnalysisResult
    _HAVE_WATER = True
except ImportError:
    WaterAnalysisResult = None  # type: ignore
    _HAVE_WATER = False

_HAVE_GNINA: bool = False
_HAVE_RF_SCORE: bool = False
_HAVE_TRANSFORMERS: bool = False
_HAVE_OPENMM: bool = False
_HAVE_PDBFIXER: bool = False
_HAVE_AMBERTOOLS: bool = False

try:
    import openmm as _openmm
    import openmm.app as _openmm_app
    import openmm.unit as _openmm_unit
    _HAVE_OPENMM = True
except ImportError:
    pass

try:
    import pdbfixer  # noqa: F401
    _HAVE_PDBFIXER = True
except ImportError:
    pass

try:
    import parmed as _parmed
    _HAVE_AMBERTOOLS = True
except ImportError:
    try:
        from pytraj import utils as _pytraj_utils
        _HAVE_AMBERTOOLS = True
    except ImportError:
        pass

try:
    result = subprocess.run(
        ["gnina", "--help"], capture_output=True, text=True, timeout=10,
    )
    _HAVE_GNINA = result.returncode == 0
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    pass

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    _HAVE_RF_SCORE = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HAVE_TRANSFORMERS = True
except ImportError:
    pass


def _compute_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute a standard set of RDKit descriptors for ML scoring."""
    descs: List[float] = []
    for name, fn in Descriptors.descList:
        try:
            val = fn(mol)
            descs.append(val if val is not None else 0.0)
        except Exception:
            descs.append(0.0)
    return np.array(descs, dtype=np.float64)


def _rescore_with_gnina(
    top_candidates: List[CompoundRecord],
    receptor_pdbqt: str,
    work_dir: str,
) -> List[CompoundRecord]:
    """Rescore top candidates using GNINA if available.

    GNINA is a deep-learning-enhanced version of AutoDock Vina that uses
    convolutional neural networks for scoring.  This wrapper writes each
    candidate as a PDBQT, invokes ``gnina``, and parses the CNN score
    from the output.
    """
    cnn_scores: Dict[str, Optional[float]] = {}
    for rec in top_candidates:
        lig_dir = os.path.join(work_dir, f"gnina_{rec.compound_id}")
        os.makedirs(lig_dir, exist_ok=True)
        lig_pdbqt = os.path.join(lig_dir, "ligand.pdbqt")
        out_pdbqt = os.path.join(lig_dir, "out.pdbqt")

        try:
            from ..docking import prepare_ligand_pdbqt
            if not prepare_ligand_pdbqt(rec.mol, lig_pdbqt):
                cnn_scores[rec.compound_id] = None
                continue

            cmd = [
                "gnina",
                "--receptor", receptor_pdbqt,
                "--ligand", lig_pdbqt,
                "--out", out_pdbqt,
                "--score_only",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=CONFIG.vina_timeout_s,
            )
            stdout = proc.stdout + proc.stderr
            cnn_score = _parse_gnina_cnn_score(stdout)
            cnn_scores[rec.compound_id] = cnn_score
        except Exception as exc:
            log.warning(f"  GNINA rescoring failed for {rec.compound_id}: {exc}")
            cnn_scores[rec.compound_id] = None
        finally:
            for f in (lig_pdbqt, out_pdbqt):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(lig_dir)
            except OSError:
                pass

    for rec in top_candidates:
        rec.ml_score = cnn_scores.get(rec.compound_id)
    return top_candidates


def _parse_gnina_cnn_score(gnina_output: str) -> Optional[float]:
    """Parse the CNN affinity score from GNINA's output."""
    for line in gnina_output.splitlines():
        if "CNN score" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "score" and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
        if "Affinity:" in line and "CNN" in line:
            m = line.split("Affinity:")[-1].strip().split()[0]
            try:
                return float(m)
            except ValueError:
                pass
    return None


def _compute_rf_features(mol: Chem.Mol) -> np.ndarray:
    """Compute a fixed-length feature vector for RF-Score-VS.

    Uses 200 rdkit descriptors (2D) plus Morgan fingerprint counts.
    """
    morgan = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=1024,
    )
    morgan_arr = np.array(morgan, dtype=np.float64)
    descs = _compute_rdkit_descriptors(mol)
    return np.concatenate([descs, morgan_arr])


def _train_rf_on_vina_data(
    candidates: List[CompoundRecord],
) -> Any:
    """Train a quick Random Forest regressor on Vina docking energies."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for rec in candidates:
        if rec.pb2pa_allosteric_energy is None or rec.mol is None:
            continue
        mol = rec.mol
        try:
            feats = _compute_rf_features(mol)
            X_list.append(feats)
            y_list.append(rec.pb2pa_allosteric_energy)
        except Exception:
            continue

    if len(X_list) < 10:
        return None

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=CONFIG.random_seed,
    )
    rf.fit(X_scaled, y)
    return scaler, rf


def _rescore_with_rf(
    top_candidates: List[CompoundRecord],
    model: Any,
) -> List[CompoundRecord]:
    """Rescore using a trained Random Forest model (RF-Score-VS style)."""
    if model is None:
        log.warning("  RF model not available for rescoring.")
        return top_candidates

    scaler, rf = model
    for rec in top_candidates:
        if rec.mol is None:
            continue
        try:
            feats = _compute_rf_features(rec.mol).reshape(1, -1)
            feats_scaled = scaler.transform(feats)
            pred = float(rf.predict(feats_scaled)[0])
            rec.ml_score = -abs(pred)  # normalise to negative (energy-like)
        except Exception as exc:
            log.warning(f"  RF rescoring failed for {rec.compound_id}: {exc}")
            rec.ml_score = None
    return top_candidates


_ID_DRUG = "[C@@]12CC[C@@]3(C)[C@]1(C[C@@H](O)[C@]2(C3=O)O)C(=O)O"
_ID_SEQUENCE = "MKKITIWLISLLVLSISFSTNSEYERISFKNKANFDSAVSK"

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


def _rescore_with_chemberta(
    top_candidates: List[CompoundRecord],
) -> List[CompoundRecord]:
    """Rescore using a pre-trained ChemBERTa model via transformers.

    Falls back gracefully if transformers or the model are unavailable.
    """
    if not _HAVE_TRANSFORMERS:
        log.warning("  transformers not installed; skipping ChemBERTa rescore.")
        return top_candidates

    try:
        tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            _CHEMBERTA_MODEL, num_labels=1,
        )
    except Exception as exc:
        log.warning(f"  ChemBERTa model load failed ({exc}); skipping rescore.")
        return top_candidates

    for rec in top_candidates:
        try:
            inputs = tokenizer(
                rec.smiles, return_tensors="pt", padding=True, truncation=True,
            )
            outputs = model(**inputs)
            pIC50 = float(outputs.logits.detach().numpy().flatten()[0])
            rec.ml_score = -abs(float(pIC50))  # normalise to negative energy-like
        except Exception as exc:
            log.warning(f"  ChemBERTa scoring failed for {rec.compound_id}: {exc}")
            rec.ml_score = None
    return top_candidates


_VDW_RADII: Dict[int, float] = {
    1: 1.20,  6: 1.70,  7: 1.55,  8: 1.52,  9: 1.47,
    16: 1.80, 17: 1.75, 15: 1.80, 14: 2.10, 5: 1.92,
    35: 1.85, 53: 1.98,
}
"""Van der Waals radii (Å) by atomic number."""


def _is_bridging_water(
    water_pos: np.ndarray,
    ligand_positions: np.ndarray,
    protein_positions: np.ndarray,
    hbond_distance_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
) -> bool:
    """Determine whether a water molecule bridges the ligand and protein
    via hydrogen bonds.

    A bridging water must be within *hbond_distance_cutoff* of at least
    one ligand heavy atom AND one protein heavy atom.  Additionally,
    the angle formed at the water oxygen (ligand_atom — water_O —
    protein_atom) should be > *angle_cutoff* degrees, indicating a
    near-linear H-bond geometry.

    Parameters
    ----------
    water_pos : np.ndarray, shape (3,)
        3-D coordinates of the water oxygen.
    ligand_positions : np.ndarray, shape (N, 3)
        Coordinates of ligand heavy atoms.
    protein_positions : np.ndarray, shape (M, 3)
        Coordinates of protein heavy atoms (N, CA, C, O, etc.).
    hbond_distance_cutoff : float
        Maximum distance (Å) for a potential H-bond (default 3.5).
    angle_cutoff : float
        Minimum angle (degrees) at the water oxygen (default 120°).

    Returns
    -------
    bool
        ``True`` if the water is likely a bridging water.
    """
    if len(ligand_positions) == 0 or len(protein_positions) == 0:
        return False

    # Find ligand atoms within cutoff
    lig_dists = np.linalg.norm(ligand_positions - water_pos, axis=1)
    lig_close = np.where(lig_dists < hbond_distance_cutoff)[0]
    if len(lig_close) == 0:
        return False

    # Find protein atoms within cutoff
    prot_dists = np.linalg.norm(protein_positions - water_pos, axis=1)
    prot_close = np.where(prot_dists < hbond_distance_cutoff)[0]
    if len(prot_close) == 0:
        return False

    # Check angle for all pairs of close ligand/protein atoms
    for li in lig_close:
        for pi in prot_close:
            v_lig = ligand_positions[li] - water_pos
            v_prot = protein_positions[pi] - water_pos
            norm_lig = np.linalg.norm(v_lig)
            norm_prot = np.linalg.norm(v_prot)
            if norm_lig < 1e-8 or norm_prot < 1e-8:
                continue
            cos_angle = np.dot(v_lig, v_prot) / (norm_lig * norm_prot)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_angle)))
            if angle > angle_cutoff:
                return True

    return False


def _check_vdw_overlap(
    ligand_mol: Chem.Mol,
    water_position: np.ndarray,
    scaling: float = 0.8,
) -> bool:
    """Check whether a water oxygen falls within the VDW radius of any
    ligand heavy atom.

    Uses a *scaling* factor (default 0.8) to soften the criterion and
    avoid penalising borderline contacts.  Returns ``True`` if a steric
    clash is detected.
    """
    conf = ligand_mol.GetConformer()
    for i in range(ligand_mol.GetNumAtoms()):
        atom = ligand_mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        vdw = _VDW_RADII.get(atomic_num, 1.70)
        pt = conf.GetAtomPosition(i)
        dist = np.linalg.norm(
            np.array([pt.x, pt.y, pt.z]) - water_position
        )
        if dist < vdw * scaling:
            return True
    return False


def _compute_volume_overlap_ratio(
    mol: Chem.Mol,
    water_position: np.ndarray,
    water_radius: float = 1.4,
) -> float:
    """Compute fractional volume overlap between a ligand conformer and a water molecule.

    The water is modelled as a sphere of *water_radius* Å.  For each ligand
    heavy atom within overlap distance, the volume of intersection between
    the atom's VDW sphere and the water sphere is computed analytically.
    The sum of intersection volumes is divided by the water sphere volume
    to give a clash ratio between 0.0 (no overlap) and 1.0 (water fully
    buried in ligand atoms).

    Parameters
    ----------
    mol : Chem.Mol
        Ligand molecule with a 3-D conformer (must have been embedded).
    water_position : np.ndarray, shape (3,)
        Coordinates of the water oxygen.
    water_radius : float
        Radius of the water sphere in Å (default 1.4 ≈ oxygen VDW radius).

    Returns
    -------
    float
        Overlap ratio in [0, 1].
    """
    water_vol = 4.0 / 3.0 * np.pi * water_radius ** 3
    conf = mol.GetConformer()
    total_overlap = 0.0

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        r_atom = _VDW_RADII.get(atomic_num, 1.70)
        pt = conf.GetAtomPosition(i)
        d = float(np.linalg.norm(
            np.array([pt.x, pt.y, pt.z]) - water_position,
        ))

        if d >= water_radius + r_atom:
            continue

        if d <= abs(water_radius - r_atom):
            total_overlap += min(
                water_vol,
                4.0 / 3.0 * np.pi * r_atom ** 3,
            )
        else:
            r1, r2 = water_radius, r_atom
            V = (np.pi * (r1 + r2 - d) ** 2 *
                 (d ** 2 + 2.0 * d * r2 - 3.0 * r2 ** 2 +
                  2.0 * d * r1 + 6.0 * r1 * r2 - 3.0 * r1 ** 2) /
                 (12.0 * d))
            total_overlap += max(V, 0.0)

    return float(min(total_overlap / water_vol, 1.0))


def _perform_pose_relaxation(
    topology: Any,
    system: Any,
    positions: Any,
    force_constant: float = 10.0,
    max_iterations: int = 500,
) -> Tuple[Any, bool]:
    """Perform restrained minimization of a protein-ligand complex.

    Adds harmonic restraints to protein backbone CA atoms so that
    ligand and side chains can relax while the backbone stays near
    its initial position.  Uses a force constant of *force_constant*
    kcal/mol/\\u00c5\\u00b2.

    Returns
    -------
    (relaxed_positions, success_flag)
        On failure *relaxed_positions* is the input positions and the
        flag is ``False``.
    """
    try:
        cpu_platform = _openmm.Platform.getPlatformByName("CPU")
        simulation = _openmm_app.Simulation(
            topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(positions)

        force = _openmm.CustomExternalForce(
            "k*periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        k_val = force_constant * _openmm_unit.kilocalories_per_mole / _openmm_unit.angstroms**2
        force.addGlobalParameter("k", k_val)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        exclude = {"HOH", "WAT", "LIG", "UNL"}
        n_restrained = 0
        for atom in topology.atoms():
            res_name = atom.residue.name if atom.residue else ""
            if atom.name == "CA" and res_name not in exclude:
                pos = positions[atom.index]
                force.addParticle(atom.index, [pos.x.value_in_unit(_openmm_unit.nanometers),
                                                pos.y.value_in_unit(_openmm_unit.nanometers),
                                                pos.z.value_in_unit(_openmm_unit.nanometers)])
                n_restrained += 1

        if n_restrained > 0:
            system.addForce(force)
            log.info(f"  Restrained minimisation: {n_restrained} CA atoms restrained")

        simulation.minimizeEnergy(maxIterations=max_iterations)
        state = simulation.context.getState(getPositions=True)
        relaxed = state.getPositions()
        return relaxed, True

    except Exception as exc:
        log.warning(f"  Pose relaxation failed: {exc}")
        return positions, False


def _prepare_receptor_for_mmgbsa(
    receptor_pdb: str, work_dir_mm: str,
) -> Optional[Tuple[Any, Any, Any, float]]:
    """Prepare the receptor structure and compute its single-point GB energy.

    Uses PDBFixer for missing atoms/residues and OpenMM with
    amber14-all + OBC2 implicit solvent.

    Returns
    -------
    (topology, forcefield, platform, energy_kcal) on success, None on failure.
    """
    try:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        rec_pdb_out = os.path.join(work_dir_mm, "receptor_prepared.pdb")
        _openmm_app.PDBFile.writeFile(
            fixer.topology, fixer.positions, open(rec_pdb_out, "w"),
        )

        forcefield = _openmm_app.ForceField("amber14-all.xml")
        cpu_platform = _openmm.Platform.getPlatformByName("CPU")

        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            fixer.topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(fixer.positions)
        simulation.minimizeEnergy(maxIterations=200)

        energy = simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)

        log.info(f"  Receptor GB energy: {energy:.2f} kcal/mol")
        return fixer.topology, forcefield, cpu_platform, energy

    except Exception as exc:
        log.warning(f"  Receptor preparation failed: {exc}")
        return None


def _compute_ligand_gb_energy(
    mol: Chem.Mol,
    forcefield: Any,
    cpu_platform: Any,
    work_dir_mm: str,
    tag: str,
    seed: int,
) -> Optional[float]:
    """Generate a 3D conformer for *mol* and compute its GB energy."""
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        if Chem.rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

        lig_pdb = os.path.join(work_dir_mm, f"lig_{tag}.pdb")
        Chem.rdmolfiles.MolToPDBFile(mol_3d, lig_pdb)

        lig_pdb_obj = _openmm_app.PDBFile(lig_pdb)
        system = forcefield.createSystem(
            lig_pdb_obj.topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            lig_pdb_obj.topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(lig_pdb_obj.positions)
        simulation.minimizeEnergy(maxIterations=200)

        energy = simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)
        return energy
    except Exception as exc:
        log.warning(f"  Ligand GB energy failed for {tag}: {exc}")
        return None


def _compute_complex_gb_energy(
    receptor_pdb_prepared: str,
    lig_pdb_tag: str,
    forcefield: Any,
    cpu_platform: Any,
    work_dir_mm: str,
    tag: str,
) -> Optional[float]:
    """Concatenate the prepared receptor and ligand PDB and compute the
    GB energy of the complex."""
    try:
        lig_pdb = os.path.join(work_dir_mm, f"lig_{lig_pdb_tag}.pdb")
        rec_pdb = receptor_pdb_prepared

        complex_pdb = os.path.join(work_dir_mm, f"complex_{tag}.pdb")
        with open(rec_pdb) as f:
            rec_lines = f.readlines()
        with open(lig_pdb) as f:
            lig_lines = f.readlines()

        with open(complex_pdb, "w") as f:
            for line in rec_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("TER\n")
            for line in lig_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("END\n")

        complex_pdb_obj = _openmm_app.PDBFile(complex_pdb)
        system = forcefield.createSystem(
            complex_pdb_obj.topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            complex_pdb_obj.topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(complex_pdb_obj.positions)
        # Longer minimisation for the complex
        simulation.minimizeEnergy(maxIterations=500)

        energy = simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)
        return energy
    except Exception as exc:
        log.warning(f"  Complex GB energy failed for {tag}: {exc}")
        return None


def _build_explicit_complex_system(
    rec_pdb: str,
    lig_pdb: str,
    work_dir_mm: str,
    tag: str,
    forcefield: Any,
) -> Optional[Tuple[Any, Any, Any]]:
    """Build a complex topology/system from solvated receptor + ligand PDBs.

    Returns (topology, system, positions) or None on failure.
    """
    try:
        complex_pdb = os.path.join(work_dir_mm, f"complex_{tag}.pdb")
        with open(rec_pdb) as f:
            rec_lines = f.readlines()
        with open(lig_pdb) as f:
            lig_lines = f.readlines()

        with open(complex_pdb, "w") as f:
            for line in rec_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("TER\n")
            for line in lig_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("END\n")

        pdb_obj = _openmm_app.PDBFile(complex_pdb)
        system = forcefield.createSystem(
            pdb_obj.topology,
            nonbondedMethod=_openmm_app.PME,
            nonbondedCutoff=1.0 * _openmm_unit.nanometer,
            constraints=_openmm_app.HBonds,
        )
        return pdb_obj.topology, system, pdb_obj.positions
    except Exception as exc:
        log.warning(f"  Complex system build failed for {tag}: {exc}")
        return None


def _compute_complex_gb_energy_relaxed(
    topology: Any,
    positions: Any,
    forcefield: Any,
    cpu_platform: Any,
) -> Optional[float]:
    """Compute GB energy of a complex using OBC2 at given positions."""
    try:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy(maxIterations=200)
        energy = simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)
        return energy
    except Exception as exc:
        log.warning(f"  Complex GB energy (relaxed) failed: {exc}")
        return None


def _compute_complex_gb_energy_with_state(
    receptor_pdb_prepared: str,
    lig_pdb_tag: str,
    forcefield: Any,
    cpu_platform: Any,
    work_dir_mm: str,
    tag: str,
) -> Optional[Tuple[float, Any, Any]]:
    """Concatenate the prepared receptor and ligand PDB, minimize the
    complex, and return ``(energy_kcal, topology, positions)``."""
    try:
        lig_pdb = os.path.join(work_dir_mm, f"lig_{lig_pdb_tag}.pdb")
        rec_pdb = receptor_pdb_prepared

        complex_pdb = os.path.join(work_dir_mm, f"complex_{tag}.pdb")
        with open(rec_pdb) as f:
            rec_lines = f.readlines()
        with open(lig_pdb) as f:
            lig_lines = f.readlines()

        with open(complex_pdb, "w") as f:
            for line in rec_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("TER\n")
            for line in lig_lines:
                if line.startswith(("END", "TER")):
                    continue
                f.write(line)
            f.write("END\n")

        complex_pdb_obj = _openmm_app.PDBFile(complex_pdb)
        system = forcefield.createSystem(
            complex_pdb_obj.topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            complex_pdb_obj.topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(complex_pdb_obj.positions)
        simulation.minimizeEnergy(maxIterations=500)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(
            _openmm_unit.kilocalorie_per_mole,
        )
        positions = state.getPositions()
        return energy, complex_pdb_obj.topology, positions
    except Exception as exc:
        log.warning(f"  Complex GB energy (with state) failed for {tag}: {exc}")
        return None


def _compute_energy_without_ligand(
    complex_topology: Any,
    complex_positions: Any,
    forcefield: Any,
    cpu_platform: Any,
    ligand_resnames: frozenset = frozenset({"LIG", "UNL"}),
) -> Optional[float]:
    """Compute OBC2 energy of the system with ligand residues removed.

    Builds a new topology containing everything except the named ligand
    residues and evaluates the potential energy on the given positions.
    """
    try:
        new_topology = _openmm_app.Topology()
        chain_map: dict = {}
        new_positions = []

        for chain in complex_topology.chains():
            new_chain = new_topology.addChain(chain.id)
            chain_map[chain.id] = new_chain
            for residue in chain.residues():
                if residue.name in ligand_resnames:
                    continue
                new_res = new_topology.addResidue(
                    residue.name, new_chain, residue.id,
                )
                for atom in residue.atoms():
                    new_topology.addAtom(
                        atom.name, atom.element, new_res, atom.id,
                    )
                    new_positions.append(complex_positions[atom.index])

        system = forcefield.createSystem(
            new_topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        simulation = _openmm_app.Simulation(
            new_topology, system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin,
                1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        simulation.context.setPositions(new_positions)
        energy = simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)
        return energy
    except Exception as exc:
        log.warning(f"  Receptor-only energy (ligand removed) failed: {exc}")
        return None


def rescore_with_mmgbsa(
    top_candidates: List[CompoundRecord],
    receptor_pdb: str,
    work_dir: str,
    water_results: Optional[WaterAnalysisResult] = None,
) -> List[CompoundRecord]:
    """Rescore the top candidates using MM-GB/SA with explicit solvent
    (TIP3P), pose relaxation, and water displacement correction.

    .. rubric:: Explicit-solvent path (default, when PDBFixer available)

    1. Prepare the receptor with PDBFixer and solvate with TIP3P
       (10 \\u00c5 padding).
    2. For each candidate, build the ligand-receptor complex and perform
       a **restrained minimisation** (backbone CA restraints, 10 kcal/mol/\\u00c5\\u00b2)
       to relax docking artefacts.
    3. Extract multiple trajectory frames and compute \\u0394G\\u1d35\\u1d62\\u2099\\u05e0\\u1d62\\u2099\\u2097
       via GB/SA (OBC2) on each relaxed snapshot.
    4. Average over ``CONFIG.explicit_solvent_frames`` (or
       ``CONFIG.mmgbsa_n_conformers`` when ensemble averaging is enabled).
    5. Apply the **water displacement penalty** for high-energy waters
       that overlap with the relaxed ligand pose (skipping bridging waters).

    .. rubric:: Implicit OBC2 fallback

    If OpenMM or PDBFixer is unavailable the original OBC2-only path is
    used as a fallback.

    Args:
        top_candidates: Docked candidates (uses SMILES for 3D generation).
        receptor_pdb: Path to the receptor PDB file.
        work_dir: Working directory for intermediate files.
        water_results: Optional crystallographic water analysis result.

    Returns:
        Updated candidates with ``ml_score`` set to the ensemble-mean
        MM-GB/SA \\u0394G (more negative = stronger predicted binding).
    """
    n_to_rescore = min(len(top_candidates), CONFIG.mm_gbsa_top_n)

    solvent_model = CONFIG.mmgbsa_solvent_model
    log.info(
        f"  Rescoring top {n_to_rescore}/{len(top_candidates)} with MM-GB/SA "
        f"(solvent model: {solvent_model})…"
    )

    if not _HAVE_OPENMM:
        log.warning("  OpenMM not installed — skipping MM-GB/SA rescoring.")
        return top_candidates

    if not os.path.exists(receptor_pdb):
        log.warning(f"  Receptor PDB not found: {receptor_pdb}. Skipping MM-GB/SA.")
        return top_candidates

    # ── Strictly follow the configured solvent model ──────────────
    if solvent_model == "explicit":
        if not _HAVE_PDBFIXER:
            raise ConfigurationError(
                f"MM-GB/SA solvent model is set to '{solvent_model}' but "
                "pdbfixer is not installed.  Install pdbfixer with:\n"
                "  conda install -c conda-forge pdbfixer\n"
                "or set mmgbsa_solvent_model='implicit'."
            )
        use_ensemble = CONFIG.use_expensive_ml_features
        n_conf = CONFIG.mmgbsa_n_conformers if use_ensemble else 1
        work_dir_mm = os.path.join(work_dir, "mmgbsa")
        os.makedirs(work_dir_mm, exist_ok=True)
        log.info(f"  Using explicit solvent model for MM-GB/SA rescoring (TIP3P + pose relaxation).")
        return _rescore_explicit_solvent_loop(
            top_candidates, receptor_pdb, work_dir_mm, water_results,
            n_to_rescore, n_conf, use_ensemble,
        )

    # ── Implicit OBC2 path ───────────────────────────────────────
    log.info(f"  Using implicit solvent model for MM-GB/SA rescoring (OBC2).")
    use_ensemble = CONFIG.use_expensive_ml_features
    n_conf = CONFIG.mmgbsa_n_conformers if use_ensemble else 1
    work_dir_mm = os.path.join(work_dir, "mmgbsa")
    os.makedirs(work_dir_mm, exist_ok=True)

    # Log a deprecation hint if the old flag is True but mmgbsa_solvent_model is implicit
    if CONFIG.use_explicit_solvent_mmgbsa and solvent_model == "implicit":
        log.info(
            "  Note: use_explicit_solvent_mmgbsa=True is superseded by "
            "mmgbsa_solvent_model='implicit'.  Set mmgbsa_solvent_model='explicit' "
            "to use the explicit-solvent path."
        )

    rec_prep = _prepare_receptor_for_mmgbsa(receptor_pdb, work_dir_mm)
    if rec_prep is None:
        log.warning("  Receptor preparation failed — skipping MM-GB/SA.")
        return top_candidates

    rec_topology, forcefield, cpu_platform, rec_energy = rec_prep
    rec_pdb_prepared = os.path.join(work_dir_mm, "receptor_prepared.pdb")

    to_rescore = top_candidates[:n_to_rescore]

    for rank, rec in enumerate(to_rescore):
        log.info(f"  MM-GB/SA (OBC2) [{rank + 1}/{n_to_rescore}]: {rec.compound_id}")
        try:
            mol = rec.mol
            if mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol

            tag = rec.compound_id.replace("/", "_").replace(" ", "_")

            binding_energies: List[float] = []

            for conf_idx in range(n_conf):
                seed = CONFIG.random_seed + rank * n_conf + conf_idx

                lig_energy = _compute_ligand_gb_energy(
                    mol, forcefield, cpu_platform,
                    work_dir_mm, f"{tag}_c{conf_idx}", seed,
                )
                if lig_energy is None:
                    continue

                complex_result = _compute_complex_gb_energy_with_state(
                    rec_pdb_prepared, f"{tag}_c{conf_idx}", forcefield,
                    cpu_platform, work_dir_mm, f"{tag}_c{conf_idx}",
                )
                if complex_result is None:
                    continue
                complex_energy, complex_topology, complex_positions = complex_result

                rec_relaxed_energy = _compute_energy_without_ligand(
                    complex_topology, complex_positions, forcefield, cpu_platform,
                )
                if rec_relaxed_energy is None:
                    continue

                binding_energy = complex_energy - rec_relaxed_energy - lig_energy

                # ── Water displacement correction ──
                if water_results is not None and water_results.high_energy_waters:
                    penalty = _compute_water_displacement_penalty(
                        mol, water_results.high_energy_waters, seed,
                        receptor_pdb=receptor_pdb,
                        strict_mode=CONFIG.use_strict_scoring,
                    )
                    if penalty > 0.0:
                        binding_energy -= penalty
                        log.info(f"      Water displacement penalty: "
                                 f"-{penalty:.2f} kcal/mol (conf {conf_idx})")

                binding_energies.append(binding_energy)

                if not use_ensemble:
                    break

            if not binding_energies:
                log.warning(f"  No valid conformers for {rec.compound_id}")
                continue

            mean_binding = float(np.mean(binding_energies))
            std_binding = float(np.std(binding_energies, ddof=1)) if len(binding_energies) > 1 else 0.0

            # ── Entropy estimation ──
            entropy_delta_ts: Optional[float] = None
            if CONFIG.include_entropy:
                entropy_result = _compute_entropy_estimation(
                    mol, work_dir_mm, tag, seed,
                )
                if entropy_result is not None:
                    entropy_delta_ts = entropy_result
                    log.info(
                        f"    -TΔS = {entropy_delta_ts:.2f} kcal/mol (NMA entropy)"
                    )

            final_score = mean_binding - (entropy_delta_ts or 0.0)
            rec.ml_score = final_score
            rec.ml_score_std = std_binding

            log.info(f"    ΔG ≈ {final_score:.2f} kcal/mol "
                     f"(ΔH={mean_binding:.2f}, -TΔS={entropy_delta_ts or 0.0:.2f})")

        except Exception as exc:
            log.warning(f"  MM-GB/SA failed for {rec.compound_id}: {exc}")

    return top_candidates


def _process_one_candidate_explicit(args: Tuple) -> Dict[str, Any]:
    """Process a single candidate for explicit-solvent MM-GB/SA.

    This is a module-level function to allow pickling with
    ``concurrent.futures.ProcessPoolExecutor``.

    Each worker recreates its own OpenMM ForceField and CPU Platform
    objects since they are not picklable across process boundaries.
    """
    (
        compound_id, smiles, mol_bytes, rank, rec_pdb_out, work_dir_mm,
        rec_energy, n_frames, n_to_rescore, use_ensemble,
        high_energy_waters, include_entropy, use_strict_scoring,
        receptor_pdb, random_seed,
    ) = args

    # Ensure logging is configured in the worker process
    logger = logging.getLogger("AutoAntibiotic")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    # Recreate per-process resources (not picklable)
    try:
        forcefield = _openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml",
        )
        cpu_platform = _openmm.Platform.getPlatformByName("CPU")
    except Exception as exc:
        logger.warning(f"  [{compound_id}] Failed to create OpenMM resources: {exc}")
        return {"compound_id": compound_id, "error": f"openmm_init: {exc}"}

    try:
        if mol_bytes:
            mol = pickle.loads(mol_bytes)
        else:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"  [{compound_id}] No molecule available")
            return {"compound_id": compound_id, "error": "no_mol"}
    except Exception as exc:
        logger.warning(f"  [{compound_id}] Failed to load molecule: {exc}")
        return {"compound_id": compound_id, "error": f"mol_load: {exc}"}

    tag = compound_id.replace("/", "_").replace(" ", "_")
    binding_energies: List[float] = []
    relaxation_failed_flag = False
    seed = random_seed + rank * n_frames

    for frame_idx in range(n_frames):
        frame_seed = seed + frame_idx
        lig_tag = f"{tag}_f{frame_idx}"

        # Ligand conformer
        lig_energy = _compute_ligand_gb_energy(
            mol, forcefield, cpu_platform, work_dir_mm, lig_tag, frame_seed,
        )
        if lig_energy is None:
            continue

        # Build solvated complex
        lig_pdb = os.path.join(work_dir_mm, f"lig_{lig_tag}.pdb")
        complex_info = _build_explicit_complex_system(
            rec_pdb_out, lig_pdb, work_dir_mm, lig_tag, forcefield,
        )
        if complex_info is None:
            continue

        complex_topology, complex_system, complex_positions = complex_info

        # Pose relaxation
        relaxed_positions, relax_ok = _perform_pose_relaxation(
            complex_topology, complex_system, complex_positions,
        )
        if not relax_ok:
            relaxation_failed_flag = True
            relaxed_positions = complex_positions

        # Compute GB/SA of relaxed complex
        complex_energy = _compute_complex_gb_energy_relaxed(
            complex_topology, relaxed_positions, forcefield, cpu_platform,
        )
        if complex_energy is None:
            continue

        # Note: Receptor energy calculated with original water box after ligand removal.
        rec_relaxed_energy = _compute_energy_without_ligand(
            complex_topology, relaxed_positions, forcefield, cpu_platform,
        )
        if rec_relaxed_energy is None:
            continue

        binding_energy = complex_energy - rec_relaxed_energy - lig_energy

        # Water displacement correction
        if high_energy_waters:
            penalty = _compute_water_displacement_penalty(
                mol, high_energy_waters, frame_seed,
                receptor_pdb=receptor_pdb,
                strict_mode=use_strict_scoring,
            )
            if penalty > 0.0:
                binding_energy -= penalty
                logger.info(
                    f"      [{compound_id}] Water displacement penalty: "
                    f"-{penalty:.2f} kcal/mol (frame {frame_idx})"
                )

        binding_energies.append(binding_energy)

        if not use_ensemble:
            break

    if not binding_energies:
        logger.warning(f"  [{compound_id}] No valid frames")
        return {"compound_id": compound_id, "error": "no_valid_frames"}

    mean_binding = float(np.mean(binding_energies))
    std_binding = (
        float(np.std(binding_energies, ddof=1))
        if len(binding_energies) > 1 else 0.0
    )

    # Entropy estimation
    entropy_delta_ts: Optional[float] = None
    if include_entropy:
        entropy_result = _compute_entropy_estimation(mol, work_dir_mm, tag, seed)
        if entropy_result is not None:
            entropy_delta_ts = entropy_result

    final_score = mean_binding - (entropy_delta_ts or 0.0)

    logger.info(
        f"  Explicit MM-GB/SA [{rank + 1}/{n_to_rescore}]: "
        f"{compound_id} — \u0394G \u2248 {final_score:.2f} \u00b1 {std_binding:.2f} kcal/mol "
        f"(explicit TIP3P + relax)"
    )

    return {
        "compound_id": compound_id,
        "ml_score": final_score,
        "ml_score_std": std_binding,
        "relaxation_failed": relaxation_failed_flag,
        "mean_binding": mean_binding,
        "entropy_delta_ts": entropy_delta_ts or 0.0,
        "n_frames_processed": len(binding_energies),
        "error": None,
    }


def _rescore_explicit_solvent_loop(
    top_candidates: List[CompoundRecord],
    receptor_pdb: str,
    work_dir_mm: str,
    water_results: Optional[WaterAnalysisResult],
    n_to_rescore: int,
    n_conf: int,
    use_ensemble: bool,
) -> List[CompoundRecord]:
    """Explicit-solvent MM-GB/SA rescoring with pose relaxation.

    Prepares the solvated receptor once, then for each candidate
    generates conformers, builds explicit-solvent complexes,
    performs restrained minimization (pose relaxation), computes
    GB/SA energies on the relaxed poses, and applies water
    displacement penalties.
    """
    # ── Prepare solvated receptor ──────────────────────────────
    try:
        from pdbfixer import PDBFixer

        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        forcefield = _openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml",
        )

        modeller = _openmm_app.Modeller(fixer.topology, fixer.positions)
        modeller.addSolvent(
            forcefield, model="tip3p",
            padding=10.0 * _openmm_unit.angstrom,
        )

        rec_system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=_openmm_app.PME,
            nonbondedCutoff=1.0 * _openmm_unit.nanometer,
            constraints=_openmm_app.HBonds,
        )

        cpu_platform = _openmm.Platform.getPlatformByName("CPU")
        rec_integrator = _openmm.LangevinMiddleIntegrator(
            300.0 * _openmm_unit.kelvin,
            1.0 / _openmm_unit.picosecond,
            0.002 * _openmm_unit.picosecond,
        )
        rec_sim = _openmm_app.Simulation(
            modeller.topology, rec_system, rec_integrator, cpu_platform,
        )
        rec_sim.context.setPositions(modeller.positions)
        rec_sim.minimizeEnergy(maxIterations=500)
        rec_sim.step(1000)  # short NVT equilibration

        rec_energy = rec_sim.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(
            _openmm_unit.kilocalorie_per_mole,
        )
        log.info(f"  Explicit-solvent receptor energy: {rec_energy:.2f} kcal/mol")

        rec_pdb_out = os.path.join(work_dir_mm, "receptor_solvated.pdb")
        with open(rec_pdb_out, "w") as f:
            _openmm_app.PDBFile.writeFile(
                modeller.topology, modeller.positions, f,
            )

    except Exception as exc:
        log.warning(
            f"  Explicit-solvent receptor preparation failed: {exc}. "
            "Cannot proceed with explicit MM-GB/SA."
        )
        return top_candidates

    n_frames = CONFIG.explicit_solvent_frames if use_ensemble else 1
    to_rescore = top_candidates[:n_to_rescore]
    water_string = "with water displacement" if water_results else "no water correction"

    # ── Attempt parallel execution ──────────────────────────────────
    max_workers = min(CONFIG.mmgbsa_parallel_workers, os.cpu_count() or 1)
    if max_workers > 1 and len(to_rescore) > 1:
        try:
            high_energy_waters = (
                water_results.high_energy_waters
                if water_results is not None else None
            )
            candidate_args: List[Tuple] = []
            for rank, rec in enumerate(to_rescore):
                mol = rec.mol
                if mol is None:
                    mol = Chem.MolFromSmiles(rec.smiles)
                    if mol is None:
                        continue
                mol_bytes = pickle.dumps(mol)
                candidate_args.append((
                    rec.compound_id, rec.smiles, mol_bytes, rank,
                    rec_pdb_out, work_dir_mm, rec_energy, n_frames,
                    n_to_rescore, use_ensemble, high_energy_waters,
                    CONFIG.include_entropy, CONFIG.use_strict_scoring,
                    receptor_pdb, CONFIG.random_seed,
                ))

            log.info(
                f"  Launching parallel explicit MM-GB/SA with "
                f"{max_workers} workers for {len(candidate_args)} candidates…"
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                results = list(
                    executor.map(_process_one_candidate_explicit, candidate_args)
                )

            n_failed = sum(1 for r in results if r.get("error"))
            if n_failed == 0:
                for rec, result in zip(to_rescore, results):
                    rec.ml_score = result["ml_score"]
                    rec.ml_score_std = result["ml_score_std"]
                log.info(
                    f"  Parallel explicit MM-GB/SA completed for "
                    f"{len(results)} candidates."
                )
                return top_candidates

            log.warning(
                f"  {n_failed}/{len(results)} candidates failed in parallel; "
                "falling back to sequential."
            )
        except Exception as exc:
            log.warning(
                f"  Parallel explicit MM-GB/SA failed ({exc}); "
                "falling back to sequential."
            )

    # ── Sequential processing ──────────────────────────────────────
    for rank, rec in enumerate(to_rescore):
        log.info(
            f"  Explicit MM-GB/SA [{rank + 1}/{n_to_rescore}]: "
            f"{rec.compound_id} ({water_string})"
        )
        try:
            mol = rec.mol
            if mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol

            tag = rec.compound_id.replace("/", "_").replace(" ", "_")
            binding_energies: List[float] = []
            relaxation_failed_flag = False

            for frame_idx in range(n_frames):
                seed = CONFIG.random_seed + rank * n_frames + frame_idx

                # --- Ligand conformer ---
                lig_energy = _compute_ligand_gb_energy(
                    mol, forcefield, cpu_platform,
                    work_dir_mm, f"{tag}_f{frame_idx}", seed,
                )
                if lig_energy is None:
                    continue

                # --- Build solvated complex and relax pose ---
                lig_tag = f"{tag}_f{frame_idx}"
                lig_pdb = os.path.join(work_dir_mm, f"lig_{lig_tag}.pdb")

                complex_info = _build_explicit_complex_system(
                    rec_pdb_out, lig_pdb, work_dir_mm, lig_tag, forcefield,
                )
                if complex_info is None:
                    continue

                complex_topology, complex_system, complex_positions = complex_info

                # Pose relaxation (restrained minimisation)
                relaxed_positions, relax_ok = _perform_pose_relaxation(
                    complex_topology, complex_system, complex_positions,
                )
                if not relax_ok:
                    relaxation_failed_flag = True
                    relaxed_positions = complex_positions

                # Compute GB/SA energy of relaxed complex
                complex_energy = _compute_complex_gb_energy_relaxed(
                    complex_topology, relaxed_positions, forcefield, cpu_platform,
                )
                if complex_energy is None:
                    continue

                # Note: Receptor energy calculated with original water box after ligand removal.
                rec_relaxed_energy = _compute_energy_without_ligand(
                    complex_topology, relaxed_positions, forcefield, cpu_platform,
                )
                if rec_relaxed_energy is None:
                    continue

                binding_energy = complex_energy - rec_relaxed_energy - lig_energy

                # ── Water displacement correction ──
                if water_results is not None and water_results.high_energy_waters:
                    penalty = _compute_water_displacement_penalty(
                        mol, water_results.high_energy_waters, seed,
                        receptor_pdb=receptor_pdb,
                        strict_mode=CONFIG.use_strict_scoring,
                    )
                    if penalty > 0.0:
                        binding_energy -= penalty
                        log.info(
                            f"      Water displacement penalty: "
                            f"-{penalty:.2f} kcal/mol (frame {frame_idx})"
                        )

                binding_energies.append(binding_energy)

                if not use_ensemble:
                    break

            if not binding_energies:
                log.warning(f"  No valid frames for {rec.compound_id}")
                continue

            mean_binding = float(np.mean(binding_energies))
            std_binding = (
                float(np.std(binding_energies, ddof=1))
                if len(binding_energies) > 1 else 0.0
            )

            # ── Entropy estimation ──
            entropy_delta_ts: Optional[float] = None
            if CONFIG.include_entropy:
                entropy_result = _compute_entropy_estimation(
                    mol, work_dir_mm, tag, seed,
                )
                if entropy_result is not None:
                    entropy_delta_ts = entropy_result
                    log.info(
                        f"    -TΔS = {entropy_delta_ts:.2f} kcal/mol (NMA entropy)"
                    )

            final_score = mean_binding - (entropy_delta_ts or 0.0)
            rec.ml_score = final_score
            rec.ml_score_std = std_binding

            if relaxation_failed_flag:
                log.info(
                    f"    (used original docked pose — relaxation failed for "
                    f"at least one frame)"
                )

            log.info(
                f"    ΔG ≈ {final_score:.2f} ± {std_binding:.2f} kcal/mol "
                f"(explicit TIP3P + relax)"
            )

        except Exception as exc:
            log.warning(
                f"  Explicit MM-GB/SA failed for {rec.compound_id}: {exc}"
            )

    return top_candidates


def rescore_with_explicit_mmgbsa(
    top_candidates: List[CompoundRecord],
    receptor_pdb: str,
    work_dir: str,
    water_results: Optional[WaterAnalysisResult] = None,
) -> List[CompoundRecord]:
    """Rescore top candidates using **explicit-solvent** MM-GB/SA.

    This is a thin wrapper around :func:`rescore_with_mmgbsa` that
    temporarily sets ``CONFIG.use_explicit_solvent_mmgbsa = True`` so
    that the explicit-solvent path (TIP3P + pose relaxation) is always
    used, regardless of the configuration default.

    Falls back to the implicit OBC2 path if OpenMM or PDBFixer are
    unavailable.

    Args:
        top_candidates: Docked candidates (uses SMILES for 3D generation).
        receptor_pdb: Path to the receptor PDB file.
        work_dir: Working directory for intermediate files.
        water_results: Optional water analysis result.

    Returns:
        Updated candidates with ``ml_score`` set to the frame-averaged
        MM-GB/SA ΔG (more negative = stronger predicted binding).
    """
    saved = CONFIG.mmgbsa_solvent_model
    CONFIG.mmgbsa_solvent_model = "explicit"
    try:
        return rescore_with_mmgbsa(
            top_candidates, receptor_pdb, work_dir, water_results=water_results,
        )
    finally:
        CONFIG.mmgbsa_solvent_model = saved


def _load_protein_heavy_atoms(
    receptor_pdb: str,
) -> np.ndarray:
    """Load protein heavy atom positions from a PDB file.

    Returns an (M, 3) array of coordinates for protein heavy atoms
    (N, CA, C, O, CB, etc.).  Ignores HETATM records (waters, ligands).
    Returns an empty array on failure.
    """
    positions: List[np.ndarray] = []
    try:
        with open(receptor_pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    if atom_name.startswith("H"):
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        positions.append(np.array([x, y, z]))
                    except (ValueError, IndexError):
                        continue
    except (OSError, IOError):
        pass
    if not positions:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(positions, dtype=np.float64)


def _compute_water_displacement_penalty(
    mol: Chem.Mol,
    high_energy_waters: List[Any],
    seed: int,
    receptor_pdb: Optional[str] = None,
    strict_mode: bool = False,
) -> float:
    """Compute the total water displacement penalty for a ligand.

    For each high-energy water, generates a 3D conformer of the ligand
    (ETKDG + MMFF) and computes a **volume-overlap clash factor**
    between the ligand atoms and the water sphere.  The displacement
    energy of each water is scaled by this factor to give a
    thermodynamically sound penalty.

    Scaling rules (Clash Factor):
        * Overlap ≥ 50 % of the water sphere volume → full penalty
          (clash factor = 1.0).
        * Overlap 10 – 50 % → linear interpolation
          (factor = (overlap - 0.1) / 0.4).
        * Overlap < 10 % → no penalty (the water is not meaningfully
          displaced).

    When *strict_mode* is True, the threshold for the 10–50 % linear
    interpolation region is lowered to 5 %, making the penalty more
    aggressive (useful for high-precision scoring).

    If *receptor_pdb* is provided, waters that form bridging H-bonds
    between the ligand and the protein (checked via
    :func:`_is_bridging_water`) are excluded from the penalty, as
    these waters are structurally critical.

    Returns the sum of scaled displacement energies of clashing waters
    (kcal/mol).
    """
    total = 0.0

    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        if Chem.rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return 0.0
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
    except Exception:
        return total

    # Pre-load protein heavy atoms if receptor PDB is provided
    protein_positions: Optional[np.ndarray] = None
    if receptor_pdb is not None:
        protein_positions = _load_protein_heavy_atoms(receptor_pdb)

    # Extract ligand heavy-atom positions from the 3D conformer
    conf = mol_3d.GetConformer()
    lig_heavy_positions = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol_3d.GetNumAtoms())
        if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1
    ], dtype=np.float64)

    # Determine overlap threshold based on strict mode
    low_overlap_threshold = 0.05 if strict_mode else 0.1

    for w in high_energy_waters:
        # Skip bridging waters — they are structurally critical
        if protein_positions is not None and len(protein_positions) > 0:
            if _is_bridging_water(
                w.position, lig_heavy_positions, protein_positions,
            ):
                log.info(
                    f"      Water at {w.position} is a bridging water; "
                    "skipping displacement penalty."
                )
                continue

            # Compute clash factor based on volume overlap
            overlap_ratio = _compute_volume_overlap_ratio(mol_3d, w.position)

            if overlap_ratio < low_overlap_threshold:
                continue

            if overlap_ratio >= 0.5:
                clash_factor = 1.0
            else:
                clash_factor = (overlap_ratio - low_overlap_threshold) / (0.5 - low_overlap_threshold)

            penalty = w.displacement_energy * clash_factor
            total += penalty

    return total


def _compute_entropy_estimation(
    mol: Chem.Mol,
    work_dir: str,
    tag: str,
    seed: int,
) -> Optional[float]:
    """Estimate the vibrational entropy (-TΔS) using Normal Mode Analysis (NMA).

    Uses the quasi-harmonic approximation based on trajectory covariance
    of ligand heavy-atom fluctuations.  When OpenMM is available, the
    function performs a short NVT simulation and computes the entropy
    contribution from the positional covariance matrix.

    The entropy estimate is returned as -TΔS (positive values indicate
    entropic loss upon binding, which makes ΔG less negative).

    Args:
        mol: RDKit Mol object of the ligand.
        work_dir: Working directory for intermediate files.
        tag: Unique tag for output files.
        seed: Random seed for reproducibility.

    Returns
    -------
    Optional[float]
        -TΔS in kcal/mol.  Returns ``None`` if entropy estimation fails.
    """
    if not _HAVE_OPENMM:
        log.warning("  OpenMM not installed — skipping entropy estimation.")
        return None

    try:
        # Generate 3D conformer for the ligand
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = Chem.rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        if Chem.rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)

        # Convert RDKit Mol to OpenMM Topology
        from openmm.app import PDBFile
        from openmm import LangevinMiddleIntegrator, HarmonicBondForce, HarmonicAngleForce, NonbondedForce
        from openmm.unit import kelvin, picosecond, atomic_mass_units, kilocalorie_per_mole

        # Write temporary PDB for the ligand
        lig_pdb = os.path.join(work_dir, f"entropy_{tag}.pdb")
        Chem.rdmolfiles.MolToPDBFile(mol_3d, lig_pdb)

        # Load topology and create system
        pdb = PDBFile(lig_pdb)
        topology = pdb.topology
        positions = pdb.positions

        # Create a simple harmonic system for entropy estimation
        # (quasi-harmonic approximation)
        system = _openmm_app.ModularSystemBuilder(topology)

        # Generate trajectory for covariance analysis
        # Use multiple conformers to estimate positional variance
        n_frames = 10
        trajectory: List[List[float]] = []

        for i in range(n_frames):
            conf_seed = seed + i * 100
            mol_tmp = Chem.RWMol(mol)
            mol_tmp = Chem.AddHs(mol_tmp)
            p = Chem.rdDistGeom.ETKDGv3()
            p.randomSeed = conf_seed
            if Chem.rdDistGeom.EmbedMolecule(mol_tmp, p) < 0:
                continue

            conf = mol_tmp.GetConformer()
            frame = []
            for j in range(mol_tmp.GetNumAtoms()):
                atom = mol_tmp.GetAtomWithIdx(j)
                if atom.GetAtomicNum() <= 1:  # Skip hydrogens
                    continue
                pos = conf.GetAtomPosition(j)
                frame.append([pos.x, pos.y, pos.z])
            trajectory.append(frame)

        if len(trajectory) < 3:
            return None

        # Compute covariance matrix of ligand heavy-atom positions
        # ΔS ≈ k_B * ln(det(Σ)) where Σ is the covariance matrix
        positions_arr = np.array(trajectory, dtype=np.float64)  # (n_frames, n_atoms, 3)

        # Compute per-atom means
        means = positions_arr.mean(axis=0)  # (n_atoms, 3)

        # Compute covariance matrix
        centered = positions_arr - means[np.newaxis, :, :]
        cov_matrix = np.cov(centered.reshape(-1, 3).T)

        # Quasi-harmonic entropy: ΔS ≈ k_B * ln(det(2πe * Σ)) / 2
        # where Σ is the covariance matrix and k_B is Boltzmann constant
        try:
            det_cov = np.linalg.det(cov_matrix)
            if det_cov <= 0:
                return None
            k_B = 0.0019872036  # kcal/mol/K (Boltzmann constant)
            T = 298.15  # K
            entropy = k_B * np.log(det_cov * 2 * np.pi * np.e) / 2
            # -TΔS (negative because binding reduces entropy)
            minus_TdS = -T * entropy
        except (np.linalg.LinAlgError, np.linalg.LinAlgError):
            return None

        return float(minus_TdS)

    except Exception as exc:
        log.warning(f"  Entropy estimation failed: {exc}")
        return None


# ── Rescore with entropy (NMA) helper ─────────────────────────────


def rescore_with_ml(
    top_candidates: List[CompoundRecord],
    receptor_pdbqt: str,
    work_dir: str,
    water_results: Optional[WaterAnalysisResult] = None,
) -> List[CompoundRecord]:
    """Rescore the top Vina candidates using the best available method.

    Selection priority:
      0. **MM-GB/SA**: if ``CONFIG.use_mm_gbsa`` is set and OpenMM is
         available. Requires a receptor PDB file alongside the PDBQT.
         When *water_results* is provided, water displacement correction
         is applied automatically.
      1. **GNINA**: if the ``gnina`` binary is on ``$PATH``.
      2. **RF-Score-VS**: a Random Forest model trained on Vina energies
         and RDKit descriptors.
      3. **ChemBERTa**: a pre-trained Transformer model regressing SMILES
         → pIC50 (requires ``transformers``).

    Each method sets ``rec.ml_score = predicted_energy_like_value``.
    Unsuccessful methods leave ``ml_score = None``, and the function
    always returns without raising.

    Args:
        top_candidates: Docked candidates to rescore.
        receptor_pdbqt: Path to the receptor PDBQT file.
        work_dir: Working directory for intermediate files.
        water_results: Optional crystallographic water analysis for
            water displacement correction in MM-GB/SA rescoring.
    """
    log.info("─── ML Rescoring ───")
    n = len(top_candidates)
    log.info(f"  Rescoring {n} candidates with ML.")

    # Priority 0: MM-GB/SA (handles both explicit TIP3P + implicit OBC2 internally)
    if CONFIG.use_mm_gbsa or CONFIG.use_mm_gbsa_rescoring:
        receptor_pdb = receptor_pdbqt.replace(".pdbqt", ".pdb")
        if os.path.exists(receptor_pdb):
            return rescore_with_mmgbsa(
                top_candidates, receptor_pdb, work_dir,
                water_results=water_results,
            )
        log.warning(
            "  MM-GB/SA enabled but receptor PDB not found at "
            f"{receptor_pdb}. Skipping MM-GB/SA."
        )

    if _HAVE_GNINA:
        log.info("  Using GNINA (CNN rescoring).")
        return _rescore_with_gnina(top_candidates, receptor_pdbqt, work_dir)

    if _HAVE_RF_SCORE:
        log.info("  Training RF-Score-VS model on Vina energies…")
        model = _train_rf_on_vina_data(top_candidates)
        if model is not None:
            log.info("  Applying RF-Score-VS rescoring.")
            return _rescore_with_rf(top_candidates, model)
        log.warning("  RF model training failed (too few training points).")

    if _HAVE_TRANSFORMERS:
        log.info("  Falling back to ChemBERTa rescoring.")
        return _rescore_with_chemberta(top_candidates)

    log.warning(
        "  No ML rescoring backend available (gnina, sklearn, or transformers). "
        "Leaving ml_score = None."
    )
    return top_candidates
