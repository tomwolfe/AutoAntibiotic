from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .config import CONFIG, CompoundRecord
from .io_utils import log

_HAVE_GNINA: bool = False
_HAVE_RF_SCORE: bool = False
_HAVE_TRANSFORMERS: bool = False
_HAVE_OPENMM: bool = False
_HAVE_AMBERTOOLS: bool = False

try:
    import openmm as _openmm
    import openmm.app as _openmm_app
    import openmm.unit as _openmm_unit
    _HAVE_OPENMM = True
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
            from .docking import prepare_ligand_pdbqt
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


def rescore_with_mmgbsa(
    top_candidates: List[CompoundRecord],
    receptor_pdb: str,
    work_dir: str,
) -> List[CompoundRecord]:
    """Rescore the top candidates using a simplified MM-GB/SA approach.

    Uses OpenMM's Generalized Born (OBC2) implicit solvent model to
    estimate the binding free energy for each docked pose:

        ΔG_binding ≈ G(complex) - G(receptor) - G(ligand)

    where each G = E_GB + E_MM (bonded + van der Waals + Coulomb).

    Args:
        top_candidates: Docked candidates to rescore (uses SMILES to
            generate 3D poses). Only the top 10 are rescored.
        receptor_pdb: Path to the receptor PDB file.
        work_dir: Working directory for intermediate files.

    Returns:
        Updated candidates with ``ml_score`` set to the MM-GB/SA ΔG
        (more negative = stronger binding predicted).
    """
    n = len(top_candidates)
    log.info(f"  Rescoring top {n} candidates with MM-GB/SA…")

    if not _HAVE_OPENMM:
        log.warning("  OpenMM not installed — install with: conda install -c conda-forge openmm")
        log.warning("  Falling back to existing scoring (ml_score unchanged).")
        return top_candidates

    if not os.path.exists(receptor_pdb):
        log.warning(f"  Receptor PDB not found: {receptor_pdb}. Skipping MM-GB/SA.")
        return top_candidates

    to_rescore = top_candidates[:10]
    work_dir_mm = os.path.join(work_dir, "mmgbsa")
    os.makedirs(work_dir_mm, exist_ok=True)

    try:
        from pdbfixer import PDBFixer

        # ── Prepare receptor ──
        log.info("  Preparing receptor structure with PDBFixer…")
        fixer = PDBFixer(filename=receptor_pdb)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        rec_pdb_out = os.path.join(work_dir_mm, "receptor_prepared.pdb")
        _openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(rec_pdb_out, "w"))

        forcefield = _openmm_app.ForceField("amber14-all.xml")
        cpu_platform = _openmm.Platform.getPlatformByName("CPU")

        # Receptor energy
        rec_system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=_openmm_app.NoCutoff,
            constraints=_openmm_app.HBonds,
            implicitSolvent=_openmm_app.OBC2,
        )
        rec_simulation = _openmm_app.Simulation(
            fixer.topology, rec_system,
            _openmm.LangevinMiddleIntegrator(
                300 * _openmm_unit.kelvin, 1.0 / _openmm_unit.picosecond,
                0.002 * _openmm_unit.picosecond,
            ),
            cpu_platform,
        )
        rec_simulation.context.setPositions(fixer.positions)
        rec_simulation.minimizeEnergy(maxIterations=100)
        rec_energy = rec_simulation.context.getState(
            getEnergy=True,
        ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)

        mm_gbsa_scores: Dict[str, Optional[float]] = {}

        for rank, rec in enumerate(to_rescore):
            log.info(f"  MM-GB/SA rescoring {rank + 1}/{len(to_rescore)}: {rec.compound_id}")
            try:
                mol = rec.mol
                if mol is None:
                    mol = Chem.MolFromSmiles(rec.smiles)
                    if mol is None:
                        mm_gbsa_scores[rec.compound_id] = None
                        continue
                    rec.mol = mol

                # Generate 3D conformer for the ligand
                mol_3d = Chem.RWMol(mol)
                mol_3d = Chem.AddHs(mol_3d)
                params = Chem.rdDistGeom.ETKDGv3()
                params.randomSeed = CONFIG.random_seed + rank
                if Chem.rdDistGeom.EmbedMolecule(mol_3d, params) < 0:
                    mm_gbsa_scores[rec.compound_id] = None
                    continue

                lig_pdb_file = os.path.join(work_dir_mm, f"lig_{rec.compound_id}.pdb")
                Chem.rdmolfiles.MolToPDBFile(mol_3d, lig_pdb_file)

                # Ligand energy (vacuum-like, but with GB implicit solvent)
                lig_pdb = _openmm_app.PDBFile(lig_pdb_file)
                lig_system = forcefield.createSystem(
                    lig_pdb.topology,
                    nonbondedMethod=_openmm_app.NoCutoff,
                    constraints=_openmm_app.HBonds,
                    implicitSolvent=_openmm_app.OBC2,
                )
                lig_simulation = _openmm_app.Simulation(
                    lig_pdb.topology, lig_system,
                    _openmm.LangevinMiddleIntegrator(
                        300 * _openmm_unit.kelvin, 1.0 / _openmm_unit.picosecond,
                        0.002 * _openmm_unit.picosecond,
                    ),
                    cpu_platform,
                )
                lig_simulation.context.setPositions(lig_pdb.positions)
                lig_simulation.minimizeEnergy(maxIterations=100)
                lig_energy = lig_simulation.context.getState(
                    getEnergy=True,
                ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)

                # Complex energy: concatenate receptor + ligand PDB into one file
                with open(rec_pdb_out) as f:
                    rec_pdb_lines = f.readlines()
                with open(lig_pdb_file) as f:
                    lig_pdb_lines = f.readlines()

                complex_pdb_file = os.path.join(work_dir_mm, f"complex_{rec.compound_id}.pdb")
                with open(complex_pdb_file, "w") as f:
                    for line in rec_pdb_lines:
                        if line.startswith("END") or line.startswith("TER"):
                            continue
                        f.write(line)
                    f.write("TER\n")
                    for line in lig_pdb_lines:
                        if line.startswith("END") or line.startswith("TER"):
                            continue
                        f.write(line)
                    f.write("END\n")

                complex_pdb = _openmm_app.PDBFile(complex_pdb_file)
                complex_system = forcefield.createSystem(
                    complex_pdb.topology,
                    nonbondedMethod=_openmm_app.NoCutoff,
                    constraints=_openmm_app.HBonds,
                    implicitSolvent=_openmm_app.OBC2,
                )
                complex_simulation = _openmm_app.Simulation(
                    complex_pdb.topology, complex_system,
                    _openmm.LangevinMiddleIntegrator(
                        300 * _openmm_unit.kelvin, 1.0 / _openmm_unit.picosecond,
                        0.002 * _openmm_unit.picosecond,
                    ),
                    cpu_platform,
                )
                complex_simulation.context.setPositions(complex_pdb.positions)

                # Brief energy minimisation of the complex
                complex_simulation.minimizeEnergy(maxIterations=500)

                complex_energy = complex_simulation.context.getState(
                    getEnergy=True,
                ).getPotentialEnergy().value_in_unit(_openmm_unit.kilocalorie_per_mole)

                binding_energy = complex_energy - rec_energy - lig_energy
                mm_gbsa_scores[rec.compound_id] = binding_energy
                log.info(f"    ΔG_binding ≈ {binding_energy:.2f} kcal/mol")

            except Exception as exc:
                log.warning(f"  MM-GB/SA rescoring failed for {rec.compound_id}: {exc}")
                mm_gbsa_scores[rec.compound_id] = None

        for rec in top_candidates:
            score = mm_gbsa_scores.get(rec.compound_id)
            if score is not None:
                rec.ml_score = score

    except ImportError as exc:
        log.warning(f"  PDBFixer not available ({exc}). Skipping MM-GB/SA rescoring.")
    except Exception as exc:
        log.warning(f"  MM-GB/SA rescoring failed: {exc}")

    return top_candidates


def rescore_with_ml(
    top_candidates: List[CompoundRecord],
    receptor_pdbqt: str,
    work_dir: str,
) -> List[CompoundRecord]:
    """Rescore the top Vina candidates using the best available method.

    Selection priority:
      0. **MM-GB/SA**: if ``CONFIG.use_mm_gbsa`` is set and OpenMM is
         available. Requires a receptor PDB file alongside the PDBQT.
      1. **GNINA**: if the ``gnina`` binary is on ``$PATH``.
      2. **RF-Score-VS**: a Random Forest model trained on Vina energies
         and RDKit descriptors.
      3. **ChemBERTa**: a pre-trained Transformer model regressing SMILES
         → pIC50 (requires ``transformers``).

    Each method sets ``rec.ml_score = predicted_energy_like_value``.
    Unsuccessful methods leave ``ml_score = None``, and the function
    always returns without raising.
    """
    log.info("─── ML Rescoring ───")
    n = len(top_candidates)
    log.info(f"  Rescoring {n} candidates with ML.")

    # Priority 0: MM-GB/SA (if enabled)
    if CONFIG.use_mm_gbsa:
        receptor_pdb = receptor_pdbqt.replace(".pdbqt", ".pdb")
        if os.path.exists(receptor_pdb):
            return rescore_with_mmgbsa(top_candidates, receptor_pdb, work_dir)
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
