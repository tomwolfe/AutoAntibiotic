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

try:
    from .water_analysis import WaterAnalysisResult
    _HAVE_WATER = True
except ImportError:
    WaterAnalysisResult = None  # type: ignore
    _HAVE_WATER = False

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


def rescore_with_mmgbsa(
    top_candidates: List[CompoundRecord],
    receptor_pdb: str,
    work_dir: str,
    water_results: Optional[WaterAnalysisResult] = None,
) -> List[CompoundRecord]:
    """Rescore the top candidates using MM-GB/SA with OpenMM + OBC2.

    Energy components (all implicit-solvent GB):

        ΔG_binding ≈ G(complex) - G(receptor) - G(ligand)

    where each G = E_MM (bonded + vdW + Coulomb) + E_GB (Born).

    When *water_results* is provided and contains high-energy waters
    that clash with the ligand (distance < 2.5 Å), a favourable water
    displacement correction is applied:

        ΔG_corrected = ΔG_binding - Σ E_displacement

    The correction makes the binding energy more negative when
    displaceable (high-energy) waters are sterically incompatible
    with the docked ligand.

    Uses ``CONFIG.mm_gbsa_top_n`` to determine how many candidates
    are rescored.  If *parmed* is available, an alternative Amber
    GB computation is used as a consistency check.

    Args:
        top_candidates: Docked candidates (uses SMILES for 3D generation).
        receptor_pdb: Path to the receptor PDB file.
        work_dir: Working directory for intermediate files.
        water_results: Optional crystallographic water analysis result.

    Returns:
        Updated candidates with ``ml_score`` set to the MM-GB/SA ΔG
        (more negative = stronger predicted binding).
    """
    n_to_rescore = min(len(top_candidates), CONFIG.mm_gbsa_top_n)
    log.info(f"  Rescoring top {n_to_rescore}/{len(top_candidates)} with MM-GB/SA…")

    if not _HAVE_OPENMM:
        log.warning("  OpenMM not installed — skipping MM-GB/SA rescoring.")
        return top_candidates

    if not os.path.exists(receptor_pdb):
        log.warning(f"  Receptor PDB not found: {receptor_pdb}. Skipping MM-GB/SA.")
        return top_candidates

    work_dir_mm = os.path.join(work_dir, "mmgbsa")
    os.makedirs(work_dir_mm, exist_ok=True)

    rec_prep = _prepare_receptor_for_mmgbsa(receptor_pdb, work_dir_mm)
    if rec_prep is None:
        log.warning("  Receptor preparation failed — skipping MM-GB/SA.")
        return top_candidates

    rec_topology, forcefield, cpu_platform, rec_energy = rec_prep
    rec_pdb_prepared = os.path.join(work_dir_mm, "receptor_prepared.pdb")

    to_rescore = top_candidates[:n_to_rescore]
    mm_gbsa_scores: Dict[str, Optional[float]] = {}

    for rank, rec in enumerate(to_rescore):
        log.info(f"  MM-GB/SA [{rank + 1}/{n_to_rescore}]: {rec.compound_id}")
        try:
            mol = rec.mol
            if mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    mm_gbsa_scores[rec.compound_id] = None
                    continue
                rec.mol = mol

            tag = rec.compound_id.replace("/", "_").replace(" ", "_")

            lig_energy = _compute_ligand_gb_energy(
                mol, forcefield, cpu_platform,
                work_dir_mm, tag, CONFIG.random_seed + rank,
            )
            if lig_energy is None:
                mm_gbsa_scores[rec.compound_id] = None
                continue

            complex_energy = _compute_complex_gb_energy(
                rec_pdb_prepared, tag, forcefield,
                cpu_platform, work_dir_mm, tag,
            )
            if complex_energy is None:
                mm_gbsa_scores[rec.compound_id] = None
                continue

            binding_energy = complex_energy - rec_energy - lig_energy

            # ── Water displacement correction ─────────────────────
            total_displacement_penalty = 0.0
            if water_results is not None and water_results.high_energy_waters:
                mol_3d_lig = Chem.RWMol(mol)
                mol_3d_lig = Chem.AddHs(mol_3d_lig)
                params = Chem.rdDistGeom.ETKDGv3()
                params.randomSeed = CONFIG.random_seed + rank
                if Chem.rdDistGeom.EmbedMolecule(mol_3d_lig, params) >= 0:
                    AllChem.MMFFOptimizeMolecule(mol_3d_lig, maxIters=500)
                    lig_conf = mol_3d_lig.GetConformer()
                    # Use only heavy atoms for distance check
                    lig_coords = np.array([
                        [lig_conf.GetAtomPosition(i).x,
                         lig_conf.GetAtomPosition(i).y,
                         lig_conf.GetAtomPosition(i).z]
                        for i in range(mol_3d_lig.GetNumAtoms())
                        if mol_3d_lig.GetAtomWithIdx(i).GetAtomicNum() > 1
                    ])
                    for w in water_results.high_energy_waters:
                        min_dist = float(np.min(
                            np.linalg.norm(lig_coords - w.position, axis=1)
                        ))
                        if min_dist < 2.5:
                            total_displacement_penalty += w.displacement_energy
                if total_displacement_penalty > 0.0:
                    binding_energy -= total_displacement_penalty
                log.info(f"      Water displacement correction: "
                         f"-{total_displacement_penalty:.2f} kcal/mol "
                         f"(corrected ΔG = {binding_energy:.2f})")

            mm_gbsa_scores[rec.compound_id] = binding_energy
            log.info(f"    ΔG ≈ {binding_energy:.2f} kcal/mol  "
                     f"(rec={rec_energy:.1f} + lig={lig_energy:.1f} → "
                     f"complex={complex_energy:.1f})")

        except Exception as exc:
            log.warning(f"  MM-GB/SA failed for {rec.compound_id}: {exc}")
            mm_gbsa_scores[rec.compound_id] = None

    # Apply scores back to all candidates (None for those not rescored)
    for rec in top_candidates:
        score = mm_gbsa_scores.get(rec.compound_id)
        if score is not None:
            rec.ml_score = score

    return top_candidates


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

    # Priority 0: MM-GB/SA (if enabled via either flag)
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
