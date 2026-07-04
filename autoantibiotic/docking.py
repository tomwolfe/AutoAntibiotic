from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from .config import CONFIG, CompoundRecord
from .io_utils import log, parse_vina_energy, run_tool

try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    _tqdm = lambda x, **kw: x


def _extract_native_ligand_from_holo(
    holo_pdb_path: str,
    output_ligand_smi: str,
    output_ligand_pdbqt: str,
) -> Optional[str]:
    """Parse the holo structure (6TKO), locate the co-crystallised ligand,
    write its SMILES to *output_ligand_smi* and its PDBQT to *output_ligand_pdbqt*."""
    try:
        from Bio.PDB import PDBIO, PDBParser, Select

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("6TKO", holo_pdb_path)

        ligand_residues: list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    het_flag = residue.get_id()[0]
                    if het_flag == " " or het_flag == "W":
                        continue
                    resname = residue.get_resname().strip()
                    if resname in ("HOH", "WAT", "SOL"):
                        continue
                    ligand_residues.append((chain.get_id(), residue))

        if not ligand_residues:
            log.warning("  ⚠  No hetero-ligand found in 6TKO.")
            return None

        chain_id, lig_res = ligand_residues[0]
        log.info(f"  Native ligand found: chain {chain_id}, residue {lig_res.get_resname()}")

        pdbio = PDBIO()

        class LigSelect(Select):
            def accept_residue(self, residue):
                return residue is lig_res

        pdbio.set_structure(struct)
        lig_pdb = output_ligand_pdbqt.replace(".pdbqt", ".pdb")
        pdbio.save(lig_pdb, LigSelect())

        mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False)
        if mol is None:
            log.warning("  ⚠  RDKit could not read ligand PDB, trying obabel…")
            smi_file = output_ligand_smi
            try:
                run_tool(["obabel", lig_pdb, "-O", smi_file], timeout=CONFIG.obabel_timeout_s)
                with open(smi_file) as f:
                    smi = f.readline().strip()
                if smi:
                    return smi
            except (RuntimeError, OSError):
                pass
            return None

        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)

        with open(output_ligand_smi, "w") as f:
            f.write(smi + "\n")
        log.info(f"  Native ligand SMILES: {smi}")

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)[0]
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setup)[0]
            with open(output_ligand_pdbqt, "w") as f:
                f.write(pdbqt_str)
            log.info(f"  Native ligand PDBQT written to {output_ligand_pdbqt}")
        except Exception as exc:
            log.warning(f"  ⚠  Meeko prep failed for native ligand: {exc}")
            shutil.copy(lig_pdb, output_ligand_pdbqt)

        return smi

    except Exception as exc:
        log.error(f"  ✗  Native ligand extraction failed: {exc}")
        return None


def _compute_rmsd_docked_vs_crystal(
    docked_pdb: str, crystal_pdb: str
) -> Optional[float]:
    """Align protein Cα backbones and compute heavy-atom RMSD of the ligand."""
    try:
        from Bio.PDB import PDBParser, Superimposer

        parser = PDBParser(QUIET=True)
        docked_struct = parser.get_structure("docked", docked_pdb)
        crystal_struct = parser.get_structure("crystal", crystal_pdb)

        def _get_ca_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == " " and "CA" in residue:
                            atoms.append(residue["CA"])
            return atoms

        docked_ca = _get_ca_atoms(docked_struct)
        crystal_ca = _get_ca_atoms(crystal_struct)

        if len(docked_ca) < 3 or len(crystal_ca) < 3:
            log.warning("  ⚠  Too few Cα atoms for backbone alignment (< 3).")
            return None

        sup = Superimposer()
        docked_coords = np.array([a.get_vector().get_array() for a in docked_ca])
        crystal_coords = np.array([a.get_vector().get_array() for a in crystal_ca])
        sup.set(crystal_coords, docked_coords)
        sup.run()
        rot, tran = sup.rotran
        log.info(f"  Backbone alignment RMSD: {sup.rmsd:.3f} Å")

        def _get_ligand_atoms(structure):
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] != " ":
                            for atom in residue:
                                if atom.element != "H":
                                    atoms.append(atom)
            return atoms

        docked_lig = _get_ligand_atoms(docked_struct)
        crystal_lig = _get_ligand_atoms(crystal_struct)

        if not docked_lig or not crystal_lig:
            log.warning("  ⚠  No ligand atoms found in one or both structures.")
            return None

        if len(docked_lig) != len(crystal_lig):
            log.warning(
                f"  ⚠  Ligand atom count mismatch: docked={len(docked_lig)}, "
                f"crystal={len(crystal_lig)}. Truncating to shorter list."
            )
            n = min(len(docked_lig), len(crystal_lig))
            docked_lig = docked_lig[:n]
            crystal_lig = crystal_lig[:n]

        docked_lig_coords = np.array([a.get_vector().get_array() for a in docked_lig])
        aligned_docked = docked_lig_coords @ rot.T + tran
        crystal_lig_coords = np.array([a.get_vector().get_array() for a in crystal_lig])
        diff = aligned_docked - crystal_lig_coords
        rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        return rmsd

    except Exception as exc:
        log.error(f"  ✗  RMSD calculation failed: {exc}")
        return None


def run_redocking_validation(
    holo_pdb_path: str,
    target_pdbqt_path: str,
    work_dir: str,
    deps: Dict[str, Any],
    center: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[float]]:
    """Phase 0 — Protocol Validation.

    Extracts the native ligand from 6TKO, docks it back into the prepared
    PBP2a receptor, and computes the RMSD to the crystal pose.

    Returns (success: bool, rmsd: float | None).
    """
    log.info("─── Phase 0: Redocking Validation ───")

    lig_smi = os.path.join(work_dir, "native_ligand.smi")
    lig_pdbqt = os.path.join(work_dir, "native_ligand.pdbqt")
    docked_pdb = os.path.join(work_dir, "native_docked.pdb")

    smi = _extract_native_ligand_from_holo(holo_pdb_path, lig_smi, lig_pdbqt)
    if smi is None:
        log.warning("  ⚠  Could not extract native ligand. Skipping redocking validation.")
        return False, None

    if not deps.get("USE_VINA", False):
        log.warning("  ⚠  Vina unavailable. Redocking validation requires Vina. Skip.")
        return False, None

    log.info("  Redocking native ligand into PBP2a…")
    docked_pdbqt = docked_pdb.replace(".pdb", ".pdbqt")
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    bx, by, bz = CONFIG.redocking_box_size
    vina_cmd = [
        "vina",
        "--receptor", target_pdbqt_path,
        "--ligand", lig_pdbqt,
        "--out", docked_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{bx:.1f}", "--size_y", f"{by:.1f}", "--size_z", f"{bz:.1f}",
        "--exhaustiveness", str(CONFIG.vina_exhaustiveness),
    ]

    try:
        run_tool(vina_cmd, timeout=CONFIG.vina_timeout_s, ignore_stderr_warnings=True)
    except RuntimeError as exc:
        log.warning(f"  ⚠  Vina redocking failed: {exc}")
        return False, None

    try:
        run_tool(
            ["obabel", docked_pdbqt, "-O", docked_pdb, "--gen3d"],
            timeout=CONFIG.obabel_timeout_s,
        )
    except RuntimeError:
        log.warning("  Could not convert docked PDBQT to PDB. Trying RDKit PDBQT reader.")
        mol = Chem.MolFromPDBQT(docked_pdbqt) if hasattr(Chem, "MolFromPDBQT") else None
        if mol is None:
            log.warning("  ⚠  Cannot parse docked PDBQT. RMSD not computed.")
            return False, None
        Chem.MolToPDBFile(mol, docked_pdb)

    crystal_pdb = lig_pdbqt.replace(".pdbqt", ".pdb")
    rmsd = _compute_rmsd_docked_vs_crystal(docked_pdb, crystal_pdb)

    if rmsd is None:
        log.warning("  ⚠  RMSD could not be computed.")
        return False, None

    log.info(f"  Redocking RMSD = {rmsd:.3f} Å")
    cutoff = CONFIG.redocking_rmsd_cutoff
    if rmsd > cutoff:
        log.warning(
            f"  ⚠  Redocking RMSD ({rmsd:.3f} Å) exceeds {cutoff} Å threshold. "
            "The docking protocol may not accurately reproduce known binding modes. "
            "Proceeding with pipeline — interpret results with caution."
        )
    else:
        log.info(f"  ✓  Redocking validated (RMSD = {rmsd:.3f} Å ≤ {cutoff} Å).")

    return (rmsd <= cutoff if rmsd is not None else False), rmsd


def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
) -> bool:
    """Convert an RDKit Mol to PDBQT via Meeko.

    Attempts conversion using Meeko's MoleculePreparation and
    PDBQTWriterLegacy.  If Meeko fails, falls back to a minimal PDBQT
    writer that assigns Gasteiger charges and writes a rigid (TORSDOF 0)
    PDBQT entry.

    Args:
        mol: RDKit molecule with at least one conformer.
        output_path: Path for the output PDBQT file.

    Returns:
        True on success, False if all conversion methods failed.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            return False
        pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
        with open(output_path, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as exc:
        log.warning(f"  Meeko prep failed ({exc}), trying RDKit fallback…")
        try:
            mol_tmp = Chem.RWMol(mol)
            mol_tmp = Chem.AddHs(mol_tmp, addCoords=True)
            AllChem.ComputeGasteigerCharges(mol_tmp)

            conf = mol_tmp.GetConformer()
            lines = ["ROOT\n"]
            for i, atom in enumerate(mol_tmp.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                charge = atom.GetDoubleProp("_GasteigerCharge")
                elem = atom.GetSymbol()
                lines.append(
                    f"ATOM     {i+1:>3}  {elem:<3} LIG X   1    "
                    f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}  "
                    f"{charge:>8.3f}     {elem:<2s}\n"
                )
            lines.append("ENDROOT\n")
            lines.append("TORSDOF 0\n")
            with open(output_path, "w") as f:
                f.writelines(lines)
            return True
        except Exception as exc2:
            log.warning(f"  RDKit PDBQT fallback also failed: {exc2}")
            return False


def _run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    output_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    timeout: int = CONFIG.vina_timeout_s,
) -> Optional[float]:
    """Run a single Vina docking job via the external tool wrapper.

    Builds the ``vina`` command-line invocation with the given receptor,
    ligand, search-box centre and dimensions, then parses the best
    (lowest) binding energy from Vina's output.

    Args:
        receptor_pdbqt: Path to the receptor PDBQT file.
        ligand_pdbqt: Path to the ligand PDBQT file.
        output_pdbqt: Path to write the docked-pose PDBQT file.
        center: 3-element array of (x, y, z) box centre coordinates.
        box_size: Tuple of (x, y, z) box dimensions in Ångström.
        timeout: Maximum wall-clock seconds for the Vina subprocess.

    Returns:
        Best binding energy in kcal/mol, or None if docking failed or
        timed out.
    """
    if CONFIG.dry_run:
        return float(np.random.uniform(-10.0, -5.0))

    cmd = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", str(CONFIG.vina_exhaustiveness),
        "--num_modes", str(CONFIG.vina_num_modes),
    ]

    try:
        result = run_tool(cmd, timeout=timeout, check=False, ignore_stderr_warnings=True)
        if result.returncode != 0:
            log.warning(f"  Vina error: {result.stderr.strip()}")
            return None
        energy = parse_vina_energy(result.stdout)
        if energy is not None:
            return energy
        energy = parse_vina_energy(result.stderr)
        return energy
    except RuntimeError as exc:
        log.warning(f"  Vina execution failed: {exc}")
        return None


def dock_compound(
    record: CompoundRecord,
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str = "",
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
) -> Optional[float]:
    """Full docking pipeline for a single compound: PDBQT prep → Vina → parse."""
    if CONFIG.dry_run:
        return float(np.random.uniform(-10.0, -5.0))

    smiles_md5 = hashlib.md5(record.smiles.encode("utf-8")).hexdigest()
    cache_key = f"{smiles_md5}_{tag}"
    if use_cache and cache is not None and cache_key in cache:
        return cache[cache_key]

    if record.mol is None:
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return None
        record.mol = mol

    safe_id = record.compound_id.replace("/", "_").replace(" ", "_")
    lig_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_lig.pdbqt")
    out_pdbqt = os.path.join(work_dir, f"{safe_id}_{tag}_out.pdbqt")

    if not prepare_ligand_pdbqt(record.mol, lig_pdbqt):
        return None

    energy = _run_vina_docking(
        receptor_pdbqt, lig_pdbqt, out_pdbqt,
        center, box_size,
    )

    for f in (lig_pdbqt, out_pdbqt):
        try:
            os.remove(f)
        except OSError:
            pass

    if use_cache and cache is not None:
        cache[cache_key] = energy

    return energy


def _worker_dock_wrapper(
    args: Tuple[str, str, str, np.ndarray, Tuple[float, float, float], str, str, bool],
) -> Tuple[str, Optional[float]]:
    """Module-level worker for :func:`_parallel_dock` (pool.map compatible).

    The per-job wall-clock timeout is enforced by :func:`run_tool` via
    ``subprocess.run(timeout=...)``, so no additional alarm mechanism is
    needed here.
    """
    cid, smiles, receptor_pdbqt, center, box_size, work_dir, tag, dry_run = args

    if dry_run:
        return cid, float(np.random.uniform(-10.0, -5.0))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cid, None
    rec = CompoundRecord(compound_id=cid, smiles=smiles, mol=mol)
    energy = dock_compound(
        rec, receptor_pdbqt, center, box_size,
        work_dir, tag, cache=None, use_cache=False,
    )
    return cid, energy


def _parallel_dock(
    items: List[Tuple[str, str]],
    receptor_pdbqt: str,
    center: np.ndarray,
    box_size: Tuple[float, float, float],
    work_dir: str,
    tag: str,
    n_jobs: int = CONFIG.n_jobs,
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
) -> List[Tuple[str, Optional[float]]]:
    """Dock a list of compounds in parallel, returning (compound_id, energy)."""
    results: List[Tuple[str, Optional[float]]] = []
    to_dock: List[Tuple[str, str, str]] = []

    for cid, smiles in items:
        smiles_md5 = hashlib.md5(smiles.encode("utf-8")).hexdigest()
        cache_key = f"{smiles_md5}_{tag}"
        if use_cache and cache is not None and cache_key in cache:
            results.append((cid, cache[cache_key]))
            log.debug(f"    Cache hit: {cid} ({tag})")
        else:
            to_dock.append((cid, smiles, cache_key))

    if not to_dock:
        return results

    n_jobs_eff = min(n_jobs, len(to_dock))
    chunksize_val = max(1, len(to_dock) // (n_jobs_eff * 4))

    work_items: List[Tuple[str, str, str, np.ndarray, Tuple[float, float, float], str, str, bool]] = [
        (cid, smiles, receptor_pdbqt, center, box_size, work_dir, tag, CONFIG.dry_run)
        for cid, smiles, _ in to_dock
    ]

    with ProcessPoolExecutor(max_workers=n_jobs_eff) as pool:
        mapped = list(
            _tqdm(
                pool.map(_worker_dock_wrapper, work_items, chunksize=chunksize_val),
                total=len(work_items),
                desc=f"  Docking {tag}",
                disable=not _HAVE_TQDM,
            )
        )

    for (cid, _, cache_key), (_, energy) in zip(to_dock, mapped):
        results.append((cid, energy))
        if use_cache and cache is not None:
            cache[cache_key] = energy

    return results


def _compute_shape_fallback_score(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    seed: int = CONFIG.random_seed,
) -> Optional[float]:
    """Fallback scoring via RDKit Shape Protrude Distance."""
    try:
        mol_3d = Chem.RWMol(mol)
        mol_3d = Chem.AddHs(mol_3d)
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        status = rdDistGeom.EmbedMolecule(mol_3d, params)
        if status < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)

        ref_3d = Chem.RWMol(ref_mol)
        ref_3d = Chem.AddHs(ref_3d)
        params_ref = rdDistGeom.ETKDGv3()
        params_ref.randomSeed = seed
        status_ref = rdDistGeom.EmbedMolecule(ref_3d, params_ref)
        if status_ref < 0:
            return None
        AllChem.MMFFOptimizeMolecule(ref_3d)

        try:
            protrude = AllChem.GetShapeProtrudeDist(mol_3d, ref_3d)
        except Exception:
            try:
                protrude = AllChem.GetShapeProtrudeDist(ref_3d, mol_3d)
            except Exception:
                return None

        normalised = min(protrude / CONFIG.shape_score_norm_factor, 10.0) if protrude > 0 else 0.0
        return normalised

    except Exception:
        return None


def screen_library(
    records: List[CompoundRecord],
    targets: Dict[str, Any],
    work_dir: str,
    deps: Dict[str, Any],
    cache: Optional[Dict[str, float]] = None,
    use_cache: bool = False,
) -> List[CompoundRecord]:
    """Phase 3 — Virtual screening.

    Primary (Vina): dock against allosteric site, select top 50 for active site.
    Fallback (RDKit Shape): shape scoring vs native ligand.

    Returns top 10 candidates.
    """
    log.info("─── Phase 3: Virtual Screening ───")

    pb2pa = targets["PBP2a"]
    allosteric_center = pb2pa["allosteric_center"]
    active_center = pb2pa["active_center"]

    use_vina = deps.get("USE_VINA", False)
    if use_vina:
        log.info("  Docking all compounds against allosteric site…")
        items = [(r.compound_id, r.smiles) for r in records]
        allosteric_results = _parallel_dock(
            items, pb2pa["pdbqt"],
            allosteric_center, CONFIG.allosteric_box_size,
            work_dir, "allosteric",
            cache=cache, use_cache=use_cache,
        )

        cid_to_record = {r.compound_id: r for r in records}
        for cid, energy in allosteric_results:
            if cid in cid_to_record:
                cid_to_record[cid].pb2pa_allosteric_energy = energy

        n_scored = sum(1 for r in records if r.pb2pa_allosteric_energy is not None)
        log.info(f"  Allosteric docking complete: {n_scored}/{len(records)} scored.")

        scored = [r for r in records if r.pb2pa_allosteric_energy is not None]
        scored.sort(key=lambda r: r.pb2pa_allosteric_energy)

        top50 = scored[:50]
        log.info(f"  Docking top {len(top50)} compounds against active site…")

        active_items = [(r.compound_id, r.smiles) for r in top50]
        active_results = _parallel_dock(
            active_items, pb2pa["pdbqt"],
            active_center, CONFIG.active_box_size,
            work_dir, "active",
            cache=cache, use_cache=use_cache,
        )

        for cid, energy in active_results:
            if cid in cid_to_record:
                cid_to_record[cid].pb2pa_active_energy = energy

    else:
        log.info("  Vina unavailable. Using RDKit Shape Fallback.")

        ref_mol = None
        holo_pdb = targets.get("holo_pdb")
        if holo_pdb and os.path.exists(holo_pdb):
            lig_pdb = os.path.join(work_dir, "native_ref.pdb")
            try:
                from Bio.PDB import PDBIO, PDBParser, Select

                parser = PDBParser(QUIET=True)
                struct = parser.get_structure("ref", holo_pdb)
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0] != " " and residue.get_resname().strip() not in ("HOH", "WAT"):
                                pdbio = PDBIO()

                                class _Sel(Select):
                                    def accept_residue(self, r):
                                        return r is residue

                                pdbio.set_structure(struct)
                                pdbio.save(lig_pdb, _Sel())
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                ref_mol = Chem.MolFromPDBFile(lig_pdb)
            except Exception:
                pass

        if ref_mol is None:
            ref_smi = list(CONFIG.control_smiles.values())[0]
            ref_mol = Chem.MolFromSmiles(ref_smi)

        if ref_mol is None:
            log.error("  Cannot obtain reference molecule for shape scoring.")
            return records[:CONFIG.top_n]

        total = len(records)
        shape_iter = _tqdm(
            enumerate(records), total=total,
            desc="  Shape scoring", disable=not _HAVE_TQDM,
        )
        for i, rec in shape_iter:
            if rec.mol is None:
                mol = Chem.MolFromSmiles(rec.smiles)
                if mol is None:
                    continue
                rec.mol = mol
            score = _compute_shape_fallback_score(rec.mol, ref_mol)
            rec.shape_score = score
            if (i + 1) % 100 == 0 and not _HAVE_TQDM:
                log.info(f"  Shape scored {i + 1} / {total}")

        scored_shape = [r for r in records if r.shape_score is not None]
        scored_shape.sort(key=lambda r: r.shape_score)
        if scored_shape:
            log.info(f"  Shape scoring complete. Best score: {scored_shape[0].shape_score:.3f}")
        else:
            log.warning("  No shape scores computed.")

    use_vina = deps.get("USE_VINA", False)
    if use_vina:
        ranked = [r for r in records if r.pb2pa_allosteric_energy is not None]
        ranked.sort(key=lambda r: r.pb2pa_allosteric_energy)
    else:
        ranked = [r for r in records if r.shape_score is not None]
        ranked.sort(key=lambda r: r.shape_score)

    top10 = ranked[:CONFIG.top_n]
    log.info(f"  Top {len(top10)} candidates selected.")
    for i, r in enumerate(top10):
        energy_str = (
            f"{r.pb2pa_allosteric_energy:.2f}" if r.pb2pa_allosteric_energy is not None
            else f"{r.shape_score:.2f} (shape)"
        )
        log.info(f"    {i + 1}. {r.compound_id}: {energy_str} kcal/mol")

    log.info("─── Phase 3 complete ───")
    return top10
