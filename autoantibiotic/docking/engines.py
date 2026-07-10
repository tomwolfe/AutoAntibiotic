from __future__ import annotations

import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from ..config import CONFIG, PipelineConfig
from ..io_utils import (
    DockingParseError,
    DockingResultValidator,
    GninaError,
    ToolExecutor,
    VinaError,
    log,
    safe_run_tool,
)
from .base import DockingEngine


# ── Ligand Preparation (shared by Vina and GNINA engines) ──────────────

_AD_TYPE_MAP: Dict[str, str] = {
    "C": "C", "c": "C",
    "N": "N", "n": "N",
    "O": "O", "o": "O",
    "S": "S", "s": "S",
    "P": "P", "p": "P",
    "F": "F", "f": "F",
    "Cl": "Cl", "Br": "Br",
    "I": "I",
    "H": "H",
}


def _prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
    config: PipelineConfig = CONFIG,
) -> bool:
    """Convert an RDKit Mol to PDBQT via Meeko.

    Attempts conversion using Meeko's MoleculePreparation and
    PDBQTWriterLegacy.  If Meeko fails, falls back to a minimal PDBQT
    writer that assigns Gasteiger charges and writes a rigid (TORSDOF 0)
    PDBQT entry.
    """
    try:
        if mol.GetNumAtoms() > 150 or mol.GetNumHeavyAtoms() > 100:
            log.debug("Molecule too large for docking")
            return False

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            mol_3d = mol
            if mol_3d.GetNumConformers() == 0:
                mol_3d = Chem.RWMol(mol)
                mol_3d = Chem.AddHs(mol_3d)
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol_3d)
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
                if mol_tmp.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol_tmp, randomSeed=42)
                AllChem.ComputeGasteigerCharges(mol_tmp)

                conf = mol_tmp.GetConformer()
                lines = ["ROOT\n"]
                for i, atom in enumerate(mol_tmp.GetAtoms()):
                    pos = conf.GetAtomPosition(i)
                    charge = atom.GetDoubleProp("_GasteigerCharge")
                    elem = atom.GetSymbol()
                    ad_type = _AD_TYPE_MAP.get(elem, "C")
                    atom_name = f" {elem:<3s}"[:4]
                    lines.append(
                        f"ATOM  {i+1:>5d} {atom_name} LIG X   1    "
                        f"{pos.x:>8.3f}{pos.y:>8.3f}{pos.z:>8.3f}  0.00  0.00"
                        f"{charge:>10.4f} {ad_type:<2s}\n"
                    )
                lines.append("ENDROOT\n")
                lines.append("TORSDOF 0\n")
                with open(output_path, "w") as f:
                    f.writelines(lines)
                return True
            except Exception as exc2:
                log.warning(f"  RDKit PDBQT fallback also failed: {exc2}")
                return False
    except Exception as exc3:
        log.warning(f"  Ligand preparation failed unexpectedly: {exc3}")
        return False


# ── Base class for external docking tools ─────────────────────────────

_DOCKING_BINARY_VALIDATED: Dict[str, bool] = {}


class _ExternalToolEngine(DockingEngine):
    """Base for docking engines that call an external binary (Vina/GNINA)."""

    def __init__(self, config: PipelineConfig = CONFIG) -> None:
        self.config = config
        self._validator = DockingResultValidator()

    @property
    def tool_name(self) -> str:
        raise NotImplementedError

    @property
    def binary_path(self) -> str:
        raise NotImplementedError

    def prepare_receptor(self, receptor_path: str) -> str:
        return receptor_path

    def prepare_ligand(self, mol: Chem.Mol, path: str) -> bool:
        return _prepare_ligand_pdbqt(mol, path, config=self.config)

    def _validate_binary(self) -> None:
        name = self.tool_name
        binary = self.binary_path
        if (
            self.config.validate_docking_binaries_on_startup
            and not _DOCKING_BINARY_VALIDATED.get(name, False)
        ):
            health_executor = ToolExecutor(retry=False)
            try:
                version_result = health_executor.run(binary, ["--version"], timeout=10)
                version_out = version_result.stdout or version_result.stderr
                if not self._validator.validate_binary_health(name, version_out):
                    from ..config import ConfigurationError
                    raise ConfigurationError(
                        f"{binary} version check failed. "
                        f"Expected Vina 1.2.x or GNINA 1.x, got: "
                        f"{version_out.strip()!r}"
                    )
                log.info(f"  ✓  {binary} binary health validated.")
            except Exception as exc:
                from ..config import ConfigurationError
                raise ConfigurationError(
                    f"Cannot run {binary} for version check: {exc}"
                )
            _DOCKING_BINARY_VALIDATED[name] = True

    def _build_args(
        self,
        ligand_path: str,
        receptor_path: str,
        output_path: str,
        center: np.ndarray,
        box_size: Tuple[float, float, float],
    ) -> List[str]:
        return [
            "--receptor", receptor_path,
            "--ligand", ligand_path,
            "--out", output_path,
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{box_size[0]:.1f}",
            "--size_y", f"{box_size[1]:.1f}",
            "--size_z", f"{box_size[2]:.1f}",
            "--exhaustiveness", str(self.config.vina_exhaustiveness),
            "--num_modes", str(self.config.vina_num_modes),
        ]

    def _parse_output(self, stdout: str, stderr: str) -> Optional[float]:
        raise NotImplementedError

    def dock(
        self,
        ligand_path: str,
        receptor_path: str,
        center: np.ndarray,
        box_size: Tuple[float, float, float],
    ) -> Optional[float]:
        if self.config.dry_run:
            return self._dry_run_score()

        self._validate_binary()

        output_pdbqt = ligand_path.replace("_lig.pdbqt", "_out.pdbqt")
        if output_pdbqt == ligand_path:
            output_pdbqt = ligand_path + ".out"

        args = self._build_args(ligand_path, receptor_path, output_pdbqt, center, box_size)
        binary = self.binary_path
        timeout = self.config.vina_timeout_s

        executor = ToolExecutor(retry=True)
        try:
            result = executor.run(binary, args, timeout=timeout)
            if result.returncode != 0 or result.timed_out:
                log.warning(f"  {binary} error: {result.stderr.strip() or 'timed out'}")
                return None
            score = self._parse_output(result.stdout, result.stderr)
            return score
        except (RuntimeError, VinaError, GninaError) as exc:
            log.warning(f"  {binary} execution failed: {exc}")
            return None

    def _dry_run_score(self) -> Optional[float]:
        raise NotImplementedError


# ── Vina Engine ───────────────────────────────────────────────────────

class VinaEngine(_ExternalToolEngine):
    """Docking engine wrapping AutoDock Vina."""

    @property
    def tool_name(self) -> str:
        return "vina"

    @property
    def binary_path(self) -> str:
        return "vina"

    def _dry_run_score(self) -> Optional[float]:
        return float(np.random.uniform(-10.0, -5.0))

    def _parse_output(self, stdout: str, stderr: str) -> Optional[float]:
        energy = self._validator.parse_vina(stdout)
        if energy is not None:
            return energy
        energy = self._validator.parse_vina(stderr)
        if energy is not None:
            return energy
        raise DockingParseError(
            f"{self.binary_path} output did not contain a valid binding energy."
        )


# ── GNINA Engine ──────────────────────────────────────────────────────

class GninaEngine(_ExternalToolEngine):
    """Docking engine wrapping GNINA."""

    def __init__(self, config: PipelineConfig = CONFIG) -> None:
        super().__init__(config)
        self._binary_path: str = config.gnina_binary_path

    @property
    def tool_name(self) -> str:
        return "gnina"

    @property
    def binary_path(self) -> str:
        return self._binary_path

    def _dry_run_score(self) -> Optional[float]:
        return float(np.random.uniform(0.5, 0.95))

    def _parse_output(self, stdout: str, stderr: str) -> Optional[float]:
        score = self._validator.parse_gnina(stdout)
        if score is not None:
            return score
        score = self._validator.parse_gnina(stderr)
        if score is not None:
            return score
        raise DockingParseError(
            f"{self.binary_path} output did not contain a valid CNNscore/CNNaffinity."
        )


# ── RDKit Shape Engine ────────────────────────────────────────────────

class RdkitShapeEngine(DockingEngine):
    """Fallback scoring engine using RDKit Shape Protrude Distance.

    Uses :func:`rdkit.Chem.AllChem.GetShapeProtrudeDist` to compute a
    shape complementarity score between the ligand and a reference
    molecule (typically the co-crystallised native ligand).
    """

    def __init__(self, config: PipelineConfig = CONFIG) -> None:
        self.config = config
        self._ref_mol: Optional[Chem.Mol] = None

    def prepare_receptor(self, receptor_path: str) -> str:
        return receptor_path

    def prepare_ligand(self, mol: Chem.Mol, path: str) -> bool:
        return True

    def _ensure_ref_mol(self, reference_smiles: Optional[str] = None) -> Optional[Chem.Mol]:
        if self._ref_mol is not None:
            return self._ref_mol
        if reference_smiles:
            mol = Chem.MolFromSmiles(reference_smiles)
            if mol is not None:
                self._ref_mol = mol
                return self._ref_mol
        smi_list = list(self.config.control_smiles.values())
        if smi_list:
            mol = Chem.MolFromSmiles(smi_list[0])
            self._ref_mol = mol
        return self._ref_mol

    def set_reference_mol(self, mol: Chem.Mol) -> None:
        """Set the reference molecule for shape comparison."""
        self._ref_mol = mol

    def dock(
        self,
        ligand_path: str,
        receptor_path: str,
        center: np.ndarray,
        box_size: Tuple[float, float, float],
    ) -> Optional[float]:
        if self.config.dry_run:
            return float(np.random.uniform(0.0, 5.0))

        ligand_mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
        if ligand_mol is None:
            smi = ligand_path.replace("_lig.pdbqt", "").split("/")[-1]
            ligand_mol = Chem.MolFromSmiles(smi)
        if ligand_mol is None:
            return None

        ref_mol = self._ensure_ref_mol()
        if ref_mol is None:
            return None

        return self._compute_shape_score(ligand_mol, ref_mol)

    def _compute_shape_score(self, mol: Chem.Mol, ref_mol: Chem.Mol) -> Optional[float]:
        try:
            mol_3d = Chem.RWMol(mol)
            mol_3d = Chem.AddHs(mol_3d)
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = self.config.random_seed
            status = rdDistGeom.EmbedMolecule(mol_3d, params)
            if status < 0:
                return None
            AllChem.MMFFOptimizeMolecule(mol_3d)

            ref_3d = Chem.RWMol(ref_mol)
            ref_3d = Chem.AddHs(ref_3d)
            params_ref = rdDistGeom.ETKDGv3()
            params_ref.randomSeed = self.config.random_seed
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

            shape_score = (
                min(protrude / self.config.shape_score_norm_factor, 10.0)
                if protrude > 0
                else 0.0
            )

            from ..scoring_metrics import compute_pharmacophore_score

            pharm_score = compute_pharmacophore_score(mol, ref_mol)
            if pharm_score is not None:
                return 0.5 * shape_score + 0.5 * (1.0 - pharm_score)

            return shape_score

        except Exception:
            return None
