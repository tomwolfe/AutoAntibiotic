"""
Ligand preparation utilities
=============================

Convert RDKit molecules to AutoDock Vina's PDBQT format. Extraction of this
logic from ``discovery_pipeline.py`` keeps the main orchestration file focused
on the high-level phases while the implementation details live here.

Strategy used by :class:`LigandPreparator`:
    1. Try ``meeko`` (preferred — handles partial charges, rotatable bonds).
    2. If meeko is unavailable or fails, fall back to ``obabel`` via subprocess.
    3. If both fail, raise a clear ``RuntimeError`` with install instructions.
"""

import logging
import subprocess

from rdkit import Chem

# Shared logger: same name as the one configured in discovery_pipeline, so all
# log records route through the same handlers (stream + pipeline.log).
log = logging.getLogger("AutoAntibiotic")


class LigandPreparator:
    """
    Encapsulates the logic for converting an RDKit Mol to PDBQT format.

    Strategy:
        1. Try meeko (preferred — handles partial charges, rotatable bonds).
        2. If meeko is unavailable or fails, fall back to obabel via subprocess.
        3. If both fail, raise a clear RuntimeError with installation instructions.
    """

    def prepare(self, mol: Chem.Mol) -> str:
        """
        Convert an RDKit Mol to a PDBQT string.

        Args:
            mol: Input molecule (should have 3D coordinates).

        Returns:
            PDBQT-formatted string.

        Raises:
            RuntimeError: If neither meeko nor obabel can produce PDBQT.
        """
        meeko_error = None
        try:
            # Newer meeko releases call ``Mol.HasQuery()``, which is absent in
            # older RDKit builds (e.g. 2022.09). Patch it in if missing so the
            # meeko preparation path works across RDKit versions.
            if not hasattr(Chem.Mol, "HasQuery"):
                Chem.Mol.HasQuery = lambda self: False

            # Newer meeko (>=0.4.0) depends on gemmi; suppress any
            # pre-import noise from optional dependencies.
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if not mol_setups:
                raise RuntimeError("Meeko returned an empty setup for the input molecule")
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
            if pdbqt_str:
                return pdbqt_str
            raise RuntimeError("Meeko produced an empty PDBQT string for the input molecule")
        except (ImportError, AttributeError, RuntimeError) as exc:
            meeko_error = str(exc)
            log.warning(
                f"Meeko failed: {exc}. For better accuracy, install it via "
                "'pip install meeko'."
            )

        obabel_error = None
        try:
            import tempfile
            import os
            tmp_path = os.path.join(tempfile.gettempdir(), f"lig_prep_{id(mol)}.pdbqt")
            subprocess.run(
                ["obabel", f"-:{Chem.MolToSmiles(mol)}", "--gen3d", "-O", tmp_path],
                capture_output=True,
                timeout=30,
            )
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path) as f:
                    pdbqt_str = f.read()
                os.remove(tmp_path)
                if pdbqt_str:
                    return pdbqt_str
            raise ValueError("obabel returned empty output")
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as exc:
            obabel_error = str(exc)
            log.warning(f"obabel fallback failed: {exc}")

        raise RuntimeError(
            "PDBQT preparation failed. Please ensure either 'meeko' or "
            "'openbabel' is installed and on your PATH."
            f" meeko error: {meeko_error}; obabel error: {obabel_error}"
        )


def prepare_ligand_pdbqt(
    mol: Chem.Mol,
    output_path: str,
) -> bool:
    """
    Convert an RDKit Mol to PDBQT via LigandPreparator.

    Args:
        mol: Input molecule.
        output_path: Destination .pdbqt path.

    Returns:
        True on success.
    """
    try:
        preparator = LigandPreparator()
        pdbqt_str = preparator.prepare(mol)
        with open(output_path, "w") as f:
            f.write(pdbqt_str)
        return True
    except Exception as exc:
        log.warning(f"  Ligand preparation failed: {exc}")
        return False
