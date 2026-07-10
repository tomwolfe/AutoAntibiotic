from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem


class DockingEngine(ABC):
    """Abstract base class for molecular docking engines."""

    @abstractmethod
    def prepare_receptor(self, receptor_path: str) -> str:
        """Prepare a receptor structure for docking.

        Args:
            receptor_path: Path to the raw receptor file (e.g. PDB).

        Returns:
            Path to the prepared receptor file (e.g. PDBQT).
        """
        ...

    @abstractmethod
    def prepare_ligand(self, mol: Chem.Mol, path: str) -> bool:
        """Prepare a ligand molecule for docking.

        Args:
            mol: RDKit molecule with at least one conformer.
            path: Output path for the prepared ligand file.

        Returns:
            True on success, False if preparation failed.
        """
        ...

    @abstractmethod
    def dock(
        self,
        ligand_path: str,
        receptor_path: str,
        center: np.ndarray,
        box_size: Tuple[float, float, float],
    ) -> Optional[float]:
        """Run docking and return the best score.

        Args:
            ligand_path: Path to the prepared ligand file.
            receptor_path: Path to the prepared receptor file.
            center: 3-element array of (x, y, z) box centre coordinates.
            box_size: Tuple of (x, y, z) box dimensions in Ångström.

        Returns:
            Best score (lower is better) or None if docking failed.
        """
        ...
