#!/usr/bin/env python3
"""
Test helpers for the AutoAntibiotic Discovery Pipeline.

These utilities let unit tests build valid, minimal PDB content in memory
instead of relying on static ``.pdb`` files under ``tests/data/`` (which are
kept only for integration / smoke tests).
"""

from typing import Dict, List, Tuple


def create_minimal_pdb(
    residues_dict: Dict[Tuple[str, int, str], List[Tuple[str, float, float, float]]],
    element_fn=None,
) -> str:
    """
    Generate a minimal but valid PDB string from an in-memory residue map.

    Args:
        residues_dict: Maps ``(resname, resid, chain)`` to a list of atoms,
            where each atom is ``(atom_name, x, y, z)``.
        element_fn: Optional callable ``(atom_name) -> element_symbol``. If
            omitted, the element is inferred from the first alphabetic
            character of the atom name.

    Returns:
        The PDB content as a string (caller decides whether to write it to
        disk or parse it directly).
    """
    if element_fn is None:
        def element_fn(name: str) -> str:
            cleaned = name.strip()
            for ch in cleaned:
                if ch.isalpha():
                    return ch
            return "C"

    lines: List[str] = []
    serial = 1
    for (resname, resid, chain), atoms in residues_dict.items():
        for atom_name, x, y, z in atoms:
            element = element_fn(atom_name)
            line = (
                "ATOM  "
                + f"{serial:5d}"
                + " "
                + f"{atom_name:4s}"
                + " "
                + f"{resname:3s}"
                + " "
                + f"{chain:1s}"
                + f"{resid:4d}"
                + "    "
                + f"{x:8.3f}{y:8.3f}{z:8.3f}"
                + "  1.00  0.00           "
                + f"{element:2s}"
            )
            lines.append(line)
            serial += 1

    lines.append("END")
    return "\n".join(lines) + "\n"
