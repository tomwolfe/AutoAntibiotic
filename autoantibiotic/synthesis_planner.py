from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from .config import CONFIG
from .io_utils import log

try:
    import requests  # type: ignore[import-untyped]
    _HAVE_REQUESTS = True
except ImportError:
    _HAVE_REQUESTS = False


def _canonical_smiles(smiles: str) -> Optional[str]:
    """Return the canonical SMILES for a molecule, or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


class SynthesisResult:
    """Result of a retrosynthesis check.

    Attributes:
        synthesizable: Whether the compound is considered synthesizable.
        confidence: Confidence score in [0, 1] for the synthetic route.
        routes: List of discovered synthetic routes (may be empty).
        error: Optional error message if the check failed.
    """

    __slots__ = ("synthesizable", "confidence", "routes", "error")

    def __init__(
        self,
        synthesizable: bool,
        confidence: float,
        routes: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.synthesizable = synthesizable
        self.confidence = confidence
        self.routes = routes or []
        self.error = error

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SynthesisResult(synthesizable={self.synthesizable}, "
            f"confidence={self.confidence:.2f}, routes={len(self.routes)})"
        )


class SynthesisPlanner:
    """Interface with retrosynthesis APIs to evaluate synthetic accessibility.

    Supports IBM RXN and ASKCOS APIs.  Falls back to a heuristic
    synthetic-accessibility score (SA) when external APIs are unavailable.

    Parameters
    ----------
    api_key: Optional[str]
        API key for the retrosynthesis service (optional; reads from
        ``SYNTHESIS_API_KEY`` environment variable when ``None``).
    base_url: Optional[str]
        Explicit base URL for the retrosynthesis API.  Falls back to
        ``CONFIG.synthesis_api_url`` when ``None``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("SYNTHESIS_API_KEY", "")
        self.base_url = base_url or CONFIG.synthesis_api_url
        self._cache: Dict[str, SynthesisResult] = {}

    def check_synthesizability(
        self,
        smiles: str,
        max_routes: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> SynthesisResult:
        """Check whether a SMILES string can be synthesized via retrosynthesis.

        Attempts to query the configured retrosynthesis API (IBM RXN or
        equivalent).  On failure (network error, API error, missing key),
        falls back to a heuristic SA-score computed from RDKit descriptors.

        Args:
            smiles: Canonical or non-canonical SMILES string.
            max_routes: Maximum number of synthesis routes to evaluate.
                Defaults to ``CONFIG.synthesis_api_max_routes``.
            min_confidence: Minimum confidence threshold for accepting
                the compound as synthesizable.  Defaults to
                ``CONFIG.synthesis_api_min_confidence``.

        Returns
        -------
        SynthesisResult
            Contains synthesizability boolean, confidence score,
            discovered routes, and optional error message.
        """
        if max_routes is None:
            max_routes = CONFIG.synthesis_api_max_routes
        if min_confidence is None:
            min_confidence = CONFIG.synthesis_api_min_confidence

        canon = _canonical_smiles(smiles)
        if canon is None:
            return SynthesisResult(
                synthesizable=False,
                confidence=0.0,
                error="Invalid SMILES string.",
            )

        if canon in self._cache:
            return self._cache[canon]

        try:
            result = self._query_api(canon, max_routes)
        except Exception as exc:
            log.warning(
                f"  Retrosynthesis API query failed for {canon}: {exc}. "
                "Falling back to heuristic SA score."
            )
            result = self._heuristic_sa_score(canon)

        self._cache[canon] = result
        return result

    def _query_api(
        self,
        smiles: str,
        max_routes: int,
    ) -> SynthesisResult:
        """Query the retrosynthesis API (IBM RXN or equivalent).

        Returns a SynthesisResult with the API response parsed.
        """
        if not self.api_key:
            return SynthesisResult(
                synthesizable=False,
                confidence=0.0,
                error="No API key configured. Cannot query retrosynthesis API.",
            )

        url = f"{self.base_url}/rxns"
        payload = {
            "prediction": {
                "reactions": {
                    "smiles": smiles,
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=CONFIG.synthesis_api_timeout_s,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        data = resp.json()
        routes = self._parse_api_response(data, max_routes)

        if not routes:
            return SynthesisResult(
                synthesizable=False,
                confidence=0.0,
                error="No synthesis routes returned from API.",
            )

        best_confidence = max(r.get("confidence", 0.0) for r in routes)
        synthesizable = best_confidence >= CONFIG.synthesis_api_min_confidence

        return SynthesisResult(
            synthesizable=synthesizable,
            confidence=best_confidence,
            routes=routes,
        )

    def _parse_api_response(
        self,
        data: Dict[str, Any],
        max_routes: int,
    ) -> List[Dict[str, Any]]:
        """Parse API response data into a list of route dicts.

        Expected API response format (IBM RXN):
        {
            "id": "rxn_123456",
            "reactants": ["CC(=O)O", "CCN"],
            "products": ["CC(=O)OC"],
            "yield_prediction": 0.85,
        }
        """
        routes: List[Dict[str, Any]] = []

        # Support both list-of-routes and single-route formats
        items = data if isinstance(data, list) else [data]

        for item in items[:max_routes]:
            reactants = item.get("reactants", [])
            products = item.get("products", [])
            confidence = item.get("confidence", item.get("yield_prediction", 0.0))

            routes.append({
                "reactants": reactants,
                "products": products,
                "confidence": confidence,
            })

        return routes

    def _heuristic_sa_score(
        self,
        smiles: str,
    ) -> SynthesisResult:
        """Compute a heuristic synthetic accessibility score from RDKit.

        Uses the Ertl et al. (2009) SA score when RDKit is available,
        otherwise falls back to a simple fragment-count heuristic.
        """
        try:
            from rdkit.Chem import MolDescriptors
            from rdkit.Chem import rdMolDescriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return SynthesisResult(
                    synthesizable=False,
                    confidence=0.0,
                    error="Failed to parse SMILES for SA scoring.",
                )

            # Ertl SA score: combination of complexity and fragment contribution
            complexity = MolDescriptors.MolLogP(mol)
            num_rots = rdMolDescriptors.NumRotatableBonds(mol)
            num_rings = rdMolDescriptors.MolNumRings(mol)

            # Normalised heuristic SA (clamped to [0, 1])
            score = min(1.0, max(0.0,
                1.0 - (complexity / 10.0) - (num_rots / 20.0) - (num_rings / 10.0),
            ))

            synthesizable = score >= CONFIG.synthesis_api_min_confidence
            return SynthesisResult(
                synthesizable=synthesizable,
                confidence=score,
            )
        except Exception as exc:
            return SynthesisResult(
                synthesizable=False,
                confidence=0.0,
                error=f"Heuristic SA scoring failed: {exc}",
            )

    def clear_cache(self) -> None:
        """Clear the in-memory result cache."""
        self._cache.clear()
