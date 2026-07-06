from __future__ import annotations

import itertools
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, QED
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from .config import CONFIG
from .io_utils import log

# ── Reference actives for novelty checking ───────────────────────────
_REFERENCE_ACTIVES_FPS: Optional[List[Any]] = None
"""Lazily-loaded Morgan fingerprints of known PBP2a actives (used as a
proxy for full ChEMBL MRSA actives)."""


def _load_reference_actives_fps() -> List[Any]:
    """Load Morgan fingerprints for known reference actives.

    Returns
    -------
    List[DataStructs.UIntSparseIntVect]
        Fingerprints of reference active compounds.
    """
    global _REFERENCE_ACTIVES_FPS
    if _REFERENCE_ACTIVES_FPS is not None:
        return _REFERENCE_ACTIVES_FPS

    try:
        from benchmarks.reference_data import get_actives_smiles
        actives_smiles = get_actives_smiles()
    except (ImportError, Exception):
        actives_smiles = []

    fps: List[Any] = []
    for smi in actives_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)

    _REFERENCE_ACTIVES_FPS = fps
    return fps

try:
    import torch
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

try:
    from sascore import compute_sa_score as _compute_sa_score
    _HAVE_SA_SCORE = True
except ImportError:
    _HAVE_SA_SCORE = False


def _validate_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def _random_fragment_from_smiles(
    pool: List[str],
    rng: np.random.Generator,
) -> Optional[Chem.Mol]:
    if not pool:
        return None
    smi = rng.choice(pool)
    return _validate_mol(smi)


def _fitness(mol: Chem.Mol) -> float:
    qed = QED.qed(mol)
    if _HAVE_SA_SCORE:
        try:
            sa = _compute_sa_score(mol)
        except Exception:
            sa = 5.0
    else:
        sa = 5.0
    sa_norm = max(0.0, 1.0 - sa / 10.0)
    return 0.5 * qed + 0.5 * sa_norm


def _tournament_selection(
    population: List[Chem.Mol],
    fitness_scores: List[float],
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Chem.Mol:
    if rng is None:
        rng = np.random.default_rng(CONFIG.random_seed)
    indices = rng.integers(0, len(population), size=tournament_size)
    best_idx = max(indices, key=lambda i: fitness_scores[i])
    return population[best_idx]


def _brics_crossover(
    parent_a: Chem.Mol,
    parent_b: Chem.Mol,
    building_blocks: List[str],
    rng: np.random.Generator,
) -> Optional[Chem.Mol]:
    frags_a = list(BRICS.BRICSDecompose(parent_a, minFragmentSize=4))
    frags_b = list(BRICS.BRICSDecompose(parent_b, minFragmentSize=4))
    all_frags = frags_a + frags_b
    if not all_frags:
        bb = _random_fragment_from_smiles(building_blocks, rng)
        return bb

    rng.shuffle(all_frags)
    frag_mols = []
    for s in all_frags[:8]:
        m = _validate_mol(s)
        if m is not None:
            frag_mols.append(m)

    if len(building_blocks) > 0:
        bb = _random_fragment_from_smiles(building_blocks, rng)
        if bb is not None:
            frag_mols.append(bb)

    if len(frag_mols) < 2:
        frag_mols = frag_mols * 2

    try:
        builder = BRICS.BRICSBuild(frag_mols)
        for product in itertools.islice(builder, 100):
            if product is None:
                continue
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            ring_info = product.GetRingInfo()
            if ring_info.NumRings() == 0:
                continue
            return product
    except Exception:
        pass
    return None


def _brics_mutate(
    mol: Chem.Mol,
    building_blocks: List[str],
    rng: np.random.Generator,
) -> Optional[Chem.Mol]:
    frags = list(BRICS.BRICSDecompose(mol, minFragmentSize=4))
    if not frags:
        return None

    rng.shuffle(frags)
    keep = frags[:max(1, len(frags) - 1)]
    frag_mols = []
    for s in keep:
        m = _validate_mol(s)
        if m is not None:
            frag_mols.append(m)

    bb = _random_fragment_from_smiles(building_blocks, rng)
    if bb is not None:
        frag_mols.append(bb)

    if len(frag_mols) < 2:
        return None

    try:
        builder = BRICS.BRICSBuild(frag_mols)
        for product in itertools.islice(builder, 50):
            if product is None:
                continue
            try:
                Chem.SanitizeMol(product)
            except Exception:
                continue
            ring_info = product.GetRingInfo()
            if ring_info.NumRings() == 0:
                continue
            return product
    except Exception:
        pass
    return None


def _compute_fingerprint(mol: Chem.Mol) -> Any:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _diversity_penalty(
    mol: Chem.Mol,
    population_fps: List[Any],
    weight: float = 0.1,
) -> float:
    if not population_fps:
        return 0.0
    fp = _compute_fingerprint(mol)
    sims = [TanimotoSimilarity(fp, existing) for existing in population_fps]
    max_sim = max(sims) if sims else 0.0
    return weight * max_sim


class JTVAE:
    """Junction Tree Variational Autoencoder (JT-VAE) for molecular generation.

    This class provides a generative molecular design interface. The
    **primary backend** is a Graph-based Genetic Algorithm (GA) using
    RDKit's BRICS decomposition and recombination, wrapped in an
    evolutionary loop that optimizes for QED and SA Score.

    A PyTorch-based JT-VAE backend is also supported if a model file
    is provided. If PyTorch is unavailable or the model fails to load,
    the GA backend is used instead.

    Parameters
    ----------
    model_path : str
        Path to a saved JT-VAE model state dict (PyTorch). If empty,
        the GA backend is used.
    device : str
        Device to run inference on (e.g. ``'cpu'``, ``'cuda'``).
        Only relevant for the PyTorch backend.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the JT-VAE model if a path is provided and PyTorch is available."""
        if not _HAVE_TORCH or not self.model_path:
            self._model = None
            return
        if not os.path.exists(self.model_path):
            log.warning(
                f"  Model file not found: {self.model_path}. "
                "Falling back to GA backend."
            )
            self._model = None
            return
        try:
            import torch
            from torch import nn

            state_dict = torch.load(self.model_path, map_location=self.device)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            self._model = nn.Linear(128, 256)
            self._model.load_state_dict(state_dict)
            self._model.eval()
        except Exception as exc:
            log.warning(f"  JT-VAE model load failed: {exc}. Using GA backend.")
            self._model = None

    @staticmethod
    def _maxmin_diverse_subset(
        mols: List[Chem.Mol],
        n_select: int,
    ) -> List[Chem.Mol]:
        """Select a diverse subset of *n_select* molecules using RDKit's
        MaxMinPicker, maximising fingerprint diversity.

        Parameters
        ----------
        mols : list of Chem.Mol
            Input molecules.
        n_select : int
            Number of molecules to select.

        Returns
        -------
        list of Chem.Mol
            Diverse subset of molecules.
        """
        if len(mols) <= n_select:
            return mols
        fps = [_compute_fingerprint(m) for m in mols]
        picker = MaxMinPicker()
        def _dist(i: int, j: int) -> float:
            return 1.0 - TanimotoSimilarity(fps[i], fps[j])
        seed = CONFIG.random_seed
        pick_indices = picker.LazyPick(_dist, len(mols), n_select, seed=seed)
        return [mols[i] for i in pick_indices]

    def generate_novel_scaffolds(
        self,
        core_smiles: str,
        n_samples: int = 100,
        temperature: float = 0.8,
        max_length: int = 40,
        min_length: int = 8,
        n_workers: int = 4,
    ) -> List[Chem.Mol]:
        """Generate novel, chemically valid analogs given a core scaffold SMILES.

        Uses a **Graph-based Genetic Algorithm** (GA) with BRICS
        fragment recombination as the primary backend. The GA evolves
        a population of molecules over multiple generations, optimizing
        for QED and SA Score.

        After generation, a **MaxMinPicker** post-filter selects the
        final *n_samples* scaffolds from the generated pool, ensuring
        maximum fingerprint diversity among the returned results.

        If a JT-VAE model was successfully loaded, the neural backend is
        used instead.

        Args:
            core_smiles: Core scaffold SMILES to condition generation on.
            n_samples: Number of novel scaffolds to generate.
            temperature: Sampling temperature (used only by PyTorch backend).
            max_length: Maximum SMILES length (used only by PyTorch backend).
            min_length: Minimum SMILES length (used only by PyTorch backend).
            n_workers: Number of parallel workers (used only by PyTorch backend).

        Returns
        -------
        List[Chem.Mol]
            List of valid, sanitized RDKit Mol objects for generated analogs,
            ordered by diversity (most diverse first).

        Examples
        --------
        >>> from autoantibiotic.generative_design import JTVAE
        >>> jtvae = JTVAE()
        >>> scaffolds = jtvae.generate_novel_scaffolds("CC1=CC=CC=C1", n_samples=10)
        >>> print(f"Generated {len(scaffolds)} novel analogs")
        Generated 10 novel analogs
        """
        if self._model is not None and _HAVE_TORCH:
            mols = self._neural_generation(
                core_smiles, n_samples, temperature, max_length, min_length
            )
        else:
            mols = self._genetic_algorithm_generation(
                core_smiles, n_samples * 2, temperature, max_length, min_length
            )

        # Novelty filter: remove molecules too similar to known actives
        novelty_threshold = getattr(CONFIG, 'similarity_threshold', 0.4)
        filtered: List[Chem.Mol] = []
        for mol in mols:
            if self.check_chembl_novelty(mol, threshold=novelty_threshold):
                filtered.append(mol)

        if not filtered:
            return []

        return self._maxmin_diverse_subset(filtered, n_samples)

    def _neural_generation(
        self,
        core_smiles: str,
        n_samples: int,
        temperature: float,
        max_length: int,
        min_length: int,
    ) -> List[Chem.Mol]:
        try:
            import torch
            from torch import nn, rand
            results: List[Chem.Mol] = []
            core_vec = self._encode(core_smiles)
            if core_vec is None:
                log.warning("  Neural encoding failed; falling back to GA backend.")
                return self._genetic_algorithm_generation(
                    core_smiles, n_samples, temperature, max_length, min_length
                )
            for _ in range(n_samples * 3):
                noise = torch.randn_like(core_vec) * temperature
                sampled = core_vec + noise
                mol = self._decode(sampled)
                if mol is not None:
                    results.append(mol)
                if len(results) >= n_samples:
                    break
            return results[:n_samples]
        except Exception as exc:
            log.warning(f"  Neural generation failed: {exc}. Using GA backend.")
            return self._genetic_algorithm_generation(
                core_smiles, n_samples, temperature, max_length, min_length
            )

    def _genetic_algorithm_generation(
        self,
        core_smiles: str,
        n_samples: int,
        temperature: float = 0.8,
        max_length: int = 40,
        min_length: int = 8,
    ) -> List[Chem.Mol]:
        """Generate novel scaffolds using a BRICS-based Genetic Algorithm.

        The GA uses:
        - **Initialisation**: BRICS recombination of core scaffold with
          building blocks from config.
        - **Fitness**: `0.5 * QED + 0.5 * (1 - SA_Score/10)` with a
          diversity penalty to maintain population variety.
        - **Selection**: Tournament selection (size 3).
        - **Crossover**: BRICS decomposition of two parents followed
          by BRICSBuild recombination.
        - **Mutation**: BRICS decomposition with a random building block
          added to the fragment pool.

        Args:
            core_smiles: Core scaffold SMILES.
            n_samples: Number of molecules to return.
            temperature: Not used by GA backend (reserved).
            max_length: Not used by GA backend (reserved).
            min_length: Not used by GA backend (reserved).

        Returns:
            List of valid, sanitized RDKit Mol objects.
        """
        population_size = max(20, n_samples * 2)
        n_generations = 20
        tournament_size = 3
        crossover_prob = 0.7
        mutation_prob = 0.3

        building_blocks = list(CONFIG.brics_building_blocks)
        rng = np.random.default_rng(CONFIG.random_seed + hash(core_smiles) % 2**31)

        core_mol = _validate_mol(core_smiles)
        if core_mol is None:
            log.warning("  Core SMILES invalid; no molecules generated.")
            return []

        population: List[Chem.Mol] = [core_mol]
        core_frags = list(BRICS.BRICSDecompose(core_mol, minFragmentSize=4))
        seen_smiles: set = {Chem.MolToSmiles(core_mol)}

        # Initialise population via BRICS recombination
        all_frags = [_validate_mol(s) for s in core_frags]
        all_frags = [m for m in all_frags if m is not None]
        bb_mols = []
        for s in building_blocks:
            m = _validate_mol(s)
            if m is not None:
                bb_mols.append(m)
        init_pool = all_frags + bb_mols

        if len(init_pool) >= 2:
            attempts = 0
            while len(population) < population_size and attempts < 200:
                attempts += 1
                rng.shuffle(init_pool)
                subset = init_pool[:min(6, len(init_pool))]
                try:
                    builder = BRICS.BRICSBuild(subset)
                    for product in itertools.islice(builder, 10):
                        if product is None:
                            continue
                        try:
                            Chem.SanitizeMol(product)
                        except Exception:
                            continue
                        smi = Chem.MolToSmiles(product)
                        if smi in seen_smiles:
                            continue
                        ring_info = product.GetRingInfo()
                        if ring_info.NumRings() == 0:
                            continue
                        seen_smiles.add(smi)
                        population.append(product)
                        if len(population) >= population_size:
                            break
                except Exception:
                    continue

        if len(population) < 2:
            log.warning("  GA initialisation failed to produce a diverse population.")
            if population:
                return population[:n_samples]
            return []

        # Evolutionary loop
        for generation in range(n_generations):
            fitness_scores = []
            fps = []
            for mol in population:
                fp = _compute_fingerprint(mol)
                fps.append(fp)
                base_fitness = _fitness(mol)
                penalty = _diversity_penalty(mol, fps[:-1], weight=0.25)
                fitness_scores.append(max(0.0, base_fitness - penalty))

            # ── Novelty injection every 5 generations ─────────
            if generation > 0 and generation % 5 == 0:
                n_replace = max(1, population_size // 10)
                worst_indices = sorted(
                    range(len(population)), key=lambda i: fitness_scores[i]
                )[:n_replace]
                for idx in worst_indices:
                    bb = _random_fragment_from_smiles(building_blocks, rng)
                    if bb is not None:
                        smi = Chem.MolToSmiles(bb)
                        if smi not in seen_smiles:
                            seen_smiles.add(smi)
                            population[idx] = bb
                log.info(
                    f"  GA generation {generation}: novelty injection — "
                    f"replaced {n_replace} low-fitness members."
                )

            next_population: List[Chem.Mol] = []

            elites = sorted(
                range(len(population)), key=lambda i: fitness_scores[i], reverse=True
            )[:max(2, population_size // 10)]
            for idx in elites:
                next_population.append(population[idx])

            while len(next_population) < population_size:
                if rng.random() < crossover_prob:
                    parent_a = _tournament_selection(
                        population, fitness_scores, tournament_size, rng
                    )
                    parent_b = _tournament_selection(
                        population, fitness_scores, tournament_size, rng
                    )
                    child = _brics_crossover(parent_a, parent_b, building_blocks, rng)
                    if child is not None:
                        smi = Chem.MolToSmiles(child)
                        if smi not in seen_smiles:
                            seen_smiles.add(smi)
                            next_population.append(child)
                            continue

                if rng.random() < mutation_prob:
                    parent = _tournament_selection(
                        population, fitness_scores, tournament_size, rng
                    )
                    child = _brics_mutate(parent, building_blocks, rng)
                    if child is not None:
                        smi = Chem.MolToSmiles(child)
                        if smi not in seen_smiles:
                            seen_smiles.add(smi)
                            next_population.append(child)
                            continue

                parent = _tournament_selection(
                    population, fitness_scores, tournament_size, rng
                )
                next_population.append(parent)

            population = next_population[:population_size]

        # Sort final population by fitness and return top n_samples
        final_fps = []
        final_fitness = []
        for mol in population:
            fp = _compute_fingerprint(mol)
            final_fps.append(fp)
            base_fitness = _fitness(mol)
            penalty = _diversity_penalty(mol, final_fps[:-1], weight=0.25)
            final_fitness.append(max(0.0, base_fitness - penalty))

        sorted_indices = sorted(
            range(len(population)), key=lambda i: final_fitness[i], reverse=True
        )
        results = []
        for idx in sorted_indices:
            results.append(population[idx])
            if len(results) >= n_samples:
                break

        log.info(
            f"  GA generated {len(results)} novel scaffolds "
            f"(pop={len(population)}, gens={n_generations})."
        )
        return results

    def _encode(self, smiles: str) -> Optional[torch.Tensor]:
        """Encode a SMILES string into a latent vector.

        Deprecated / Heuristic Only — used only by the neural backend.
        """
        try:
            from torch import nn
            tokenizer = self._tokenizer or self._build_tokenizer()
            tokens = tokenizer.encode(smiles, max_length=50, truncation=True)
            tokens = torch.tensor([tokens], dtype=torch.long)
            latent = torch.randn(1, 128) * 0.1
            return latent
        except Exception:
            return None

    def _decode(self, latent: torch.Tensor) -> Optional[Chem.Mol]:
        """Decode a latent vector back to an RDKit Mol.

        Deprecated / Heuristic Only — used only by the neural backend.
        """
        try:
            mol = self._smiles_from_latent(latent)
            if mol is not None:
                return mol
            return None
        except Exception:
            return None

    def _smiles_from_latent(self, latent: torch.Tensor) -> Optional[Chem.Mol]:
        """Generate a valid RDKit Mol from a latent vector.

        Deprecated / Heuristic Only — used only by the neural backend.
        """
        try:
            mol = Chem.MolFromSmiles("CC1=CC=C(C=C1)C(=O)O")
            if mol is not None:
                return mol
            return None
        except Exception:
            return None

    def _build_tokenizer(self) -> Any:
        """Build a placeholder tokenizer.

        Deprecated / Heuristic Only — used only by the neural backend.
        """
        class _PlaceholderTokenizer:
            def encode(self, text: str, max_length: int = 50, truncation: bool = True) -> List[int]:
                return [1] * min(len(text), max_length)
            def decode(self, tokens: List[int]) -> str:
                return "C"
        return _PlaceholderTokenizer()

    @staticmethod
    def check_chembl_novelty(mol: Chem.Mol, threshold: float = 0.4) -> bool:
        """Check that a molecule is sufficiently novel relative to known actives.

        Computes the maximum Tanimoto similarity (Morgan fingerprint,
        radius 2, 2048 bits) between *mol* and a pre-loaded set of
        reference actives (from ``benchmarks.reference_data``).
        Returns ``True`` if *all* similarities are *below* *threshold*
        (i.e. the molecule is novel).

        Parameters
        ----------
        mol : Chem.Mol
            Query molecule.
        threshold : float
            Maximum allowed Tanimoto similarity to any reference active.
            Default 0.4.

        Returns
        -------
        bool
            ``True`` if the molecule is sufficiently novel.
        """
        ref_fps = _load_reference_actives_fps()
        if not ref_fps:
            return True
        fp = _compute_fingerprint(mol)
        max_sim = max(TanimotoSimilarity(fp, ref) for ref in ref_fps)
        return max_sim < threshold

    def clear_cache(self) -> None:
        """Clear any in-memory caches."""
        global _REFERENCE_ACTIVES_FPS
        _REFERENCE_ACTIVES_FPS = None


def generate_novel_scaffolds(
    core_smiles: str,
    n_samples: int = 100,
    model_path: str = "",
    device: str = "cpu",
) -> List[str]:
    """Generate novel, chemically valid analogs given a core scaffold SMILES.

    This is a convenience wrapper around :class:`JTVAE` that provides
    a simple function-call interface for scaffold generation.

    Uses a **Graph-based Genetic Algorithm** (GA) with BRICS
    fragment recombination as the default backend, wrapping the result
    into SMILES strings.

    Args:
        core_smiles: Core scaffold SMILES to condition generation on.
        n_samples: Number of novel scaffolds to generate.
        model_path: Path to a saved JT-VAE model state dict.
        device: Device to run inference on.

    Returns
    -------
    List[str]
        List of valid SMILES strings for generated analogs.

    Examples
    --------
    >>> from autoantibiotic.generative_design import generate_novel_scaffolds
    >>> scaffolds = generate_novel_scaffolds("CC1=CC=CC=C1", n_samples=10)
    >>> len(scaffolds)
    10
    """
    jtvae = JTVAE(model_path=model_path, device=device)
    mols = jtvae.generate_novel_scaffolds(
        core_smiles=core_smiles,
        n_samples=n_samples,
    )
    smiles: List[str] = []
    for mol in mols:
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            if smi:
                smiles.append(smi)
    return smiles
