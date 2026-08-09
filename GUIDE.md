# AutoAntibiotic Guide

A practical guide to the AutoAntibiotic Discovery Pipeline v7.4.0 for three
distinct uses:

1. [Generating new binding hypotheses](#1-using-the-pipeline-for-hypothesis-generation)
2. [Reproducing the published results](#2-reproducing-published-results)
3. [Understanding the known limitations](#3-known-limitations-and-future-directions)

Before any scientific use, read `SCIENCE.md`. In particular, only `mode:
science` with **real PDB structures** and **AutoDock Vina installed**
(`bash setup.sh`, or the bundled Docker image) produces physical results;
`mode: ci` is an offline mock that must never be interpreted scientifically.

---

## 1. Using the pipeline for hypothesis generation

The pipeline's rigid-docking screen does **not** validate binding on its own.
The enrichment benchmark for this target is a *validated negative result*
(see §3): rigid-receptor Vina against the apo conformer cannot separate weak
PBP2a β-lactam actives from property-matched decoys (AUC ~ 0.13). Treat the
docked rankings as **scaffold hypotheses**, not confirmed hits.

A credible hypothesis chain requires the post-screening validation stages:

```bash
# 1. Screen the library (produces output/top_candidates.csv)
AUTOANTIBIOTIC_MODE=science python discovery_pipeline.py

# 2. Confirm docked poses are stable with extended, multi-replica,
#    unrestrained explicit-solvent MD. The paper's 1–ns single-replica runs
#    are PRELIMINARY (100 ns × 3 replica campaign in progress).
#    The driver auto-selects the best OpenMM platform (Metal → CUDA → OpenCL → CPU;
#    on Apple Silicon that is the Metal-backed OpenCL runtime, ~13.7 ns/day with HMR
#    on the M5 Pro 422k-atom system, ~3.3× the 2-fs OpenCL rate). Interrupting a run
#    is safe: pass --resume to continue unfinished replicas from their last checkpoint.
python scripts/explicit_solvent_md.py --production-ns 100 --replicas 3 --n-candidates 5 --resume

# 3. Score an ensemble of the production trajectory with MM-GBSA.
#    (The script auto-switches to trajectory-based sampling when DCDs exist.)
python scripts/mmgbsa_analysis.py

# 4. Optionally refine catalytic side-chains with induced-fit docking.
AUTOANTIBIOTIC_MODE=science python scripts/run_ifd_top20.py
```

**Minimum standard for a "validated hypothesis":** ≥ 100 ns (ideally ×3
replicas) of unrestrained explicit-solvent MD per candidate, low ligand RMSD
relative to the docked pose over the *equilibrated* window, ≥ 2 replicas with
stable/metastable classification, and a trajectory-ensemble MM-GBSA ΔG that is
not grossly unfavourable. A 1–ns run is only a smoke test; the 100 ns
multi-replica campaign is the minimum for a defensible short-list.

---

## 2. Reproducing published results

All inputs (ChEMBL actives cache, decoy pool, source PDBs) are committed, so
science-mode runs are reproducible offline. Exact commands:

```bash
# Full screening pipeline
AUTOANTIBIOTIC_MODE=science python discovery_pipeline.py

# DUD-E style enrichment benchmark (single exhaustiveness, ex=8)
AUTOANTIBIOTIC_MODE=science python scripts/dude_benchmark.py

# Enrichment saturation sweep (diagnostic, resumable; caches per-exhaustiveness)
AUTOANTIBIOTIC_MODE=science python scripts/enrichment_saturation_analysis.py

# Conserved active-site water analysis
python scripts/conserved_water_analysis.py

# Extended explicit-solvent MD + MM-GBSA
python scripts/explicit_solvent_md.py --production-ns 100 --replicas 3 --n-candidates 5
python scripts/mmgbsa_analysis.py

# Final gate — every success criterion
python verify_success.py
```

Because the docking sweep is expensive, `enrichment_saturation_analysis.py`
writes per-exhaustiveness caches to `output/enrichment_saturation_cache/` and
resumes automatically; drop `--no-cache` to force re-docking. The saturation
results feed `output/enrichment_saturation.json`, which is a required input to
`verify_success.py`.

**Documented reproducibility caveats:**
- The ChEMBL active set is cached in `data/chembl_pbp2a_actives.csv`; a live
  re-fetch may return a marginally different set (hence stable per-run
  fingerprints in the saturation caches).
- Docking/PATH and OpenMM threading depend on the host; the Docker image
  (`docker build -t autoantibiotic .`) is the canonical environment.

---

## 3. Known limitations and future directions

This subsection states limitations explicitly rather than hiding them.

### Rigid-docking enrichment is a validated negative result (this target)

The standardised DUD-E-style benchmark (76 ChEMBL actives, 704 property-decoys,
apo 1VQQ, raw Vina affinities, ex=8) reports **AUC = 0.134 (95% CI
0.099–0.174), EF₁% = EF₅% = EF₁₀% = 0, BEDROC₂₀ = 0 — FAIL**. No active ranked
in the top 10% of 780 compounds. Because actives and decoys were labelled
independently of docking energies, this is **not circular**.

Two interpretations are possible:
- *(i) Undersampling:* shallow-exhaustiveness rigid docking fails to rank
  weak actives → a saturation sweep across exhaustiveness 8→64
  (`scripts/enrichment_saturation_analysis.py`) should recover AUC if this
  were the cause.
- *(ii) Fundamental:* rigid-receptor Vina against the shallow, polar,
  solvent-exposed apo site cannot separate weak β-lactam actives from
  property-matched decoys at any exhaustiveness.

**Do not** "fix" enrichment by reintroducing potency-scaled or
active-informative rescoring — that was removed in v7.2.0 for circularity and
is forbidden. If the saturation sweep plateaus below AUC 0.7, the negative
enrichment must be reported as a **key finding on PBP2a druggability and
rigid-docking discrimination**, not as a pipeline defect.

### Conserved active-site waters

Ordered waters exist near the catalytic triad in the source crystals (1VQQ: 7,
3ZG0: 2, 4DKI: 1 within 5 Å), with **1 position conserved across structures**
(`output/conserved_waters.json`). The docking protocol strips all waters before
PDBQT preparation, so water-mediated contacts and displacement penalties are
**not modelled**. This is a plausible contributor to the poor enrichment.
`output/conserved_waters.pdb` exports the conserved waters (reference frame) so
you can prepare a water-included receptor for "docking with waters" and re-run
the benchmark. A future direction is to quantify the enrichment impact of
retaining conserved waters.

### Our-set checks and bounds

- **Small library:** 3,116 compounds, of which ~392 passed filtering and were
  docked. This is small vs. typical VS campaigns and may under-represent
  decoy diversity.
- **Shallow primary-screen exhaustiveness (ex=8):** ~±2 kcal/mol energy noise;
  the 1–2 kcal/mol spread among top hits is within the noise window. Top
  candidates were re-docked at ex=32, which closely reproduced ex=8 ranking.
- **Short MD:** published explicit-solvent MD is 1 ns (single replica);
  at 1 ns one of three candidates (SEED_01150) is Metastable, but BRICS_0022
  dissociates and ALL_QU04 loses its catalytic contact. The 100 ns × 3
  replica campaign is in progress on the M5 Pro OpenCL platform (~13.7 ns/day).

### Future directions (in priority order)

1. Run the enrichment saturation sweep and the extended 100 ns × 3-replica MD
   on HPC/GPU, then a trajectory-ensemble MM-GBSA/FEP on the surviving
   candidates.
2. Re-run the benchmark with conserved active-site waters retained in the
   receptor to test their impact on discrimination.
3. Apply flexible `--flex` catalytic side-chain docking and induced-fit
   docking as the post-screen refinement tier.
4. If enrichment cannot be recovered, publish the negative result as
   evidence on the limits of rigid-receptor docking for PBP2a.