# Quickstart

Three copy-paste blocks to get running fast. See the [README](../README.md)
for full details.

## 1. Offline smoke test (CI mode)

No PDB downloads, completes in seconds — proves the install is wired up:

```bash
autoantibiotic --count 10
```

## 2. Real screening (science mode)

Create a `config.yaml` at the repo root:

```yaml
mode: science
```

Then run the full pipeline (downloads real PDBs, redocking validation, Vina):

```bash
autoantibiotic --count 50
```

## 3. Single-compound screen

Screen one SMILES instantly and print a docking summary:

```bash
autoantibiotic --smiles "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)O)C(=O)O)(C)O"
```
