#!/usr/bin/env python3
"""Generate t-SNE library characterization figure (Fix 7)."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.manifold import TSNE

BASE = Path(__file__).resolve().parent.parent
LIB_PATH = BASE / "data" / "screen_library_final.csv"
OUT_PATH = BASE / "output" / "figures" / "library_tsne.png"

# Read library
smiles_list = []
ids_list = []
with open(LIB_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        smiles_list.append(row["smiles"])
        ids_list.append(row["compound_id"])

print(f"Read {len(smiles_list)} compounds from library")

# Compute Morgan fingerprints
mfpgen = GetMorganGenerator(radius=2, fpSize=2048)
fps = []
valid_smiles = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros((1,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)
    valid_smiles.append(smi)

fps = np.array(fps)
print(f"Computed {len(fps)} Morgan fingerprints (shape: {fps.shape})")

# Scaffold family from compound ID prefix (extract alphabetic prefix)
import re
def scaffold_family(cid):
    cid = cid.replace("-", "_")
    parts = cid.split("_")
    prefix = parts[0] if parts else cid
    alpha = re.match(r"([A-Za-z]+)", prefix)
    base = alpha.group(1) if alpha else prefix
    mapping = {
        "AAB": "AAB", "ALL": "ALL", "ALLO": "ALL",
        "BRIC": "BRICS", "BRICS": "BRICS",
        "DIV": "DIV", "DECOY": "DECOY",
        "SEED": "SEED", "NEW": "NOVEL",
        "CTRL": "CONTROL", "LIT": "LITERATURE",
        "ACT": "ACTIVE",
    }
    return mapping.get(base, base)

families = [scaffold_family(cid) for cid in ids_list if Chem.MolFromSmiles(smiles_list[ids_list.index(cid)])]
# Need to align families with valid smiles
valid_families = []
for smi in valid_smiles:
    idx = smiles_list.index(smi)
    valid_families.append(families[idx])

unique_families = sorted(set(valid_families))
family_to_idx = {f: i for i, f in enumerate(unique_families)}
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_families)))

print(f"Found {len(unique_families)} scaffold families: {unique_families}")

# t-SNE
print("Running t-SNE (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords = tsne.fit_transform(fps)
print(f"t-SNE done. Shape: {coords.shape}")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
for i, fam in enumerate(unique_families):
    mask = np.array([f == fam for f in valid_families])
    ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=fam, alpha=0.6, s=8)

ax.set_title("Chemical Space Coverage of Screening Library (t-SNE)", fontsize=14)
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.legend(markerscale=3, fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300)
print(f"Figure saved: {OUT_PATH}")
