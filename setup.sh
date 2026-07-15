#!/usr/bin/env bash
#
# setup.sh — one-command install for AutoAntibiotic (non-Docker users)
# =====================================================================
#
# What this script does:
#   1. Ensures a conda/mamba distribution exists (installs Miniforge if not).
#   2. Creates a dedicated environment named "autoantibiotic".
#   3. Installs AutoDock Vina + OpenBabel from conda-forge (these are NOT pip
#      packages and are the only external binaries the pipeline needs).
#   4. Installs the Python package (with the [docking] extras) via pip.
#   5. Prints a friendly success message with the "next step" command.
#
# Usage:
#   bash setup.sh
#
# Designed to be read top-to-bottom by one person. No hidden magic.
#
set -euo pipefail

ENV_NAME="autoantibiotic"

echo "────────────────────────────────────────────────────────────"
echo "  AutoAntibiotic — environment setup"
echo "────────────────────────────────────────────────────────────"

# ── 1. Make sure conda/mamba is available ───────────────────────────────────
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
    CONDA_ROOT="$(dirname "$(dirname "$(command -v mamba)")")"
    echo "  ✓ Found mamba at: $(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
    CONDA_ROOT="$(dirname "$(dirname "$(command -v conda)")")"
    echo "  ✓ Found conda at: $(command -v conda)"
else
    echo "  ⚠ No conda/mamba found. Installing Miniforge (lightweight conda)…"
    INSTALLER="Miniforge3-$(uname)-$(uname -m).sh"
    INSTALL_DIR="${HOME}/miniforge3"
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}"

    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "$TMP/$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$URL" -O "$TMP/$INSTALLER"
    else
        echo "  ✗ Neither curl nor wget is available to download Miniforge." >&2
        echo "    Please install conda manually: https://conda.io/projects/conda/en/latest/install.html" >&2
        exit 1
    fi

    bash "$TMP/$INSTALLER" -b -p "$INSTALL_DIR"
    CONDA_BIN="$INSTALL_DIR/bin/conda"
    CONDA_ROOT="$INSTALL_DIR"
    # Make conda usable in this shell session.
    # shellcheck disable=SC1091
    source "$INSTALL_DIR/etc/profile.d/conda.sh"
    echo "  ✓ Miniforge installed to $INSTALL_DIR"
fi

# ── 2. Create the dedicated environment ─────────────────────────────────────
if "$CONDA_BIN" env list | grep -q "^${ENV_NAME}\s"; then
    echo "  • Environment '${ENV_NAME}' already exists — reusing it."
else
    echo "  • Creating conda environment '${ENV_NAME}' (Python 3.10)…"
    "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10
fi

# ── 3. Install the external binaries (vina + openbabel) ─────────────────────
echo "  • Installing AutoDock Vina + OpenBabel from conda-forge…"
"$CONDA_BIN" install -y -n "$ENV_NAME" -c conda-forge vina openbabel

# ── 4. Install the Python package ───────────────────────────────────────────
# Resolve the repository root (directory containing this script) so the
# install works regardless of the current working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "  • Installing the Python package from ${REPO_ROOT} (with [docking] extras)…"
"$CONDA_BIN" run -n "$ENV_NAME" pip install "$REPO_ROOT"[docking]

# ── 5. Success message + next step ──────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  ✅  Setup complete!"
echo ""
echo "  Activate the environment and run your first screen:"
echo ""
echo "      conda activate ${ENV_NAME}"
echo "      autoantibiotic --check          # verify Vina + OpenBabel"
echo "      autoantibiotic --count 10       # quick offline smoke test"
echo ""
echo "  Or screen a single compound:"
echo "      autoantibiotic --smiles \"CN1C(=O)C(N=C1C(=O)O)S...\""
echo "────────────────────────────────────────────────────────────"
