#!/bin/bash
# run_production_md.sh — Generate SLURM job scripts for top-5 candidate
# 100 ns × 3 replica explicit-solvent MD runs.
#
# Usage:
#   bash scripts/run_production_md.sh          # dry-run: prints job scripts
#   bash scripts/run_production_md.sh --submit  # prints + marks scripts ready
#
# Does NOT submit or run any jobs. Verifies that the generated scripts
# are syntactically valid and reference existing input files.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${REPO}/output"
MD_DIR="${OUT_DIR}/md_explicit"
CSV_PATH="${OUT_DIR}/top_candidates.csv"
SCRIPT_DIR="${REPO}/scripts"

PRODUCTION_NS=100
REPLICAS=3
N_CANDIDATES=5
PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-48:00:00}"
MEM="${MEM:-32G}"
HMR_FLAG="${HMR_FLAG:---hmr}"

echo "=== AutoAntibiotic v7.4.0 — Production MD Job Generator ==="
echo "  Production length : ${PRODUCTION_NS} ns"
echo "  Replicas per CID : ${REPLICAS}"
echo "  Candidates        : ${N_CANDIDATES}"
echo "  Partition         : ${PARTITION}"
echo "  Time limit        : ${TIME}"
echo "  Memory            : ${MEM}"
echo ""

# Verify prerequisites
if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: ${CSV_PATH} not found. Run the pipeline first."
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

# Extract top N candidate IDs from the CSV. (read-while-loop keeps this
# compatible with macOS GNU bash 3.2, which lacks the `mapfile` builtin.)
CANDIDATES=()
while IFS= read -r _cid; do
    [[ -n "$_cid" ]] && CANDIDATES+=("$_cid")
done < <(python3 -c "
import csv, sys
with open('${CSV_PATH}') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= ${N_CANDIDATES}:
            break
        print(row['Compound_ID'])
")

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: No candidates found in ${CSV_PATH}."
    exit 1
fi

echo "  Candidates:"
for cid in "${CANDIDATES[@]}"; do
    echo "    - ${cid}"
done
echo ""

# Generate a SLURM job script per candidate
mkdir -p "${SCRIPT_DIR}/slurm"

ALL_VALID=true

for cid in "${CANDIDATES[@]}"; do
    JOB_SCRIPT="${SCRIPT_DIR}/slurm/md_${cid}.sh"
    LOG_DIR="${OUT_DIR}/md_explicit/${cid}/slurm_logs"

    cat > "${JOB_SCRIPT}" << 'JOBHEADER'
#!/bin/bash
#SBATCH --job-name=md_@CID@
#SBATCH --partition=@PARTITION@
#SBATCH --time=@TIME@
#SBATCH --mem=@MEM@
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --output=@LOGDIR@/md_@CID@_%j.out
#SBATCH --error=@LOGDIR@/md_@CID@_%j.err
#SBATCH --constraint=fast

set -euo pipefail

REPO="@REPO@"
CID="@CID@"
PRODUCTION_NS=@PRODUCTION_NS@
REPLICAS=@REPLICAS@
HMR_FLAG="@HMR_FLAG@"

echo "=== Starting production MD for ${CID} ==="
echo "  Production: ${PRODUCTION_NS} ns × ${REPLICAS} replicas"
echo "  HMR: ${HMR_FLAG}"
echo "  Working directory: $(pwd)"

# Verify the top_candidates.csv exists and contains this CID
CSV_PATH="${REPO}/output/top_candidates.csv"
if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: ${CSV_PATH} not found."
    exit 1
fi

if ! grep -q "^${CID}," "${CSV_PATH}"; then
    echo "ERROR: ${CID} not found in top_candidates.csv."
    exit 1
fi

# Run the explicit-solvent MD pipeline. Platform auto-detection (Metal →
# CUDA → OpenCL → CPU, see utils/openmm_platform.py) picks the best backend;
# on Apple Silicon that is the Metal-backed OpenCL runtime (~8× CPU). Use
# --platform to force a specific backend. --resume makes a re-submitted job
# continue unfinished replicas from their last checkpoint instead of
# restarting (production writes checkpoints every --checkpoint-interval steps).
cd "${REPO}"
python3 scripts/explicit_solvent_md.py \
    --production-ns "${PRODUCTION_NS}" \
    --replicas "${REPLICAS}" \
    --candidates "${CID}" \
    --resume ${HMR_FLAG}

echo "=== Production MD for ${CID} completed ==="
JOBHEADER

    # Replace placeholders (perl -pi works identically on macOS BSD and GNU).
    perl -pi -e "s/\@CID\@/${cid}/g" "${JOB_SCRIPT}"
    perl -pi -e "s/\@PARTITION\@/${PARTITION}/g" "${JOB_SCRIPT}"
    perl -pi -e "s/\@TIME\@/${TIME}/g" "${JOB_SCRIPT}"
    perl -pi -e "s/\@MEM\@/${MEM}/g" "${JOB_SCRIPT}"
    perl -pi -e "s|\@LOGDIR\@|${LOG_DIR}|g" "${JOB_SCRIPT}"
    perl -pi -e "s|\@REPO\@|${REPO}|g" "${JOB_SCRIPT}"
    perl -pi -e "s/\@PRODUCTION_NS\@/${PRODUCTION_NS}/g" "${JOB_SCRIPT}"
    perl -pi -e "s/\@REPLICAS\@/${REPLICAS}/g" "${JOB_SCRIPT}"
    perl -pi -e "s|\@HMR_FLAG\@|${HMR_FLAG}|g" "${JOB_SCRIPT}"

    chmod +x "${JOB_SCRIPT}"

    # Validate the generated script
    if bash -n "${JOB_SCRIPT}" 2>/dev/null; then
        echo "  [VALID] ${JOB_SCRIPT}"
    else
        echo "  [INVALID] ${JOB_SCRIPT}"
        ALL_VALID=false
    fi

    # Verify the candidate directory exists (or will exist after pipeline run)
    if [[ -d "${MD_DIR}/${cid}" ]]; then
        echo "    Existing MD dir: ${MD_DIR}/${cid}"
    else
        echo "    MD dir not yet created (will be created by pipeline): ${MD_DIR}/${cid}"
    fi
done

echo ""
echo "=== Summary ==="
echo "  Generated ${#CANDIDATES[@]} SLURM job scripts in ${SCRIPT_DIR}/slurm/"
echo "  Scripts are syntactically valid: ${ALL_VALID}"
echo ""
echo "  To submit (after pipeline has prepared top_candidates.csv):"
echo "    sbatch ${SCRIPT_DIR}/slurm/md_*.sh"
echo ""
echo "  NOTE: This script does NOT submit jobs or run MD."
echo "  It only generates and validates SLURM scripts."

if [[ "${ALL_VALID}" != "true" ]]; then
    exit 1
fi