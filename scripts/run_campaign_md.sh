#!/bin/bash
# Production MD campaign launcher — Apple Silicon M5 Pro, OpenCL + HMR
# Runs 10 ns × 3 replicas × top-3 candidates (resumable to 100 ns with --resume)
# 
# Usage:
#   bash scripts/run_campaign_md.sh              # launch (background-safe)
#   bash scripts/run_campaign_md.sh --extend-100  # extend to 100 ns after 10 ns completes

set -euo pipefail

cd "$(dirname "$0")/.."

NPT_NS="${NPT_NS:-10}"
REPLICAS="${REPLICAS:-3}"
CANDIDATES="SEED_01150,BRICS_0022,ALL_QU04"
PLATFORM="${PLATFORM:-OpenCL}"
LOG="output/md_explicit/logs/campaign_$(date +%Y%m%d_%H%M%S).log"

mkdir -p output/md_explicit/logs

echo "=== AutoAntibiotic Production MD Campaign ==="
echo "  Production: ${NPT_NS} ns × ${REPLICAS} replicas"
echo "  Candidates: ${CANDIDATES}"
echo "  Platform:   ${PLATFORM} (HMR + 4 fs)"
echo "  Log:        ${LOG}"
echo "  Started:    $(date)"
echo ""

if [ "${1:-}" = "--extend-100" ]; then
    NPT_NS=100
    echo "=== EXTENDING TO 100 ns ==="
fi

exec python scripts/explicit_solvent_md.py \
    --production-ns "${NPT_NS}" \
    --replicas "${REPLICAS}" \
    --candidates "${CANDIDATES}" \
    --platform "${PLATFORM}" \
    --hmr \
    --resume \
    --checkpoint-interval 25000 \
    2>&1 | tee "${LOG}"
