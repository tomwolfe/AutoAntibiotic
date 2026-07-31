#!/usr/bin/env bash
# AutoAntibiotic full compute pipeline
# Usage:
#   ./scripts/run_all_compute.sh              # full production run
#   ./scripts/run_all_compute.sh --quick       # quick verification (minutes)
#   ./scripts/run_all_compute.sh --background  # full run via nohup
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODE="${1:-full}"
LOG_DIR="$REPO/output/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " AutoAntibiotic Compute Pipeline"
echo " Mode: $MODE"
echo " Started: $(date)"
echo "=============================================="

# ---- Phase 1: Enrichment validation (runs against known_actives.csv) ----
echo ""
echo "[Phase 1] Enrichment validation..."
AUTOANTIBIOTIC_MODE=science python scripts/enrichment_validation.py 2>&1 | tee "$LOG_DIR/phase1_enrichment.log"
echo "  -> See $LOG_DIR/phase1_enrichment.log"

# ---- Phase 2: Explicit-solvent MD ----
echo ""
echo "[Phase 2] Explicit-solvent MD..."
if [ "$MODE" == "--quick" ]; then
    python scripts/explicit_solvent_md.py --quick 2>&1 | tee "$LOG_DIR/phase2_md.log"
    echo "  -> Quick MD done. See $LOG_DIR/phase2_md.log"
elif [ "$MODE" == "--background" ] || [ "$MODE" == "full" ]; then
    echo "  Starting full MD (background via nohup)..."
    nohup python scripts/explicit_solvent_md.py > "$LOG_DIR/phase2_md.log" 2>&1 &
    MD_PID=$!
    echo "  PID: $MD_PID  (log: $LOG_DIR/phase2_md.log)"
    echo "  Check: tail -f $LOG_DIR/phase2_md.log"
fi

# ---- Phase 3: MM-GBSA (depends on Phase 2) ----
if [ "$MODE" == "--quick" ]; then
    echo ""
    echo "[Phase 3] MM-GBSA rescoring..."
    if [ -f output/md_explicit/summary.json ]; then
        python scripts/mmgbsa_analysis.py 2>&1 | tee "$LOG_DIR/phase3_mmgbsa.log"
        echo "  -> See $LOG_DIR/phase3_mmgbsa.log"
    else
        echo "  SKIP: output/md_explicit/summary.json not found (MD not run yet)"
    fi
fi

# ---- Phase 4: DUD-E benchmark (depends on Phase 1 data) ----
if [ "$MODE" == "--quick" ] && [ -f output/enrichment_results.json ]; then
    echo ""
    echo "[Phase 4] DUD-E benchmark..."
    python -c "
import json, sys
sys.path.insert(0, '.')
from scripts.enrichment_validation import run_dude_benchmark, load_benchmark, compute_roc
records, labels = load_benchmark('data')
result = run_dude_benchmark(records, labels, 'output', 'output/dude_benchmark_results.json')
if result:
    print(f'  DUD-E done: AUC={result[\"auc\"]}, BEDROC={result[\"bedrock_alpha20\"]}')
" 2>&1 | tee "$LOG_DIR/phase4_dude.log"
fi

# ---- Summary ----
echo ""
echo "=============================================="
echo " Pipeline Status"
echo "=============================================="
if [ "$MODE" == "--background" ] || [ "$MODE" == "full" ]; then
    if [ -n "${MD_PID:-}" ]; then
        echo "  MD running in background (PID $MD_PID)"
        echo "  After MD completes, run:"
        echo "    python scripts/mmgbsa_analysis.py"
        echo "    python -c 'from scripts.enrichment_validation import *; ...' (DUD-E)"
    fi
fi
echo "  Verify: python verify_success.py"
echo "  Finished: $(date)"
echo "=============================================="
