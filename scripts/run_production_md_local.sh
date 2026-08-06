#!/bin/bash
# run_production_md_local.sh — Run production MD locally on a single Apple
# Silicon Mac (no SLURM), with staggered starts and optional duty-cycling.
#
# The full campaign target is N_CANDIDATES (default top 5) × N_REPLICAS (3) ×
# PRODUCTION_NS (100) ns of explicit-solvent NPT on OpenCL. At ~4.2 ns/day for
# the 426k-atom PBP2a system this is a *multi-week* total, so each candidate is
# submitted as its own background job that uses --resume, meaning it can be
# killed (Ctrl-C / `pkill`) and re-invoked to carry on from the last
# checkpoint instead of restarting.
#
# Features:
#   * One background job per candidate. MAX_CONCURRENT (default 1) caps how many
#     candidates run at once: the 426k-atom PBP2a OpenCL system is exclusive of
#     the M5 GPU, and several concurrent MD contexts on one device stall each
#     other (observed hang at 5 parallel jobs). With the default, candidates run
#     strictly one at a time; each candidate's replicas run sequentially inside
#     one job. STAGGER seconds still space out launches.
#   * Optional duty-cycling via DUTY_ON_MIN / DUTY_OFF_MIN to limit thermal
#     throttling: each job does an "on" run then sleeps "off" minutes.
#   * --platform auto (Metal -> OpenCL -> CPU). Set OPENMM_PLATFORM to force.
#   * Checkpoint every --checkpoint-interval steps; NaN/crash auto-restart is
#     handled inside explicit_solvent_md.py.
#
# Usage:
#   bash scripts/run_production_md_local.sh                 # launch all jobs
#   PRODUCTION_NS=50 STAGGER=120 bash scripts/run_production_md_local.sh
#   DUTY_ON_MIN=55 DUTY_OFF_MIN=5 bash scripts/run_production_md_local.sh
#   MAX_CONCURRENT=2 bash scripts/run_production_md_local.sh
#   bash scripts/run_production_md_local.sh --cids BRICS_0022,ALL_QU04
#   bash scripts/run_production_md_local.sh --nano          # tiny 0.05 ns smoke
#
# State/logs:
#   output/md_explicit/<CID>/slurm_logs/md_<CID>.log   (raw runner output)
#   output/md_explicit/<CID>/summary.json               (per-candidate results)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${REPO}/output"
MD_DIR="${OUT_DIR}/md_explicit"
CSV_PATH="${OUT_DIR}/top_candidates.csv"

# ── Config (override via env) ───────────────────────────────────────────────
PRODUCTION_NS="${PRODUCTION_NS:-100}"
REPLICAS="${REPLICAS:-3}"
N_CANDIDATES="${N_CANDIDATES:-5}"
STAGGER="${STAGGER:-60}"                 # seconds between candidate launches
# Max candidates running at once. The 426k-atom PBP2a system is exclusive of
# the M5 GPU: several concurrent OpenCL MD contexts on the same GPU contend for
# the device and can stall/hang each other. Default 1 runs candidates strictly
# one at a time (each candidate's replicas run sequentially inside its job).
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25000}"
# Comma-separated CIDs to re-run even if summary.json already reports success
# (e.g. after a completed trajectory was lost). Empty by default.
FORCE_RERUN="${FORCE_RERUN:-}"
# Duty-cycle (thermal throttle mitigation). 0 disables the pause loop.
DUTY_ON_MIN="${DUTY_ON_MIN:-0}"
DUTY_OFF_MIN="${DUTY_OFF_MIN:-0}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-OpenCL}"

CIDS=""
NANO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cids) CIDS="$2"; shift 2 ;;
        --nano) NANO=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "${CIDS}" == "" ]]; then
    if [[ ! -f "${CSV_PATH}" ]]; then
        echo "ERROR: ${CSV_PATH} not found."; exit 1
    fi
    CIDS=""
    while IFS= read -r _cid; do
        [[ -n "$_cid" ]] && CIDS="${CIDS:+${CIDS},}${_cid}"
    done < <(python3 -c "
import csv,sys
with open('${CSV_PATH}') as f:
    for i,row in enumerate(csv.DictReader(f)):
        if i>=${N_CANDIDATES}: break
        print(row['Compound_ID'])
")
fi

if [[ "${NANO}" == "1" ]]; then
    echo "NANO smoke mode: 0.05 ns × 1 replica per candidate."
    PRODUCTION_NS=0.05; REPLICAS=1
fi

echo "=== AutoAntibiotic Production MD (local Apple Silicon) ==="
echo "  Candidates : ${CIDS}"
echo "  Production : ${PRODUCTION_NS} ns × ${REPLICAS} replicas"
echo "  Platform   : ${OPENMM_PLATFORM}"
echo "  Concurrency: ${MAX_CONCURRENT} job(s) at a time | Stagger: ${STAGGER}s | Duty cycle: ${DUTY_ON_MIN}on/${DUTY_OFF_MIN}off min"
echo ""

_md_complete() {
    local cid="$1"
    local s="${MD_DIR}/${cid}/summary.json"
    [[ -f "$s" ]] || return 1
    python3 -c '
import json, sys
p, target = sys.argv[1], float(sys.argv[2])
d = json.load(open(p))
sys.exit(0 if (d.get("success") and (d.get("npt_duration_ns") or 0) + 1e-9 >= target) else 1)
' "$s" "${PRODUCTION_NS}"
}

launch_one() {
    local cid="$1"
    local logdir="${MD_DIR}/${cid}/slurm_logs"
    mkdir -p "${logdir}"
    local logf="${logdir}/md_${cid}.log"
    echo "  Starting ${cid} → ${logf}"
    nohup bash -c '
        set -euo pipefail
        REPO="'"${REPO}"'"
        CID="'"${cid}"'"
        PRODUCTION_NS="'"${PRODUCTION_NS}"'"
        REPLICAS="'"${REPLICAS}"'"
        PLATFORM="'"${OPENMM_PLATFORM}"'"
        CKPT="'"${CHECKPOINT_INTERVAL}"'"
        DUTY_ON="'"${DUTY_ON_MIN}"'"
        DUTY_OFF="'"${DUTY_OFF_MIN}"'"
        cd "${REPO}"
        run_py() {
            python3 scripts/explicit_solvent_md.py \
                --production-ns "${PRODUCTION_NS}" --replicas "${REPLICAS}" \
                --candidates "${CID}" --platform "${PLATFORM}" \
                --checkpoint-interval "${CKPT}" --resume
        }
        if [[ "${DUTY_ON}" == "0" ]]; then
            run_py
        else
            # Time-boxed duty cycle: run for DUTY_ON minutes, cool for
            # DUTY_OFF minutes, resume from checkpoints (--resume) until the
            # replicas finish (exit 0). This bounds thermal throttling.
            while true; do
                run_py &
                PID=$!
                for _ in $(seq 1 "${DUTY_ON}"); do
                    if ! kill -0 "${PID}" 2>/dev/null; then break; fi
                    sleep 60
                done
                if kill -0 "${PID}" 2>/dev/null; then
                    echo "[duty-cycle] ${DUTY_ON} min window over; pausing ${DUTY_OFF} min"
                    kill "${PID}" 2>/dev/null || true
                    wait "${PID}" 2>/dev/null || true
                    sleep $((DUTY_OFF * 60))
                else
                    wait "${PID}"
                    rc=$?
                    if [[ $rc -eq 0 ]]; then
                        echo "[duty-cycle] ${CID} replicas complete"
                        break
                    fi
                    echo "[duty-cycle] run ended rc=${rc}; cooling ${DUTY_OFF} min"
                    sleep $((DUTY_OFF * 60))
                fi
            done
        fi
    ' > "${logf}" 2>&1 &
    local pid=$!
    echo "    → PID ${pid}"
    sleep "${STAGGER}"
}

OLDIFS="$IFS"; IFS=','
active=()
for cid in ${CIDS}; do
    [ -n "$cid" ] || continue
    # Skip candidates whose production already completed at the requested length
    # (summary.json success=True with npt_duration_ns >= target). Re-invoking
    # this launcher therefore continues, not repeats, the campaign.
    if _md_complete "${cid}" && [[ ",${FORCE_RERUN}," != *",${cid},"* ]]; then
        echo "  SKIP ${cid} (production already complete at requested ns)"
        continue
    fi
    # Respect the GPU concurrency cap: wait (poll) for a slot before launching.
    # Poll instead of `wait -n` (macOS ships bash 3.2, which lacks it).
    # NB: under `set -u`, expanding an empty array errors in bash 3.2, so every
    # expansion uses the guarded ${arr[@]+"${arr[@]}"} idiom.
    while [[ ${#active[@]} -ge ${MAX_CONCURRENT} ]]; do
        sleep 20
        compact=()
        for p in ${active[@]+"${active[@]}"}; do
            if kill -0 "$p" 2>/dev/null; then compact+=("$p"); fi
        done
        active=("${compact[@]+"${compact[@]}"}")
    done
    launch_one "$cid"
    active+=("$!")
done
IFS="$OLDIFS"

echo ""
echo "=== Launched candidate job(s) sequentially (concurrency ${MAX_CONCURRENT}). ==="
echo "  Logs: ${MD_DIR}/<CID>/slurm_logs/md_<CID>.log"
echo "  Re-launch this script any time to resume unfinished replicas from checkpoints."
echo "  Stop a job:  kill <PID>  ;  Resume: re-run this script."
exit 0