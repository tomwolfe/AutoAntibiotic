#!/usr/bin/env bash
# Waits for D1 (troczi_site_diagnosis.py) to finish, then launches D4
# (dude_benchmark.py) in the background. D1 and D4 both need full CPU, so
# running them sequentially avoids resource contention.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "[launcher] Waiting for D1 to finish..."
while ps aux | grep -q "[t]roczi_site_diagnosis.py"; do
    sleep 60
done
echo "[launcher] D1 finished at $(date)"

D1_JSON="output/troczi_site_diagnosis.json"
if [ ! -f "$D1_JSON" ]; then
    echo "[launcher] ERROR: $D1_JSON missing after D1 exit"
    exit 1
fi
echo "[launcher] D1 result:"
python3 -c "import json; d=json.load(open('$D1_JSON')); print('  active AUC:', d.get('active_site',{}).get('auc'), '| allosteric AUC:', d.get('allosteric_site',{}).get('auc'), '| hypothesis_supported:', d.get('hypothesis_supported'))"

echo "[launcher] Launching D4 (DUD-E benchmark) at $(date)"
AUTOANTIBIOTIC_MODE=science nohup python3 scripts/dude_benchmark.py \
    > output/logs/dude_benchmark.log 2>&1 &
echo "[launcher] D4 PID: $!  (log: output/logs/dude_benchmark.log)"
echo "[launcher] Monitor: tail -f output/logs/dude_benchmark.log"
