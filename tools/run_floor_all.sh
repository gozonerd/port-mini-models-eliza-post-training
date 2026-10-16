#!/bin/bash
# Wait until all seven pinned snapshots are complete and the download has exited, then run the floor
# on each model, smallest first, both framings, on the processor. Log: runs/floor.log
set -u
cd "$(dirname "$0")/.."
PY=~/.venvs/eval-stack/bin/python
HUB=~/.cache/huggingface/hub
declare -a MODELS=(
  "EleutherAI/pythia-70m a39f36b100fe8a5377810d56c3f4789b9c53ac42"
  "HuggingFaceTB/SmolLM2-135M 93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
  "EleutherAI/pythia-160m 50f5173d932e8e61f858120bcb800b97af589f46"
  "HuggingFaceTB/SmolLM2-360M f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
  "EleutherAI/pythia-410m 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
  "EleutherAI/pythia-1b f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"
  "HuggingFaceTB/SmolLM2-1.7B effd688a12921b4cc83e3312b6feb579f70f9c71"
)
mkdir -p runs
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a runs/floor.log; }
ready(){ for m in "${MODELS[@]}"; do set -- $m; d="$HUB/models--${1/\//--}/snapshots/$2"; for f in "$d"/*.safetensors; do [ -e "$f" ] && [ "$(stat -L -c %s "$f" 2>/dev/null || echo 0)" -gt 1000000 ] || return 1; done; [ -e "$d/config.json" ] || return 1; done; ! pgrep -f snapshot_download >/dev/null; }
log "waiting for the seven pinned snapshots"
until ready; do sleep 30; done
log "all seven present; starting floor runs"
for m in "${MODELS[@]}"; do set -- $m
  log "START $1 @ $2"
  $PY tools/measure_floor.py --model "$1" --revision "$2" --split test --batch 8 --note "floor, FLOOR_PROTOCOL.md" >> runs/floor.log 2>&1 && log "DONE $1" || log "FAILED $1"
done
log "floor complete"
