#!/bin/bash
# The pipeline on both README games (research note §10.14): detect on Core ML, ball, events (shirt rule unless a
# --referees is given as $3), timeline. usage: pipeline_r8.sh TAG WEIGHTS.pt [REFEREES] [DET_TAG]
# DET_TAG reuses another run's detections and ball track (no detect), e.g. main-c for v2's.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; W=$2; REF=${3:-shirt}; DT=${4:-$1}
A=(.venv/bin/python tools/nba_analyze.py)
for G in runs/nba/2024-12-25-LAL-GSW runs/nba/2017-06-12-CLE-GSW; do
  if [ "$G" = runs/nba/2017-06-12-CLE-GSW ]; then R=(--team-refit); else R=(); fi
  if [ ! -f $G/ball_track-$DT.json ]; then
    echo "$(date) $TAG detect $G"
    "${A[@]}" $G detect --det-dir detections-$DT --weights $W --backend coreml --workers 1 --chunks 8
    "${A[@]}" $G ball --det-dir detections-$DT --ball-track ball_track-$DT.json
  fi
  echo "$(date) $TAG events $G ($REF)"
  "${A[@]}" $G events "${R[@]}" --referees $REF --tag $TAG --det-dir detections-$DT --ball-track ball_track-$DT.json 2>&1 | tail -2
  "${A[@]}" $G timeline --tag $TAG --det-dir detections-$DT --ball-track ball_track-$DT.json 2>&1 | tail -2
done
echo "$(date) PIPELINE-DONE $TAG"
