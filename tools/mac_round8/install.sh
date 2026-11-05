#!/bin/bash
# Install a round-8 candidate as the main model, as v2 was installed (research note §10.15): the Studio's shot models,
# rim model and shot judge refitted on the candidate's own pipeline runs (runs/nba/<game>/runs/TAG, detections-TAG,
# ball_track-TAG.json), v2's Studio files backed up, then the Studio's weights and the browser app's export (its
# manifest and reference outputs). usage: install.sh TAG "how the weights were made (the manifest's training)"
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; TRAINING=$2
CK=runs/models/$TAG/last.pt
LG=2024-12-25-LAL-GSW; CG=2017-06-12-CLE-GSW
A=(.venv/bin/python tools/nba_analyze.py)
M2='{"motion": true, "freeze": false, "backbone_lr": 3e-5, "epochs": 30, "lr": 1e-3}'
echo "$(date) shot models (ball zones)"
for G in $LG $CG; do
  R=runs/nba/$G/runs
  mkdir -p $R/$TAG-zones && cp $R/$TAG/events.jsonl $R/$TAG/run.json $R/$TAG-zones/
  "${A[@]}" runs/nba/$G timeline --tag $TAG-zones --zones --det-dir detections-$TAG --ball-track ball_track-$TAG.json | grep -E "shots" || true
done
.venv/bin/python tools/nba_shot_models.py --runs $LG:$TAG-zones,$CG:$TAG-zones --zones --out runs/models/shot_models-zones-$TAG.json
echo "$(date) rim clips"
for G in $LG $CG; do
  [ -d runs/nba/$G/rim_clips-$TAG ] || .venv/bin/python tools/nba_rim_clips.py runs/nba/$G --tag $TAG --ball-track ball_track-$TAG.json --out rim_clips-$TAG
done
echo "$(date) rim model, each game held out"
.venv/bin/python tools/nba_rim_model.py --games $LG,$CG --folds games --name rim-x --clips rim_clips-$TAG --real-only \
  --config "$M2" > runs/models/rim/rim-x-games-$TAG.log 2>&1
grep -A 30 "held-out, all folds" runs/models/rim/rim-x-games-$TAG.log | tr -d '\n' | sed 's/  */ /g'; echo
.venv/bin/python tools/nba_rim_model.py --games $LG,$CG --folds none --name rim-2g --clips rim_clips-$TAG --real-only \
  --config "$M2" --save runs/models/rim/rim-2g-$TAG.pt 2>&1 | grep -v epoch | tail -2
echo "$(date) shot judge"
.venv/bin/python tools/nba_live_judge.py --games $LG,$CG --tag $TAG --det-dir detections-$TAG \
  --ball-track ball_track-$TAG.json --clips rim_clips-$TAG --rim rim-x --rim-model runs/models/rim/rim-2g-$TAG.pt \
  --out runs/models/rim/live_judge-$TAG.json | tail -24
echo "$(date) install (v2's Studio files kept in previous-v2-1eee0dd1)"
S=runs/studio/models; B=$S/previous-v2-1eee0dd1
mkdir -p $B
for f in nba-student.pt rim-model.pt shot-judge.json shot-models.json; do [ -f $B/$f ] || cp $S/$f $B/$f; done
cp $CK $S/nba-student.pt
cp runs/models/rim/rim-2g-$TAG.pt $S/rim-model.pt
TAG=$TAG .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
d = json.loads(Path(f"runs/models/rim/live_judge-{os.environ['TAG']}.json").read_text())
d["rim_model"] = "rim-model.pt"  # beside the judge's file
Path("runs/studio/models/shot-judge.json").write_text(json.dumps(d, indent=1))
PY
cp runs/models/shot_models-zones-$TAG.json $S/shot-models.json
echo "$(date) browser export"
.venv/bin/python tools/export_student_browser.py --weights $S/nba-student.pt --video runs/nba/$LG/videos/full_981-1001.mp4 \
  --training "$TRAINING" | tail -5
shasum -a 256 $S/nba-student.pt courtvision/server/static/nba-student.onnx
echo "$(date) INSTALL-DONE $TAG"
