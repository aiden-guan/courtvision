#!/bin/bash
# Round 8 on the Mac (research note §13.1-13.4), one heavy job at a time (it first waits for a labelling job already
# running, as when it took over from an earlier version of itself): the new
# clips' ball labels (prep_mps.py, 100 clips a batch, raw clips deleted after), R8a, E-BARD, R8e, SportsMOT's officials
# promoted (mi_mps.py), R8f, the scores of R8-ctrl/R8a/R8e/R8f, then court-persons, R8b and its scores.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=tools/mac_round8
F=(.venv/bin/python tools/student_finetune.py)
V2=courtvision/server/static/nba-student.onnx
RECIPE=(--init runs/models/v2.pt --steps 4000 --static-p 0.25 --lr 2e-4 --freeze stem,l1 --hue 30 --zoom-min 0.7
  --zoom-max 1.4 --ball-weight 2.0 --workers 4 --threads 6 --balance --device mps --seed 0 --eval-every 1000
  --eval-frames 6)
REST=(--data data/ft/coco3_ref --data data/ft/bard_calib_m1 --data data/ft/lal_teacher_court)
IGN=(--data data/ft/mot_train_ign --data data/ft/mot_val_ign "${REST[@]}")
REF=(--data data/ft/mot_train_ref --data data/ft/mot_val_ref "${REST[@]}")
train() {  # name, all --data args
  local name=$1; shift
  if [ ! -f runs/models/$name/summary.json ]; then
    echo "$(date) train $name: $*"
    "${F[@]}" train "$@" "${RECIPE[@]}" --out runs/models/$name > runs/logs/train_$name.log 2>&1
  fi
  ln -sfn $name/last.pt runs/models/$name.pt
  [ -f runs/models/$name.onnx ] || "${F[@]}" export runs/models/$name/last.pt --out runs/models/$name.onnx 2>&1 | tail -1
  echo "$(date) $name TRAINED"
}
while pgrep -f "$P/prep_mps.py|fetch_data.py bard --clips calib_clips_round8" > /dev/null; do sleep 20; done
echo "$(date) took over"

RAW=data/bard/split/calib8; OUT=data/ft/bard_calib8
for first in 0 100 200 300 400 500; do
  if [ -f $OUT/meta.json ] && [ "$(.venv/bin/python -c "import json;print(len(json.load(open('$OUT/meta.json'))['clips']))")" -ge $((first + 100)) ]; then continue; fi
  echo "$(date) fetch clips $first-$((first + 99))"
  .venv/bin/python tools/fetch_data.py bard --clips calib_clips_round8 --out $RAW --first $first --count 100 2>&1 | tail -1
  .venv/bin/python $P/prep_mps.py $RAW $OUT --append --model $V2 --pt runs/models/v2.pt --max-frames 60 --jpeg-quality 80 2>&1 | tail -2
  rm -rf "${RAW:?}"; df -h . | tail -1
done
echo "$(date) PREP-DONE"

( [ -f data/ebard/calib_ann.json ] || .venv/bin/python tools/fetch_data.py ebard --out data/ebard
  [ -f data/ft/ebard_calib/meta.json ] || "${F[@]}" prep-coco data/ebard/calib_ann.json data/ebard/images data/ft/ebard_calib --referees
  echo "$(date) EBARD-READY" ) > runs/logs/ebard.log 2>&1 &
EB=$!
train r8a "${IGN[@]}" --data $OUT
wait $EB; tail -3 runs/logs/ebard.log
train r8e "${IGN[@]}" --data $OUT --data data/ft/ebard_calib

for s in train val; do
  [ -f data/ft/mot_${s}_ref/meta.json ] || .venv/bin/python $P/mi_mps.py data/ft/mot_$s data/ft/mot_${s}_ref --promote > runs/logs/mi_$s.log 2>&1
  tail -2 runs/logs/mi_$s.log
done
train r8f "${REF[@]}" --data $OUT --data data/ft/ebard_calib

$P/eval.sh r8 runs/models/r8-ctrl.pt runs/models/r8a.pt runs/models/r8e.pt runs/models/r8f.pt > runs/logs/eval_r8.log 2>&1
echo "$(date) EVAL-R8-DONE"

[ -f data/ft/bard_calib8_cp/meta.json ] || .venv/bin/python $P/cp_parallel.py $OUT data/ft/bard_calib8_cp \
  runs/models/app-v1.onnx 6 6 > runs/logs/court_persons.log 2>&1
tail -1 runs/logs/court_persons.log
train r8b "${IGN[@]}" --data data/ft/bard_calib8_cp
$P/eval.sh r8b runs/models/r8b.pt > runs/logs/eval_r8b.log 2>&1
echo "$(date) CHAIN-DONE"
