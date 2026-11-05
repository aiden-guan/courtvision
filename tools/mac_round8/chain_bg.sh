#!/bin/bash
# After the pipeline comparisons: the BARD check games' linked ball (§13.10), R8b's court-geometry labels (court fits in 4 single-threaded processes), R8b and R8g
# (research note §13.1, §13.8), and their scores. One heavy job at a time.
set -euo pipefail
cd /Users/aidenguan/Documents/Projects/courtvision
S=tools/mac_round8
F=(.venv/bin/python tools/student_finetune.py)
RECIPE=(--init runs/models/v2.pt --steps 4000 --static-p 0.25 --lr 2e-4 --freeze stem,l1 --hue 30 --zoom-min 0.7
  --zoom-max 1.4 --ball-weight 2.0 --workers 4 --threads 6 --balance --device mps --seed 0 --eval-every 1000
  --eval-frames 6)
REST=(--data data/ft/coco3_ref --data data/ft/bard_calib_m1 --data data/ft/lal_teacher_court)
IGN=(--data data/ft/mot_train_ign --data data/ft/mot_val_ign "${REST[@]}")
REF=(--data data/ft/mot_train_ref --data data/ft/mot_val_ref "${REST[@]}")
train() {
  local name=$1; shift
  if [ ! -f runs/models/$name/summary.json ]; then
    echo "$(date) train $name: $*"
    "${F[@]}" train "$@" "${RECIPE[@]}" --out runs/models/$name > runs/logs/train_$name.log 2>&1
  fi
  ln -sfn $name/last.pt runs/models/$name.pt
  [ -f runs/models/$name.onnx ] || "${F[@]}" export runs/models/$name/last.pt --out runs/models/$name.onnx 2>&1 | tail -1
  echo "$(date) $name TRAINED"
}
while pgrep -f "pipe_chain.sh" > /dev/null; do sleep 30; done
echo "$(date) BARD linked ball (research note §13.10)"
[ -f runs/bench/local/r8_bard_link.json ] || .venv/bin/python $S/bench_mps.py -q --models runs/models/v2.pt,runs/models/r8-ctrl.pt,runs/models/r8a.pt,runs/models/r8e.pt,runs/models/r8f.pt \
  --ball-folder data/ft/bard_check_teacher_mac --link --json runs/bench/local/r8_bard_link.json > /dev/null
.venv/bin/python $S/summ.py runs/bench/local/r8_bard_link.json
echo "$(date) court-persons"
[ -f data/ft/bard_calib8_cp/meta.json ] || OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python $S/cp_parallel.py data/ft/bard_calib8 data/ft/bard_calib8_cp runs/models/app-v1.onnx 4 6 \
  > runs/logs/court_persons.log 2>&1
tail -1 runs/logs/court_persons.log
train r8b "${IGN[@]}" --data data/ft/bard_calib8_cp
train r8g "${REF[@]}" --data data/ft/bard_calib8_cp --data data/ft/ebard_calib
$S/eval.sh r8bg runs/models/r8b.pt runs/models/r8g.pt > runs/logs/eval_r8bg.log 2>&1
echo "$(date) CHAIN3-DONE"
