#!/bin/bash
# Score round-8 candidates against v2 on the held-out sets of research note §10.8 and §13.2, every model on the GPU
# (PyTorch MPS fp32: v2's student_from_onnx copy, the candidates' last.pt). Args: tag model.pt [model.pt ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
P=tools/mac_round8
TAG=$1; shift
V2=runs/models/v2.pt
M=$V2$(printf ",%s" "$@")
L=runs/nba/2024-12-25-LAL-GSW; C=runs/nba/2017-06-12-CLE-GSW
D=(.venv/bin/python $P/bench_mps.py -q)
O=runs/bench/local
echo "$(date) ball"
"${D[@]}" --models $M --ball-folder $L/teacher --held-out --json $O/${TAG}_lal_top.json > /dev/null
"${D[@]}" --models $M --ball-folder $L/teacher --held-out --link --json $O/${TAG}_lal_link.json > /dev/null
"${D[@]}" --models $M --ball-folder $C/teacher-check --json $O/${TAG}_cle_top.json > /dev/null
"${D[@]}" --models $M --ball-folder $C/teacher-check --link --json $O/${TAG}_cle_link.json > /dev/null
"${D[@]}" --models $M --ball-folder data/ft/bard_check_teacher_mac --json $O/${TAG}_bard_top.json > /dev/null
echo "$(date) coco"
"${D[@]}" --models $M --coco data/roboflow/heldout_ann.json --json $O/${TAG}_roboflow.json > $O/${TAG}_roboflow.txt
EB=$V2$( [ "$TAG" = r8 ] && echo ,runs/models/app-v1.pt )$(printf ",%s" "$@")
"${D[@]}" --models $EB --coco data/ebard/check_ann.json --json $O/${TAG}_ebard.json > $O/${TAG}_ebard.txt
echo "$(date) people"
echo "== people split, Lakers @ Warriors held out"
.venv/bin/python $P/people_split.py $L/teacher $L $M held 2>&1 | grep -v Warning | tail -$(( $# + 1 ))
echo "== people split, Cavaliers @ Warriors"
.venv/bin/python $P/people_split.py $C/teacher-check $C $M all 2>&1 | grep -v Warning | tail -$(( $# + 1 ))
echo "== paired players, Lakers @ Warriors"
.venv/bin/python $P/people_paired.py $L/teacher $L $M held 2>&1 | grep -v Warning | tail -20
echo "== paired players, Cavaliers @ Warriors"
.venv/bin/python $P/people_paired.py $C/teacher-check $C $M all 2>&1 | grep -v Warning | tail -20
echo "== ball"
.venv/bin/python $P/summ.py $O/${TAG}_lal_top.json $O/${TAG}_lal_link.json $O/${TAG}_cle_top.json $O/${TAG}_cle_link.json $O/${TAG}_bard_top.json
echo "== roboflow"; tail -30 $O/${TAG}_roboflow.txt
echo "== ebard check"; tail -30 $O/${TAG}_ebard.txt
echo "$(date) EVAL-DONE $TAG"
