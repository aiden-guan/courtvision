#!/bin/bash
# Double-click to set up (first run only) and open CourtVision Studio. See scripts/mac/courtvision.sh.
cd "$(dirname "$0")" || exit 1
scripts/mac/courtvision.sh start || { status=$?; read -r -p "Press Return to close." _; exit "$status"; }
