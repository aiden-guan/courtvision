#!/usr/bin/env bash
# The heavy NBA jobs, one at a time (several at once ran an 18 GB Mac out of memory). Research use (docs/NBA.md).
#
#   tools/nba_mac_jobs.sh q1            # Q1 evaluation on the existing camera tracks and detection cache
#   tools/nba_mac_jobs.sh retrack-q1    # re-estimate the camera centre, re-track Q1 into $G-v2, evaluate it the same way
#   tools/nba_mac_jobs.sh train         # resume the student detector from runs/models/student/last.pt
#   tools/nba_mac_jobs.sh retrack-full  # re-track the whole game into $G-v2 (hours; resumes nothing: one file)
#   tools/nba_mac_jobs.sh full [ROOT]   # whole game in ROOT ($G or $G-v2): detection (into detections-full/, resumable
#                                       # chunks), ball, events, fusion, evaluation
#
# A step refuses to start while another CourtVision job runs or with less than MIN_FREE % of memory free, and a
# watchdog stops it if free memory falls below WATCH_FREE % while it runs. Each step logs to $G/logs/<step>.log.
# G, PY, EPOCHS (the training run's --epochs), DATA (teacher folders, space separated), WEIGHTS, DET_DIR, DET_WORKERS
# (parallel detection processes of full, 1 by default), MIN_FREE and WATCH_FREE can be set in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v caffeinate >/dev/null && [ -z "${NBA_AWAKE:-}" ]; then  # macOS: no sleep while a job runs
    NBA_AWAKE=1 exec caffeinate -i "$0" "$@"
fi

G=${G:-runs/nba/2024-12-25-LAL-GSW}
PY=${PY:-.venv/bin/python}
Q1_END=${Q1_END:-871}
EPOCHS=${EPOCHS:-12}
DATA=${DATA:-$G/teacher}
WEIGHTS=${WEIGHTS:-runs/models/student/best.pt}
DET_WORKERS=${DET_WORKERS:-1}  # each detection process takes about 1.7 GB; two share the Neural Engine about 1.5x faster
MIN_FREE=${MIN_FREE:-25}
WATCH_FREE=${WATCH_FREE:-8}
JOBS="tools/nba_(analyze|track_court|teacher_labels|scan_scorebug|ball_bench|diagnose|fusion_ab|calibrate_fusion|hard_examples|bug_study|loop)\.py|courtvision\.training\.(train_student|pseudo_labels|nba_cue_rows|fit_cue_models|ball_labels)"
V2=$G-v2  # the same game with new camera tracks; everything else is linked, so the old tracks stay for comparison
mkdir -p "$G/logs"

free_pct() {  # macOS system-wide free memory, % (empty elsewhere)
    if command -v memory_pressure >/dev/null; then
        memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: *\([0-9]*\)%.*/\1/p'
    fi
}

kill_tree() {  # a process and all its descendants
    local c
    for c in $(pgrep -P "$1"); do kill_tree "$c"; done
    kill -TERM "$1" 2>/dev/null || true
}

watchdog() {  # stop the whole step when free memory falls below WATCH_FREE % (before macOS starts killing apps)
    local main=$1 me free c
    me=$(sh -c 'echo $PPID')  # this subshell (macOS bash 3.2 has no BASHPID)
    while sleep 15; do
        free=$(free_pct)
        if [ -n "$free" ] && [ "$free" -lt "$WATCH_FREE" ]; then
            echo "watchdog: only ${free}% of memory free: stopping the step" >&2
            for c in $(pgrep -P "$main"); do
                if [ "$c" != "$me" ]; then kill_tree "$c"; fi
            done
            kill -TERM "$main" 2>/dev/null || true
            return
        fi
    done
}

guard() {
    if pgrep -f "$JOBS" >/dev/null; then
        echo "another CourtVision job is running (pgrep -fl nba_ / train_student): wait for it or stop it" >&2
        exit 3
    fi
    local free
    free=$(free_pct)
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then
        echo "only ${free}% of memory free (need ${MIN_FREE}%): close other apps first" >&2
        exit 3
    fi
    watchdog "$$" &
    WATCHDOG=$!
    trap 'kill "$WATCHDOG" 2>/dev/null || true' EXIT
}

video() { "$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['video'])" "$G/game.json"; }

link_v2() {
    mkdir -p "$V2/camera_parts"
    for f in game.json scorebug.jsonl espn_summary.json nba_pbp.csv nba_tracking.json detections; do
        if [ -e "$G/$f" ] && [ ! -e "$V2/$f" ]; then ln -s "$(cd "$G" && pwd)/$f" "$V2/$f"; fi
    done
}

evaluate() {  # game folder, tag, start s, end s (0: the whole game), detection folder
    local root=$1 tag=$2 start=$3 end=$4 det=${5:-detections}
    "$PY" tools/nba_analyze.py "$root" ball --det-dir "$det"
    "$PY" tools/nba_analyze.py "$root" events --tag "$tag" --start "$start" --end "$end" --det-dir "$det"
    "$PY" tools/nba_analyze.py "$root" fuse --tag "$tag"
    "$PY" tools/nba_analyze.py "$root" eval --tag "$tag"
    "$PY" tools/nba_analyze.py "$root" eval --tag "$tag" --file fused.jsonl
    "$PY" tools/nba_diagnose.py "$root" --tag "$tag" --file fused.jsonl
}

step=${1:-}
case "$step" in
    q1)
        guard
        evaluate "$G" q1 0 "$Q1_END" 2>&1 | tee "$G/logs/q1.log"
        ;;
    retrack-q1)
        guard
        link_v2
        {
            "$PY" tools/nba_track_court.py "$(video)" --scorebug "$G/scorebug.jsonl" \
                --out "$V2/camera_parts/part0.jsonl" --start 0 --duration "$Q1_END"
            evaluate "$V2" q1 0 "$Q1_END"
        } 2>&1 | tee "$G/logs/retrack-q1.log"
        ;;
    train)
        guard
        args=()
        for d in $DATA; do args+=(--data "$d"); done
        "$PY" -m courtvision.training.train_student "${args[@]}" --out runs/models/student --epochs "$EPOCHS" \
            --resume 2>&1 | tee "$G/logs/train.log"
        ;;
    retrack-full)
        guard
        link_v2
        center=$("$PY" -c "import json, sys; print(','.join(str(v) for v in json.loads(open(sys.argv[1]).readline())['center']))" \
            "$V2/camera_parts/part0.jsonl")  # the centre the Q1 re-track estimated
        "$PY" tools/nba_track_court.py "$(video)" --scorebug "$G/scorebug.jsonl" --center "$center" \
            --out "$V2/camera_parts/part0.jsonl.full" 2>&1 | tee "$G/logs/retrack-full.log"
        mv "$V2/camera_parts/part0.jsonl.full" "$V2/camera_parts/part0.jsonl"
        ;;
    full)
        guard
        root=${2:-$G}
        det=${DET_DIR:-detections-full}  # its own folder: detections/ holds the Q1 cache of the earlier weights
        {
            "$PY" tools/nba_analyze.py "$root" detect --workers "$DET_WORKERS" --chunks 8 --weights "$WEIGHTS" \
                --det-dir "$det"
            evaluate "$root" full 0 0 "$det"
        } 2>&1 | tee "$G/logs/full.log"
        ;;
    *)
        sed -n '2,14p' "$0"
        exit 2
        ;;
esac
