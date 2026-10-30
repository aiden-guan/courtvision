#!/bin/bash
# One-step local setup and launcher for CourtVision Studio on macOS.
#
#   scripts/mac/courtvision.sh start        set up if needed, load keys, open the Studio (default)
#   scripts/mac/courtvision.sh setup        create/refresh .venv and install dependencies
#   scripts/mac/courtvision.sh keys         add, replace or remove API keys (stored in the macOS Keychain)
#   scripts/mac/courtvision.sh install-app  put a CourtVision app in ~/Applications
#
# Double-clicking "Start CourtVision.command" runs `start`. API keys never touch the repo or a
# dotfile: they live in the login Keychain under the service name below and are exported only
# into the Studio process. Written for the bash 3.2 that ships with macOS.
set -eu

REPO="$(cd "$(dirname "$0")/../.." && pwd -P)"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
STAMP="$VENV/.courtvision-install"   # sha of pyproject.toml + extras at the last successful install
ASKED="$VENV/.courtvision-keys-asked"
EXTRAS="${COURTVISION_EXTRAS:-cv,tts}"
KEYCHAIN_SERVICE="CourtVision"
KEY_NAMES="ANTHROPIC_API_KEY TYPESAFE_API_KEY"
HOST="127.0.0.1"
PORT="${COURTVISION_PORT:-8000}"
URL="http://$HOST:$PORT/"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
ask_yes() {  # ask_yes "Question" default(y|n)
    local reply hint="[y/N]"
    [ "$2" = y ] && hint="[Y/n]"
    read -r -p "$1 $hint " reply || reply=""
    [ -z "$reply" ] && reply="$2"
    case "$reply" in [yY]*) return 0 ;; *) return 1 ;; esac
}

# ------------------------------------------------------------------ python

py_ok() {  # true when $1 is Python 3.11 or newer
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

find_python() {
    # 3.12 first: it is what the project is developed and tested on.
    local c
    for c in python3.12 python3.11 python3.13 python3 \
             /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
        if command -v "$c" >/dev/null 2>&1 && py_ok "$(command -v "$c")"; then
            command -v "$c"
            return 0
        fi
    done
    return 1
}

make_venv() {
    local python
    if python="$(find_python)"; then
        say "Creating .venv with $("$python" --version 2>&1) ($python)"
        "$python" -m venv "$VENV"
        return
    fi
    warn "CourtVision needs Python 3.11 or newer, and none was found."
    if command -v brew >/dev/null 2>&1; then
        if ask_yes "Install Python 3.12 with Homebrew now?" y; then
            brew install python@3.12
            python="$(brew --prefix python@3.12)/bin/python3.12"
            "$python" -m venv "$VENV"
            return
        fi
    elif command -v uv >/dev/null 2>&1 || ask_yes "Install uv (Astral's Python manager, no admin password) to fetch Python 3.12?" y; then
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            PATH="$HOME/.local/bin:$PATH"
        fi
        uv venv --seed --python 3.12 "$VENV"
        return
    fi
    open "https://www.python.org/downloads/macos/" 2>/dev/null || true
    die "Install Python 3.12 from python.org, then double-click Start CourtVision again."
}

install_hash() {
    { cat "$REPO/pyproject.toml"; echo "$EXTRAS"; } | shasum -a 256 | cut -d' ' -f1
}

needs_setup() {
    [ -x "$PY" ] || return 0
    py_ok "$PY" || return 0
    [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$(install_hash)" ] && return 1
    return 0
}

cmd_setup() {
    [ "$(uname -s)" = Darwin ] || warn "This script targets macOS; continuing anyway."
    if [ -x "$PY" ] && ! py_ok "$PY"; then
        warn "The existing .venv uses an older Python; recreating it."
        rm -rf "$VENV"
    fi
    [ -x "$PY" ] || make_venv
    say "Installing CourtVision and its video-analysis dependencies (the first run downloads ~1-2 GB, mostly PyTorch)..."
    "$PY" -m pip install --upgrade --quiet pip
    "$PY" -m pip install -e "$REPO[$EXTRAS]"
    install_hash > "$STAMP"
    say "Setup complete."
}

# ------------------------------------------------------------------ keys

keychain_get() {
    security find-generic-password -s "$KEYCHAIN_SERVICE" -a "$1" -w 2>/dev/null
}

key_label() {
    case "$1" in
        ANTHROPIC_API_KEY) echo "Anthropic API key (Claude colour commentary)" ;;
        TYPESAFE_API_KEY) echo "TypeSafe API key (Jev decision layer)" ;;
        *) echo "$1" ;;
    esac
}

store_key() {  # prompt for one key with hidden input and save it to the Keychain
    local name="$1" value
    read -r -s -p "Paste your $(key_label "$name") and press Return (leave empty to skip): " value || value=""
    echo
    value="$(printf '%s' "$value" | tr -d '[:space:]')"
    [ -z "$value" ] && { echo "Skipped."; return 0; }
    # `security -w` with a value puts the key on a command line for a moment; feeding it via
    # `security -i` keeps it out of the process list.
    printf 'add-generic-password -U -s %s -a %s -l %s -w %s\n' \
        "$KEYCHAIN_SERVICE" "$name" "$KEYCHAIN_SERVICE" "$value" | security -i >/dev/null
    echo "Saved to your login Keychain."
}

cmd_keys() {
    local name
    say "CourtVision API keys (both optional: without them the Studio uses local rules and template commentary)"
    for name in $KEY_NAMES; do
        if keychain_get "$name" >/dev/null; then
            echo "$(key_label "$name"): saved"
            if ask_yes "  Replace it?" n; then
                store_key "$name"
            elif ask_yes "  Remove it?" n; then
                security delete-generic-password -s "$KEYCHAIN_SERVICE" -a "$name" >/dev/null
                echo "  Removed."
            fi
        else
            echo "$(key_label "$name"): not set"
            store_key "$name"
        fi
    done
    mkdir -p "$VENV" && : > "$ASKED"
}

load_keys() {  # export saved keys unless the shell already provides them
    local name value
    command -v security >/dev/null 2>&1 || return 0
    for name in $KEY_NAMES; do
        eval "[ -n \"\${$name:-}\" ]" && continue
        if value="$(keychain_get "$name")" && [ -n "$value" ]; then
            export "$name=$value"
        fi
    done
}

# ------------------------------------------------------------------ app

cmd_install_app() {
    local dest="$HOME/Applications/CourtVision.app" launcher="$REPO/Start CourtVision.command" esc
    command -v osacompile >/dev/null 2>&1 || die "osacompile not found (it ships with macOS)."
    mkdir -p "$HOME/Applications"
    esc="$(printf '%s' "$launcher" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    rm -rf "$dest"
    osacompile -o "$dest" -e "do shell script \"open -a Terminal \" & quoted form of \"$esc\""
    say "Installed $dest"
    echo "Open it from Launchpad or Spotlight, or drag it to the Dock. It runs this checkout: $REPO"
    echo "If you move the repo folder, run this again."
}

# ------------------------------------------------------------------ start

studio_running() {
    curl -fsS -o /dev/null --max-time 2 "${URL}api/studio" 2>/dev/null
}

port_busy() {
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
}

cmd_start() {
    cd "$REPO"
    if studio_running; then
        say "CourtVision Studio is already running; opening $URL"
        open "$URL"
        return 0
    fi
    port_busy && die "Port $PORT is in use by another program. Quit it, or run with COURTVISION_PORT=8001."
    local first_run=0
    [ -x "$PY" ] || first_run=1
    needs_setup && cmd_setup
    if [ ! -f "$ASKED" ]; then
        echo
        cmd_keys
    fi
    if [ "$first_run" = 1 ] && [ ! -d "$HOME/Applications/CourtVision.app" ] && command -v osacompile >/dev/null 2>&1; then
        echo
        ask_yes "Add a CourtVision app to your Applications folder for next time?" y && cmd_install_app
    fi
    load_keys
    echo
    say "Starting CourtVision Studio at $URL (close this window or press Ctrl-C to stop)"
    exec "$PY" -m courtvision studio --host "$HOST" --port "$PORT" --open
}

case "${1:-start}" in
    start) cmd_start ;;
    setup) cmd_setup ;;
    keys) cmd_keys ;;
    install-app) cmd_install_app ;;
    *) die "usage: $0 [start|setup|keys|install-app]" ;;
esac
