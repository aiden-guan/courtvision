#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Set up CourtVision first:'
  print '  python3 -m venv .venv'
  print '  .venv/bin/pip install -e ".[cv,dev]"'
  read '?Press Return to close.'
  exit 1
fi
exec .venv/bin/python -m courtvision studio --open
