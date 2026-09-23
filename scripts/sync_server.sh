#!/bin/bash
# Push the code to the GPU server, with a record of which commit it is.
#
# The server copy is not a git checkout, so runs there cannot ask git what code
# they are. REVISION carries the commit and how many files under src/, scripts/
# and tests/ differed from it; src/runinfo.py copies it into run_manifest.json.
set -euo pipefail
cd "$(dirname "$0")/.."

commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain -- src scripts tests | wc -l | tr -d ' ')
echo "$commit dirty_files=$dirty synced=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > REVISION
if [ "$dirty" != "0" ]; then
  echo "warning: $dirty uncommitted file(s) under src/ scripts/ tests/ — runs will say so" >&2
fi

rsync -az src scripts tests REVISION kiis-mvf-gpu:~/jev-wm/
echo "synced $(cat REVISION)"
