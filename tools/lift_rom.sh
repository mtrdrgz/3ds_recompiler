#!/usr/bin/env bash
# Step 1 of every build: turn YOUR ROM into C++ (gen/, gitignored).
#   tools/lift_rom.sh ["path/to/game.3ds"]
# Extracts code.bin + exheader.bin from the (decrypted) .3ds into extracted/
# and runs the static recompiler. Needs only Python 3 (standard library). ~20 s.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
ROM=${1:-${R3DS_ROM:-game.3ds}}
if [ ! -f "$ROM" ]; then
  echo "ROM not found: '$ROM'" >&2
  echo "Pass the path to your decrypted .3ds, or put it at the repo root as 'game.3ds'." >&2
  exit 1
fi
echo "== extracting code.bin from $ROM"
"$PY" scripts/extract_rom.py "$ROM" extracted >/dev/null
echo "== lifting ARM code to C++ (gen/)"
"$PY" lift/lift.py
echo "== done: $(ls gen/lifted_*.cpp | wc -l) files in gen/"
