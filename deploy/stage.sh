#!/usr/bin/env bash
# Stage the web build for the Cloudflare Worker (src/worker.js):
#   deploy/stage.sh            -> deploy/site/<file>.br
#   deploy/stage.sh --rom      -> also deploy/rom/ (tools/pack_rom.py)
# Every file is brotli-compressed at maximum quality; the Worker serves the
# .br copy with Content-Encoding: br (the 30 MB wasm travels as ~6 MB).
# Upload (the dataset must stay private):
#   hf upload <repo> deploy/site web --repo-type dataset
#   hf upload <repo> deploy/rom . --repo-type dataset
set -euo pipefail
cd "$(dirname "$0")/.."
FILES="index.html app.js compositor.js rom_worker.js audio_worklet.js coi-sw.js manifest.webmanifest icon.png recomp3ds.js recomp3ds.wasm"
rm -rf deploy/site && mkdir -p deploy/site
src() { case $1 in recomp3ds.*) echo build-web/$1 ;; *) echo web/$1 ;; esac; }   # page files straight from web/
for f in $FILES; do
  [ -f "$(src $f)" ] || { echo "missing $(src $f): build the web target first" >&2; exit 1; }
  brotli -f -q 11 -w 24 -o deploy/site/$f.br "$(src $f)" &
done
wait
for f in $FILES; do printf '%-18s %8d -> %8d\n' $f $(stat -f%z "$(src $f)" 2>/dev/null || stat -c%s "$(src $f)") $(stat -f%z deploy/site/$f.br 2>/dev/null || stat -c%s deploy/site/$f.br); done
# the default save (tools/make_default_save.py), when there is one
if [ -f deploy/default-save.json ]; then brotli -f -q 11 -o deploy/site/default-save.json.br deploy/default-save.json; echo "default-save.json staged"; fi
if [ "${1:-}" = "--rom" ]; then python3 tools/pack_rom.py "${R3DS_ROM:-../game.3ds}" deploy/rom; fi
