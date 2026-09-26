#!/usr/bin/env bash
# Build the WebAssembly version.
#   tools/build_web.sh ["path/to/game.3ds"]
# Needs: Python 3, CMake >= 3.16, git, and the Emscripten SDK. If `emcc` is
# not on PATH, the SDK is installed into emsdk/ (gitignored, ~1 GB).
# Output: build-web/ — a static site (index.html, app.js, recomp3ds.js,
# recomp3ds.wasm, workers). It contains code generated from YOUR ROM: keep it
# for yourself, do not publish it.
# Serve it:  python3 tools/serve_web.py   then open http://localhost:8080/
set -euo pipefail
cd "$(dirname "$0")/.."
tools/lift_rom.sh "${1:-${R3DS_ROM:-game.3ds}}"
if ! command -v emcc >/dev/null 2>&1; then
  if [ ! -f emsdk/emsdk_env.sh ]; then
    echo "== installing the Emscripten SDK into emsdk"
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git emsdk
    emsdk/emsdk install latest
    emsdk/emsdk activate latest
  fi
  # shellcheck disable=SC1091
  source emsdk/emsdk_env.sh >/dev/null
fi
JOBS=${JOBS:-$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)}
mkdir -p build-web && cd build-web
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DR3DS_WEB_DEBUG=OFF
emmake make -j"$JOBS"
echo
echo "built build-web/ ($(du -h recomp3ds.wasm | cut -f1) wasm)"
echo "serve it:  python3 tools/serve_web.py    then open http://localhost:8080/"
