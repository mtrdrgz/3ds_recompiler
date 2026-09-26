#!/usr/bin/env bash
# Build recomp3ds.exe for Windows x64.
#   * on Linux / WSL: cross-compiles with mingw-w64 (apt install g++-mingw-w64-x86-64-posix)
#   * on Windows: run from an MSYS2 "MINGW64" shell with
#       pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make python
# Output: build-win/ (recomp3ds.exe + SDL2.dll)
#   tools/build_windows.sh ["path/to/game.3ds"]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
tools/lift_rom.sh "${1:-${R3DS_ROM:-game.3ds}}"
[ -d winsdk/SDL2-2.30.9 ] || "$PY" tools/fetch_win_deps.py
SDLDIR=winsdk/SDL2-2.30.9/x86_64-w64-mingw32
EXTRA=()
if [ "$(uname -s | cut -c1-5)" != "MINGW" ]; then EXTRA=(-DCMAKE_TOOLCHAIN_FILE=../mingw-toolchain.cmake); fi
mkdir -p build-win && cd build-win
cmake .. "${EXTRA[@]}" -DCMAKE_BUILD_TYPE=Release -DSDL2_DIR="$PWD/../$SDLDIR/lib/cmake/SDL2" -G "Unix Makefiles"
make -j"$(nproc 2>/dev/null || echo 4)"
cp "../$SDLDIR/bin/SDL2.dll" .
echo "built build-win/recomp3ds.exe (+ SDL2.dll next to it)"
echo "run it from the repo root:  build-win\\recomp3ds.exe \"game.3ds\""
