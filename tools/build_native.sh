#!/usr/bin/env bash
# Build the native PC version (Linux or macOS).
#   tools/build_native.sh ["path/to/game.3ds"]
# Needs: Python 3, CMake >= 3.16, a C++20 compiler (clang++ recommended, g++ works),
# and SDL2 development files for a window/audio/gamepads
#   Debian/Ubuntu: sudo apt install cmake clang libsdl2-dev
#   Fedora:        sudo dnf install cmake clang SDL2-devel
#   Arch:          sudo pacman -S cmake clang sdl2
#   macOS:         brew install cmake sdl2 sse2neon wgpu-native
# Without SDL2 it still builds, but runs headless (screenshots only).
# With wgpu-native (brew, or WGPU_NATIVE_ROOT=<release dir>) the GPU renderer
# is built in: higher resolution and widescreen (docs/RECOMP2.md 2.7).
# Output: build/recomp3ds
set -euo pipefail
cd "$(dirname "$0")/.."
tools/lift_rom.sh "${1:-${R3DS_ROM:-game.3ds}}"
CXX=${CXX:-$(command -v clang++ || command -v g++)}
JOBS=${JOBS:-$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)}
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX"
make -j"$JOBS"
echo
echo "built build/recomp3ds"
echo "run it from the repo root:  build/recomp3ds \"${1:-game.3ds}\""
