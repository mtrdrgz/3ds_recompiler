# 3ds_recompiler — agent notes

A static recompiler for Nintendo 3DS games: `lift/` turns `code.bin` (ARMv6K)
into C++, `rt/` is a high-level emulation of the 3DS OS (Horizon kernel,
services, PICA200 software renderer, DSP audio, interpreter fallback), and
`web/` + `deploy/` serve the WebAssembly build.

## Build & test

```bash
tools/build_native.sh "game.3ds"     # extract + lift + cmake + make
tools/build_web.sh "game.3ds"        # same, via emscripten -> build-web/
python3 lift/difftest.py             # gen_test/tests.cpp (needs capstone, unicorn)
```

`tools/lift_rom.sh` alone re-runs extract+lift (~20 s, pure Python stdlib).

## Layout conventions

* All guest-memory layout is **exheader-driven** (`rt/rom.cpp:read_layout`,
  `lift/lift.py:load_layout` reading `extracted/manifest.json`). Do not
  hardcode .text/.rodata/.data extents.
* Env vars / CMake defines / JS globals use the `R3DS_` prefix; the binary and
  wasm are `recomp3ds`. No game names in the tree — everything is
  per-ROM-parameterized.
* Runtime miss loop: run → `miss.log` (R3DS_MISS_LOG) → append to
  `lift/seeds.txt` → re-lift. `lift/seeds.txt` holds only hex addresses (no
  game data).
* Host hooks for SDK busy-waits: `R3DS_HOOK_DELAY=<pcs>` in
  `rt/hle_hooks.cpp` — mechanism is generic, addresses are per game.

## Hard rules

* **Never commit ROM data or derived artifacts**: `*.3ds`, `*.cci`, `*.cia`,
  `extracted/`, `gen/`, `build*/`, `save/`, `miss.log`, `deploy/rom`,
  `deploy/site`, `deploy/default-save.json`. `.gitignore` covers them; keep it
  that way.
* The web build must stay private: the wasm is generated from the ROM.
* Keep the interpreter (`rt/armint.cpp`) instruction-exact — it is the oracle
  fallback for everything the lifter misses. `difftest.py` verifies it per
  instruction against Unicorn.
* HLE services are demand-driven: when adding a command, implement the real
  semantics where known; a wrong silent stub is worse than an `unknown()` log.
