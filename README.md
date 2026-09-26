# 3ds_recompiler — static 3DS recompiler for PC and Web

`3ds_recompiler` turns a Nintendo 3DS game's `code.bin` into C++ at build time
and links it against a high-level emulation (HLE) of the 3DS OS, a software
PICA200 renderer and a DSP audio mixer. The result runs the game as a native
program on Linux, macOS or Windows, or as WebAssembly in a browser — with RomFS
streamed on demand from your own `.3ds` file.

It has been brought up on one full retail title end to end (boot → title →
gameplay → town, audio, 60 fps, same saves native and web). Support for
further games is incremental: the lifter, kernel, GPU and runtime are
game-agnostic, while the HLE service layer is demand-driven — see
[Porting to another game](#7-porting-to-another-game).

> **No game data is included.** You need your own **decrypted** `.3ds` dump
> (for example from GodMode9 on your own console). The build generates C++
> from *your* ROM into `gen/`. That directory, the binaries and the web build
> are derived from the ROM: keep them for yourself and never commit or publish
> them (`.gitignore` enforces this).

---

## 1. Quick start

Every build starts the same way: put a decrypted ROM at the repo root as
`game.3ds`, or pass its path to the scripts.

| Target | One command | Output |
|---|---|---|
| Linux / macOS | `tools/build_native.sh "game.3ds"` | `build/recomp3ds` |
| Windows (x64) | `tools/build_windows.sh "game.3ds"` (MSYS2 or cross) | `build-win/recomp3ds.exe` |
| Web (WASM) | `tools/build_web.sh "game.3ds"` | `build-web/` (static site) |

Each script first runs `tools/lift_rom.sh`: it extracts `code.bin` and the
exheader from the ROM (`scripts/extract_rom.py`) and lifts the ARM code to C++
(about 20 s, Python 3 standard library only). Compiling takes a few minutes.

## 2. PC build in detail

### 2.1 Dependencies

```bash
# Debian/Ubuntu: sudo apt install python3 cmake clang libsdl2-dev
# Fedora:        sudo dnf install python3 cmake clang SDL2-devel
# Arch:          sudo pacman -S python cmake sdl2
# macOS:         xcode-select --install && brew install cmake sdl2 python sse2neon wgpu-native
```

`clang++` is preferred and picked automatically. Without SDL2 the program
still builds but runs headless, writing PNG snapshots only (`R3DS_HEADLESS=1`
+ `R3DS_SHOTS=dir`). With `wgpu-native` the WebGPU renderer is built in
(§5). On ARM64 hosts (Apple Silicon, Linux arm64) the shader interpreter's
SSE2 intrinsics go through `sse2neon` and guest FPSCR state maps straight onto
the host FPCR.

### 2.2 Windows

Either build natively in an **MSYS2 MINGW64** shell
(`pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make python git`) or
cross-compile from Linux/WSL (`sudo apt install g++-mingw-w64-x86-64-posix
cmake python3`), then:

```bash
tools/build_windows.sh "game.3ds"
```

The script downloads SDL2 into `winsdk/`. The result is
`build-win/recomp3ds.exe` plus `SDL2.dll`. The cross-built exe has been tested
under Wine.

### 2.3 Running

```
recomp3ds [path/to/rom.3ds] [--save DIR] [-v] [--svclog] [--ipclog] [--interp]
```

* The ROM path defaults to `game.3ds` in the current directory. Assets are
  read from it on demand: nothing is extracted at run time.
* Saves (savedata, extdata, sdmc) go to `save/`; change with `--save DIR` or
  `R3DS_SAVE_DIR`.
* At boot the runtime checks that the ROM's code matches the code the build
  was lifted from (FNV-1a hash). A different ROM needs a re-lift;
  `R3DS_FORCE=1` runs anyway.

### 2.4 Controls

| 3DS | Keyboard | Gamepad (by position, Nintendo layout) |
|---|---|---|
| D-pad | arrow keys | D-pad |
| Circle pad | W A S D | left stick |
| A / B / X / Y | K / J / I / U | right / bottom / top / left face button |
| L / R | Q / E | shoulders (and triggers on the web) |
| Start / Select | Enter / Backspace | Start / Back |
| Touch screen | left mouse on the bottom screen | — (on phones: your finger) |
| screenshot / quit | F12 / Esc (PC) | |

## 3. Display: the "remaster" layout

The 3DS has two screens. Stacking them works poorly on a PC monitor, so by
default the top screen fills the window and the bottom screen **floats over it
without its background**. Buttons, text and cursors stay; the backdrop becomes
transparent. Clicking through where the background used to be does nothing.

This is generic — no menu is special-cased. A coverage mask per render target
marks backdrop draws (triangles covering ≥ 85% of the viewport) versus content;
the mask rides along display transfers into the bottom screen's alpha channel.
A mostly-content bottom screen becomes a **minimap** in a corner instead of a
large overlay; a mostly-black one is keyed out by brightness.

| Setting | Web (Display ⚙) | Native |
|---|---|---|
| Layout: remaster / classic / large / side by side / top only | ✓ (`1`) | `F1` · `R3DS_LAYOUT=remaster\|classic\|large\|side\|top` |
| Bottom screen: automatic / always large / always minimap | ✓ | `F2` · `R3DS_BOTTOM=auto\|ui\|minimap` |
| Remove the bottom screen's background | ✓ (`3`) | `F3` · `R3DS_KEEP_BG=1` to disable |
| UI position | ✓ (`2`) | `F8` · `R3DS_UI_POS=right\|centre\|left\|top-right\|top-left` |
| UI scale | ✓ | `F9`/`F10` · `R3DS_UI_SCALE=0.62` |
| Renderer: GPU (WebGPU) / software | ✓ | `R3DS_RENDERER=soft` |
| Resolution: automatic / 1×–6× | ✓ | `F7` · `R3DS_RES=N`, `R3DS_MAX_RES=N` |
| Upscaling: bicubic / bilinear / sharp | ✓ | `F4` · `R3DS_FILTER=hq\|smooth\|pixel` |
| Aspect: auto / 16:9 / 5:3 / stretched | ✓ | `F5` · `R3DS_ASPECT=auto\|16:9\|5:3\|stretch` |
| Side bars: blurred / black | ✓ | `F6` |
| Fullscreen | button | `F11` · `R3DS_FULLSCREEN=1` |

## 4. Web build in detail

```bash
tools/build_web.sh "game.3ds"     # needs cmake + git; installs emsdk/ if no emcc
python3 tools/serve_web.py        # http://localhost:8080/
```

The output is a static site (`index.html`, `app.js`, `recomp3ds.js`,
`recomp3ds.wasm`, `rom_worker.js`, `audio_worklet.js`, `coi-sw.js`). It needs
cross-origin isolation (`SharedArrayBuffer` for threads) — `serve_web.py` sends
the COOP/COEP headers; `coi-sw.js` does it from a service worker for static
hosts that can't.

In the page: **Choose ROM file…** (optionally kept in OPFS for next time), or
**stream from a URL** — any server with HTTP Range support works.
`tools/serve_web.py --rom game.3ds` exposes it locally as `/rom.3ds`:
`http://localhost:8080/?rom=rom.3ds`. Never point this at a network you don't
control — it serves your ROM.

URL parameters: `?rom=<url>`, `&autostart=1`, `&env=NAME=VALUE` (repeatable),
`?layout=`, `?filter=`, `?aspect=`, `?bottomMode=`, `?renderer=`, `?res=`,
`?extdata=<id>` (save import target, see §6).

**Download-all mode** (`?romMode=download`, the default):
`tools/pack_rom.py` splits the ROM into independent 8 MiB brotli streams served
as `/rompack/<i>`; the page inflates them natively into OPFS and streams the
rest on demand until the copy is complete. It resumes after an interruption.

### 4.1 Hosting (privately)

`deploy/` holds a Cloudflare Worker (`src/worker.js`) that serves the site and
the ROM pack out of a private Hugging Face dataset behind an access cookie.
`deploy/stage.sh` brotli-compresses the site; edit `wrangler.toml` (`HF_REPO`,
your route) and `wrangler secret put HF_TOKEN` / `ACCESS_KEY`. A Durable
Object (`MultiplayerRoom`) relays `/multiplayer` websocket rooms — see
`rt/multiplayer.cpp`.

### 4.2 Browser requirements

WebAssembly threads + SIMD and ideally 4+ cores (the runtime uses about a
dozen threads). The guest address space is a 512 MB block; actual memory use
is roughly 600–900 MB.

## 5. GPU renderer (WebGPU)

Next to the software renderer (which stays the source of truth for guest
memory), `rt/hwr_rec.cpp` records every PICA draw before clipping and
`rt/hwr_gpu.cpp` replays the stream through WebGPU (wgpu-native on PC,
emdawnwebgpu in the browser) into surfaces at up to 6× resolution with real
widescreen. GPU surfaces are invalidated and reloaded from software results
whenever the guest touches memory the GPU missed, so a gap costs resolution,
never correctness.

High-resolution text: if `gen/fonts_gen.cpp` exists (a per-game font table,
see §7) the executor recognises the game's glyph-sheet textures and redraws
them from a vector font. `R3DS_HWR_BITMAP_TEXT=1` keeps the original glyphs.
`rt/hwr_icons.cpp` similarly rewrites small button-prompt textures with
keyboard keycaps (`R3DS_HWR_BUTTONS=0` disables).

Texture replacement: `rt/hwr_texrepl.cpp` serves `<key>.rgba` files from
`R3DS_TEX_DIR` natively, or a `textures.pack` on the web — an external tool
produces the pack keyed by content hash.

## 6. Saves

* Native: `save/` in the repo (or `--save DIR` / `R3DS_SAVE_DIR`).
* Web: `/save` is IndexedDB (IDBFS), synced periodically. **Export save** /
  **Import save** move saves between browsers and to/from the PC build as a
  JSON file map (`r3ds-save-1` format).
* Loose file import goes to the game's extdata directory — discovered once the
  game creates it, or forced with `?extdata=<id>`.
* `tools/make_default_save.py DIR` packs a save directory into
  `default-save.json`, seeded once on first load when the game has no save.

## 7. Porting to another game

The pipeline is ROM-agnostic; these are the places a new title meets:

* **Layout**: free. `scripts/extract_rom.py` reads the exheader's CodeSetInfo
  into `extracted/manifest.json`; `lift/lift.py` lifts against it and the
  runtime parses the same fields at boot (`rt/rom.cpp` `read_layout`). No
  constants to edit.
* **Code discovery**: automatic. The lifter works from the entry point, data
  pointers and jump tables; at run time the dispatcher hands misses to the
  interpreter (`rt/armint.cpp`) and logs them (`R3DS_MISS_LOG`, default
  `miss.log`). Feed misses back into `lift/seeds.txt` and re-lift.
* **Busy-wait hooks**: the SDK delay loop is patched via
  `R3DS_HOOK_DELAY=<pcs>` (see `rt/hle_hooks.cpp`). Find the game's
  `subs; bgt` spin function and pass its pc; without the hook a spin-polling
  main loop can starve lower-priority guest threads.
* **Services**: `rt/services.cpp` + `rt/hle_*.cpp` implement the IPC commands
  observed so far (srv, APT, fs, gsp, hid, cfg, ptm, dsp, y2r, err:f, plus
  generic OK stubs for the online/camera/etc ports). A new game hitting an
  unhandled command logs `[ipc] <svc>: unhandled cmd 0x…` — implement it there.
  Known gaps with real depth: `ldr:ro` (dynamic CRO modules), `csnd:SND`,
  library applets (swkbd/mii/error), camera, friends/NEX online.
* **Fonts** (optional, GPU renderer): `gen/fonts_gen.cpp` is generated by a
  per-game tool describing the game's font glyph order (`tools/` wrote one for
  the reference title). Without it, text renders at native resolution — no
  breakage.
* **Input scripts for tests**: `tools/web_smoke.js` takes `SMOKE_STEPS=file.json`
  (a JSON array of wait/key/tap/shot/probe steps) to drive a game headless.

## 8. Runtime knobs (environment; on the web use `?env=NAME=VALUE`)

| Variable | Effect |
|---|---|
| `R3DS_HEADLESS=1` | no window |
| `R3DS_SHOTS=dir`, `R3DS_SHOT_EVERY=N` | write a PNG every N frames |
| `R3DS_INPUT=...` | scripted input: `frame:buttonsHex[:dur]`, `frame:T:x:y` (touch), `frame:C:x:y` (circle pad) |
| `R3DS_SAVE_DIR=dir` | save location (native) |
| `R3DS_FORCE=1` | run even if the ROM's code differs from the lifted code |
| `R3DS_INTERP=1` | run everything on the interpreter |
| `R3DS_MISS_LOG=file` | log pcs the interpreter had to take (feed into `lift/seeds.txt`) |
| `R3DS_HOOK_DELAY=0x…,…` | host-replace SDK delay loops at these pcs |
| `R3DS_WATCHDOG=N`, `R3DS_PEEK=a,b` | dump guest thread states every N s |
| `R3DS_PROFILE=1` | interpreter entry profile → `profile.txt` |
| `R3DS_PICA_DBG=N`, `R3DS_REF_RASTER=1`, `R3DS_PICA_DUMP=F` | renderer debugging |
| `R3DS_GX_LOG=1`, `R3DS_GX_WATCH=va`, `R3DS_SYNC_GPU=1`, `R3DS_GX_SNAP=0`, `R3DS_GX_AHEAD=N`, `R3DS_VBLANK_WAIT=1`, `R3DS_VBLANK_HZ=hz` | GX pipeline control |
| `R3DS_RASTER_THREADS=N` | rasterizer / vertex threads |
| `R3DS_SHADER_CHECK=1` / `R3DS_SHADER_REF=1` | compare with / use the reference shader interpreter |
| `R3DS_WAV=file.wav` | record the audio output |
| `R3DS_LAYOUT`, `R3DS_BOTTOM`, `R3DS_KEEP_BG`, `R3DS_UI_POS`, `R3DS_UI_SCALE`, `R3DS_SWAP_KEYS`, `R3DS_RES`, `R3DS_MAX_RES`, `R3DS_FILTER`, `R3DS_ASPECT`, `R3DS_FULLSCREEN`, `R3DS_SCALE` | window layout (§3) |
| `R3DS_RENDERER=soft`, `R3DS_TEX_DIR=dir`, `R3DS_HWR_*` | renderer control and GPU-renderer debugging |
| `--svclog`, `--ipclog`, `-v` | log SVCs / IPC / verbose |

## 9. Architecture

```
game.3ds ──scripts/extract_rom.py──> extracted/code.bin + exheader/manifest ──lift/lift.py──> gen/lifted_XXXX.cpp ─┐
                                                                                                                  ├─> recomp3ds(.exe|.wasm)
rt/ runtime: kernel HLE, services, PICA200 renderer, DSP, interpreter ───────────────────────────────────────────────┘
       at run time: code.bin and every asset come straight from the .3ds (rt/rom.cpp)
```

* `lift/armdec.py` — bit-level ARMv6K decoder for ARM, Thumb-1 (with BL/BLX
  pairs) and VFPv2; one C++ statement per guest instruction.
* `lift/lift.py` — recursive-descent discovery, ARM jump tables, literal-pool
  pointer seeds, miss-log seeds; emits chunks with `goto` control flow. There
  are no function boundaries: a guest `pc` (bit 0 = Thumb) is the only state
  crossing chunk edges, so tail calls, shared epilogues, mid-function entries
  and `longjmp` are exact.
* `rt/dispatch.cpp` maps `pc -> chunk`; a miss goes to the interpreter.
* `rt/kernel.*` — HLE Horizon kernel: one host thread per guest thread under a
  strict single-core priority scheduler; events, mutexes, semaphores, timers,
  address arbiters, shared memory, IPC sessions.
* `rt/services.cpp`, `rt/hle_*.cpp` — Horizon services (see §7).
* `rt/pica*.cpp` — software PICA200: command lists, vertex/geometry shaders,
  clipping, texture cache (all formats incl. ETC1/A4), band-parallel fragment
  shading with TEV/blending/stencil/depth, async GX thread.
* `rt/hle_dsp_audio.cpp` — DSP voices (PCM8/16, DSP-ADPCM), resampling, mixers;
  SDL on PC, AudioWorklet on the web.
* `rt/rom.cpp` — NCSD/NCCH/ExeFS parsing, `.code` decompression (reverse LZSS),
  RomFS level-3 reads, exheader layout. `rt/platform.cpp` covers POSIX, Win32
  and Emscripten; `rt/web.cpp` is the browser bridge.

### 9.1 Floating point fidelity

Games may run VFP in RunFast (round-toward-zero, flush-to-zero, default NaN).
Natively the host FP control register mirrors the guest FPSCR; on wasm — where
there is no rounding-mode control — `lift_rt.h` emulates it exactly
(double-precision intermediates with TwoSum error terms, correctly-rounded
divide/sqrt, denormal flushing). Build natively with `-DR3DS_SOFT_FP` to
exercise the software path.

### 9.2 Verification

`lift/difftest.py` + `rt_test/difftest_main.cpp` run a differential test
against Unicorn: single guest instructions from random register, flag, VFP and
memory states (`DT_SPECIAL=1` adds ±0, ±inf, NaNs, denormals), across lifted
code or the interpreter (`DT_INTERP=1`), host or soft FPSCR
(`-DR3DS_SOFT_FP`). `R3DS_INTERP=1` runs the whole game on the interpreter as
an end-to-end check; `tools/web_smoke.js` does the same for the web build.

---

Progress on a new game is measured the same way it was made: run it, read the
`[ipc] unhandled cmd` / `ConnectToPort` / miss logs, and close the gaps one by
one.
