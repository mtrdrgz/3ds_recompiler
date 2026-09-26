// Hardware (GPU) presentation renderer — shared between the recorder and the
// WebGPU executor.
//
// The software renderer (pica_raster.cpp / pica_fast.cpp) stays the source of
// truth for guest memory: everything the game can read back is produced by it.
// Next to it, the recorder (hwr_rec.cpp, GX thread) captures every PICA draw
// before clipping -- vertex shader outputs plus the fragment pipeline state --
// and the GPU transfer operations, and publishes them as a command stream. The
// executor (hwr_gpu.cpp, the frontend's thread) replays that stream with
// WebGPU at a higher resolution into its own surfaces, and presents the
// resulting screens. Whatever the GPU side does not have (a buffer the CPU
// wrote, a transfer from memory it never saw) falls back to the software
// image, so a gap costs resolution, never correctness.
#pragma once
#include "common.h"
#include <vector>
#include <atomic>

struct OutVertex;

// ---- recorder hooks (GX thread). Cheap no-ops unless an executor is attached.
extern std::atomic<bool> g_hwr_on;   // an executor is attached and consuming
void hwr_draw_begin();
bool hwr_skip_draw();   // the draw goes to a pass nobody sees: skip it
void hwr_draw_end(int cls);   // raster_draw_end(): 0 nothing drawn, 1 backdrop, 2 content
void hwr_triangle(const OutVertex &a, const OutVertex &b, const OutVertex &c);
void hwr_fill(u32 start, u32 end, u32 value, u32 width);
void hwr_display_transfer(u32 in_va, u32 out_va, u32 in_w, u32 in_h, u32 out_w, u32 out_h,
                          u32 in_fmt, u32 out_fmt, bool flip, bool to_tiled, u32 hs, u32 vs);
void hwr_texture_copy(u32 in_va, u32 out_va, u32 size, u32 in_wg, u32 out_wg);
void hwr_mem_written(u32 va, u32 size);   // DMA or any other write the GPU side did not do
void hwr_publish();                       // end of a GX command: hand the recorded commands over

// executor -> recorder
extern std::atomic<u32> g_hwr_req_scale;   // resolution multiplier the executor wants
extern std::atomic<u32> g_hwr_req_wide;    // top screen width factor (float bits), 1 = 5:3
extern std::atomic<u64> g_hwr_alive_ns;    // last time the executor took commands
bool hwr_take(std::vector<std::vector<u8>> &out);   // executor: all published chunks, in order
void hwr_request_reset();

// diagnostic counters (cumulative unless noted), read by the web page's log capture
enum HwrDbg {
    HD_RESETS, HD_QUEUE_BYTES /* now */, HD_INVALID_RANGE, HD_INVALID_HASH, HD_LOAD_COLOR, HD_LOAD_DEPTH,
    HD_TEX_UP, HD_TEX_BYTES, HD_TEX_SAME, HD_SKIP_PASS, HD_DRAWS, HD_TRIS,
    HD_EXE_FRAMES, HD_EXE_SOFT_TOP, HD_EXE_SOFT_BOT, HD_EXE_REPLAY_MS, HD_EXE_SUBMIT_MS, HD_EXE_PRESENT_MS,
    HD_FS_READS, HD_FS_BYTES, HD_FS_MS, HD_FS_MAX_MS /* max since last read */, HD_GX_QUEUE_MAX /* since last read */, HD_GX_CMDS, HD_GX_WAIT_MS, HD_REC_WAIT_MS, HD_VBLANK_WAIT_MS, HD_SNAP_MS, HD_SNAP_BYTES,
    HD_GAME_FRAMES, HD_GAME_GAP_MAX_MS /* since last read */, HD_GPU_BUSY_MS, HD_CPU_IDLE_MS, HD_GX_WAIT_MAX_MS /* since last read */, HD_CTX_SWITCHES, HD_PACE_MS /* now */, HD_PACE_NEW,
    HD_COUNT
};
extern double g_hwr_dbg[HD_COUNT];

// fast content hash (recorder and display must agree)
u64 hwr_hash(const u8 *p, size_t n);

// ---- what display_present() saw at the last vblank (display.cpp)
struct HwrScreen { u32 va = 0, bytes = 0; u64 hash = 0; bool valid = false; };
struct HwrFrameInfo {
    HwrScreen top, bot;
    bool masked = false;   // the bottom screen's alpha is a content mask
    bool blank = false;    // mostly-black bottom screen: key it by brightness
    float cover = 1.0f;    // mean mask of the bottom screen
    u32 frame = 0;
    u64 t_ns = 0;          // when the vblank happened (now_ns clock)
};
bool display_get_frame_info(std::vector<u32> &out, HwrFrameInfo &info);
// every frame info published since the last call, oldest first (frame pacing)
void display_frame_history(std::vector<HwrFrameInfo> &out);

// ---- command stream
enum HwrOp : u32 {
    HOP_RESET = 1,        // scale, wide_num, wide_den: drop every surface / texture
    HOP_SURF_NEW,         // id, kind, lw, lh, pw, ph
    HOP_SURF_FREE,        // id
    HOP_SURF_LOAD,        // id, lw, lh, has_cov, then lw*lh RGBA8 (color) or depth24|stencil<<24 (depth), then cov bytes
    HOP_SURF_CLEAR,       // id, rgba8 or depth24|stencil<<24, clear_cov
    HOP_TEX,              // id, w, h, hash lo, hash hi (of the decoded pixels), then w*h RGBA8 (rows in memory order)
    HOP_TEX_FREE,         // id
    HOP_DRAW,             // HwrDraw, then n * HWR_VFLOATS floats
    HOP_BLIT,             // HwrBlit
    HOP_LCD,              // HwrLcd: a display transfer into an LCD framebuffer
    HOP_TEX_FONT,         // id, w, h, scale, font, cw, ch, cols, cells, then w*h RGBA8 (memory order): a glyph sheet to redraw
};
enum HwrSurfKind : u32 { HSK_COLOR = 0, HSK_DEPTH = 1, HSK_LCD = 2 };

// one vertex as the GPU sees it: PICA clip-space position and attributes
// pos.xyzw, color.rgba, tc0.uv tc1.uv, tc2.uv view.xy, quat.xyzw, view.z
constexpr int HWR_VFLOATS = 21;

// per-draw uniform block (WGSL: array<vec4<u32>, HWR_UVEC4>); see hwr_gpu.cpp
constexpr int HWR_UVEC4 = 32;
enum : u32 {
    // u[2].x flags
    HUF_WBUF = 1u << 0, HUF_LIGHT = 1u << 1, HUF_TEX2_TC1 = 1u << 2, HUF_ALPHA_TEST = 1u << 3,
    HUF_EXCLUDE = 1u << 4, HUF_NOALPHA = 1u << 5, HUF_SQUEEZE = 1u << 6,
    HUF_TEX0 = 1u << 8, HUF_TEX1 = 1u << 9, HUF_TEX2 = 1u << 10,
};

// fixed-function state that becomes a render pipeline
struct HwrPipeKey {
    u32 blend;     // PICA 0x101 when blending, else 0
    u32 misc;      // bit0 blend on, bits1-4 logic-op class, bits 8-11 color mask, bits 12-13 cov mode, bits 14-15 cull
    u32 depth;     // bit0 test, bits1-3 func, bit4 write
    u32 stencil;   // bit0 on, bits1-3 func, bits4-6 fail, 7-9 zfail, 10-12 pass, 16-23 read mask, 24-31 write mask
    bool operator==(const HwrPipeKey &o) const { return blend == o.blend && misc == o.misc && depth == o.depth && stencil == o.stencil; }
};

struct HwrDraw {
    u32 color_id, depth_id;       // target surfaces (depth_id 0: none)
    u32 tex_id[3];                // surface or texture ids (0: unit off)
    HwrPipeKey pipe;
    u32 scissor[4];               // x0, y0, x1, y1 in the target's logical pixels, GPU orientation (y down)
    u32 stencil_ref;
    u32 blend_const;              // rgba8
    u32 nverts;
    u32 u[HWR_UVEC4 * 4];         // uniform block
};

struct HwrBlit {                   // copy src region -> whole dst (both logical coords, GPU orientation)
    u32 src_id, dst_id;
    u32 sx, sy, sw, sh;
    u32 flip;                     // mirror rows
    u32 cov;                      // also carry the coverage mask
    u32 noalpha;                  // destination format has no alpha: force 1
};

struct HwrLcd {
    u32 id;                        // HSK_LCD surface holding this image
    u32 va, bytes;                 // guest framebuffer and its size
    u64 hash;                      // content of the guest framebuffer after the transfer
};

// ---- texture replacements (hwr_texrepl.cpp, tools/texlab.py), executor side.
// Keyed by hwr_hash of the decoded RGBA as pica_tex_decoded returns it (bottom row
// first); tools/tex_extract.py computes the same keys straight from the ROM.
// query: 0 no replacement, 1 being loaded (ask again later), 2 ready (*w, *h set)
int hwr_texrepl_query(u64 hash, u32 *w, u32 *h);
void hwr_texrepl_copy(u64 hash, u32 *dst);   // after 2: w*h RGBA8, top row first (as HOP_TEX)

// ---- executor API (hwr_gpu.cpp; the frontend's thread)
enum HwrLayout { HWL_REMASTER = 0, HWL_CLASSIC, HWL_SIDE, HWL_TOP, HWL_LARGE };   // LARGE: Citra's "large screen"
enum HwrBottomMode { HWB_AUTO = 0, HWB_UI, HWB_MINIMAP };
enum HwrAspect { HWA_NATIVE = 0, HWA_AUTO, HWA_WIDE, HWA_STRETCH };   // 5:3, match the window, 16:9, 16:9 stretched
struct HwrView {
    int layout = HWL_REMASTER, bottom_mode = HWB_AUTO, remove_bg = 1;
    float ui_scale = 0.62f, ui_alpha = 0.96f, mini_scale = 0.36f, mini_alpha = 0.9f;
    int mini_corner = 0;        // bit0 left, bit1 top
    int ui_anchor = 0;          // 0 bottom right, 1 bottom centre, 2 bottom left, 3 top right, 4 top left
    int filter = 2;             // 0 sharp pixels, 1 bilinear, 2 bicubic
    int aspect = HWA_AUTO;
    int side_blur = 1, show_full = 0;
    int res = 0, max_res = 6;   // resolution multiplier: 0 = from the window size
};
struct HwrOut {
    bool ready = false, bottom_on = false, hit = false, animating = false, top_gpu = false, bot_gpu = false;
    bool bottom_focus = false;   // the bottom screen shows a menu (not a picture such as the map)
    float bx = 0, by = 0, bw = 0, bh = 0;   // bottom screen rectangle in target pixels
};
enum HwrTargetKind { HWT_NONE = 0, HWT_CANVAS, HWT_METAL, HWT_WIN32, HWT_X11, HWT_WAYLAND };
struct HwrSurfaceTarget { int kind = HWT_NONE; void *a = nullptr, *b = nullptr; const char *canvas = nullptr; };
bool hwr_gpu_start(const HwrSurfaceTarget &t);       // asynchronous on the web: poll hwr_gpu_state()
int hwr_gpu_state();                                 // 0 idle, 1 starting, 2 ready, 3 failed
bool hwr_gpu_frame(u32 w, u32 h, const HwrView &v, HwrOut &out, bool force_present);   // true: a new game frame was shown
u32 hwr_gpu_soft_pixel(int x, int y);               // the software frame (touch hit tests)
void hwr_gpu_test_loop(const char *dir, u32 every, u32 w, u32 h, const HwrView &v);
HwrView hwr_view_from_env();
