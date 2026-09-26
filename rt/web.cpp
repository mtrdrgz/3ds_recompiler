// WebAssembly frontend glue. The browser page (web/) drives this:
//
//  * ROM streaming: guest threads block in web_stream_read() while a
//    dedicated JS worker (rom_worker.js) fills the request straight into
//    wasm memory from the user's File, an OPFS copy of it, or an HTTP URL
//    with Range requests. Only the bytes the game asks for are ever read.
//  * frames: the page polls web_frame() from requestAnimationFrame.
//  * input: the page calls web_input() with the pad / touch state.
//  * audio: a pthread moves DSP output into a ring buffer in wasm memory
//    that an AudioWorklet reads directly.
#ifdef __EMSCRIPTEN__
#include "common.h"
#include "input.h"
#include <emscripten.h>
#include <emscripten/threading.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>

// ------------------------------------------------------------ ROM stream
struct RomCtl {
    s32 state;    // 0 idle, 1 request posted, 2 done
    s32 ready;    // set to 1 by the page once the worker is serving
    double size;  // ROM size in bytes
    double off;   // request
    s32 len, dst;
    s32 result;   // bytes read, or -1
    s32 pad;
};
alignas(16) static RomCtl g_romctl;
static std::mutex g_rom_mtx;

extern "C" EMSCRIPTEN_KEEPALIVE RomCtl *web_rom_ctl() { return &g_romctl; }

static void wait_while(s32 *p, s32 v) {
    while (__atomic_load_n(p, __ATOMIC_SEQ_CST) == v) emscripten_futex_wait(p, (u32)v, 1000.0);
}
static void wait_ready() {
    if (__atomic_load_n(&g_romctl.ready, __ATOMIC_SEQ_CST)) return;
    LOG("[web] waiting for the page to provide the ROM...");
    wait_while(&g_romctl.ready, 0);
}

#include "hwr.h"
s64 web_stream_read_impl(void *buf, size_t n, u64 off);
// timed: the page's log capture reports ROM read latency
s64 web_stream_read(void *buf, size_t n, u64 off) {
    double t0 = emscripten_get_now();
    s64 r = web_stream_read_impl(buf, n, off);
    double ms = emscripten_get_now() - t0;
    g_hwr_dbg[HD_FS_READS]++; g_hwr_dbg[HD_FS_BYTES] += r > 0 ? (double)r : 0; g_hwr_dbg[HD_FS_MS] += ms;
    if (ms > g_hwr_dbg[HD_FS_MAX_MS]) g_hwr_dbg[HD_FS_MAX_MS] = ms;
    return r;
}
s64 web_stream_read_impl(void *buf, size_t n, u64 off) {
    std::lock_guard<std::mutex> lk(g_rom_mtx);
    wait_ready();
    s64 total = 0;
    u8 *p = (u8 *)buf;
    while (n) {
        size_t chunk = n > (8u << 20) ? (8u << 20) : n;
        g_romctl.off = (double)off;
        g_romctl.len = (s32)chunk;
        g_romctl.dst = (s32)(uintptr_t)p;
        g_romctl.result = -1;
        __atomic_store_n(&g_romctl.state, 1, __ATOMIC_SEQ_CST);
        emscripten_futex_wake(&g_romctl.state, 1);
        wait_while(&g_romctl.state, 1);
        s32 r = g_romctl.result;
        __atomic_store_n(&g_romctl.state, 0, __ATOMIC_SEQ_CST);
        if (r < 0) return total ? total : -1;
        total += r; p += r; off += r; n -= r;
        if ((size_t)r < chunk) break;
    }
    return total;
}
u64 web_stream_size() { wait_ready(); return (u64)g_romctl.size; }

// ---------------------------------------------------------------- frames
bool display_get_frame(std::vector<u32> &out);
u32 display_frame_count();
static std::vector<u32> g_web_frame(400 * 480, 0xFF000000u);
extern "C" EMSCRIPTEN_KEEPALIVE u32 *web_frame() { return display_get_frame(g_web_frame) ? g_web_frame.data() : nullptr; }
extern "C" EMSCRIPTEN_KEEPALIVE u32 web_frame_count() { return display_frame_count(); }
// bottom screen of the last frame: alpha carries the content mask (0 =
// backdrop) when known; coverage = mean alpha, masked = mask available
float display_bottom_coverage();
bool display_bottom_masked();
extern "C" EMSCRIPTEN_KEEPALIVE float web_bottom_coverage() { return display_bottom_coverage(); }
extern "C" EMSCRIPTEN_KEEPALIVE int web_bottom_masked() { return display_bottom_masked() ? 1 : 0; }

// ------------------------------------------------------- WebGPU renderer
// The page drives the hardware renderer (hwr_gpu.cpp) from its
// requestAnimationFrame loop on the browser's main thread, where the
// WebGPU objects live.
#include "hwr.h"
static float g_hwr_out[9];
extern "C" EMSCRIPTEN_KEEPALIVE int web_hwr_start() {
    HwrSurfaceTarget t; t.kind = HWT_CANVAS; t.canvas = "#screen";
    hwr_gpu_start(t);
    return hwr_gpu_state();
}
// returns -1 while not ready (hwr_gpu_state() says why), else bit 0: a new frame was shown
extern "C" EMSCRIPTEN_KEEPALIVE int web_hwr_frame(int w, int h, int layout, int bmode, int remove_bg, float ui_scale, float ui_alpha,
                                                  float mini_scale, float mini_alpha, int mini_corner, int ui_anchor, int filter,
                                                  int aspect, int side_blur, int show_full, int res, int max_res) {
    if (hwr_gpu_state() != 2) return -1;
    HwrView v;
    v.layout = layout; v.bottom_mode = bmode; v.remove_bg = remove_bg; v.ui_scale = ui_scale; v.ui_alpha = ui_alpha;
    v.mini_scale = mini_scale; v.mini_alpha = mini_alpha; v.mini_corner = mini_corner; v.ui_anchor = ui_anchor;
    v.filter = filter; v.aspect = aspect; v.side_blur = side_blur; v.show_full = show_full; v.res = res; v.max_res = max_res;
    HwrOut o;
    bool shown = hwr_gpu_frame((u32)w, (u32)h, v, o, false);
    g_hwr_out[0] = o.bx; g_hwr_out[1] = o.by; g_hwr_out[2] = o.bw; g_hwr_out[3] = o.bh;
    g_hwr_out[4] = o.bottom_on; g_hwr_out[5] = o.hit; g_hwr_out[6] = o.top_gpu; g_hwr_out[7] = (float)g_hwr_req_scale.load();
    g_hwr_out[8] = o.bottom_focus;
    return shown ? 1 : 0;
}
extern "C" EMSCRIPTEN_KEEPALIVE int web_hwr_state() { return hwr_gpu_state(); }
u32 display_frame_count();
// diagnostic counters for the page's log capture (see HwrDbg); [HD_COUNT] = vblanks so far
static double g_dbg_out[HD_COUNT + 1];
extern "C" EMSCRIPTEN_KEEPALIVE double *web_debug_stats() {
    memcpy(g_dbg_out, g_hwr_dbg, sizeof g_hwr_dbg);
    g_dbg_out[HD_COUNT] = display_frame_count();
    g_hwr_dbg[HD_FS_MAX_MS] = 0; g_hwr_dbg[HD_GX_QUEUE_MAX] = 0; g_hwr_dbg[HD_GAME_GAP_MAX_MS] = 0; g_hwr_dbg[HD_GX_WAIT_MAX_MS] = 0;   // maxima since the last read
    return g_dbg_out;
}
extern "C" EMSCRIPTEN_KEEPALIVE float *web_hwr_out() { return g_hwr_out; }
extern "C" EMSCRIPTEN_KEEPALIVE int web_hwr_soft_alpha(int x, int y) { return (int)(hwr_gpu_soft_pixel(40 + x, 240 + y) >> 24); }

// ----------------------------------------------------------------- input
extern "C" EMSCRIPTEN_KEEPALIVE void web_input(u32 buttons, int cx, int cy, int touch, int tx, int ty) {
    InputState in{};
    in.buttons = buttons;
    in.cx = (s16)cx; in.cy = (s16)cy;
    if (cx > 80) in.buttons |= 1u << 28;
    if (cx < -80) in.buttons |= 1u << 29;
    if (cy > 80) in.buttons |= 1u << 30;
    if (cy < -80) in.buttons |= 1u << 31;
    if (touch && tx >= 0 && tx < 320 && ty >= 0 && ty < 240) { in.touch = 1; in.tx = (u16)tx; in.ty = (u16)ty; }
    input_set(in);
}

// ----------------------------------------------------------------- audio
void dsp_audio_pull(s16 *dst, int frames);
size_t dsp_audio_queued();
struct AudioRing {
    s32 wr, rd;          // frame counters (monotonic, wrap mod 2^31)
    s32 cap;             // frames
    s32 rate;            // source sample rate
    s16 data[2 * 16384]; // interleaved stereo
};
alignas(16) static AudioRing g_ring{0, 0, 16384, 32728, {}};
extern "C" EMSCRIPTEN_KEEPALIVE AudioRing *web_audio_ring() { return &g_ring; }

static void audio_thread() {
    const s32 target = 32728 / 10;   // keep ~100 ms queued for the worklet
    s16 tmp[2 * 256];
    for (;;) {
        s32 fill = __atomic_load_n(&g_ring.wr, __ATOMIC_SEQ_CST) - __atomic_load_n(&g_ring.rd, __ATOMIC_SEQ_CST);
        if (fill < target && dsp_audio_queued() >= 256) {
            dsp_audio_pull(tmp, 256);
            s32 w = g_ring.wr;
            for (int i = 0; i < 256; i++) {
                s32 k = (w + i) & (g_ring.cap - 1);
                g_ring.data[2 * k] = tmp[2 * i];
                g_ring.data[2 * k + 1] = tmp[2 * i + 1];
            }
            __atomic_store_n(&g_ring.wr, w + 256, __ATOMIC_SEQ_CST);
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
}

bool frontend_run() {
    std::thread(audio_thread).detach();
    LOG("[web] frontend ready");
    return false;   // main() parks this thread; the page does the rest
}
#endif
