// Display: composes both LCDs into one 400x480 image. Headless builds
// write PNG snapshots (R3DS_SHOTS=dir, R3DS_SHOT_EVERY=N frames); the last
// frame always goes to last_frame.png on request.
#include "gpu.h"
#include "mem.h"
#include "pica_fmt.h"
#include "input.h"
#include "platform.h"
#include "hwr.h"
u64 now_ns();
#include <vector>
#include <algorithm>
#include <string>
#include <atomic>
#include <deque>
#include <mutex>
#include <mutex>

static std::atomic<u32> g_frames{0};
static std::vector<u32> g_screen(400 * 480);
static std::string g_shot_dir;
static u32 g_shot_every = 0;
static std::mutex g_input_mtx;
static InputState g_input{};
struct ScriptEv { u32 frame, buttons, dur; u16 tx, ty; bool touch; s16 cx, cy; bool circle; };
static std::vector<ScriptEv> g_script;

u32 display_frame_count() { return g_frames; }

InputState input_get() {
    std::lock_guard<std::mutex> lk(g_input_mtx);
    InputState s = g_input;
    u32 f = g_frames;
    for (auto &e : g_script)
        if (f >= e.frame && f < e.frame + e.dur) {
            s.buttons |= e.buttons;
            if (e.touch) { s.touch = 1; s.tx = e.tx; s.ty = e.ty; }
            if (e.circle) { s.cx = e.cx; s.cy = e.cy; }
        }
    return s;
}
void input_set(const InputState &s) { std::lock_guard<std::mutex> lk(g_input_mtx); g_input = s; }

static u32 crc32_png(const u8 *d, size_t n, u32 c = 0) {
    static u32 t[256];
    if (!t[1]) for (u32 k = 0; k < 256; k++) { u32 v = k; for (int b = 0; b < 8; b++) v = (v >> 1) ^ (0xEDB88320u & (0u - (v & 1))); t[k] = v; }
    c = ~c;
    for (size_t i = 0; i < n; i++) c = t[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}
static void write_png(const std::string &path) {
    // minimal PNG: RGBA8, zlib stream made of stored (uncompressed) blocks
    const u32 W = 400, H = 480;
    std::vector<u8> raw;
    raw.reserve((W * 4 + 1) * H);
    for (u32 y = 0; y < H; y++) {
        raw.push_back(0);
        for (u32 x = 0; x < W; x++) {
            u32 c = g_screen[y * W + x] | 0xFF000000u;   // snapshots are opaque
            raw.push_back(c & 0xFF); raw.push_back((c >> 8) & 0xFF); raw.push_back((c >> 16) & 0xFF); raw.push_back(0xFF);
        }
    }
    std::vector<u8> z = {0x78, 0x01};
    u32 a = 1, b = 0;
    for (u8 v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw.size(); off += 65535) {
        u32 n = (u32)std::min<size_t>(65535, raw.size() - off);
        z.push_back(off + n == raw.size() ? 1 : 0);
        z.push_back(n & 0xFF); z.push_back(n >> 8); z.push_back(~n & 0xFF); z.push_back((~n >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    }
    u32 ad = (b << 16) | a;
    z.push_back(ad >> 24); z.push_back(ad >> 16); z.push_back(ad >> 8); z.push_back(ad);
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    auto be32 = [&](u32 v) { u8 x[4] = {(u8)(v >> 24), (u8)(v >> 16), (u8)(v >> 8), (u8)v}; fwrite(x, 1, 4, f); };
    auto chunk = [&](const char *type, const u8 *data, u32 n) {
        be32(n);
        std::vector<u8> td(type, type + 4);
        td.insert(td.end(), data, data + n);
        fwrite(td.data(), 1, td.size(), f);
        be32(crc32_png(td.data(), td.size()));
    };
    static const u8 sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    fwrite(sig, 1, 8, f);
    u8 ihdr[13] = {0, 0, W >> 8, W & 0xFF, 0, 0, H >> 8, H & 0xFF, 8, 6, 0, 0, 0};
    chunk("IHDR", ihdr, 13);
    chunk("IDAT", z.data(), (u32)z.size());
    chunk("IEND", nullptr, 0);
    fclose(f);
}

bool cov_fb_mask(u32 va, std::vector<u8> &out);   // pica_fast.cpp
static bool g_bottom_masked = false;
static float g_bottom_cover = 1.0f;   // mean alpha of the bottom screen (0..1)
bool display_bottom_masked() { return g_bottom_masked; }
float display_bottom_coverage() { return g_bottom_cover; }

// statistics over the masked bottom screen: returns the coverage (drives the automatic
// UI / minimap layout) and blank-screen keying: a bottom screen that is
// almost entirely black (cutscenes, fades) keeps only its non-black pixels
static float bottom_postprocess(bool *blank_out) {
    u64 n = 0, black = 0, asum = 0;
    for (int y = 240; y < 480; y++) for (int x = 40; x < 360; x++) {
        u32 c = g_screen[y * 400 + x];
        u32 a = c >> 24, r = c & 0xFF, g = (c >> 8) & 0xFF, b = (c >> 16) & 0xFF;
        n++;
        if (a > 128 && r + g + b < 40) black++;
    }
    bool blank = black * 100 >= n * 85;
    *blank_out = blank;
    for (int y = 240; y < 480; y++) for (int x = 40; x < 360; x++) {
        u32 &c = g_screen[y * 400 + x];
        if (blank) {
            u32 r = c & 0xFF, g = (c >> 8) & 0xFF, b = (c >> 16) & 0xFF, l = (r + g + b) / 3;
            u32 a = std::min<u32>(c >> 24, l < 12 ? 0 : std::min<u32>(255, (l - 12) * 8));
            c = (c & 0x00FFFFFFu) | (a << 24);
        }
        asum += c >> 24;
    }
    return n ? (float)asum / (255.0f * n) : 1.0f;
}

// with_mask: alpha = the renderer's coverage of the screen's content
// (0 = backdrop), when known; otherwise opaque.
// Returns BLIT_NONE when the screen has no framebuffer (the image keeps the
// previous frame's pixels), else whether the alpha carries a mask.
enum { BLIT_NONE = -1, BLIT_OPAQUE = 0, BLIT_MASKED = 1 };
static int blit(const FbInfo &fb, int w, int ox, int oy, bool with_mask) {
    if (!fb.valid || !fb.addr_va) return BLIT_NONE;
    u32 va = fb.addr_va;
    if (!mem_is_mapped(va)) return BLIT_NONE;
    u32 bpp = fb_bpp(fb.format);
    u32 stride = fb.stride ? fb.stride : 240 * bpp;
    static std::vector<u8> mask;
    bool m = with_mask && cov_fb_mask(va, mask) && mask.size() >= (size_t)w * 240;
    u32 mw = m ? (u32)(mask.size() / w) : 0;
    for (int sx = 0; sx < w; sx++) {
        u32 row = va + sx * stride;
        for (int sy = 0; sy < 240; sy++) {
            u32 c = fb_decode(gp(row + (239 - sy) * bpp), fb.format) & 0x00FFFFFFu;
            u32 a = m ? mask[(size_t)sx * mw + (239 - sy)] : 255;
            g_screen[(oy + sy) * 400 + ox + sx] = c | (a << 24);
        }
    }
    return m ? BLIT_MASKED : BLIT_OPAQUE;
}

// R3DS_SHOT_OVERLAY=1: also write the "remaster" composite (bottom screen
// without backdrop over the top one) next to each snapshot
static void write_overlay_png(const std::string &path) {
    std::vector<u32> top(g_screen.begin(), g_screen.begin() + 400 * 240), save = g_screen;
    // bottom at 75 % scale, centered at the bottom edge
    const float sc = 0.75f; int bw = (int)(320 * sc), bh = (int)(240 * sc), bx0 = (400 - bw) / 2, by0 = 240 - bh;
    for (int y = 0; y < bh; y++) for (int x = 0; x < bw; x++) {
        u32 s2 = g_screen[(240 + (int)(y / sc)) * 400 + 40 + (int)(x / sc)];
        u32 a = s2 >> 24;
        u32 &d = top[(by0 + y) * 400 + bx0 + x];
        u32 r = ((s2 & 0xFF) * a + (d & 0xFF) * (255 - a)) / 255;
        u32 g = (((s2 >> 8) & 0xFF) * a + ((d >> 8) & 0xFF) * (255 - a)) / 255;
        u32 b = (((s2 >> 16) & 0xFF) * a + ((d >> 16) & 0xFF) * (255 - a)) / 255;
        d = r | (g << 8) | (b << 16) | 0xFF000000u;
    }
    std::fill(g_screen.begin(), g_screen.end(), 0xFF000000u);
    std::copy(top.begin(), top.end(), g_screen.begin());
    write_png(path);
    g_screen = save;
}

void display_init() {
    const char *d = getenv("R3DS_SHOTS");
    if (d && *d) { g_shot_dir = d; plat_mkdir(d); }
    const char *e = getenv("R3DS_SHOT_EVERY");
    g_shot_every = e ? atoi(e) : 60;
    // R3DS_INPUT="frame:buttonsHex[:dur][,..]" or "frame:T:x:y[:dur]" for touch
    const char *in = getenv("R3DS_INPUT");
    if (in) {
        std::string s = in;
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            std::string it = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
            ScriptEv ev{0, 0, 6, 0, 0, false, 0, 0, false};
            unsigned a = 0, b = 0, c = 0, dd = 6;
            int cx = 0, cy = 0;
            if (sscanf(it.c_str(), "%u:C:%d:%d:%u", &a, &cx, &cy, &dd) >= 3) { ev.frame = a; ev.circle = true; ev.cx = cx; ev.cy = cy; ev.dur = dd; }
            else if (sscanf(it.c_str(), "%u:T:%u:%u:%u", &a, &b, &c, &dd) >= 3) { ev.frame = a; ev.touch = true; ev.tx = b; ev.ty = c; ev.dur = dd; }
            else if (sscanf(it.c_str(), "%u:%x:%u", &a, &b, &dd) >= 2) { ev.frame = a; ev.buttons = b; ev.dur = dd; }
            g_script.push_back(ev);
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
}

static std::mutex g_frame_mtx;
static std::vector<u32> g_frame_out(400 * 480);
static bool g_frame_new = false;
static bool g_frame_masked = false;    // mask state of g_frame_out, published with it
static float g_frame_cover = 1.0f;
static HwrFrameInfo g_frame_info;      // ... and what the hardware renderer needs to match it
static std::deque<HwrFrameInfo> g_frame_hist;   // every one since the executor last looked
void display_frame_history(std::vector<HwrFrameInfo> &out) {
    std::lock_guard<std::mutex> lk(g_frame_mtx);
    out.assign(g_frame_hist.begin(), g_frame_hist.end());
    g_frame_hist.clear();
}
bool display_get_frame(std::vector<u32> &out) {
    std::lock_guard<std::mutex> lk(g_frame_mtx);
    if (!g_frame_new) return false;
    out = g_frame_out;
    g_frame_new = false;
    g_bottom_masked = g_frame_masked;   // what display_bottom_*() report: this frame's
    g_bottom_cover = g_frame_cover;
    return true;
}
bool display_get_frame_info(std::vector<u32> &out, HwrFrameInfo &info) {
    std::lock_guard<std::mutex> lk(g_frame_mtx);
    if (!g_frame_new) return false;
    out = g_frame_out;
    info = g_frame_info;
    g_frame_new = false;
    g_bottom_masked = g_frame_masked;
    g_bottom_cover = g_frame_cover;
    return true;
}

// LCD framebuffers seen at the last vblanks, per screen (the hardware
// recorder asks which screen a display transfer feeds)
static std::atomic<u32> g_lcd_va[2][2];
int display_screen_of(u32 va) {
    for (int s = 0; s < 2; s++)
        if (va && (g_lcd_va[s][0] == va || g_lcd_va[s][1] == va)) return s;
    return -1;
}
static void note_lcd(int s, u32 va) {
    if (!va || g_lcd_va[s][0] == va) return;
    g_lcd_va[s][1] = g_lcd_va[s][0].load();
    g_lcd_va[s][0] = va;
}
u64 hwr_hash(const u8 *p, size_t n) {
    u64 h = 0x9E3779B97F4A7C15ull ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { u64 w; memcpy(&w, p + i, 8); h = (h ^ w) * 0xFF51AFD7ED558CCDull; h ^= h >> 29; }
    for (; i < n; i++) { h = (h ^ p[i]) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}
static HwrScreen screen_info(const FbInfo &fb, u32 rows) {
    HwrScreen s;
    if (!fb.valid || !fb.addr_va || !mem_is_mapped(fb.addr_va)) return s;
    u32 stride = fb.stride ? fb.stride : 240 * fb_bpp(fb.format);
    s.va = fb.addr_va; s.bytes = stride * rows; s.valid = true;
    if (mem_is_mapped(fb.addr_va + s.bytes - 1)) s.hash = hwr_hash(gp(fb.addr_va), s.bytes);
    return s;
}

void display_present(const FbInfo &top, const FbInfo &bottom, u64 vblank_ns) {
    u32 f = ++g_frames;
    blit(top, 400, 0, 0, false);
    // no bottom framebuffer: g_screen keeps the last bottom pixels, and with
    // them their mask state (else a masked image would be shown as opaque)
    static bool masked = false;
    static float cover = 1.0f;
    int b = blit(bottom, 320, 40, 240, true);
    static bool blank = false;
    if (b != BLIT_NONE) {
        masked = b == BLIT_MASKED;
        blank = false;
        cover = masked ? bottom_postprocess(&blank) : 1.0f;
    }
    note_lcd(0, top.valid ? top.addr_va : 0);
    note_lcd(1, bottom.valid ? bottom.addr_va : 0);
    HwrFrameInfo info;
    if (g_hwr_on) {
        info.top = screen_info(top, 400);
        info.bot = b != BLIT_NONE ? screen_info(bottom, 320) : HwrScreen{};
    }
    info.masked = masked; info.blank = blank; info.cover = cover; info.frame = f;
    info.t_ns = vblank_ns ? vblank_ns : now_ns();
    {
        std::lock_guard<std::mutex> lk(g_frame_mtx);
        g_frame_out = g_screen;
        g_frame_masked = masked;
        g_frame_cover = cover;
        g_frame_info = info;
        g_frame_new = true;
        g_frame_hist.push_back(info);
        while (g_frame_hist.size() > 32) g_frame_hist.pop_front();
    }
    if (!g_shot_dir.empty() && g_shot_every && f % g_shot_every == 0) {
        char b[64]; snprintf(b, sizeof b, "/f%05u.png", f);
        write_png(g_shot_dir + b);
        static bool ov = getenv("R3DS_SHOT_OVERLAY") != nullptr;
        if (ov) { snprintf(b, sizeof b, "/f%05u_ov.png", f); write_overlay_png(g_shot_dir + b); }
    }
}
