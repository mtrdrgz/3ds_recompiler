// Hardware renderer, recorder side (GX thread). See hwr.h.
//
// Mirrors every write the PICA does to render targets as commands for the
// executor, and keeps a model of what the executor holds:
//  * color / depth surfaces, keyed by guest address and size. A surface is
//    "valid" while its GPU content mirrors guest memory; an invalid one is
//    reloaded from memory (1x, produced by the software renderer) before
//    the next draw into it.
//  * decoded textures uploaded from memory, keyed by the software texture
//    cache's decode id (re-uploaded when the software re-decodes).
//  * LCD images: the result of each display transfer into a framebuffer,
//    tagged with a hash of the guest framebuffer, so the display can tell
//    whether the image still matches what the software shows.
// Widescreen: a render target whose passes end up on the top screen gets a
// wider surface for those passes. Draws with a perspective projection that
// fill the viewport see more of the scene; everything else (2D, UI) is
// stretched. The game renders both screens through one render target, so
// the class of each pass (top or not) is predicted from the order observed
// in earlier frames.
#include "hwr.h"
#include "pica.h"
#include "pica_fmt.h"
#include "mem.h"
#include "kernel.h"
#include "hwr_fonts.h"
#include "pica_snap.h"
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#define R (g_pica->regs)
std::atomic<bool> g_hwr_on{false};
double g_hwr_dbg[HD_COUNT];
std::atomic<u32> g_hwr_req_scale{1};
std::atomic<u32> g_hwr_req_wide{0x3F800000u};
std::atomic<u64> g_hwr_alive_ns{0};
static std::atomic<bool> g_req_reset{false};
void hwr_request_reset() { g_req_reset = true; }

extern u32 g_pica_state_gen;
u32 display_frame_count();
int display_screen_of(u32 va);
void pica_flush();
void pica_flush_if_queued(u32 va, u32 size);
const u32 *pica_tex_decoded(u32 va, u32 w, u32 h, u32 fmt, u64 *id);
u32 pica_tex_bytes(u32 w, u32 h, u32 fmt);
bool pica_cov_get(u32 va, u32 w, u32 h, std::vector<u8> &out);

// ------------------------------------------------------------ queue
static std::mutex g_qm;
static std::deque<std::vector<u8>> g_queue;
static size_t g_queue_bytes = 0;
static std::condition_variable g_qcv;         // the executor took the queue
static std::vector<u8> g_cur;                 // being recorded
static bool g_active = false;                 // recording (executor alive)
bool hwr_take(std::vector<std::vector<u8>> &out) {
    g_hwr_alive_ns = now_ns();
    std::lock_guard<std::mutex> lk(g_qm);
    if (g_queue.empty()) return false;
    for (auto &c : g_queue) out.push_back(std::move(c));
    g_queue.clear(); g_queue_bytes = 0;
    g_qcv.notify_all();
    return true;
}

static inline void put(const void *p, size_t n) { const u8 *b = (const u8 *)p; g_cur.insert(g_cur.end(), b, b + n); }
static inline void put32(u32 v) { put(&v, 4); }
static size_t begin_cmd(u32 op) { size_t at = g_cur.size(); put32(op); put32(0); return at; }
static void end_cmd(size_t at) { u32 len = (u32)(g_cur.size() - at - 8); memcpy(&g_cur[at + 4], &len, 4); }
template <class T> static void cmd(u32 op, const T &payload) { size_t at = begin_cmd(op); put(&payload, sizeof payload); end_cmd(at); }
static void cmd_ids(u32 op, std::initializer_list<u32> v) { size_t at = begin_cmd(op); for (u32 x : v) put32(x); end_cmd(at); }

// ------------------------------------------------------------ model
struct Surf {
    u32 id, kind, va, lw, lh, fmt, bytes, pw, ph;
    bool wide;
    bool valid = false;
    bool hash_pending = true; u64 memhash = 0; u32 hash_frame = ~0u;
    u32 last_use = 0;
};
struct Rt {                                   // pass bookkeeping of a color render target
    u32 va, lw, lh, fmt;
    int cls = 0;                              // class of the current pass: 0 bottom / other, 1 top, 2 never shown
    bool closed = false;                      // transferred out since the last draw
    // classes of the last two passes and what followed each such pair (and
    // each single class): the game renders right eye, left eye, bottom
    int hist[2] = {-1, -1};
    int learned2[4][4], learned1[4];
    Rt() { for (auto &r : learned2) for (int &x : r) x = -1; for (int &x : learned1) x = -1; }
};
struct TexRec { u32 id = 0; u64 decode = 0, memhash = 0; u32 last_use = 0; };
struct TexKey { u32 va, w, h, fmt; bool operator==(const TexKey &o) const { return va == o.va && w == o.w && h == o.h && fmt == o.fmt; } };
struct TexKeyHash { size_t operator()(const TexKey &k) const { return k.va * 2654435761u ^ (k.w << 16) ^ k.h ^ (k.fmt << 28); } };

static std::vector<Surf> g_surfs;
static std::vector<Rt> g_rts;
static std::unordered_map<TexKey, TexRec, TexKeyHash> g_texs;
static std::unordered_map<u32, std::deque<u32>> g_lcds;   // framebuffer va -> recent LCD image ids
static u32 g_next_id = 1;
static u32 g_scale = 1;
static float g_wide = 1.0f;
static u32 g_frame = 0, g_gc_frame = 0;

static bool wide_on() { return g_wide > 1.001f; }
static u32 depth_bpp(u32 dfmt) { return dfmt == 0 ? 2 : dfmt == 3 ? 4 : 3; }

static void reset_model(bool emit) {
    g_surfs.clear(); g_rts.clear(); g_texs.clear(); g_lcds.clear();
    if (emit) cmd_ids(HOP_RESET, {g_scale, (u32)lroundf(g_wide * 65536.0f)});
}

static void phys_size(u32 lw, u32 lh, bool wide, u32 &pw, u32 &ph) {
    float k = wide ? g_wide : 1.0f;
    u32 s = g_scale;
    while (s > 1 && std::max(lw, lh) * s * k > 8192) s--;
    pw = lw * s; ph = lh * s;
    if (wide) { if (lh >= lw) ph = (u32)lroundf(lh * s * k); else pw = (u32)lroundf(lw * s * k); }
}

static Surf &new_surf(u32 kind, u32 va, u32 lw, u32 lh, u32 fmt, u32 bytes, bool wide, u32 pw = 0, u32 ph = 0) {
    Surf s{};
    s.id = g_next_id++; s.kind = kind; s.va = va; s.lw = lw; s.lh = lh; s.fmt = fmt; s.bytes = bytes; s.wide = wide;
    if (pw) { s.pw = pw; s.ph = ph; } else phys_size(lw, lh, wide, s.pw, s.ph);
    s.last_use = g_frame;
    cmd_ids(HOP_SURF_NEW, {s.id, kind, lw, lh, s.pw, s.ph});
    g_surfs.push_back(s);
    return g_surfs.back();
}
static void free_surf(size_t i) {
    cmd_ids(HOP_SURF_FREE, {g_surfs[i].id});
    g_surfs.erase(g_surfs.begin() + i);
}
static Surf *find_surf(u32 id) {
    for (auto &s : g_surfs) if (s.id == id) return &s;
    return nullptr;
}
static Surf *find_color(u32 va, u32 lw, u32 lh, u32 fmt, bool wide) {
    for (auto &s : g_surfs)
        if (s.kind == HSK_COLOR && s.va == va && s.lw == lw && s.lh == lh && s.fmt == fmt && s.wide == wide) return &s;
    return nullptr;
}
static Rt *find_rt(u32 va) {
    for (auto &r : g_rts) if (r.va == va) return &r;
    return nullptr;
}
static Rt &rt_state(u32 va, u32 lw, u32 lh, u32 fmt) {
    for (auto &r : g_rts)
        if (r.va == va) {
            if (r.lw != lw || r.lh != lh || r.fmt != fmt) { r = Rt{}; r.va = va; r.lw = lw; r.lh = lh; r.fmt = fmt; }
            return r;
        }
    Rt r; r.va = va; r.lw = lw; r.lh = lh; r.fmt = fmt;
    g_rts.push_back(r);
    return g_rts.back();
}
// the surface currently standing for render target rt
static Surf *rt_current(const Rt &rt) { return find_color(rt.va, rt.lw, rt.lh, rt.fmt, rt.cls == 1 && wide_on()); }
static inline int hidx(int c) { return c < 0 ? 3 : c; }
static int predict_class(const Rt &rt) {
    if (rt.hist[0] < 0) return rt.cls;
    int l = rt.learned2[hidx(rt.hist[1])][hidx(rt.hist[0])];
    if (l >= 0) return l;
    l = rt.learned1[hidx(rt.hist[0])];
    return l >= 0 ? l : rt.hist[0];
}
// a new pass starts in rt (cleared: by a full memory fill)
static void start_pass(Rt &rt, bool cleared) {
    rt.closed = false;
    int c = wide_on() ? predict_class(rt) : 0;
    if (c == rt.cls) return;
    rt.cls = c;
    if (!cleared) {   // the other class's surface does not hold the current content
        if (Surf *s = rt_current(rt)) { s->valid = false; g_hwr_dbg[HD_INVALID_RANGE]++; }
    }
}

// ---- loads from memory (the software renderer's 1x result)
static void load_color(Surf &s) {
    g_hwr_dbg[HD_LOAD_COLOR]++;
    pica_flush();
    u32 bpp = fb_bpp(s.fmt);
    std::vector<u8> cov;
    bool has_cov = pica_cov_get(s.va, s.lw, s.lh, cov);
    size_t at = begin_cmd(HOP_SURF_LOAD);
    put32(s.id); put32(s.lw); put32(s.lh); put32(has_cov);
    size_t px0 = g_cur.size();
    g_cur.resize(px0 + (size_t)s.lw * s.lh * 4);
    u32 *px = (u32 *)&g_cur[px0];
    bool ok = mem_is_mapped(s.va) && mem_is_mapped(s.va + s.bytes - 1);
    for (u32 y = 0; y < s.lh; y++)
        for (u32 x = 0; x < s.lw; x++)
            px[y * s.lw + x] = ok ? fb_decode(gp(s.va + tiled_offset(x, y, s.lw, bpp)), s.fmt) : 0;
    if (has_cov) put(cov.data(), (size_t)s.lw * s.lh);
    end_cmd(at);
    s.valid = true; s.hash_pending = true;
}
static void load_depth(Surf &s) {
    g_hwr_dbg[HD_LOAD_DEPTH]++;
    pica_flush();
    u32 bpp = depth_bpp(s.fmt);
    size_t at = begin_cmd(HOP_SURF_LOAD);
    put32(s.id); put32(s.lw); put32(s.lh); put32(0);
    size_t px0 = g_cur.size();
    g_cur.resize(px0 + (size_t)s.lw * s.lh * 4);
    u32 *px = (u32 *)&g_cur[px0];
    bool ok = mem_is_mapped(s.va) && mem_is_mapped(s.va + s.bytes - 1);
    for (u32 y = 0; y < s.lh; y++)
        for (u32 x = 0; x < s.lw; x++) {
            u32 v = 0;
            if (ok) {
                const u8 *p = gp(s.va + tiled_offset(x, y, s.lw, bpp));
                if (s.fmt == 0) v = (u32)(((u64)(p[0] | (p[1] << 8)) * 0xFFFFFF) / 0xFFFF);
                else { v = p[0] | (p[1] << 8) | (p[2] << 16); if (s.fmt == 3) v |= (u32)p[3] << 24; }
            }
            px[y * s.lw + x] = v;
        }
    end_cmd(at);
    s.valid = true; s.hash_pending = true;
}

// invalidate what overlaps a guest range the GPU side did not write
static void invalidate_range(u32 va, u32 size, u32 except_id = 0) {
    if (!size) return;
    for (auto &s : g_surfs)
        if (s.kind != HSK_LCD && s.id != except_id && va < s.va + s.bytes && s.va < va + size && s.valid) { s.valid = false; g_hwr_dbg[HD_INVALID_RANGE]++; }
}

// a rendered surface about to be sampled: still what memory holds?
static bool surface_current(Surf &s) {
    pica_flush_if_queued(s.va, s.bytes);
    if (!mem_is_mapped(s.va) || !mem_is_mapped(s.va + s.bytes - 1)) return false;
    if (s.hash_pending) {
        s.memhash = hwr_hash(gp(s.va), s.bytes);
        s.hash_pending = false; s.hash_frame = g_frame;
        return true;
    }
    if (s.hash_frame == g_frame) return true;
    s.hash_frame = g_frame;
    if (hwr_hash(gp(s.va), s.bytes) != s.memhash) { s.valid = false; g_hwr_dbg[HD_INVALID_HASH]++; return false; }
    return true;
}

// ------------------------------------------------------------ textures
static const u32 TEXFMT_TO_FB[5] = {0, 1, 3, 2, 4};
static u32 resolve_tex(u32 va, u32 w, u32 h, u32 fmt) {
    if (!va || !w || !h || w > 1024 || h > 1024) return 0;
    if (fmt <= 4) {   // rendered by the GPU side?
        Surf *best = nullptr;
        if (Rt *rt = find_rt(va)) {
            Surf *c = rt_current(*rt);
            if (c && c->lw == w && c->lh == h && c->fmt == TEXFMT_TO_FB[fmt]) best = c;
        }
        if (!best)
            for (auto &s : g_surfs)
                if (s.kind == HSK_COLOR && s.va == va && s.lw == w && s.lh == h && s.fmt == TEXFMT_TO_FB[fmt] && !find_rt(va)) { best = &s; break; }
        if (best && best->valid && surface_current(*best)) { best->last_use = g_frame; return best->id; }
    }
    u64 did = 0;
    const u32 *px = pica_tex_decoded(va, w, h, fmt, &did);
    if (!px) return 0;
    TexRec &t = g_texs[TexKey{va, w, h, fmt}];
    if (t.id && t.decode != did) {
        // re-decoded after a cache flush: upload again only if the texture changed
        u32 n = pica_tex_bytes(w, h, fmt);
        u64 mh = mem_is_mapped(va + n - 1) ? hwr_hash(gpu_ptr(va, n), n) : 0;
        if (mh == t.memhash) { t.decode = did; g_hwr_dbg[HD_TEX_SAME]++; }
    }
    if (!t.id || t.decode != did) {
        if (t.id) cmd_ids(HOP_TEX_FREE, {t.id});
        t.id = g_next_id++; t.decode = did;
        g_hwr_dbg[HD_TEX_UP]++; g_hwr_dbg[HD_TEX_BYTES] += (double)w * h * 4;
        u32 n = pica_tex_bytes(w, h, fmt);
        t.memhash = mem_is_mapped(va + n - 1) ? hwr_hash(gpu_ptr(va, n), n) : 0;
        // a glyph sheet of one of the game's fonts: the executor redraws it from a vector font
        HwrSheet sheet;
        u32 ts = std::min<u32>(std::min(g_scale, 4u), 4096 / std::max(w, h));
        static bool vector_text = !getenv("R3DS_HWR_BITMAP_TEXT");
        bool font = vector_text && ts >= 2 && hwr_font_detect(px, w, h, fmt, sheet);
        // the texture's identity for replacements (tools/texlab.py): the hash of
        // its decoded pixels, which does not depend on where the game put it
        const u64 ph = hwr_hash((const u8 *)px, (size_t)w * h * 4);
        {   // R3DS_HWR_TEXDUMP=dir: every texture once, named by that hash (textures the
            // game builds at run time; tools/tex_extract.py gets the ROM's without playing)
            static const char *dir = getenv("R3DS_HWR_TEXDUMP");
            static std::unordered_set<u64> dumped;
            if (dir && !font && w * h >= 16 && dumped.insert(ph).second) {
                char path[512];
                snprintf(path, sizeof path, "%s/%016llx_%ux%u_f%u.rgba", dir, (unsigned long long)ph, w, h, fmt);
                if (FILE *f = fopen(path, "wb")) {   // top row first (an ordinary image)
                    for (u32 row = 0; row < h; row++) fwrite(px + (size_t)(h - 1 - row) * w, 4, w, f);
                    fclose(f);
                }
            }
        }
        size_t at = begin_cmd(font ? HOP_TEX_FONT : HOP_TEX);
        put32(t.id); put32(w); put32(h);
        if (!font) { put32((u32)ph); put32((u32)(ph >> 32)); }
        if (font) {
            put32(ts); put32((u32)(sheet.font - g_hwr_fonts)); put32(sheet.cw); put32(sheet.ch); put32(sheet.cols); put32(sheet.cells);
            static int logged = 0;
            if (logged++ < 8) LOG("[hwr] glyph sheet %08x %ux%u: %ux%u font, %u glyphs, redrawn at %ux", va, w, h, sheet.font->w, sheet.font->h, sheet.cells, ts);
        }
        for (u32 row = 0; row < h; row++) put(px + (size_t)(h - 1 - row) * w, w * 4);   // memory row order
        end_cmd(at);
    }
    t.last_use = g_frame;
    return t.id;
}

// ------------------------------------------------------------ draws
static size_t g_draw_at = SIZE_MAX;   // open HOP_DRAW record in g_cur
static u32 g_draw_gen = 0, g_skip_gen = 0;
static bool g_draw_persp = false, g_draw_squeezable = false;
static Surf *g_draw_target = nullptr;

static inline u32 fbits(float f) { u32 u; memcpy(&u, &f, 4); return u; }

static bool open_draw() {
    u32 color_pa = R[0x11D] << 3, depth_pa = R[0x11C] << 3;
    u32 fbw = R[0x11E] & 0x7FF, fbh = ((R[0x11E] >> 12) & 0x3FF) + 1;
    u32 cva = pa_to_va(color_pa), dva = pa_to_va(depth_pa);
    if (!cva || !fbw || !pa_ptr(color_pa)) return false;
    u32 cfmt = (R[0x117] >> 16) & 7, dfmt = R[0x116] & 3;
    if (cfmt > 4) return false;

    Rt &rt = rt_state(cva, fbw, fbh, cfmt);
    if (rt.closed) start_pass(rt, false);
    if (rt.cls == 2) {   // a pass nobody will see (the right eye): leave it to the software renderer
        g_hwr_dbg[HD_SKIP_PASS]++;
        for (auto &s : g_surfs)
            if ((s.kind == HSK_COLOR && s.va == cva) || (s.kind == HSK_DEPTH && s.va == dva)) s.valid = false;
        return false;
    }
    bool wide = rt.cls == 1 && wide_on();
    Surf *cs = find_color(cva, fbw, fbh, cfmt, wide);
    if (!cs) cs = &new_surf(HSK_COLOR, cva, fbw, fbh, cfmt, fbw * fbh * fb_bpp(cfmt), wide);
    u32 color_id = cs->id;
    if (!cs->valid) load_color(*cs);
    u32 cpw = cs->pw, cph = cs->ph;
    u32 depth_id = 0;
    if (dva && pa_ptr(depth_pa)) {
        Surf *ds = nullptr;
        for (auto &s : g_surfs)
            if (s.kind == HSK_DEPTH && s.va == dva && s.lw == fbw && s.lh == fbh && s.fmt == dfmt && s.pw == cpw && s.ph == cph) { ds = &s; break; }
        if (!ds) ds = &new_surf(HSK_DEPTH, dva, fbw, fbh, dfmt, fbw * fbh * depth_bpp(dfmt), false, cpw, cph);
        if (!ds->valid) load_depth(*ds);
        ds->last_use = g_frame; ds->hash_pending = true;
        depth_id = ds->id;
    }

    HwrDraw d{};
    // textures first: their uploads must precede the draw record
    u32 tcfg = R[0x80];
    const u32 tbase[3] = {0x81, 0x91, 0x99};
    u32 flags = 0;
    for (int i = 0; i < 3; i++) {
        u32 b = tbase[i];
        if (!((tcfg >> i) & 1)) continue;
        u32 h = R[b + 1] & 0x7FF, w = (R[b + 1] >> 16) & 0x7FF;
        u32 va = pa_to_va(R[b + 4] << 3);
        u32 fmt = (i == 0 ? R[0x8E] : R[b + 5]) & 0xF;
        d.tex_id[i] = resolve_tex(va, w, h, fmt);
        if (d.tex_id[i]) flags |= HUF_TEX0 << i;
    }
    // resolving textures may have flushed / invalidated; the targets stay valid
    cs = find_surf(color_id);
    if (!cs) return false;
    cs->last_use = g_frame; cs->hash_pending = true;
    g_draw_target = cs;
    d.color_id = color_id; d.depth_id = depth_id;

    // viewport -> GPU clip transform, scissor
    float hw = f24(R[0x41]), hh = f24(R[0x43]);
    float vx = (float)(s32)((R[0x68] & 0x3FF) << 22 >> 22);
    float vy = (float)(s32)(((R[0x68] >> 16) & 0x3FF) << 22 >> 22);
    float Ax = 2 * hw / fbw, Bx = 2 * (hw + vx) / fbw - 1, Ay = 2 * hh / fbh, By = 2 * (hh + vy) / fbh - 1;
    float x0 = vx, x1 = vx + 2 * hw, y0 = vy, y1 = vy + 2 * hh;
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    g_draw_squeezable = false;
    if (wide) {   // only a viewport spanning the whole widened axis can show more
        if (fbh >= fbw) g_draw_squeezable = y0 <= 0.5f && y1 >= fbh - 0.5f;
        else g_draw_squeezable = x0 <= 0.5f && x1 >= fbw - 0.5f;
    }
    x0 = std::max(x0, 0.0f); y0 = std::max(y0, 0.0f); x1 = std::min(x1, (float)fbw); y1 = std::min(y1, (float)fbh);
    u32 smode = R[0x65] & 3;
    u32 ex1 = R[0x66] & 0x3FF, ey1 = (R[0x66] >> 16) & 0x3FF, ex2 = R[0x67] & 0x3FF, ey2 = (R[0x67] >> 16) & 0x3FF;
    if (smode == 3) {
        x0 = std::max(x0, (float)ex1); y0 = std::max(y0, (float)ey1);
        x1 = std::min(x1, (float)ex2 + 1); y1 = std::min(y1, (float)ey2 + 1);
    }
    if (x0 >= x1 || y0 >= y1) { g_draw_target = nullptr; return false; }
    d.scissor[0] = fbits(x0); d.scissor[1] = fbits(fbh - y1); d.scissor[2] = fbits(x1); d.scissor[3] = fbits(fbh - y0);

    // fixed-function state
    u32 colop = R[0x100], blend = R[0x101], lop = R[0x102] & 0xF;
    u32 at = R[0x104], st = R[0x105], sop = R[0x106], dcm = R[0x107];
    bool can_cw = R[0x113] & 0xF, can_dw = (R[0x115] & 2) != 0, can_dr = (R[0x114] & 2) != 0, can_sw = (R[0x115] & 1) != 0;
    bool noalpha = cfmt == 1 || cfmt == 2;
    u32 wmask = can_cw ? (dcm >> 8) & 0xF : 0;
    if (noalpha) wmask &= 7;
    bool blend_on = colop & 0x100;
    u32 lclass = 0;
    if (!blend_on) {
        switch (lop) {
        case 3: lclass = 0; break;   // copy
        case 0: lclass = 1; break;   // clear
        case 4: lclass = 2; break;   // set
        case 5: lclass = 3; break;   // copy inverted
        case 6: lclass = 4; wmask = 0; break;   // no-op
        case 7: lclass = 5; break;   // invert
        default: { static u32 seen = 0; if (!(seen & (1u << lop))) { seen |= 1u << lop; LOG("[hwr] logic op %u drawn as copy", lop); } }
        }
    }
    u32 cull = R[0x40] & 3;
    if (hw * hh < 0 && cull) cull ^= 3;
    d.pipe.blend = blend_on ? blend : 0;
    d.pipe.misc = (blend_on ? 1 : 0) | (lclass << 1) | (wmask << 8) | (cull << 14) | (can_cw ? 1u << 16 : 0);
    bool dtest = (dcm & 1) && can_dr && depth_id;
    bool dwrite = ((dcm >> 12) & 1) && can_dw && depth_id;
    d.pipe.depth = (dtest ? 1 : 0) | (((dcm >> 4) & 7) << 1) | (dwrite ? 16 : 0);
    bool st_en = (st & 1) && dfmt == 3 && depth_id;
    if (st_en)
        d.pipe.stencil = 1 | (((st >> 4) & 7) << 1) | ((sop & 7) << 4) | (((sop >> 4) & 7) << 7) | (((sop >> 8) & 7) << 10) |
                         ((st >> 24) << 16) | ((can_sw ? (st >> 8) & 0xFF : 0) << 24);
    d.stencil_ref = (st >> 16) & 0xFF;
    d.blend_const = R[0x103];

    // uniforms
    u32 *u = d.u;
    u[0] = fbits(Ax); u[1] = fbits(Bx); u[2] = fbits(Ay); u[3] = fbits(By);
    u[4] = fbits(f24(R[0x4D])); u[5] = fbits(f24(R[0x4E]));
    u[6] = fbits(wide ? 1.0f / g_wide : 1.0f); u[7] = wide ? (fbh >= fbw ? 2 : 1) : 0;
    if (!(R[0x6D] & 1)) flags |= HUF_WBUF;
    if (R[0x8F] & 1) flags |= HUF_LIGHT;
    if ((tcfg >> 13) & 1) flags |= HUF_TEX2_TC1;
    if (at & 1) flags |= HUF_ALPHA_TEST;
    if (smode == 1) flags |= HUF_EXCLUDE;
    if (noalpha) flags |= HUF_NOALPHA;
    u[8] = flags; u[9] = R[0xFD]; u[10] = ((at >> 4) & 7) | (((at >> 8) & 0xFF) << 8); u[11] = R[0x1C0];
    u[12] = (R[0x1C2] & 7) + 1; u[14] = ex1 | (ey1 << 16); u[15] = (ex2 + 1) | ((ey2 + 1) << 16);
    const u32 tev_base[6] = {0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8};
    u32 upd = R[0xE0], ntev = 0;
    for (int s = 0; s < 6; s++) {
        u32 b = tev_base[s];
        u32 cmb = R[b + 2] & 0x000F000F, sc = R[b + 4];
        cmb |= (sc & 3) << 8 | ((sc >> 16) & 3) << 24;
        if (s < 4) cmb |= ((upd >> (8 + s)) & 1) << 4 | ((upd >> (12 + s)) & 1) << 5;
        u[16 + s * 4 + 0] = R[b]; u[16 + s * 4 + 1] = R[b + 1]; u[16 + s * 4 + 2] = cmb; u[16 + s * 4 + 3] = R[b + 3];
        // stages after the last non-passthrough one change nothing (as pica_fast.cpp's ntev)
        bool pass = (cmb & 0x0F0F0F3F) == 0 && (R[b] & 0xF) == 15 && ((R[b] >> 16) & 0xF) == 15 && (R[b + 1] & 0xF) == 0 && ((R[b + 1] >> 12) & 7) == 0;
        if (!pass) ntev = s + 1;
    }
    u[13] = lclass | (ntev << 16);
    for (int i = 0; i < 3; i++) { u[40 + i * 4] = R[tbase[i] + 2]; u[41 + i * 4] = R[tbase[i]]; }
    u[52] = fbits((float)fbw); u[53] = fbits((float)fbh); u[54] = fbits((float)cpw); u[55] = fbits((float)cph);
    u32 nl = (R[0x1C2] & 7) + 1;
    for (u32 i = 0; i < 8; i++) {
        if (i >= nl) break;
        u32 id = (R[0x1D9] >> (4 * i)) & 7, lb = 0x140 + id * 0x10;
        u32 *L = &u[56 + i * 8];
        L[0] = R[lb + 0]; L[1] = R[lb + 2]; L[2] = R[lb + 3]; L[3] = R[lb + 9] & 1;
        L[4] = fbits(f16(R[lb + 4] & 0xFFFF)); L[5] = fbits(f16(R[lb + 4] >> 16)); L[6] = fbits(f16(R[lb + 5] & 0xFFFF)); L[7] = 0;
    }

    g_draw_at = begin_cmd(HOP_DRAW);
    put(&d, sizeof d);
    g_draw_gen = g_pica_state_gen;
    g_draw_persp = false;
    return true;
}

// R3DS_HWR_DRAWLOG=frame: log every draw of that display frame with its texture,
// texture-coordinate bounds and the rectangle it covers on the render target
static void drawlog(const HwrDraw *d) {
    static int want = getenv("R3DS_HWR_DRAWLOG") ? atoi(getenv("R3DS_HWR_DRAWLOG")) : -1;
    if (want < 0 || (int)display_frame_count() != want) return;
    const float *v = (const float *)(d + 1);
    float u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9, x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    float Ax, Bx, Ay, By; memcpy(&Ax, &d->u[0], 4); memcpy(&Bx, &d->u[1], 4); memcpy(&Ay, &d->u[2], 4); memcpy(&By, &d->u[3], 4);
    float lw, lh; memcpy(&lw, &d->u[52], 4); memcpy(&lh, &d->u[53], 4);
    for (u32 i = 0; i < d->nverts; i++, v += HWR_VFLOATS) {
        u0 = std::min(u0, v[8]); u1 = std::max(u1, v[8]); v0 = std::min(v0, v[9]); v1 = std::max(v1, v[9]);
        float w = v[3] ? v[3] : 1, nx = (Ax * v[0] + Bx * w) / w, ny = (Ay * v[1] + By * w) / w;
        float px = (nx + 1) * 0.5f * lw, py = (1 - (ny + 1) * 0.5f) * lh;   // target pixels, y down
        x0 = std::min(x0, px); x1 = std::max(x1, px); y0 = std::min(y0, py); y1 = std::max(y1, py);
    }
    u32 tcfg = R[0x80];
    LOG("[drawlog] rt %08x tex0 %08x %ux%u fmt %u | uv %.4f-%.4f %.4f-%.4f | rect %.0f-%.0f x %.0f-%.0f | verts %u",
        R[0x11D] << 3, (tcfg & 1) ? R[0x85] << 3 : 0, (R[0x82] >> 16) & 0x7FF, R[0x82] & 0x7FF, R[0x8E] & 0xF,
        u0, u1, v0, v1, x0, x1, y0, y1, d->nverts);
}

static void close_draw(int cls) {
    if (g_draw_at == SIZE_MAX) return;
    HwrDraw *d = (HwrDraw *)&g_cur[g_draw_at + 8];
    if (!d->nverts) { g_cur.resize(g_draw_at); g_draw_at = SIZE_MAX; return; }
    drawlog(d);
    g_hwr_dbg[HD_DRAWS]++; g_hwr_dbg[HD_TRIS] += d->nverts / 3;
    bool blend_on = d->pipe.misc & 1, can_cw = (d->pipe.misc >> 16) & 1;
    u32 cov = 0;   // 0 keep, 1 max(alpha), 2 set, 3 clear -- as pica_fast.cpp's shade_rows
    if (can_cw) cov = cls == 1 ? (blend_on ? 0 : 3) : (blend_on ? 1 : 2);
    d->pipe.misc = (d->pipe.misc & ~(3u << 12)) | (cov << 12);
    d->u[13] = (d->u[13] & ~0xFF00u) | (cov << 8);
    if (g_draw_persp && g_draw_squeezable) d->u[8] |= HUF_SQUEEZE;
    end_cmd(g_draw_at);
    g_draw_at = SIZE_MAX;
    g_draw_target = nullptr;
}

// Before a draw's vertices are processed: true when it renders into a pass
// nobody sees (predicted to be the 3D right eye, whose framebuffer is never
// shown). The whole draw is skipped: no vertex shading, no software
// rasterization, which in stereo scenes is ~40% of the GPU thread's draws.
// The next pass into the shared render target starts with a clear, so the
// missing right-eye pixels are never read. R3DS_HWR_SKIP_HIDDEN=0: draw them in
// software as before.
bool hwr_skip_draw() {
    static int on = getenv("R3DS_HWR_SKIP_HIDDEN") ? atoi(getenv("R3DS_HWR_SKIP_HIDDEN")) : 1;
    if (!g_active || !on) return false;
    u32 color_pa = R[0x11D] << 3, depth_pa = R[0x11C] << 3;
    u32 fbw = R[0x11E] & 0x7FF, fbh = ((R[0x11E] >> 12) & 0x3FF) + 1;
    u32 cva = pa_to_va(color_pa), dva = pa_to_va(depth_pa);
    if (!cva || !fbw || !pa_ptr(color_pa)) return false;
    u32 cfmt = (R[0x117] >> 16) & 7;
    if (cfmt > 4) return false;
    Rt &rt = rt_state(cva, fbw, fbh, cfmt);
    if (rt.closed) start_pass(rt, false);
    if (rt.cls != 2) return false;
    g_hwr_dbg[HD_SKIP_PASS]++;
    for (auto &sf : g_surfs)
        if ((sf.kind == HSK_COLOR && sf.va == cva) || (sf.kind == HSK_DEPTH && sf.va == dva)) sf.valid = false;
    return true;
}

void hwr_draw_begin() {
    if (!g_active) return;
    close_draw(2);
}
void hwr_draw_end(int cls) {
    if (!g_active) return;
    close_draw(cls);
}

static inline void put_vertex(const OutVertex &v) {
    float f[HWR_VFLOATS] = {v.pos.x, v.pos.y, v.pos.z, v.pos.w, v.color.x, v.color.y, v.color.z, v.color.w,
                            v.tc0[0], v.tc0[1], v.tc1[0], v.tc1[1], v.tc2[0], v.tc2[1], v.view[0], v.view[1],
                            v.quat.x, v.quat.y, v.quat.z, v.quat.w, v.view[2]};
    put(f, sizeof f);
}

void hwr_triangle(const OutVertex &a, const OutVertex &b, const OutVertex &c) {
    if (!g_active) return;
    if (g_draw_at == SIZE_MAX || g_draw_gen != g_pica_state_gen) {
        if (g_draw_at != SIZE_MAX) close_draw(2);
        if (g_skip_gen == g_pica_state_gen) return;
        if (!open_draw()) { g_skip_gen = g_pica_state_gen; return; }
    }
    put_vertex(a); put_vertex(b); put_vertex(c);
    if (!g_draw_persp && (a.pos.w != 1.0f || b.pos.w != 1.0f || c.pos.w != 1.0f)) g_draw_persp = true;
    ((HwrDraw *)&g_cur[g_draw_at + 8])->nverts += 3;
}

// ------------------------------------------------------------ transfers
void hwr_fill(u32 start, u32 end, u32 value, u32 width) {
    if (!g_active || end <= start) return;
    close_draw(2);
    u8 unit = width == 0 ? 2 : width == 1 ? 3 : 4, pat[8];
    for (int i = 0; i < 8; i++) pat[i] = (u8)(value >> (8 * (i % unit)));
    // render targets cleared as a whole start a new pass
    for (auto &rt : g_rts)
        if (start <= rt.va && end >= rt.va + rt.lw * rt.lh * fb_bpp(rt.fmt)) start_pass(rt, true);
    for (size_t i = 0; i < g_surfs.size(); i++) {
        Surf &s = g_surfs[i];
        if (s.kind == HSK_LCD || !(start < s.va + s.bytes && s.va < end)) continue;
        bool full = start <= s.va && end >= s.va + s.bytes;
        Rt *rt = s.kind == HSK_COLOR ? find_rt(s.va) : nullptr;
        bool current = !rt || rt_current(*rt) == &s;
        if (!full || !current) { if (s.valid) g_hwr_dbg[HD_INVALID_RANGE]++; s.valid = false; continue; }
        u32 v;
        if (s.kind == HSK_COLOR) v = fb_decode(pat, s.fmt);
        else if (s.fmt == 0) v = (u32)(((u64)(pat[0] | (pat[1] << 8)) * 0xFFFFFF) / 0xFFFF);
        else v = pat[0] | (pat[1] << 8) | (pat[2] << 16) | (s.fmt == 3 ? (u32)pat[3] << 24 : 0);
        cmd_ids(HOP_SURF_CLEAR, {s.id, v, 1});
        s.valid = true; s.hash_pending = true;
    }
}

void hwr_display_transfer(u32 in_va, u32 out_va, u32 in_w, u32 in_h, u32 out_w, u32 out_h,
                          u32 in_fmt, u32 out_fmt, bool flip, bool to_tiled, u32 hs, u32 vs) {
    if (!g_active) return;
    close_draw(2);
    u32 out_bytes = out_w * out_h * fb_bpp(out_fmt);
    invalidate_range(out_va, out_bytes);
    if (to_tiled) return;
    Rt *rt = find_rt(in_va);
    if (!rt) return;
    Surf *src = rt_current(*rt);
    // observe which screen this pass fed, to predict the next pass's class
    // top screen: its framebuffer, or any 400-row image (the right eye's
    // framebuffer is not the one displayed)
    int scr = display_screen_of(out_va);
    int obs = scr == 1 ? 0 : scr == 0 ? 1 : out_h >= 400 ? 2 : 0;
    static int dbg = getenv("R3DS_HWR_DBG") ? 40 : 0;
    if (dbg > 0) { dbg--; LOG("[hwr] xfer rt %08x -> %08x screen %d pass class %d wide %.3f surf %s", in_va, out_va, scr, rt->cls, g_wide, src ? (src->wide ? "wide" : "normal") : "none"); }
    if (rt->hist[0] >= 0) {
        rt->learned1[hidx(rt->hist[0])] = obs;
        rt->learned2[hidx(rt->hist[1])][hidx(rt->hist[0])] = obs;
    }
    rt->hist[1] = rt->hist[0]; rt->hist[0] = obs;
    rt->closed = true;
    // the transfer reads (out_w << hs) x (out_h << vs) of the input
    u32 rw = std::min(in_w, out_w << hs), rh = std::min(in_h, out_h << vs);
    if (!src || !src->valid || src->lw != in_w || src->lh < rh || src->fmt != in_fmt) return;
    float rx = (float)src->pw / src->lw, ry = (float)src->ph / src->lh;
    u32 pw = std::max(1u, (u32)lroundf(rw * rx) >> hs), ph = std::max(1u, (u32)lroundf(rh * ry) >> vs);
    Surf &img = new_surf(HSK_LCD, out_va, out_w, out_h, 0, out_bytes, false, pw, ph);
    u32 img_id = img.id;
    HwrBlit b{src->id, img_id, 0, 0, rw, rh, flip ? 1u : 0u, 1, 0};
    cmd(HOP_BLIT, b);
    HwrLcd l{img_id, out_va, out_bytes, mem_is_mapped(out_va + out_bytes - 1) ? hwr_hash(gp(out_va), out_bytes) : 0};
    cmd(HOP_LCD, l);
    auto &ring = g_lcds[out_va];
    ring.push_back(img_id);
    while (ring.size() > 3) {
        u32 old = ring.front(); ring.pop_front();
        for (size_t i = 0; i < g_surfs.size(); i++) if (g_surfs[i].id == old) { free_surf(i); break; }
    }
}

void hwr_texture_copy(u32 in_va, u32 out_va, u32 size, u32 in_wg, u32 out_wg) {
    if (!g_active) return;
    close_draw(2);
    u32 iw = (in_wg & 0xFFFF) * 16, ig = (in_wg >> 16) * 16, ow = (out_wg & 0xFFFF) * 16, og = (out_wg >> 16) * 16;
    u32 out_extent = size + (ow ? (size / ow) * og : 0);
    Surf *src = nullptr;
    for (auto &s : g_surfs)
        if (s.kind == HSK_COLOR && s.va == in_va && s.bytes == size && s.valid) {
            Rt *rt = find_rt(in_va);
            if (!rt || rt_current(*rt) == &s) { src = &s; break; }
        }
    bool contiguous = (!iw || !ig) && (!ow || !og);
    if (!src || !contiguous) { invalidate_range(out_va, out_extent); return; }
    u32 lw = src->lw, lh = src->lh, fmt = src->fmt, sid = src->id;
    Surf *dst = find_color(out_va, lw, lh, fmt, false);
    if (Rt *rt = find_rt(out_va)) { if (rt->lw == lw && rt->lh == lh && rt->fmt == fmt) dst = rt_current(*rt); }
    if (!dst) dst = &new_surf(HSK_COLOR, out_va, lw, lh, fmt, size, false);
    u32 did = dst->id;
    invalidate_range(out_va, out_extent, did);
    dst = find_surf(did);
    HwrBlit b{sid, did, 0, 0, lw, lh, 0, 1, (fmt == 1 || fmt == 2) ? 1u : 0u};
    cmd(HOP_BLIT, b);
    dst->valid = true; dst->hash_pending = true; dst->last_use = g_frame;
}

void hwr_mem_written(u32 va, u32 size) {
    if (!g_active) return;
    close_draw(2);
    invalidate_range(va, size);
}

// ------------------------------------------------------------ publish
static void gc() {
    for (auto it = g_texs.begin(); it != g_texs.end();) {
        if (g_frame - it->second.last_use > 600) { cmd_ids(HOP_TEX_FREE, {it->second.id}); it = g_texs.erase(it); }
        else ++it;
    }
    for (size_t i = 0; i < g_surfs.size();) {
        Surf &s = g_surfs[i];
        bool lcd = s.kind == HSK_LCD;
        if (!lcd && g_frame - s.last_use > 1200) free_surf(i); else i++;
    }
}

void hwr_publish() {
    bool on = g_hwr_on.load(std::memory_order_relaxed);
    bool alive = on && now_ns() - g_hwr_alive_ns.load(std::memory_order_relaxed) < 4000000000ull;
    if (!alive) {
        if (g_active) { g_active = false; g_cur.clear(); g_draw_at = SIZE_MAX; reset_model(false); }
        return;
    }
    close_draw(2);
    u32 want_scale = std::max(1u, std::min(g_hwr_req_scale.load(), 8u));
    float want_wide; { u32 b = g_hwr_req_wide.load(); memcpy(&want_wide, &b, 4); }
    want_wide = std::clamp(want_wide, 1.0f, 2.0f);
    if (!g_active || want_scale != g_scale || fabsf(want_wide - g_wide) > 1e-4f || g_req_reset.exchange(false)) {
        g_scale = want_scale; g_wide = want_wide;
        g_cur.clear(); g_draw_at = SIZE_MAX;
        reset_model(true);
        g_active = true;
        g_hwr_dbg[HD_RESETS]++;
        std::lock_guard<std::mutex> lk(g_qm);
        g_queue.clear(); g_queue_bytes = 0;
    }
    g_frame = display_frame_count();
    if (g_frame - g_gc_frame > 120) { g_gc_frame = g_frame; gc(); }
    if (g_cur.empty()) return;
    std::unique_lock<std::mutex> lk(g_qm);
    // The executor (the browser's main thread on the web) is behind: hold
    // the GPU thread, and through the GX throttle the game, until it catches
    // up. Starting over instead reloads every surface at 1x, which shows as
    // a low-resolution flicker. A hidden tab stops the executor altogether;
    // the wait gives up after a while and the alive check takes over.
    if (g_queue_bytes > (32u << 20)) {
        const u64 t0 = now_ns();
        g_qcv.wait_for(lk, std::chrono::milliseconds(1500), [] { return g_queue_bytes <= (8u << 20); });
        g_hwr_dbg[HD_REC_WAIT_MS] += (now_ns() - t0) / 1e6;
    }
    if (g_queue_bytes > (512u << 20)) {   // the executor stalled: start over
        g_queue.clear(); g_queue_bytes = 0; g_cur.clear();
        reset_model(true);
        g_hwr_dbg[HD_RESETS]++;
    }
    g_queue_bytes += g_cur.size();
    g_hwr_dbg[HD_QUEUE_BYTES] = (double)g_queue_bytes;
    g_queue.push_back(std::move(g_cur));
    g_cur.clear();
}
