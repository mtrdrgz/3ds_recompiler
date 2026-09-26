// Optimized PICA200 rasterizer: per-draw decoded state, incremental edge
// functions and attribute planes, decoded-texture cache, row bands split
// across worker threads for large triangles.
#include "pica.h"
#include "pica_fmt.h"
#include "mem.h"
#include "pica_snap.h"
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <set>
#ifdef __EMSCRIPTEN__
#define CSR_GET() 0u
#define CSR_SET(x) (void)(x)
#elif defined(__aarch64__)
// FPCR carries the rounding mode / flush-to-zero here (see host_fpscr)
static inline u64 fpcr_get() { u64 v; __asm__ volatile("mrs %0, fpcr" : "=r"(v)); return v; }
#define CSR_GET() fpcr_get()
#define CSR_SET(x) __asm__ volatile("msr fpcr, %0" : : "r"((u64)(x)))
#else
#include <xmmintrin.h>
#define CSR_GET() _mm_getcsr()
#define CSR_SET(x) _mm_setcsr(x)
#endif

#define R (g_pica->regs)
u32 pica_tex_texel(u32 va, u32 w, u32 h, u32 fmt, u32 s, u32 t);   // pica_raster.cpp
void pica_lighting(const OutVertex &v, const float *view, const float *normal_q, u32 *prim, u32 *sec);
extern u64 g_frag_count;

// ------------------------------------------------------------ texture cache
struct TexKey { u32 va, w, h, fmt; bool operator==(const TexKey &o) const { return va == o.va && w == o.w && h == o.h && fmt == o.fmt; } };
struct TexKeyHash { size_t operator()(const TexKey &k) const { return k.va * 2654435761u ^ (k.w << 16) ^ k.h ^ (k.fmt << 28); } };
struct TexEntry { std::vector<u32> px; u64 id; };   // id: unique per decode (hwr_rec.cpp re-uploads on change)
static std::unordered_map<TexKey, TexEntry, TexKeyHash> g_texcache;
static u64 g_tex_decode_id = 0;
static std::mutex g_texcache_mtx;
// decodes dropped while triangles that sample them may still be queued:
// freed only after the queue has been drained (pica_flush)
static std::vector<std::vector<u32>> g_tex_graveyard;
static u32 g_state_built = 0xFFFFFFFF;   // g_pica_state_gen the cached DrawState was built for

// ------------------------------------------------------ screen coverage
// For render targets that end up on an LCD (learned from display
// transfers) a per-pixel coverage mask records which pixels hold UI /
// content as opposed to the screen's backdrop. A draw is "backdrop" when it
// covers (nearly) its whole viewport: background quads, patterns,
// gradients, dimming layers. An opaque backdrop clears the mask where it
// paints and a translucent one leaves it alone; every other draw is content
// and marks its pixels with its alpha. So the mask is right whatever the
// draw order. The mask travels with the display
// transfer so the frontend can show the bottom screen without its
// background, over the top screen (see docs/RECOMP2.md, "Remaster layout").
struct Coverage { u32 w = 0, h = 0; std::vector<u8> m; };
static std::unordered_map<u32, Coverage> g_cov;          // render target VA -> coverage
static std::unordered_map<u32, std::vector<u8>> g_fbmask; // LCD framebuffer VA -> mask (linear, as the fb)
static std::mutex g_cov_mtx;
static Coverage *cov_find(u32 va, u32 w, u32 h) {   // GPU thread
    auto it = g_cov.find(va);
    if (it == g_cov.end()) return nullptr;
    Coverage &c = it->second;
    if (c.w != w || c.h != h) { c.w = w; c.h = h; c.m.assign((size_t)w * h, 0); }
    return &c;
}

static u32 tex_bytes(u32 w, u32 h, u32 fmt) {
    static const u32 bpp4[16] = {32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8, 0, 0};
    return w * h * bpp4[fmt & 15] / 8;
}

void pica_flush();
void pica_texcache_invalidate(u32 va, u32 size) {
    pica_flush();
    std::lock_guard<std::mutex> lk(g_texcache_mtx);
    bool dropped = false;
    for (auto it = g_texcache.begin(); it != g_texcache.end();) {
        u32 a = it->first.va, e = a + tex_bytes(it->first.w, it->first.h, it->first.fmt);
        if (size == 0xFFFFFFFFu || (e > va && a < va + size)) {
            g_tex_graveyard.push_back(std::move(it->second.px));
            it = g_texcache.erase(it);
            dropped = true;
        } else ++it;
    }
    // the cached DrawState may point at a dropped decode: rebuild it
    if (dropped) g_state_built = 0xFFFFFFFF;
}

// decode a whole tiled texture into linear RGBA8 (row t = 0 at the bottom,
// matching the sampler's orientation)
static void etc1_block(u64 blk, u32 *out4x4, u64 alpha, bool has_alpha) {
    static const int mods[8][2] = {{2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183}};
    bool flip = (blk >> 32) & 1, diff = (blk >> 33) & 1;
    u32 tbl[2] = {(u32)(blk >> 37) & 7, (u32)(blk >> 34) & 7};
    int rgb[2][3];
    if (diff) {
        int r = (blk >> 59) & 31, g = (blk >> 51) & 31, b = (blk >> 43) & 31;
        int dr = (int)(((blk >> 56) & 7) << 29) >> 29, dg = (int)(((blk >> 48) & 7) << 29) >> 29, db = (int)(((blk >> 40) & 7) << 29) >> 29;
        int c1[3] = {r, g, b}, c2[3] = {(r + dr) & 31, (g + dg) & 31, (b + db) & 31};
        for (int k = 0; k < 3; k++) { rgb[0][k] = (c1[k] << 3) | (c1[k] >> 2); rgb[1][k] = (c2[k] << 3) | (c2[k] >> 2); }
    } else {
        rgb[0][0] = ((blk >> 60) & 15) * 17; rgb[1][0] = ((blk >> 56) & 15) * 17;
        rgb[0][1] = ((blk >> 52) & 15) * 17; rgb[1][1] = ((blk >> 48) & 15) * 17;
        rgb[0][2] = ((blk >> 44) & 15) * 17; rgb[1][2] = ((blk >> 40) & 15) * 17;
    }
    for (u32 x = 0; x < 4; x++)
        for (u32 y = 0; y < 4; y++) {
            int sb = flip ? (y >= 2) : (x >= 2);
            u32 i = x * 4 + y;
            u32 lsb = (blk >> i) & 1, msb = (blk >> (16 + i)) & 1;
            int m = mods[tbl[sb]][lsb];
            if (msb) m = -m;
            int r = std::clamp(rgb[sb][0] + m, 0, 255), g = std::clamp(rgb[sb][1] + m, 0, 255), b = std::clamp(rgb[sb][2] + m, 0, 255);
            u32 a = has_alpha ? ((alpha >> (4 * i)) & 0xF) * 17 : 255;
            out4x4[y * 4 + x] = r | (g << 8) | (b << 16) | (a << 24);
        }
}

static void tex_decode_full(u32 va, u32 w, u32 h, u32 fmt, u32 *out) {
    static const u8 MX[64] = {0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,4,5,4,5,6,7,6,7};
    static const u8 MY[64] = {0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3,0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3,4,4,5,5,4,4,5,5,6,6,7,7,6,6,7,7,4,4,5,5,4,4,5,5,6,6,7,7,6,6,7,7};
    const u8 *p = gpu_ptr(va, tex_bytes(w, h, fmt));
    u32 tiles_x = w / 8, tiles_y = h / 8;
    if (fmt == 0xC || fmt == 0xD) {
        bool alpha = fmt == 0xD;
        u32 bs = alpha ? 16 : 8;
        for (u32 ty = 0; ty < tiles_y; ty++)
            for (u32 tx = 0; tx < tiles_x; tx++) {
                const u8 *tp = p + (ty * tiles_x + tx) * 4 * bs;
                for (u32 sub = 0; sub < 4; sub++) {
                    const u8 *bp = tp + sub * bs;
                    u64 a = 0, blk;
                    if (alpha) { memcpy(&a, bp, 8); bp += 8; }
                    memcpy(&blk, bp, 8);
                    u32 px[16];
                    etc1_block(blk, px, a, alpha);
                    u32 bx = tx * 8 + (sub & 1) * 4, by = ty * 8 + (sub >> 1) * 4;
                    for (u32 y = 0; y < 4; y++)
                        for (u32 x = 0; x < 4; x++) {
                            u32 my = by + y, t = h - 1 - my;
                            out[(size_t)t * w + bx + x] = px[y * 4 + x];
                        }
                }
            }
        return;
    }
    static const u32 bits[16] = {32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 0, 0, 0, 0};
    u32 bpp = bits[fmt & 15];
    u32 tile_bytes = 64 * bpp / 8;
    for (u32 ty = 0; ty < tiles_y; ty++)
        for (u32 tx = 0; tx < tiles_x; tx++) {
            const u8 *tp = p + (ty * tiles_x + tx) * tile_bytes;
            for (u32 i = 0; i < 64; i++) {
                u32 x = tx * 8 + MX[i], my = ty * 8 + MY[i], t = h - 1 - my;
                u32 c;
                switch (fmt) {
                case 0: c = fb_decode(tp + i * 4, 0); break;
                case 1: c = fb_decode(tp + i * 3, 1); break;
                case 2: c = fb_decode(tp + i * 2, 3); break;
                case 3: c = fb_decode(tp + i * 2, 2); break;
                case 4: c = fb_decode(tp + i * 2, 4); break;
                case 5: { u32 a = tp[i * 2], l = tp[i * 2 + 1]; c = l | (l << 8) | (l << 16) | (a << 24); break; }
                case 6: { u32 g = tp[i * 2], r = tp[i * 2 + 1]; c = r | (g << 8) | 0xFF000000u; break; }
                case 7: { u32 l = tp[i]; c = l | (l << 8) | (l << 16) | 0xFF000000u; break; }
                case 8: c = (u32)tp[i] << 24; break;
                case 9: { u32 b = tp[i]; u32 l = (b >> 4) * 17, a = (b & 15) * 17; c = l | (l << 8) | (l << 16) | (a << 24); break; }
                case 10: { u32 b = tp[i / 2]; u32 l = ((i & 1) ? b >> 4 : b & 15) * 17; c = l | (l << 8) | (l << 16) | 0xFF000000u; break; }
                case 11: { u32 b = tp[i / 2]; c = (((i & 1) ? b >> 4 : b & 15) * 17) << 24; break; }
                default: c = 0xFFFF00FF;
                }
                out[(size_t)t * w + x] = c;
            }
        }
}

static const u32 *tex_get(u32 va, u32 w, u32 h, u32 fmt, u64 *id = nullptr) {
    std::lock_guard<std::mutex> lk(g_texcache_mtx);
    TexKey k{va, w, h, fmt};
    auto it = g_texcache.find(k);
    if (it != g_texcache.end()) { if (id) *id = it->second.id; return it->second.px.data(); }
    std::vector<u32> v((size_t)w * h);
    {
        static u32 wlo = getenv("R3DS_GX_WATCH") ? (u32)strtoul(getenv("R3DS_GX_WATCH"), 0, 16) : 0;
        if (wlo && va == wlo) {
            u32 hh = 2166136261u, n = tex_bytes(w, h, fmt);
            for (u32 i = 0; i < n; i++) { hh ^= rd8(va + i); hh *= 16777619u; }
            LOG("[texwatch] decode %08x %ux%u fmt %u hash %08x", va, w, h, fmt, hh);
        }
    }
    if ((w & 7) == 0 && (h & 7) == 0 && !getenv("R3DS_SLOW_TEXDEC")) {
        tex_decode_full(va, w, h, fmt, v.data());
        static bool check = getenv("R3DS_TEXCHECK") != nullptr;
        if (check) {
            u32 bad = 0;
            for (u32 t = 0; t < h; t++)
                for (u32 s2 = 0; s2 < w; s2++)
                    if (v[(size_t)t * w + s2] != pica_tex_texel(va, w, h, fmt, s2, t)) bad++;
            static std::set<u32> seenfmt;
            if (bad || seenfmt.insert(fmt).second) LOG("[texcheck] fmt %u %ux%u mismatches %u", fmt, w, h, bad);
        }
    }
    else
        for (u32 t = 0; t < h; t++)
            for (u32 s = 0; s < w; s++) v[(size_t)t * w + s] = pica_tex_texel(va, w, h, fmt, s, t);
    {   // R3DS_TEX_DUMP=va[,dir]: write each decode of the texture at va as raw RGBA (debugging)
        static u32 dva = getenv("R3DS_TEX_DUMP") ? (u32)strtoul(getenv("R3DS_TEX_DUMP"), 0, 16) : 0;
        static int dumps = 0;
        if (dva && va == dva && dumps < 64) {
            char path[512];
            snprintf(path, sizeof path, "%s/tex_%08x_%ux%u_f%u_%03d.rgba", getenv("R3DS_TEX_DUMP_DIR") ? getenv("R3DS_TEX_DUMP_DIR") : ".", va, w, h, fmt, dumps++);
            if (FILE *f = fopen(path, "wb")) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        }
    }
    auto &slot = g_texcache[k];
    slot.px = std::move(v);
    slot.id = ++g_tex_decode_id;
    if (id) *id = slot.id;
    return slot.px.data();
}

// ------------------------------------------------------------ state
struct TexView { const u32 *px; int w, h; u32 ws, wt; bool linear, on; u32 border; bool pow2; float fw, fh; };
struct TevStage { u8 cs[3], as[3], co[3], ao[3], cop, aop, cscale, ascale; u32 konst; bool upd_c, upd_a; };
struct DrawState {
    u8 *cbuf, *dbuf; u32 fbw, fbh, cfmt, dfmt, cbpp, dbpp;
    u32 cva, dva;   // guest addresses of the color / depth buffers
    u8 *cov;        // coverage mask of a screen render target (row-major, fbw per row) or null
    Coverage *covr;
    bool can_cw, can_dw, can_dr, can_sw;
    int sx0, sy0, sx1, sy1; u32 smode; int ex1, ey1, ex2, ey2;
    TexView tex[3]; bool tex2_tc1;
    TevStage tev[6]; int ntev; u32 buf_init;
    bool lighting;
    u32 colop, blend, lop, bcol;
    bool alpha_test; u32 at_func; int at_ref;
    bool st_en; u32 st_func, sop; u8 st_wmask, st_ref, st_imask;
    bool depth_test, dwrite; u32 dfunc, wmask;
    float dscale, doff; bool wbuf;
    bool need_color, need_tc0, need_tc1, need_tc2;
};

static std::vector<DrawState> g_qstates;   // states of the queued triangles (deferred shading)
// distinct color / depth buffer ranges written by the queued triangles
static std::vector<std::pair<u32, u32>> g_q_targets;
static void note_targets(const DrawState &S) {
    auto add = [](u32 a, u32 n) {
        if (!a || !n) return;
        for (auto &t : g_q_targets) if (t.first == a && t.second == n) return;
        g_q_targets.push_back({a, n});
    };
    add(S.cva, S.fbw * S.fbh * S.cbpp);
    add(S.dva, S.fbw * S.fbh * S.dbpp);
}

static void build_state(DrawState &S) {
    u32 color_pa = R[0x11D] << 3, depth_pa = R[0x11C] << 3;
    S.fbw = R[0x11E] & 0x7FF; S.fbh = ((R[0x11E] >> 12) & 0x3FF) + 1;
    S.cbuf = pa_ptr(color_pa); S.dbuf = pa_ptr(depth_pa);
    S.cva = pa_to_va(color_pa); S.dva = pa_to_va(depth_pa);
    S.covr = S.cbuf ? cov_find(S.cva, R[0x11E] & 0x7FF, ((R[0x11E] >> 12) & 0x3FF) + 1) : nullptr;
    S.cov = S.covr ? S.covr->m.data() : nullptr;
    if (S.cbuf) {   // rendering into a buffer that may be sampled later: drop stale decodes
        std::lock_guard<std::mutex> lk(g_texcache_mtx);
        u32 va = pa_to_va(color_pa), size = S.fbw * S.fbh * 4;
        for (auto it = g_texcache.begin(); it != g_texcache.end();) {
            u32 a = it->first.va, e = a + tex_bytes(it->first.w, it->first.h, it->first.fmt);
            if (e > va && a < va + size) {
                // queued triangles of earlier draws may still sample it: keep the
                // pixels alive until the next flush instead of freeing them now
                g_tex_graveyard.push_back(std::move(it->second.px));
                it = g_texcache.erase(it);
            } else ++it;
        }
    }
    S.cfmt = (R[0x117] >> 16) & 7; S.dfmt = R[0x116] & 3;
    S.cbpp = fb_bpp(S.cfmt); S.dbpp = S.dfmt == 0 ? 2 : S.dfmt == 3 ? 4 : 3;
    S.can_cw = R[0x113] & 0xF; S.can_dw = (R[0x115] & 2) != 0; S.can_dr = (R[0x114] & 2) != 0; S.can_sw = (R[0x115] & 1) != 0;
    S.smode = R[0x65] & 3;
    S.sx0 = 0; S.sy0 = 0; S.sx1 = S.fbw; S.sy1 = S.fbh;
    S.ex1 = R[0x66] & 0x3FF; S.ey1 = (R[0x66] >> 16) & 0x3FF; S.ex2 = R[0x67] & 0x3FF; S.ey2 = (R[0x67] >> 16) & 0x3FF;
    if (S.smode == 3) {
        S.sx0 = std::max(S.sx0, S.ex1); S.sy0 = std::max(S.sy0, S.ey1);
        S.sx1 = std::min(S.sx1, S.ex2 + 1); S.sy1 = std::min(S.sy1, S.ey2 + 1);
    }
    u32 tcfg = R[0x80];
    const u32 tbase[3] = {0x81, 0x91, 0x99};
    for (int i = 0; i < 3; i++) {
        u32 b = tbase[i];
        TexView &t = S.tex[i];
        t.on = (tcfg >> i) & 1;
        t.border = R[b];
        t.h = R[b + 1] & 0x7FF; t.w = (R[b + 1] >> 16) & 0x7FF;
        u32 param = R[b + 2];
        t.ws = (param >> 12) & 7; t.wt = (param >> 8) & 7; t.linear = (param >> 1) & 1;
        u32 va = pa_to_va(R[b + 4] << 3);
        u32 fmt = (i == 0 ? R[0x8E] : R[b + 5]) & 0xF;
        t.px = nullptr;
        if (t.on && va && t.w && t.h && t.w <= 1024 && t.h <= 1024) {
            // sampling a buffer that queued (not yet rasterized) triangles
            // render into: draw those first so the texture sees their pixels
            u32 te = va + tex_bytes(t.w, t.h, fmt);
            for (auto &q : g_q_targets)
                if (te > q.first && va < q.first + q.second) { pica_flush(); break; }
            t.px = tex_get(va, t.w, t.h, fmt);
        }
        if (!t.px) t.on = false;
        t.pow2 = t.w && t.h && !(t.w & (t.w - 1)) && !(t.h & (t.h - 1));
        t.fw = (float)t.w; t.fh = (float)t.h;
    }
    S.tex2_tc1 = (tcfg >> 13) & 1;
    const u32 tev_base[6] = {0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8};
    u32 upd = R[0xE0];
    S.ntev = 0;
    for (int s = 0; s < 6; s++) {
        u32 b = tev_base[s];
        TevStage &T = S.tev[s];
        u32 src = R[b], op = R[b + 1], cmb = R[b + 2], sc = R[b + 4];
        for (int k = 0; k < 3; k++) {
            T.cs[k] = (src >> (4 * k)) & 0xF; T.as[k] = (src >> (16 + 4 * k)) & 0xF;
            T.co[k] = (op >> (4 * k)) & 0xF; T.ao[k] = (op >> (12 + 4 * k)) & 7;
        }
        T.cop = cmb & 0xF; T.aop = (cmb >> 16) & 0xF;
        T.cscale = sc & 3; T.ascale = (sc >> 16) & 3;
        T.konst = R[b + 3];
        T.upd_c = s < 4 && ((upd >> (8 + s)) & 1);
        T.upd_a = s < 4 && ((upd >> (12 + s)) & 1);
        bool pass = T.cop == 0 && T.aop == 0 && T.cs[0] == 15 && T.as[0] == 15 && T.co[0] == 0 && T.ao[0] == 0 &&
                    T.cscale == 0 && T.ascale == 0 && !T.upd_c && !T.upd_a;
        if (!pass) S.ntev = s + 1;
    }
    S.buf_init = R[0xFD];
    {
        static bool fog_seen = false, proc_seen = false, shadow_seen = false;
        u32 fogmode = R[0xE0] & 7;
        if (fogmode && !fog_seen) { fog_seen = true; LOG("[pica] fog/gas mode %u used (E0=%08x E1=%08x) -- not implemented", fogmode, R[0xE0], R[0xE1]); }
        if ((R[0x80] & 8) && !proc_seen) { proc_seen = true; LOG("[pica] procedural texture (tex3) used -- not implemented"); }
        if ((R[0x100] & 3) == 3 && !shadow_seen) { shadow_seen = true; LOG("[pica] shadow rendering mode used -- not implemented"); }
    }
    S.lighting = R[0x8F] & 1;
    if (S.lighting && getenv("R3DS_LIGHT_DUMP")) {
        static int n = 0;
        if (n++ < 3) {
            fprintf(stderr, "[light] 0x1C0-0x1D9:");
            for (u32 r = 0x1C0; r <= 0x1D9; r++) fprintf(stderr, " %03x=%08x", r, R[r]);
            fprintf(stderr, "\n");
            for (int l = 0; l < 8; l++) {
                fprintf(stderr, "[light] L%d:", l);
                for (int k = 0; k < 10; k++) fprintf(stderr, " %08x", R[0x140 + l * 0x10 + k]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "[light] tev:");
            for (int s2 = 0; s2 < 6; s2++) { const u32 tb[6] = {0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8}; fprintf(stderr, " [%08x %08x %08x %08x %08x]", R[tb[s2]], R[tb[s2]+1], R[tb[s2]+2], R[tb[s2]+3], R[tb[s2]+4]); }
            fprintf(stderr, " upd=%08x fog=%08x\n", R[0xE0], R[0xE1]);
        }
    }
    S.colop = R[0x100]; S.blend = R[0x101]; S.lop = R[0x102] & 0xF; S.bcol = R[0x103];
    u32 at = R[0x104], st = R[0x105], dcm = R[0x107];
    S.alpha_test = at & 1; S.at_func = (at >> 4) & 7; S.at_ref = (at >> 8) & 0xFF;
    S.st_en = (st & 1) && S.dfmt == 3; S.st_func = (st >> 4) & 7; S.st_wmask = (st >> 8) & 0xFF; S.st_ref = (st >> 16) & 0xFF; S.st_imask = st >> 24;
    S.sop = R[0x106];
    S.depth_test = dcm & 1; S.dfunc = (dcm >> 4) & 7; S.dwrite = (dcm >> 12) & 1; S.wmask = (dcm >> 8) & 0xF;
    S.dscale = f24(R[0x4D]); S.doff = f24(R[0x4E]); S.wbuf = !(R[0x6D] & 1);
    // which interpolants are needed
    S.need_color = S.need_tc0 = S.need_tc1 = S.need_tc2 = false;
    for (int s = 0; s < S.ntev; s++)
        for (int k = 0; k < 3; k++) {
            u8 a = S.tev[s].cs[k], b = S.tev[s].as[k];
            if (a == 0 || b == 0) S.need_color = true;
        }
    S.need_tc0 = S.tex[0].on; S.need_tc1 = S.tex[1].on || (S.tex[2].on && S.tex2_tc1); S.need_tc2 = S.tex[2].on && !S.tex2_tc1;
}

// ---- for the hardware recorder (hwr_rec.cpp), GX thread
// make memory [va, va+size) current: rasterize queued triangles that write it
void pica_flush_if_queued(u32 va, u32 size) {
    for (auto &q : g_q_targets)
        if (va + size > q.first && va < q.first + q.second) { pica_flush(); return; }
}
// decoded texture as the software sampler sees it (RGBA8, row t = 0 at the
// bottom) and the id of that decode; nullptr when it cannot be sampled
const u32 *pica_tex_decoded(u32 va, u32 w, u32 h, u32 fmt, u64 *id) {
    if (!va || !w || !h || w > 1024 || h > 1024) return nullptr;
    pica_flush_if_queued(va, tex_bytes(w, h, fmt));
    return tex_get(va, w, h, fmt, id);
}
u32 pica_tex_bytes(u32 w, u32 h, u32 fmt) { return tex_bytes(w, h, fmt); }
// copy of the coverage mask of render target va (row-major, memory row order)
bool pica_cov_get(u32 va, u32 w, u32 h, std::vector<u8> &out) {
    auto it = g_cov.find(va);
    if (it == g_cov.end() || it->second.w != w || it->second.h != h || it->second.m.empty()) return false;
    out = it->second.m;
    return true;
}

// ------------------------------------------------------------ helpers
struct C4 { int r, g, b, a; };
static inline C4 unpack(u32 c) { return {(int)(c & 0xFF), (int)((c >> 8) & 0xFF), (int)((c >> 16) & 0xFF), (int)(c >> 24)}; }
static inline u32 pack(C4 c) { return (u32)std::clamp(c.r, 0, 255) | ((u32)std::clamp(c.g, 0, 255) << 8) | ((u32)std::clamp(c.b, 0, 255) << 16) | ((u32)std::clamp(c.a, 0, 255) << 24); }
static inline bool cmpf(u32 f, int a, int b) {
    switch (f) { case 0: return false; case 1: return true; case 2: return a == b; case 3: return a != b;
    case 4: return a < b; case 5: return a <= b; case 6: return a > b; default: return a >= b; }
}
static inline int wrapc(int c, int size, u32 mode, bool &border) {
    switch (mode) {
    case 2: c %= size; if (c < 0) c += size; return c;
    case 3: { int p = size * 2; c %= p; if (c < 0) c += p; return c < size ? c : p - 1 - c; }
    case 1: if (c < 0 || c >= size) border = true; return c < 0 ? 0 : c >= size ? size - 1 : c;
    default: return c < 0 ? 0 : c >= size ? size - 1 : c;
    }
}
static inline int ffloor(float f) { int i = f2i(f); return i - (f < (float)i); }
static inline int fwrap(int c, int size, u32 mode, bool pow2, bool &border) {
    if (mode == 2 && pow2) return c & (size - 1);
    if (mode == 0 || mode >= 4) return c < 0 ? 0 : c >= size ? size - 1 : c;
    return wrapc(c, size, mode, border);
}
static inline u32 lerp4(u32 c00, u32 c10, u32 c01, u32 c11, u32 ws, u32 wt) {
    if (c00 == c10 && c00 == c01 && c00 == c11) return c00;
    u32 out = 0;
    for (int ch = 0; ch < 32; ch += 8) {
        u32 a = (c00 >> ch) & 0xFF, b = (c10 >> ch) & 0xFF, c = (c01 >> ch) & 0xFF, d = (c11 >> ch) & 0xFF;
        u32 top = a * (256 - ws) + b * ws, bot = c * (256 - ws) + d * ws;
        u32 r = (top * (256 - wt) + bot * wt + 32768) >> 16;
        out |= (r > 255 ? 255 : r) << ch;
    }
    return out;
}
// keep texel coordinates in a range where every int conversion below is
// exact (NaN -> the low bound): stray NaN / huge texture coordinates must
// not reach the index math
static inline float tc_clamp(float x) { return fminf(fmaxf(x, -8388608.0f), 8388608.0f); }
static inline __attribute__((always_inline)) u32 sample_fast(const TexView &t, float u, float v) {
    float fu = tc_clamp(u * t.fw), fv = tc_clamp(v * t.fh);
    bool b0 = false;
    if (!t.linear) {
        int s = fwrap(ffloor(fu), t.w, t.ws, t.pow2, b0), tt = fwrap(ffloor(fv), t.h, t.wt, t.pow2, b0);
        return b0 ? t.border : t.px[tt * t.w + s];
    }
    fu -= 0.5f; fv -= 0.5f;
    int s0 = ffloor(fu), t0 = ffloor(fv);
    u32 ws = f2u((fu - s0) * 256.0f), wt = f2u((fv - t0) * 256.0f);
    bool b1 = false, b2 = false, b3 = false;
    int sa = fwrap(s0, t.w, t.ws, t.pow2, b0), sb = fwrap(s0 + 1, t.w, t.ws, t.pow2, b1);
    int ta = fwrap(t0, t.h, t.wt, t.pow2, b2), tb = fwrap(t0 + 1, t.h, t.wt, t.pow2, b3);
    const u32 *ra = t.px + ta * t.w, *rb = t.px + tb * t.w;
    if (!(b0 | b1 | b2 | b3)) return lerp4(ra[sa], ra[sb], rb[sa], rb[sb], ws, wt);
    return lerp4((b0 || b2) ? t.border : ra[sa], (b1 || b2) ? t.border : ra[sb],
                 (b0 || b3) ? t.border : rb[sa], (b1 || b3) ? t.border : rb[sb], ws, wt);
}
static inline u32 sample(const TexView &t, float u, float v) {
    float fu = tc_clamp(u * t.w), fv = tc_clamp(v * t.h);
    if (!t.linear) {
        bool bd = false;
        int s = wrapc(f2i(floorf(fu)), t.w, t.ws, bd), tt = wrapc(f2i(floorf(fv)), t.h, t.wt, bd);
        return bd ? t.border : t.px[tt * t.w + s];
    }
    fu -= 0.5f; fv -= 0.5f;
    float ffu = floorf(fu), ffv = floorf(fv);
    int s0 = f2i(ffu), t0 = f2i(ffv);
    u32 ws = f2u((fu - ffu) * 256.0f), wt = f2u((fv - ffv) * 256.0f);
    bool b0 = false, b1 = false, b2 = false, b3 = false;
    int sa = wrapc(s0, t.w, t.ws, b0), sb = wrapc(s0 + 1, t.w, t.ws, b1);
    int ta = wrapc(t0, t.h, t.wt, b2), tb = wrapc(t0 + 1, t.h, t.wt, b3);
    u32 c00 = (b0 || b2) ? t.border : t.px[ta * t.w + sa];
    u32 c10 = (b1 || b2) ? t.border : t.px[ta * t.w + sb];
    u32 c01 = (b0 || b3) ? t.border : t.px[tb * t.w + sa];
    u32 c11 = (b1 || b3) ? t.border : t.px[tb * t.w + sb];
    if (c00 == c10 && c00 == c01 && c00 == c11) return c00;
    u32 out = 0;
    for (int ch = 0; ch < 32; ch += 8) {
        u32 a = (c00 >> ch) & 0xFF, b = (c10 >> ch) & 0xFF, c = (c01 >> ch) & 0xFF, d = (c11 >> ch) & 0xFF;
        u32 top = a * (256 - ws) + b * ws, bot = c * (256 - ws) + d * ws;
        u32 r = (top * (256 - wt) + bot * wt + 32768) >> 16;
        out |= (r > 255 ? 255 : r) << ch;
    }
    return out;
}

static inline void color_mod(u32 op, C4 v, int out[3]) {
    switch (op) {
    case 0: out[0] = v.r; out[1] = v.g; out[2] = v.b; break;
    case 1: out[0] = 255 - v.r; out[1] = 255 - v.g; out[2] = 255 - v.b; break;
    case 2: out[0] = out[1] = out[2] = v.a; break;
    case 3: out[0] = out[1] = out[2] = 255 - v.a; break;
    case 4: out[0] = out[1] = out[2] = v.r; break;
    case 5: out[0] = out[1] = out[2] = 255 - v.r; break;
    case 8: out[0] = out[1] = out[2] = v.g; break;
    case 9: out[0] = out[1] = out[2] = 255 - v.g; break;
    case 12: out[0] = out[1] = out[2] = v.b; break;
    case 13: out[0] = out[1] = out[2] = 255 - v.b; break;
    default: out[0] = v.r; out[1] = v.g; out[2] = v.b;
    }
}
static inline int alpha_mod(u32 op, C4 v) {
    switch (op) {
    case 0: return v.a; case 1: return 255 - v.a; case 2: return v.r; case 3: return 255 - v.r;
    case 4: return v.g; case 5: return 255 - v.g; case 6: return v.b; default: return 255 - v.b;
    }
}
static inline int combine1(u32 op, int a, int b, int c) {
    switch (op) {
    case 0: return a;
    case 1: return (a * b + 127) / 255;
    case 2: return std::min(a + b, 255);
    case 3: return std::clamp(a + b - 128, 0, 255);
    case 4: return (a * c + b * (255 - c) + 127) / 255;
    case 5: return std::max(a - b, 0);
    case 8: return std::min((a * b + 255 * c) / 255, 255);
    case 9: return std::min(a + b, 255) * c / 255;
    default: return a;
    }
}
static inline float blend_factor(u32 f, int ch, C4 s, C4 d, C4 k) {
    const float n = 1.0f / 255.0f;
    int sc = ch == 0 ? s.r : ch == 1 ? s.g : ch == 2 ? s.b : s.a;
    int dc = ch == 0 ? d.r : ch == 1 ? d.g : ch == 2 ? d.b : d.a;
    int kc = ch == 0 ? k.r : ch == 1 ? k.g : ch == 2 ? k.b : k.a;
    switch (f) {
    case 0: return 0; case 1: return 1;
    case 2: return sc * n; case 3: return 1 - sc * n;
    case 4: return dc * n; case 5: return 1 - dc * n;
    case 6: return s.a * n; case 7: return 1 - s.a * n;
    case 8: return d.a * n; case 9: return 1 - d.a * n;
    case 10: return kc * n; case 11: return 1 - kc * n;
    case 12: return k.a * n; case 13: return 1 - k.a * n;
    case 14: return ch == 3 ? 1 : std::min(s.a * n, 1 - d.a * n);
    }
    return 1;
}
static inline u32 logic_op(u32 op, u32 s, u32 d) {
    switch (op) {
    case 0: return 0; case 1: return s & d; case 2: return s & ~d; case 3: return s; case 4: return 0xFFFFFFFF;
    case 5: return ~s; case 6: return d; case 7: return ~d; case 8: return ~(s & d); case 9: return s | d;
    case 10: return ~(s | d); case 11: return s ^ d; case 12: return ~(s ^ d); case 13: return ~s & d;
    case 14: return s | ~d; default: return ~s | d;
    }
}
static inline u8 stencil_op(u32 op, u8 old, u8 ref) {
    switch (op) {
    case 0: return old; case 1: return 0; case 2: return ref; case 3: return old == 255 ? 255 : old + 1;
    case 4: return old == 0 ? 0 : old - 1; case 5: return ~old; case 6: return old + 1; default: return old - 1;
    }
}

// ------------------------------------------------------------ triangle setup
enum { A_CR, A_CG, A_CB, A_CA, A_U0, A_V0, A_U1, A_V1, A_U2, A_V2, A_QX, A_QY, A_QZ, A_QW, A_VX, A_VY, A_VZ, A_N };
struct Plane { float a, b, c; };
struct Tri {
    int x[3], y[3];
    s64 bias[3];
    int minx, maxx, miny, maxy;
    Plane invw, z, attr[A_N];
    const OutVertex *v0;
    bool backdrop;
};

static inline Plane make_plane(const float *sx, const float *sy, const float *val) {
    float x1 = sx[1] - sx[0], y1 = sy[1] - sy[0], x2 = sx[2] - sx[0], y2 = sy[2] - sy[0];
    float v1 = val[1] - val[0], v2 = val[2] - val[0];
    float det = x1 * y2 - x2 * y1;
    Plane p;
    if (det == 0) { p.a = p.b = 0; p.c = val[0]; return p; }
    float id = 1.0f / det;
    p.a = (v1 * y2 - v2 * y1) * id;
    p.b = (x1 * v2 - x2 * v1) * id;
    p.c = val[0] - p.a * sx[0] - p.b * sy[0];
    return p;
}

static void shade_rows(const DrawState &S, const Tri &T, int row_mod, int row_rem);

// deferred triangle queue
struct QTri { int state; Tri tri; };
static std::vector<QTri> g_queue;
static double g_draw_area = 0;           // screen area of the current draw's triangles
static double g_draw_vp_area = 0;        // its viewport area
static size_t g_draw_q0 = 0;             // queue index where the current draw started
static u64 g_flush_gen = 0, g_draw_flush_gen = 0;
static bool g_in_draw = false;
static void run_queue_band(int row_mod, int row_rem) {
    for (auto &q : g_queue) shade_rows(g_qstates[q.state], q.tri, row_mod, row_rem);
}

// ------------------------------------------------------------ thread pool
struct Pool {
    std::vector<std::thread> th;
    std::mutex m; std::condition_variable cv, done_cv;
    u64 gen = 0; int pending = 0; int n = 1;
    const std::function<void(int, int)> *job = nullptr;
    unsigned csr = 0;
    void start(int workers) {
        n = workers + 1;
        for (int i = 0; i < workers; i++)
            th.emplace_back([this, i] {
                u64 seen = 0;
                for (;;) {
                    const std::function<void(int, int)> *j;
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [&] { return gen != seen; });
                        seen = gen; j = job;
                        CSR_SET(csr);   // same rounding/FTZ as the submitting thread
                    }
                    (*j)(i + 1, n);
                    {
                        std::lock_guard<std::mutex> lk(m);
                        if (--pending == 0) done_cv.notify_one();
                    }
                }
            });
    }
    void run(const std::function<void(int, int)> &f) {
        {
            std::lock_guard<std::mutex> lk(m);
            job = &f; pending = n - 1; gen++; csr = CSR_GET();
        }
        cv.notify_all();
        f(0, n);
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return pending == 0; });
    }
};
static Pool *g_pool;

// state cache: rebuilt when any register changed since last build
extern u32 g_pica_state_gen;
static DrawState g_state;

static u64 g_queue_pixels = 0;
static Pool *pool() {
    if (!g_pool) {
        g_pool = new Pool();
        int n = std::thread::hardware_concurrency();
#ifdef __EMSCRIPTEN__
        n = std::min(n, 8);   // each worker is a Web Worker
#endif
        if (getenv("R3DS_RASTER_THREADS")) n = atoi(getenv("R3DS_RASTER_THREADS"));
        g_pool->start(std::max(0, std::min(n, 16) - 1));
    }
    return g_pool;
}
// run f(part, nparts) on all pool threads (caller included)
void pica_parallel(const std::function<void(int, int)> &f) {
    Pool *p = pool();
    if (p->n > 1) p->run(f); else f(0, 1);
}

void pica_flush() {
    if (g_queue.empty()) return;
    Pool *p = pool();
    if (p->n > 1 && g_queue_pixels >= 2048) {
        std::function<void(int, int)> f = [](int part, int n) { run_queue_band(n, part); };
        p->run(f);
    }
    else run_queue_band(1, 0);
    g_queue.clear();
    g_queue_pixels = 0;
    g_flush_gen++;
    {   // no queued triangle can reference a retired decode any more
        std::lock_guard<std::mutex> lk(g_texcache_mtx);
        if (!g_tex_graveyard.empty()) { g_tex_graveyard.clear(); g_state_built = 0xFFFFFFFF; }
    }
    // keep only the current state snapshot
    g_q_targets.clear();
    if (!g_qstates.empty()) { DrawState last = g_qstates.back(); g_qstates.clear(); g_qstates.push_back(last); note_targets(last); }
}

void raster_draw_begin() {
    g_draw_vp_area = (double)f24(R[0x41]) * 2 * (double)f24(R[0x43]) * 2;
    g_draw_area = 0; g_draw_q0 = g_queue.size(); g_draw_flush_gen = g_flush_gen; g_in_draw = true;
}
int raster_draw_end(double *area, double *rt_area) {
    if (!g_in_draw) return 0;
    g_in_draw = false;
    if (area) *area = g_draw_area;
    if (g_qstates.empty() || (g_queue.size() == g_draw_q0 && g_draw_flush_gen == g_flush_gen)) return 0;   // drew nothing
    const DrawState &S = g_qstates.back();
    Coverage *c = S.covr;
    if (!c) return 0;
    double ref = std::min(g_draw_vp_area, (double)c->w * c->h);
    if (ref <= 0) ref = (double)c->w * c->h;
    if (rt_area) *rt_area = ref;
    if (g_draw_area >= 0.85 * ref) {
        if (g_draw_flush_gen == g_flush_gen)
            for (size_t i = g_draw_q0; i < g_queue.size(); i++) g_queue[i].tri.backdrop = true;
        return 1;
    }
    return 2;
}

void raster_tri_fast(const OutVertex &v0in, const OutVertex &v1in, const OutVertex &v2in) {
    const OutVertex *v[3] = {&v0in, &v1in, &v2in};
    float area = (v[1]->sx - v[0]->sx) * (v[2]->sy - v[0]->sy) - (v[1]->sy - v[0]->sy) * (v[2]->sx - v[0]->sx);
    u32 cull = R[0x40] & 3;
    if (cull == 1 && area > 0) return;
    if (cull == 2 && area < 0) return;
    if (area == 0) return;
    if (area < 0) std::swap(v[1], v[2]);
    if (g_state_built != g_pica_state_gen || g_qstates.empty()) {
        build_state(g_state); g_state_built = g_pica_state_gen;
        g_qstates.push_back(g_state);
        note_targets(g_state);
    }
    const DrawState &S = g_qstates.back();
    if (!S.cbuf || !S.fbw) return;

    Tri T;
    for (int i = 0; i < 3; i++) { T.x[i] = (int)llrintf(v[i]->sx * 16.0f); T.y[i] = (int)llrintf(v[i]->sy * 16.0f); }
    auto is_topleft = [](int ax, int ay, int bx, int by) { return (ay == by && bx < ax) || (by < ay); };
    T.bias[0] = is_topleft(T.x[1], T.y[1], T.x[2], T.y[2]) ? 0 : -1;
    T.bias[1] = is_topleft(T.x[2], T.y[2], T.x[0], T.y[0]) ? 0 : -1;
    T.bias[2] = is_topleft(T.x[0], T.y[0], T.x[1], T.y[1]) ? 0 : -1;
    s64 tot = (s64)(T.x[1] - T.x[0]) * (T.y[2] - T.y[0]) - (s64)(T.y[1] - T.y[0]) * (T.x[2] - T.x[0]);
    if (tot <= 0) return;
    T.minx = std::max(std::min({T.x[0], T.x[1], T.x[2]}) >> 4, S.sx0);
    T.maxx = std::min((std::max({T.x[0], T.x[1], T.x[2]}) + 15) >> 4, S.sx1 - 1);
    T.miny = std::max(std::min({T.y[0], T.y[1], T.y[2]}) >> 4, S.sy0);
    T.maxy = std::min((std::max({T.y[0], T.y[1], T.y[2]}) + 15) >> 4, S.sy1 - 1);
    if (T.minx > T.maxx || T.miny > T.maxy) return;

    float sx[3], sy[3], val[3];
    for (int i = 0; i < 3; i++) { sx[i] = T.x[i] / 16.0f; sy[i] = T.y[i] / 16.0f; }
    for (int i = 0; i < 3; i++) val[i] = v[i]->invw;
    T.invw = make_plane(sx, sy, val);
    for (int i = 0; i < 3; i++) val[i] = v[i]->sz;
    T.z = make_plane(sx, sy, val);
    auto P = [&](int slot, auto get) {
        for (int i = 0; i < 3; i++) val[i] = get(*v[i]) * v[i]->invw;
        T.attr[slot] = make_plane(sx, sy, val);
    };
    if (S.need_color) {
        P(A_CR, [](const OutVertex &o) { return o.color.x; }); P(A_CG, [](const OutVertex &o) { return o.color.y; });
        P(A_CB, [](const OutVertex &o) { return o.color.z; }); P(A_CA, [](const OutVertex &o) { return o.color.w; });
    }
    if (S.need_tc0) { P(A_U0, [](const OutVertex &o) { return o.tc0[0]; }); P(A_V0, [](const OutVertex &o) { return o.tc0[1]; }); }
    if (S.need_tc1) { P(A_U1, [](const OutVertex &o) { return o.tc1[0]; }); P(A_V1, [](const OutVertex &o) { return o.tc1[1]; }); }
    if (S.need_tc2) { P(A_U2, [](const OutVertex &o) { return o.tc2[0]; }); P(A_V2, [](const OutVertex &o) { return o.tc2[1]; }); }
    if (S.lighting) {
        P(A_QX, [](const OutVertex &o) { return o.quat.x; }); P(A_QY, [](const OutVertex &o) { return o.quat.y; });
        P(A_QZ, [](const OutVertex &o) { return o.quat.z; }); P(A_QW, [](const OutVertex &o) { return o.quat.w; });
        P(A_VX, [](const OutVertex &o) { return o.view[0]; }); P(A_VY, [](const OutVertex &o) { return o.view[1]; });
        P(A_VZ, [](const OutVertex &o) { return o.view[2]; });
    }
    T.v0 = nullptr;
    T.backdrop = false;
    {   // screen area this draw covers (for backdrop classification)
        double bb = (double)(T.maxx - T.minx + 1) * (T.maxy - T.miny + 1);
        g_draw_area += std::min(fabs((double)area) * 0.5, bb);
    }
    g_queue_pixels += (u64)(T.maxx - T.minx + 1) * (T.maxy - T.miny + 1);
    g_queue.push_back({(int)g_qstates.size() - 1, T});
    if (g_queue.size() >= 20000) pica_flush();
}

static void shade_rows(const DrawState &S, const Tri &T, int row_mod, int row_rem) {
    auto edge = [](int ax, int ay, int bx, int by, int px, int py) -> s64 {
        return (s64)(bx - ax) * (py - ay) - (s64)(by - ay) * (px - ax);
    };
    const int *x = T.x, *y = T.y;
    s64 dx0 = -(s64)(y[2] - y[1]) * 16, dx1 = -(s64)(y[0] - y[2]) * 16, dx2 = -(s64)(y[1] - y[0]) * 16;
    u64 frags = 0;
    for (int fy = T.miny; fy <= T.maxy; fy++) {
        if (((fy >> 3) % row_mod) != row_rem) continue;
        int py = fy * 16 + 8, px0 = T.minx * 16 + 8;
        s64 w0 = edge(x[1], y[1], x[2], y[2], px0, py) + T.bias[0];
        s64 w1 = edge(x[2], y[2], x[0], y[0], px0, py) + T.bias[1];
        s64 w2 = edge(x[0], y[0], x[1], y[1], px0, py) + T.bias[2];
        u32 row = S.fbh - 1 - fy;
        float cy = fy + 0.5f;
        const u32 crow = (row & ~7u) * S.fbw, rlow = row & 7;
        static const u8 MORTON[8][8] = {
            {0, 1, 4, 5, 16, 17, 20, 21}, {2, 3, 6, 7, 18, 19, 22, 23}, {8, 9, 12, 13, 24, 25, 28, 29}, {10, 11, 14, 15, 26, 27, 30, 31},
            {32, 33, 36, 37, 48, 49, 52, 53}, {34, 35, 38, 39, 50, 51, 54, 55}, {40, 41, 44, 45, 56, 57, 60, 61}, {42, 43, 46, 47, 58, 59, 62, 63}};
        const u8 *mrow = MORTON[rlow];
        for (int fx = T.minx; fx <= T.maxx; fx++, w0 += dx0, w1 += dx1, w2 += dx2) {
            if ((w0 | w1 | w2) < 0) continue;
            if (S.smode == 1 && fx >= S.ex1 && fx <= S.ex2 && fy >= S.ey1 && fy <= S.ey2) continue;
            float cx = fx + 0.5f;
            float iw = T.invw.a * cx + T.invw.b * cy + T.invw.c;
            float W = 1.0f / iw;
            auto I = [&](int a) { const Plane &p = T.attr[a]; return (p.a * cx + p.b * cy + p.c) * W; };
            float z = T.z.a * cx + T.z.b * cy + T.z.c;
            float depth = z * S.dscale + S.doff;
            if (S.wbuf) depth *= W;
            depth = depth < 0 ? 0 : depth > 1 ? 1 : depth;

            C4 prim = {255, 255, 255, 255};
            if (S.need_color) {
                auto c8 = [](float f) { int v = f2i(f * 255.0f + 0.5f); return v < 0 ? 0 : v > 255 ? 255 : v; };
                prim = {c8(I(A_CR)), c8(I(A_CG)), c8(I(A_CB)), c8(I(A_CA))};
            }
            C4 texc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
            if (S.tex[0].on) texc[0] = unpack(sample_fast(S.tex[0], I(A_U0), I(A_V0)));
            if (S.tex[1].on) texc[1] = unpack(sample_fast(S.tex[1], I(A_U1), I(A_V1)));
            if (S.tex[2].on) texc[2] = unpack(S.tex2_tc1 ? sample_fast(S.tex[2], I(A_U1), I(A_V1)) : sample_fast(S.tex[2], I(A_U2), I(A_V2)));
            C4 fprim = {0, 0, 0, 0}, fsec = {0, 0, 0, 0};
            if (S.lighting) {
                float q[4] = {I(A_QX), I(A_QY), I(A_QZ), I(A_QW)};
                float vw[3] = {I(A_VX), I(A_VY), I(A_VZ)};
                u32 pp, ss;
                static OutVertex dummy;
                pica_lighting(dummy, vw, q, &pp, &ss);
                fprim = unpack(pp); fsec = unpack(ss);
            }
            C4 out = {0, 0, 0, 0}, next_buf = unpack(S.buf_init);
            C4 srcs[16];
            srcs[0] = prim; srcs[1] = fprim; srcs[2] = fsec;
            srcs[3] = texc[0]; srcs[4] = texc[1]; srcs[5] = texc[2]; srcs[6] = texc[3];
            for (int k = 7; k < 13; k++) srcs[k] = prim;
            srcs[13] = {0, 0, 0, 0}; srcs[15] = out;
            for (int s = 0; s < S.ntev; s++) {
                const TevStage &Tv = S.tev[s];
                srcs[14] = unpack(Tv.konst);
                auto src = [&](u32 id) -> const C4 & { return srcs[id]; };
                int c[3][3], a[3];
                for (int k = 0; k < 3; k++) color_mod(Tv.co[k], src(Tv.cs[k]), c[k]);
                for (int k = 0; k < 3; k++) a[k] = alpha_mod(Tv.ao[k], src(Tv.as[k]));
                int rc[3];
                if (Tv.cop == 6 || Tv.cop == 7) {
                    int d = ((c[0][0] - 128) * (c[1][0] - 128) + (c[0][1] - 128) * (c[1][1] - 128) + (c[0][2] - 128) * (c[1][2] - 128)) * 4 / 255;
                    d = d < 0 ? 0 : d > 255 ? 255 : d;
                    rc[0] = rc[1] = rc[2] = d;
                } else for (int k = 0; k < 3; k++) rc[k] = combine1(Tv.cop, c[0][k], c[1][k], c[2][k]);
                int ra = (Tv.cop == 7) ? rc[0] : combine1(Tv.aop == 6 || Tv.aop == 7 ? 0 : Tv.aop, a[0], a[1], a[2]);
                int cs = 1 << Tv.cscale, as = 1 << Tv.ascale;
                out = {std::min(rc[0] * cs, 255), std::min(rc[1] * cs, 255), std::min(rc[2] * cs, 255), std::min(ra * as, 255)};
                srcs[15] = out;
                srcs[13] = next_buf;
                if (Tv.upd_c) { next_buf.r = out.r; next_buf.g = out.g; next_buf.b = out.b; }
                if (Tv.upd_a) next_buf.a = out.a;
            }
            if (S.alpha_test && !cmpf(S.at_func, out.a, S.at_ref)) continue;
            const u32 pix = crow + ((u32)fx & ~7u) * 8 + mrow[fx & 7];
            u8 *dp = S.dbuf ? S.dbuf + pix * S.dbpp : nullptr;
            u32 dmax = S.dfmt == 0 ? 0xFFFF : 0xFFFFFF;
            u32 zval = f2u(depth * dmax);
            u32 stored = 0; u8 sten = 0;
            if (dp) {
                if (S.dfmt == 0) stored = dp[0] | (dp[1] << 8);
                else { stored = dp[0] | (dp[1] << 8) | (dp[2] << 16); if (S.dfmt == 3) sten = dp[3]; }
            }
            if (S.st_en && dp) {
                if (!cmpf(S.st_func, S.st_ref & S.st_imask, sten & S.st_imask)) {
                    if (S.can_sw) { u8 ns = stencil_op(S.sop & 7, sten, S.st_ref); dp[3] = (sten & ~S.st_wmask) | (ns & S.st_wmask); }
                    continue;
                }
            }
            if (S.depth_test && dp && S.can_dr && !cmpf(S.dfunc, (int)zval, (int)stored)) {
                if (S.st_en && S.can_sw) { u8 ns = stencil_op((S.sop >> 4) & 7, sten, S.st_ref); dp[3] = (sten & ~S.st_wmask) | (ns & S.st_wmask); }
                continue;
            }
            if (S.st_en && dp && S.can_sw) { u8 ns = stencil_op((S.sop >> 8) & 7, sten, S.st_ref); dp[3] = (sten & ~S.st_wmask) | (ns & S.st_wmask); }
            if (S.dwrite && dp && S.can_dw) {
                dp[0] = zval & 0xFF; dp[1] = (zval >> 8) & 0xFF;
                if (S.dfmt != 0) dp[2] = (zval >> 16) & 0xFF;
            }
            if (!S.can_cw) continue;
            u8 *cp = S.cbuf + pix * S.cbpp;
            C4 res;
            if (S.colop & 0x100) {
                u32 eqc = S.blend & 7, eqa = (S.blend >> 8) & 7;
                u32 sfc = (S.blend >> 16) & 0xF, dfc = (S.blend >> 20) & 0xF, sfa = (S.blend >> 24) & 0xF, dfa = S.blend >> 28;
                if (eqc == 0 && eqa == 0 && sfc == 1 && dfc == 0 && sfa == 1 && dfa == 0) {
                    res = out;
                } else {
                    C4 dst = unpack(fb_decode(cp, S.cfmt));
                    C4 bc = unpack(S.bcol);
                    float r[4];
                    for (int ch = 0; ch < 4; ch++) {
                        int sc = ch == 0 ? out.r : ch == 1 ? out.g : ch == 2 ? out.b : out.a;
                        int dc = ch == 0 ? dst.r : ch == 1 ? dst.g : ch == 2 ? dst.b : dst.a;
                        float sf = blend_factor(ch == 3 ? sfa : sfc, ch, out, dst, bc);
                        float df = blend_factor(ch == 3 ? dfa : dfc, ch, out, dst, bc);
                        u32 eq = ch == 3 ? eqa : eqc;
                        float Sv = sc * sf, Dv = dc * df;
                        switch (eq) {
                        case 0: r[ch] = Sv + Dv; break; case 1: r[ch] = Sv - Dv; break; case 2: r[ch] = Dv - Sv; break;
                        case 3: r[ch] = std::min(sc, dc); break; default: r[ch] = std::max(sc, dc);
                        }
                    }
                    res = {f2i(r[0] + 0.5f), f2i(r[1] + 0.5f), f2i(r[2] + 0.5f), f2i(r[3] + 0.5f)};
                    if (S.wmask != 0xF) {
                        if (!(S.wmask & 1)) res.r = dst.r;
                        if (!(S.wmask & 2)) res.g = dst.g;
                        if (!(S.wmask & 4)) res.b = dst.b;
                        if (!(S.wmask & 8)) res.a = dst.a;
                    }
                    fb_encode(cp, S.cfmt, pack(res));
                    if (S.cov && !T.backdrop) {   // translucent content: its alpha is its coverage
                        u8 &m = S.cov[row * S.fbw + fx];
                        u8 a = (u8)std::clamp(out.a, 0, 255);
                        if (a > m) m = a;
                    }
                    frags++;
                    continue;
                }
            } else {
                C4 dst = unpack(fb_decode(cp, S.cfmt));
                res = unpack(logic_op(S.lop, pack(out), pack(dst)));
            }
            if (S.wmask != 0xF) {
                C4 dst = unpack(fb_decode(cp, S.cfmt));
                if (!(S.wmask & 1)) res.r = dst.r;
                if (!(S.wmask & 2)) res.g = dst.g;
                if (!(S.wmask & 4)) res.b = dst.b;
                if (!(S.wmask & 8)) res.a = dst.a;
            }
            fb_encode(cp, S.cfmt, pack(res));
            if (S.cov) S.cov[row * S.fbw + fx] = T.backdrop ? 0 : 255;   // opaque backdrop clears, content marks
            frags++;
        }
    }
    __atomic_fetch_add(&g_frag_count, frags, __ATOMIC_RELAXED);
}

// ---- coverage API (gpu_xfer.cpp, display.cpp, frontends)
// a display transfer presents render target in_va (w x h) at LCD buffer
// out_va: remember it as a screen target and hand its mask to the buffer
void cov_on_display_transfer(u32 in_va, u32 in_w, u32 in_h, u32 out_va, u32 out_w, u32 out_h, bool flip, u32 hs, u32 vs) {
    auto it = g_cov.find(in_va);
    if (it == g_cov.end()) { g_cov[in_va]; return; }   // learn it; dims are set by the next draw into it
    Coverage &c = it->second;
    // the transfer may read only the first rows of the target (the game renders
    // the 320-row bottom screen into a 400-row target): same width, row prefix
    if (c.w == in_w && in_h <= c.h) {
        std::vector<u8> out((size_t)out_w * out_h);
        for (u32 y = 0; y < out_h; y++) {
            u32 oy = flip ? out_h - 1 - y : y, iy = y << vs;
            if (iy >= in_h) continue;
            for (u32 x = 0; x < out_w; x++) {
                u32 ix = x << hs;
                out[(size_t)oy * out_w + x] = ix < in_w ? c.m[(size_t)iy * in_w + ix] : 0;
            }
        }
        std::lock_guard<std::mutex> lk(g_cov_mtx);
        g_fbmask[out_va] = std::move(out);
    }
}
// a memory fill (clear) over a screen render target starts a new frame there
void cov_on_fill(u32 start, u32 end) {
    for (auto &kv : g_cov)
        if (kv.first >= start && kv.first < end) std::fill(kv.second.m.begin(), kv.second.m.end(), 0);
}
// copy of the mask of LCD buffer va (fb layout: row r = one 240-pixel column of the screen)
bool cov_fb_mask(u32 va, std::vector<u8> &out) {
    std::lock_guard<std::mutex> lk(g_cov_mtx);
    auto it = g_fbmask.find(va);
    if (it == g_fbmask.end()) return false;
    out = it->second;
    return true;
}
