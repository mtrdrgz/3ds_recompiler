// High-resolution text for the GPU renderer (see hwr_fonts.h).
#include "hwr_fonts.h"
#include <cmath>
#include <algorithm>
#include <map>
#ifdef R3DS_HAVE_FONTS
#else
const HwrFontTable g_hwr_fonts[1] = {{0, 0, 0, nullptr}};
const int g_hwr_font_count = 0;
#endif

static bool blank_code(u16 c) { return c == 0x0a || c == 0x20 || c == 0xa0 || c == 0x3000; }

// ink of cell k: any texel with a visible alpha. rgba rows are t = 0 at the
// bottom; cells run along t (cell 0 at t = 0).
static bool cell_ink(const u32 *rgba, u32 w, u32 cw, u32 ch, u32 cols, u32 k) {
    u32 x0 = (k % cols) * cw, t0 = (k / cols) * ch;
    for (u32 t = t0; t < t0 + ch; t++)
        for (u32 x = x0; x < x0 + cw; x++)
            if ((rgba[t * w + x] >> 24) > 34) return true;
    return false;
}

bool hwr_font_detect(const u32 *rgba, u32 w, u32 h, u32 fmt, HwrSheet &out) {
    if ((fmt != 11 && fmt != 8) || w < 128 || h < 128) return false;
    double best = 0; HwrSheet pick;
    for (int f = 0; f < g_hwr_font_count; f++) {
        const HwrFontTable &F = g_hwr_fonts[f];
        if (F.n < 64) continue;
        u32 cw = (F.w + 1) & ~1u, ch = F.h + 1, cols = w / cw, rows = h / ch;
        if (!cols || !rows) continue;
        u32 cells = std::min<u32>(F.n, cols * rows);
        // the first cells hold '\n', '\n', ' ' (blank) and then printable characters
        if (cell_ink(rgba, w, cw, ch, cols, 0) || cell_ink(rgba, w, cw, ch, cols, 2) || !cell_ink(rgba, w, cw, ch, cols, 3)) continue;
        u32 agree = 0, checked = 0;
        for (u32 k = 0; k < cells; k++) {
            bool want = !blank_code(F.codes[k]);
            agree += cell_ink(rgba, w, cw, ch, cols, k) == want;
            checked++;
        }
        double score = (double)agree / checked;
        // among equally good fits, the font that explains more cells
        double rank = score + cells * 1e-7;
        if (score >= 0.97 && checked >= 64 && rank > best) {
            best = rank;
            pick.cw = cw; pick.ch = ch; pick.cols = cols; pick.cells = cells; pick.font = &F;
        }
    }
    if (!pick.font) return false;
    out = pick;
    return true;
}

// ------------------------------------------------------------ glyph rasterisers
// hwr_raster_glyph: draw one character into a w x h 8-bit alpha buffer (row 0
// at the top), pen at (x, baseline) in pixels. false: no vector font here.
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
EM_JS(int, js_raster_glyph, (int code, float px, int w, int h, float x, float base, unsigned char *out, int plain), {
    if (!self.__glyphCache) {
        const c = new OffscreenCanvas(64, 64);
        self.__glyphCache = { c, g: c.getContext('2d', { willReadFrequently: true }) };
    }
    const G = self.__glyphCache;
    if (G.c.width < w || G.c.height < h) { G.c.width = Math.max(G.c.width, w); G.c.height = Math.max(G.c.height, h); G.g = G.c.getContext('2d', { willReadFrequently: true }); }
    const g = G.g;
    g.clearRect(0, 0, w, h);
    g.font = plain ? '700 ' + px + 'px Arial, Helvetica, "Liberation Sans", sans-serif'
                   : '700 ' + px + 'px "M PLUS Rounded 1c", "Arial Rounded MT Bold", "Varela Round", "Nunito", system-ui, sans-serif';
    g.fillStyle = '#fff';
    g.textBaseline = 'alphabetic';
    g.fillText(String.fromCharCode(code), x, base);
    const d = g.getImageData(0, 0, w, h).data;
    const mem = new Uint8Array(wasmMemory.buffer, out, w * h);
    for (let i = 0; i < w * h; i++) mem[i] = d[i * 4 + 3];
    return 1;
});
static bool hwr_raster_glyph(u16 code, float px, int w, int h, float x, float base, u8 *out, bool plain = false) {
    return js_raster_glyph(code, px, w, h, x, base, out, plain ? 1 : 0) != 0;
}
#elif defined(__APPLE__)
#include <CoreText/CoreText.h>
#include <CoreGraphics/CoreGraphics.h>
static CTFontRef mac_font(float px, bool plain) {
    static std::map<int, CTFontRef> cache;
    int key = (int)lroundf(px * 8) * 2 + plain;
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    // rounded like the game's own face; CoreText falls back per character
    CTFontRef f = CTFontCreateWithName(plain ? CFSTR("Arial-BoldMT") : CFSTR("ArialRoundedMTBold"), px, nullptr);
    if (!f) f = CTFontCreateWithName(CFSTR("HiraMaruProN-W4"), px, nullptr);
    cache[key] = f;
    return f;
}
static bool hwr_raster_glyph(u16 code, float px, int w, int h, float x, float base, u8 *out, bool plain = false) {
    memset(out, 0, (size_t)w * h);
    CTFontRef font = mac_font(px, plain);
    if (!font) return false;
    CGContextRef ctx = CGBitmapContextCreate(out, w, h, 8, w, nullptr, (CGBitmapInfo)kCGImageAlphaOnly);
    if (!ctx) return false;
    CGContextSetShouldAntialias(ctx, true);
    CGContextSetAllowsFontSmoothing(ctx, false);
    CGContextSetRGBFillColor(ctx, 1, 1, 1, 1);
    UniChar ch = code;
    CFStringRef str = CFStringCreateWithCharacters(nullptr, &ch, 1);
    const void *keys[] = {kCTFontAttributeName, kCTForegroundColorFromContextAttributeName};
    const void *vals[] = {font, kCFBooleanTrue};
    CFDictionaryRef attrs = CFDictionaryCreate(nullptr, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFAttributedStringRef as = CFAttributedStringCreate(nullptr, str, attrs);
    CTLineRef line = CTLineCreateWithAttributedString(as);
    CGContextSetTextPosition(ctx, x, h - base);   // CoreGraphics: y up
    CTLineDraw(line, ctx);
    CFRelease(line); CFRelease(as); CFRelease(attrs); CFRelease(str);
    CGContextRelease(ctx);
    return true;
}
#else
static bool hwr_raster_glyph(u16, float, int, int, float, float, u8 *, bool = false) { return false; }
#endif

bool hwr_raster_glyph_pub(u16 code, float px, int w, int h, float x, float base, u8 *out, bool plain) {
    return hwr_raster_glyph(code, px, w, h, x, base, out, plain);
}

// ink bounds of an alpha buffer
static bool ink_box(const u8 *a, int w, int h, int &x0, int &y0, int &x1, int &y1) {
    x0 = w; y0 = h; x1 = -1; y1 = -1;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (a[y * w + x] > 40) { x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
    return x1 >= 0;
}

// Redraw a recognised sheet at scale s. src: the 1x texture (RGBA, rows in
// memory order: row 0 = the last texture row t). out: (w*s) x (h*s), same order.
// Cells whose character has no vector glyph keep the original, magnified.
bool hwr_font_build(const u32 *src, u32 w, u32 h, const HwrSheet &S, u32 s, std::vector<u32> &out) {
    const u32 W = w * s, H = h * s;
    out.assign((size_t)W * H, 0);
    for (u32 y = 0; y < H; y++)
        for (u32 x = 0; x < W; x++) out[(size_t)y * W + x] = src[(y / s) * w + x / s];
    const HwrFontTable &F = *S.font;
    // original glyph boxes, in cell coordinates with y down (t grows downward in a cell)
    auto alpha1 = [&](u32 x, u32 t) { return src[(h - 1 - t) * w + x] >> 24; };   // t: texture row (0 = bottom)
    auto box1 = [&](u32 k, int &x0, int &y0, int &x1, int &y1) {
        u32 cx = (k % S.cols) * S.cw, ct = (k / S.cols) * S.ch;
        x0 = S.cw; y0 = S.ch; x1 = -1; y1 = -1;
        for (u32 yy = 0; yy < S.ch; yy++)
            for (u32 xx = 0; xx < S.cw; xx++)
                if (alpha1(cx + xx, ct + yy) > 34) { x0 = std::min<int>(x0, xx); x1 = std::max<int>(x1, xx); y0 = std::min<int>(y0, yy); y1 = std::max<int>(y1, yy); }
        return x1 >= 0;
    };
    // size and baseline from a reference capital (cap height, bottom = baseline)
    int ref = -1;
    for (u16 c : {u16('H'), u16('E'), u16('I'), u16('A')})
        for (u32 k = 0; k < S.cells && ref < 0; k++) if (F.codes[k] == c) ref = (int)k;
    if (ref < 0) return false;
    int rx0, ry0, rx1, ry1;
    if (!box1(ref, rx0, ry0, rx1, ry1)) return false;
    const float cap = (float)(ry1 - ry0 + 1) * s, base = (float)(ry1 + 1) * s;
    // calibrate the vector font: pixel size whose 'H' is cap pixels tall
    std::vector<u8> tmp;
    const int TW = 256, TH = 256;
    tmp.resize(TW * TH);
    if (!hwr_raster_glyph(F.codes[ref], 100.0f, TW, TH, 20.0f, 180.0f, tmp.data())) return false;
    int ax0, ay0, ax1, ay1;
    if (!ink_box(tmp.data(), TW, TH, ax0, ay0, ax1, ay1)) return false;
    const float px = 100.0f * cap / (float)(ay1 - ay0 + 1);
    // per glyph: pen on the original baseline, centred on the original ink
    const int cw = (int)(S.cw * s), chh = (int)(S.ch * s);
    const int GW = cw * 3, GH = chh * 2;
    tmp.assign((size_t)GW * GH, 0);
    u32 drawn = 0;
    for (u32 k = 0; k < S.cells; k++) {
        u16 code = F.codes[k];
        if (blank_code(code)) continue;
        int ox0, oy0, ox1, oy1;
        if (!box1(k, ox0, oy0, ox1, oy1)) continue;
        const float pen_base = base + (float)chh / 2;   // tmp has half a cell of room above
        if (!hwr_raster_glyph(code, px, GW, GH, (float)cw, pen_base, tmp.data())) continue;
        int gx0, gy0, gx1, gy1;
        if (!ink_box(tmp.data(), GW, GH, gx0, gy0, gx1, gy1)) continue;
        // horizontal: centre on the original ink; vertical: baseline as rendered
        const float ocx = (ox0 + ox1 + 1) * 0.5f * s, gcx = (gx0 + gx1 + 1) * 0.5f;
        const int dx = (int)lroundf(ocx - gcx), dy = -chh / 2;
        const u32 cx = (k % S.cols) * S.cw * s, ct = (k / S.cols) * S.ch * s;   // cell origin, t-down texel units
        for (int yy = 0; yy < chh; yy++)
            for (int xx = 0; xx < cw; xx++) {
                int sx = xx - dx, sy = yy - dy;
                u32 a = (sx >= 0 && sx < GW && sy >= 0 && sy < GH) ? tmp[sy * GW + sx] : 0;
                u32 t = ct + yy;                  // texture row from the bottom
                out[(size_t)(H - 1 - t) * W + cx + xx] = a << 24;
            }
        drawn++;
    }
    return drawn > 0;
}
