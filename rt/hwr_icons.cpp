// Button icons as keyboard keys, for the GPU renderer.
//
// The game shows its button prompts as small textures: a white circle (A B X
// Y) or rounded square (L R) with a dark outline and a black letter, or a
// white cross (the D-pad). hwr_icon_build() recognises those shapes in any
// small texture the executor uploads -- the letter is read by comparing it
// with the same letter drawn by the vector font -- and redraws the texture at
// a higher resolution with keyboard keycaps showing the key each button is on
// (A -> K, B -> J, X -> I, Y -> U, L -> Q, R -> E, D-pad -> arrows).
#include "common.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

bool hwr_raster_glyph_pub(u16 code, float px, int w, int h, float x, float base, u8 *out, bool plain = false);   // hwr_fonts.cpp

namespace {
struct Img {   // RGBA8, row 0 at the top (memory order of an upright icon)
    u32 w, h; const u32 *p;
    u32 at(int x, int y) const { return p[y * w + x]; }
};
inline int luma(u32 c) { return ((c & 0xFF) * 3 + ((c >> 8) & 0xFF) * 6 + ((c >> 16) & 0xFF)) / 10; }
inline u32 alpha(u32 c) { return c >> 24; }

struct Comp { int x0, y0, x1, y1, n; std::vector<int> px; };

// 4-connected components of opaque texels
std::vector<Comp> components(const Img &im) {
    std::vector<int> lab(im.w * im.h, -1);
    std::vector<Comp> out;
    for (u32 y = 0; y < im.h; y++)
        for (u32 x = 0; x < im.w; x++) {
            if (lab[y * im.w + x] >= 0 || alpha(im.at(x, y)) < 128) continue;
            Comp c{(int)x, (int)y, (int)x, (int)y, 0, {}};
            std::vector<int> st{(int)(y * im.w + x)};
            lab[y * im.w + x] = (int)out.size();
            while (!st.empty()) {
                int i = st.back(); st.pop_back();
                int cx = i % im.w, cy = i / im.w;
                c.px.push_back(i); c.n++;
                c.x0 = std::min(c.x0, cx); c.x1 = std::max(c.x1, cx); c.y0 = std::min(c.y0, cy); c.y1 = std::max(c.y1, cy);
                const int d[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (auto &k : d) {
                    int nx = cx + k[0], ny = cy + k[1];
                    if (nx < 0 || ny < 0 || nx >= (int)im.w || ny >= (int)im.h) continue;
                    int j = ny * im.w + nx;
                    if (lab[j] < 0 && alpha(im.p[j]) >= 128) { lab[j] = lab[i]; st.push_back(j); }
                }
            }
            out.push_back(std::move(c));
        }
    return out;
}

// binary 16x16 mask of the ink inside [x0,x1] x [y0,y1] of a predicate
constexpr int M = 16;
template <class F> bool norm_mask(int x0, int y0, int x1, int y1, F ink, std::vector<u8> &m) {
    int bx0 = x1, by0 = y1, bx1 = x0 - 1, by1 = y0 - 1;
    for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) if (ink(x, y)) { bx0 = std::min(bx0, x); bx1 = std::max(bx1, x); by0 = std::min(by0, y); by1 = std::max(by1, y); }
    if (bx1 < bx0) return false;
    // keep the aspect: square box around the ink
    int s = std::max(bx1 - bx0, by1 - by0) + 1;
    float cx = (bx0 + bx1 + 1) * 0.5f, cy = (by0 + by1 + 1) * 0.5f;
    m.assign(M * M, 0);
    for (int j = 0; j < M; j++)
        for (int i = 0; i < M; i++) {
            int hit = 0, tot = 0;
            for (int sy = 0; sy < 3; sy++) for (int sx = 0; sx < 3; sx++) {
                float fx = cx - s * 0.5f + (i + (sx + 0.5f) / 3) * s / M, fy = cy - s * 0.5f + (j + (sy + 0.5f) / 3) * s / M;
                int xi = (int)floorf(fx), yi = (int)floorf(fy);
                tot++;
                if (xi >= x0 && xi <= x1 && yi >= y0 && yi <= y1 && ink(xi, yi)) hit++;
            }
            m[j * M + i] = hit * 2 >= tot;
        }
    return true;
}
// exact overlap (intersection over union) of two masks
float iou(const std::vector<u8> &a, const std::vector<u8> &b) {
    int i = 0, u = 0;
    for (int k = 0; k < M * M; k++) { i += a[k] && b[k]; u += a[k] || b[k]; }
    return u ? (float)i / u : 0;
}
// overlap tolerant to a one-cell misalignment: the share of each mask's ink
// that the other one, dilated by a cell, covers
float near(const std::vector<u8> &a, const std::vector<u8> &b) {
    auto dil = [](const std::vector<u8> &m) {
        std::vector<u8> d(M * M, 0);
        for (int y = 0; y < M; y++) for (int x = 0; x < M; x++) if (m[y * M + x])
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int X = x + dx, Y = y + dy;
                if (X >= 0 && Y >= 0 && X < M && Y < M) d[Y * M + X] = 1;
            }
        return d;
    };
    std::vector<u8> da = dil(a), db = dil(b);
    int na = 0, nb = 0, ca = 0, cb = 0;
    for (int k = 0; k < M * M; k++) { na += a[k]; nb += b[k]; ca += a[k] && db[k]; cb += b[k] && da[k]; }
    return na && nb ? std::min((float)ca / na, (float)cb / nb) : 0;
}

// the vector font's letter, as a normalised mask
const std::vector<u8> *letter_mask(u16 code) {
    static std::map<u16, std::vector<u8>> cache;
    auto it = cache.find(code);
    if (it != cache.end()) return it->second.empty() ? nullptr : &it->second;
    std::vector<u8> a(160 * 160), m;
    if (hwr_raster_glyph_pub(code, 100, 160, 160, 30, 125, a.data(), true))
        norm_mask(0, 0, 159, 159, [&](int x, int y) { return a[y * 160 + x] > 110; }, m);
    cache[code] = m;
    return m.empty() ? nullptr : &cache[code];
}

enum Key { K_NONE, K_LETTER, K_ARROWS };
struct Found { int x0, y0, x1, y1; bool round; Key kind; u16 label; };

// what button is component c? letters are dark ink inside a light, outlined cap
bool classify(const Img &im, const Comp &c, Found &f) {
    int bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
    if (bw < 9 || bh < 9 || bw > 3 * bh / 2 || bh > 3 * bw / 2) return false;
    int light = 0, dark = 0;
    for (int i : c.px) { int l = luma(im.p[i]); light += l > 190; dark += l < 90; }
    if (light * 10 < c.n * 3 || dark * 20 < c.n) return false;   // a light cap with a dark outline / letter
    float fill = (float)c.n / (bw * bh);
    f = {c.x0, c.y0, c.x1, c.y1, fill < 0.86f, K_NONE, 0};
    // the D-pad cross: arms through the middle, empty quarter points (a
    // circle or a rounded square is filled there)
    auto op = [&](int x, int y) { return alpha(im.at(x, y)) >= 128; };
    int mxc = (c.x0 + c.x1) / 2, myc = (c.y0 + c.y1) / 2;
    bool quarters_empty = !op(c.x0 + bw / 4, c.y0 + bh / 4) && !op(c.x1 - bw / 4, c.y0 + bh / 4) &&
                          !op(c.x0 + bw / 4, c.y1 - bh / 4) && !op(c.x1 - bw / 4, c.y1 - bh / 4);
    bool arms = op(c.x0 + 1, myc) && op(c.x1 - 1, myc) && op(mxc, c.y0 + 1) && op(mxc, c.y1 - 1) && op(mxc, myc);
    if (quarters_empty && arms && fill > 0.3f && fill < 0.75f) { f.kind = K_ARROWS; f.round = false; return true; }
    // the letter: dark texels away from the cap's outline
    int mx = std::max(2, bw * 22 / 100), my = std::max(2, bh * 22 / 100);
    auto inner_dark = [&](int x, int y) { u32 p = im.at(x, y); return alpha(p) >= 128 && luma(p) < 110; };
    std::vector<u8> m;
    if (!norm_mask(c.x0 + mx, c.y0 + my, c.x1 - mx, c.y1 - my, inner_dark, m)) return false;
    static const u16 cand[] = {'A', 'B', 'X', 'Y', 'L', 'R'};
    // rank by exact overlap (tells letters apart), accept by tolerant overlap (fonts differ)
    float best = 0, second = 0, fit = 0; u16 which = 0;
    for (u16 ch : cand) {
        const std::vector<u8> *t = letter_mask(ch);
        if (!t) continue;
        float sc = iou(m, *t);
        if (sc > best) { second = best; best = sc; which = ch; fit = near(m, *t); } else if (sc > second) second = sc;
    }
    if (fit < 0.8f || best - second < 0.06f) {
        static int logged = 0;
        if (logged++ < 6) LOG("[hwr] button-like icon not read: best %c %.2f (fit %.2f), next %.2f", which ? (char)which : '?', best, fit, second);
        return false;
    }
    static const std::map<u16, u16> key = {{'A', 'K'}, {'B', 'J'}, {'X', 'I'}, {'Y', 'U'}, {'L', 'Q'}, {'R', 'E'}};
    f.kind = K_LETTER; f.label = key.at(which);
    return true;
}

// ---- drawing (supersampled coverage, premultiplied into RGBA8)
struct Canvas {
    int w, h; std::vector<float> r, g, b, a;
    Canvas(int W, int H) : w(W), h(H), r(W * H, 0), g(W * H, 0), b(W * H, 0), a(W * H, 0) {}
    void over(int i, float cr, float cg, float cb, float cov) {   // source-over, straight colour
        if (cov <= 0) return;
        r[i] = cr * cov + r[i] * (1 - cov); g[i] = cg * cov + g[i] * (1 - cov); b[i] = cb * cov + b[i] * (1 - cov);
        a[i] = cov + a[i] * (1 - cov);
    }
};
float rrect_sd(float px, float py, float cx, float cy, float hw, float hh, float rad) {   // signed distance to a rounded rectangle
    float qx = fabsf(px - cx) - (hw - rad), qy = fabsf(py - cy) - (hh - rad);
    float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f);
    return sqrtf(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.0f) - rad;
}
void fill_rrect(Canvas &cv, float cx, float cy, float hw, float hh, float rad, float cr, float cg, float cb) {
    for (int y = 0; y < cv.h; y++)
        for (int x = 0; x < cv.w; x++) {
            float d = rrect_sd(x + 0.5f, y + 0.5f, cx, cy, hw, hh, rad);
            cv.over(y * cv.w + x, cr, cg, cb, std::clamp(0.5f - d, 0.0f, 1.0f));
        }
}
void fill_tri(Canvas &cv, float x0, float y0, float x1, float y1, float x2, float y2, float c) {
    auto edge = [](float ax, float ay, float bx, float by, float px, float py) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); };
    float area = edge(x0, y0, x1, y1, x2, y2);
    for (int y = 0; y < cv.h; y++)
        for (int x = 0; x < cv.w; x++) {
            int in = 0;
            for (int s = 0; s < 4; s++) {
                float px = x + 0.25f + (s & 1) * 0.5f, py = y + 0.25f + (s >> 1) * 0.5f;
                float e0 = edge(x1, y1, x2, y2, px, py), e1 = edge(x2, y2, x0, y0, px, py), e2 = edge(x0, y0, x1, y1, px, py);
                if (area > 0 ? (e0 >= 0 && e1 >= 0 && e2 >= 0) : (e0 <= 0 && e1 <= 0 && e2 <= 0)) in++;
            }
            cv.over(y * cv.w + x, c, c, c, in / 4.0f);
        }
}
// a keycap: dark outline, light top, a darker lip at the bottom like a real key
void keycap(Canvas &cv, float x0, float y0, float x1, float y1) {
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hw = (x1 - x0) / 2, hh = (y1 - y0) / 2, s = std::min(hw, hh) * 2;
    float rad = s * 0.24f, ol = std::max(1.0f, s * 0.08f), lip = s * 0.09f;
    fill_rrect(cv, cx, cy, hw, hh, rad, 0.12f, 0.12f, 0.13f);
    fill_rrect(cv, cx, cy - lip * 0.5f, hw - ol, hh - ol - lip * 0.5f, std::max(0.0f, rad - ol), 0.62f, 0.63f, 0.66f);
    fill_rrect(cv, cx, cy - lip, hw - ol, hh - ol - lip, std::max(0.0f, rad - ol), 0.98f, 0.98f, 0.97f);
}
}   // namespace

// icon: the uploaded texture (RGBA8, memory row order = upright). s: scale.
// out: (w*s) x (h*s) RGBA8 when at least one button was recognised.
bool hwr_icon_build(const u32 *px, u32 w, u32 h, u32 s, std::vector<u32> &out) {
    if (w > 64 || h > 64 || w < 12 || h < 12) return false;
    Img im{w, h, px};
    std::vector<Found> keys;
    for (const Comp &c : components(im)) {
        if (c.n < 40) continue;
        Found f;
        if (classify(im, c, f)) keys.push_back(f);
    }
    if (keys.empty()) return false;
    Canvas cv(w * s, h * s);
    // everything that is not a recognised button stays as it was (magnified)
    for (u32 y = 0; y < h * s; y++)
        for (u32 x = 0; x < w * s; x++) {
            u32 p = px[(y / s) * w + x / s];
            bool in_key = false;
            for (auto &k : keys) in_key |= (int)(x / s) >= k.x0 && (int)(x / s) <= k.x1 && (int)(y / s) >= k.y0 && (int)(y / s) <= k.y1;
            if (in_key) continue;
            float a = (p >> 24) / 255.0f;
            cv.over(y * cv.w + x, (p & 0xFF) / 255.0f, ((p >> 8) & 0xFF) / 255.0f, ((p >> 16) & 0xFF) / 255.0f, a);
        }
    std::vector<u8> glyph;
    for (auto &k : keys) {
        float x0 = k.x0 * (float)s, y0 = k.y0 * (float)s, x1 = (k.x1 + 1) * (float)s, y1 = (k.y1 + 1) * (float)s;
        float sz = std::min(x1 - x0, y1 - y0);
        if (k.kind == K_LETTER) {
            keycap(cv, x0, y0, x1, y1);
            // the key's letter, bold, centred on the cap (above its lip)
            int gw = (int)(x1 - x0), gh = (int)(y1 - y0);
            glyph.assign((size_t)gw * gh, 0);
            float px_ = sz * 0.62f;
            if (hwr_raster_glyph_pub(k.label, px_, gw, gh, 0, gh * 0.5f + px_ * 0.36f - sz * 0.05f, glyph.data())) {
                int gx0 = gw, gx1 = -1;
                for (int y = 0; y < gh; y++) for (int x = 0; x < gw; x++) if (glyph[y * gw + x] > 60) { gx0 = std::min(gx0, x); gx1 = std::max(gx1, x); }
                int dx = gx1 >= 0 ? (int)lroundf(gw * 0.5f - (gx0 + gx1 + 1) * 0.5f) : 0;
                for (int y = 0; y < gh; y++)
                    for (int x = 0; x < gw; x++) {
                        int sx = x - dx;
                        float cov = (sx >= 0 && sx < gw) ? glyph[y * gw + sx] / 255.0f : 0;
                        int X = (int)x0 + x, Y = (int)y0 + y;
                        if (X < cv.w && Y < cv.h) cv.over(Y * cv.w + X, 0.08f, 0.08f, 0.1f, cov);
                    }
            }
        } else {   // D-pad: the arrow keys, laid out as on a keyboard (inverted T)
            float gap = sz * 0.02f, c = (sz - 2 * gap) / 3;
            float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
            struct { float kx, ky; int dir; } a[4] = {{cx, cy - (c + gap) / 2, 0}, {cx - c - gap, cy + (c + gap) / 2, 1},
                                                      {cx + c + gap, cy + (c + gap) / 2, 2}, {cx, cy + (c + gap) / 2, 3}};
            for (auto &q : a) {
                float kx = q.kx, ky = q.ky;
                keycap(cv, kx - c / 2, ky - c / 2, kx + c / 2, ky + c / 2);
                float t = c * 0.24f, ty = ky - c * 0.05f;
                if (q.dir == 0) fill_tri(cv, kx, ty - t, kx - t, ty + t * 0.7f, kx + t, ty + t * 0.7f, 0.1f);
                if (q.dir == 3) fill_tri(cv, kx, ty + t, kx + t, ty - t * 0.7f, kx - t, ty - t * 0.7f, 0.1f);
                if (q.dir == 1) fill_tri(cv, kx - t, ty, kx + t * 0.7f, ty + t, kx + t * 0.7f, ty - t, 0.1f);
                if (q.dir == 2) fill_tri(cv, kx + t, ty, kx - t * 0.7f, ty - t, kx - t * 0.7f, ty + t, 0.1f);
            }
        }
    }
    out.resize((size_t)cv.w * cv.h);
    for (int i = 0; i < cv.w * cv.h; i++) {
        auto b8 = [](float v) { return (u32)std::clamp((int)lroundf(v * 255), 0, 255); };
        out[i] = b8(cv.r[i]) | (b8(cv.g[i]) << 8) | (b8(cv.b[i]) << 16) | (b8(cv.a[i]) << 24);
    }
    static int logged = 0;
    if (logged++ < 12) {
        std::string what;
        for (auto &k : keys) { what += k.kind == K_ARROWS ? "arrows " : std::string(1, (char)k.label) + " "; }
        LOG("[hwr] button icon %ux%u -> keys %s", w, h, what.c_str());
    }
    return true;
}
