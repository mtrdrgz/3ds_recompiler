// Pixel format helpers shared by the transfer engines, the rasterizer and
// the display. Colors are passed around as 0xAABBGGRR.
#pragma once
#include "common.h"
#include <algorithm>

static inline u32 morton8(u32 x, u32 y) {
    return (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
}
// offset of pixel (x,y) in an 8x8-tiled surface of width w
static inline u32 tiled_offset(u32 x, u32 y, u32 w, u32 bpp) {
    return (((y & ~7u) * w) + ((x & ~7u) * 8) + morton8(x & 7, y & 7)) * bpp;
}
static inline u32 fb_bpp(u32 fmt) {
    switch (fmt) { case 0: return 4; case 1: return 3; default: return 2; }
}
static inline u32 c5to8(u32 v) { return (v << 3) | (v >> 2); }
static inline u32 c6to8(u32 v) { return (v << 2) | (v >> 4); }
static inline u32 c4to8(u32 v) { return v * 17; }

static inline u32 fb_decode(const u8 *p, u32 fmt) {
    switch (fmt) {
    case 0: return (u32)p[3] | ((u32)p[2] << 8) | ((u32)p[1] << 16) | ((u32)p[0] << 24);
    case 1: return (u32)p[2] | ((u32)p[1] << 8) | ((u32)p[0] << 16) | 0xFF000000u;
    case 2: { u16 v = p[0] | (p[1] << 8); return c5to8(v >> 11) | (c6to8((v >> 5) & 63) << 8) | (c5to8(v & 31) << 16) | 0xFF000000u; }
    case 3: { u16 v = p[0] | (p[1] << 8); return c5to8(v >> 11) | (c5to8((v >> 6) & 31) << 8) | (c5to8((v >> 1) & 31) << 16) | ((v & 1) ? 0xFF000000u : 0); }
    case 4: { u16 v = p[0] | (p[1] << 8); return c4to8(v >> 12) | (c4to8((v >> 8) & 15) << 8) | (c4to8((v >> 4) & 15) << 16) | (c4to8(v & 15) << 24); }
    }
    return 0;
}
static inline void fb_encode(u8 *p, u32 fmt, u32 c) {
    u32 r = c & 0xFF, g = (c >> 8) & 0xFF, b = (c >> 16) & 0xFF, a = c >> 24;
    switch (fmt) {
    case 0: p[0] = a; p[1] = b; p[2] = g; p[3] = r; break;
    case 1: p[0] = b; p[1] = g; p[2] = r; break;
    case 2: { u16 v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); p[0] = v; p[1] = v >> 8; break; }
    case 3: { u16 v = ((r >> 3) << 11) | ((g >> 3) << 6) | ((b >> 3) << 1) | (a >> 7); p[0] = v; p[1] = v >> 8; break; }
    case 4: { u16 v = ((r >> 4) << 12) | ((g >> 4) << 8) | ((b >> 4) << 4) | (a >> 4); p[0] = v; p[1] = v >> 8; break; }
    }
}
