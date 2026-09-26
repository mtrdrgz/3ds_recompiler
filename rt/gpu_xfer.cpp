// GPU transfer engines: memory fill (PSC), display transfer / texture copy (PPF).
#include "gpu.h"
#include "mem.h"
#include "pica_fmt.h"
#include "hwr.h"
void pica_flush();
void cov_on_display_transfer(u32 in_va, u32 in_w, u32 in_h, u32 out_va, u32 out_w, u32 out_h, bool flip, u32 hs, u32 vs);
void cov_on_fill(u32 start, u32 end);

void gpu_memory_fill(u32 start, u32 end, u32 value, u32 control) {
    pica_flush();
    if (!mem_is_mapped(start)) { LOG("[gpu] fill to unmapped %08x", start); return; }
    cov_on_fill(start, end);
    u32 width = (control >> 8) & 3;
    if (width == 0) {
        for (u32 a = start; a + 2 <= end; a += 2) wr16(a, (u16)value);
    } else if (width == 1) {
        for (u32 a = start; a + 3 <= end; a += 3) { wr8(a, value & 0xFF); wr8(a + 1, (value >> 8) & 0xFF); wr8(a + 2, (value >> 16) & 0xFF); }
    } else {
        for (u32 a = start; a + 4 <= end; a += 4) wr32(a, value);
    }
    hwr_fill(start, end, value, width);
}

void gpu_display_transfer(u32 in_va, u32 out_va, u32 in_dim, u32 out_dim, u32 flags) {
    pica_flush();
    if (!mem_is_mapped(in_va) || !mem_is_mapped(out_va)) { LOG("[gpu] xfer unmapped %08x->%08x", in_va, out_va); return; }
    if (flags & 8) {  // texture copy mode handled by caller normally
        return;
    }
    u32 in_w = in_dim & 0xFFFF, in_h = in_dim >> 16;
    u32 out_w = out_dim & 0xFFFF, out_h = out_dim >> 16;
    u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7;
    bool flip = flags & 1, to_tiled = flags & 2;
    u32 scale = (flags >> 24) & 3;
    u32 hs = scale != 0 ? 1 : 0, vs = scale == 2 ? 1 : 0;
    if (scale) { out_w = in_w >> hs; out_h = in_h >> vs; }
    u32 ib = fb_bpp(in_fmt), ob = fb_bpp(out_fmt);
    if (flags & 0x20) {  // raw copy, no conversion
        out_fmt = in_fmt; ob = ib;
    }
    if (!to_tiled) cov_on_display_transfer(in_va, in_w, in_h, out_va, out_w, out_h, flip, hs, vs);
    u8 *src = gp(in_va), *dst = gp(out_va);
    for (u32 y = 0; y < out_h; y++) {
        for (u32 x = 0; x < out_w; x++) {
            u32 ix = x << hs, iy = y << vs;
            u32 oy = flip ? out_h - 1 - y : y;
            u32 so, doff;
            if (!to_tiled) {
                so = tiled_offset(ix, iy, in_w, ib);
                doff = (x + oy * out_w) * ob;
            } else {
                so = (ix + iy * in_w) * ib;
                doff = tiled_offset(x, oy, out_w, ob);
            }
            u32 c;
            if (scale) {
                u32 r = 0, g = 0, b = 0, a = 0, n = 0;
                for (u32 dy = 0; dy <= vs; dy++) for (u32 dx = 0; dx <= hs; dx++) {
                    u32 s2 = to_tiled ? ((ix + dx) + (iy + dy) * in_w) * ib : tiled_offset(ix + dx, iy + dy, in_w, ib);
                    u32 p = fb_decode(src + s2, in_fmt);
                    r += p & 0xFF; g += (p >> 8) & 0xFF; b += (p >> 16) & 0xFF; a += p >> 24; n++;
                }
                c = (r / n) | ((g / n) << 8) | ((b / n) << 16) | ((a / n) << 24);
            } else {
                c = fb_decode(src + so, in_fmt);
            }
            fb_encode(dst + doff, out_fmt, c);
        }
    }
    hwr_display_transfer(in_va, out_va, in_w, in_h, out_w, out_h, in_fmt, out_fmt, flip, to_tiled, hs, vs);
}

void gpu_texture_copy(u32 in_va, u32 out_va, u32 size, u32 in_wg, u32 out_wg, u32 flags) {
    pica_flush();
    if (!mem_is_mapped(in_va) || !mem_is_mapped(out_va)) return;
    u32 iw = (in_wg & 0xFFFF) * 16, ig = (in_wg >> 16) * 16;
    u32 ow = (out_wg & 0xFFFF) * 16, og = (out_wg >> 16) * 16;
    if (!iw) iw = size, ig = 0;
    if (!ow) ow = size, og = 0;
    u32 si = 0, di = 0, sp = 0, dp = 0, left = size;
    u8 *s = gp(in_va), *d = gp(out_va);
    while (left) {
        u32 n = std::min({iw - sp, ow - dp, left});
        memmove(d + di, s + si, n);
        si += n; di += n; sp += n; dp += n; left -= n;
        if (sp == iw) { sp = 0; si += ig; }
        if (dp == ow) { dp = 0; di += og; }
    }
    hwr_texture_copy(in_va, out_va, size, in_wg, out_wg);
}
