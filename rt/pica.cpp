// PICA200 command processor front-end (register file). The rasterizer
// lives in pica_raster.cpp.
#include "gpu.h"
#include "mem.h"
#include "pica.h"
#include "pica_snap.h"

Pica *g_pica;

void pica_init() { g_pica = new Pica(); memset((void *)g_pica, 0, sizeof(Pica)); g_pica->vs.dec_dirty = g_pica->gs.dec_dirty = true; }

extern u32 g_cmdbuf_jump_pa, g_cmdbuf_jump_size;
void pica_flush();

void pica_run_cmdlist(u32 addr, u32 size, bool is_va) {
    if (!mem_is_mapped(addr)) { LOG("[pica] cmdlist unmapped %08x", addr); return; }
    u32 p = addr, end = addr + size;
    g_cmdbuf_jump_pa = 0;
    int jumps = 0;
    while (p + 8 <= end) {
        if (g_cmdbuf_jump_pa) {
            u32 va = pa_to_va(g_cmdbuf_jump_pa);
            if (!va || ++jumps > 10000) break;
            p = va; end = va + g_cmdbuf_jump_size;
            g_cmdbuf_jump_pa = 0;
            continue;
        }
        u32 param = gpu_rd32(p), hdr = gpu_rd32(p + 4);
        p += 8;
        u32 reg = hdr & 0xFFFF, mask = (hdr >> 16) & 0xF, extra = (hdr >> 20) & 0xFF;
        bool seq = hdr >> 31;
        pica_write(reg, param, mask);
        for (u32 i = 0; i < extra; i++) {
            if (seq) reg++;
            pica_write(reg, gpu_rd32(p), mask);
            p += 4;
        }
        if (extra & 1) p += 4;   // keep 8-byte alignment
    }
    pica_flush();
}

