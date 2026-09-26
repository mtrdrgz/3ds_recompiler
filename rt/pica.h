// PICA200 state. Register numbers follow 3dbrew "GPU/Internal Registers".
#pragma once
#include "common.h"
#include <cmath>

struct Vec4 { float x, y, z, w; float &operator[](int i) { return (&x)[i]; } float operator[](int i) const { return (&x)[i]; } };

struct DSrc { u8 reg, sel[4]; bool neg; };
struct DIns {                  // pre-decoded shader instruction
    u8 op, dst, mask, idx, rel;          // rel: which source (1..3) is address-register relative
    DSrc s[3];
    u16 dest; u8 num, condop, refx, refy, bool_id, int_id, cmpx, cmpy;
};
struct ShaderUnitSetup {       // one of VS / GS
    u32 code[4096];
    u32 swizzle[128];
    DIns dec[4096];
    bool dec_dirty = true;
    Vec4 f[96];
    u32 b;                     // bool uniforms
    u8 i[4][4];                // int uniforms (x,y,z,w)
    u32 f_index; bool f32mode; u32 f_buf[4]; u32 f_count;
    u32 code_off, swz_off;
};

// rasterizer-level vertex (after VS + output mapping)
struct OutVertex {
    Vec4 pos;
    Vec4 quat;
    Vec4 color;
    float tc0[2], tc1[2], tc0w, pad0;
    float view[3], pad1;
    float tc2[2];
    // derived
    float sx, sy, sz, invw;
};

struct Pica {
    u32 regs[0x400];
    u64 draws;
    ShaderUnitSetup vs, gs;
    Vec4 fixed_attr[16];
    u32 fixed_buf[3]; u32 fixed_count; u32 fixed_index;
    Vec4 imm_attr[16]; u32 imm_count;
    // primitive assembler
    OutVertex pa_buf[2]; u32 pa_index; bool pa_strip_ready;
    u32 fog_lut[128]; u32 lighting_lut[24][256];
    u32 proctex_lut[6][128]; u32 proctex_idx;
    u32 lut_index;
};
extern Pica *g_pica;
void pica_write(u32 reg, u32 value, u32 mask);

static inline float f24(u32 raw) {
    raw &= 0xFFFFFF;
    if ((raw & 0x7FFFFF) == 0) return (raw >> 23) ? -0.0f : 0.0f;
    u32 sign = raw >> 23, exp = (raw >> 16) & 0x7F, man = raw & 0xFFFF;
    u32 bits;
    if (exp == 0x7F) bits = (sign << 31) | (0xFFu << 23) | (man << 7);
    else if (exp == 0) bits = sign << 31;
    else bits = (sign << 31) | ((exp + 64) << 23) | (man << 7);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float f16(u32 raw) {
    raw &= 0xFFFF;
    u32 sign = raw >> 15, exp = (raw >> 10) & 0x1F, man = raw & 0x3FF;
    u32 bits;
    if (exp == 0) bits = sign << 31;
    else if (exp == 0x1F) bits = (sign << 31) | (0xFFu << 23) | (man << 13);
    else bits = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float f31(u32 raw) {
    u32 sign = (raw >> 30) & 1, exp = (raw >> 23) & 0x7F, man = raw & 0x7FFFFF;
    if (!exp && !man) return 0.0f;
    u32 bits = (sign << 31) | ((exp + 64) << 23) | man;
    float f; memcpy(&f, &bits, 4); return f;
}

void shader_run(ShaderUnitSetup &su, Vec4 *input, Vec4 *output, u32 entry, bool is_gs);
void pica_draw(bool indexed);
void pica_imm_vertex();
