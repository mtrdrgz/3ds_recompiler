// PICA200 shader interpreter (vertex and geometry shaders).
#include "pica.h"
#if defined(__aarch64__) && !defined(__EMSCRIPTEN__)
#include <sse2neon.h>   // SSE2 intrinsics on NEON (brew install sse2neon / apt install sse2neon)
#else
#include <emmintrin.h>
#endif
#include <functional>

static inline float pmul(float a, float b) {
    float r = a * b;
    if (std::isnan(r) && !std::isnan(a) && !std::isnan(b)) return 0.0f;
    return r;
}

struct ShaderState {
    Vec4 in[16], tmp[16], out[16];
    s32 a0[2], aL;
    bool cond[2];
};

std::function<void(Vec4 *out, int vtx_id, bool prim_emit, bool winding)> g_gs_emit;

static inline Vec4 get_src(ShaderState &s, ShaderUnitSetup &su, u32 reg, u32 addr_idx) {
    if (reg < 0x10) return s.in[reg];
    if (reg < 0x20) return s.tmp[reg - 0x10];
    s32 off = 0;
    if (addr_idx == 1) off = s.a0[0];
    else if (addr_idx == 2) off = s.a0[1];
    else if (addr_idx == 3) off = s.aL;
    s32 idx = (s32)(reg - 0x20) + off;
    if (idx < 0 || idx >= 96) return Vec4{1, 1, 1, 1};
    return su.f[idx];
}

static inline Vec4 swz(const Vec4 &v, u32 sel, bool neg) {
    Vec4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = v[(sel >> (2 * (3 - i))) & 3];
        if (neg) r[i] = -r[i];
    }
    return r;
}

static inline void write_dst(ShaderState &s, u32 dst, u32 mask, const Vec4 &v) {
    Vec4 *d = dst < 0x10 ? &s.out[dst] : &s.tmp[(dst - 0x10) & 15];
    for (int i = 0; i < 4; i++)
        if (mask & (8 >> i)) (*d)[i] = v[i];
}

static inline bool eval_cond(ShaderState &s, u32 ins) {
    u32 op = (ins >> 22) & 3;
    bool refy = (ins >> 24) & 1, refx = (ins >> 25) & 1;
    bool x = refx == s.cond[0], y = refy == s.cond[1];
    switch (op) { case 0: return x || y; case 1: return x && y; case 2: return x; default: return y; }
}

static inline bool cmp(float a, float b, u32 op) {
    switch (op) {
    case 0: return a == b; case 1: return a != b; case 2: return a < b;
    case 3: return a <= b; case 4: return a > b; case 5: return a >= b;
    }
    return true;
}

int g_nan_hunt = -1;
struct CallFrame { u32 final_addr, return_addr; u32 repeat; u32 inc; u32 loop_addr; };

void shader_run_ref(ShaderUnitSetup &su, Vec4 *input, Vec4 *output, u32 entry, bool is_gs) {
    if (g_nan_hunt < 0) g_nan_hunt = getenv("R3DS_NAN_HUNT") ? 10 : 0;
    ShaderState s;
    if (g_nan_hunt) for (int i = 0; i < 16; i++) for (int k = 0; k < 4; k++) if (std::isnan(input[i][k])) { static int n = 0; if (n++ < 5) LOG("[shader] NaN input v%d.%d", i, k); }
    memcpy(s.in, input, sizeof s.in);
    memset(s.tmp, 0, sizeof s.tmp);
    for (auto &o : s.out) o = Vec4{0, 0, 0, 1};
    s.a0[0] = s.a0[1] = 0; s.aL = 0; s.cond[0] = s.cond[1] = false;
    CallFrame stack[16]; int sp = 0;
    u32 pc = entry;
    u32 emit_vid = 0; bool emit_prim = false, emit_wind = false;
    for (int steps = 0; steps < 200000; steps++) {
        // handle end of call/loop/if blocks
        while (sp > 0 && pc == stack[sp - 1].final_addr) {
            CallFrame &f = stack[sp - 1];
            if (f.repeat) {
                f.repeat--;
                s.aL += (s8)f.inc;
                pc = f.loop_addr;
                break;
            }
            pc = f.return_addr;
            sp--;
        }
        if (pc >= 4096) break;
        u32 ins = su.code[pc];
        u32 op = ins >> 26;
        u32 next = pc + 1;
        auto do_arith = [&](bool inverted) {
            u32 desc = su.swizzle[ins & 0x7F];
            u32 dst = (ins >> 21) & 0x1F;
            u32 idx = (ins >> 19) & 3;
            u32 r1, r2;
            Vec4 a, b;
            if (!inverted) { r1 = (ins >> 12) & 0x7F; r2 = (ins >> 7) & 0x1F;
                a = get_src(s, su, r1, idx); b = get_src(s, su, r2, 0); }
            else { r1 = (ins >> 14) & 0x1F; r2 = (ins >> 7) & 0x7F;
                a = get_src(s, su, r1, 0); b = get_src(s, su, r2, idx); }
            a = swz(a, (desc >> 5) & 0xFF, (desc >> 4) & 1);
            b = swz(b, (desc >> 14) & 0xFF, (desc >> 13) & 1);
            u32 mask = desc & 0xF;
            Vec4 r{0, 0, 0, 0};
            switch (op) {
            case 0x00: for (int i = 0; i < 4; i++) r[i] = a[i] + b[i]; break;
            case 0x01: { float d = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z); r = {d, d, d, d}; break; }
            case 0x02: { float d = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z) + pmul(a.w, b.w); r = {d, d, d, d}; break; }
            case 0x03: case 0x18: { float d = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z) + b.w; r = {d, d, d, d}; break; }
            case 0x04: case 0x19: r = {1.0f, pmul(a.y, b.y), a.z, b.w}; break;
            case 0x05: { float d = exp2f(a.x); r = {d, d, d, d}; break; }
            case 0x06: { float d = log2f(a.x); r = {d, d, d, d}; break; }
            case 0x07: {  // LITP
                float x = std::max(a.x, 0.0f), y = std::max(a.y, 0.0f);
                float w = std::clamp(a.w, -127.9961f, 127.9961f);
                r = {x, y, a.z, w};
                s.cond[0] = a.x > 0; s.cond[1] = a.y > 0;
                break;
            }
            case 0x08: for (int i = 0; i < 4; i++) r[i] = pmul(a[i], b[i]); break;
            case 0x09: case 0x1A: for (int i = 0; i < 4; i++) r[i] = a[i] >= b[i] ? 1.0f : 0.0f; break;
            case 0x0A: case 0x1B: for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? 1.0f : 0.0f; break;
            case 0x0B: for (int i = 0; i < 4; i++) r[i] = floorf(a[i]); break;
            case 0x0C: for (int i = 0; i < 4; i++) r[i] = a[i] > b[i] ? a[i] : b[i]; break;
            case 0x0D: for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? a[i] : b[i]; break;
            case 0x0E: { float d = 1.0f / a.x; r = {d, d, d, d}; break; }
            case 0x0F: { float d = 1.0f / sqrtf(a.x); r = {d, d, d, d}; break; }
            case 0x12:  // MOVA
                if (mask & 8) s.a0[0] = f2i(a.x);
                if (mask & 4) s.a0[1] = f2i(a.y);
                return;
            case 0x13: r = a; break;
            }
            if (g_nan_hunt) {
                bool rn = false, in_nan = false;
                for (int i = 0; i < 4; i++) { if ((mask & (8 >> i)) && std::isnan(r[i])) rn = true; if (std::isnan(a[i]) || std::isnan(b[i])) in_nan = true; }
                if (rn && !in_nan && g_nan_hunt-- > 0)
                    LOG("[shader] NaN from finite inputs: pc=%u op=%02x a=(%g %g %g %g) b=(%g %g %g %g) -> (%g %g %g %g) mask=%x r1=%02x r2=%02x",
                        pc, op, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, r.x, r.y, r.z, r.w, mask, r1, r2);
            }
            write_dst(s, dst, mask, r);
        };
        switch (op) {
        case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: case 0x0F:
        case 0x12: case 0x13:
            do_arith(false); break;
        case 0x18: case 0x19: case 0x1A: case 0x1B:
            do_arith(true); break;
        case 0x20: // BREAK
            if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; }
            break;
        case 0x21: break;  // NOP
        case 0x22: goto done;  // END
        case 0x23:  // BREAKC
            if (eval_cond(s, ins)) {
                // pop to innermost loop
                while (sp > 0 && !stack[sp - 1].loop_addr) sp--;
                if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; stack[sp - 1].return_addr = stack[sp - 1].final_addr; sp--; }
            }
            break;
        case 0x24: case 0x25: case 0x26: {  // CALL / CALLC / CALLU
            bool take = op == 0x24 ? true : op == 0x25 ? eval_cond(s, ins) : ((su.b >> ((ins >> 22) & 0xF)) & 1);
            if (take && sp < 16) {
                u32 dest = (ins >> 10) & 0xFFF, num = ins & 0xFF;
                stack[sp++] = {dest + num, pc + 1, 0, 0, 0};
                next = dest;
            }
            break;
        }
        case 0x27: case 0x28: {  // IFU / IFC
            bool c = op == 0x27 ? ((su.b >> ((ins >> 22) & 0xF)) & 1) : eval_cond(s, ins);
            u32 dest = (ins >> 10) & 0xFFF, num = ins & 0xFF;
            if (sp < 16) {
                if (c) { stack[sp++] = {dest, dest + num, 0, 0, 0}; next = pc + 1; }
                else { stack[sp++] = {dest + num, dest + num, 0, 0, 0}; next = dest; }
            }
            break;
        }
        case 0x29: {  // LOOP
            u32 id = (ins >> 22) & 3, dest = (ins >> 10) & 0xFFF;
            s.aL = su.i[id][1];
            if (sp < 16) { stack[sp++] = {dest + 1, dest + 1, su.i[id][0], su.i[id][2], pc + 1}; }
            next = pc + 1;
            break;
        }
        case 0x2A:  // EMIT
            if (is_gs && g_gs_emit) g_gs_emit(s.out, emit_vid, emit_prim, emit_wind);
            break;
        case 0x2B:  // SETEMIT
            emit_vid = (ins >> 24) & 3; emit_prim = (ins >> 23) & 1; emit_wind = (ins >> 22) & 1;
            break;
        case 0x2C: case 0x2D: {  // JMPC / JMPU
            bool c;
            if (op == 0x2C) c = eval_cond(s, ins);
            else c = ((su.b >> ((ins >> 22) & 0xF)) & 1) == !(ins & 1);
            if (c) next = (ins >> 10) & 0xFFF;
            break;
        }
        case 0x2E: case 0x2F: {  // CMP
            u32 desc = su.swizzle[ins & 0x7F];
            u32 idx = (ins >> 19) & 3;
            Vec4 a = swz(get_src(s, su, (ins >> 12) & 0x7F, idx), (desc >> 5) & 0xFF, (desc >> 4) & 1);
            Vec4 b = swz(get_src(s, su, (ins >> 7) & 0x1F, 0), (desc >> 14) & 0xFF, (desc >> 13) & 1);
            s.cond[0] = cmp(a.x, b.x, (ins >> 24) & 7);
            s.cond[1] = cmp(a.y, b.y, (ins >> 21) & 7);
            break;
        }
        default:
            if (op >= 0x30) {  // MAD / MADI
                bool inv = op < 0x38;
                u32 desc = su.swizzle[ins & 0x1F];
                u32 dst = (ins >> 24) & 0x1F, idx = (ins >> 22) & 3;
                u32 r1 = (ins >> 17) & 0x1F;
                Vec4 a = get_src(s, su, r1, 0), b, c;
                if (!inv) { b = get_src(s, su, (ins >> 10) & 0x7F, idx); c = get_src(s, su, (ins >> 5) & 0x1F, 0); }
                else { b = get_src(s, su, (ins >> 12) & 0x1F, 0); c = get_src(s, su, (ins >> 5) & 0x7F, idx); }
                a = swz(a, (desc >> 5) & 0xFF, (desc >> 4) & 1);
                b = swz(b, (desc >> 14) & 0xFF, (desc >> 13) & 1);
                c = swz(c, (desc >> 23) & 0xFF, (desc >> 22) & 1);
                Vec4 r;
                for (int i = 0; i < 4; i++) r[i] = pmul(a[i], b[i]) + c[i];
                if (g_nan_hunt) {
                    bool rn = false, in_nan = false;
                    for (int i = 0; i < 4; i++) { if (((desc & 0xF) & (8 >> i)) && std::isnan(r[i])) rn = true; if (std::isnan(a[i]) || std::isnan(b[i]) || std::isnan(c[i])) in_nan = true; }
                    if (rn && !in_nan && g_nan_hunt-- > 0)
                        LOG("[shader] NaN from finite MAD inputs: pc=%u a=(%g %g %g %g) b=(%g %g %g %g) c=(%g %g %g %g)", pc, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w);
                }
                write_dst(s, dst, desc & 0xF, r);
            } else {
                static int warned = 0;
                if (warned++ < 5) LOG("[shader] unknown opcode %02x at %u", op, pc);
            }
        }
        pc = next;
    }
done:
    memcpy(output, s.out, sizeof s.out);
}


// ------------------------------------------------------------ fast path
static inline void dsrc(DSrc &d, u32 reg, u32 sel, bool neg) {
    d.reg = (u8)reg; d.neg = neg;
    for (int i = 0; i < 4; i++) d.sel[i] = (sel >> (2 * (3 - i))) & 3;
}

void shader_decode(ShaderUnitSetup &su) {
    for (u32 pc = 0; pc < 4096; pc++) {
        u32 ins = su.code[pc];
        DIns &d = su.dec[pc];
        memset(&d, 0, sizeof d);
        u32 op = ins >> 26;
        d.op = (u8)op;
        if (op >= 0x30) {   // MAD / MADI
            bool inv = op < 0x38;
            d.op = inv ? 0x30 : 0x38;
            u32 desc = su.swizzle[ins & 0x1F];
            d.dst = (ins >> 24) & 0x1F; d.idx = (ins >> 22) & 3; d.mask = desc & 0xF;
            dsrc(d.s[0], (ins >> 17) & 0x1F, (desc >> 5) & 0xFF, (desc >> 4) & 1);
            if (!inv) { dsrc(d.s[1], (ins >> 10) & 0x7F, (desc >> 14) & 0xFF, (desc >> 13) & 1); dsrc(d.s[2], (ins >> 5) & 0x1F, (desc >> 23) & 0xFF, (desc >> 22) & 1); d.rel = 2; }
            else { dsrc(d.s[1], (ins >> 12) & 0x1F, (desc >> 14) & 0xFF, (desc >> 13) & 1); dsrc(d.s[2], (ins >> 5) & 0x7F, (desc >> 23) & 0xFF, (desc >> 22) & 1); d.rel = 3; }
            continue;
        }
        if (op <= 0x13 || (op >= 0x18 && op <= 0x1B) || op == 0x2E || op == 0x2F) {
            u32 desc = su.swizzle[ins & 0x7F];
            d.mask = desc & 0xF;
            d.dst = (ins >> 21) & 0x1F;
            d.idx = (ins >> 19) & 3;
            bool inv = op >= 0x18 && op <= 0x1B;
            if (!inv) { dsrc(d.s[0], (ins >> 12) & 0x7F, (desc >> 5) & 0xFF, (desc >> 4) & 1); dsrc(d.s[1], (ins >> 7) & 0x1F, (desc >> 14) & 0xFF, (desc >> 13) & 1); d.rel = 1; }
            else { dsrc(d.s[0], (ins >> 14) & 0x1F, (desc >> 5) & 0xFF, (desc >> 4) & 1); dsrc(d.s[1], (ins >> 7) & 0x7F, (desc >> 14) & 0xFF, (desc >> 13) & 1); d.rel = 2; }
            if (op == 0x2E || op == 0x2F) { d.op = 0x2E; d.cmpx = (ins >> 24) & 7; d.cmpy = (ins >> 21) & 7; }
            continue;
        }
        d.dest = (ins >> 10) & 0xFFF; d.num = ins & 0xFF;
        d.condop = (ins >> 22) & 3; d.refy = (ins >> 24) & 1; d.refx = (ins >> 25) & 1;
        d.bool_id = (ins >> 22) & 0xF; d.int_id = (ins >> 22) & 3;
        if (op == 0x2B) { d.cmpx = (ins >> 24) & 3; d.cmpy = (ins >> 23) & 1; d.refx = (ins >> 22) & 1; }
    }
    su.dec_dirty = false;
}

static inline bool cond_d(const ShaderState &s, const DIns &d) {
    bool x = d.refx == s.cond[0], y = d.refy == s.cond[1];
    switch (d.condop) { case 0: return x || y; case 1: return x && y; case 2: return x; default: return y; }
}

static inline Vec4 fetch(const ShaderState &s, const ShaderUnitSetup &su, const DSrc &src, u32 rel_idx) {
    const Vec4 *v;
    Vec4 tmp;
    u32 reg = src.reg;
    if (reg < 0x10) v = &s.in[reg];
    else if (reg < 0x20) v = &s.tmp[reg - 0x10];
    else {
        s32 off = rel_idx == 1 ? s.a0[0] : rel_idx == 2 ? s.a0[1] : rel_idx == 3 ? s.aL : 0;
        s32 idx = (s32)(reg - 0x20) + off;
        if (idx < 0 || idx >= 96) { tmp = Vec4{1, 1, 1, 1}; v = &tmp; }
        else v = &su.f[idx];
    }
    Vec4 r;
    r.x = (*v)[src.sel[0]]; r.y = (*v)[src.sel[1]]; r.z = (*v)[src.sel[2]]; r.w = (*v)[src.sel[3]];
    if (src.neg) { r.x = -r.x; r.y = -r.y; r.z = -r.z; r.w = -r.w; }
    return r;
}

void shader_run_fast(ShaderUnitSetup &su, Vec4 *input, Vec4 *output, u32 entry, bool is_gs) {
    ShaderState s;
    memcpy(s.in, input, sizeof s.in);
    memset(s.tmp, 0, sizeof s.tmp);
    for (auto &o : s.out) o = Vec4{0, 0, 0, 1};
    s.a0[0] = s.a0[1] = 0; s.aL = 0; s.cond[0] = s.cond[1] = false;
    CallFrame stack[16]; int sp = 0;
    u32 pc = entry;
    u32 emit_vid = 0; bool emit_prim = false, emit_wind = false;
    for (int steps = 0; steps < 200000; steps++) {
        while (sp > 0 && pc == stack[sp - 1].final_addr) {
            CallFrame &f = stack[sp - 1];
            if (f.repeat) { f.repeat--; s.aL += (s8)f.inc; pc = f.loop_addr; break; }
            pc = f.return_addr; sp--;
        }
        if (pc >= 4096) break;
        const DIns &d = su.dec[pc];
        u32 next = pc + 1;
        switch (d.op) {
        case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: case 0x0F:
        case 0x12: case 0x13: case 0x18: case 0x19: case 0x1A: case 0x1B: {
            Vec4 a = fetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0);
            Vec4 b = fetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0);
            Vec4 r{0, 0, 0, 0};
            switch (d.op) {
            case 0x00: for (int i = 0; i < 4; i++) r[i] = a[i] + b[i]; break;
            case 0x01: { float x = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z); r = {x, x, x, x}; break; }
            case 0x02: { float x = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z) + pmul(a.w, b.w); r = {x, x, x, x}; break; }
            case 0x03: case 0x18: { float x = pmul(a.x, b.x) + pmul(a.y, b.y) + pmul(a.z, b.z) + b.w; r = {x, x, x, x}; break; }
            case 0x04: case 0x19: r = {1.0f, pmul(a.y, b.y), a.z, b.w}; break;
            case 0x05: { float x = exp2f(a.x); r = {x, x, x, x}; break; }
            case 0x06: { float x = log2f(a.x); r = {x, x, x, x}; break; }
            case 0x07: {
                float x = std::max(a.x, 0.0f), y = std::max(a.y, 0.0f);
                r = {x, y, a.z, std::clamp(a.w, -127.9961f, 127.9961f)};
                s.cond[0] = a.x > 0; s.cond[1] = a.y > 0;
                break;
            }
            case 0x08: for (int i = 0; i < 4; i++) r[i] = pmul(a[i], b[i]); break;
            case 0x09: case 0x1A: for (int i = 0; i < 4; i++) r[i] = a[i] >= b[i] ? 1.0f : 0.0f; break;
            case 0x0A: case 0x1B: for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? 1.0f : 0.0f; break;
            case 0x0B: for (int i = 0; i < 4; i++) r[i] = floorf(a[i]); break;
            case 0x0C: for (int i = 0; i < 4; i++) r[i] = a[i] > b[i] ? a[i] : b[i]; break;
            case 0x0D: for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? a[i] : b[i]; break;
            case 0x0E: { float x = 1.0f / a.x; r = {x, x, x, x}; break; }
            case 0x0F: { float x = 1.0f / sqrtf(a.x); r = {x, x, x, x}; break; }
            case 0x12:
                if (d.mask & 8) s.a0[0] = f2i(a.x);
                if (d.mask & 4) s.a0[1] = f2i(a.y);
                goto next_ins;
            case 0x13: r = a; break;
            }
            write_dst(s, d.dst, d.mask, r);
            break;
        }
        case 0x20:
            if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; }
            break;
        case 0x21: break;
        case 0x22: goto done;
        case 0x23:
            if (cond_d(s, d)) {
                while (sp > 0 && !stack[sp - 1].loop_addr) sp--;
                if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; stack[sp - 1].return_addr = stack[sp - 1].final_addr; sp--; }
            }
            break;
        case 0x24: case 0x25: case 0x26: {
            bool take = d.op == 0x24 ? true : d.op == 0x25 ? cond_d(s, d) : ((su.b >> d.bool_id) & 1);
            if (take && sp < 16) { stack[sp++] = {(u32)d.dest + d.num, pc + 1, 0, 0, 0}; next = d.dest; }
            break;
        }
        case 0x27: case 0x28: {
            bool c = d.op == 0x27 ? ((su.b >> d.bool_id) & 1) : cond_d(s, d);
            if (sp < 16) {
                if (c) { stack[sp++] = {d.dest, (u32)d.dest + d.num, 0, 0, 0}; next = pc + 1; }
                else { stack[sp++] = {(u32)d.dest + d.num, (u32)d.dest + d.num, 0, 0, 0}; next = d.dest; }
            }
            break;
        }
        case 0x29:
            s.aL = su.i[d.int_id][1];
            if (sp < 16) stack[sp++] = {(u32)d.dest + 1, (u32)d.dest + 1, su.i[d.int_id][0], su.i[d.int_id][2], pc + 1};
            break;
        case 0x2A:
            if (is_gs && g_gs_emit) g_gs_emit(s.out, emit_vid, emit_prim, emit_wind);
            break;
        case 0x2B:
            emit_vid = d.cmpx; emit_prim = d.cmpy; emit_wind = d.refx;
            break;
        case 0x2C: case 0x2D: {
            bool c = d.op == 0x2C ? cond_d(s, d) : (((su.b >> d.bool_id) & 1) == !(d.num & 1));
            if (c) next = d.dest;
            break;
        }
        case 0x2E: {
            Vec4 a = fetch(s, su, d.s[0], d.idx), b = fetch(s, su, d.s[1], 0);
            s.cond[0] = cmp(a.x, b.x, d.cmpx);
            s.cond[1] = cmp(a.y, b.y, d.cmpy);
            break;
        }
        case 0x30: case 0x38: {
            Vec4 a = fetch(s, su, d.s[0], 0);
            Vec4 b = fetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0);
            Vec4 c = fetch(s, su, d.s[2], d.rel == 3 ? d.idx : 0);
            Vec4 r;
            for (int i = 0; i < 4; i++) r[i] = pmul(a[i], b[i]) + c[i];
            write_dst(s, d.dst, d.mask, r);
            break;
        }
        default: break;
        }
    next_ins:
        pc = next;
    }
done:
    memcpy(output, s.out, sizeof s.out);
}

// ------------------------------------------------------------ SSE path
typedef __m128 V;
static inline V vload(const Vec4 &v) { return _mm_loadu_ps(&v.x); }
static inline V vswz(V v, const u8 *sel) {
    if (sel[0] == 0 && sel[1] == 1 && sel[2] == 2 && sel[3] == 3) return v;
    alignas(16) float t[4];
    _mm_store_ps(t, v);
    return _mm_setr_ps(t[sel[0]], t[sel[1]], t[sel[2]], t[sel[3]]);
}
static inline V vpmul(V a, V b) {
    V p = _mm_mul_ps(a, b);
    // PICA: NaN from non-NaN inputs (0 * inf) is 0
    V bad = _mm_and_ps(_mm_cmpunord_ps(p, p), _mm_and_ps(_mm_cmpord_ps(a, a), _mm_cmpord_ps(b, b)));
    return _mm_andnot_ps(bad, p);
}
static inline float hsum3(V p) { alignas(16) float t[4]; _mm_store_ps(t, p); return t[0] + t[1] + t[2]; }
static inline float hsum4(V p) { alignas(16) float t[4]; _mm_store_ps(t, p); return (t[0] + t[1]) + t[2] + t[3]; }

struct SState {
    V in[16], tmp[16], out[16];
    s32 a0[0x2], aL;
    bool cond[2];
};

static const V k_masks[16] = {
#define M(b) _mm_castsi128_ps(_mm_setr_epi32((b & 8) ? -1 : 0, (b & 4) ? -1 : 0, (b & 2) ? -1 : 0, (b & 1) ? -1 : 0))
    M(0), M(1), M(2), M(3), M(4), M(5), M(6), M(7), M(8), M(9), M(10), M(11), M(12), M(13), M(14), M(15)
#undef M
};

static inline V sfetch(const SState &s, const ShaderUnitSetup &su, const DSrc &src, u32 rel_idx) {
    V v;
    u32 reg = src.reg;
    if (reg < 0x10) v = s.in[reg];
    else if (reg < 0x20) v = s.tmp[reg - 0x10];
    else {
        s32 off = rel_idx == 1 ? s.a0[0] : rel_idx == 2 ? s.a0[1] : rel_idx == 3 ? s.aL : 0;
        s32 idx = (s32)(reg - 0x20) + off;
        v = (idx < 0 || idx >= 96) ? _mm_set1_ps(1.0f) : vload(su.f[idx]);
    }
    v = vswz(v, src.sel);
    if (src.neg) v = _mm_xor_ps(v, _mm_set1_ps(-0.0f));
    return v;
}
static inline void swrite(SState &s, u32 dst, u32 mask, V r) {
    V *d = dst < 0x10 ? &s.out[dst] : &s.tmp[(dst - 0x10) & 15];
    if (mask == 0xF) { *d = r; return; }
    V m = k_masks[mask];
    *d = _mm_or_ps(_mm_and_ps(m, r), _mm_andnot_ps(m, *d));
}
static inline bool scond(const SState &s, const DIns &d) {
    bool x = d.refx == s.cond[0], y = d.refy == s.cond[1];
    switch (d.condop) { case 0: return x || y; case 1: return x && y; case 2: return x; default: return y; }
}

void shader_run_sse(ShaderUnitSetup &su, Vec4 *input, Vec4 *output, u32 entry, bool is_gs) {
    SState s;
    for (int i = 0; i < 16; i++) { s.in[i] = vload(input[i]); s.tmp[i] = _mm_setzero_ps(); s.out[i] = _mm_setr_ps(0, 0, 0, 1); }
    s.a0[0] = s.a0[1] = 0; s.aL = 0; s.cond[0] = s.cond[1] = false;
    CallFrame stack[16]; int sp = 0;
    u32 pc = entry;
    u32 emit_vid = 0; bool emit_prim = false, emit_wind = false;
    for (int steps = 0; steps < 200000; steps++) {
        while (sp > 0 && pc == stack[sp - 1].final_addr) {
            CallFrame &f = stack[sp - 1];
            if (f.repeat) { f.repeat--; s.aL += (s8)f.inc; pc = f.loop_addr; break; }
            pc = f.return_addr; sp--;
        }
        if (pc >= 4096) break;
        const DIns &d = su.dec[pc];
        u32 next = pc + 1;
        switch (d.op) {
        case 0x00: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_add_ps(a, b)); break; }
        case 0x08: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, vpmul(a, b)); break; }
        case 0x01: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_set1_ps(hsum3(vpmul(a, b)))); break; }
        case 0x02: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_set1_ps(hsum4(vpmul(a, b)))); break; }
        case 0x03: case 0x18: {
            V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0);
            alignas(16) float tb[4]; _mm_store_ps(tb, b);
            swrite(s, d.dst, d.mask, _mm_set1_ps(hsum3(vpmul(a, b)) + tb[3])); break;
        }
        case 0x0C: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_max_ps(a, b)); break; }
        case 0x0D: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_min_ps(a, b)); break; }
        case 0x09: case 0x1A: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_and_ps(_mm_cmpge_ps(a, b), _mm_set1_ps(1.0f))); break; }
        case 0x0A: case 0x1B: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0), b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0); swrite(s, d.dst, d.mask, _mm_and_ps(_mm_cmplt_ps(a, b), _mm_set1_ps(1.0f))); break; }
        case 0x13: { V a = sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0); swrite(s, d.dst, d.mask, a); break; }
        case 0x04: case 0x05: case 0x06: case 0x07: case 0x0B: case 0x0E: case 0x0F: case 0x12: case 0x19: {
            // rarer ops: scalar
            alignas(16) float a[4], b[4];
            _mm_store_ps(a, sfetch(s, su, d.s[0], d.rel == 1 ? d.idx : 0));
            _mm_store_ps(b, sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0));
            float r[4] = {0, 0, 0, 0};
            switch (d.op) {
            case 0x04: case 0x19: r[0] = 1.0f; r[1] = pmul(a[1], b[1]); r[2] = a[2]; r[3] = b[3]; break;
            case 0x05: r[0] = r[1] = r[2] = r[3] = exp2f(a[0]); break;
            case 0x06: r[0] = r[1] = r[2] = r[3] = log2f(a[0]); break;
            case 0x07:
                r[0] = std::max(a[0], 0.0f); r[1] = std::max(a[1], 0.0f); r[2] = a[2]; r[3] = std::clamp(a[3], -127.9961f, 127.9961f);
                s.cond[0] = a[0] > 0; s.cond[1] = a[1] > 0;
                break;
            case 0x0B: for (int i = 0; i < 4; i++) r[i] = floorf(a[i]); break;
            case 0x0E: r[0] = r[1] = r[2] = r[3] = 1.0f / a[0]; break;
            case 0x0F: r[0] = r[1] = r[2] = r[3] = 1.0f / sqrtf(a[0]); break;
            case 0x12:
                if (d.mask & 8) s.a0[0] = f2i(a[0]);
                if (d.mask & 4) s.a0[1] = f2i(a[1]);
                goto next_ins;
            }
            swrite(s, d.dst, d.mask, _mm_loadu_ps(r));
            break;
        }
        case 0x20:
            if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; }
            break;
        case 0x21: break;
        case 0x22: goto done;
        case 0x23:
            if (scond(s, d)) {
                while (sp > 0 && !stack[sp - 1].loop_addr) sp--;
                if (sp > 0) { next = stack[sp - 1].final_addr; stack[sp - 1].repeat = 0; stack[sp - 1].return_addr = stack[sp - 1].final_addr; sp--; }
            }
            break;
        case 0x24: case 0x25: case 0x26: {
            bool take = d.op == 0x24 ? true : d.op == 0x25 ? scond(s, d) : ((su.b >> d.bool_id) & 1);
            if (take && sp < 16) { stack[sp++] = {(u32)d.dest + d.num, pc + 1, 0, 0, 0}; next = d.dest; }
            break;
        }
        case 0x27: case 0x28: {
            bool c = d.op == 0x27 ? ((su.b >> d.bool_id) & 1) : scond(s, d);
            if (sp < 16) {
                if (c) { stack[sp++] = {d.dest, (u32)d.dest + d.num, 0, 0, 0}; next = pc + 1; }
                else { stack[sp++] = {(u32)d.dest + d.num, (u32)d.dest + d.num, 0, 0, 0}; next = d.dest; }
            }
            break;
        }
        case 0x29:
            s.aL = su.i[d.int_id][1];
            if (sp < 16) stack[sp++] = {(u32)d.dest + 1, (u32)d.dest + 1, su.i[d.int_id][0], su.i[d.int_id][2], pc + 1};
            break;
        case 0x2A:
            if (is_gs && g_gs_emit) {
                Vec4 o[16];
                for (int i = 0; i < 16; i++) _mm_storeu_ps(&o[i].x, s.out[i]);
                g_gs_emit(o, emit_vid, emit_prim, emit_wind);
            }
            break;
        case 0x2B: emit_vid = d.cmpx; emit_prim = d.cmpy; emit_wind = d.refx; break;
        case 0x2C: case 0x2D: {
            bool c = d.op == 0x2C ? scond(s, d) : (((su.b >> d.bool_id) & 1) == !(d.num & 1));
            if (c) next = d.dest;
            break;
        }
        case 0x2E: {
            alignas(16) float a[4], b[4];
            _mm_store_ps(a, sfetch(s, su, d.s[0], d.idx));
            _mm_store_ps(b, sfetch(s, su, d.s[1], 0));
            s.cond[0] = cmp(a[0], b[0], d.cmpx);
            s.cond[1] = cmp(a[1], b[1], d.cmpy);
            break;
        }
        case 0x30: case 0x38: {
            V a = sfetch(s, su, d.s[0], 0);
            V b = sfetch(s, su, d.s[1], d.rel == 2 ? d.idx : 0);
            V c = sfetch(s, su, d.s[2], d.rel == 3 ? d.idx : 0);
            swrite(s, d.dst, d.mask, _mm_add_ps(vpmul(a, b), c));
            break;
        }
        default: break;
        }
    next_ins:
        pc = next;
    }
done:
    for (int i = 0; i < 16; i++) _mm_storeu_ps(&output[i].x, s.out[i]);
}

void shader_run(ShaderUnitSetup &su, Vec4 *input, Vec4 *output, u32 entry, bool is_gs) {
    static int mode = -1;
    if (mode < 0) mode = getenv("R3DS_SHADER_REF") ? 1 : getenv("R3DS_SHADER_CHECK") ? 2 : 0;
    if (su.dec_dirty) shader_decode(su);
    if (mode == 1) { shader_run_ref(su, input, output, entry, is_gs); return; }
    shader_run_sse(su, input, output, entry, is_gs);
    if (mode == 2 && !is_gs) {
        Vec4 ref[16];
        shader_run_ref(su, input, ref, entry, is_gs);
        if (memcmp(ref, output, sizeof ref)) {
            static int n = 0;
            if (n++ < 10) LOG("[shader] fast/ref mismatch at entry %u: o0 fast (%g %g %g %g) ref (%g %g %g %g)", entry,
                              output[0].x, output[0].y, output[0].z, output[0].w, ref[0].x, ref[0].y, ref[0].z, ref[0].w);
        }
    }
}
