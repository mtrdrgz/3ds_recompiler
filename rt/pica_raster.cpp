// PICA200 software renderer: register side effects, vertex fetch, primitive
// assembly, clipping, rasterization and the per-fragment pipeline.
#include "pica.h"
#include "pica_snap.h"
#include "pica_fmt.h"
#include "mem.h"
#include "gpu.h"
#include "hwr.h"
#include <vector>
#include <functional>

#define R (g_pica->regs)
u32 g_cmdbuf_jump_pa = 0, g_cmdbuf_jump_size = 0;
int g_pica_dbg = -1;
u32 g_pica_state_gen = 1;
void raster_tri_fast(const OutVertex &v0, const OutVertex &v1, const OutVertex &v2);
extern std::function<void(Vec4 *out, int vtx_id, bool prim_emit, bool winding)> g_gs_emit;
static void triangle(const OutVertex &a, const OutVertex &b, const OutVertex &c);
void shader_decode(ShaderUnitSetup &su);
void pica_lighting(const OutVertex &v, const float *view, const float *normal_q, u32 *prim, u32 *sec);

// ------------------------------------------------------------ uniform upload
static void upload_float(ShaderUnitSetup &su, u32 v) {
    su.f_buf[su.f_count++] = v;
    u32 need = su.f32mode ? 4 : 3;
    if (su.f_count < need) return;
    su.f_count = 0;
    Vec4 u;
    if (su.f32mode) {
        memcpy(&u.w, &su.f_buf[0], 4); memcpy(&u.z, &su.f_buf[1], 4);
        memcpy(&u.y, &su.f_buf[2], 4); memcpy(&u.x, &su.f_buf[3], 4);
    } else {
        u32 *b = su.f_buf;
        u.w = f24(b[0] >> 8);
        u.z = f24(((b[0] & 0xFF) << 16) | (b[1] >> 16));
        u.y = f24(((b[1] & 0xFFFF) << 8) | (b[2] >> 24));
        u.x = f24(b[2] & 0xFFFFFF);
    }
    if (su.f_index < 96) su.f[su.f_index] = u;
    su.f_index++;
}

static void shader_reg(ShaderUnitSetup &su, u32 off, u32 v) {
    // off relative to 0x280 (GS) / 0x2B0 (VS)
    switch (off) {
    case 0x00: su.b = v & 0xFFFF; break;
    case 0x01: case 0x02: case 0x03: case 0x04:
        su.i[off - 1][0] = v & 0xFF; su.i[off - 1][1] = (v >> 8) & 0xFF; su.i[off - 1][2] = (v >> 16) & 0xFF; su.i[off - 1][3] = v >> 24; break;
    case 0x10: su.f_index = v & 0xFF; su.f32mode = v >> 31; su.f_count = 0; break;
    case 0x11: case 0x12: case 0x13: case 0x14: case 0x15: case 0x16: case 0x17: case 0x18: upload_float(su, v); break;
    case 0x1B: su.code_off = v & 0xFFF; break;
    case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x20: case 0x21: case 0x22: case 0x23:
        if (su.code_off < 4096) { su.code[su.code_off] = v; su.dec_dirty = true; }
        su.code_off++; break;
    case 0x25: su.swz_off = v & 0x7F; break;
    case 0x26: case 0x27: case 0x28: case 0x29: case 0x2A: case 0x2B: case 0x2C: case 0x2D:
        if (su.swz_off < 128) { su.swizzle[su.swz_off] = v; su.dec_dirty = true; }
        su.swz_off++; break;
    }
}

static Vec4 unpack_f24x4(const u32 *b) {
    Vec4 u;
    u.w = f24(b[0] >> 8);
    u.z = f24(((b[0] & 0xFF) << 16) | (b[1] >> 16));
    u.y = f24(((b[1] & 0xFFFF) << 8) | (b[2] >> 24));
    u.x = f24(b[2] & 0xFFFFFF);
    return u;
}

void pica_write(u32 reg, u32 value, u32 mask) {
    if (reg >= 0x400) return;
    u32 m = (mask & 1 ? 0xFF : 0) | (mask & 2 ? 0xFF00 : 0) | (mask & 4 ? 0xFF0000 : 0) | (mask & 8 ? 0xFF000000 : 0);
    u32 v = (R[reg] & ~m) | (value & m);
    if (R[reg] != v) g_pica_state_gen++;
    R[reg] = v;
    switch (reg) {
    case 0x22E: pica_draw(false); break;
    case 0x22F: pica_draw(true); break;
    case 0x232:
        g_pica->fixed_index = v & 0xF; g_pica->fixed_count = 0;
        if ((v & 0xF) == 0xF) { g_pica->imm_count = 0; g_pica->pa_index = 0; g_pica->pa_strip_ready = false; }
        break;
    case 0x233: case 0x234: case 0x235:
        g_pica->fixed_buf[g_pica->fixed_count++] = v;
        if (g_pica->fixed_count == 3) {
            g_pica->fixed_count = 0;
            Vec4 a = unpack_f24x4(g_pica->fixed_buf);
            if (g_pica->fixed_index == 0xF) {
                g_pica->imm_attr[g_pica->imm_count++] = a;
                if (g_pica->imm_count >= (R[0x242] & 0xF) + 1) { pica_imm_vertex(); g_pica->imm_count = 0; }
            } else {
                g_pica->fixed_attr[g_pica->fixed_index & 15] = a;
                g_pica->fixed_index++;
            }
        }
        break;
    case 0x23C: g_cmdbuf_jump_pa = R[0x23A] << 3; g_cmdbuf_jump_size = R[0x238] << 3; break;
    case 0x23D: g_cmdbuf_jump_pa = R[0x23B] << 3; g_cmdbuf_jump_size = R[0x239] << 3; break;
    case 0x245: case 0x25F: g_pica->pa_index = 0; g_pica->pa_strip_ready = false; break;
    case 0x1C5: g_pica->lut_index = v; break;
    case 0x1C8: case 0x1C9: case 0x1CA: case 0x1CB: case 0x1CC: case 0x1CD: case 0x1CE: case 0x1CF: {
        u32 lut = (g_pica->lut_index >> 8) & 0x1F, idx = g_pica->lut_index & 0xFF;
        if (lut < 24) g_pica->lighting_lut[lut][idx] = v;
        g_pica->lut_index = (g_pica->lut_index & ~0xFFu) | ((idx + 1) & 0xFF);
        break;
    }
    case 0x0E6: g_pica->lut_index = v; R[0x3FF] = v & 0x7F; break;
    case 0x0E8: case 0x0E9: case 0x0EA: case 0x0EB: case 0x0EC: case 0x0ED: case 0x0EE: case 0x0EF:
        g_pica->fog_lut[R[0x3FF] & 0x7F] = v; R[0x3FF] = (R[0x3FF] + 1) & 0x7F; break;
    case 0x0AF: g_pica->proctex_idx = v; break;
    case 0x0B0: case 0x0B1: case 0x0B2: case 0x0B3: case 0x0B4: case 0x0B5: case 0x0B6: case 0x0B7: {
        u32 t = (g_pica->proctex_idx >> 8) & 7, i = g_pica->proctex_idx & 0xFF;
        if (t < 6 && i < 128) g_pica->proctex_lut[t][i] = v;
        g_pica->proctex_idx = (g_pica->proctex_idx & ~0xFFu) | ((i + 1) & 0xFF);
        break;
    }
    default:
        if (reg >= 0x2B0 && reg < 0x2E0) {
            shader_reg(g_pica->vs, reg - 0x2B0, v);
            if (!(R[0x244] & 1)) shader_reg(g_pica->gs, reg - 0x2B0, v);
        } else if (reg >= 0x280 && reg < 0x2B0) {
            shader_reg(g_pica->gs, reg - 0x280, v);
        }
    }
}

// ------------------------------------------------------------ vertex pipeline
// attrs: output registers compacted in output-mask order
static void map_compact(const Vec4 *attrs, u32 n, OutVertex &ov) {
    float slots[24];
    for (int i = 0; i < 24; i++) slots[i] = 0;
    slots[3] = 1; slots[11] = 1;
    u32 total = R[0x4F] & 7;
    for (u32 idx = 0; idx < total && idx < n && idx < 7; idx++) {
        u32 map = R[0x50 + idx];
        for (int c = 0; c < 4; c++) {
            u32 sem = (map >> (8 * c)) & 0x1F;
            if (sem < 24) slots[sem] = attrs[idx][c];
        }
    }
    ov.pos = {slots[0], slots[1], slots[2], slots[3]};
    ov.quat = {slots[4], slots[5], slots[6], slots[7]};
    for (int c = 0; c < 4; c++) ov.color[c] = std::min(fabsf(slots[8 + c]), 1.0f);
    ov.tc0[0] = slots[12]; ov.tc0[1] = slots[13];
    ov.tc1[0] = slots[14]; ov.tc1[1] = slots[15];
    ov.tc0w = slots[16];
    ov.view[0] = slots[18]; ov.view[1] = slots[19]; ov.view[2] = slots[20];
    ov.tc2[0] = slots[22]; ov.tc2[1] = slots[23];
}
static u32 compact(const Vec4 *regs, u32 mask, Vec4 *out) {
    u32 n = 0;
    for (u32 i = 0; i < 16; i++) if (mask & (1u << i)) out[n++] = regs[i];
    return n;
}
static void map_outputs(const Vec4 *o, OutVertex &ov) {
    Vec4 c[16];
    u32 n = compact(o, R[0x2BD], c);
    map_compact(c, n, ov);
}

static void gs_emit_vertex(const OutVertex &v);
static bool gs_enabled() { return (R[0x229] & 3) == 2; }

static bool g_pa_winding = false;
static void assemble(const OutVertex &v) {
    u32 topo = (R[0x25E] >> 8) & 3;
    Pica &p = *g_pica;
    if (topo == 0 || topo == 3) {
        if (p.pa_index < 2) { p.pa_buf[p.pa_index++] = v; }
        else {
            p.pa_index = 0;
            if (topo == 3 && g_pa_winding) { triangle(p.pa_buf[1], p.pa_buf[0], v); g_pa_winding = false; }
            else triangle(p.pa_buf[0], p.pa_buf[1], v);
        }
    } else {
        if (p.pa_strip_ready) triangle(p.pa_buf[0], p.pa_buf[1], v);
        p.pa_buf[p.pa_index] = v;
        p.pa_strip_ready |= (p.pa_index == 1);
        if (topo == 1) p.pa_index = !p.pa_index;
        else p.pa_index = 1;
    }
}

// geometry shader: EMIT stores the compacted GS outputs into emitter slot
// `vid`; with prim_emit set all three slots go to the primitive assembler.
static Vec4 g_gs_emit_buf[3][16];
static u32 g_gs_emit_n[3];
static void gs_emit_cb(Vec4 *out, int vid, bool prim, bool winding) {
    vid = vid > 2 ? 2 : vid;
    g_gs_emit_n[vid] = compact(out, R[0x28D], g_gs_emit_buf[vid]);
    if (prim) {
        if (winding) g_pa_winding = true;
        for (int i = 0; i < 3; i++) {
            OutVertex ov;
            map_compact(g_gs_emit_buf[i], g_gs_emit_n[i], ov);
            assemble(ov);
        }
    }
}

static std::vector<Vec4> g_gs_inputs;

static bool run_vertex(Vec4 *attr_in, u32 nattr, OutVertex &ov);
static void process_vertex(Vec4 *attr_in, u32 nattr) {
    OutVertex ov;
    if (run_vertex(attr_in, nattr, ov)) assemble(ov);
}
// returns false when the geometry shader consumed the vertex
static bool run_vertex(Vec4 *attr_in, u32 nattr, OutVertex &ov) {
    Vec4 in[16];
    for (auto &x : in) x = Vec4{0, 0, 0, 1};
    u64 perm = (u64)R[0x2BB] | ((u64)R[0x2BC] << 32);
    for (u32 i = 0; i < nattr && i < 16; i++) in[(perm >> (4 * i)) & 0xF] = attr_in[i];
    Vec4 out[16];
    shader_run(g_pica->vs, in, out, R[0x2BA] & 0xFFFF, false);
    if (gs_enabled()) {
        // point mode: buffer the compacted VS outputs until the GS input
        // count is reached, then load them through the GS input permutation
        Vec4 c[16];
        u32 n = compact(out, R[0x2BD], c);
        for (u32 i = 0; i < n; i++) g_gs_inputs.push_back(c[i]);
        u32 need = (R[0x289] & 0xF) + 1;
        if (g_gs_inputs.size() >= need) {
            Vec4 gin[16];
            for (auto &x : gin) x = Vec4{0, 0, 0, 1};
            u64 perm = (u64)R[0x28B] | ((u64)R[0x28C] << 32);
            for (u32 i = 0; i < need && i < 16; i++) gin[(perm >> (4 * i)) & 0xF] = g_gs_inputs[i];
            g_gs_inputs.clear();
            Vec4 gout[16];
            g_gs_emit = gs_emit_cb;
            shader_run(g_pica->gs, gin, gout, R[0x28A] & 0xFFFF, true);
        }
        return false;
    }
    map_outputs(out, ov);
    return true;
}

void pica_imm_vertex() {
    process_vertex(g_pica->imm_attr, (R[0x242] & 0xF) + 1);
}

static float read_attr_elem(u32 va, u32 type) {
    switch (type) {
    case 0: return (float)(s8)gpu_rd8(va);
    case 1: return (float)gpu_rd8(va);
    case 2: return (float)(s16)gpu_rd16(va);
    default: { u32 b = gpu_rd32(va); float f; memcpy(&f, &b, 4); return f; }
    }
}

// R3DS_PICA_DUMP=N: log a summary of every draw of display frame N (for
// comparing hosts, e.g. native vs wasm). Flushes after each draw.
u64 g_dump_tris = 0;
static int g_dump_frame = -2;
static void pica_draw_impl(bool indexed);
void pica_flush();
extern u64 g_frag_count;
u32 display_frame_count();
void raster_draw_begin();
int raster_draw_end(double *area, double *rt_area);   // 0 not a screen target, 1 backdrop, 2 content, 3 after backdrop
void pica_draw(bool indexed) {
    if (g_hwr_on && hwr_skip_draw()) return;
    if (g_dump_frame == -2) g_dump_frame = getenv("R3DS_PICA_DUMP") ? atoi(getenv("R3DS_PICA_DUMP")) : -1;
    double area = 0, rta = 0;
    static int dump_n = getenv("R3DS_PICA_DUMP_N") ? atoi(getenv("R3DS_PICA_DUMP_N")) : 4;
    if (g_dump_frame < 0 || (int)display_frame_count() < g_dump_frame || (int)display_frame_count() >= g_dump_frame + dump_n) {
        raster_draw_begin(); hwr_draw_begin();
        pica_draw_impl(indexed);
        hwr_draw_end(raster_draw_end(&area, &rta));   // backdrop classification
        return;
    }
    pica_flush();
    u64 f0 = __atomic_load_n(&g_frag_count, __ATOMIC_RELAXED), t0 = g_dump_tris;
    raster_draw_begin(); hwr_draw_begin();
    pica_draw_impl(indexed);
    int cls = raster_draw_end(&area, &rta);
    hwr_draw_end(cls);
    pica_flush();
    LOG("[dump]   rt=%08x cls=%d area=%.0f/%.0f vp=%.0fx%.0f", R[0x11D] << 3, cls, area, rta, f24(R[0x41]) * 2, f24(R[0x43]) * 2);
    u64 f1 = __atomic_load_n(&g_frag_count, __ATOMIC_RELAXED);
    LOG("[dump] f%u draw%s n=%u tris=%llu frags=%llu vs=%x tex0=%08x/%08x/%x cfg0=%08x depth=%08x light=%x lut=%08x cull=%x blend=%08x combs=%08x,%08x",
        display_frame_count(), indexed ? "E" : "A", R[0x228], (unsigned long long)(g_dump_tris - t0), (unsigned long long)(f1 - f0), R[0x2BA] & 0xFFFF,
        R[0x85] << 3, R[0x82], R[0x8E], R[0x80], R[0x107], R[0x08F], R[0x1C4], R[0x40], R[0x101], R[0xC0], R[0xC8]);
}

static void pica_draw_impl(bool indexed) {
    g_pica->draws++;
    if (gs_enabled()) { static int n = 0; if (n++ < 5) LOG("[pica] draw with geometry shader: geostage=%08x gs_entry=%x inputs=%u", R[0x229], R[0x28A] & 0xFFFF, (R[0x289] & 0xF) + 1); }
    if (g_pica_dbg > 0) LOG("[pica] draw%s count=%u first=%u nattr=%u base=%08x topo=%u entry=%x", indexed ? "E" : "A", R[0x228], R[0x22A], (R[0x202] >> 28) + 1, R[0x200] << 3, (R[0x25E] >> 8) & 3, R[0x2BA] & 0xFFFF);
    u32 base_pa = R[0x200] << 3;
    u32 nattr = (R[0x202] >> 28) + 1;
    u32 fixed_mask = (R[0x202] >> 16) & 0xFFF;
    u64 fmt = (u64)R[0x201] | ((u64)(R[0x202] & 0xFFFF) << 32);
    u32 count = R[0x228];
    u32 first = R[0x22A];
    u32 icfg = R[0x227];
    u32 ibase = base_pa + (icfg & 0x0FFFFFFF);
    bool i16 = icfg >> 31;

    // per-attribute source address/stride
    u32 src[12] = {}, stride[12] = {}, elems[12] = {}, etype[12] = {};
    for (u32 l = 0; l < 12; l++) {
        u32 off = R[0x203 + l * 3], lo = R[0x204 + l * 3], hi = R[0x205 + l * 3];
        u32 ncomp = hi >> 28, bytes = (hi >> 16) & 0xFF;
        u64 comps = (u64)lo | ((u64)(hi & 0xFFFF) << 32);
        u32 o = 0;
        for (u32 c = 0; c < ncomp && c < 12; c++) {
            u32 a = (comps >> (4 * c)) & 0xF;
            if (a < 12) {
                u32 f = (fmt >> (4 * a)) & 0xF;
                u32 t = f & 3, n = (f >> 2) + 1;
                u32 es = t == 0 || t == 1 ? 1 : t == 2 ? 2 : 4;
                o = (o + es - 1) & ~(es - 1);
                src[a] = base_pa + off + o;
                stride[a] = bytes;
                elems[a] = n; etype[a] = t;
                o += n * es;
            } else {
                o = (o + 3) & ~3u;
                o += (a - 11) * 4;
            }
        }
    }
    // large draws without a geometry shader: shade the unique vertices in parallel
    if (!gs_enabled() && count >= 96 && !getenv("R3DS_SERIAL_VS")) {
        std::vector<u32> idxs(count);
        for (u32 n = 0; n < count; n++) {
            if (indexed) {
                u32 ia = pa_to_va(ibase + n * (i16 ? 2 : 1));
                if (!ia) return;
                idxs[n] = i16 ? gpu_rd16(ia) : gpu_rd8(ia);
            } else idxs[n] = first + n;
        }
        static std::vector<s32> slot_of;
        std::vector<u32> uniq;
        u32 maxi = 0;
        for (u32 v : idxs) maxi = std::max(maxi, v);
        if (maxi < 0x100000) {
            if (slot_of.size() < maxi + 1) slot_of.resize(maxi + 1);
            for (u32 v : idxs) slot_of[v] = -1;
            for (u32 v : idxs) if (slot_of[v] < 0) { slot_of[v] = (s32)uniq.size(); uniq.push_back(v); }
            std::vector<OutVertex> outv(uniq.size());
            if (g_pica->vs.dec_dirty) shader_decode(g_pica->vs);
            extern void pica_parallel(const std::function<void(int, int)> &f);
            std::function<void(int, int)> job = [&](int part, int np) {
                for (size_t k = part; k < uniq.size(); k += np) {
                    u32 idx = uniq[k];
                    Vec4 attrs[16];
                    for (u32 a = 0; a < nattr && a < 12; a++) {
                        if (fixed_mask & (1u << a) || !elems[a]) { attrs[a] = g_pica->fixed_attr[a]; continue; }
                        Vec4 v{0, 0, 0, 1};
                        u32 va = pa_to_va(src[a] + idx * stride[a]);
                        if (va) {
                            u32 es = etype[a] == 0 || etype[a] == 1 ? 1 : etype[a] == 2 ? 2 : 4;
                            for (u32 e = 0; e < elems[a]; e++) v[e] = read_attr_elem(va + e * es, etype[a]);
                        }
                        attrs[a] = v;
                    }
                    run_vertex(attrs, nattr, outv[k]);
                }
            };
            pica_parallel(job);
            for (u32 v : idxs) assemble(outv[slot_of[v]]);
            return;
        }
    }
    g_pica->pa_index = 0; g_pica->pa_strip_ready = false;
    g_gs_inputs.clear();

    // post-transform cache keyed by index (indexed draws without a GS)
    static OutVertex vcache[256];
    static u32 vtag[256];
    bool use_cache = indexed && !gs_enabled();
    if (use_cache) memset(vtag, 0xFF, sizeof vtag);
    for (u32 n = 0; n < count; n++) {
        u32 idx;
        if (indexed) {
            u32 ia = pa_to_va(ibase + n * (i16 ? 2 : 1));
            if (!ia) return;
            idx = i16 ? gpu_rd16(ia) : gpu_rd8(ia);
        } else idx = first + n;
        if (use_cache && vtag[idx & 255] == idx) { assemble(vcache[idx & 255]); continue; }
        Vec4 attrs[16];
        for (u32 a = 0; a < nattr && a < 12; a++) {
            if (fixed_mask & (1u << a) || !elems[a]) { attrs[a] = g_pica->fixed_attr[a]; continue; }
            Vec4 v{0, 0, 0, 1};
            u32 va = pa_to_va(src[a] + idx * stride[a]);
            if (!va) { attrs[a] = v; continue; }
            u32 es = etype[a] == 0 || etype[a] == 1 ? 1 : etype[a] == 2 ? 2 : 4;
            for (u32 e = 0; e < elems[a]; e++) v[e] = read_attr_elem(va + e * es, etype[a]);
            if (std::isnan(v.x) || std::isnan(v.y)) {
                static int n = 0;
                if (n++ < 6) LOG("[pica] NaN attribute %u idx=%u va=%08x (pa %08x) type=%u elems=%u stride=%u base=%08x words: %08x %08x %08x",
                                 a, idx, va, src[a] + idx * stride[a], etype[a], elems[a], stride[a], base_pa, rd32(va), rd32(va + 4), rd32(va + 8));
            }
            attrs[a] = v;
        }
        if (use_cache) {
            OutVertex ov;
            if (run_vertex(attrs, nattr, ov)) { vcache[idx & 255] = ov; vtag[idx & 255] = idx; assemble(ov); }
        } else process_vertex(attrs, nattr);
    }
}

// ------------------------------------------------------------ clipping
struct ClipV { OutVertex v; };

static OutVertex lerp_v(const OutVertex &a, const OutVertex &b, float t) {
    OutVertex r;
    const float *pa = (const float *)&a, *pb = (const float *)&b;
    float *pr = (float *)&r;
    for (size_t i = 0; i < offsetof(OutVertex, sx) / sizeof(float); i++) pr[i] = pa[i] + (pb[i] - pa[i]) * t;
    return r;
}

static float plane_dist(const OutVertex &v, int p) {
    const Vec4 &q = v.pos;
    switch (p) {
    case 0: return q.w - q.x;
    case 1: return q.w + q.x;
    case 2: return q.w - q.y;
    case 3: return q.w + q.y;
    case 4: return q.z;           // z >= 0 ... PICA clip volume is -w <= z <= 0 in its convention
    case 5: return q.w + q.z;
    case 6: return q.w - 0.00001f;
    }
    return 1;
}

static void raster_tri(const OutVertex &v0, const OutVertex &v1, const OutVertex &v2);

static void triangle(const OutVertex &a, const OutVertex &b, const OutVertex &c) {
    g_dump_tris++;
    if (g_hwr_on) hwr_triangle(a, b, c);
    if (g_pica_dbg < 0) g_pica_dbg = getenv("R3DS_PICA_DBG") ? atoi(getenv("R3DS_PICA_DBG")) : 0;
    if (g_pica_dbg > 0) {
        g_pica_dbg--;
        LOG("[pica] tri pos (%g %g %g %g) (%g %g %g %g) (%g %g %g %g) col %g %g %g %g tc %g %g",
            a.pos.x, a.pos.y, a.pos.z, a.pos.w, b.pos.x, b.pos.y, b.pos.z, b.pos.w, c.pos.x, c.pos.y, c.pos.z, c.pos.w,
            a.color.x, a.color.y, a.color.z, a.color.w, a.tc0[0], a.tc0[1]);
        LOG("[pica]   vp %g x %g xy %08x fb %08x/%08x dim %08x fmt %08x outmap total %u mask %04x map0 %08x map1 %08x",
            f24(R[0x41]), f24(R[0x43]), R[0x68], R[0x11D] << 3, R[0x11C] << 3, R[0x11E], R[0x117], R[0x4F], R[0x2BD], R[0x50], R[0x51]);
    }
    OutVertex buf1[16], buf2[16];
    int n = 3;
    buf1[0] = a; buf1[1] = b; buf1[2] = c;
    OutVertex *in = buf1, *out = buf2;
    for (int p = 0; p < 7; p++) {
        if (p == 4) {
            // PICA z range: 0 >= z >= -w  (z/w in [-1, 0])
        }
        int m = 0;
        for (int i = 0; i < n; i++) {
            const OutVertex &s = in[i], &e = in[(i + 1) % n];
            float ds, de;
            if (p == 4) { ds = -s.pos.z; de = -e.pos.z; }
            else { ds = plane_dist(s, p); de = plane_dist(e, p); }
            if (ds >= 0) { if (m < 15) out[m++] = s; }
            if ((ds >= 0) != (de >= 0)) {
                float t = ds / (ds - de);
                if (m < 15) out[m++] = lerp_v(s, e, t);
            }
        }
        n = m;
        std::swap(in, out);
        if (n < 3) return;
    }
    // viewport transform
    float hw = f24(R[0x41]), hh = f24(R[0x43]);
    float vx = (float)(s32)((R[0x68] & 0x3FF) << 22 >> 22);
    float vy = (float)(s32)(((R[0x68] >> 16) & 0x3FF) << 22 >> 22);
    for (int i = 0; i < n; i++) {
        OutVertex &v = in[i];
        if (!std::isfinite(v.pos.x) || !std::isfinite(v.pos.y) || !std::isfinite(v.pos.z) || !std::isfinite(v.pos.w)) {
            static int bad = 0;
            if (bad++ < 5) LOG("[pica] non-finite clipped vertex (%g %g %g %g) from tri (%g %g %g %g) (%g %g %g %g) (%g %g %g %g)",
                v.pos.x, v.pos.y, v.pos.z, v.pos.w, a.pos.x, a.pos.y, a.pos.z, a.pos.w, b.pos.x, b.pos.y, b.pos.z, b.pos.w, c.pos.x, c.pos.y, c.pos.z, c.pos.w);
            return;
        }
        float iw = 1.0f / v.pos.w;
        v.invw = iw;
        v.sx = (v.pos.x * iw + 1.0f) * hw + vx;
        v.sy = (v.pos.y * iw + 1.0f) * hh + vy;
        v.sz = v.pos.z * iw;
    }
    static int ref = -1;
    if (ref < 0) ref = getenv("R3DS_REF_RASTER") ? 1 : 0;
    for (int i = 1; i + 1 < n; i++) {
        if (ref) raster_tri(in[0], in[i], in[i + 1]);
        else raster_tri_fast(in[0], in[i], in[i + 1]);
    }
}

// ------------------------------------------------------------ textures
static inline u32 etc1_texel(u64 blk, u32 x, u32 y) {
    static const int mods[8][2] = {{2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183}};
    bool flip = (blk >> 32) & 1, diff = (blk >> 33) & 1;
    u32 tbl1 = (blk >> 37) & 7, tbl2 = (blk >> 34) & 7;
    int r1, g1, b1, r2, g2, b2;
    if (diff) {
        int r = (blk >> 59) & 31, g = (blk >> 51) & 31, b = (blk >> 43) & 31;
        int dr = (int)(((blk >> 56) & 7) << 29) >> 29, dg = (int)(((blk >> 48) & 7) << 29) >> 29, db = (int)(((blk >> 40) & 7) << 29) >> 29;
        r1 = r; g1 = g; b1 = b; r2 = r + dr; g2 = g + dg; b2 = b + db;
        r1 = (r1 << 3) | (r1 >> 2); g1 = (g1 << 3) | (g1 >> 2); b1 = (b1 << 3) | (b1 >> 2);
        r2 = ((r2 & 31) << 3) | ((r2 & 31) >> 2); g2 = ((g2 & 31) << 3) | ((g2 & 31) >> 2); b2 = ((b2 & 31) << 3) | ((b2 & 31) >> 2);
    } else {
        r1 = ((blk >> 60) & 15) * 17; r2 = ((blk >> 56) & 15) * 17;
        g1 = ((blk >> 52) & 15) * 17; g2 = ((blk >> 48) & 15) * 17;
        b1 = ((blk >> 44) & 15) * 17; b2 = ((blk >> 40) & 15) * 17;
    }
    bool second = flip ? (y >= 2) : (x >= 2);
    int r = second ? r2 : r1, g = second ? g2 : g1, b = second ? b2 : b1;
    u32 tbl = second ? tbl2 : tbl1;
    u32 i = x * 4 + y;
    u32 lsb = (blk >> i) & 1, msb = (blk >> (16 + i)) & 1;
    int m = mods[tbl][lsb];
    if (msb) m = -m;
    r = std::clamp(r + m, 0, 255); g = std::clamp(g + m, 0, 255); b = std::clamp(b + m, 0, 255);
    return r | (g << 8) | (b << 16) | 0xFF000000u;
}

struct TexUnit { u32 va, w, h, fmt, param, border; bool on; };

static u32 tex_texel(const TexUnit &t, u32 s, u32 tt);
u32 pica_tex_texel(u32 va, u32 w, u32 h, u32 fmt, u32 s, u32 t) {
    TexUnit tu{va, w, h, fmt, 0, 0, true};
    return tex_texel(tu, s, t);
}
static u32 tex_texel(const TexUnit &t, u32 s, u32 tt) {
    u32 y = t.h - 1 - tt;
    u32 fmt = t.fmt;
    if (fmt == 0xC || fmt == 0xD) {
        bool alpha = fmt == 0xD;
        u32 bs = alpha ? 16 : 8;
        u32 tile = (y / 8) * (t.w / 8) + (s / 8);
        u32 sub = ((s / 4) & 1) + 2 * ((y / 4) & 1);
        u32 a = t.va + (tile * 4 + sub) * bs;
        u32 av = 255;
        if (alpha) { u64 pa = gpu_rd64(a); av = ((pa >> (4 * ((s & 3) * 4 + (y & 3)))) & 0xF) * 17; a += 8; }
        u32 c = etc1_texel(gpu_rd64(a), s & 3, y & 3);
        return (c & 0xFFFFFF) | (av << 24);
    }
    u32 texel = ((y & ~7u) * t.w) + ((s & ~7u) * 8) + morton8(s & 7, y & 7);
    switch (fmt) {
    case 0: return fb_decode(gpu_ptr(t.va + texel * 4), 0);
    case 1: return fb_decode(gpu_ptr(t.va + texel * 3), 1);
    case 2: return fb_decode(gpu_ptr(t.va + texel * 2), 3);
    case 3: return fb_decode(gpu_ptr(t.va + texel * 2), 2);
    case 4: return fb_decode(gpu_ptr(t.va + texel * 2), 4);
    case 5: { u32 a = gpu_rd8(t.va + texel * 2), i = gpu_rd8(t.va + texel * 2 + 1); return i | (i << 8) | (i << 16) | (a << 24); }
    case 6: { u32 g = gpu_rd8(t.va + texel * 2), r = gpu_rd8(t.va + texel * 2 + 1); return r | (g << 8) | 0xFF000000u; }
    case 7: { u32 i = gpu_rd8(t.va + texel); return i | (i << 8) | (i << 16) | 0xFF000000u; }
    case 8: { u32 a = gpu_rd8(t.va + texel); return a << 24; }
    case 9: { u32 b = gpu_rd8(t.va + texel); u32 i = (b >> 4) * 17, a = (b & 15) * 17; return i | (i << 8) | (i << 16) | (a << 24); }
    case 10: { u32 b = gpu_rd8(t.va + texel / 2); u32 i = ((texel & 1) ? b >> 4 : b & 15) * 17; return i | (i << 8) | (i << 16) | 0xFF000000u; }
    case 11: { u32 b = gpu_rd8(t.va + texel / 2); u32 a = ((texel & 1) ? b >> 4 : b & 15) * 17; return a << 24; }
    }
    return 0xFFFF00FF;
}

static inline int wrap(int c, int size, u32 mode, bool &border) {
    switch (mode) {
    case 2: c %= size; if (c < 0) c += size; return c;
    case 3: { int p = size * 2; c %= p; if (c < 0) c += p; return c < size ? c : p - 1 - c; }
    case 1: if (c < 0 || c >= size) border = true; return std::clamp(c, 0, size - 1);
    default: return std::clamp(c, 0, size - 1);
    }
}

static u32 tex_sample(const TexUnit &t, float u, float v) {
    if (!t.va || !t.w || !t.h) return 0;
    u32 ws = (t.param >> 12) & 7, wt = (t.param >> 8) & 7;
    bool linear = (t.param >> 1) & 1;
    float fu = fminf(fmaxf(u * t.w, -8388608.0f), 8388608.0f), fv = fminf(fmaxf(v * t.h, -8388608.0f), 8388608.0f);
    if (!linear) {
        bool bd = false;
        int s = wrap(f2i(floorf(fu)), t.w, ws, bd), tt = wrap(f2i(floorf(fv)), t.h, wt, bd);
        if (bd) return t.border;
        return tex_texel(t, s, tt);
    }
    fu -= 0.5f; fv -= 0.5f;
    int s0 = f2i(floorf(fu)), t0 = f2i(floorf(fv));
    float fs = fu - s0, ft = fv - t0;
    u32 c[4];
    for (int k = 0; k < 4; k++) {
        bool bd = false;
        int s = wrap(s0 + (k & 1), t.w, ws, bd), tt = wrap(t0 + (k >> 1), t.h, wt, bd);
        c[k] = bd ? t.border : tex_texel(t, s, tt);
    }
    u32 out = 0;
    for (int ch = 0; ch < 4; ch++) {
        float a = (c[0] >> (8 * ch)) & 0xFF, b = (c[1] >> (8 * ch)) & 0xFF, cc = (c[2] >> (8 * ch)) & 0xFF, d = (c[3] >> (8 * ch)) & 0xFF;
        float top = a + (b - a) * fs, bot = cc + (d - cc) * fs;
        u32 r = f2u(top + (bot - top) * ft + 0.5f);
        out |= std::min(r, 255u) << (8 * ch);
    }
    return out;
}

// ------------------------------------------------------------ fragment ops
struct C4 { int r, g, b, a; };
static inline C4 unpack(u32 c) { return {(int)(c & 0xFF), (int)((c >> 8) & 0xFF), (int)((c >> 16) & 0xFF), (int)(c >> 24)}; }
static inline u32 pack(C4 c) { return (u32)std::clamp(c.r, 0, 255) | ((u32)std::clamp(c.g, 0, 255) << 8) | ((u32)std::clamp(c.b, 0, 255) << 16) | ((u32)std::clamp(c.a, 0, 255) << 24); }

static inline bool cmpf(u32 f, int a, int b) {
    switch (f) { case 0: return false; case 1: return true; case 2: return a == b; case 3: return a != b;
    case 4: return a < b; case 5: return a <= b; case 6: return a > b; default: return a >= b; }
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
    case 1: return a * b / 255;
    case 2: return std::min(a + b, 255);
    case 3: return std::clamp(a + b - 128, 0, 255);
    case 4: return (a * c + b * (255 - c)) / 255;
    case 5: return std::max(a - b, 0);
    case 8: return std::min((a * b + 255 * c) / 255, 255);
    case 9: return std::min(a + b, 255) * c / 255;
    default: return a;
    }
}

static inline float blend_factor(u32 f, int ch, C4 s, C4 d, C4 k) {
    auto n = [](int v) { return v / 255.0f; };
    int sc = ch == 0 ? s.r : ch == 1 ? s.g : ch == 2 ? s.b : s.a;
    int dc = ch == 0 ? d.r : ch == 1 ? d.g : ch == 2 ? d.b : d.a;
    int kc = ch == 0 ? k.r : ch == 1 ? k.g : ch == 2 ? k.b : k.a;
    switch (f) {
    case 0: return 0; case 1: return 1;
    case 2: return n(sc); case 3: return 1 - n(sc);
    case 4: return n(dc); case 5: return 1 - n(dc);
    case 6: return n(s.a); case 7: return 1 - n(s.a);
    case 8: return n(d.a); case 9: return 1 - n(d.a);
    case 10: return n(kc); case 11: return 1 - n(kc);
    case 12: return n(k.a); case 13: return 1 - n(k.a);
    case 14: return ch == 3 ? 1 : std::min(n(s.a), 1 - n(d.a));
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

u64 g_frag_count = 0;

static void raster_tri(const OutVertex &v0in, const OutVertex &v1in, const OutVertex &v2in) {
    const OutVertex *v[3] = {&v0in, &v1in, &v2in};
    auto orient = [](const OutVertex *a, const OutVertex *b, const OutVertex *c) {
        return (b->sx - a->sx) * (c->sy - a->sy) - (b->sy - a->sy) * (c->sx - a->sx);
    };
    float area = orient(v[0], v[1], v[2]);
    u32 cull = R[0x40] & 3;
    if (cull == 1 && area > 0) return;
    if (cull == 2 && area < 0) return;
    if (area == 0) return;
    if (area < 0) { std::swap(v[1], v[2]); area = -area; }

    u32 color_pa = R[0x11D] << 3, depth_pa = R[0x11C] << 3;
    u32 fbw = R[0x11E] & 0x7FF, fbh = ((R[0x11E] >> 12) & 0x3FF) + 1;
    u8 *colbuf = pa_ptr(color_pa), *dbuf = pa_ptr(depth_pa);
    if (!colbuf || !fbw) return;
    u32 cfmt = (R[0x117] >> 16) & 7, dfmt = R[0x116] & 3;
    u32 cbpp = fb_bpp(cfmt), dbpp = dfmt == 0 ? 2 : dfmt == 3 ? 4 : 3;
    bool can_cw = R[0x113] & 0xF, can_cr = R[0x112] & 0xF;
    bool can_dw = (R[0x115] & 2) != 0, can_dr = (R[0x114] & 2) != 0;
    bool can_sw = (R[0x115] & 1) != 0;

    // fixed-point 1/16 pixel
    int x[3], y[3];
    for (int i = 0; i < 3; i++) { x[i] = (int)llrintf(v[i]->sx * 16.0f); y[i] = (int)llrintf(v[i]->sy * 16.0f); }
    int minx = std::min({x[0], x[1], x[2]}) & ~0xF, maxx = std::max({x[0], x[1], x[2]});
    int miny = std::min({y[0], y[1], y[2]}) & ~0xF, maxy = std::max({y[0], y[1], y[2]});
    int sx0 = 0, sy0 = 0, sx1 = (int)fbw * 16, sy1 = (int)fbh * 16;
    u32 smode = R[0x65] & 3;
    if (smode == 3) {
        sx0 = std::max(sx0, (int)(R[0x66] & 0x3FF) * 16); sy0 = std::max(sy0, (int)((R[0x66] >> 16) & 0x3FF) * 16);
        sx1 = std::min(sx1, ((int)(R[0x67] & 0x3FF) + 1) * 16); sy1 = std::min(sy1, ((int)((R[0x67] >> 16) & 0x3FF) + 1) * 16);
    }
    minx = std::max(minx, sx0 & ~0xF); miny = std::max(miny, sy0 & ~0xF);
    maxx = std::min(maxx, sx1 - 1); maxy = std::min(maxy, sy1 - 1);
    if (minx > maxx || miny > maxy) return;

    // edge setup (CCW, y up)
    auto edge = [](int ax, int ay, int bx, int by, int px, int py) -> s64 {
        return (s64)(bx - ax) * (py - ay) - (s64)(by - ay) * (px - ax);
    };
    auto is_topleft = [](int ax, int ay, int bx, int by) {
        return (ay == by && bx < ax) || (by < ay);
    };
    int bias[3];
    bias[0] = is_topleft(x[1], y[1], x[2], y[2]) ? 0 : -1;
    bias[1] = is_topleft(x[2], y[2], x[0], y[0]) ? 0 : -1;
    bias[2] = is_topleft(x[0], y[0], x[1], y[1]) ? 0 : -1;
    s64 tot = edge(x[0], y[0], x[1], y[1], x[2], y[2]);
    if (tot <= 0) return;

    // texture units
    TexUnit tex[3];
    u32 tcfg = R[0x80];
    const u32 tbase[3] = {0x81, 0x91, 0x99};
    for (int i = 0; i < 3; i++) {
        u32 b = tbase[i];
        tex[i].on = (tcfg >> i) & 1;
        tex[i].border = R[b];
        tex[i].h = R[b + 1] & 0x7FF; tex[i].w = (R[b + 1] >> 16) & 0x7FF;
        tex[i].param = R[b + 2];
        tex[i].va = pa_to_va(R[b + 4] << 3);
        tex[i].fmt = (i == 0 ? R[0x8E] : R[b + 5]) & 0xF;
        tex[i].border = R[b];
    }
    bool tex2_uses_tc1 = (tcfg >> 13) & 1;

    // tev
    const u32 tev_base[6] = {0xC0, 0xC8, 0xD0, 0xD8, 0xF0, 0xF8};
    u32 upd = R[0xE0];
    C4 buf_init = unpack(R[0xFD]);
    bool lighting = R[0x8F] & 1;

    // output merger
    u32 colop = R[0x100], blend = R[0x101], lop = R[0x102] & 0xF;
    C4 bcol = unpack(R[0x103]);
    u32 at = R[0x104], st = R[0x105], sop = R[0x106], dcm = R[0x107];
    bool alpha_test = at & 1; u32 at_func = (at >> 4) & 7; int at_ref = (at >> 8) & 0xFF;
    bool st_en = (st & 1) && dfmt == 3; u32 st_func = (st >> 4) & 7; u8 st_wmask = (st >> 8) & 0xFF, st_ref = (st >> 16) & 0xFF, st_imask = st >> 24;
    bool depth_test = dcm & 1; u32 dfunc = (dcm >> 4) & 7; bool dwrite = (dcm >> 12) & 1;
    u32 wmask = (dcm >> 8) & 0xF;
    float dscale = f24(R[0x4D]), doff = f24(R[0x4E]);
    bool wbuf = !(R[0x6D] & 1);

    const OutVertex &A = *v[0], &B = *v[1], &C = *v[2];
    for (int py = miny + 8; py <= maxy; py += 16) {
        for (int px = minx + 8; px <= maxx; px += 16) {
            s64 w0 = edge(x[1], y[1], x[2], y[2], px, py) + bias[0];
            s64 w1 = edge(x[2], y[2], x[0], y[0], px, py) + bias[1];
            s64 w2 = edge(x[0], y[0], x[1], y[1], px, py) + bias[2];
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            int fx = px >> 4, fy = py >> 4;
            if (smode == 1) {  // exclude
                int x1 = R[0x66] & 0x3FF, y1 = (R[0x66] >> 16) & 0x3FF, x2 = R[0x67] & 0x3FF, y2 = (R[0x67] >> 16) & 0x3FF;
                if (fx >= x1 && fx <= x2 && fy >= y1 && fy <= y2) continue;
            }
            float b0 = (float)w0 / tot, b1 = (float)w1 / tot, b2 = (float)w2 / tot;
            float iw = b0 * A.invw + b1 * B.invw + b2 * C.invw;
            float W = 1.0f / iw;
            float p0 = b0 * A.invw * W, p1 = b1 * B.invw * W, p2 = b2 * C.invw * W;
            auto I = [&](float a, float b, float c) { return a * p0 + b * p1 + c * p2; };
            float z = b0 * A.sz + b1 * B.sz + b2 * C.sz;
            float depth = z * dscale + doff;
            if (wbuf) depth = (z * dscale + doff) * W;  // approximate W-buffer
            depth = std::clamp(depth, 0.0f, 1.0f);

            u32 row = fbh - 1 - fy;
            u32 coff = tiled_offset(fx, row, fbw, cbpp);
            u32 doffs = tiled_offset(fx, row, fbw, dbpp);

            C4 prim = {f2i(I(A.color.x, B.color.x, C.color.x) * 255 + 0.5f), f2i(I(A.color.y, B.color.y, C.color.y) * 255 + 0.5f),
                       f2i(I(A.color.z, B.color.z, C.color.z) * 255 + 0.5f), f2i(I(A.color.w, B.color.w, C.color.w) * 255 + 0.5f)};
            C4 texc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
            if (tex[0].on) texc[0] = unpack(tex_sample(tex[0], I(A.tc0[0], B.tc0[0], C.tc0[0]), I(A.tc0[1], B.tc0[1], C.tc0[1])));
            if (tex[1].on) texc[1] = unpack(tex_sample(tex[1], I(A.tc1[0], B.tc1[0], C.tc1[0]), I(A.tc1[1], B.tc1[1], C.tc1[1])));
            if (tex[2].on) {
                float u = tex2_uses_tc1 ? I(A.tc1[0], B.tc1[0], C.tc1[0]) : I(A.tc2[0], B.tc2[0], C.tc2[0]);
                float vv = tex2_uses_tc1 ? I(A.tc1[1], B.tc1[1], C.tc1[1]) : I(A.tc2[1], B.tc2[1], C.tc2[1]);
                texc[2] = unpack(tex_sample(tex[2], u, vv));
            }
            C4 fprim = {0, 0, 0, 0}, fsec = {0, 0, 0, 0};
            if (lighting) {
                float q[4] = {I(A.quat.x, B.quat.x, C.quat.x), I(A.quat.y, B.quat.y, C.quat.y), I(A.quat.z, B.quat.z, C.quat.z), I(A.quat.w, B.quat.w, C.quat.w)};
                float vw[3] = {I(A.view[0], B.view[0], C.view[0]), I(A.view[1], B.view[1], C.view[1]), I(A.view[2], B.view[2], C.view[2])};
                u32 pp, ss;
                pica_lighting(A, vw, q, &pp, &ss);
                fprim = unpack(pp); fsec = unpack(ss);
            }
            // texture combiners
            C4 out = {0, 0, 0, 0}, cbuf = {0, 0, 0, 0}, next_buf = buf_init;
            for (int s = 0; s < 6; s++) {
                u32 b = tev_base[s];
                u32 srcr = R[b], opr = R[b + 1], cmb = R[b + 2];
                C4 kc = unpack(R[b + 3]);
                u32 scale = R[b + 4];
                auto src = [&](u32 id) -> C4 {
                    switch (id) {
                    case 0: return prim; case 1: return fprim; case 2: return fsec;
                    case 3: return texc[0]; case 4: return texc[1]; case 5: return texc[2]; case 6: return texc[3];
                    case 13: return cbuf; case 14: return kc; case 15: return out;
                    }
                    return prim;
                };
                int c[3][3];
                for (int k = 0; k < 3; k++) color_mod((opr >> (4 * k)) & 0xF, src((srcr >> (4 * k)) & 0xF), c[k]);
                int a[3];
                for (int k = 0; k < 3; k++) a[k] = alpha_mod((opr >> (12 + 4 * k)) & 7, src((srcr >> (16 + 4 * k)) & 0xF));
                u32 cop = cmb & 0xF, aop = (cmb >> 16) & 0xF;
                int rc[3];
                if (cop == 6 || cop == 7) {
                    int d = ((c[0][0] - 128) * (c[1][0] - 128) + (c[0][1] - 128) * (c[1][1] - 128) + (c[0][2] - 128) * (c[1][2] - 128)) * 4 / 255;
                    d = std::clamp(d, 0, 255);
                    rc[0] = rc[1] = rc[2] = d;
                } else for (int k = 0; k < 3; k++) rc[k] = combine1(cop, c[0][k], c[1][k], c[2][k]);
                int ra = (cop == 7) ? rc[0] : combine1(aop == 6 || aop == 7 ? 0 : aop, a[0], a[1], a[2]);
                int cs = 1 << (scale & 3), as = 1 << ((scale >> 16) & 3);
                out = {std::min(rc[0] * cs, 255), std::min(rc[1] * cs, 255), std::min(rc[2] * cs, 255), std::min(ra * as, 255)};
                cbuf = next_buf;
                if (s < 4) {
                    if ((upd >> (8 + s)) & 1) { next_buf.r = out.r; next_buf.g = out.g; next_buf.b = out.b; }
                    if ((upd >> (12 + s)) & 1) next_buf.a = out.a;
                }
            }
            if (alpha_test && !cmpf(at_func, out.a, at_ref)) continue;

            // depth / stencil
            u8 *dp = dbuf ? dbuf + doffs : nullptr;
            u32 dmax = dfmt == 0 ? 0xFFFF : 0xFFFFFF;
            u32 zval = f2u(depth * dmax);
            u32 stored = 0; u8 sten = 0;
            if (dp) {
                if (dfmt == 0) stored = dp[0] | (dp[1] << 8);
                else { stored = dp[0] | (dp[1] << 8) | (dp[2] << 16); if (dfmt == 3) sten = dp[3]; }
            }
            if (st_en && dp) {
                bool pass = cmpf(st_func, st_ref & st_imask, sten & st_imask);
                if (!pass) {
                    if (can_sw) { u8 ns = stencil_op(sop & 7, sten, st_ref); dp[3] = (sten & ~st_wmask) | (ns & st_wmask); }
                    continue;
                }
            }
            if (depth_test && dp && can_dr) {
                if (!cmpf(dfunc, (int)zval, (int)stored)) {
                    if (st_en && can_sw) { u8 ns = stencil_op((sop >> 4) & 7, sten, st_ref); dp[3] = (sten & ~st_wmask) | (ns & st_wmask); }
                    continue;
                }
            }
            if (st_en && dp && can_sw) { u8 ns = stencil_op((sop >> 8) & 7, sten, st_ref); dp[3] = (sten & ~st_wmask) | (ns & st_wmask); }
            if (dwrite && dp && can_dw) {
                dp[0] = zval & 0xFF; dp[1] = (zval >> 8) & 0xFF;
                if (dfmt != 0) dp[2] = (zval >> 16) & 0xFF;
            }
            if (!can_cw) continue;
            // blending
            u8 *cp = colbuf + coff;
            C4 dst = unpack(fb_decode(cp, cfmt));
            C4 res;
            if (colop & 0x100) {
                u32 eqc = blend & 7, eqa = (blend >> 8) & 7;
                u32 sfc = (blend >> 16) & 0xF, dfc = (blend >> 20) & 0xF, sfa = (blend >> 24) & 0xF, dfa = blend >> 28;
                float r[4];
                for (int ch = 0; ch < 4; ch++) {
                    int sc = ch == 0 ? out.r : ch == 1 ? out.g : ch == 2 ? out.b : out.a;
                    int dc = ch == 0 ? dst.r : ch == 1 ? dst.g : ch == 2 ? dst.b : dst.a;
                    float sf = blend_factor(ch == 3 ? sfa : sfc, ch, out, dst, bcol);
                    float df = blend_factor(ch == 3 ? dfa : dfc, ch, out, dst, bcol);
                    u32 eq = ch == 3 ? eqa : eqc;
                    float S = sc * sf, D = dc * df;
                    switch (eq) {
                    case 0: r[ch] = S + D; break; case 1: r[ch] = S - D; break; case 2: r[ch] = D - S; break;
                    case 3: r[ch] = std::min(sc, dc); break; default: r[ch] = std::max(sc, dc);
                    }
                }
                res = {f2i(r[0] + 0.5f), f2i(r[1] + 0.5f), f2i(r[2] + 0.5f), f2i(r[3] + 0.5f)};
            } else {
                res = unpack(logic_op(lop, pack(out), pack(dst)));
            }
            if (wmask != 0xF) {
                if (!(wmask & 1)) res.r = dst.r;
                if (!(wmask & 2)) res.g = dst.g;
                if (!(wmask & 4)) res.b = dst.b;
                if (!(wmask & 8)) res.a = dst.a;
            }
            fb_encode(cp, cfmt, pack(res));
            g_frag_count++;
            (void)can_cr;
        }
    }
}

// Fragment lighting (subset): diffuse + specular from up to 8 lights, LUT D0.
void pica_lighting(const OutVertex &, const float *view, const float *q, u32 *prim, u32 *sec) {
    // normal from quaternion: rotate (0,0,1)
    float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    float ql = sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
    if (ql > 0) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; }
    float nx = 2 * (qx * qz + qw * qy), ny = 2 * (qy * qz - qw * qx), nz = 1 - 2 * (qx * qx + qy * qy);
    float vl = sqrtf(view[0] * view[0] + view[1] * view[1] + view[2] * view[2]);
    float vx = vl > 0 ? view[0] / vl : 0, vy = vl > 0 ? view[1] / vl : 0, vz = vl > 0 ? view[2] / vl : 1;
    u32 amb = R[0x1C0];
    float pr = ((amb >> 20) & 0xFF) / 255.0f, pg = ((amb >> 10) & 0xFF) / 255.0f, pb = (amb & 0xFF) / 255.0f;
    float sr = 0, sg = 0, sb = 0;
    u32 nl = (R[0x1C2] & 7) + 1;
    for (u32 i = 0; i < nl; i++) {
        u32 id = (R[0x1D9] >> (4 * i)) & 7;
        u32 lb = 0x140 + id * 0x10;
        auto col = [](u32 c, float &r, float &g, float &b) { r = ((c >> 20) & 0xFF) / 255.0f; g = ((c >> 10) & 0xFF) / 255.0f; b = (c & 0xFF) / 255.0f; };
        float s0r, s0g, s0b, s1r, s1g, s1b, dr, dg, db, ar, ag, ab;
        col(R[lb + 0], s0r, s0g, s0b); col(R[lb + 1], s1r, s1g, s1b); col(R[lb + 2], dr, dg, db); col(R[lb + 3], ar, ag, ab);
        float lx = f16(R[lb + 4] & 0xFFFF), ly = f16(R[lb + 4] >> 16), lz = f16(R[lb + 5] & 0xFFFF);
        bool directional = (R[lb + 9] & 1);
        if (!directional) { lx += view[0]; ly += view[1]; lz += view[2]; }
        float ll = sqrtf(lx * lx + ly * ly + lz * lz);
        if (ll > 0) { lx /= ll; ly /= ll; lz /= ll; }
        float ndl = nx * lx + ny * ly + nz * lz;
        float dot = std::max(ndl, 0.0f);
        pr += ar + dr * dot; pg += ag + dg * dot; pb += ab + db * dot;
        float hx = lx + vx, hy = ly + vy, hz = lz + vz, hl = sqrtf(hx * hx + hy * hy + hz * hz);
        if (hl > 0 && ndl > 0) {
            float nh = std::max((nx * hx + ny * hy + nz * hz) / hl, 0.0f);
            float spec = powf(nh, 16.0f);
            sr += s0r * spec; sg += s0g * spec; sb += s0b * spec;
        }
    }
    auto p8 = [](float r, float g, float b) { return (u32)(std::clamp(r, 0.f, 1.f) * 255) | ((u32)(std::clamp(g, 0.f, 1.f) * 255) << 8) | ((u32)(std::clamp(b, 0.f, 1.f) * 255) << 16) | 0xFF000000u; };
    *prim = p8(pr, pg, pb);
    *sec = p8(sr, sg, sb);
}
