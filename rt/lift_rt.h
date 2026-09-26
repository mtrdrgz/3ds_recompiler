// Support header for generated (lifted) code.
#pragma once
#include "cpu.h"
#include "mem.h"
#include <atomic>
#include <cmath>
#include <cstring>

extern std::atomic<int> g_irq_pending;
[[noreturn]] void rt_trap(Cpu &c, u32 pc, const char *msg);

#define RD8(a)  (*(const u8 *)(M + (u32)(a)))
#define RD16(a) ld16(M + (u32)(a))
#define RD32(a) ld32(M + (u32)(a))
#define RD64(a) ld64(M + (u32)(a))
#define WR8(a, v)  (*(u8 *)(M + (u32)(a)) = (u8)(v))
#define WR16(a, v) st16(M + (u32)(a), (u16)(v))
#define WR32(a, v) st32(M + (u32)(a), (u32)(v))
#define WR64(a, v) st64(M + (u32)(a), (u64)(v))

static inline u16 ld16(const u8 *p) { u16 v; memcpy(&v, p, 2); return v; }
static inline u32 ld32(const u8 *p) { u32 v; memcpy(&v, p, 4); return v; }
static inline u64 ld64(const u8 *p) { u64 v; memcpy(&v, p, 8); return v; }
static inline void st16(u8 *p, u16 v) { memcpy(p, &v, 2); }
static inline void st32(u8 *p, u32 v) { memcpy(p, &v, 4); }
static inline void st64(u8 *p, u64 v) { memcpy(p, &v, 8); }

#define PROLOGUE \
    u8 *const M = g_base; \
    u32 r0 = c.r[0], r1 = c.r[1], r2 = c.r[2], r3 = c.r[3], r4 = c.r[4], r5 = c.r[5], r6 = c.r[6], r7 = c.r[7]; \
    u32 r8 = c.r[8], r9 = c.r[9], r10 = c.r[10], r11 = c.r[11], r12 = c.r[12], r13 = c.r[13], r14 = c.r[14]; \
    u32 fn = c.n, fz = c.z, fc = c.c, fv = c.v; \
    u32 npc;
#define SAVE() do { c.r[0] = r0; c.r[1] = r1; c.r[2] = r2; c.r[3] = r3; c.r[4] = r4; c.r[5] = r5; c.r[6] = r6; c.r[7] = r7; \
    c.r[8] = r8; c.r[9] = r9; c.r[10] = r10; c.r[11] = r11; c.r[12] = r12; c.r[13] = r13; c.r[14] = r14; \
    c.n = fn; c.z = fz; c.c = fc; c.v = fv; } while (0)
#define LOAD() do { r0 = c.r[0]; r1 = c.r[1]; r2 = c.r[2]; r3 = c.r[3]; r4 = c.r[4]; r5 = c.r[5]; r6 = c.r[6]; r7 = c.r[7]; \
    r8 = c.r[8]; r9 = c.r[9]; r10 = c.r[10]; r11 = c.r[11]; r12 = c.r[12]; r13 = c.r[13]; r14 = c.r[14]; \
    fn = c.n; fz = c.z; fc = c.c; fv = c.v; } while (0)
#define EXIT(e) do { npc = (e); goto out; } while (0)
#define OUT out: SAVE(); return npc;
#define POLL(t) do { if (__builtin_expect(g_irq_pending.load(std::memory_order_relaxed), 0)) EXIT(t); } while (0)
#define MISS(p) do { SAVE(); return rt_miss(c, p); } while (0)
#define FALLBACK(p) do { SAVE(); return interp_run(c, p); } while (0)
#define SVC(imm, next) do { SAVE(); c.r[15] = (next) & ~1u; c.thumb = (next) & 1; rt_svc(c, imm); LOAD(); } while (0)
#define TRAP(msg) do { SAVE(); rt_trap(c, pc, msg); } while (0)

static inline u32 ror32(u32 v, u32 s) { s &= 31; return s ? (v >> s) | (v << (32 - s)) : v; }
static inline u32 clz32(u32 v) { return v ? __builtin_clz(v) : 32; }
static inline u32 rev16(u32 v) { return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu); }

static inline u32 adds(u32 a, u32 b, u32 cin, u32 &n, u32 &z, u32 &c, u32 &v) {
    u64 r = (u64)a + b + cin;
    u32 res = (u32)r;
    c = (u32)(r >> 32);
    v = ((a ^ res) & (b ^ res)) >> 31;
    n = res >> 31; z = res == 0;
    return res;
}

// register-specified shifts (amount already masked to 8 bits)
static inline u32 lsl_r(u32 v, u32 a) { return a >= 32 ? 0 : v << a; }
static inline u32 lsr_r(u32 v, u32 a) { return a >= 32 ? 0 : v >> a; }
static inline u32 asr_r(u32 v, u32 a) { return a >= 32 ? (u32)((s32)v >> 31) : (u32)((s32)v >> a); }
static inline u32 ror_r(u32 v, u32 a) { return ror32(v, a & 31); }
static inline u32 lsl_rc(u32 v, u32 a, u32 &c) {
    if (a == 0) return v;
    if (a < 32) { c = (v >> (32 - a)) & 1; return v << a; }
    if (a == 32) { c = v & 1; return 0; }
    c = 0; return 0;
}
static inline u32 lsr_rc(u32 v, u32 a, u32 &c) {
    if (a == 0) return v;
    if (a < 32) { c = (v >> (a - 1)) & 1; return v >> a; }
    if (a == 32) { c = v >> 31; return 0; }
    c = 0; return 0;
}
static inline u32 asr_rc(u32 v, u32 a, u32 &c) {
    if (a == 0) return v;
    if (a < 32) { c = ((u32)((s32)v >> (a - 1))) & 1; return (u32)((s32)v >> a); }
    c = v >> 31; return c ? 0xFFFFFFFFu : 0;
}
static inline u32 ror_rc(u32 v, u32 a, u32 &c) {
    if (a == 0) return v;
    a &= 31;
    if (a == 0) { c = v >> 31; return v; }
    c = (v >> (a - 1)) & 1;
    return ror32(v, a);
}

// saturating
static inline s32 sat32(Cpu &c, s64 v) {
    if (v > 0x7FFFFFFF) { c.q = 1; return 0x7FFFFFFF; }
    if (v < -0x80000000ll) { c.q = 1; return (s32)0x80000000; }
    return (s32)v;
}
static inline u32 qadd(Cpu &c, u32 m, u32 n) { return (u32)sat32(c, (s64)(s32)m + (s32)n); }
static inline u32 qsub(Cpu &c, u32 m, u32 n) { return (u32)sat32(c, (s64)(s32)m - (s32)n); }
static inline u32 qdadd(Cpu &c, u32 m, u32 n) { s32 d = sat32(c, 2 * (s64)(s32)n); return (u32)sat32(c, (s64)(s32)m + d); }
static inline u32 qdsub(Cpu &c, u32 m, u32 n) { s32 d = sat32(c, 2 * (s64)(s32)n); return (u32)sat32(c, (s64)(s32)m - d); }
static inline u32 smla_q(Cpu &c, u32 prod, u32 acc) {
    s64 r = (s64)(s32)prod + (s32)acc;
    if (r != (s32)r) c.q = 1;
    return (u32)r;
}
static inline u32 ssat(Cpu &c, s32 v, int bits) {
    s32 mx = (1 << (bits - 1)) - 1, mn = -(1 << (bits - 1));
    if (v > mx) { c.q = 1; return (u32)mx; }
    if (v < mn) { c.q = 1; return (u32)mn; }
    return (u32)v;
}
static inline u32 usat(Cpu &c, s32 v, int bits) {
    s32 mx = bits >= 31 ? 0x7FFFFFFF : (1 << bits) - 1;
    if (v > mx) { c.q = 1; return (u32)mx; }
    if (v < 0) { c.q = 1; return 0; }
    return (u32)v;
}
static inline u32 ssat16(Cpu &c, u32 v, int bits) {
    return (ssat(c, (s16)(v & 0xFFFF), bits) & 0xFFFF) | (ssat(c, (s16)(v >> 16), bits) << 16);
}
static inline u32 usat16(Cpu &c, u32 v, int bits) {
    return (usat(c, (s16)(v & 0xFFFF), bits) & 0xFFFF) | (usat(c, (s16)(v >> 16), bits) << 16);
}
static inline u32 sel_ge(Cpu &c, u32 n, u32 m) {
    u32 r = 0;
    for (int i = 0; i < 4; i++) r |= (((c.ge >> i) & 1) ? n : m) & (0xFFu << (8 * i));
    return r;
}
static inline u32 usad8(u32 a, u32 b) {
    u32 s = 0;
    for (int i = 0; i < 4; i++) { int x = (a >> (8 * i)) & 0xFF, y = (b >> (8 * i)) & 0xFF; s += x > y ? x - y : y - x; }
    return s;
}

// ARMv6 parallel add/subtract. prefixes: s (signed, sets GE), u (unsigned, sets GE),
// q/uq (saturating), sh/uh (halving)
template <int K> static inline s32 lane(u32 v, int i, bool sgn) {
    if (K == 16) { u32 x = (v >> (16 * i)) & 0xFFFF; return sgn ? (s32)(s16)x : (s32)x; }
    u32 x = (v >> (8 * i)) & 0xFF; return sgn ? (s32)(s8)x : (s32)x;
}
// helper for exchange variants: halfword j of m
static inline s32 b16x(u32 m, int j, bool sgn) { u32 x = (m >> (16 * j)) & 0xFFFF; return sgn ? (s32)(s16)x : (s32)x; }
enum ParOp { P_ADD16, P_ASX, P_SAX, P_SUB16, P_ADD8, P_SUB8 };
template <int MODE> static inline u32 par(Cpu &c, u32 n, u32 m, ParOp op) {
    // MODE: 0 s, 1 q, 2 sh, 3 u, 4 uq, 5 uh
    bool sgn = MODE <= 2;
    bool b8 = op == P_ADD8 || op == P_SUB8;
    int nl = b8 ? 4 : 2, bits = b8 ? 8 : 16;
    u32 r = 0, ge = 0;
    for (int i = 0; i < nl; i++) {
        s32 a, b, res;
        if (b8) { a = lane<8>(n, i, sgn); b = lane<8>(m, i, sgn); }
        else { a = lane<16>(n, i, sgn); b = lane<16>(m, i, sgn); }
        bool sub;
        switch (op) {
        case P_ADD16: case P_ADD8: sub = false; break;
        case P_SUB16: case P_SUB8: sub = true; break;
        case P_ASX: sub = (i == 0); b = b16x(m, i == 0 ? 1 : 0, sgn); break;
        case P_SAX: sub = (i == 1); b = b16x(m, i == 0 ? 1 : 0, sgn); break;
        }
        res = sub ? a - b : a + b;
        s32 lo = sgn ? -(1 << (bits - 1)) : 0, hi = sgn ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
        if (MODE == 0) { if (res >= 0) ge |= (b8 ? 1 : 3) << (i * (b8 ? 1 : 2)); }
        if (MODE == 3) { if (sub ? res >= 0 : res > hi) ge |= (b8 ? 1 : 3) << (i * (b8 ? 1 : 2)); }
        if (MODE == 1 || MODE == 4) res = res < lo ? lo : res > hi ? hi : res;
        if (MODE == 2 || MODE == 5) res >>= 1;
        r |= ((u32)res & ((1u << bits) - 1)) << (i * bits);
    }
    if (MODE == 0 || MODE == 3) c.ge = ge;
    return r;
}
#define PAR_DEF(pfx, mode) \
    static inline u32 par_##pfx##add16(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_ADD16); } \
    static inline u32 par_##pfx##asx(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_ASX); } \
    static inline u32 par_##pfx##sax(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_SAX); } \
    static inline u32 par_##pfx##sub16(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_SUB16); } \
    static inline u32 par_##pfx##add8(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_ADD8); } \
    static inline u32 par_##pfx##sub8(Cpu &c, u32 n, u32 m) { return par<mode>(c, n, m, P_SUB8); }
PAR_DEF(s, 0) PAR_DEF(q, 1) PAR_DEF(sh, 2) PAR_DEF(u, 3) PAR_DEF(uq, 4) PAR_DEF(uh, 5)

// VFP
#if defined(__EMSCRIPTEN__) || defined(R3DS_SOFT_FP)
// WebAssembly has no rounding-mode / flush-to-zero control: arithmetic is
// always round-to-nearest with denormals (see docs/RECOMP2.md, "Web build").
// Conversions that depend on the mode go through vfp_round(), which is exact.
static inline void host_fpscr(u32) {}
#elif defined(__aarch64__)
extern int g_host_fp_plain;   // R3DS_FP_PLAIN=1: ignore FPSCR (mimics the wasm build)
// AArch64 FPCR has the guest FPSCR layout for these bits: RMode [23:22]
// (RN, RP, RM, RZ) and FZ [24], which flushes denormal inputs and results.
static inline void host_fpscr(u32 fpscr) {
    if (g_host_fp_plain) fpscr = 0;
    u64 cur;
    __asm__ volatile("mrs %0, fpcr" : "=r"(cur));
    u64 m = (cur & ~0x01C00000ull) | (fpscr & 0x01C00000u);
    if (m != cur) __asm__ volatile("msr fpcr, %0" : : "r"(m));
}
#else
#include <xmmintrin.h>
extern int g_host_fp_plain;   // R3DS_FP_PLAIN=1: ignore FPSCR (mimics the wasm build)
// keep the host SSE rounding / flush-to-zero in sync with the guest FPSCR
static inline void host_fpscr(u32 fpscr) {
    if (g_host_fp_plain) fpscr = 0;
    u32 m = _mm_getcsr() & ~0xE040u;
    static const u32 rc[4] = {0x0000, 0x4000, 0x2000, 0x6000};   // RN, RP, RM, RZ
    m |= rc[(fpscr >> 22) & 3];
    if (fpscr & (1u << 24)) m |= 0x8040;   // FZ -> FTZ + DAZ
    _mm_setcsr(m);
}
#endif
static inline float dn_f(float x) { if (__builtin_expect(x != x, 0)) { u32 n = 0x7FC00000u; memcpy(&x, &n, 4); } return x; }
static inline double dn_d(double x) { if (__builtin_expect(x != x, 0)) { u64 n = 0x7FF8000000000000ull; memcpy(&x, &n, 8); } return x; }
static inline float vsqrt_f(float x) { if (x < 0 || x != x) { u32 n = 0x7FC00000u; float f; memcpy(&f, &n, 4); return f; } return sqrtf(x); }
static inline double vsqrt_d(double x) { if (x < 0 || x != x) { u64 n = 0x7FF8000000000000ull; double f; memcpy(&f, &n, 8); return f; } return sqrt(x); }

// ---- VFP arithmetic. Natively the host MXCSR mirrors FPSCR (rounding mode,
// flush-to-zero), so the plain C++ operator is exact. Where the host cannot
// do that (WebAssembly, or -DFL_SOFT_FP to test the path natively) the
// FPSCR modes are emulated: single-precision results are computed exactly
// (or correctly rounded) in double and rounded to float per FPSCR.RMode,
// with FZ flushing denormal operands and results. Double precision gets FZ
// only (round-to-nearest).
#if defined(__EMSCRIPTEN__) || defined(R3DS_SOFT_FP)
#define R3DS_SOFTFP 1
static inline bool is_denf(float x) { u32 b; memcpy(&b, &x, 4); return !(b & 0x7F800000u) && (b & 0x7FFFFFu); }
static inline bool is_dend(double x) { u64 b; memcpy(&b, &x, 8); return !(b & 0x7FF0000000000000ull) && (b & 0xFFFFFFFFFFFFFull); }
static inline float fzf(float x, u32 fs) { return ((fs >> 24) & 1) && is_denf(x) ? copysignf(0.0f, x) : x; }
static inline double fzd(double x, u32 fs) { return ((fs >> 24) & 1) && is_dend(x) ? copysign(0.0, x) : x; }
// round the exact value r + e (e = rounding error of r, 0 if r is exact)
// to float according to FPSCR
static inline float rnd_f2(double r, double e, u32 fs) {
    if (r != r) return dn_f((float)r);
    if (((fs >> 24) & 1) && r != 0 && fabs(r) < 1.17549435082228750797e-38) return copysignf(0.0f, (float)r);
    float f = (float)r;   // round to nearest
    u32 mode = (fs >> 22) & 3;
    if (!mode) return f;
    double fd = f;
    bool above = fd > r || (fd == r && e < 0);   // f > exact value
    bool below = fd < r || (fd == r && e > 0);   // f < exact value
    switch (mode) {
    case 1: if (below) f = nextafterf(f, INFINITY); break;
    case 2: if (above) f = nextafterf(f, -INFINITY); break;
    default: if (f > 0 ? above : f < 0 ? below : false) f = nextafterf(f, 0.0f); break;
    }
    return f;
}
static inline float rnd_f(double r, u32 fs) { return rnd_f2(r, 0.0, fs); }
static inline float add_rnd_f(double a, double b, u32 fs) {
    double s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);   // TwoSum: a + b == s + e exactly
    return rnd_f2(s, e, fs);
}
// double: r is already a double; e (exact error, or just its sign) decides
// the directed roundings. fin: the operands were finite (overflow handling).
static inline double rnd_d2(double r, double e, bool fin, u32 fs) {
    if (r != r) return dn_d(r);
    u32 mode = (fs >> 22) & 3;
    if (std::isinf(r) && fin && mode) {   // overflow under a directed rounding
        double mx = copysign(1.7976931348623157e308, r);
        if (mode == 3 || (mode == 1 && r < 0) || (mode == 2 && r > 0)) r = mx;
        return r;
    }
    if (((fs >> 24) & 1) && is_dend(r)) return copysign(0.0, r);
    switch (mode) {
    case 1: if (e > 0) r = nextafter(r, INFINITY); break;
    case 2: if (e < 0) r = nextafter(r, -INFINITY); break;
    case 3: if ((r > 0 && e < 0) || (r < 0 && e > 0)) r = nextafter(r, 0.0); break;
    default: break;
    }
    if (((fs >> 24) & 1) && is_dend(r)) return copysign(0.0, r);
    return r;
}
static inline double two_prod_err(double a, double b, double p) {   // a*b - p exactly (Dekker)
    const double C = 134217729.0;
    double t = C * a, u = C * b;
    if (!std::isfinite(t) || !std::isfinite(u) || !std::isfinite(p)) return 0;
    double ah = t - (t - a), al = a - ah, bh = u - (u - b), bl = b - bh;
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}
static inline double rnd_d(double r, u32 fs) { return rnd_d2(r, 0, true, fs); }
static inline double vmul_d(u32 fs, double a, double b) {
    a = fzd(a, fs); b = fzd(b, fs);
    double p = a * b;
    return rnd_d2(p, two_prod_err(a, b, p), std::isfinite(a) && std::isfinite(b), fs);
}
static inline double vadd_d(u32 fs, double a, double b) {
    a = fzd(a, fs); b = fzd(b, fs);
    double s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);
    return rnd_d2(s, std::isfinite(s) ? e : 0, std::isfinite(a) && std::isfinite(b), fs);
}
static inline double vsub_d(u32 fs, double a, double b) { return vadd_d(fs, a, -b); }
static inline double vdiv_d(u32 fs, double a, double b) {
    a = fzd(a, fs); b = fzd(b, fs);
    double q = a / b, e = 0;
    if (std::isfinite(q) && q != 0) { double p = q * b; double rem = (a - p) - two_prod_err(q, b, p); e = b > 0 ? rem : -rem; }
    return rnd_d2(q, e, std::isfinite(a) && std::isfinite(b) && b != 0, fs);
}
static inline double vsqrt_d2(u32 fs, double a) {
    a = fzd(a, fs);
    if (a < 0 || a != a) return vsqrt_d(a);
    double r = sqrt(a), e = 0;
    if (std::isfinite(r) && r != 0) { double p = r * r; e = (a - p) - two_prod_err(r, r, p); }
    return rnd_d2(r, e, true, fs);
}
static inline float vmul_f(u32 fs, float a, float b) { return rnd_f((double)fzf(a, fs) * fzf(b, fs), fs); }
static inline float vadd_f(u32 fs, float a, float b) { return add_rnd_f(fzf(a, fs), fzf(b, fs), fs); }
static inline float vsub_f(u32 fs, float a, float b) { return add_rnd_f(fzf(a, fs), -(double)fzf(b, fs), fs); }
static inline float vdiv_f(u32 fs, float a, float b) { return rnd_f((double)fzf(a, fs) / fzf(b, fs), fs); }
static inline float vsqrt_f2(u32 fs, float a) { a = fzf(a, fs); return a < 0 || a != a ? dn_f(NAN) : rnd_f(sqrt((double)a), fs); }
static inline float vcvt_f_d(u32 fs, double a) { return rnd_f(fzd(a, fs), fs); }
static inline double vcvt_d_f(u32 fs, float a) { return dn_d((double)fzf(a, fs)); }
static inline float vcvt_f_i(u32 fs, double a) { return rnd_f(a, fs); }
static inline float fz_cmp_f(float x, u32 fs) { return fzf(x, fs); }
static inline double fz_cmp_d(double x, u32 fs) { return fzd(x, fs); }
#else
static inline float vmul_f(u32, float a, float b) { return dn_f(a * b); }
static inline float vadd_f(u32, float a, float b) { return dn_f(a + b); }
static inline float vsub_f(u32, float a, float b) { return dn_f(a - b); }
static inline float vdiv_f(u32, float a, float b) { return dn_f(a / b); }
static inline float vsqrt_f2(u32, float a) { return vsqrt_f(a); }
static inline float vcvt_f_d(u32, double a) { return dn_f((float)a); }
static inline double vcvt_d_f(u32, float a) { return dn_d((double)a); }
static inline float vcvt_f_i(u32, double a) { return (float)a; }
static inline double vmul_d(u32, double a, double b) { return dn_d(a * b); }
static inline double vadd_d(u32, double a, double b) { return dn_d(a + b); }
static inline double vsub_d(u32, double a, double b) { return dn_d(a - b); }
static inline double vdiv_d(u32, double a, double b) { return dn_d(a / b); }
static inline double vsqrt_d2(u32, double a) { return vsqrt_d(a); }
static inline float fz_cmp_f(float x, u32) { return x; }
static inline double fz_cmp_d(double x, u32) { return x; }
#endif

static inline u32 vfp_cmp(u32 fpscr, double a, double b) {
    u32 f;
    if (std::isnan(a) || std::isnan(b)) f = 0x3;
    else if (a == b) f = 0x6;
    else if (a < b) f = 0x8;
    else f = 0x2;
    return (fpscr & 0x0FFFFFFFu) | (f << 28);
}
static inline double vfp_round(double v, u32 fpscr) {
    switch ((fpscr >> 22) & 3) {
    case 0: return nearbyint(v);    // host default is round-to-nearest-even
    case 1: return ceil(v);
    case 2: return floor(v);
    default: return trunc(v);
    }
}
static inline u32 f2s_rz(double v, u32) {
    if (std::isnan(v)) return 0;
    if (v >= 2147483647.0) return 0x7FFFFFFF;
    if (v <= -2147483648.0) return 0x80000000u;
    return (u32)(s32)v;
}
static inline u32 f2u_rz(double v, u32) {
    if (std::isnan(v) || v <= 0) return 0;
    if (v >= 4294967295.0) return 0xFFFFFFFFu;
    return (u32)v;
}
static inline u32 f2s(double v, u32 fpscr) { return f2s_rz(vfp_round(v, fpscr), fpscr); }
static inline u32 f2u(double v, u32 fpscr) { return f2u_rz(vfp_round(v, fpscr), fpscr); }
