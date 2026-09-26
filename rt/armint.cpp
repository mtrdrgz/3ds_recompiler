// Portable fallback interpreter for ARMv6K (ARM + Thumb-1 + VFPv2).
//
// It runs guest code the static lifter did not cover (and the whole game in
// R3DS_INTERP=1 mode), then hands control back to lifted code at the first pc
// that has a lifted entry. It replaces Unicorn as the runtime fallback so
// the recompilation has no native-code-generating dependency and can be
// built for WebAssembly. Semantics mirror lift/armdec.py instruction by
// instruction and are checked against Unicorn by rt_test/difftest_main.cpp
// (DT_INTERP=1).
#include "cpu.h"
#include "mem.h"
#include "lift_rt.h"
#include <atomic>
#include <mutex>
#include <set>

extern int g_interp_only;
extern std::atomic<u64> g_stat_interp;
#ifndef R3DS_USE_UNICORN
FILE *g_miss_log;
static std::set<u32> g_miss_seen;
static std::mutex g_miss_mtx;
#endif



namespace {

struct St {
    Cpu &c;
    u32 pc;       // address of the current instruction
    u32 npc;      // next pc (bit0 = thumb)
    bool svc = false;
    u32 svc_imm = 0;
};

static inline bool cond_ok(const Cpu &c, u32 cond) {
    switch (cond) {
    case 0: return c.z;
    case 1: return !c.z;
    case 2: return c.c;
    case 3: return !c.c;
    case 4: return c.n;
    case 5: return !c.n;
    case 6: return c.v;
    case 7: return !c.v;
    case 8: return c.c && !c.z;
    case 9: return !c.c || c.z;
    case 10: return c.n == c.v;
    case 11: return c.n != c.v;
    case 12: return !c.z && c.n == c.v;
    case 13: return c.z || c.n != c.v;
    default: return true;
    }
}

static inline s32 sxb(u32 v, int bits) { u32 m = 1u << (bits - 1); return (s32)((v & (m - 1)) - (v & m)); }

[[noreturn]] static void undef(St &s, u32 w, const char *what) {
    LOG("[armint] unsupported %s instruction %08x at %08x (%s)", s.c.thumb ? "thumb" : "arm", w, s.pc, what);
    rt_trap(s.c, s.pc, "undefined instruction in interpreter");
}

// ------------------------------------------------------------------ ARM
struct Arm {
    St &s;
    Cpu &c;
    u32 w, pc8;   // pc8 = address + 8
    Arm(St &st, u32 word) : s(st), c(st.c), w(word), pc8(st.pc + 8) {}

    u32 rr(u32 n) const { return n == 15 ? pc8 : c.r[n]; }
    u32 rr12(u32 n) const { return n == 15 ? pc8 + 4 : c.r[n]; }
    void wr(u32 n, u32 v, bool interwork = false) {
        if (n == 15) s.npc = interwork ? v : (v & ~3u);
        else c.r[n] = v;
    }
    void nz(u32 r) { c.n = r >> 31; c.z = r == 0; }

    u32 imm_shift(u32 v, u32 typ, u32 amt, u32 &carry) const {
        switch (typ) {
        case 0:
            if (amt == 0) return v;
            carry = (v >> (32 - amt)) & 1; return v << amt;
        case 1:
            if (amt == 0) amt = 32;
            if (amt == 32) { carry = v >> 31; return 0; }
            carry = (v >> (amt - 1)) & 1; return v >> amt;
        case 2:
            if (amt == 0) amt = 32;
            if (amt == 32) { carry = v >> 31; return (u32)((s32)v >> 31); }
            carry = (v >> (amt - 1)) & 1; return (u32)((s32)v >> amt);
        default:
            if (amt == 0) { carry = v & 1; return (c.c << 31) | (v >> 1); }
            carry = (v >> (amt - 1)) & 1; return ror32(v, amt);
        }
    }

    void exec() {
        u32 cond = w >> 28;
        if (cond == 15) { uncond(); return; }
        if (!cond_ok(c, cond)) return;
        u32 op = (w >> 25) & 7;
        switch (op) {
        case 0:
            if ((w & 0x90) == 0x90) {
                if (((w >> 5) & 3) == 0) mul_swp(); else extra_ls();
            } else if (((w >> 23) & 3) == 2 && !((w >> 20) & 1)) misc();
            else dp();
            return;
        case 1:
            if (((w >> 23) & 3) == 2 && !((w >> 20) & 1)) {
                if ((w >> 21) & 1) {
                    u32 mask = (w >> 16) & 0xF;
                    if (mask) msr(mask, ror32(w & 0xFF, ((w >> 8) & 0xF) * 2));
                } else undef(s, w, "movw/movt");
            } else dp();
            return;
        case 2: ls(); return;
        case 3: if ((w >> 4) & 1) media(); else ls(); return;
        case 4: ldm(); return;
        case 5: {
            u32 t = pc8 + ((u32)sxb(w & 0xFFFFFF, 24) << 2);
            if ((w >> 24) & 1) c.r[14] = s.pc + 4;
            s.npc = t;
            return;
        }
        case 6: ldc(); return;
        default:
            if ((w >> 24) & 1) { s.svc = true; s.svc_imm = w & 0xFFFFFF; return; }
            cp();
            return;
        }
    }

    void msr(u32 mask, u32 v) {
        if (mask & 8) { c.n = v >> 31; c.z = (v >> 30) & 1; c.c = (v >> 29) & 1; c.v = (v >> 28) & 1; c.q = (v >> 27) & 1; }
        if (mask & 4) c.ge = (v >> 16) & 0xF;
    }

    void uncond() {
        if ((w & 0x0E000000) == 0x0A000000) {   // BLX imm
            u32 off = ((u32)sxb(w & 0xFFFFFF, 24) << 2) | (((w >> 24) & 1) << 1);
            c.r[14] = s.pc + 4;
            s.npc = (pc8 + off) | 1;
            return;
        }
        if ((w & 0x0D70F000) == 0x0550F000) return;   // PLD
        if (w == 0xF57FF01F) { c.excl_valid = 0; return; }
        if ((w & 0xFFFFFFF0) == 0xF57FF040 || (w & 0xFFFFFFF0) == 0xF57FF050 || (w & 0xFFFFFFF0) == 0xF57FF060) return;
        if ((w & 0xFFF1FE20) == 0xF1000000) return;   // CPS
        undef(s, w, "uncond");
    }

    void dp() {
        u32 opc = (w >> 21) & 0xF, sf = (w >> 20) & 1, rn = (w >> 16) & 0xF, rd = (w >> 12) & 0xF;
        bool regsh = !((w >> 25) & 1) && ((w >> 4) & 1);
        u32 sc = c.c, op2;
        if ((w >> 25) & 1) {
            u32 rot = ((w >> 8) & 0xF) * 2;
            op2 = ror32(w & 0xFF, rot);
            if (rot) sc = op2 >> 31;
        } else if (regsh) {
            u32 v = rr12(w & 0xF), a = rr12((w >> 8) & 0xF) & 0xFF;
            switch ((w >> 5) & 3) {
            case 0: op2 = lsl_rc(v, a, sc); break;
            case 1: op2 = lsr_rc(v, a, sc); break;
            case 2: op2 = asr_rc(v, a, sc); break;
            default: op2 = ror_rc(v, a, sc); break;
            }
        } else {
            op2 = imm_shift(rr(w & 0xF), (w >> 5) & 3, (w >> 7) & 0x1F, sc);
        }
        u32 a = regsh ? rr12(rn) : rr(rn);
        if (sf && rd == 15 && !(opc >= 8 && opc <= 11)) undef(s, w, "exception return");
        u32 res;
        switch (opc) {
        case 0: case 8: res = a & op2; break;
        case 1: case 9: res = a ^ op2; break;
        case 12: res = a | op2; break;
        case 13: res = op2; break;
        case 14: res = a & ~op2; break;
        case 15: res = ~op2; break;
        default: {
            u32 x, y, cin;
            switch (opc) {
            case 2: case 10: x = a; y = ~op2; cin = 1; break;
            case 3: x = op2; y = ~a; cin = 1; break;
            case 4: case 11: x = a; y = op2; cin = 0; break;
            case 5: x = a; y = op2; cin = c.c; break;
            case 6: x = a; y = ~op2; cin = c.c; break;
            default: x = op2; y = ~a; cin = c.c; break;
            }
            u32 n, z, cc, v;
            res = adds(x, y, cin, n, z, cc, v);
            if (sf || opc == 10 || opc == 11) { c.n = n; c.z = z; c.c = cc; c.v = v; }
            if (opc != 10 && opc != 11) wr(rd, res);
            return;
        }
        }
        if (sf || opc == 8 || opc == 9) { nz(res); c.c = sc; }
        if (opc != 8 && opc != 9) wr(rd, res);
    }

    void misc() {
        u32 op = (w >> 21) & 3, op2 = (w >> 4) & 0xF, rd = (w >> 12) & 0xF, rm = w & 0xF;
        if (op2 == 0) {
            if ((w >> 22) & 1) undef(s, w, "spsr");
            if (op & 1) msr((w >> 16) & 0xF, c.r[rm]);
            else c.r[rd] = (c.n << 31) | (c.z << 30) | (c.c << 29) | (c.v << 28) | (c.q << 27) | (c.ge << 16) | 0x10u;
            return;
        }
        if ((op2 == 1 || op2 == 2) && op == 1) { s.npc = rr(rm); return; }   // BX / BXJ
        if (op2 == 1 && op == 3) { c.r[rd] = clz32(c.r[rm]); return; }
        if (op2 == 3 && op == 1) { u32 t = rr(rm); c.r[14] = s.pc + 4; s.npc = t; return; }
        if (op2 == 5) {
            u32 rn = (w >> 16) & 0xF, m = c.r[rm], n = c.r[rn];
            c.r[rd] = op == 0 ? qadd(c, m, n) : op == 1 ? qsub(c, m, n) : op == 2 ? qdadd(c, m, n) : qdsub(c, m, n);
            return;
        }
        if (op2 == 7 && op == 1) rt_trap(c, s.pc, "bkpt");
        if ((op2 & 9) == 8) {
            u32 x = (w >> 5) & 1, y = (w >> 6) & 1;
            u32 rdh = (w >> 16) & 0xF, rn = (w >> 12) & 0xF, rs = (w >> 8) & 0xF;
            s32 hm = (s16)(x ? c.r[rm] >> 16 : c.r[rm]);
            s32 hs = (s16)(y ? c.r[rs] >> 16 : c.r[rs]);
            if (op == 0) c.r[rdh] = smla_q(c, (u32)(hm * hs), c.r[rn]);
            else if (op == 1) {
                s64 p = ((s64)(s32)c.r[rm] * (s64)hs) >> 16;
                if (x) c.r[rdh] = (u32)p;
                else c.r[rdh] = smla_q(c, (u32)p, c.r[rn]);
            } else if (op == 2) {
                s64 acc = (s64)(((u64)c.r[rdh] << 32) | c.r[rn]);
                acc += (s64)(hm * hs);
                c.r[rn] = (u32)acc; c.r[rdh] = (u32)((u64)acc >> 32);
            } else c.r[rdh] = (u32)(hm * hs);
            return;
        }
        undef(s, w, "misc");
    }

    void mul_swp() {
        if ((w >> 24) & 1) {
            u32 rn = (w >> 16) & 0xF, rt = (w >> 12) & 0xF, rt2 = w & 0xF;
            if ((w >> 23) & 1) {
                u32 kind = (w >> 21) & 3, a = c.r[rn];
                if ((w >> 20) & 1) {
                    c.excl_addr = a; c.excl_valid = 1; c.excl_epoch = excl_epoch_now();
                    switch (kind) {
                    case 0: c.r[rt] = rd32(a); break;
                    case 1: c.r[rt] = rd32(a); c.r[rt + 1] = rd32(a + 4); break;
                    case 2: c.r[rt] = rd8(a); break;
                    default: c.r[rt] = rd16(a); break;
                    }
                } else {
                    if (c.excl_valid && c.excl_addr == a && c.excl_epoch == excl_epoch_now()) {
                        switch (kind) {
                        case 0: wr32(a, c.r[rt2]); break;
                        case 1: wr32(a, c.r[rt2]); wr32(a + 4, c.r[rt2 + 1]); break;
                        case 2: wr8(a, (u8)c.r[rt2]); break;
                        default: wr16(a, (u16)c.r[rt2]); break;
                        }
                        c.r[rt] = 0;
                    } else c.r[rt] = 1;
                    c.excl_valid = 0;
                }
                return;
            }
            u32 a = c.r[rn];
            if ((w >> 22) & 1) { u32 t = rd8(a); wr8(a, (u8)c.r[rt2]); c.r[rt] = t; }
            else { u32 t = rd32(a); wr32(a, c.r[rt2]); c.r[rt] = t; }
            return;
        }
        u32 op = (w >> 21) & 7, sf = (w >> 20) & 1;
        u32 rd = (w >> 16) & 0xF, rn = (w >> 12) & 0xF, rs = (w >> 8) & 0xF, rm = w & 0xF;
        switch (op) {
        case 0: c.r[rd] = c.r[rm] * c.r[rs]; if (sf) nz(c.r[rd]); return;
        case 1: c.r[rd] = c.r[rm] * c.r[rs] + c.r[rn]; if (sf) nz(c.r[rd]); return;
        case 2: { u64 r = (u64)c.r[rm] * c.r[rs] + c.r[rn] + c.r[rd]; c.r[rn] = (u32)r; c.r[rd] = (u32)(r >> 32); return; }
        case 3: c.r[rd] = c.r[rn] - c.r[rm] * c.r[rs]; return;
        default: {
            bool sg = op >= 6;
            u64 r = sg ? (u64)((s64)(s32)c.r[rm] * (s64)(s32)c.r[rs]) : (u64)c.r[rm] * c.r[rs];
            if (op & 1) r += ((u64)c.r[rd] << 32) | c.r[rn];
            c.r[rn] = (u32)r; c.r[rd] = (u32)(r >> 32);
            if (sf) { c.n = (u32)(r >> 63); c.z = r == 0; }
            return;
        }
        }
    }

    void extra_ls() {
        u32 p = (w >> 24) & 1, u = (w >> 23) & 1, i = (w >> 22) & 1, wb = (w >> 21) & 1, l = (w >> 20) & 1;
        u32 rn = (w >> 16) & 0xF, rt = (w >> 12) & 0xF, sh = (w >> 5) & 3;
        u32 off = i ? (((w >> 4) & 0xF0) | (w & 0xF)) : c.r[w & 0xF];
        u32 base = rr(rn);
        u32 a = p ? (u ? base + off : base - off) : base;
        u32 nb = p ? a : (u ? base + off : base - off);
        bool dowb = !p || wb;
        if (l) {
            if (rt == 15) undef(s, w, "ldrh pc");
            u32 v;
            if (sh == 1) v = rd16(a);
            else if (sh == 2) v = (u32)(s32)(s8)rd8(a);
            else v = (u32)(s32)(s16)rd16(a);
            if (dowb) c.r[rn] = nb;
            c.r[rt] = v;
        } else if (sh == 1) {
            wr16(a, (u16)rr(rt));
            if (dowb) c.r[rn] = nb;
        } else if (sh == 2) {   // LDRD
            if ((rt & 1) || rt == 14) undef(s, w, "ldrd odd");
            u32 v0 = rd32(a), v1 = rd32(a + 4);
            if (dowb) c.r[rn] = nb;
            c.r[rt] = v0; c.r[rt + 1] = v1;
        } else {                // STRD
            if ((rt & 1) || rt == 14) undef(s, w, "strd odd");
            wr32(a, c.r[rt]); wr32(a + 4, c.r[rt + 1]);
            if (dowb) c.r[rn] = nb;
        }
    }

    void ls() {
        u32 p = (w >> 24) & 1, u = (w >> 23) & 1, b = (w >> 22) & 1, wb = (w >> 21) & 1, l = (w >> 20) & 1;
        u32 rn = (w >> 16) & 0xF, rt = (w >> 12) & 0xF;
        u32 off;
        if ((w >> 25) & 1) { u32 dummy = 0; off = imm_shift(rr(w & 0xF), (w >> 5) & 3, (w >> 7) & 0x1F, dummy); }
        else off = w & 0xFFF;
        u32 base = rr(rn);
        u32 a = p ? (u ? base + off : base - off) : base;
        u32 nb = p ? a : (u ? base + off : base - off);
        bool dowb = wb || !p;
        if (l) {
            u32 v = b ? rd8(a) : rd32(a);
            if (rt == 15) {
                if (b) undef(s, w, "ldrb pc");
                if (dowb) c.r[rn] = nb;
                s.npc = v;
                return;
            }
            if (dowb) c.r[rn] = nb;
            c.r[rt] = v;
        } else {
            u32 v = rr(rt);
            if (b) wr8(a, (u8)v); else wr32(a, v);
            if (dowb) c.r[rn] = nb;
        }
    }

    void ldm() {
        u32 p = (w >> 24) & 1, u = (w >> 23) & 1, sb = (w >> 22) & 1, wb = (w >> 21) & 1, l = (w >> 20) & 1;
        u32 rn = (w >> 16) & 0xF, list = w & 0xFFFF;
        u32 n = __builtin_popcount(list);
        if (!n) undef(s, w, "ldm empty");
        if (sb) undef(s, w, "ldm user/exception");
        u32 base = c.r[rn];
        u32 a = u ? (p ? base + 4 : base) : (p ? base - 4 * n : base - 4 * n + 4);
        u32 nbase = u ? base + 4 * n : base - 4 * n;
        if (l) {
            u32 k = 0, tpc = 0, nbv = 0; bool haspc = false, hasrn = false;
            for (u32 r = 0; r < 16; r++) {
                if (!((list >> r) & 1)) continue;
                u32 v = rd32(a + 4 * k++);
                if (r == 15) { tpc = v; haspc = true; }
                else if (r == rn) { nbv = v; hasrn = true; }
                else c.r[r] = v;
            }
            if (wb && !hasrn) c.r[rn] = nbase;
            if (hasrn) c.r[rn] = nbv;
            if (haspc) s.npc = tpc;
        } else {
            u32 k = 0;
            for (u32 r = 0; r < 16; r++)
                if ((list >> r) & 1) wr32(a + 4 * k++, rr(r));
            if (wb) c.r[rn] = nbase;
        }
    }

    void media() {
        u32 op1 = (w >> 20) & 0x1F, op2 = (w >> 5) & 7;
        u32 rd = (w >> 12) & 0xF, rn = (w >> 16) & 0xF, rm = w & 0xF, rs = (w >> 8) & 0xF;
        u32 N = c.r[rn], Mv = c.r[rm];
        if ((op1 >> 3) == 0) {
            u32 kind = op1 & 7;
            static const int modes[8] = {-1, 0, 1, 2, -1, 3, 4, 5};
            int mode = modes[kind];
            ParOp po;
            switch (op2) {
            case 0: po = P_ADD16; break; case 1: po = P_ASX; break; case 2: po = P_SAX; break;
            case 3: po = P_SUB16; break; case 4: po = P_ADD8; break; case 7: po = P_SUB8; break;
            default: mode = -1; po = P_ADD16; break;
            }
            u32 r;
            switch (mode) {
            case 0: r = par<0>(c, N, Mv, po); break;
            case 1: r = par<1>(c, N, Mv, po); break;
            case 2: r = par<2>(c, N, Mv, po); break;
            case 3: r = par<3>(c, N, Mv, po); break;
            case 4: r = par<4>(c, N, Mv, po); break;
            case 5: r = par<5>(c, N, Mv, po); break;
            default: undef(s, w, "parallel");
            }
            c.r[rd] = r;
            return;
        }
        if ((op1 >> 3) == 1) {
            u32 o = op1 & 7;
            u32 rot = ((w >> 10) & 3) * 8;
            u32 rme = ror32(Mv, rot);
            u32 dummy = 0;
            if (o == 0 && !(op2 & 1)) {   // PKH
                u32 sh = (w >> 7) & 0x1F;
                if ((w >> 6) & 1) c.r[rd] = (N & 0xFFFF0000u) | (imm_shift(Mv, 2, sh, dummy) & 0xFFFFu);
                else c.r[rd] = (N & 0xFFFFu) | ((Mv << sh) & 0xFFFF0000u);
                return;
            }
            if (o == 0 && op2 == 3) {   // SXTAB16 / SXTB16
                u32 b = rn == 15 ? 0 : N;
                u32 lo = (u32)((s32)(s8)(rme & 0xFF) + (s32)(s16)(b & 0xFFFF)) & 0xFFFF;
                u32 hi = (u32)((s32)(s8)((rme >> 16) & 0xFF) + (s32)(s16)(b >> 16)) & 0xFFFF;
                c.r[rd] = lo | (hi << 16);
                return;
            }
            if (o == 0 && op2 == 5) { c.r[rd] = sel_ge(c, N, Mv); return; }
            if ((o >> 1) == 1 && !(op2 & 1)) {
                u32 v = imm_shift(Mv, (w >> 6) & 1 ? 2 : 0, (w >> 7) & 0x1F, dummy);
                c.r[rd] = ssat(c, (s32)v, ((w >> 16) & 0x1F) + 1); return;
            }
            if (o == 2 && op2 == 1) { c.r[rd] = ssat16(c, Mv, ((w >> 16) & 0xF) + 1); return; }
            if ((o >> 1) == 3 && !(op2 & 1)) {
                u32 v = imm_shift(Mv, (w >> 6) & 1 ? 2 : 0, (w >> 7) & 0x1F, dummy);
                c.r[rd] = usat(c, (s32)v, (w >> 16) & 0x1F); return;
            }
            if (o == 6 && op2 == 1) { c.r[rd] = usat16(c, Mv, (w >> 16) & 0xF); return; }
            if (op2 == 3) {
                if (o == 4) {   // UXTAB16 / UXTB16
                    u32 b = rn == 15 ? 0 : N;
                    u32 lo = ((rme & 0xFF) + (b & 0xFFFF)) & 0xFFFF;
                    u32 hi = (((rme >> 16) & 0xFF) + (b >> 16)) & 0xFFFF;
                    c.r[rd] = lo | (hi << 16);
                    return;
                }
                u32 v;
                bool ok = true;
                switch (o) {
                case 2: v = (u32)(s32)(s8)rme; break;
                case 3: v = (u32)(s32)(s16)rme; break;
                case 6: v = (u32)(u8)rme; break;
                case 7: v = (u32)(u16)rme; break;
                default: ok = false; v = 0; break;
                }
                if (ok) { c.r[rd] = rn == 15 ? v : N + v; return; }
            }
            if (o == 3 && op2 == 1) { c.r[rd] = __builtin_bswap32(Mv); return; }
            if (o == 3 && op2 == 5) { c.r[rd] = rev16(Mv); return; }
            if (o == 7 && op2 == 5) { c.r[rd] = (u32)(s32)(s16)__builtin_bswap16((u16)Mv); return; }
            undef(s, w, "media pack");
        }
        if ((op1 >> 3) == 2) {
            u32 o = op1 & 7;
            u32 rdd = (w >> 16) & 0xF, ra = (w >> 12) & 0xF;
            u32 rsv = (op2 & 1) ? ror32(c.r[rs], 16) : c.r[rs];
            if (o == 0) {
                if ((op2 >> 2) & 1) undef(s, w, "smlad op2");
                s32 lo = (s32)(s16)Mv * (s32)(s16)rsv;
                s32 hi = (s32)(s16)(Mv >> 16) * (s32)(s16)(rsv >> 16);
                s64 acc = ra == 15 ? 0 : (s64)(s32)c.r[ra];
                s64 r = (op2 >> 1) & 1 ? (s64)lo - (s64)hi + acc : (s64)lo + (s64)hi + acc;
                c.r[rdd] = (u32)r;
                if (r != (s64)(s32)r) c.q = 1;
                return;
            }
            if (o == 4) {
                s64 lo = (s64)(s16)Mv * (s16)rsv;
                s64 hi = (s64)(s16)(Mv >> 16) * (s16)(rsv >> 16);
                s64 acc = (s64)(((u64)c.r[rdd] << 32) | c.r[ra]);
                acc += (op2 >> 1) & 1 ? lo - hi : lo + hi;
                c.r[ra] = (u32)acc; c.r[rdd] = (u32)((u64)acc >> 32);
                return;
            }
            if (o == 5) {
                s64 rnd = (op2 & 1) ? 0x80000000ll : 0;
                s64 prod = (s64)(s32)Mv * (s32)c.r[rs];
                if ((op2 >> 1) == 0) {
                    s64 acc = ra == 15 ? 0 : (s64)((u64)c.r[ra] << 32);
                    c.r[rdd] = (u32)((u64)(acc + prod + rnd) >> 32);
                } else if ((op2 >> 1) == 3) {
                    c.r[rdd] = (u32)((u64)((s64)((u64)c.r[ra] << 32) - prod + rnd) >> 32);
                } else undef(s, w, "smm");
                return;
            }
            undef(s, w, "smul");
        }
        if (op1 == 0x18 && op2 == 0) {
            u32 rdd = (w >> 16) & 0xF, ra = (w >> 12) & 0xF;
            c.r[rdd] = usad8(Mv, c.r[rs]) + (ra == 15 ? 0 : c.r[ra]);
            return;
        }
        undef(s, w, "media");
    }

    static u32 sreg(u32 v, u32 bit) { return (v << 1) | bit; }
    static u32 dreg(u32 v, u32 bit) { return (bit << 4) | v; }

    void ldc() {
        u32 cpn = (w >> 8) & 0xF;
        if (cpn != 10 && cpn != 11) undef(s, w, "ldc");
        if (((w >> 21) & 0x7F) == 0x62) {   // VMOV two core <-> D / two S
            u32 l = (w >> 20) & 1, rt = (w >> 12) & 0xF, rt2 = (w >> 16) & 0xF, m = (w >> 5) & 1, vm = w & 0xF;
            u32 a, b;
            if (cpn == 11) { a = dreg(vm, m) * 2; b = a + 1; }
            else { a = sreg(vm, m); b = (a + 1) & 31; }
            if (l) { c.r[rt] = c.f.sw[a]; c.r[rt2] = c.f.sw[b]; }
            else { c.f.sw[a] = c.r[rt]; c.f.sw[b] = c.r[rt2]; }
            return;
        }
        u32 p = (w >> 24) & 1, u = (w >> 23) & 1, d = (w >> 22) & 1, wb = (w >> 21) & 1, l = (w >> 20) & 1;
        u32 rn = (w >> 16) & 0xF, vd = (w >> 12) & 0xF, imm8 = w & 0xFF;
        bool dbl = cpn == 11;
        if (p && !wb) {   // VLDR / VSTR
            u32 base = rn == 15 ? (pc8 & ~3u) : c.r[rn];
            u32 a = u ? base + imm8 * 4 : base - imm8 * 4;
            if (dbl) { u32 r = dreg(vd, d); if (l) c.f.dw[r] = rd64(a); else wr64(a, c.f.dw[r]); }
            else { u32 r = sreg(vd, d); if (l) c.f.sw[r] = rd32(a); else wr32(a, c.f.sw[r]); }
            return;
        }
        u32 n = dbl ? imm8 / 2 : imm8;
        u32 first = dbl ? dreg(vd, d) : sreg(vd, d);
        if (!n) undef(s, w, "vldm 0");
        u32 size = imm8 * 4;
        u32 a;
        if (p == 0 && u == 1) a = c.r[rn];
        else if (p == 1 && u == 0) a = c.r[rn] - size;
        else undef(s, w, "vldm mode");
        for (u32 k = 0; k < n; k++) {
            if (dbl) {
                if (l) c.f.dw[(first + k) & 15] = rd64(a + 8 * k); else wr64(a + 8 * k, c.f.dw[(first + k) & 15]);
            } else {
                if (l) c.f.sw[(first + k) & 31] = rd32(a + 4 * k); else wr32(a + 4 * k, c.f.sw[(first + k) & 31]);
            }
        }
        if (wb) c.r[rn] = u ? c.r[rn] + size : c.r[rn] - size;
    }

    void cp() {
        u32 cpn = (w >> 8) & 0xF;
        if ((w >> 4) & 1) {
            u32 l = (w >> 20) & 1, rt = (w >> 12) & 0xF, opc1 = (w >> 21) & 7;
            u32 crn = (w >> 16) & 0xF, crm = w & 0xF, opc2 = (w >> 5) & 7;
            if (cpn == 15) {
                if (l && crn == 13 && crm == 0 && opc1 == 0 && opc2 == 3) { c.r[rt] = c.tpidruro; return; }
                if (l && crn == 13 && crm == 0 && opc1 == 0 && opc2 == 2) { c.r[rt] = c.tpidrurw; return; }
                if (!l && crn == 13 && crm == 0 && opc2 == 2) { c.tpidrurw = c.r[rt]; return; }
                if (!l && crn == 7) return;
                undef(s, w, "cp15");
            }
            if (cpn == 10) {
                if (opc1 == 0) {
                    u32 sn = sreg(crn, (w >> 7) & 1);
                    if (l) c.r[rt] = c.f.sw[sn]; else c.f.sw[sn] = c.r[rt];
                    return;
                }
                if (opc1 == 7) {
                    if (l) {
                        if (crn == 1) {
                            if (rt == 15) { c.n = c.fpscr >> 31; c.z = (c.fpscr >> 30) & 1; c.c = (c.fpscr >> 29) & 1; c.v = (c.fpscr >> 28) & 1; }
                            else c.r[rt] = c.fpscr;
                        } else if (crn == 8) c.r[rt] = 0x40000000u;
                        else if (crn == 0) c.r[rt] = 0x410120B4u;
                        else undef(s, w, "vmrs");
                    } else if (crn == 1) { c.fpscr = c.r[rt]; host_fpscr(c.fpscr); }
                    return;
                }
            }
            if (cpn == 11 && (opc1 & 6) == 0) {
                u32 dd = dreg(crn, (w >> 7) & 1), x = opc1 & 1;
                if (l) c.r[rt] = c.f.sw[dd * 2 + x]; else c.f.sw[dd * 2 + x] = c.r[rt];
                return;
            }
            undef(s, w, "cp transfer");
        }
        if (cpn != 10 && cpn != 11) undef(s, w, "cdp");
        if (cpn == 11) vfp<double>(); else vfp<float>();
    }

    template <class T> T &fr(u32 v, u32 bit);
    template <class T> void vfp() {
        constexpr bool dbl = sizeof(T) == 8;
        u32 p = (w >> 23) & 1, q = (w >> 21) & 1, r = (w >> 20) & 1, sb = (w >> 6) & 1;
        u32 D = (w >> 22) & 1, N = (w >> 7) & 1, Mb = (w >> 5) & 1;
        u32 vn = (w >> 16) & 0xF, vd = (w >> 12) & 0xF, vm = w & 0xF;
        T &d = fr<T>(vd, D);
        T n = fr<T>(vn, N), m = fr<T>(vm, Mb);
        u32 opc = (p << 3) | (q << 2) | (r << 1) | sb;
        u32 F = c.fpscr;
        auto mul = [&](T x, T y) -> T { if constexpr (dbl) return vmul_d(F, x, y); else return vmul_f(F, x, y); };
        auto add = [&](T x, T y) -> T { if constexpr (dbl) return vadd_d(F, x, y); else return vadd_f(F, x, y); };
        auto sub = [&](T x, T y) -> T { if constexpr (dbl) return vsub_d(F, x, y); else return vsub_f(F, x, y); };
        auto fzc = [&](T x) -> T { if constexpr (dbl) return fz_cmp_d(x, F); else return fz_cmp_f(x, F); };
        switch (opc) {
        case 0: d = add(d, mul(n, m)); return;
        case 1: d = sub(d, mul(n, m)); return;
        case 2: d = sub(mul(n, m), d); return;
        case 3: d = sub(-d, mul(n, m)); return;
        case 4: d = mul(n, m); return;
        case 5: d = -mul(n, m); return;
        case 6: d = add(n, m); return;
        case 7: d = sub(n, m); return;
        case 8: if constexpr (dbl) d = vdiv_d(F, n, m); else d = vdiv_f(F, n, m); return;
        default: break;
        }
        if ((opc & 0xE) != 0xE) undef(s, w, "vfp dp");
        u32 o3 = (w >> 6) & 3;
        if (vn == 0 && o3 == 1) { d = m; return; }
        if (vn == 0 && o3 == 3) { if constexpr (dbl) d = fabs(m); else d = fabsf(m); return; }
        if (vn == 1 && o3 == 1) { d = -m; return; }
        if (vn == 1 && o3 == 3) { if constexpr (dbl) d = vsqrt_d2(F, m); else d = vsqrt_f2(F, m); return; }
        if (vn == 4 || vn == 5) {
            T rhs = vn == 5 ? (T)0 : fzc(m);
            c.fpscr = vfp_cmp(c.fpscr, (double)fzc(d), (double)rhs);
            return;
        }
        if (vn == 7 && o3 == 3) {
            if constexpr (dbl) c.f.s[sreg(vd, D)] = vcvt_f_d(F, m);
            else c.f.d[dreg(vd, D)] = vcvt_d_f(F, c.f.s[sreg(vm, Mb)]);
            return;
        }
        if (vn == 8) {
            u32 src = c.f.sw[sreg(vm, Mb)];
            double v = (w >> 7) & 1 ? (double)(s32)src : (double)(u32)src;
            if constexpr (dbl) d = v; else d = vcvt_f_i(F, v);
            return;
        }
        if (vn == 12 || vn == 13) {
            u32 &dst = c.f.sw[sreg(vd, D)];
            bool rz = (w >> 7) & 1;
            double x = (double)m;
            if (vn == 13) dst = rz ? f2s_rz(x, c.fpscr) : f2s(x, c.fpscr);
            else dst = rz ? f2u_rz(x, c.fpscr) : f2u(x, c.fpscr);
            return;
        }
        undef(s, w, "vfp ext");
    }
};
template <> float &Arm::fr<float>(u32 v, u32 bit) { return c.f.s[sreg(v, bit)]; }
template <> double &Arm::fr<double>(u32 v, u32 bit) { return c.f.d[dreg(v, bit)]; }

// ---------------------------------------------------------------- Thumb
static void thumb(St &s, u32 h) {
    Cpu &c = s.c;
    u32 pc4 = s.pc + 4;
    auto lo = [h](int sh) { return (h >> sh) & 7; };
    auto nz = [&c](u32 r) { c.n = r >> 31; c.z = r == 0; };
    auto addf = [&c](u32 a, u32 b, u32 cin) { return adds(a, b, cin, c.n, c.z, c.c, c.v); };
    u32 top5 = h >> 11;
    if (top5 < 3) {
        u32 imm = (h >> 6) & 0x1F, rm = lo(3), rd = lo(0), v = c.r[rm];
        if (top5 == 0 && imm == 0) { c.r[rd] = v; nz(v); return; }
        u32 cc = c.c, r;
        u32 amt = imm;
        switch (top5) {
        case 0: cc = (v >> (32 - amt)) & 1; r = v << amt; break;
        case 1: if (!amt) amt = 32; cc = (v >> (amt - 1)) & 1; r = amt == 32 ? 0 : v >> amt; break;
        default: if (!amt) amt = 32; cc = amt == 32 ? v >> 31 : (v >> (amt - 1)) & 1; r = (u32)((s32)v >> (amt == 32 ? 31 : amt)); break;
        }
        c.r[rd] = r; nz(r); c.c = cc;
        return;
    }
    if (top5 == 3) {
        u32 b = (h >> 10) & 1 ? (h >> 6) & 7 : c.r[(h >> 6) & 7];
        c.r[lo(0)] = (h >> 9) & 1 ? addf(c.r[lo(3)], ~b, 1) : addf(c.r[lo(3)], b, 0);
        return;
    }
    if (top5 < 8) {
        u32 rd = lo(8), imm = h & 0xFF;
        switch (top5 - 4) {
        case 0: c.r[rd] = imm; nz(imm); break;
        case 1: addf(c.r[rd], ~imm, 1); break;
        case 2: c.r[rd] = addf(c.r[rd], imm, 0); break;
        default: c.r[rd] = addf(c.r[rd], ~imm, 1); break;
        }
        return;
    }
    if ((h >> 10) == 0x10) {
        u32 op = (h >> 6) & 0xF, rd = lo(0);
        u32 &D = c.r[rd]; u32 Mv = c.r[lo(3)];
        switch (op) {
        case 0: D &= Mv; nz(D); break;
        case 1: D ^= Mv; nz(D); break;
        case 2: { u32 sc = c.c; D = lsl_rc(D, Mv & 0xFF, sc); c.c = sc; nz(D); break; }
        case 3: { u32 sc = c.c; D = lsr_rc(D, Mv & 0xFF, sc); c.c = sc; nz(D); break; }
        case 4: { u32 sc = c.c; D = asr_rc(D, Mv & 0xFF, sc); c.c = sc; nz(D); break; }
        case 5: D = addf(D, Mv, c.c); break;
        case 6: D = addf(D, ~Mv, c.c); break;
        case 7: { u32 sc = c.c; D = ror_rc(D, Mv & 0xFF, sc); c.c = sc; nz(D); break; }
        case 8: nz(D & Mv); break;
        case 9: D = addf(0, ~Mv, 1); break;
        case 10: addf(D, ~Mv, 1); break;
        case 11: addf(D, Mv, 0); break;
        case 12: D |= Mv; nz(D); break;
        case 13: D = Mv * D; nz(D); break;
        case 14: D &= ~Mv; nz(D); break;
        default: D = ~Mv; nz(D); break;
        }
        return;
    }
    if ((h >> 10) == 0x11) {
        u32 op = (h >> 8) & 3, rd = ((h >> 4) & 8) | (h & 7), rm = (h >> 3) & 0xF;
        u32 mv = rm == 15 ? pc4 : c.r[rm];
        switch (op) {
        case 0:
            if (rd == 15) s.npc = (pc4 + mv) | 1;
            else c.r[rd] = c.r[rd] + mv;
            return;
        case 1: addf(rd == 15 ? pc4 : c.r[rd], ~mv, 1); return;
        case 2:
            if (rd == 15) s.npc = mv | 1;
            else c.r[rd] = mv;
            return;
        default:
            if ((h >> 7) & 1) c.r[14] = (s.pc + 2) | 1;
            s.npc = mv;
            return;
        }
    }
    if (top5 == 9) { c.r[lo(8)] = rd32((pc4 & ~3u) + (h & 0xFF) * 4); return; }
    if ((h >> 12) == 5) {
        u32 a = c.r[lo(3)] + c.r[lo(6)], rt = lo(0);
        switch ((h >> 9) & 7) {
        case 0: wr32(a, c.r[rt]); break;
        case 1: wr16(a, (u16)c.r[rt]); break;
        case 2: wr8(a, (u8)c.r[rt]); break;
        case 3: c.r[rt] = (u32)(s32)(s8)rd8(a); break;
        case 4: c.r[rt] = rd32(a); break;
        case 5: c.r[rt] = rd16(a); break;
        case 6: c.r[rt] = rd8(a); break;
        default: c.r[rt] = (u32)(s32)(s16)rd16(a); break;
        }
        return;
    }
    if ((h >> 13) == 3) {
        u32 b = (h >> 12) & 1, l = (h >> 11) & 1;
        u32 a = c.r[lo(3)] + ((h >> 6) & 0x1F) * (b ? 1 : 4), rt = lo(0);
        if (l) c.r[rt] = b ? rd8(a) : rd32(a);
        else if (b) wr8(a, (u8)c.r[rt]); else wr32(a, c.r[rt]);
        return;
    }
    if ((h >> 12) == 8) {
        u32 a = c.r[lo(3)] + ((h >> 6) & 0x1F) * 2;
        if ((h >> 11) & 1) c.r[lo(0)] = rd16(a); else wr16(a, (u16)c.r[lo(0)]);
        return;
    }
    if ((h >> 12) == 9) {
        u32 a = c.r[13] + (h & 0xFF) * 4;
        if ((h >> 11) & 1) c.r[lo(8)] = rd32(a); else wr32(a, c.r[lo(8)]);
        return;
    }
    if ((h >> 12) == 10) {
        u32 imm = (h & 0xFF) * 4;
        c.r[lo(8)] = (h >> 11) & 1 ? c.r[13] + imm : (pc4 & ~3u) + imm;
        return;
    }
    if ((h >> 12) == 11) {
        if ((h >> 8) == 0xB0) {
            u32 imm = (h & 0x7F) * 4;
            if ((h >> 7) & 1) c.r[13] -= imm; else c.r[13] += imm;
            return;
        }
        if ((h & 0x0600) == 0x0400) {
            u32 l = (h >> 11) & 1, rbit = (h >> 8) & 1;
            if (l) {
                u32 a = c.r[13], k = 0;
                for (u32 r = 0; r < 8; r++) if ((h >> r) & 1) c.r[r] = rd32(a + 4 * k++);
                u32 tpc = 0;
                if (rbit) tpc = rd32(a + 4 * k++);
                c.r[13] = a + 4 * k;
                if (rbit) s.npc = tpc;
            } else {
                u32 n = __builtin_popcount(h & 0xFF) + rbit;
                u32 a = c.r[13] - 4 * n, k = 0;
                for (u32 r = 0; r < 8; r++) if ((h >> r) & 1) wr32(a + 4 * k++, c.r[r]);
                if (rbit) wr32(a + 4 * k++, c.r[14]);
                c.r[13] = a;
            }
            return;
        }
        if ((h >> 8) == 0xB2) {
            u32 v = c.r[lo(3)], &d = c.r[lo(0)];
            switch ((h >> 6) & 3) {
            case 0: d = (u32)(s32)(s16)v; break;
            case 1: d = (u32)(s32)(s8)v; break;
            case 2: d = (u16)v; break;
            default: d = (u8)v; break;
            }
            return;
        }
        if ((h >> 8) == 0xBA) {
            u32 v = c.r[lo(3)], &d = c.r[lo(0)];
            switch ((h >> 6) & 3) {
            case 0: d = __builtin_bswap32(v); return;
            case 1: d = rev16(v); return;
            case 3: d = (u32)(s32)(s16)__builtin_bswap16((u16)v); return;
            default: undef(s, h, "rev");
            }
        }
        if ((h >> 8) == 0xBE) rt_trap(c, s.pc, "bkpt");
        if ((h >> 8) == 0xBF || (h & 0xFFE8) == 0xB660 || (h & 0xFFF7) == 0xB650) return;
        undef(s, h, "thumb misc");
    }
    if ((h >> 12) == 12) {
        u32 l = (h >> 11) & 1, rn = lo(8), list = h & 0xFF;
        if (!list) undef(s, h, "ldm empty");
        u32 a = c.r[rn], k = 0;
        if (l) {
            for (u32 r = 0; r < 8; r++) if ((list >> r) & 1) c.r[r] = rd32(a + 4 * k++);
            if (!((list >> rn) & 1)) c.r[rn] = a + 4 * k;
        } else {
            for (u32 r = 0; r < 8; r++) if ((list >> r) & 1) wr32(a + 4 * k++, c.r[r]);
            c.r[rn] = a + 4 * k;
        }
        return;
    }
    if ((h >> 12) == 13) {
        u32 cond = (h >> 8) & 0xF;
        if (cond == 15) { s.svc = true; s.svc_imm = h & 0xFF; return; }
        if (cond == 14) rt_trap(c, s.pc, "udf");
        if (cond_ok(c, cond)) s.npc = (pc4 + ((u32)sxb(h & 0xFF, 8) << 1)) | 1;
        return;
    }
    if (top5 == 0x1C) { s.npc = (pc4 + ((u32)sxb(h & 0x7FF, 11) << 1)) | 1; return; }
    // BL/BLX as two separate halves, like the hardware: the prefix parks the
    // high offset in LR, the suffix branches.
    if (top5 == 0x1E) { c.r[14] = pc4 + ((u32)sxb(h & 0x7FF, 11) << 12); return; }
    if (top5 == 0x1F || top5 == 0x1D) {
        u32 t = c.r[14] + ((h & 0x7FF) << 1);
        c.r[14] = (s.pc + 2) | 1;
        s.npc = top5 == 0x1F ? (t | 1) : (t & ~3u);
        return;
    }
    undef(s, h, "thumb");
}

} // namespace

// Execute exactly one instruction at pc (bit0 = thumb). Returns the next pc.
// An SVC is reported through *svc (imm) instead of being dispatched.
u32 armint_step(Cpu &c, u32 pc, int *svc) {
    St s{c, pc & ~1u, 0};
    c.thumb = pc & 1;
    if (pc & 1) {
        u32 h = rd16(s.pc);
        s.npc = (s.pc + 2) | 1;
        // a BL prefix immediately followed by its suffix runs as one unit
        if ((h >> 11) == 0x1E) {
            u32 h2 = rd16(s.pc + 2);
            if ((h2 >> 11) == 0x1F || (h2 >> 11) == 0x1D) {
                thumb(s, h);
                s.pc += 2; s.npc = (s.pc + 2) | 1;
                thumb(s, h2);
                goto done;
            }
        }
        thumb(s, h);
    } else {
        s.npc = s.pc + 4;
        Arm a(s, rd32(s.pc));
        a.exec();
    }
done:
    if (svc) *svc = s.svc ? (int)s.svc_imm : -1;
    return s.npc;
}

#ifndef R3DS_USE_UNICORN
#include <unordered_map>
std::unordered_map<u32, u64> g_prof[64];   // R3DS_PROFILE: interpreter entries per thread
int g_profiling = 0;
extern struct Thread *cur_thread();
u32 thread_id_of(struct Thread *t);

// Run from pc until control reaches a lifted entry (other than the start),
// an SVC has been serviced, or an interrupt is pending.
u32 interp_run(Cpu &c, u32 pc) {
    if (!g_interp_only && g_miss_log) {
        std::lock_guard<std::mutex> lk(g_miss_mtx);
        if (g_miss_seen.insert(pc).second) { fprintf(g_miss_log, "%08x\n", pc); fflush(g_miss_log); }
    }
    g_stat_interp++;
    if (g_profiling) g_prof[thread_id_of(cur_thread()) & 63][pc]++;
    for (u32 n = 0;; n++) {
        int svc;
        u32 next = armint_step(c, pc, &svc);
        if (svc >= 0) {
            c.r[15] = next & ~1u; c.thumb = next & 1;
            rt_svc(c, (u32)svc);
            return next;
        }
        pc = next;
        c.thumb = pc & 1;
        if (!g_interp_only && disp_lookup(pc)) return pc;
        if ((n & 63) == 63 && g_irq_pending.load(std::memory_order_relaxed)) return pc;
    }
}

void interp_init() {
    const char *m = getenv("R3DS_MISS_LOG");
#ifdef __EMSCRIPTEN__
    if (!m) return;
#endif
    g_miss_log = fopen(m && *m ? m : "miss.log", "a");
}
void interp_kick() {}
void uc_sync_map(u32, u32, u32) {}
void uc_sync_unmap(u32, u32) {}
void uc_sync_protect(u32, u32, u32) {}
#endif
