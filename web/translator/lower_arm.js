// ARM-state lowering. Semantics follow lift/armdec.py statement by statement.
import { Emitter, CPU, IMPORTS } from './emit.js';
import { OP, EMPTY } from './wasm.js';
import { ror } from './arm.js';

const P = Emitter.prototype;
const L_T0 = 22, L_T1 = 23, L_T2 = 24, L_T3 = 25, L_T4 = 26, L_T5 = 27, L_X0 = 28, L_LBL = 1, L_CPU = 0, L_NPC = 21;
const L_N = 17, L_Z = 18, L_C = 19, L_V = 20;

// operand descriptors: push() / not() push the value / its complement
P.dReg = function (n, pcval) {
    return { push: () => this.rv(n, pcval), not: () => { this.rv(n, pcval); this.i32c(-1); this.xor(); } };
};
P.dConst = function (v) { return { push: () => this.i32c(v), not: () => this.i32c(~v) }; };
P.dLocal = function (i) {
    return { push: () => this.lget(i), not: () => { this.lget(i); this.i32c(-1); this.xor(); } };
};

// shift by immediate on the value in local `src` (already holding rm). Pushes the result;
// carry (if wanted) is left in T5. typ 0 LSL 1 LSR 2 ASR 3 ROR/RRX
P.immShift = function (src, typ, amt, wantCarry) {
    const g = () => this.lget(src);
    if (typ === 0) {
        if (amt === 0) { g(); return false; }
        if (wantCarry) { g(); this.i32c(32 - amt); this.shru(); this.i32c(1); this.and(); this.lset(L_T5); }
        g(); this.i32c(amt); this.shl();
        return wantCarry;
    }
    if (typ === 1) {
        if (amt === 0) amt = 32;
        if (amt === 32) {
            if (wantCarry) { g(); this.i32c(31); this.shru(); this.lset(L_T5); }
            this.i32c(0);
            return wantCarry;
        }
        if (wantCarry) { g(); this.i32c(amt - 1); this.shru(); this.i32c(1); this.and(); this.lset(L_T5); }
        g(); this.i32c(amt); this.shru();
        return wantCarry;
    }
    if (typ === 2) {
        if (amt === 0) amt = 32;
        if (amt === 32) {
            if (wantCarry) { g(); this.i32c(31); this.shru(); this.lset(L_T5); }
            g(); this.i32c(31); this.shrs();
            return wantCarry;
        }
        if (wantCarry) { g(); this.i32c(amt - 1); this.shru(); this.i32c(1); this.and(); this.lset(L_T5); }
        g(); this.i32c(amt); this.shrs();
        return wantCarry;
    }
    if (amt === 0) {   // RRX
        if (wantCarry) { g(); this.i32c(1); this.and(); this.lset(L_T5); }
        this.gF(2); this.i32c(31); this.shl(); g(); this.i32c(1); this.shru(); this.or();
        return wantCarry;
    }
    if (wantCarry) { g(); this.i32c(amt - 1); this.shru(); this.i32c(1); this.and(); this.lset(L_T5); }
    g(); this.i32c(amt); this.op(OP.i32_rotr);
    return wantCarry;
};

// operand 2 of a data-processing instruction. Returns {d: descriptor, carry: null | 'T5' | 0 | 1}
// or null when this form needs the interpreter (register shift that must produce a carry).
P.operand2 = function (ins, w, wantCarry) {
    const pc = ins.addr + 8;
    if ((w >>> 25) & 1) {
        const rot = ((w >>> 8) & 0xF) * 2;
        const v = ror(w & 0xFF, rot);
        return { d: this.dConst(v | 0), carry: rot ? (v >>> 31) : null };
    }
    const rm = w & 0xF, typ = (w >>> 5) & 3;
    if ((w >>> 4) & 1) {   // register-specified shift
        if (wantCarry) return null;
        const rs = (w >>> 8) & 0xF;
        this.rv(rm, pc + 4); this.lset(L_T3);          // T3 = value
        this.rv(rs, pc + 4); this.i32c(0xFF); this.and(); this.lset(L_T5);   // T5 = amount
        const v = () => this.lget(L_T3), a = () => this.lget(L_T5), lt32 = () => { a(); this.i32c(32); this.op(OP.i32_lt_u); };
        if (typ === 0) { v(); a(); this.shl(); this.i32c(0); lt32(); this.op(OP.select); }
        else if (typ === 1) { v(); a(); this.shru(); this.i32c(0); lt32(); this.op(OP.select); }
        else if (typ === 2) { v(); a(); this.shrs(); v(); this.i32c(31); this.shrs(); lt32(); this.op(OP.select); }
        else { v(); a(); this.op(OP.i32_rotr); }
        this.lset(L_T4);
        return { d: this.dLocal(L_T4), carry: null };
    }
    const amt = (w >>> 7) & 0x1F;
    if (typ === 0 && amt === 0) return { d: this.dReg(rm, pc), carry: null };
    this.rv(rm, pc); this.lset(L_T3);
    const hasCarry = this.immShift(L_T3, typ, amt, wantCarry);
    this.lset(L_T4);
    return { d: this.dLocal(L_T4), carry: hasCarry ? 'T5' : null };
};

P.setCarry = function (c) {
    if (c === null) return;
    if (c === 'T5') this.lget(L_T5); else this.i32c(c);
    this.sF(2);
};

P.jump = function (ins, t) {
    const ci = this.labelIndex(t);
    if (ci < 0) { this.exitConst(t); return; }
    if ((t & ~1) <= (ins.addr & ~1)) {   // backward: poll for interrupts, re-enter the loop
        this.i32c(0); this.c.u8(0xFE).u8(0x10).uleb(2).uleb(this.irqAddr);
        this.beginIf(); this.exitConst(t); this.endIf();
        this.i32c(ci); this.lset(L_LBL); this.c.u8(OP.br).uleb(this.dLoop());
    } else {
        this.c.u8(OP.br).uleb(this.dLabel(ci));
    }
};

// spill everything, run one instruction on the interpreter, reload
P.spill = function () {
    for (let r = 0; r < 15; r++) { this.lget(L_CPU); this.gR(r); this.cpuSt(CPU.R + 4 * r); }
    for (let f = 0; f < 4; f++) { this.lget(L_CPU); this.gF(f); this.cpuSt(CPU.N + 4 * f); }
};
P.reload = function () {
    for (let r = 0; r < 15; r++) { this.cpuLd(CPU.R + 4 * r); this.sR(r); }
    for (let f = 0; f < 4; f++) { this.cpuLd(CPU.N + 4 * f); this.sF(f); }
};
P.step = function (ins) {
    const pc = (ins.addr | (ins.thumb ? 1 : 0)) >>> 0;
    this.spill();
    this.lget(L_CPU); this.i32c(pc); this.c.u8(OP.call).uleb(IMPORTS.STEP);
    this.lset(L_NPC);
    this.reload();
    const next = (pc + ins.size) >>> 0;
    if (ins.bad) { this.c.u8(OP.br).uleb(this.dExit()); return; }
    this.lget(L_NPC); this.i32c(next); this.op(OP.i32_ne);
    this.beginIf(); this.c.u8(OP.br).uleb(this.dExit()); this.endIf();
};

// ------------------------------------------------------------------ instructions
const NATIVE_ARM = new Set(['dp', 'mul', 'mull', 'ls', 'lsx', 'ldm', 'b', 'blx_imm', 'bx', 'blx_reg', 'clz', 'svc']);

// true when the instruction has a native lowering (otherwise it runs on the interpreter)
export function armIsNative(ins) {
    if (ins.bad || !NATIVE_ARM.has(ins.kind)) return false;
    if (ins.kind === 'dp') {   // register-specified shift that must produce a carry: interpreter
        const w = ins.w, opc = (w >>> 21) & 0xF, s = (w >>> 20) & 1;
        const logical = opc === 0 || opc === 1 || opc === 8 || opc === 9 || opc === 12 || opc === 13 || opc === 14 || opc === 15;
        if (s && logical && !((w >>> 25) & 1) && ((w >>> 4) & 1)) return false;
    }
    return true;
}

P.lowerArm = function (ins) {
    const cc = ins.cond;
    if (ins.kind === 'nop') return;
    if (!armIsNative(ins)) { this.step(ins); return; }
    if (cc !== 14) { this.cond(cc); this.beginIf(); }
    this.lowerArmBody(ins);
    if (cc !== 14) this.endIf();
};

P.lowerArmBody = function (ins) {
    const w = ins.w, pc8 = ins.addr + 8;
    switch (ins.kind) {
    case 'dp': return this.armDp(ins, w, pc8);
    case 'mul': return this.armMul(ins, w);
    case 'mull': return this.armMull(ins, w);
    case 'ls': return this.armLs(ins, w, pc8);
    case 'lsx': return this.armLsx(ins, w, pc8);
    case 'ldm': return this.armLdm(ins, w, pc8);
    case 'b': {
        if (ins.call) { this.i32c(ins.addr + 4); this.sR(14); }
        this.jump(ins, ins.t);
        return;
    }
    case 'blx_imm': this.i32c(ins.addr + 4); this.sR(14); this.jump(ins, ins.t); return;
    case 'bx': this.rv(w & 0xF, pc8); this.exitStack(); return;
    case 'blx_reg': this.rv(w & 0xF, pc8); this.lset(L_T0); this.i32c(ins.addr + 4); this.sR(14); this.lget(L_T0); this.exitStack(); return;
    case 'clz': this.gR(w & 0xF); this.op(OP.i32_clz); this.sR((w >>> 12) & 0xF); return;
    case 'svc': {
        this.spill();
        this.lget(L_CPU); this.i32c(w & 0xFFFFFF); this.i32c(ins.addr + 4); this.c.u8(OP.call).uleb(IMPORTS.SVC);
        this.reload();
        return;
    }
    default: throw new Error('no lowering for ' + ins.kind);
    }
};

P.armDp = function (ins, w, pc8) {
    const opc = (w >>> 21) & 0xF, s = (w >>> 20) & 1, rn = (w >>> 16) & 0xF, rd = (w >>> 12) & 0xF;
    const logical = opc === 0 || opc === 1 || opc === 8 || opc === 9 || opc === 12 || opc === 13 || opc === 14 || opc === 15;
    const regShiftForm = !((w >>> 25) & 1) && ((w >>> 4) & 1);
    const o = this.operand2(ins, w, !!s && logical);
    if (o === null) throw new Error('dp: register shift with carry');
    const rnPc = regShiftForm ? pc8 + 4 : pc8;
    const dn = this.dReg(rn, rnPc);
    if (logical) {
        const rnop = opc === 13 || opc === 15 ? null : dn;
        const value = () => {
            switch (opc) {
            case 0: case 8: rnop.push(); o.d.push(); this.and(); break;
            case 1: case 9: rnop.push(); o.d.push(); this.xor(); break;
            case 12: rnop.push(); o.d.push(); this.or(); break;
            case 13: o.d.push(); break;
            case 14: rnop.push(); o.d.not(); this.and(); break;
            case 15: o.d.not(); break;
            }
        };
        if (opc === 8 || opc === 9) { value(); this.lset(L_T2); this.nz(); this.setCarry(o.carry); return; }
        if (s) {
            value(); this.lset(L_T2); this.nz(); this.setCarry(o.carry);
            this.lget(L_T2); this.wreg(ins, rd);
        } else { value(); this.wreg(ins, rd); }
        return;
    }
    // arithmetic
    let A, B, cin;
    switch (opc) {
    case 2: A = dn; B = { push: o.d.not }; cin = 1; break;
    case 3: A = o.d; B = { push: dn.not }; cin = 1; break;
    case 4: A = dn; B = o.d; cin = 0; break;
    case 5: A = dn; B = o.d; cin = 'c'; break;
    case 6: A = dn; B = { push: o.d.not }; cin = 'c'; break;
    case 7: A = o.d; B = { push: dn.not }; cin = 'c'; break;
    case 10: A = dn; B = { push: o.d.not }; cin = 1; break;
    case 11: A = dn; B = o.d; cin = 0; break;
    }
    if (opc === 10 || opc === 11) { this.adds(A.push, B.push, cin); return; }
    if (s) { this.adds(A.push, B.push, cin); this.lget(L_T2); this.wreg(ins, rd); return; }
    // no flags: plain arithmetic
    A.push(); B.push(); this.add();
    if (cin === 1) { this.i32c(1); this.add(); } else if (cin === 'c') { this.gF(2); this.add(); }
    this.wreg(ins, rd);
};

P.armMul = function (ins, w) {
    const op = (w >>> 21) & 7, s = (w >>> 20) & 1;
    const rd = (w >>> 16) & 0xF, rn = (w >>> 12) & 0xF, rs = (w >>> 8) & 0xF, rm = w & 0xF;
    this.gR(rm); this.gR(rs); this.op(OP.i32_mul);
    if (op === 1) { this.gR(rn); this.add(); }
    if (s) { this.ltee(L_T2); this.sR(rd); this.nz(); } else this.sR(rd);
};

P.armMull = function (ins, w) {
    const op = (w >>> 21) & 7, s = (w >>> 20) & 1;
    const rd = (w >>> 16) & 0xF, rn = (w >>> 12) & 0xF, rs = (w >>> 8) & 0xF, rm = w & 0xF;
    const signed = op >= 6, acc = op & 1;
    this.gR(rm); this.op(signed ? OP.i64_extend_i32_s : OP.i64_extend_i32_u);
    this.gR(rs); this.op(signed ? OP.i64_extend_i32_s : OP.i64_extend_i32_u);
    this.op(OP.i64_mul);
    if (acc) {
        this.gR(rd); this.op(OP.i64_extend_i32_u); this.c.u8(OP.i64_const).sleb64(32n); this.op(OP.i64_shl);
        this.gR(rn); this.op(OP.i64_extend_i32_u); this.op(OP.i64_or);
        this.op(OP.i64_add);
    }
    this.c.u8(OP.local_tee).uleb(L_X0);
    this.op(OP.i32_wrap_i64); this.sR(rn);
    this.c.u8(OP.local_get).uleb(L_X0); this.c.u8(OP.i64_const).sleb64(32n); this.op(OP.i64_shr_u); this.op(OP.i32_wrap_i64); this.sR(rd);
    if (s) {
        this.c.u8(OP.local_get).uleb(L_X0); this.c.u8(OP.i64_const).sleb64(63n); this.op(OP.i64_shr_u); this.op(OP.i32_wrap_i64); this.sF(0);
        this.c.u8(OP.local_get).uleb(L_X0); this.op(OP.i64_eqz); this.sF(1);
    }
};

// effective address into T0, register writeback code shared by ls/lsx.
// off: {imm} or {reg, typ, amt} ; loads pass `skipWb` when the load target is the base
P.eaAddr = function (ins, rn, p, u, off, pcval) {
    // T0 = address used for the access
    this.rv(rn, pcval);
    if (p) { this.pushOff(off); u ? this.add() : this.sub(); }
    this.lset(L_T0);
};
P.pushOff = function (off) {
    if (off.imm !== undefined) { this.i32c(off.imm); return; }
    // register offset, optionally shifted by an immediate
    this.rv(off.reg, off.pcval); this.lset(L_T3);
    this.immShift(L_T3, off.typ, off.amt, false);
};
P.writeback = function (rn, p, u, off) {
    if (p) { this.lget(L_T0); this.sR(rn); return; }
    // post-indexed: rn = a +/- off
    this.lget(L_T0); this.pushOff(off); u ? this.add() : this.sub(); this.sR(rn);
};

P.armLs = function (ins, w, pc8) {
    const reg = (w >>> 25) & 1, p = (w >>> 24) & 1, u = (w >>> 23) & 1, b = (w >>> 22) & 1, wb = (w >>> 21) & 1, l = (w >>> 20) & 1;
    const rn = (w >>> 16) & 0xF, rt = (w >>> 12) & 0xF;
    const wback = wb || !p;
    const off = reg ? { reg: w & 0xF, typ: (w >>> 5) & 3, amt: (w >>> 7) & 0x1F, pcval: pc8 } : { imm: w & 0xFFF };
    // literal loads from read-only memory fold to constants
    if (l && !b && rn === 15 && p && !reg && rt !== 15 && this.foldLiterals) {
        const a = (pc8 + (u ? 1 : -1) * (w & 0xFFF)) >>> 0;
        if (a >= this.img.base && a + 4 <= this.roEnd) { this.i32c(this.img.w32(a)); this.sR(rt); return; }
    }
    this.eaAddr(ins, rn, p, u, off, pc8);
    if (l) {
        if (rt === 15) {
            this.lget(L_T0);
            if (!p) { /* post-index: address is the base itself */ }
            b ? this.rd8() : this.rd32(); this.lset(L_T1);
            if (wback) this.writeback(rn, p, u, off);
            this.lget(L_T1); this.exitStack();
            return;
        }
        if (wback && rn === rt) { this.lget(L_T0); b ? this.rd8() : this.rd32(); this.sR(rt); return; }
        if (wback) this.writeback(rn, p, u, off);
        this.lget(L_T0); b ? this.rd8() : this.rd32(); this.sR(rt);
    } else {
        this.lget(L_T0); this.rv(rt, pc8); b ? this.wr8() : this.wr32();
        if (wback) this.writeback(rn, p, u, off);
    }
};

P.armLsx = function (ins, w, pc8) {
    const p = (w >>> 24) & 1, u = (w >>> 23) & 1, i = (w >>> 22) & 1, wb = (w >>> 21) & 1, l = (w >>> 20) & 1;
    const rn = (w >>> 16) & 0xF, rt = (w >>> 12) & 0xF, sh = (w >>> 5) & 3;
    const wback = wb || !p;
    const off = i ? { imm: ((w >>> 4) & 0xF0) | (w & 0xF) } : { reg: w & 0xF, typ: 0, amt: 0, pcval: pc8 };
    if (l && i && rn === 15 && p && this.foldLiterals) {
        const a = (pc8 + (u ? 1 : -1) * off.imm) >>> 0;
        if (a >= this.img.base && a + 2 <= this.roEnd) {
            const v = sh === 1 ? this.img.w16(a) : sh === 2 ? ((this.img.w16(a) & 0xFF) << 24) >> 24 : (this.img.w16(a) << 16) >> 16;
            this.i32c(v); this.sR(rt); return;
        }
    }
    this.eaAddr(ins, rn, p, u, off, pc8);
    if (l) {
        const wbFirst = wback && rn !== rt;
        if (wbFirst) this.writeback(rn, p, u, off);
        this.lget(L_T0);
        if (sh === 1) this.rd16(); else if (sh === 2) this.rd8s(); else this.rd16s();
        this.sR(rt);
    } else {
        this.lget(L_T0); this.rv(rt, pc8); this.wr16();
        if (wback) this.writeback(rn, p, u, off);
    }
};

P.armLdm = function (ins, w, pc8) {
    const p = (w >>> 24) & 1, u = (w >>> 23) & 1, wb = (w >>> 21) & 1, l = (w >>> 20) & 1, rn = (w >>> 16) & 0xF;
    const regs = [];
    for (let i = 0; i < 16; i++) if ((w >>> i) & 1) regs.push(i);
    const n = regs.length;
    // T0 = start address
    this.gR(rn);
    if (u) { if (p) this.addc(4); }
    else this.addc(p ? -4 * n : -4 * n + 4);
    this.lset(L_T0);
    const newbase = () => { this.gR(rn); this.addc(u ? 4 * n : -4 * n); };
    if (l) {
        regs.forEach((r, k) => {
            this.lget(L_T0); this.addc(0); this.c.u8(OP.i32_load).uleb(0).uleb((this.gbase + 4 * k) >>> 0);
            if (r === 15) this.lset(L_T1);
            else if (r === rn) this.lset(L_T2);
            else this.sR(r);
        });
        const hasBase = regs.includes(rn);
        if (wb && !hasBase) { newbase(); this.sR(rn); }
        if (hasBase) { this.lget(L_T2); this.sR(rn); }
        if (regs.includes(15)) { this.lget(L_T1); this.exitStack(); }
    } else {
        regs.forEach((r, k) => {
            this.lget(L_T0); this.rv(r, pc8); this.c.u8(OP.i32_store).uleb(0).uleb((this.gbase + 4 * k) >>> 0);
        });
        if (wb) { newbase(); this.sR(rn); }
    }
};
