// Chunk emitter: lowers decoded ARM / Thumb instructions to WebAssembly.
//
// One chunk = one wasm function `(cpu: i32, entry: i32) -> i32` returning the
// next guest pc (bit 0 = Thumb), exactly the contract of the lifter's C++
// chunks. Guest registers live in wasm locals for the duration of the chunk;
// only the registers the chunk touches are loaded from / stored to the Cpu
// struct. Control flow inside a chunk is a `loop` over nested `block`s
// entered through one `br_table` on the label index (forward gotos are plain
// `br`s to the enclosing block, backward gotos re-enter the loop).
//
// Anything without a native lowering is `step`: spill the registers, run one
// instruction on the runtime interpreter (the oracle the lifter's FALLBACK
// used), reload.

import { ByteBuf, OP, EMPTY } from './wasm.js';
import { ror } from './arm.js';

// ------------------------------------------------------------------ layout
export const CPU = {
    R: 0, N: 64, Z: 68, C: 72, V: 76, Q: 80, GE: 84, FPSCR: 88, TPIDRURO: 92, TPIDRURW: 96,
    EXCL_VALID: 100, EXCL_ADDR: 104, EXCL_EPOCH: 108, F: 112,
};

// wasm local indices
const L_CPU = 0, L_LBL = 1;
const L_R0 = 2;                       // r0..r14 -> 2..16
const L_N = 17, L_Z = 18, L_C = 19, L_V = 20;
const L_NPC = 21;
const L_T0 = 22, L_T1 = 23, L_T2 = 24, L_T3 = 25, L_T4 = 26, L_T5 = 27;
const L_X0 = 28, L_X1 = 29;           // i64
const L_F0 = 30, L_F1 = 31;           // f32
const L_D0 = 32, L_D1 = 33;           // f64
const FLAG_BASE = 15;                 // bit index in the use masks for N (15..18 = N,Z,C,V)

export const IMPORTS = { STEP: 0, SVC: 1 };   // function indices of the host imports used by chunks

function pcConst(pc) { return pc | 0; }

export class Emitter {
    constructor(img, opts) {
        this.img = img;
        this.gbase = opts.gbase >>> 0;        // guest address 0 inside the wasm linear memory
        this.irqAddr = opts.irqAddr >>> 0;    // &g_irq_pending
        this.foldLiterals = opts.foldLiterals !== false;
        this.roEnd = img.roEnd;
        this.body = new ByteBuf(4096);
        this.pro = new ByteBuf(256);
        this.epi = new ByteBuf(256);
        this.c = this.body;
        this.readMask = 0; this.writeMask = 0;
        this.d = 0; this.seg = 0; this.k = 0;
        this.labelOf = null; this.chunkPos = null;
    }

    // ---- primitive emitters (all write to this.c)
    op(b) { this.c.u8(b); }
    i32c(x) { this.c.u8(OP.i32_const).sleb(x | 0); }
    lget(i) { this.c.u8(OP.local_get).uleb(i); }
    lset(i) { this.c.u8(OP.local_set).uleb(i); }
    ltee(i) { this.c.u8(OP.local_tee).uleb(i); }
    gR(r) { this.readMask |= 1 << r; this.lget(L_R0 + r); }
    sR(r) { this.writeMask |= 1 << r; this.lset(L_R0 + r); }
    gF(f) { this.readMask |= 1 << (FLAG_BASE + 1 + f); this.lget(L_N + f); }   // f: 0 N, 1 Z, 2 C, 3 V
    sF(f) { this.writeMask |= 1 << (FLAG_BASE + 1 + f); this.lset(L_N + f); }
    // guest memory access: address (u32) on the stack
    memarg(align, off = 0) { this.c.uleb(align).uleb((this.gbase + off) >>> 0); }
    rd8() { this.c.u8(OP.i32_load8_u); this.memarg(0); }
    rd8s() { this.c.u8(OP.i32_load8_s); this.memarg(0); }
    rd16() { this.c.u8(OP.i32_load16_u); this.memarg(0); }
    rd16s() { this.c.u8(OP.i32_load16_s); this.memarg(0); }
    rd32() { this.c.u8(OP.i32_load); this.memarg(0); }
    rd64() { this.c.u8(OP.i64_load); this.memarg(0); }
    wr8() { this.c.u8(OP.i32_store8); this.memarg(0); }
    wr16() { this.c.u8(OP.i32_store16); this.memarg(0); }
    wr32() { this.c.u8(OP.i32_store); this.memarg(0); }
    wr64() { this.c.u8(OP.i64_store); this.memarg(0); }
    // Cpu struct access (cpu pointer local)
    cpuLd(off) { this.lget(L_CPU); this.c.u8(OP.i32_load).uleb(2).uleb(off); }
    cpuSt(off) { this.c.u8(OP.i32_store).uleb(2).uleb(off); }   // [addr, value] on the stack
    add() { this.op(OP.i32_add); }
    sub() { this.op(OP.i32_sub); }
    and() { this.op(OP.i32_and); }
    or() { this.op(OP.i32_or); }
    xor() { this.op(OP.i32_xor); }
    shl() { this.op(OP.i32_shl); }
    shru() { this.op(OP.i32_shr_u); }
    shrs() { this.op(OP.i32_shr_s); }
    eqz() { this.op(OP.i32_eqz); }
    addc(x) { if (x) { this.i32c(x); this.add(); } }   // + constant

    // ---- control flow depth bookkeeping
    dExit() { return this.k - this.seg + this.d; }
    dLoop() { return this.k - 1 - this.seg + this.d; }
    dLabel(j) { return j - this.seg - 1 + this.d; }
    exitConst(pc) { this.i32c(pc); this.lset(L_NPC); this.c.u8(OP.br).uleb(this.dExit()); }
    exitStack() { this.lset(L_NPC); this.c.u8(OP.br).uleb(this.dExit()); }   // next pc on the stack
    beginIf() { this.c.u8(OP.if).u8(EMPTY); this.d++; }
    endIf() { this.c.u8(OP.end); this.d--; }

    // ---- conditions: push a boolean
    cond(cc) {
        switch (cc) {
        case 0: this.gF(1); break;
        case 1: this.gF(1); this.eqz(); break;
        case 2: this.gF(2); break;
        case 3: this.gF(2); this.eqz(); break;
        case 4: this.gF(0); break;
        case 5: this.gF(0); this.eqz(); break;
        case 6: this.gF(3); break;
        case 7: this.gF(3); this.eqz(); break;
        case 8: this.gF(2); this.gF(1); this.eqz(); this.and(); break;                 // hi
        case 9: this.gF(2); this.gF(1); this.eqz(); this.and(); this.eqz(); break;     // ls
        case 10: this.gF(0); this.gF(3); this.op(OP.i32_eq); break;                     // ge
        case 11: this.gF(0); this.gF(3); this.op(OP.i32_ne); break;                     // lt
        case 12: this.gF(1); this.eqz(); this.gF(0); this.gF(3); this.op(OP.i32_eq); this.and(); break;   // gt
        case 13: this.gF(1); this.gF(0); this.gF(3); this.op(OP.i32_ne); this.or(); break;                // le
        default: this.i32c(1);
        }
    }

    // ---- register value helpers. `rv(n, pcval)` pushes r[n], or the constant pc value for r15
    rv(n, pcval) { if (n === 15) this.i32c(pcval); else this.gR(n); }
    // write to a register; the value is on the stack. r15 = branch.
    wreg(ins, n, interwork = false) {
        if (n === 15) {
            if (!interwork) { this.i32c(ins.thumb ? 1 : -4); this.op(ins.thumb ? OP.i32_or : OP.i32_and); }
            this.exitStack();
        } else this.sR(n);
    }

    // ---- flags
    nz() {   // result in T2
        this.lget(L_T2); this.i32c(31); this.shru(); this.sF(0);
        this.lget(L_T2); this.eqz(); this.sF(1);
    }

    // adds(a, b, cin) -> res in T2 and NZCV set. pa()/pb() push the operands; each may be
    // evaluated more than once, so they must be side-effect free (locals / constants).
    adds(pa, pb, cin) {   // cin: 0, 1 or 'c'
        pa(); pb(); this.add();
        if (cin === 1) this.i32c(1), this.add();
        else if (cin === 'c') this.gF(2), this.add();
        this.lset(L_T2);
        this.lget(L_T2); this.i32c(31); this.shru(); this.sF(0);
        this.lget(L_T2); this.eqz(); this.sF(1);
        if (cin === 0) { this.lget(L_T2); pa(); this.op(OP.i32_lt_u); }
        else if (cin === 1) { this.lget(L_T2); pa(); this.op(OP.i32_le_u); }
        else {
            // c = res < a || (cin && res == a); cin is the old C, still intact
            this.lget(L_T2); pa(); this.op(OP.i32_lt_u);
            this.gF(2); this.lget(L_T2); pa(); this.op(OP.i32_eq); this.and(); this.or();
        }
        this.sF(2);
        pa(); this.lget(L_T2); this.xor(); pb(); this.lget(L_T2); this.xor(); this.and(); this.i32c(31); this.shru(); this.sF(3);
    }
}
