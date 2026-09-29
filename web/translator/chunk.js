// Chunk driver: turns the instructions of one chunk into a complete wasm function body.
import { ByteBuf, OP, EMPTY, T } from './wasm.js';
import { Emitter, CPU } from './emit.js';
import './lower_arm.js';
import './lower_thumb.js';
import { F_ENT_ARM, F_ENT_THUMB } from './discover.js';

const L_LBL = 1, L_NPC = 21;
const FLAG_BASE = 15;

Emitter.prototype.labelIndex = function (t) {
    // binary search of t among the chunk's pcs
    const pcs = this.pcs;
    let lo = this.pcLo, hi = this.pcHi - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >>> 1, v = pcs[mid];
        if (v === t) return this.labelOfInsn[mid - this.pcLo];
        if (v < t) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
};

// Emit chunk `pcs[lo..hi)`. Returns {body: ByteBuf, entries: [[pc, label], ...]}
Emitter.prototype.emitChunk = function (pcs, lo, hi, flags) {
    const img = this.img, n = hi - lo;
    this.pcs = pcs; this.pcLo = lo; this.pcHi = hi;
    const insns = new Array(n);
    for (let i = 0; i < n; i++) insns[i] = img.decode(pcs[lo + i]);

    // labels: entry points and in-chunk jump targets
    const isLabel = new Uint8Array(n);
    isLabel[0] = 1;
    const base = img.base;
    const entryPcs = [];
    for (let i = 0; i < n; i++) {
        const pc = pcs[lo + i], f = flags[((pc & ~1) - base) >>> 1];
        if (i === 0 || (f & ((pc & 1) ? F_ENT_THUMB : F_ENT_ARM))) { isLabel[i] = 1; entryPcs.push(pc); }
        const ins = insns[i];
        if (ins.t >= 0 && !ins.bad && (ins.kind === 'b' || ins.kind === 't_bcond' || ins.kind === 't_b' || ins.kind === 't_bl' || ins.kind === 'blx_imm')) {
            // in-chunk target?
            let a = lo, b = hi - 1, found = -1;
            while (a <= b) { const m = (a + b) >>> 1, v = pcs[m]; if (v === ins.t) { found = m; break; } if (v < ins.t) a = m + 1; else b = m - 1; }
            if (found >= 0) isLabel[found - lo] = 1;
        }
    }
    const labelOfInsn = new Int32Array(n).fill(-1);
    const labelInsn = [];
    for (let i = 0; i < n; i++) if (isLabel[i]) { labelOfInsn[i] = labelInsn.length; labelInsn.push(i); }
    this.labelOfInsn = labelOfInsn;
    const k = labelInsn.length;
    this.k = k; this.d = 0; this.seg = 0;
    this.readMask = 0; this.writeMask = 0;

    const c = this.body; c.reset(); this.c = c;
    c.u8(OP.block).u8(EMPTY);
    c.u8(OP.loop).u8(EMPTY);
    for (let i = 0; i < k; i++) c.u8(OP.block).u8(EMPTY);
    c.u8(OP.local_get).uleb(L_LBL);
    c.u8(OP.br_table).uleb(k);
    for (let i = 0; i < k; i++) c.uleb(i);
    c.uleb(0);
    for (let s = 0; s < k; s++) {
        c.u8(OP.end);   // block s: its end is the start of label s
        this.seg = s;
        const from = labelInsn[s], to = s + 1 < k ? labelInsn[s + 1] : n;
        for (let i = from; i < to; i++) {
            const ins = insns[i];
            if (ins.thumb) this.lowerThumb(ins); else this.lowerArm(ins);
        }
    }
    const last = insns[n - 1];
    if (last.falls && !last.bad) {
        this.i32c((pcs[hi - 1] & 1 ? 1 : 0) | ((last.addr + last.size) >>> 0)); this.lset(L_NPC);
    } else if (last.bad) { /* the step already exits */ }
    c.u8(OP.end);   // loop
    c.u8(OP.end);   // exit block

    // prologue / epilogue from the use masks
    const loadMask = this.readMask | this.writeMask;
    const pro = this.pro; pro.reset();
    const epi = this.epi; epi.reset();
    for (let r = 0; r < 15; r++) if (loadMask & (1 << r)) pro.u8(OP.local_get).uleb(0).u8(OP.i32_load).uleb(2).uleb(CPU.R + 4 * r).u8(OP.local_set).uleb(2 + r);
    for (let f = 0; f < 4; f++) if (loadMask & (1 << (FLAG_BASE + 1 + f))) pro.u8(OP.local_get).uleb(0).u8(OP.i32_load).uleb(2).uleb(CPU.N + 4 * f).u8(OP.local_set).uleb(17 + f);
    for (let r = 0; r < 15; r++) if (this.writeMask & (1 << r)) epi.u8(OP.local_get).uleb(0).u8(OP.local_get).uleb(2 + r).u8(OP.i32_store).uleb(2).uleb(CPU.R + 4 * r);
    for (let f = 0; f < 4; f++) if (this.writeMask & (1 << (FLAG_BASE + 1 + f))) epi.u8(OP.local_get).uleb(0).u8(OP.local_get).uleb(17 + f).u8(OP.i32_store).uleb(2).uleb(CPU.N + 4 * f);
    epi.u8(OP.local_get).uleb(L_NPC).u8(OP.end);

    const out = new ByteBuf(pro.length + c.length + epi.length + 32);
    // locals: 26 x i32 (2..27), 2 x i64, 2 x f32, 2 x f64
    out.uleb(4).uleb(26).u8(T.i32).uleb(2).u8(T.i64).uleb(2).u8(T.f32).uleb(2).u8(T.f64);
    out.buf(pro).buf(c).buf(epi);

    const entries = [];
    for (const pc of entryPcs) {
        // label number of the entry pc
        let a = lo, b = hi - 1;
        while (a <= b) { const m = (a + b) >>> 1, v = pcs[m]; if (v === pc) { entries.push([pc, labelOfInsn[m - lo]]); break; } if (v < pc) a = m + 1; else b = m - 1; }
    }
    return { body: out, entries };
};

export { Emitter };
