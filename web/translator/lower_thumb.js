// Thumb-1 lowering (semantics from lift/armdec.py `_dec_thumb`).
import { Emitter } from './emit.js';
import { OP } from './wasm.js';
import './lower_arm.js';

const P = Emitter.prototype;
const L_T0 = 22, L_T1 = 23, L_T2 = 24, L_T3 = 25, L_T5 = 27;

const NATIVE_THUMB = new Set(['t_shift', 't_addsub', 't_imm', 't_alu', 't_hi', 't_ldrlit', 't_lsreg', 't_lsimm', 't_lsh',
                              't_sp', 't_addpc', 't_spadj', 't_pushpop', 't_ext', 't_ldm', 't_bcond', 't_b', 't_bl', 'svc']);

export function thumbIsNative(ins) {
    if (ins.bad || !NATIVE_THUMB.has(ins.kind)) return false;
    if (ins.kind === 't_alu') {
        const op = (ins.w >>> 6) & 0xF;
        if (op === 2 || op === 3 || op === 4 || op === 7) return false;   // shifts by register produce a carry
    }
    return true;
}

P.lowerThumb = function (ins) {
    if (ins.kind === 'nop') return;
    if (!thumbIsNative(ins)) { this.step(ins); return; }
    if (ins.cond !== 14) { this.cond(ins.cond); this.beginIf(); }
    this.lowerThumbBody(ins);
    if (ins.cond !== 14) this.endIf();
};

P.lowerThumbBody = function (ins) {
    const h = ins.w, pc = ins.addr + 4;
    const lo = (s) => (h >>> s) & 7;
    switch (ins.kind) {
    case 't_shift': {
        const op = h >>> 11, imm = (h >>> 6) & 0x1F, rm = lo(3), rd = lo(0);
        if (op === 0 && imm === 0) { this.gR(rm); this.lset(L_T2); this.lget(L_T2); this.sR(rd); this.nz(); return; }
        this.gR(rm); this.lset(L_T3);
        this.immShift(L_T3, op, imm, true);
        this.lset(L_T2); this.lget(L_T2); this.sR(rd); this.nz();
        this.lget(L_T5); this.sF(2);
        return;
    }
    case 't_addsub': {
        const imm = (h >>> 10) & 1, sub = (h >>> 9) & 1, rn = lo(3), rd = lo(0);
        const a = this.dReg(rn, pc);
        const b = imm ? this.dConst((h >>> 6) & 7) : this.dReg(lo(6), pc);
        if (sub) this.adds(a.push, b.not, 1); else this.adds(a.push, b.push, 0);
        this.lget(L_T2); this.sR(rd);
        return;
    }
    case 't_imm': {
        const op = (h >>> 11) - 4, rd = lo(8), imm = h & 0xFF;
        const a = this.dReg(rd, pc), b = this.dConst(imm);
        if (op === 0) { this.i32c(imm); this.lset(L_T2); this.lget(L_T2); this.sR(rd); this.nz(); }
        else if (op === 1) this.adds(a.push, b.not, 1);
        else if (op === 2) { this.adds(a.push, b.push, 0); this.lget(L_T2); this.sR(rd); }
        else { this.adds(a.push, b.not, 1); this.lget(L_T2); this.sR(rd); }
        return;
    }
    case 't_alu': {
        const op = (h >>> 6) & 0xF, rm = lo(3), rd = lo(0);
        const D = this.dReg(rd, pc), M = this.dReg(rm, pc);
        const logic = () => { this.lset(L_T2); this.lget(L_T2); this.sR(rd); this.nz(); };
        switch (op) {
        case 0: D.push(); M.push(); this.and(); logic(); break;
        case 1: D.push(); M.push(); this.xor(); logic(); break;
        case 5: this.adds(D.push, M.push, 'c'); this.lget(L_T2); this.sR(rd); break;
        case 6: this.adds(D.push, M.not, 'c'); this.lget(L_T2); this.sR(rd); break;
        case 8: D.push(); M.push(); this.and(); this.lset(L_T2); this.nz(); break;
        case 9: this.adds(() => this.i32c(0), M.not, 1); this.lget(L_T2); this.sR(rd); break;
        case 10: this.adds(D.push, M.not, 1); break;
        case 11: this.adds(D.push, M.push, 0); break;
        case 12: D.push(); M.push(); this.or(); logic(); break;
        case 13: M.push(); D.push(); this.op(OP.i32_mul); logic(); break;
        case 14: D.push(); M.not(); this.and(); logic(); break;
        default: M.not(); logic(); break;
        }
        return;
    }
    case 't_hi': {
        const op = (h >>> 8) & 3, rd = ((h >>> 4) & 8) | (h & 7), rm = (h >>> 3) & 0xF;
        const rmv = () => this.rv(rm, pc), rdv = () => this.rv(rd, pc);
        if (op === 0) {
            if (rd === 15) { this.i32c(pc); rmv(); this.add(); this.i32c(1); this.or(); this.exitStack(); }
            else { this.gR(rd); rmv(); this.add(); this.sR(rd); }
        } else if (op === 1) {
            this.adds(rdv, () => { rmv(); this.i32c(-1); this.xor(); }, 1);
        } else if (op === 2) {
            if (rd === 15) { rmv(); this.i32c(1); this.or(); this.exitStack(); }
            else { rmv(); this.sR(rd); }
        } else if ((h >>> 7) & 1) {   // BLX reg
            rmv(); this.lset(L_T0); this.i32c((ins.addr + 2) | 1); this.sR(14); this.lget(L_T0); this.exitStack();
        } else { rmv(); this.exitStack(); }   // BX
        return;
    }
    case 't_ldrlit': {
        const a = (((pc & ~3) >>> 0) + (h & 0xFF) * 4) >>> 0, rt = lo(8);
        if (this.foldLiterals && a >= this.img.base && a + 4 <= this.roEnd) { this.i32c(this.img.w32(a)); this.sR(rt); return; }
        this.i32c(a); this.rd32(); this.sR(rt);
        return;
    }
    case 't_lsreg': {
        const op = (h >>> 9) & 7, rm = lo(6), rn = lo(3), rt = lo(0);
        this.gR(rn); this.gR(rm); this.add();
        if (op < 3) {
            this.gR(rt);
            (op === 0 ? this.wr32 : op === 1 ? this.wr16 : this.wr8).call(this);
        } else {
            (op === 3 ? this.rd8s : op === 4 ? this.rd32 : op === 5 ? this.rd16 : op === 6 ? this.rd8 : this.rd16s).call(this);
            this.sR(rt);
        }
        return;
    }
    case 't_lsimm': {
        const b = (h >>> 12) & 1, l = (h >>> 11) & 1, imm = ((h >>> 6) & 0x1F) * (b ? 1 : 4), rn = lo(3), rt = lo(0);
        this.gR(rn); this.addc(imm);
        if (l) { b ? this.rd8() : this.rd32(); this.sR(rt); }
        else { this.gR(rt); b ? this.wr8() : this.wr32(); }
        return;
    }
    case 't_lsh': {
        const l = (h >>> 11) & 1, imm = ((h >>> 6) & 0x1F) * 2, rn = lo(3), rt = lo(0);
        this.gR(rn); this.addc(imm);
        if (l) { this.rd16(); this.sR(rt); } else { this.gR(rt); this.wr16(); }
        return;
    }
    case 't_sp': {
        const l = (h >>> 11) & 1, rt = lo(8);
        this.gR(13); this.addc((h & 0xFF) * 4);
        if (l) { this.rd32(); this.sR(rt); } else { this.gR(rt); this.wr32(); }
        return;
    }
    case 't_addpc': {
        const rd = lo(8), imm = (h & 0xFF) * 4;
        if ((h >>> 11) & 1) { this.gR(13); this.addc(imm); } else this.i32c(((pc & ~3) >>> 0) + imm);
        this.sR(rd);
        return;
    }
    case 't_spadj': {
        const imm = (h & 0x7F) * 4;
        this.gR(13); this.addc((h >>> 7) & 1 ? -imm : imm); this.sR(13);
        return;
    }
    case 't_pushpop': {
        const l = (h >>> 11) & 1, rbit = (h >>> 8) & 1;
        const regs = [];
        for (let i = 0; i < 8; i++) if ((h >>> i) & 1) regs.push(i);
        if (l) {
            if (rbit) regs.push(15);
            this.gR(13); this.lset(L_T0);
            regs.forEach((r, k) => {
                this.lget(L_T0); this.c.u8(OP.i32_load).uleb(0).uleb((this.gbase + 4 * k) >>> 0);
                if (r === 15) this.lset(L_T1); else this.sR(r);
            });
            this.lget(L_T0); this.addc(4 * regs.length); this.sR(13);
            if (rbit) { this.lget(L_T1); this.exitStack(); }
        } else {
            if (rbit) regs.push(14);
            this.gR(13); this.addc(-4 * regs.length); this.lset(L_T0);
            regs.forEach((r, k) => {
                this.lget(L_T0); this.gR(r); this.c.u8(OP.i32_store).uleb(0).uleb((this.gbase + 4 * k) >>> 0);
            });
            this.lget(L_T0); this.sR(13);
        }
        return;
    }
    case 't_ext': {
        const op = (h >>> 6) & 3, rm = lo(3), rd = lo(0);
        this.gR(rm);
        if (op === 0) this.op(OP.i32_extend16_s);
        else if (op === 1) this.op(OP.i32_extend8_s);
        else if (op === 2) { this.i32c(0xFFFF); this.and(); }
        else { this.i32c(0xFF); this.and(); }
        this.sR(rd);
        return;
    }
    case 't_ldm': {
        const l = (h >>> 11) & 1, rn = lo(8);
        const regs = [];
        for (let i = 0; i < 8; i++) if ((h >>> i) & 1) regs.push(i);
        this.gR(rn); this.lset(L_T0);
        if (l) {
            regs.forEach((r, k) => { this.lget(L_T0); this.c.u8(OP.i32_load).uleb(0).uleb((this.gbase + 4 * k) >>> 0); this.sR(r); });
            if (!regs.includes(rn)) { this.lget(L_T0); this.addc(4 * regs.length); this.sR(rn); }
        } else {
            regs.forEach((r, k) => { this.lget(L_T0); this.gR(r); this.c.u8(OP.i32_store).uleb(0).uleb((this.gbase + 4 * k) >>> 0); });
            this.lget(L_T0); this.addc(4 * regs.length); this.sR(rn);
        }
        return;
    }
    case 't_bcond': case 't_b': this.jump(ins, ins.t); return;
    case 't_bl': this.i32c((ins.addr + 4) | 1); this.sR(14); this.jump(ins, ins.t); return;
    case 'svc': {
        this.spill();
        this.lget(0); this.i32c(h & 0xFF); this.i32c((ins.addr + 2) | 1); this.c.u8(OP.call).uleb(1);
        this.reload();
        return;
    }
    default: throw new Error('no thumb lowering for ' + ins.kind);
    }
};
