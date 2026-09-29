// ARMv6K ARM / Thumb-1 instruction classification for the translator.
//
// Mirrors lift/armdec.py exactly where it matters for code discovery (size,
// condition, static target, fall-through, call, literal pointer loads, jump
// tables, "bad" = unsupported): a different answer here would discover
// different code than the reference lifter. The semantics are not kept as text
// any more; `kind` names how the emitter (emit.js) lowers the instruction and
// everything without a native lowering is `step` (one instruction executed by
// the runtime interpreter, the same oracle the lifter's FALLBACK used).

export function mkInsn(addr, thumb, size, w) {
    return { addr, thumb, size, cond: 14, kind: 'step', t: -1, falls: true, call: false, bad: null,
             table: null, tableReg: 0, ptr: -1, w, w2: 0 };
}

function sx(v, bits) { const m = 1 << (bits - 1); return (v & (m - 1)) - (v & m); }
export function ror(v, s) { s &= 31; return s ? ((v >>> s) | (v << (32 - s))) >>> 0 : v >>> 0; }

class Unsupported extends Error {}
function unsupported(msg) { throw new Unsupported(msg); }

// ------------------------------------------------------------------- ARM
export function decodeArm(addr, w) {
    const ins = mkInsn(addr, false, 4, w);
    const cond = w >>> 28;
    const pc = (addr + 8) >>> 0;
    if (cond === 15) return decArmUncond(ins, w);
    ins.cond = cond;
    const op = (w >>> 25) & 7;
    try {
        if (op === 0) {
            if ((w & 0x90) === 0x90) {
                if (((w >>> 5) & 3) === 0) decArmMulSwp(ins, w);
                else decArmExtraLs(ins, w, pc);
            } else if (((w >>> 23) & 3) === 2 && !((w >>> 20) & 1)) decArmMisc(ins, w);
            else decArmDp(ins, w);
        } else if (op === 1) {
            if (((w >>> 23) & 3) === 2 && !((w >>> 20) & 1)) {
                if ((w >>> 21) & 1) {
                    if (((w >>> 16) & 0xF) === 0) ins.kind = 'nop';   // nop / yield / wfe / wfi / sev
                    else ins.kind = 'step';                            // MSR imm
                } else unsupported('movw/movt');
            } else decArmDp(ins, w);
        } else if (op === 2 || (op === 3 && !((w >>> 4) & 1))) decArmLs(ins, w, pc);
        else if (op === 3) decArmMedia(ins, w);
        else if (op === 4) decArmLdm(ins, w);
        else if (op === 5) {
            const off = sx(w & 0xFFFFFF, 24) << 2;
            const t = (pc + off) >>> 0;
            ins.t = t;
            ins.kind = 'b';
            if ((w >>> 24) & 1) ins.call = true;
            else if (cond === 14) ins.falls = false;
        } else if (op === 6) decArmLdc(ins, w, pc);
        else {
            if ((w >>> 24) & 1) ins.kind = 'svc';
            else decArmCp(ins, w);
        }
    } catch (e) {
        if (!(e instanceof Unsupported)) throw e;
        ins.bad = e.message || 'unimpl';
        ins.kind = 'step';
    }
    return ins;
}

// lift/armdec.py rejects any lowered body that names the variable r15 (register
// slots it does not special-case: unpredictable encodings, usually data decoded
// as code). r15(...) reproduces that per slot so both decoders stop discovery at
// the same places.
function r15(...regs) { for (const r of regs) if (r === 15) unsupported('r15 operand'); }

function decArmUncond(ins, w) {
    ins.cond = 14;
    if ((w & 0x0E000000) === 0x0A000000) {   // BLX imm
        const off = (sx(w & 0xFFFFFF, 24) << 2) | (((w >>> 24) & 1) << 1);
        const t = (ins.addr + 8 + off) >>> 0;
        ins.call = true;
        ins.kind = 'blx_imm';
        ins.t = (t | 1) >>> 0;
        return ins;
    }
    if ((w & 0x0D70F000) === 0x0550F000) { ins.kind = 'nop'; return ins; }   // PLD
    if (w === 0xF57FF01F) { ins.kind = 'step'; return ins; }                  // CLREX
    if (((w & 0xFFFFFFF0) >>> 0) === 0xF57FF040 || ((w & 0xFFFFFFF0) >>> 0) === 0xF57FF050 ||
        ((w & 0xFFFFFFF0) >>> 0) === 0xF57FF060) { ins.kind = 'nop'; return ins; }   // DSB/DMB/ISB
    if (((w & 0xFFF1FE20) >>> 0) === 0xF1000000) { ins.kind = 'nop'; return ins; }   // CPS
    ins.bad = 'uncond ' + w.toString(16);
    return ins;
}

function decArmDp(ins, w) {
    const opc = (w >>> 21) & 0xF, s = (w >>> 20) & 1, rn = (w >>> 16) & 0xF, rd = (w >>> 12) & 0xF;
    if (s && rd === 15 && (opc < 8 || opc > 11)) unsupported('exception return');
    ins.kind = 'dp';
    if (rd === 15 && (opc < 8 || opc > 11)) {
        // computed jump: add pc, pc, rX, lsl #2 jump tables
        if (opc === 4 && rn === 15 && !((w >>> 25) & 1) && !((w >>> 4) & 1) && ((w >>> 5) & 3) === 0 && ((w >>> 7) & 0x1F) === 2) {
            ins.table = 'addpc'; ins.tableReg = w & 0xF;
        }
        if (ins.cond === 14) ins.falls = false;
    }
    // r15 among the operand registers is fine for ARM data processing (reads pc + 8/12);
    // armdec.py only rejects the r15 spelled out by write_reg/reg_read misses, which
    // cannot occur here.
}

function decArmMisc(ins, w) {
    const op = (w >>> 21) & 3, op2 = (w >>> 4) & 0xF;
    const rd = (w >>> 12) & 0xF, rm = w & 0xF;
    if (op2 === 0) {
        if (op & 1) { if ((w >>> 22) & 1) unsupported('msr spsr'); r15(rm); }
        else { if ((w >>> 22) & 1) unsupported('mrs spsr'); r15(rd); }
        ins.kind = 'step';
        return;
    }
    if ((op2 === 1 && op === 1) || (op2 === 2 && op === 1)) {   // BX / BXJ
        ins.kind = 'bx';
        if (ins.cond === 14) ins.falls = false;
        return;
    }
    if (op2 === 1 && op === 3) { r15(rd, rm); ins.kind = 'clz'; return; }
    if (op2 === 3 && op === 1) { ins.kind = 'blx_reg'; ins.call = true; return; }
    if (op2 === 5) { r15(rd, rm, (w >>> 16) & 0xF); ins.kind = 'step'; return; }   // QADD/QSUB/QDADD/QDSUB
    if (op2 === 7 && op === 1) { ins.kind = 'step'; return; }    // BKPT
    if ((op2 & 9) === 8) {                                       // signed halfword multiplies
        const rdh = (w >>> 16) & 0xF, rn = (w >>> 12) & 0xF, rs = (w >>> 8) & 0xF, x = (w >>> 5) & 1;
        if (op === 0) r15(rdh, rm, rs, rn);
        else if (op === 1) { r15(rdh, rm, rs); if (!x) r15(rn); }
        else if (op === 2) r15(rdh, rn, rm, rs);
        else r15(rdh, rm, rs);
        ins.kind = 'step';
        return;
    }
    unsupported('misc');
}

function decArmMulSwp(ins, w) {
    const rn = (w >>> 16) & 0xF, rt = (w >>> 12) & 0xF, rt2 = w & 0xF;
    if ((w >>> 24) & 1) {   // swap / exclusives
        ins.kind = 'step';
        if ((w >>> 23) & 1) {
            const kind = (w >>> 21) & 3, load = (w >>> 20) & 1;
            if (load) { r15(rn, rt); if (kind === 1) r15(rt + 1); }
            else { r15(rn, rt2, rt); if (kind === 1) r15(rt2 + 1); }
        } else r15(rn, rt2, rt);
        return;
    }
    const op = (w >>> 21) & 7;
    const rd = (w >>> 16) & 0xF, rn2 = (w >>> 12) & 0xF, rs = (w >>> 8) & 0xF, rm = w & 0xF;
    if (op === 0) r15(rd, rm, rs);
    else if (op === 1) r15(rd, rm, rs, rn2);
    else r15(rd, rn2, rm, rs);
    ins.kind = (op === 0 || op === 1) ? 'mul' : (op >= 4 ? 'mull' : 'step');
}

function decArmExtraLs(ins, w, pc) {
    const p = (w >>> 24) & 1, u = (w >>> 23) & 1, i = (w >>> 22) & 1, l = (w >>> 20) & 1;
    const rn = (w >>> 16) & 0xF, rt = (w >>> 12) & 0xF, sh = (w >>> 5) & 3;
    const wbk = (w >>> 21) & 1;
    if (rn === 15 && i && p) ins.ptr = (pc + (u ? 1 : -1) * (((w >>> 4) & 0xF0) | (w & 0xF))) >>> 0;
    if (!i) r15(w & 0xF);                       // register offset is spelled R(rm)
    if (!p || wbk) r15(rn);                     // writeback target is spelled R(rn)
    if (l) {
        if (rt === 15) unsupported('ldrh pc');
        ins.kind = 'lsx';
    } else if (sh === 1) ins.kind = 'lsx';
    else {
        if ((rt & 1) || rt === 14) unsupported(sh === 2 ? 'ldrd odd' : 'strd odd');
        ins.kind = 'step';   // LDRD / STRD
    }
}

function decArmLs(ins, w, pc) {
    const reg = (w >>> 25) & 1, p = (w >>> 24) & 1, u = (w >>> 23) & 1, b = (w >>> 22) & 1, l = (w >>> 20) & 1;
    const rn = (w >>> 16) & 0xF, rt = (w >>> 12) & 0xF;
    if (!reg && rn === 15 && p) ins.ptr = (pc + (w & 0xFFF) * (u ? 1 : -1)) >>> 0;
    ins.kind = 'ls';
    if (!p || ((w >>> 21) & 1)) r15(rn);        // writeback target is spelled R(rn)
    if (l && rt === 15) { /* pc load handled below */ }
    if (l && rt === 15) {
        if (b) unsupported('ldrb pc');
        if (ins.cond === 14) ins.falls = false;
        if (reg && rn === 15 && p && ((w >>> 5) & 3) === 0 && ((w >>> 7) & 0x1F) === 2) { ins.table = 'ldrpc'; ins.tableReg = w & 0xF; }
    }
}

function decArmLdm(ins, w) {
    const l = (w >>> 20) & 1, s = (w >>> 22) & 1;
    if ((w & 0xFFFF) === 0) unsupported('ldm empty');
    if (s && !(l && ((w >>> 15) & 1))) unsupported('ldm user regs');
    if (s) unsupported('ldm exc return');
    r15((w >>> 16) & 0xF);
    ins.kind = 'ldm';
    if (l && ((w >>> 15) & 1) && ins.cond === 14) ins.falls = false;
}

function decArmMedia(ins, w) {
    const op1 = (w >>> 20) & 0x1F, op2 = (w >>> 5) & 7;
    const rd = (w >>> 12) & 0xF, rn = (w >>> 16) & 0xF, rm = w & 0xF, rs = (w >>> 8) & 0xF;
    ins.kind = 'step';
    const hi = op1 >>> 3;
    if (hi === 0) {
        const kind = op1 & 7;
        if (![1, 2, 3, 5, 6, 7].includes(kind) || ![0, 1, 2, 3, 4, 7].includes(op2)) unsupported('parallel');
        r15(rd, rn, rm);
        return;
    }
    if (hi === 1) {
        const o = op1 & 7;
        if (o === 0 && !(op2 & 1)) { r15(rd, rn, rm); return; }   // PKH
        if (o === 0 && op2 === 3) { r15(rd, rm); return; }        // SXTAB16 / SXTB16
        if (o === 0 && op2 === 5) { r15(rd, rn, rm); return; }    // SEL
        if ((o >>> 1) === 1 && !(op2 & 1)) { r15(rd, rm); return; }   // SSAT
        if (o === 2 && op2 === 1) { r15(rd, rm); return; }        // SSAT16
        if ((o >>> 1) === 3 && !(op2 & 1)) { r15(rd, rm); return; }   // USAT
        if (o === 6 && op2 === 1) { r15(rd, rm); return; }        // USAT16
        if (op2 === 3) {
            if (o === 4) { r15(rd, rm); return; }                 // UXTAB16 / UXTB16
            if (o === 2 || o === 3 || o === 6 || o === 7) { r15(rd, rm); return; }   // SXTB/SXTH/UXTB/UXTH (+accumulate)
        }
        if (o === 3 && op2 === 1) { r15(rd, rm); return; }        // REV
        if (o === 3 && op2 === 5) { r15(rd, rm); return; }        // REV16
        if (o === 7 && op2 === 5) { r15(rd, rm); return; }        // REVSH
        unsupported('media pack');
    }
    if (hi === 2) {
        const o = op1 & 7;
        if (o === 0) { if ((op2 >>> 2) & 1) unsupported('smlad op2'); r15(rn, rm, rs); return; }
        if (o === 4) { r15(rn, rd, rm, rs); return; }
        if (o === 5) {
            if ((op2 >>> 1) === 0) { r15(rn, rm, rs); return; }
            if ((op2 >>> 1) === 3) { r15(rn, rd, rm, rs); return; }
            unsupported('smm');
        }
        unsupported('smul');
    }
    if (op1 === 0x18 && op2 === 0) { r15(rn, rm, rs); return; }   // USAD8 / USADA8
    unsupported('media');
}

function decArmLdc(ins, w, pc) {
    const cp = (w >>> 8) & 0xF;
    if (cp !== 10 && cp !== 11) unsupported('ldc cp' + cp);
    ins.kind = 'step';
    if (((w >>> 21) & 0x7F) === 0x62) { r15((w >>> 12) & 0xF, (w >>> 16) & 0xF); return; }   // VMOV two core regs
    const p = (w >>> 24) & 1, u = (w >>> 23) & 1, wb = (w >>> 21) & 1;
    const rn = (w >>> 16) & 0xF, imm8 = w & 0xFF;
    if (p && !wb) {                                             // VLDR / VSTR
        if (rn === 15) ins.ptr = (((pc & ~3) >>> 0) + (u ? imm8 * 4 : -imm8 * 4)) >>> 0;
        ins.kind = 'vfp_ls';
        return;
    }
    const n = cp === 11 ? imm8 >>> 1 : imm8;
    if (n === 0) unsupported('vldm 0');
    if (!((p === 0 && u === 1) || (p === 1 && u === 0))) unsupported('vldm mode');
    r15(rn);
}

function decArmCp(ins, w) {
    const cp = (w >>> 8) & 0xF;
    ins.kind = 'step';
    if ((w >>> 4) & 1) {   // register transfer
        const l = (w >>> 20) & 1, opc1 = (w >>> 21) & 7, crn = (w >>> 16) & 0xF, crm = w & 0xF, opc2 = (w >>> 5) & 7;
        const rt = (w >>> 12) & 0xF;
        if (cp === 15) {
            if (l && crn === 13 && crm === 0 && opc1 === 0 && (opc2 === 3 || opc2 === 2)) { r15(rt); return; }
            if (!l && crn === 13 && crm === 0 && opc2 === 2) { r15(rt); return; }
            if (!l && crn === 7) { ins.kind = 'nop'; return; }
            unsupported('cp15');
        }
        if (cp === 10) {
            if (opc1 === 0) { r15(rt); return; }
            if (opc1 === 7) {
                if (l) {
                    if (crn === 1) { if (rt !== 15) r15(rt); }
                    else if (crn === 8 || crn === 0) r15(rt);
                    else unsupported('vmrs');
                } else if (crn === 1) r15(rt);
                return;
            }
        }
        if (cp === 11 && (opc1 & 6) === 0) { r15(rt); return; }
        unsupported('cp reg transfer');
    }
    if (cp !== 10 && cp !== 11) unsupported('cdp cp' + cp);
    const p = (w >>> 23) & 1, q = (w >>> 21) & 1, r = (w >>> 20) & 1, s = (w >>> 6) & 1;
    const opc = (p << 3) | (q << 2) | (r << 1) | s;
    if (opc <= 8) return;
    if ((opc & 0xE) === 0xE) {
        const vn = (w >>> 16) & 0xF, o3 = (w >>> 6) & 3;
        if ((vn === 0 || vn === 1) && (o3 === 1 || o3 === 3)) return;
        if (vn === 4 || vn === 5) return;
        if (vn === 7 && o3 === 3) return;
        if (vn === 8 || vn === 12 || vn === 13) return;
        unsupported('vfp ext');
    }
    unsupported('vfp dp');
}

// ------------------------------------------------------------------ Thumb
export function decodeThumb(addr, h, h2) {
    const ins = mkInsn(addr, true, 2, h);
    try { decThumb(ins, addr, h, h2); } catch (e) {
        if (!(e instanceof Unsupported)) throw e;
        ins.bad = e.message || 'unimpl';
        ins.kind = 'step';
    }
    return ins;
}

function decThumb(ins, addr, h, h2) {
    const pc = (addr + 4) >>> 0;
    const top5 = h >>> 11;
    if (top5 < 3) { ins.kind = 't_shift'; return; }
    if (top5 === 3) { ins.kind = 't_addsub'; return; }
    if (top5 < 8) { ins.kind = 't_imm'; return; }
    if ((h >>> 10) === 0x10) { ins.kind = 't_alu'; return; }
    if ((h >>> 10) === 0x11) {
        const op = (h >>> 8) & 3, rd = ((h >>> 4) & 8) | (h & 7);
        ins.kind = 't_hi';
        if (op === 0) { if (rd === 15) ins.falls = false; }
        else if (op === 2) { if (rd === 15) ins.falls = false; }
        else if (op === 3) { if ((h >>> 7) & 1) ins.call = true; else ins.falls = false; }
        return;
    }
    if (top5 === 9) { ins.kind = 't_ldrlit'; ins.ptr = (((pc & ~3) >>> 0) + (h & 0xFF) * 4) >>> 0; return; }
    if ((h >>> 12) === 5) { ins.kind = 't_lsreg'; return; }
    if ((h >>> 13) === 3) { ins.kind = 't_lsimm'; return; }
    if ((h >>> 12) === 8) { ins.kind = 't_lsh'; return; }
    if ((h >>> 12) === 9) { ins.kind = 't_sp'; return; }
    if ((h >>> 12) === 10) { ins.kind = 't_addpc'; return; }
    if ((h >>> 12) === 11) {
        if ((h >>> 8) === 0xB0) { ins.kind = 't_spadj'; return; }
        if ((h & 0x0600) === 0x0400) {
            ins.kind = 't_pushpop';
            if (((h >>> 11) & 1) && ((h >>> 8) & 1)) ins.falls = false;
            return;
        }
        if ((h >>> 8) === 0xB2) { ins.kind = 't_ext'; return; }
        if ((h >>> 8) === 0xBA) {
            if (((h >>> 6) & 3) === 2) unsupported('rev?');
            ins.kind = 't_rev';
            return;
        }
        if ((h >>> 8) === 0xBE) { ins.kind = 'step'; return; }
        if ((h >>> 8) === 0xBF || (h & 0xFFE8) === 0xB660 || (h & 0xFFF7) === 0xB650) { ins.kind = 'nop'; return; }
        unsupported('thumb misc');
    }
    if ((h >>> 12) === 12) {
        if (((h & 0xFF)) === 0) unsupported('ldm empty');
        ins.kind = 't_ldm';
        return;
    }
    if ((h >>> 12) === 13) {
        const cond = (h >>> 8) & 0xF;
        if (cond === 15) { ins.kind = 'svc'; return; }
        if (cond === 14) { ins.kind = 'step'; return; }   // udf
        ins.cond = cond;
        ins.kind = 't_bcond';
        ins.t = ((pc + (sx(h & 0xFF, 8) << 1)) | 1) >>> 0;
        return;
    }
    if (top5 === 0x1C) {
        ins.kind = 't_b';
        ins.t = ((pc + (sx(h & 0x7FF, 11) << 1)) | 1) >>> 0;
        ins.falls = false;
        return;
    }
    if (top5 === 0x1E) {
        if (h2 === undefined || h2 === null || ((h2 >>> 11) !== 0x1F && (h2 >>> 11) !== 0x1D)) unsupported('lone bl prefix');
        ins.size = 4;
        ins.w2 = h2;
        const off = (sx(h & 0x7FF, 11) << 12) | ((h2 & 0x7FF) << 1);
        const base = (pc + off) >>> 0;
        ins.call = true;
        ins.kind = 't_bl';
        ins.t = (h2 >>> 11) === 0x1F ? (base | 1) >>> 0 : (base & ~3) >>> 0;
        return;
    }
    unsupported('thumb');
}
