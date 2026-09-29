// Code discovery and chunking, ported from lift/lift.py.
//
// Recursive descent from the entry point, every function pointer found in
// .rodata/.data and literal pools, ARM jump tables, and extra seeds. The result
// is compact (bitmaps over the .text halfwords instead of an object per
// instruction) so 2M instructions stay cheap in a browser worker.

import { decodeArm, decodeThumb, ror } from './arm.js';

// per-halfword flags
export const F_ARM = 1, F_THUMB = 2, F_ENT_ARM = 4, F_ENT_THUMB = 8, F_FN_ARM = 16, F_FN_THUMB = 32, F_T32 = 64;

export const MAX_CHUNK = 600;
export const SOFT_CHUNK = 120;

export class Image {
    // code: Uint8Array of the guest image starting at `base`; text is [base, textEnd)
    constructor(code, layout) {
        this.code = code;
        this.dv = new DataView(code.buffer, code.byteOffset, code.byteLength);
        this.base = layout.base >>> 0;
        this.textEnd = layout.textEnd >>> 0;
        this.dataEnd = layout.dataEnd >>> 0;
        this.roEnd = layout.roEnd >>> 0;
    }
    w32(a) { return this.dv.getUint32(a - this.base, true); }
    w16(a) { return this.dv.getUint16(a - this.base, true); }
    decode(pc) {
        const a = pc & ~1;
        if (a < this.base || a >= this.textEnd) return null;
        if (pc & 1) {
            const h = this.w16(a);
            const h2 = a + 2 < this.textEnd ? this.w16(a + 2) : null;
            return decodeThumb(a >>> 0, h, h2);
        }
        if (a & 3) return null;
        return decodeArm(a >>> 0, this.w32(a));
    }
}

export function discover(img, extraSeeds = [], log = () => {}) {
    const base = img.base, textEnd = img.textEnd, dataEnd = img.dataEnd;
    const nHalf = ((textEnd - base) >>> 1) + 2;
    const flags = new Uint8Array(nHalf);
    const idx = (a) => (a - base) >>> 1;
    const seen = (pc) => (flags[idx(pc & ~1)] & ((pc & 1) ? F_THUMB : F_ARM)) !== 0;
    const addEntry = (pc) => { flags[idx(pc & ~1)] |= (pc & 1) ? F_ENT_THUMB : F_ENT_ARM; };
    const addFn = (pc) => { flags[idx(pc & ~1)] |= (pc & 1) ? F_FN_THUMB : F_FN_ARM; };

    function validSeed(pc, n = 3) {
        let cur = pc;
        for (let k = 0; k < n; k++) {
            const i = img.decode(cur);
            if (i === null || i.bad) return false;
            if (!i.falls) return true;
            cur = ((cur & 1) | (((cur & ~1) + i.size) >>> 0)) >>> 0;
        }
        return true;
    }

    // --- seeds
    const seeds = [base];
    let nptr = 0;
    for (let a = textEnd; a < dataEnd - 3; a += 4) {
        const v = img.w32(a);
        if (v >= base && v < textEnd && ((v & 1) || !(v & 3)) && validSeed(v)) { seeds.push(v); nptr++; }
    }
    log(`pointer seeds from data: ${nptr}`);
    let nextra = 0;
    for (const v of extraSeeds) if (v >= base && v < textEnd && validSeed(v, 1)) { seeds.push(v); nextra++; }
    log(`extra seeds: ${nextra}`);
    for (const s of seeds) { addEntry(s); addFn(s); }

    // --- recursive descent
    const work = seeds.slice();
    let ndec = 0, nbad = 0, nthumb = 0;

    function handleTable(pc, i) {
        let bound = -1;
        for (const back of [4, 8, 12]) {
            const p = pc - back;
            if (p < base) break;
            const w = img.w32(p);
            if ((w & 0x0FF00000) === 0x03500000 && ((w >>> 16) & 0xF) === i.tableReg) {
                bound = ror(w & 0xFF, ((w >>> 8) & 0xF) * 2);
                break;
            }
        }
        if (bound < 0 || bound > 512) return;
        if (i.table === 'addpc') {
            const b = ((pc & ~1) + 8) >>> 0;
            for (let k = 0; k <= bound; k++) { const t = b + 4 * k; addEntry(t); work.push(t); }
            const t = b - 4;   // default case: the instruction right after the add
            addEntry(t); work.push(t);
        } else {           // ldr pc, [pc, rX, lsl #2]: table at pc + 8
            const b = ((pc & ~1) + 8) >>> 0;
            for (let k = 0; k <= bound; k++) {
                const v = img.w32(b + 4 * k);
                if (v >= base && v < textEnd) { addEntry(v); work.push(v); }
            }
        }
    }

    while (work.length) {
        const pc = work.pop();
        if (seen(pc)) continue;
        let cur = pc;
        for (;;) {
            if (seen(cur)) break;
            const i = img.decode(cur);
            if (i === null) break;
            const fi = idx(cur & ~1);
            flags[fi] |= (cur & 1) ? F_THUMB : F_ARM;
            if (i.size === 4 && (cur & 1)) flags[fi] |= F_T32;
            ndec++;
            if (cur & 1) nthumb++;
            if (i.t >= 0) {
                addEntry(i.t);
                if (!seen(i.t)) work.push(i.t);
                if (i.call) addFn(i.t);
            }
            const nxt = ((cur & 1) | (((cur & ~1) + i.size) >>> 0)) >>> 0;
            if (i.call) addEntry(nxt);
            if (i.ptr >= 0 && i.ptr >= base && i.ptr < dataEnd - 3) {
                const v = img.w32(i.ptr);
                if (v >= base && v < textEnd && ((v & 1) || !(v & 3)) && !seen(v) && validSeed(v)) {
                    addEntry(v); addFn(v); work.push(v);
                }
            }
            if (i.table) handleTable(cur, i);
            if (i.bad) { nbad++; break; }
            if (!i.falls) break;
            cur = nxt;
        }
    }
    log(`decoded ${ndec} instructions (${ndec - nthumb} ARM, ${nthumb} Thumb), ${nbad} bad stops`);
    return { flags, ndec, nbad };
}

// Split the decoded instructions into chunks: maximal runs of contiguous
// instructions of one ISA, cut at MAX_CHUNK, or at SOFT_CHUNK when the next
// instruction is a known function start.
export function makeChunks(img, flags) {
    const base = img.base;
    const n = flags.length;
    const pcs = [];
    const starts = [];
    let cur = 0, prevEnd = -1, prevMode = -1;
    for (let i = 0; i < n; i++) {
        const f = flags[i];
        if (!(f & (F_ARM | F_THUMB))) continue;
        const a = base + i * 2;
        for (let mode = 0; mode < 2; mode++) {
            if (!(f & (mode ? F_THUMB : F_ARM))) continue;
            const size = mode ? ((f & F_T32) ? 4 : 2) : 4;
            const isFn = (f & (mode ? F_FN_THUMB : F_FN_ARM)) !== 0;
            const contiguous = prevEnd === a && prevMode === mode;
            const startNew = !contiguous || cur >= MAX_CHUNK || (cur >= SOFT_CHUNK && isFn);
            if (startNew) { starts.push(pcs.length); cur = 0; }
            pcs.push(a | mode);
            cur++;
            prevEnd = a + size; prevMode = mode;
        }
    }
    starts.push(pcs.length);
    return { pcs: Uint32Array.from(pcs), starts: Uint32Array.from(starts), nchunks: starts.length - 1 };
}
