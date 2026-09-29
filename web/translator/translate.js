// ARM -> WASM translation of a whole 3DS code image, ahead of boot.
//
//   const r = translate(codeBytes, layout, { gbase, irqAddr, tableAddr }, log);
//   r.wasm   Uint8Array: a module importing {env.memory, env.step, env.svc}, exporting `run`
//   r.table  Uint32Array over the .text halfwords: pc -> (chunk + 1) << 11 | thumb << 10 | label
//
// The module is derived from the caller's own ROM at runtime; nothing about any
// game is embedded in this file or in the published page.

import { ByteBuf, OP, EMPTY, T, assemble } from './wasm.js';
import { Image, discover, makeChunks } from './discover.js';
import { Emitter } from './chunk.js';

export const ENTRY_HOST = 1;   // table value: hand this pc back to the runtime (hooks)

export function translate(code, layout, opts, log = () => {}) {
    const t0 = Date.now();
    const img = new Image(code, layout);
    const { flags } = discover(img, opts.seeds || [], log);
    const { pcs, starts, nchunks } = makeChunks(img, flags);
    log(`${nchunks} chunks (${Date.now() - t0} ms)`);

    const em = new Emitter(img, opts);
    const nHalf = flags.length;
    const table = new Uint32Array(nHalf);
    const funcs = [];
    for (let ci = 0; ci < nchunks; ci++) {
        const { body, entries } = em.emitChunk(pcs, starts[ci], starts[ci + 1], flags);
        // keep a private copy: the emitter reuses its buffers
        const copy = new ByteBuf(body.length); copy.buf(body);
        funcs.push({ type: 0, body: copy });
        for (const [pc, label] of entries) {
            if (label >= 1024) throw new Error('chunk with more than 1023 labels');
            table[((pc & ~1) - img.base) >>> 1] = ((ci + 1) << 11) | ((pc & 1) << 10) | label;
        }
    }
    log(`emitted ${nchunks} functions (${Date.now() - t0} ms)`);

    const nImp = 2;   // step, svc (functions) — memory is not in the function index space
    const runIdx = nImp + nchunks;
    funcs.push({ type: 0, body: dispatcher(img, opts, nHalf) });
    const elem = [nImp];
    for (let ci = 0; ci < nchunks; ci++) elem.push(nImp + ci);
    const wasm = assemble({
        types: [{ params: [T.i32, T.i32], results: [T.i32] }, { params: [T.i32, T.i32, T.i32], results: [] }],
        imports: [
            { module: 'env', name: 'step', kind: 'func', type: 0 },
            { module: 'env', name: 'svc', kind: 'func', type: 1 },
            { module: 'env', name: 'memory', kind: 'memory', min: opts.memMinPages || 1, max: opts.memMaxPages || 65536 },
        ],
        funcs,
        tableSize: nchunks + 1,
        elem,
        exports: [{ name: 'run', kind: 'func', index: runIdx }],
    });
    log(`module ${(wasm.length / 1048576).toFixed(1)} MiB (${Date.now() - t0} ms)`);
    return { wasm, table, nchunks, ndec: pcs.length };
}

// run(cpu, pc) -> pc: dispatch chunks until the runtime is needed (interrupt pending,
// unknown or hooked pc). The table lives in guest-visible linear memory at opts.tableAddr.
function dispatcher(img, opts, nHalf) {
    const c = new ByteBuf(256);
    const textSize = img.textEnd - img.base;
    c.uleb(1).uleb(2).u8(T.i32);           // locals: d (2), e (3)
    c.u8(OP.loop).u8(EMPTY);
    // interrupt pending -> return pc
    c.u8(OP.i32_const).sleb(0).u8(0xFE).u8(0x10).uleb(2).uleb(opts.irqAddr);
    c.u8(OP.if).u8(EMPTY).u8(OP.local_get).uleb(1).u8(OP.return).u8(OP.end);
    // d = (pc & ~1) - base; outside .text -> return pc
    c.u8(OP.local_get).uleb(1).u8(OP.i32_const).sleb(-2).u8(OP.i32_and).u8(OP.i32_const).sleb(img.base | 0).u8(OP.i32_sub).u8(OP.local_tee).uleb(2);
    c.u8(OP.i32_const).sleb(textSize | 0).u8(OP.i32_ge_u);
    c.u8(OP.if).u8(EMPTY).u8(OP.local_get).uleb(1).u8(OP.return).u8(OP.end);
    // e = table[d >> 1]  (4-byte entries: byte offset d * 2)
    c.u8(OP.local_get).uleb(2).u8(OP.i32_const).sleb(1).u8(OP.i32_shl).u8(OP.i32_load).uleb(2).uleb(opts.tableAddr >>> 0).u8(OP.local_tee).uleb(3);
    c.u8(OP.i32_const).sleb(1).u8(OP.i32_le_u);
    c.u8(OP.if).u8(EMPTY).u8(OP.local_get).uleb(1).u8(OP.return).u8(OP.end);
    // ISA of the entry must match the pc's
    c.u8(OP.local_get).uleb(3).u8(OP.i32_const).sleb(10).u8(OP.i32_shr_u).u8(OP.i32_const).sleb(1).u8(OP.i32_and);
    c.u8(OP.local_get).uleb(1).u8(OP.i32_const).sleb(1).u8(OP.i32_and).u8(OP.i32_ne);
    c.u8(OP.if).u8(EMPTY).u8(OP.local_get).uleb(1).u8(OP.return).u8(OP.end);
    // pc = chunk(cpu, label)
    c.u8(OP.local_get).uleb(0);
    c.u8(OP.local_get).uleb(3).u8(OP.i32_const).sleb(1023).u8(OP.i32_and);
    c.u8(OP.local_get).uleb(3).u8(OP.i32_const).sleb(11).u8(OP.i32_shr_u);
    c.u8(OP.call_indirect).uleb(0).uleb(0);
    c.u8(OP.local_set).uleb(1);
    c.u8(OP.br).uleb(0);
    c.u8(OP.end);
    c.u8(OP.unreachable);
    c.u8(OP.end);
    const out = new ByteBuf(c.length + 8); out.buf(c);
    return out;
}

// Test helper: one function per given pc (a chunk of exactly that instruction), exported as
// t0..tN. Used by tests/wasm_diff to check single instructions against Unicorn.
export function translateSingles(code, layout, opts, pcList) {
    const img = new Image(code, layout);
    const em = new Emitter(img, opts);
    const flags = new Uint8Array(((layout.textEnd - layout.base) >>> 1) + 2);
    const funcs = [], exports = [];
    pcList.forEach((pc, i) => {
        const { body } = em.emitChunk(Uint32Array.of(pc), 0, 1, flags);
        const copy = new ByteBuf(body.length); copy.buf(body);
        funcs.push({ type: 0, body: copy });
        exports.push({ name: 't' + i, kind: 'func', index: 2 + i });
    });
    return assemble({
        types: [{ params: [T.i32, T.i32], results: [T.i32] }, { params: [T.i32, T.i32, T.i32], results: [] }],
        imports: [
            { module: 'env', name: 'step', kind: 'func', type: 0 },
            { module: 'env', name: 'svc', kind: 'func', type: 1 },
            { module: 'env', name: 'memory', kind: 'memory', min: opts.memMinPages || 1, max: opts.memMaxPages || 65536 },
        ],
        funcs, tableSize: 1, elem: null, exports,
    });
}
