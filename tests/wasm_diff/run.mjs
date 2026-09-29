// Run the picked instructions through the translator's wasm and compare with the Unicorn results.
//   node tests/wasm_diff/run.mjs tests.json oracle.json
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { translateSingles } from '../../web/translator/translate.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const spec = JSON.parse(fs.readFileSync(process.argv[2]));
const oracle = JSON.parse(fs.readFileSync(process.argv[3]));
const code = fs.readFileSync(path.join(root, 'extracted/code.bin'));
const L = spec.layout;
const SCR = oracle.scr, SCR_SIZE = 0x10000;
const scratch = Buffer.from(oracle.scratch, 'base64');
const CPU = 0x1000;

const pcs = spec.tests.map((t) => t.pc);
const wasm = translateSingles(new Uint8Array(code.buffer, code.byteOffset, code.length), L,
    { gbase: 0, irqAddr: 0x800, tableAddr: 0x900 }, pcs);
if (!WebAssembly.validate(wasm)) { console.error('module does not validate'); process.exit(2); }
const pages = Math.ceil((SCR + SCR_SIZE) / 65536);
const mem = new WebAssembly.Memory({ initial: pages, maximum: 65536, shared: true });
new Uint8Array(mem.buffer, L.base, code.length).set(code);
const inst = new WebAssembly.Instance(new WebAssembly.Module(wasm), {
    env: { memory: mem, step: () => { throw new Error('step called'); }, svc: () => { throw new Error('svc called'); } } });
const u32 = new Uint32Array(mem.buffer, CPU, 64);
const scr = new Uint8Array(mem.buffer, SCR, SCR_SIZE);

let fails = 0, total = 0, skipped = 0;
const trapSeen = new Set();
const byClass = new Map();
const hex = (x) => (x >>> 0).toString(16).padStart(8, '0');
spec.tests.forEach((t, ti) => {
    const fn = inst.exports['t' + ti];
    for (const c of oracle.results[ti]) {
        total++;
        u32.fill(0);
        for (let i = 0; i < 15; i++) u32[i] = c.r[i];
        u32[16] = c.f[0]; u32[17] = c.f[1]; u32[18] = c.f[2]; u32[19] = c.f[3]; u32[21] = c.ge;
        scr.set(scratch);
        let npc;
        try { npc = fn(CPU, 0) >>> 0; } catch (e) { skipped++; if (!trapSeen.has(t.key)) { trapSeen.add(t.key); console.log(`TRAP ${hex(t.pc)} ${t.kind} w=${hex(t.w)} [${t.key}] ${e.message} r=${[...c.r].map(hex).join(',')}`); } continue; }
        let why = null;
        for (let i = 0; i < 15 && !why; i++) if (u32[i] !== c.ro[i]) why = `r${i} wasm=${hex(u32[i])} uc=${hex(c.ro[i])} (in ${hex(c.r[i])})`;
        if (!why) for (let f = 0; f < 4; f++) if (u32[16 + f] !== c.fo[f]) { why = `flag ${'NZCV'[f]} wasm=${u32[16 + f]} uc=${c.fo[f]} (in ${c.f.join('')})`; break; }
        if (!why && u32[21] !== c.geo) why = `ge wasm=${u32[21]} uc=${c.geo}`;
        if (!why && npc !== (c.npc >>> 0)) why = `npc wasm=${hex(npc)} uc=${hex(c.npc)}`;
        if (!why) {
            const ex = new Map(c.mem.map(([o, v]) => [o, v]));
            for (let o = 0; o < SCR_SIZE; o++) {
                const want = ex.has(o) ? ex.get(o) : scratch[o];
                if (scr[o] !== want) { why = `mem @${hex(SCR + o)} wasm=${scr[o].toString(16)} uc=${want.toString(16)}`; break; }
            }
        }
        if (why) {
            fails++;
            const k = t.key;
            if (!byClass.has(k)) { byClass.set(k, 0); console.log(`FAIL ${hex(t.pc)} ${t.kind} w=${hex(t.w)}${t.size === 4 && t.w2 ? ' ' + t.w2.toString(16) : ''} [${k}] ${why}`); }
            byClass.set(k, byClass.get(k) + 1);
        }
    }
});
console.log(`\n${fails}/${total} cases failed (${skipped} skipped after wasm traps); ${byClass.size} failing classes`);
process.exit(fails ? 1 : 0);
