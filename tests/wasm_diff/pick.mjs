// Pick a spread of real instructions from the ROM's discovered code for the differential test.
//   node tests/wasm_diff/pick.mjs [per_class=6] > tests.json
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Image, discover, F_ARM, F_THUMB, F_T32 } from '../../web/translator/discover.js';
import { armIsNative } from '../../web/translator/lower_arm.js';
import { thumbIsNative } from '../../web/translator/lower_thumb.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const per = +(process.argv[2] || 6);
const m = JSON.parse(fs.readFileSync(path.join(root, 'extracted/manifest.json')));
const buf = fs.readFileSync(path.join(root, 'extracted/code.bin'));
const t = m.text, r = m.rodata, d = m.data;
const textEnd = r.addr || t.addr + t.pages * 0x1000;
const roEnd = d.addr || r.addr + r.pages * 0x1000;
const layout = { base: t.addr, textEnd, roEnd, dataEnd: (d.addr || roEnd) + d.size };
const img = new Image(new Uint8Array(buf.buffer, buf.byteOffset, buf.length), layout);
const { flags } = discover(img, []);

const NATIVE_A = new Set(['dp', 'mul', 'mull', 'ls', 'lsx', 'ldm', 'b', 'blx_imm', 'bx', 'blx_reg', 'clz']);
const NATIVE_T = new Set(['t_shift', 't_addsub', 't_imm', 't_alu', 't_hi', 't_ldrlit', 't_lsreg', 't_lsimm', 't_lsh', 't_sp', 't_addpc', 't_spadj', 't_pushpop', 't_ext', 't_ldm', 't_bcond', 't_b', 't_bl']);
const groups = new Map();
let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed; };
for (let i = 0; i < flags.length; i++) {
    const f = flags[i];
    for (const mode of [0, 1]) {
        if (!(f & (mode ? F_THUMB : F_ARM))) continue;
        const pc = ((layout.base + i * 2) | mode) >>> 0;
        const ins = img.decode(pc);
        if (!ins || ins.bad) continue;
        if (!(mode ? NATIVE_T : NATIVE_A).has(ins.kind)) continue;
        if (!(mode ? thumbIsNative(ins) : armIsNative(ins))) continue;
        if (!mode && ins.kind === 'dp' && !((ins.w >>> 25) & 1) && ((ins.w >>> 4) & 1)) {
            // register-specified shift reading pc: the reference (and real hardware) gives pc + 12, Unicorn pc + 8
            const w = ins.w;
            if (((w >>> 16) & 0xF) === 15 || (w & 0xF) === 15 || ((w >>> 8) & 0xF) === 15) continue;
        }
        const w = ins.w;
        const key = mode
            ? `T:${ins.kind}:${(w >>> 8).toString(16)}`
            : `A:${ins.kind}:${((w >>> 20) & 0xFF).toString(16)}:${ins.cond === 14 ? 'al' : 'c'}:${(w >>> 4) & 0xF}`;
        let g = groups.get(key);
        if (!g) groups.set(key, g = { n: 0, items: [] });
        g.n++;
        // reservoir sampling
        if (g.items.length < per) g.items.push(pc);
        else { const j = rnd() % g.n; if (j < per) g.items[j] = pc; }
    }
}
const tests = [];
for (const [key, g] of [...groups].sort()) for (const pc of g.items) {
    const ins = img.decode(pc);
    tests.push({ pc, key, kind: ins.kind, w: ins.w, w2: ins.w2, size: ins.size });
}
console.error(`${tests.length} tests in ${groups.size} classes`);
console.log(JSON.stringify({ layout, tests }));
