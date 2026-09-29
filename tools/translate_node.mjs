#!/usr/bin/env node
// Translate extracted/code.bin to WebAssembly on the command line (development tool).
//   node tools/translate_node.mjs [out.wasm]
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { translate } from '../web/translator/translate.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const m = JSON.parse(fs.readFileSync(path.join(root, 'extracted/manifest.json')));
const buf = fs.readFileSync(path.join(root, 'extracted/code.bin'));
const t = m.text, r = m.rodata, d = m.data;
const textEnd = r.addr || t.addr + t.pages * 0x1000;
const roEnd = d.addr || r.addr + r.pages * 0x1000;
const layout = { base: t.addr, textEnd, roEnd, dataEnd: (d.addr || roEnd) + d.size };
const seeds = fs.readFileSync(path.join(root, 'lift/seeds.txt'), 'utf8').split('\n')
    .map((l) => l.split('#')[0].trim().split(/\s+/)[0]).filter(Boolean).map((x) => parseInt(x, 16));
const t0 = Date.now();
const res = translate(new Uint8Array(buf.buffer, buf.byteOffset, buf.length), layout,
    { gbase: 0x1000000, irqAddr: 0x100, tableAddr: 0x2000000, seeds }, (s) => console.error(`[translate ${Date.now() - t0}ms] ${s}`));
console.error('validate:', WebAssembly.validate(res.wasm));
if (process.argv[2]) fs.writeFileSync(process.argv[2], res.wasm);
