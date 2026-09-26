#!/usr/bin/env node
// Headless smoke test of the web build: loads the page in Chromium (Playwright),
// streams the ROM from serve_web.py --rom, and drives it with real keyboard /
// mouse events, saving canvas screenshots. A playthrough script for the game
// under test is supplied as JSON via SMOKE_STEPS (see below).
//
//   python3 tools/serve_web.py --rom "game.3ds" &
//   node tools/web_smoke.js http://127.0.0.1:8080/ shots/web
//
// Needs Playwright (npm i -g playwright; `npx playwright install chromium`
// if no Chromium is available). Exit code 0 on success.
'use strict';
const path = require('path');
const fs = require('fs');
let pw;
try { pw = require('playwright'); } catch (e) {
  pw = require(path.join(require('child_process').execSync('npm root -g').toString().trim(), 'playwright'));
}
const base = process.argv[2] || 'http://127.0.0.1:8080/';
const out = process.argv[3] || 'shots/web';
fs.mkdirSync(path.dirname(out), { recursive: true });

(async () => {
  const browser = await pw.chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => { errors.push(e.message); console.log('[pageerror]', e.message); });
  const logf = process.env.SMOKE_LOG;   // optional: full console log to a file
  page.on('console', (m) => { const t = m.text(); if (logf) fs.appendFileSync(logf, t + '\n'); if (/fatal|trap|Abort|error/i.test(t) && !/unhandled cmd/.test(t)) console.log('[page]', t.slice(0, 200)); });
  await page.goto(base + (base.includes('?') ? '&' : '?') + 'rom=rom.3ds&autostart=1&filter=smooth&layout=' + (process.env.SMOKE_LAYOUT || 'classic'));
  const canvas = await page.waitForSelector('#screen', { state: 'visible', timeout: 60000 });
  await page.waitForFunction(() => window.r3ds && r3ds.comp && r3ds.comp.bottomRect, null, { timeout: 60000 });
  let n = 0;
  const shot = async (tag) => {
    const f = `${out}_${String(n++).padStart(2, '0')}_${tag}.png`;
    await canvas.screenshot({ path: f });
    const st = await page.evaluate(() => ({ frames: r3ds.frames, fps: +r3ds.fps.toFixed(1), romMiB: +(r3ds.bytesRead / 1048576).toFixed(1) }));
    console.log(tag.padEnd(14), JSON.stringify(st), f);
  };
  const wait = (ms) => page.waitForTimeout(ms);
  const key = async (code, ms = 120) => { await page.keyboard.down(code); await wait(ms); await page.keyboard.up(code); };
  const tap = async (x, y, ms = 120) => {   // 3DS bottom-screen coordinates, through the current layout
    const r = await canvas.boundingBox();
    const b = await page.evaluate(() => r3ds.comp && r3ds.comp.bottomRect);
    if (!b) return;
    await page.mouse.move(r.x + b.x + (x + 0.5) / 320 * b.w, r.y + b.y + (y + 0.5) / 240 * b.h);
    await page.mouse.down(); await wait(ms); await page.mouse.up();
  };
  // pixel probes on the 400x480 canvas (x, y, w, h) -> count of pixels matching pred
  const probe = (x, y, w, h, pred) => page.evaluate(([x, y, w, h, pred]) => {
    const d = r3ds.getFrame(), f = new Function('r', 'g', 'b', 'return ' + pred);   // raw 400x480 frame
    let n = 0;
    for (let yy = y; yy < y + h; yy++) for (let xx = x; xx < x + w; xx++) { const i = (yy * 400 + xx) * 4; if (f(d[i], d[i + 1], d[i + 2])) n++; }
    return n;
  }, [x, y, w, h, pred]);
  const waitFor = async (what, fn, ms = 30000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { if (await fn()) return true; await wait(250); }
    console.log('timeout waiting for', what); return false;
  };

  // A game's playthrough is data, not code. SMOKE_STEPS=file.json holds an
  // array of steps, each a verb + arguments:
  //   ["wait", ms]                 sleep
  //   ["key", "KeyK" [, ms]]       key hold (KeyboardEvent code, hold ms)
  //   ["tap", x, y [, ms]]         bottom-screen touch at 320x240 coordinates
  //   ["shot", "tag"]              screenshot
  //   ["probe", x,y,w,h,"expr",min,["tag"]] wait until >=min pixels match expr (r/g/b)
  //   ["probeOnce", x,y,w,h,"expr",min,["tag"]] same, no waiting -> fail hard
  // With no script the test boots, checks frames render, and screenshots.
  const steps = process.env.SMOKE_STEPS ? JSON.parse(fs.readFileSync(process.env.SMOKE_STEPS)) : null;
  const doProbe = async (s, once) => {
    const [_, x, y, w, h, expr, min, tag] = s;
    const fn = async () => (await probe(x, y, w, h, expr)) >= min;
    return once ? await fn() : waitFor(tag || `probe(${x},${y})`, fn, 120000);
  };
  let ok = true;
  await wait(20000); await shot('boot');
  if (!steps) {
    ok = await waitFor('first frames', async () => await page.evaluate(() => r3ds.frames) > 30, 120000);
    await shot('frames');
  } else for (const s of steps) {
    const v = s[0];
    if (v === 'wait') await wait(s[1]);
    else if (v === 'key') await key(s[1], s[2]);
    else if (v === 'tap') await tap(s[1], s[2], s[3]);
    else if (v === 'shot') await shot(s[1]);
    else if (v === 'probe') ok = await doProbe(s, false) && ok;
    else if (v === 'probeOnce') { if (!(ok = await doProbe(s, true))) { console.log('probe failed:', JSON.stringify(s)); break; } }
    else { console.log('unknown step', JSON.stringify(s)); break; }
  }
  await browser.close();
  if (errors.length) { console.log('FAIL:', errors.length, 'page errors'); process.exit(1); }
  if (!ok) { console.log('FAIL: playthrough did not complete (timing or game-specific expectations; see screenshots)'); process.exit(2); }
  console.log('OK');
})();
