// 3DS static recompiler — browser frontend.
//
// Loads recomp3ds.js/.wasm (the recompiled game + HLE runtime, built with
// Emscripten), hands it a ROM source for streaming, and does display,
// input, audio and save persistence. URL parameters (handy for testing):
//   ?rom=<url>        stream the ROM from this URL (HTTP Range required)
//   ?romMode=download (default) download the whole ROM once (rom.pack) and play it
//                     from the browser's storage; 'stream' reads on demand
//   &autostart=1      start without waiting for a click (audio starts muted
//                     until the first click, as browsers require)
//   &env=NAME=VALUE   set a runtime environment variable (repeatable), e.g.
//                     env=R3DS_RASTER_THREADS=4
'use strict';
(() => {
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const BTN = { A: 1, B: 2, SELECT: 4, START: 8, RIGHT: 16, LEFT: 32, UP: 64, DOWN: 128, R: 256, L: 512, X: 1024, Y: 2048 };
const KEYMAP = {
  KeyK: BTN.A, KeyJ: BTN.B, KeyI: BTN.X, KeyU: BTN.Y, KeyQ: BTN.L, KeyE: BTN.R,
  Enter: BTN.START, Backspace: BTN.SELECT,
  ArrowUp: BTN.UP, ArrowDown: BTN.DOWN, ArrowLeft: BTN.LEFT, ArrowRight: BTN.RIGHT,
};
// display settings (compositor.js), persisted; ?layout=... overrides for one session
const stored = (() => { try { return JSON.parse(localStorage.getItem('r3ds-display') || '{}'); } catch (e) { return {}; } })();
if (!(stored.v >= 2)) { delete stored.uiScale; delete stored.aspect; }   // new defaults: smaller, bottom right, widescreen
if (!(stored.v >= 3)) delete stored.romMode;   // new default: download the whole game (in the background)
const settings = Object.assign({}, window.R3DS_DEFAULTS, stored, { v: 3 });
// touch screens: the virtual pad covers the bottom corners, so the floating
// bottom screen UI starts at the bottom centre (once; the setting is kept after)
if (('ontouchstart' in window || navigator.maxTouchPoints > 0) && !stored.touchInit) { settings.uiAnchor = 'bc'; settings.touchInit = 1; }
for (const k of ['renderer', 'res', 'romMode']) if (params.get(k)) settings[k] = params.get(k);
for (const k of ['layout', 'filter', 'aspect', 'bottomMode']) if (params.get(k)) settings[k] = params.get(k);
const saveSettings = () => { try { localStorage.setItem('r3ds-display', JSON.stringify(settings)); } catch (e) {} };
const r3ds = window.r3ds = { Module: null, frames: 0, fps: 0, bytesRead: 0, started: false, errors: [] };

// ------------------------------------------------ single-file build (tools/bundle_html.py)
// In the bundled page every runtime payload lives in a
// <script type="text/plain" data-r3ds="name"> element holding base64.
// All of these are null in the normal multi-file build.
const EMBED = {};
for (const el of document.querySelectorAll('script[data-r3ds]')) EMBED[el.dataset.r3ds] = el.textContent.trim();
const embBytes = (k) => EMBED[k] ? Uint8Array.from(atob(EMBED[k]), (c) => c.charCodeAt(0)) : null;
const embURL = (k, type) => { const b = embBytes(k); return b ? URL.createObjectURL(new Blob([b], { type })) : null; };

// ---- ROM sanity check for picked files: the same magics rt/rom.cpp checks,
// run here so a bad file fails with a readable error instead of a black canvas
async function checkRomFile(f) {
  const rd = async (o, n) => new Uint8Array(await f.slice(o, Math.min(o + n, f.size)).arrayBuffer());
  const isMagic = (b, s) => b.length >= 4 && String.fromCharCode(b[0], b[1], b[2], b[3]) === s;
  const err = (msg) => ({ ok: false, msg });
  if (f.size < 0x400) return err('too small to be a 3DS image');
  const h = await rd(0, 0x200);
  const u32 = (b, o) => b[o] | b[o + 1] << 8 | b[o + 2] << 16 | b[o + 3] << 24;
  let ncchOff = 0;
  if (isMagic(h.subarray(0x100), 'NCSD')) {
    ncchOff = u32(h, 0x120) * 0x200;   // partition 0 offset, in media units
    if (!ncchOff || ncchOff >= f.size) return err('NCSD partition table looks wrong');
  } else if (isMagic(h.subarray(0x100), 'NCCH')) {
    ncchOff = 0;                       // bare CXI
  } else if (h[0] === 0x20 && h[1] === 0x20) {
    return err("this is a .cia install archive — installable apps aren't supported; use a .3ds/.cci cartridge image or a .cxi");
  } else return err('no NCSD/NCCH magic — not a .3ds/.cci/.cxi image (or badly damaged)');
  const ncch = await rd(ncchOff, 0x200);
  if (!isMagic(ncch.subarray(0x100), 'NCCH')) return err('no NCCH header at partition 0 — not a normal cartridge image');
  if (!(ncch[0x18F] & 0x04))
    return err('this ROM is still encrypted — decrypt it first (GodMode9: "Decrypt file…"), or use an already-decrypted .3ds/.cxi');
  const product = new TextDecoder('ascii').decode(ncch.subarray(0x150, 0x160)).replace(/\0.*$/, '').trim();
  return { ok: true, product };
}

// a short note over the status bar (quick keys)
let flashTimer = 0;
function flash(msg) {
  const el = $('status'); if (!el) return;
  r3ds.flash = msg; clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { r3ds.flash = ''; updateHud(); }, 1500);
  updateHud();
}
function showError(msg) {
  $('err').textContent = msg; r3ds.errors.push(msg); logLine('[error] ' + msg);
  // while playing, the setup panel is hidden — surface failures in the bar
  // and open the log, where the [rom]/[boot]/[ipc] lines say what happened
  if (r3ds.started) { $('log').style.display = 'block'; if ($('status')) $('status').textContent = 'error: ' + String(msg).split('\n')[0]; }
}
window.addEventListener('unhandledrejection', (e) => showError('unhandled runtime error: ' + (e.reason && e.reason.message || e.reason)));
window.addEventListener('error', (e) => { if (r3ds.started && e.message) showError('runtime error: ' + e.message); });
const logBuf = [];
function logLine(s) {
  logBuf.push(s); if (logBuf.length > 400) logBuf.shift();
  if (capture) capture.lines.push(`${((performance.now() - capture.t0) / 1000).toFixed(2)}s ${s}`);
  const el = $('log'); if (el.style.display === 'block') { el.textContent = logBuf.join('\n'); el.scrollTop = el.scrollHeight; }
  console.log(s);
}

// ------------------------------------------------ cross-origin isolation
async function ensureIsolation() {
  if (self.crossOriginIsolated) return true;
  // a coi-sw.js served next to the page fixes isolation on static hosts
  // (GitHub Pages and friends send no headers); on file:// there is no
  // navigator.serviceWorker at all, so this fails through to the message
  if ('serviceWorker' in navigator && !sessionStorage.getItem('coi-reloaded')) {
    try {
      await navigator.serviceWorker.register('coi-sw.js');
      await navigator.serviceWorker.ready;
      sessionStorage.setItem('coi-reloaded', '1');
      location.reload();
      return false;
    } catch (e) { /* fall through */ }
  }
  const how = EMBED.mod ?
    'Serve this file over HTTP(S) together with the repo\'s coi-sw.js — e.g. tools/serve_web.py (it sets the headers), or any static host where coi-sw.js can install as a service worker. file:// pages can never be cross-origin isolated.' :
    'Serve it with tools/serve_web.py (it sets the headers), or over https/localhost so the bundled service worker can add them.';
  showError('This page needs cross-origin isolation (SharedArrayBuffer for threads).\n' +
            '  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n' + how);
  return false;
}

// ------------------------------------------------------------- the game
let audioCtx = null, audioNode = null, muted = false;
function makeAudio() {
  try {
    try { audioCtx = new AudioContext({ sampleRate: 32728, latencyHint: 'interactive' }); }
    catch (e) { audioCtx = new AudioContext({ latencyHint: 'interactive' }); }
  } catch (e) { audioCtx = null; }
}
async function startAudio(Module) {
  if (!audioCtx) return;
  try {
    await audioCtx.audioWorklet.addModule(embURL('audioWorklet', 'text/javascript') || 'audio_worklet.js');
    audioNode = new AudioWorkletNode(audioCtx, 'r3ds-audio', { numberOfInputs: 0, outputChannelCount: [2] });
    audioNode.port.postMessage({ mem: Module.wasmMemory, ring: Module._web_audio_ring() });
    audioNode.connect(audioCtx.destination);
    const resume = () => { if (audioCtx.state !== 'running' && !muted) audioCtx.resume(); };
    resume();
    for (const ev of ['pointerdown', 'keydown', 'touchend']) window.addEventListener(ev, resume);
  } catch (e) { logLine('[web] audio unavailable: ' + e); }
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = () => rej(new Error('cannot load ' + src + ' (did you build the web target?)'));
    document.head.appendChild(s);
  });
}

async function start(source) {
  if (r3ds.started) return;
  r3ds.started = true;
  makeAudio();   // inside the click handler: browsers only allow audio after a gesture
  $('setup').style.display = 'none';
  $('stage').style.display = 'flex';
  document.querySelector('main').classList.add('playing');
  document.body.classList.add('playing');
  for (const p of document.querySelectorAll('main > .panel')) p.style.display = 'none';
  $('status').textContent = 'loading the recompiled game…';
  try {
    // embedded single-file build: load the module from a blob URL — its src
    // becomes _scriptName, so pthread workers spawn from the same blob
    if (!window.createR3DS) await loadScript(embURL('mod', 'text/javascript') || 'recomp3ds.js');
    const wasmURL = embURL('wasm', 'application/wasm');
    const wasmBytes = embBytes('wasm');
    const env = { R3DS_MMO: params.get('multiplayer') === '0' ? '0' : '1' };
    if (params.get('room')) env.R3DS_MMO_ROOM = params.get('room').slice(0, 64);
    if (params.get('multiplayerUrl')) env.R3DS_MMO_URL = params.get('multiplayerUrl');
    for (const kv of params.getAll('env')) { const i = kv.indexOf('='); if (i > 0) env[kv.slice(0, i)] = kv.slice(i + 1); }
    const Module = await window.createR3DS({
      ...(wasmURL ? { locateFile: (p) => p.endsWith('.wasm') ? wasmURL : p } : {}),
      ...(wasmBytes ? { wasmBinary: wasmBytes.buffer } : {}),
      onAbort: (w) => showError('guest runtime aborted' + (w ? ': ' + w : '') + ' — see the log'),
      onExit: (c) => { if (c) showError('guest exited with code ' + c + ' — see the log'); },
      print: (s) => logLine(s),
      printErr: (s) => { logLine(s); if (/^\[fatal\]|^abort\(|wasm exception/i.test(s)) showError(s); },
      preRun: [(M) => {
        for (const k in env) M.ENV[k] = env[k];
        M.FS.mkdir('/save');
        M.FS.mount(M.IDBFS, {}, '/save');
        M.addRunDependency('idbfs');
        M.FS.syncfs(true, (err) => {
          if (err) logLine('[web] save load: ' + err);
          seedDefaultSave(M).finally(() => M.removeRunDependency('idbfs'));
        });
      }],
    });
    r3ds.Module = Module;
    loadTexturePack(Module);
    // ROM streaming worker
    const worker = new Worker((embURL('romWorker', 'text/javascript') || 'rom_worker.js') +
                              (params.get('block') ? '?block=' + params.get('block') : ''));
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === 'error') { showError(d.msg); $('status').textContent = 'error: ' + d.msg; }
      else if (d.type === 'ready') { r3ds.romLocal = source.kind === 'file' || !!(r3ds.dl && !r3ds.dl.failed);
        logLine(`[web] ROM ready (${(d.size / 1048576).toFixed(0)} MiB, ${source.kind === 'file' ? 'local file' : r3ds.romLocal ? 'downloaded copy' : 'streamed'})`); }
      else if (d.type === 'download') {
        r3ds.dl = d;
        if (d.done >= d.total) r3ds.romLocal = true;
        if (d.failed) { logLine('[web] ROM download failed, streaming instead: ' + d.failed); flash('download failed, streaming instead'); }
      }
      else if (d.type === 'stats') { r3ds.bytesRead = d.bytesRead; r3ds.net = d; }
    };
    worker.onerror = (e) => showError('ROM worker error: ' + (e.message || 'unknown'));
    worker.postMessage({ mem: Module.wasmMemory, ctl: Module._web_rom_ctl(), source });
    r3ds.worker = worker;
    if (source.kind === 'file') $('log').style.display = 'block';   // boot diagnostics are the only feedback a generic build has
    // the compositor ticks at 60 fps whether the guest lives or not — call
    // out a game that never presents a frame instead of a silent black screen
    let bootSecs = 0;
    const bootWatch = setInterval(() => {
      if (!r3ds.Module || Module._web_frame_count() > 0) { clearInterval(bootWatch); return; }
      if ((bootSecs += 5) === 45) {
        logLine('[web] 45s with no guest frame — the game is probably stuck waiting on a service the runtime does not implement yet (see the last "unhandled cmd" line below)');
        $('status').textContent = 'booting… no guest frame yet — likely stalled on an unimplemented service (log below)';
      }
      if (bootSecs >= 300) clearInterval(bootWatch);
    }, 5000);
    startAudio(Module);
    startSaveSync(Module);
    startLoop(Module);
    setupInput(Module);
  } catch (e) {
    showError(String(e && e.message || e));
    $('setup').style.display = '';
  }
}

// ---------------------------------------------------------- texture pack
// HD replacements for the game's textures (tools/texlab.py builds them):
// textures.json maps a texture's key (the hash of its decoded pixels, the one
// rt/hwr_texrepl.cpp asks for) to [offset, length, width, height, type] in
// textures.pack, a run of WebP / PNG images. Until the pack is here, and while
// an image decodes, the game shows the original texture.
const texPack = { idx: null, buf: null, ready: new Map(), busy: new Set(), failed: new Set(), wanted: new Set() };
const texKey = (lo, hi) => (hi >>> 0).toString(16).padStart(8, '0') + (lo >>> 0).toString(16).padStart(8, '0');
async function texDecode(k) {
  const e = texPack.idx[k];
  texPack.busy.add(k);
  try {
    const blob = new Blob([texPack.buf.subarray(e[0], e[0] + e[1])], { type: 'image/' + (e[4] || 'webp') });
    const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(bmp.width, bmp.height) : Object.assign(document.createElement('canvas'), { width: bmp.width, height: bmp.height });
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    texPack.ready.set(k, { w: bmp.width, h: bmp.height, data: ctx.getImageData(0, 0, bmp.width, bmp.height).data });
    bmp.close && bmp.close();
  } catch (err) { texPack.failed.add(k); }
  texPack.busy.delete(k);
}
async function loadTexturePack(Module) {
  if (params.get('textures') === '0' || settings.hdTextures === false) return;
  const mem = () => Module.wasmMemory.buffer;
  // set before the index arrives: a texture uploaded meanwhile waits (1) and is asked for again
  Module.r3dsTexQuery = (lo, hi, wp, hp) => {
    if (!texPack.idx) return 1;
    const k = texKey(lo, hi);
    if (!texPack.idx[k] || texPack.failed.has(k)) return 0;
    const r = texPack.ready.get(k);
    if (r) { const u = new Uint32Array(mem()); u[(wp >>> 0) >> 2] = r.w; u[(hp >>> 0) >> 2] = r.h; return 2; }
    if (!texPack.buf) texPack.wanted.add(k);
    else if (!texPack.busy.has(k)) texDecode(k);
    return 1;
  };
  Module.r3dsTexCopy = (lo, hi, dst) => {
    const k = texKey(lo, hi), r = texPack.ready.get(k);
    if (!r) return;
    new Uint8Array(mem()).set(r.data, dst >>> 0);
    texPack.ready.delete(k);
  };
  let j = null;
  try { const r = await fetch('textures.json', { cache: 'no-cache' }); if (r.ok) j = await r.json(); } catch (e) {}
  if (!j || !j.entries) { texPack.idx = {}; return; }
  texPack.idx = j.entries;
  const n = Object.keys(j.entries).length;
  logLine(`[web] HD texture pack: ${n} textures, ${(j.bytes / 1048576 || 0).toFixed(0)} MiB, loading`);
  try {
    const r = await fetch('textures.pack', { cache: 'no-cache' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    texPack.buf = new Uint8Array(await r.arrayBuffer());
    logLine(`[web] HD texture pack ready`);
    for (const k of texPack.wanted) texDecode(k);
    texPack.wanted.clear();
  } catch (e) { logLine('[web] HD texture pack: ' + e); texPack.idx = {}; }
}

// ---------------------------------------------------------------- saves
let syncing = false;
function syncSaves(Module) {
  if (syncing) return;
  syncing = true;
  Module.FS.syncfs(false, (err) => { syncing = false; if (err) logLine('[web] save sync: ' + err); });
}
function startSaveSync(Module) {
  setInterval(() => syncSaves(Module), 3000);
  document.addEventListener('visibilitychange', () => { if (document.hidden) syncSaves(Module); });
  window.addEventListener('pagehide', () => syncSaves(Module));
}
function walk(FS, dir, out) {
  for (const n of FS.readdir(dir)) {
    if (n === '.' || n === '..') continue;
    const p = dir + '/' + n, st = FS.stat(p);
    if (FS.isDir(st.mode)) walk(FS, p, out);
    else out[p] = btoa(Array.from(FS.readFile(p), (b) => String.fromCharCode(b)).join(''));
  }
}
// A device without saves starts with the site's default save (default-save.json,
// tools/make_default_save.py). Only once per device, and never over an
// existing save.
// Which extdata directory the game uses is its own business: it is taken
// from ?extdata=<id> when given, else discovered under /save/extdata once the
// game has created one.
function extdataDir(FS) {
  if (params.get('extdata')) return '/save/extdata/' + params.get('extdata');
  try {
    const n = FS.readdir('/save/extdata').find((x) => x !== '.' && x !== '..');
    if (n) return '/save/extdata/' + n;
  } catch (e) {}
  return null;
}
function hasAnySave(FS) {
  const dirs = ['/save/savedata'];
  try { for (const n of FS.readdir('/save/extdata')) if (n !== '.' && n !== '..') dirs.push('/save/extdata/' + n); } catch (e) {}
  for (const d of dirs) try { if (FS.readdir(d).some((n) => n !== '.' && n !== '..')) return true; } catch (e) {}
  return false;
}
function writeSaveFiles(FS, files) {
  for (const [p, bytes] of Object.entries(files)) {
    if (!p.startsWith('/save/') || p.includes('..')) continue;
    const parts = p.split('/').slice(1, -1); let d = '';
    for (const x of parts) { d += '/' + x; try { FS.mkdir(d); } catch (e) {} }
    FS.writeFile(p, bytes);
  }
}
const unb64 = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
async function seedDefaultSave(M) {
  const has = hasAnySave(M.FS);
  let seeded = false;
  try { seeded = localStorage.getItem('r3ds-default-save') === '1'; } catch (e) {}
  if (has || seeded || params.get('defaultSave') === '0') return;
  try {
    const r = await fetch('default-save.json', { cache: 'no-store' });
    if (!r.ok) return;
    const j = await r.json();
    if (j.format !== 'r3ds-save-1') return;
    const files = {};
    for (const [p, b64] of Object.entries(j.files)) files[p] = unb64(b64);
    writeSaveFiles(M.FS, files);
    try { localStorage.setItem('r3ds-default-save', '1'); } catch (e) {}
    logLine(`[web] default save installed (${Object.keys(files).length} files)${j.note ? ': ' + j.note : ''}`);
  } catch (e) { logLine('[web] default save: ' + e); }
}
function exportSave() {
  const out = {}; walk(r3ds.Module.FS, '/save', out);
  const blob = new Blob([JSON.stringify({ format: 'r3ds-save-1', files: out })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'r3ds-save.json'; a.click();
}
// an export of this page (.json), or the game's own save files (from a 3DS
// with Checkpoint / JKSM, or an emulator's dump): a bare file goes into the
// game's extdata directory, resolved by extdataDir(). If the game has not
// created one yet, name it with ?extdata=<id>.
async function importSave(list) {
  const FS = r3ds.Module.FS;
  const files = {};
  for (const file of list) {
    if (/\.json$/i.test(file.name)) {
      const j = JSON.parse(await file.text());
      if (j.format !== 'r3ds-save-1') throw new Error(file.name + ': not a save export from this page');
      for (const [p, b64] of Object.entries(j.files)) files[p] = unb64(b64);
      continue;
    }
    const dir = extdataDir(FS);
    if (!dir) throw new Error(file.name + ': no extdata directory yet; open the page with ?extdata=<id> or let the game create its data first');
    files[`${dir}/${file.name.replace(/[^A-Za-z0-9_.-]/g, '_')}`] = new Uint8Array(await file.arrayBuffer());
  }
  writeSaveFiles(FS, files);
  FS.syncfs(false, () => { alert(`Save imported (${Object.keys(files).length} files). The page will reload so the game picks it up.`); location.reload(); });
}

// ------------------------------------------------------- log capture
// "Record log": one line of counters per second plus the whole runtime log,
// downloaded as a text file when stopped (for debugging a session).
let capture = null;
const DBG = ['resets', 'queueBytes', 'invalidRange', 'invalidHash', 'loadColor', 'loadDepth', 'texUp', 'texBytes', 'texSame',
             'skipPass', 'draws', 'tris', 'exeFrames', 'softTop', 'softBot', 'replayMs', 'submitMs', 'presentMs',
             'fsReads', 'fsBytes', 'fsMs', 'fsMaxMs', 'gxQueueMax', 'gxCmds', 'gxWaitMs', 'recWaitMs', 'vblankWaitMs', 'snapMs', 'snapBytes', 'gameFrames', 'gameGapMax', 'gpuBusyMs', 'cpuIdleMs', 'gxWaitMax', 'ctxSwitches', 'paceMs', 'paceNew', 'vblanks'];
function readDbg() {
  const M = r3ds.Module;
  if (!M || !M._web_debug_stats) return null;
  const a = new Float64Array(M.wasmMemory.buffer, M._web_debug_stats(), DBG.length);
  const o = {}; DBG.forEach((k, i) => { o[k] = a[i]; }); return o;
}
function captureTick() {
  if (!capture) return;
  const now = performance.now(), d = readDbg(), p = capture.prev;
  if (d && p) {
    const x = (k) => d[k] - p[k], ms = (now - capture.tPrev) / 1000;
    const n = r3ds.net || {}, pn = capture.prevNet || {};
    const netReq = (n.netReqs || 0) - (pn.netReqs || 0), netMs = (n.netMs || 0) - (pn.netMs || 0), netB = (n.netBytes || 0) - (pn.netBytes || 0);
    capture.stats.push(`${((now - capture.t0) / 1000).toFixed(1)}s | shown ${(x('exeFrames') / ms || r3ds.fps || 0).toFixed(1)} fps, vblank ${(x('vblanks') / ms).toFixed(1)}/s, ` +
      `rAF max ${capture.maxDt.toFixed(0)} ms, bottom ${r3ds.bottomFocus ? 'menu' : 'picture'} | ` +
      `GAME ${(x('gameFrames') / ms).toFixed(1)} fps, longest gap ${d.gameGapMax.toFixed(0)} ms, GPU thread busy ${(x('gpuBusyMs') / ms / 10).toFixed(0)}%, emulated CPU idle ${(x('cpuIdleMs') / ms / 10).toFixed(0)}%, longest hold ${d.gxWaitMax.toFixed(0)} ms, thread switches ${(x('ctxSwitches') / ms).toFixed(0)}/s, display delay ${d.paceMs.toFixed(0)} ms | ` +
      `ROM reads ${x('fsReads')} (${(x('fsBytes') / 1048576).toFixed(2)} MiB) ${x('fsReads') ? (x('fsMs') / x('fsReads')).toFixed(1) : 0} ms avg, max ${d.fsMaxMs.toFixed(0)} ms, ` +
      `net ${netReq} req ${(netB / 1048576).toFixed(2)} MiB ${netReq ? (netMs / netReq).toFixed(0) : 0} ms/req | ` +
      `GX cmds ${x('gxCmds')} queue max ${d.gxQueueMax} game held ${x('gxWaitMs').toFixed(0)} ms, GPU thread held ${x('recWaitMs').toFixed(0)} ms, vblank held ${x('vblankWaitMs').toFixed(0)} ms, snapshots ${(x('snapBytes') / 1048576).toFixed(1)} MiB in ${x('snapMs').toFixed(0)} ms | draws ${x('draws')} tris ${x('tris')} skipped passes ${x('skipPass')} | ` +
      `tex up ${x('texUp')} (${(x('texBytes') / 1048576).toFixed(2)} MiB) same ${x('texSame')} | invalid range ${x('invalidRange')} hash ${x('invalidHash')} | ` +
      `loads color ${x('loadColor')} depth ${x('loadDepth')} | resets ${x('resets')} queue ${(d.queueBytes / 1048576).toFixed(1)} MiB | ` +
      `software image top ${x('softTop')} bottom ${x('softBot')} | cpu replay ${x('replayMs').toFixed(1)} submit ${x('submitMs').toFixed(1)} present ${x('presentMs').toFixed(1)} ms`);
  }
  capture.prev = d; capture.prevNet = Object.assign({}, r3ds.net || {}); capture.tPrev = now; capture.maxDt = 0;
}
function toggleCapture() {
  const btn = $('rec');
  if (!capture) {
    capture = { t0: performance.now(), tPrev: performance.now(), lines: [], stats: [], prev: readDbg(), prevNet: Object.assign({}, r3ds.net || {}), maxDt: 0 };
    capture.timer = setInterval(captureTick, 1000);
    if (btn) { btn.textContent = '■ Stop & save log'; btn.classList.add('rec'); }
    flash('recording log');
    return;
  }
  clearInterval(capture.timer);
  captureTick();
  const url = new URL(location.href); url.searchParams.delete('key');
  const head = [
    `recomp3ds log capture, ${new Date().toISOString()}`,
    `page ${url.href}`, `browser ${navigator.userAgent}`,
    `threads ${navigator.hardwareConcurrency} · cross-origin isolated ${self.crossOriginIsolated} · device pixel ratio ${devicePixelRatio} · canvas ${$('screen').width}x${$('screen').height}`,
    `renderer ${r3ds.gpuScale ? 'WebGPU ' + r3ds.gpuScale + 'x' : 'software'} · settings ${JSON.stringify(settings)}`,
    `duration ${((performance.now() - capture.t0) / 1000).toFixed(1)} s`, '',
    '== per second', ...capture.stats, '', '== log', ...capture.lines, '',
    '== last 400 log lines before the capture', ...logBuf.slice(0, Math.max(0, logBuf.length - capture.lines.length)),
  ];
  const blob = new Blob([head.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = `r3ds-log-${Date.now()}.txt`; a.click();
  capture = null;
  if (btn) { btn.textContent = '● Record log'; btn.classList.remove('rec'); }
}

// --------------------------------------------------------- display loop
// Two display paths: the WebGPU renderer (hwr_gpu.cpp: the game drawn again
// on the GPU at a higher resolution, and the screens composed there), or the
// software frame composed by compositor.js with WebGL2.
const MAP = {
  layout: { remaster: 0, classic: 1, side: 2, top: 3, large: 4 }, bottomMode: { auto: 0, ui: 1, minimap: 2 },
  miniCorner: { br: 0, bl: 1, tr: 2, tl: 3 }, uiAnchor: { br: 0, bc: 1, bl: 2, tr: 3, tl: 4 }, filter: { pixel: 0, smooth: 1, hq: 2 },
  aspect: { native: 0, auto: 1, wide: 2, stretch: 3 },
};
function startLoop(Module) {
  let cv = $('screen');
  let mode = 'soft', comp = null;
  if (settings.renderer !== 'soft' && navigator.gpu && Module._web_hwr_start) {
    mode = Module._web_hwr_start() === 3 ? 'soft' : 'gpu';
  }
  const toSoft = (why) => {
    if (why) logLine('[web] ' + why + ' -- using the software renderer');
    if (mode === 'gpu') {   // the canvas belongs to WebGPU now: replace it
      const n = cv.cloneNode(false); cv.replaceWith(n); cv = n; bindPointer(cv);
    }
    mode = 'soft';
    try { comp = r3ds.comp = new R3DSCompositor(cv); } catch (e) { showError(String(e.message || e)); }
  };
  if (mode === 'soft') toSoft(settings.renderer === 'soft' ? '' : (navigator.gpu ? 'WebGPU did not start' : 'this browser has no WebGPU'));
  // what the pointer handlers need from the GPU path (rectangle in CSS px, mask)
  const gpuComp = { bottomRect: null, showFull: false, bottomAlpha: (x, y) => Module._web_hwr_soft_alpha(x, y) };
  r3ds.getFrame = () => comp && comp.frame;   // raw 400x480 RGBA of the last software frame (tests)
  let last = performance.now(), count = 0, dirtyUntil = 0, lastSettings = '', waitSince = performance.now();
  let prevTick = performance.now();
  const tick = () => {
    const now0 = performance.now();
    if (capture) capture.maxDt = Math.max(capture.maxDt, now0 - prevTick);
    prevTick = now0;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
    let dirty = false;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; dirty = true; }
    if (mode === 'gpu') {
      const S = settings;
      const r = Module._web_hwr_frame(w, h, MAP.layout[S.layout] | 0, MAP.bottomMode[S.bottomMode] | 0, S.removeBg ? 1 : 0,
        +S.uiScale, +S.uiOpacity, +S.miniScale, +S.miniOpacity, MAP.miniCorner[S.miniCorner] | 0, MAP.uiAnchor[S.uiAnchor] | 0,
        MAP.filter[S.filter] ?? 2, MAP.aspect[S.aspect] ?? 1, S.sideFill === 'blur' ? 1 : 0, keys.has('Tab') ? 1 : 0,
        S.res === 'auto' ? 0 : (+S.res | 0), 6);
      if (r < 0) {
        const st = Module._web_hwr_state();
        if (st === 3) toSoft('WebGPU failed');
        else if (now0 - waitSince > 15000) toSoft('WebGPU did not become ready');
      } else {
        if (r & 1) { r3ds.frames++; count++; }
        const o = new Float32Array(Module.wasmMemory.buffer, Module._web_hwr_out(), 9);
        gpuComp.bottomRect = o[4] ? { x: o[0] / dpr, y: o[1] / dpr, w: o[2] / dpr, h: o[3] / dpr, hit: !!o[5] } : null;
        r3ds.gpuScale = o[7]; r3ds.gpuTop = !!o[6]; r3ds.bottomFocus = !!o[8];
        r3ds.comp = gpuComp;
      }
    } else if (comp) {
      const p = Module._web_frame();
      if (p) {
        comp.upload(new Uint8Array(Module.wasmMemory.buffer, p, 400 * 480 * 4), Module._web_bottom_coverage(), !!Module._web_bottom_masked());
        r3ds.frames++; count++; dirty = true;
      }
      const full = keys.has('Tab'), sk = JSON.stringify(settings);
      if (full !== comp.showFull || sk !== lastSettings) { comp.showFull = full; lastSettings = sk; dirty = true; dirtyUntil = now0 + 400; }
      if (dirty || now0 < dirtyUntil) comp.render(settings, dpr);   // redraw on new frames and while animating
      r3ds.bottomFocus = comp.masked && comp.mode === 'ui';
    }
    pollInput(Module);
    const now = performance.now();
    if (now - last >= 1000) {
      r3ds.fps = count * 1000 / (now - last); count = 0; last = now;
      let s = settings.showFps ? `${r3ds.fps.toFixed(1)} fps · ` : '';
      s += mode === 'gpu' ? (r3ds.gpuScale ? `WebGPU ${r3ds.gpuScale}× · ` : 'WebGPU starting · ') : 'software · ';
      const dl = r3ds.dl;
      if (dl && !dl.failed && dl.done < dl.total)
        s += `downloading the game ${(dl.done / dl.total * 100).toFixed(0)}% (${(dl.done / 1048576).toFixed(0)} / ${(dl.total / 1048576).toFixed(0)} MiB` +
             (dl.rate ? `, ${(dl.rate / 1048576).toFixed(1)} MiB/s, ${Math.ceil((dl.total - dl.done) / dl.rate / 60)} min left)` : ')');
      else if (r3ds.romLocal) s += 'ROM on this device';
      else s += `ROM streamed ${(r3ds.bytesRead / 1048576).toFixed(1)} MiB`;
      if (r3ds.flash) s = r3ds.flash + ' · ' + s;
      const n = r3ds.net;   // network: throughput while fetching, and time per request
      if (n && n.netReqs) s += ` · net ${(n.netBytes / 1048576 / (n.netMs / 1000)).toFixed(1)} MiB/s, ${(n.netMs / n.netReqs).toFixed(0)} ms/request`;
      if (r3ds.storeProgress !== undefined && r3ds.storeProgress < 1) s += ` · storing ROM copy ${(r3ds.storeProgress * 100).toFixed(0)}%`;
      $('status').textContent = s;
      updateHud();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// settings panel
function setupSettings() {
  const panel = $('settings');
  const sync = () => {
    for (const el of panel.querySelectorAll('[data-k]')) {
      const k = el.dataset.k;
      if (el.type === 'checkbox') el.checked = !!settings[k]; else el.value = settings[k];
    }
  };
  for (const el of panel.querySelectorAll('[data-k]')) {
    el.addEventListener('input', () => {
      const k = el.dataset.k;
      settings[k] = el.type === 'checkbox' ? el.checked : el.type === 'range' ? parseFloat(el.value) : el.value;
      saveSettings();
    });
  }
  $('gear').onclick = () => { sync(); if (!r3ds.started || document.body.classList.contains('playing')) setMenu(true); panel.style.display = panel.style.display === 'block' ? 'none' : 'block'; };
  $('closeSettings').onclick = () => { panel.style.display = 'none'; };
  $('resetSettings').onclick = () => { Object.assign(settings, window.R3DS_DEFAULTS); saveSettings(); sync(); };
  sync();
  // game data: the downloaded copy of the ROM (OPFS rom.bin / rom.meta)
  const romHint = async () => {
    let t = 'Download (the default) stores the whole game on this device once, in the background while you play: until it is done the parts not yet here are streamed. Afterwards loading is smooth and the game plays offline. Streaming only reads what the game needs, when it needs it. Changes apply after a reload.';
    try { const e = await navigator.storage.estimate(); if (e.usage > 64 << 20) t += ` This site stores ${(e.usage / 1048576).toFixed(0)} MB now.`; } catch (e) {}
    $('romHint').textContent = t;
  };
  romHint();
  panel.querySelector('[data-k="romMode"]').addEventListener('change', () => {
    if (r3ds.started && confirm('Reload now to apply? Progress since your last in-game save is lost.')) location.reload();
  });
  $('romDelete').onclick = async () => {
    if (!confirm('Delete the downloaded copy of the game from this device?')) return;
    if (r3ds.source && r3ds.source.kind === 'pack') {   // in use by this session: delete on the next load
      settings.romMode = 'stream'; saveSettings();
      try { localStorage.setItem('r3ds-rom-delete', '1'); } catch (e) {}
      location.reload();
      return;
    }
    await deleteRomCopy(); romHint(); flash('downloaded copy deleted');
  };
}

async function deleteRomCopy() {
  try {
    const root = await navigator.storage.getDirectory();
    for (const n of ['rom.bin', 'rom.meta']) { try { await root.removeEntry(n); } catch (e) {} }
  } catch (e) {}
}

// ------------------------------------------------- full screen and the menu
// While playing, the game fills the screen and the page never scrolls. The
// browser's full screen is entered on the first click / key / touch (it needs
// a gesture) and again after leaving it. Esc -- which also leaves full screen
// -- opens the menu (status, buttons, Display ⚙); on touch screens the small ⋯
// button at the top does. On an iPhone, Safari has no full screen for pages:
// Share → Add to Home Screen starts the game full screen (manifest.webmanifest).
function updateHud() {
  const el = $('hud'); if (!el) return;
  const parts = [];
  if (r3ds.flash) parts.push(r3ds.flash);
  if (settings.showFps && r3ds.fps) parts.push(`${r3ds.fps.toFixed(0)} fps`);
  const dl = r3ds.dl;
  if (dl && !dl.failed && dl.done < dl.total) parts.push(`downloading ${(dl.done / dl.total * 100).toFixed(0)}%`);
  el.textContent = parts.join(' · ');
}
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
const fsAvailable = () => !!(document.fullscreenEnabled || document.webkitFullscreenEnabled);
function enterFullscreen() {
  if (!settings.autoFs || fsElement() || !fsAvailable()) return;
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen;
  try {
    const p = req.call(el, { navigationUI: 'hide' });
    const lock = () => { if (document.body.classList.contains('touch') && screen.orientation && screen.orientation.lock) screen.orientation.lock('landscape').catch(() => {}); };
    if (p && p.then) p.then(lock, () => {}); else lock();
  } catch (e) {}
}
function setMenu(open) {
  document.body.classList.toggle('menu', open);
  if (!open) { $('settings').style.display = 'none'; $('log').style.display = 'none'; }
}
function setupScreen() {
  const playing = () => document.body.classList.contains('playing');
  const menuOpen = () => document.body.classList.contains('menu');
  const isIos = /iP(hone|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1 && !fsAvailable());
  const standalone = matchMedia('(display-mode: fullscreen), (display-mode: standalone)').matches || navigator.standalone;
  if (isIos && !standalone) $('status').title = 'Full screen on iPhone: Share → Add to Home Screen';
  $('resume').onclick = () => { setMenu(false); enterFullscreen(); };
  $('menuBtn').onclick = (e) => { e.stopPropagation(); setMenu(true); };
  window.addEventListener('keydown', (e) => {
    if (e.code !== 'Escape' || !playing()) return;
    e.preventDefault();
    // the same Esc may also have left full screen (fullscreenchange opens the menu)
    if (fsElement() || performance.now() - lastFsExit < 600) { setMenu(true); return; }
    if ($('settings').style.display === 'block') { $('settings').style.display = 'none'; return; }
    setMenu(!menuOpen());
  });
  // leaving full screen (Esc, a system gesture) opens the menu
  let lastFsExit = -1e9;
  const fsChange = () => { if (!fsElement()) { lastFsExit = performance.now(); if (playing() && settings.autoFs) setMenu(true); } };
  document.addEventListener('fullscreenchange', fsChange);
  document.addEventListener('webkitfullscreenchange', fsChange);
  // any gesture in the game goes back to full screen (browsers need one)
  const gesture = (e) => {
    if (!playing() || menuOpen() || (e.code === 'Escape')) return;
    if (e.target && e.target.closest && e.target.closest('#settings, #bar, #log')) return;
    enterFullscreen();
  };
  for (const ev of ['pointerup', 'keydown']) window.addEventListener(ev, gesture);
  // nothing scrolls, zooms or selects text over the game
  document.addEventListener('touchmove', (e) => {
    if (playing() && !(e.target.closest && e.target.closest('#settings, #log, #bar'))) e.preventDefault();
  }, { passive: false });
  for (const ev of ['gesturestart', 'gesturechange', 'dblclick']) document.addEventListener(ev, (e) => { if (playing()) e.preventDefault(); }, { passive: false });
  document.addEventListener('contextmenu', (e) => { if (playing() && !(e.target.closest && e.target.closest('#settings, #log, #bar'))) e.preventDefault(); });
}

// ---------------------------------------------------------------- input
const keys = new Set();
const virt = { buttons: 0 };
let touch = null;   // {x, y} in bottom-screen coordinates
function setupInput(Module) {
  const typing = (e) => e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA');
  // quick display keys: 1 layout, 2 bottom screen position, 3 remove the bottom screen's background
  const cycle = (k, values) => { settings[k] = values[(values.indexOf(settings[k]) + 1) % values.length]; saveSettings(); flash(`${k}: ${settings[k]}`); };
  window.addEventListener('keydown', (e) => {
    if (typing(e) || e.repeat) return;
    if (e.code === 'Digit1') { cycle('layout', ['remaster', 'classic', 'large', 'side', 'top']); e.preventDefault(); return; }
    if (e.code === 'Digit2') { cycle('uiAnchor', ['br', 'bc', 'bl', 'tr', 'tl']); e.preventDefault(); return; }
    if (e.code === 'Digit3') { settings.removeBg = !settings.removeBg; saveSettings(); flash(`remove background: ${settings.removeBg ? 'on' : 'off'}`); e.preventDefault(); }
  });
  window.addEventListener('keydown', (e) => {
    if (typing(e)) return;
    if (KEYMAP[e.code] !== undefined || ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Tab'].includes(e.code)) { keys.add(e.code); e.preventDefault(); }
  });
  window.addEventListener('keyup', (e) => { keys.delete(e.code); });
  window.addEventListener('blur', () => keys.clear());
  bindPointer($('screen'));
  if ('ontouchstart' in window || navigator.maxTouchPoints > 0) buildVirtualPad();
}
function bindPointer(cv) {
  // canvas CSS px -> bottom-screen pixel through the current layout
  const pos = (e) => {
    const r = cv.getBoundingClientRect(), b = r3ds.comp && r3ds.comp.bottomRect;
    if (!b) return null;
    const x = Math.floor((e.clientX - r.left - b.x) / b.w * 320), y = Math.floor((e.clientY - r.top - b.y) / b.h * 240);
    if (x < 0 || y < 0 || x >= 320 || y >= 240) return null;
    return { x, y, hit: b.hit };
  };
  cv.addEventListener('pointerdown', (e) => {
    const p = pos(e);
    // over the game, a click on a removed background is not a touch
    if (!p || (p.hit && r3ds.comp.bottomAlpha(p.x, p.y) < 24)) return;
    cv.setPointerCapture(e.pointerId); touch = p; e.preventDefault();
  });
  cv.addEventListener('pointermove', (e) => { if (touch) { const p = pos(e); if (p) touch = p; } });
  const up = () => { touch = null; };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
}
// while the bottom screen shows a menu, WASD drive the D-pad and the arrows the circle pad
const SWAP = { KeyW: 'ArrowUp', KeyA: 'ArrowLeft', KeyS: 'ArrowDown', KeyD: 'ArrowRight',
               ArrowUp: 'KeyW', ArrowLeft: 'KeyA', ArrowDown: 'KeyS', ArrowRight: 'KeyD' };
function pollInput(Module) {
  let b = virt.buttons, cx = 0, cy = 0;
  const swap = settings.swapKeys && r3ds.bottomFocus;
  const held = swap ? new Set([...keys].map((k) => SWAP[k] || k)) : keys;
  for (const k of held) if (KEYMAP[k]) b |= KEYMAP[k];
  if (held.has('KeyA')) cx -= 156;
  if (held.has('KeyD')) cx += 156;
  if (held.has('KeyW')) cy += 156;
  if (held.has('KeyS')) cy -= 156;
  for (const g of (navigator.getGamepads ? navigator.getGamepads() : [])) {
    if (!g) continue;
    const p = (i) => g.buttons[i] && g.buttons[i].pressed;
    if (p(1)) b |= BTN.A; if (p(0)) b |= BTN.B; if (p(3)) b |= BTN.X; if (p(2)) b |= BTN.Y;
    if (p(4) || p(6)) b |= BTN.L; if (p(5) || p(7)) b |= BTN.R;
    if (p(9)) b |= BTN.START; if (p(8)) b |= BTN.SELECT;
    if (p(12)) b |= BTN.UP; if (p(13)) b |= BTN.DOWN; if (p(14)) b |= BTN.LEFT; if (p(15)) b |= BTN.RIGHT;
    const ax = g.axes[0] || 0, ay = g.axes[1] || 0;
    if (Math.abs(ax) > 0.18) cx = Math.round(ax * 156);
    if (Math.abs(ay) > 0.18) cy = Math.round(-ay * 156);
  }
  const t = touch && touch.x >= 0 && touch.x < 320 && touch.y >= 0 && touch.y < 240;
  Module._web_input(b >>> 0, cx, cy, t ? 1 : 0, t ? touch.x : 0, t ? touch.y : 0);
}
function buildVirtualPad() {
  document.body.classList.add('touch');
  const pad = $('pad');
  const mk = (label, mask, css) => {
    const d = document.createElement('div');
    d.className = 'vb'; d.textContent = label; Object.assign(d.style, css);
    const on = (e) => { virt.buttons |= mask; d.classList.add('on'); e.preventDefault(); };
    const off = (e) => { virt.buttons &= ~mask; d.classList.remove('on'); e.preventDefault(); };
    d.addEventListener('pointerdown', on); d.addEventListener('pointerup', off);
    d.addEventListener('pointercancel', off); d.addEventListener('pointerleave', off);
    pad.appendChild(d);
  };
  mk('▲', BTN.UP, { left: '62px', bottom: '130px' }); mk('▼', BTN.DOWN, { left: '62px', bottom: '14px' });
  mk('◀', BTN.LEFT, { left: '6px', bottom: '72px' }); mk('▶', BTN.RIGHT, { left: '118px', bottom: '72px' });
  mk('A', BTN.A, { right: '6px', bottom: '72px' }); mk('B', BTN.B, { right: '62px', bottom: '14px' });
  mk('X', BTN.X, { right: '62px', bottom: '130px' }); mk('Y', BTN.Y, { right: '118px', bottom: '72px' });
  mk('L', BTN.L, { left: '6px', bottom: '200px' }); mk('R', BTN.R, { right: '6px', bottom: '200px' });
  mk('ST', BTN.START, { right: '130px', bottom: '200px', width: '44px', height: '34px' });
  mk('SE', BTN.SELECT, { left: '130px', bottom: '200px', width: '44px', height: '34px' });
}

// ------------------------------------------------------------------ UI
async function init() {
  const ok = await ensureIsolation();
  $('env').textContent = `threads: ${navigator.hardwareConcurrency || '?'} · cross-origin isolated: ${self.crossOriginIsolated}`;
  $('fs').onclick = () => {
    if (fsElement()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    else { const a = settings.autoFs; settings.autoFs = true; setMenu(false); enterFullscreen(); settings.autoFs = a; }
  };
  setupScreen();
  setupSettings();
  $('mute').onclick = () => {
    muted = !muted; $('mute').textContent = muted ? 'Unmute' : 'Mute';
    if (audioCtx) muted ? audioCtx.suspend() : audioCtx.resume();
  };
  $('toggleLog').onclick = () => {
    const el = $('log'); el.style.display = el.style.display === 'block' ? 'none' : 'block';
    el.textContent = logBuf.join('\n'); el.scrollTop = el.scrollHeight;
  };
  $('exportSave').onclick = () => exportSave();
  $('rec').onclick = () => toggleCapture();
  if (params.get('debug') === '1') setTimeout(() => toggleCapture(), 0);
  $('importSave').onclick = () => $('importFile').click();
  $('importFile').onchange = () => { const f = $('importFile').files; if (f.length) importSave([...f]).catch((e) => showError(String(e))); };
  if (!ok) return;
  // the text font must be ready before the game's glyph sheets are redrawn with it
  try { await Promise.race([document.fonts.load('700 48px "M PLUS Rounded 1c"', 'AaK'), new Promise((r) => setTimeout(r, 4000))]); } catch (e) {}
  let del = false;
  try { del = localStorage.getItem('r3ds-rom-delete') === '1'; localStorage.removeItem('r3ds-rom-delete'); } catch (e) {}
  if (del) await deleteRomCopy();
  // single-file build: nothing is hosted next to the page, so unless ?rom=
  // points at a URL the user supplies the file (picker or drag & drop)
  if (EMBED.mod && !params.get('rom')) {
    $('loading').textContent = 'This is the standalone build — pick a 3DS ROM to play.';
    $('romPick').hidden = false;
    $('romBtn').onclick = () => $('romFile').click();
    const useFile = async (f) => {
      if (!f || r3ds.started) return;
      $('loading').textContent = 'checking ' + f.name + '…';
      const c = await checkRomFile(f).catch((e) => ({ ok: false, msg: String(e && e.message || e) }));
      if (!c.ok) { showError(c.msg); $('loading').textContent = 'Pick a decrypted .3ds/.cci cartridge image or a .cxi.'; return; }
      $('loading').textContent = 'booting ' + (c.product ? c.product + ' — ' : '') + f.name + '…';
      r3ds.product = c.product;
      start({ kind: 'file', file: f });
    };
    $('romFile').onchange = () => useFile($('romFile').files[0]);
    window.addEventListener('dragover', (e) => { e.preventDefault(); });
    window.addEventListener('drop', (e) => { e.preventDefault(); useFile(e.dataTransfer.files && e.dataTransfer.files[0]); });
    return;
  }
  const url = new URL(params.get('rom') || 'rom.3ds', location.href).href;
  if (settings.romMode === 'download' && !params.get('rom') && navigator.storage && navigator.storage.getDirectory) {
    try { if (navigator.storage.persist) await navigator.storage.persist(); } catch (e) {}   // keep it from being evicted
    r3ds.source = { kind: 'pack', url, index: new URL('rom.pack.json', location.href).href, base: new URL('rompack/', location.href).href };
  } else r3ds.source = { kind: 'url', url };
  start(r3ds.source);
}
init();
})();
