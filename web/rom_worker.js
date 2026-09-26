// ROM streaming worker. Guest threads post a read request into a control
// block in (shared) wasm memory and block on it; this worker fills the
// bytes straight into wasm memory and wakes them. Sources:
//   file : a File the user picked (FileReaderSync, random access, no copy)
//   opfs : a copy of the ROM kept in the browser's Origin Private File System
//   url  : an HTTP(S) URL read with Range requests, with an LRU block cache
//          (only the parts of the ROM the game touches are ever read)
//   pack : the whole ROM, downloaded once as brotli chunks (tools/pack_rom.py,
//          served by the Worker) into OPFS in the background while the game
//          plays, streaming (url) what is not there yet. Interrupted
//          downloads resume; a complete copy also plays offline. Without a
//          pack on the server it falls back to url.
'use strict';

const BLOCK = (() => { const v = Number(new URLSearchParams(self.location.search).get('block')); return v >= 16 ? v << 10 : 256 << 10; })();   // URL mode: block size (KiB via ?block=, default 256 KiB)
const MAX_BLOCKS = (256 << 20) / BLOCK;   // URL mode: keep up to 256 MiB cached
const MAX_AHEAD = Math.max(1, (4 << 20) / BLOCK);   // read-ahead up to 4 MiB

let read = null, size = 0, bytesRead = 0, requests = 0, lastStats = 0;
let netBytes = 0, netMs = 0, netReqs = 0;   // URL mode: what the network delivered, and how long it took

function openFile(file) {
  const fr = new FileReaderSync();
  size = file.size;
  read = (off, len) => new Uint8Array(fr.readAsArrayBuffer(file.slice(off, off + len)));
}

async function openOpfs(name) {
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(name);
  const h = await fh.createSyncAccessHandle();
  size = h.getSize();
  read = (off, len) => {
    const b = new Uint8Array(len);
    const n = h.read(b, { at: off });
    return n === len ? b : b.subarray(0, n);
  };
}

function xhrRange(url, a, b) {
  const x = new XMLHttpRequest();
  x.open('GET', url, false);
  x.responseType = 'arraybuffer';
  x.setRequestHeader('Range', `bytes=${a}-${b}`);
  const t0 = performance.now();
  x.send();
  netMs += performance.now() - t0; netReqs++;
  if (x.response) netBytes += x.response.byteLength;
  if (x.status !== 206 && x.status !== 200) throw new Error(`HTTP ${x.status} for ${url}`);
  if (x.status === 200 && (a !== 0 || x.response.byteLength > b - a + 1))
    throw new Error('the server ignored the Range header; it must support byte ranges to stream the ROM');
  return { data: new Uint8Array(x.response), range: x.getResponseHeader('Content-Range') };
}

function openUrl(url) {
  const r = xhrRange(url, 0, 0);
  const m = r.range && r.range.match(/\/(\d+)\s*$/);
  if (!m) throw new Error('no Content-Range total in the response; the server must support byte ranges');
  size = Number(m[1]);
  const cache = new Map();   // block index -> Uint8Array, insertion order = LRU
  const nblocks = Math.ceil(size / BLOCK);
  const put = (i, v) => { cache.set(i, v); if (cache.size > MAX_BLOCKS) cache.delete(cache.keys().next().value); };
  // one request per run of missing blocks of a read; while reads continue
  // where the previous one ended (loading a file), also fetch 1, 2, then 4
  // blocks ahead: every request is a round trip to the server
  let nextOff = -1, ahead = 0;
  read = (off, len) => {
    len = Math.max(0, Math.min(len, size - off));
    const out = new Uint8Array(len);
    if (!len) return out;
    ahead = off === nextOff ? Math.min(ahead ? ahead * 2 : 1, MAX_AHEAD) : 0;
    nextOff = off + len;
    const i0 = Math.floor(off / BLOCK), i1 = Math.floor((off + len - 1) / BLOCK);
    for (let i = i0; i <= i1; i++) {
      if (cache.has(i)) continue;
      let j = i;
      while (j + 1 <= i1 && !cache.has(j + 1)) j++;
      if (j === i1) { let k = 0; while (k < ahead && j + 1 < nblocks && !cache.has(j + 1)) { j++; k++; } }
      const data = xhrRange(url, i * BLOCK, Math.min(size, (j + 1) * BLOCK) - 1).data;
      for (let k = i; k <= j; k++) put(k, data.subarray((k - i) * BLOCK, Math.min(data.length, (k - i + 1) * BLOCK)));
      i = j;
    }
    let done = 0;
    while (done < len) {
      const p = off + done, i = Math.floor(p / BLOCK), o = p - i * BLOCK;
      let b = cache.get(i);
      if (!b) throw new Error('short read at block ' + i);
      cache.delete(i); cache.set(i, b);   // LRU touch
      const n = Math.min(len - done, b.length - o);
      if (n <= 0) break;
      out.set(b.subarray(o, o + n), done);
      done += n;
    }
    return done === len ? out : out.subarray(0, done);
  };
}

// ---------------------------------------------------------------- pack
const enc = new TextEncoder(), dec = new TextDecoder();
function readSmall(h) { const b = new Uint8Array(h.getSize()); h.read(b, { at: 0 }); return b; }
function writeSmall(h, str) { const b = enc.encode(str); h.truncate(0); h.write(b, { at: 0 }); h.flush(); }
// OPFS access handles are exclusive: after a reload the previous page's
// worker may hold them for a moment longer, so wait for it
async function syncHandle(root, name) {
  const fh = await root.getFileHandle(name, { create: true });
  for (let t = 0; ; t++) {
    try { return await fh.createSyncAccessHandle(); } catch (e) {
      if (t >= 40) throw new Error('the game data is in use by another tab of this site (' + e.name + ')');   // browsers name this error differently
      await new Promise((r) => setTimeout(r, 250));
    }
  }
}

// Plays at once: parts already on this device are read from OPFS, the rest is
// streamed (url reader) while the download fills the copy in the background.
// A part the game streams moves to the front of the download queue.
async function openPack(src) {
  let idx = null;
  try {
    const r = await fetch(src.index, { cache: 'no-store' });
    if (r.ok) idx = await r.json();
  } catch (e) {}
  const root = await navigator.storage.getDirectory();
  const mh = await syncHandle(root, 'rom.meta');
  let meta = {};
  try { meta = JSON.parse(dec.decode(readSmall(mh)) || '{}'); } catch (e) {}
  const complete = meta.id && Array.isArray(meta.done) && meta.done.length === meta.n;
  if (!idx && !complete) { mh.close(); return false; }   // no pack on the server and no complete copy: stream
  if (!idx || (complete && meta.id === idx.id)) {   // complete copy (also offline)
    mh.close();
    const h = await syncHandle(root, 'rom.bin');
    size = h.getSize();
    read = (off, len) => { const b = new Uint8Array(Math.max(0, Math.min(len, size - off))); const n = h.read(b, { at: off }); return n === b.length ? b : b.subarray(0, n); };
    postMessage({ type: 'download', done: size, total: size, local: true });
    return true;
  }
  if (meta.id !== idx.id || !Array.isArray(meta.done)) meta = { id: idx.id, size: idx.size, n: idx.parts.length, done: [] };
  const h = await syncHandle(root, 'rom.bin');
  if (h.getSize() !== idx.size) h.truncate(idx.size);
  const done = new Set(meta.done);
  let have = 0;
  for (const i of done) have += idx.parts[i][2];
  const t0 = performance.now(), have0 = have;
  const report = () => postMessage({ type: 'download', done: have, total: idx.size, rate: (have - have0) / Math.max(0.001, (performance.now() - t0) / 1000) });
  report();
  // the streaming reader for what is not here yet
  openUrl(src.url);
  const stream = read;
  size = idx.size;
  const local = (off, len) => { const b = new Uint8Array(Math.max(0, Math.min(len, size - off))); const n = h.read(b, { at: off }); return n === b.length ? b : b.subarray(0, n); };
  const urgent = [];
  const queued = new Set();
  let partial = true;
  read = (off, len) => {
    if (!partial) return local(off, len);
    const i0 = Math.floor(off / idx.chunk), i1 = Math.floor((off + Math.max(1, len) - 1) / idx.chunk);
    let here = true;
    for (let i = i0; i <= i1; i++) if (!done.has(i)) { here = false; if (!queued.has(i)) { queued.add(i); urgent.push(i); } }
    return here ? local(off, len) : stream(off, len);
  };
  let next = 0, failed = null, busy = new Set();
  const order = idx.parts.map((_, i) => i).filter((i) => !done.has(i));
  const pick = () => {
    while (urgent.length) { const i = urgent.shift(); if (!done.has(i) && !busy.has(i)) return i; }
    while (next < order.length) { const i = order[next++]; if (!done.has(i) && !busy.has(i)) return i; }
    return -1;
  };
  const one = async (i) => {
    for (let attempt = 0; ; attempt++) {
      try {
        const r = await fetch(`${src.base}${i}?v=${idx.id}`, { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const b = new Uint8Array(await r.arrayBuffer());   // inflated by the browser (Content-Encoding: br)
        if (b.length !== idx.parts[i][2]) throw new Error(`part ${i}: ${b.length} bytes, expected ${idx.parts[i][2]}`);
        h.write(b, { at: i * idx.chunk });
        return;
      } catch (e) {
        if (attempt >= 4) throw e;
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
      }
    }
  };
  const lane = async () => {
    for (let i; !failed && (i = pick()) >= 0;) {
      busy.add(i);
      try { await one(i); } catch (e) { failed = e; return; } finally { busy.delete(i); }
      h.flush();   // the data is on disk before the index says so
      done.add(i); have += idx.parts[i][2];
      meta.done = [...done];
      writeSmall(mh, JSON.stringify(meta));
      report();
    }
  };
  Promise.all([lane(), lane(), lane(), lane()]).then(() => {
    mh.close();
    if (failed) postMessage({ type: 'download', failed: String(failed && failed.message || failed), partial: true });
    else partial = false;   // everything is local now
  });
  return true;
}

onmessage = async (e) => {
  const { mem, ctl, source } = e.data;
  try {
    if (source.kind === 'file') openFile(source.file);
    else if (source.kind === 'opfs') await openOpfs(source.name);
    else if (source.kind === 'url') openUrl(source.url);
    else if (source.kind === 'pack') {
      let ok = false;
      try { ok = await openPack(source); } catch (err) { postMessage({ type: 'download', failed: String(err && err.message || err) }); }
      if (!ok) openUrl(source.url);   // stream instead
    }
    else throw new Error('unknown ROM source ' + source.kind);
  } catch (err) {
    postMessage({ type: 'error', msg: String(err && err.message || err) });
    return;
  }
  // control block layout (rt/web.cpp RomCtl):
  //   i32[0] state, i32[1] ready, f64[1] size, f64[2] off, i32[6] len, i32[7] dst, i32[8] result
  const i32 = new Int32Array(mem.buffer, ctl, 10);
  const f64 = new Float64Array(mem.buffer, ctl, 3);
  f64[1] = size;
  Atomics.store(i32, 1, 1);
  Atomics.notify(i32, 1);
  postMessage({ type: 'ready', size });
  for (;;) {
    const s = Atomics.load(i32, 0);
    if (s !== 1) {   // idle: let the event loop run (background download)
      if (Atomics.waitAsync) { const w = Atomics.waitAsync(i32, 0, s, 1000); if (w.async) await w.value; }
      else { Atomics.wait(i32, 0, s, 2); await new Promise((r) => setTimeout(r, 0)); }
      continue;
    }
    const off = f64[2], len = i32[6], dst = i32[7] >>> 0;   // wasm addresses above 2 GB are negative as i32
    let n = -1;
    try {
      const data = read(off, len);
      new Uint8Array(mem.buffer).set(data, dst);   // buffer may have grown: take the current one
      n = data.length;
      bytesRead += n; requests++;
    } catch (err) {
      postMessage({ type: 'error', msg: 'ROM read failed: ' + String(err && err.message || err) });
    }
    i32[8] = n;
    Atomics.store(i32, 0, 2);
    Atomics.notify(i32, 0);
    const now = performance.now();
    if (now - lastStats > 500) { lastStats = now; postMessage({ type: 'stats', bytesRead, requests, size, netBytes, netMs, netReqs }); }
  }
};
