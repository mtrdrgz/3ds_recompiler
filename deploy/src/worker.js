// Cloudflare Worker: serves the web build and the ROM from a PRIVATE Hugging
// Face dataset, behind an access key, with the headers the page needs.
//
//   dataset layout:  rom.3ds, rom.pack, rom.pack.json (tools/pack_rom.py),
//                    web/index.html, web/app.js, web/recomp3ds.wasm, ... and
//                    web/<file>.br (brotli; served pre-compressed when present)
//   secrets:         HF_TOKEN   (read access to the dataset)
//                    ACCESS_KEY (open https://<host>/?key=<ACCESS_KEY> once per device)
//   vars:            HF_REPO    (e.g. you/your-private-dataset)
//
// Without ACCESS_KEY or HF_TOKEN nothing is served (fail closed): the ROM and
// the code generated from it must not be public.

const FILES = new Set(['index.html', 'app.js', 'compositor.js', 'rom_worker.js', 'audio_worklet.js', 'coi-sw.js', 'recomp3ds.js', 'recomp3ds.wasm', 'default-save.json', 'manifest.webmanifest', 'icon.png', 'textures.json', 'textures.pack']);
const TYPES = { html: 'text/html; charset=utf-8', js: 'text/javascript', json: 'application/json', webmanifest: 'application/manifest+json', png: 'image/png', pack: 'application/octet-stream', wasm: 'application/wasm', '3ds': 'application/octet-stream' };
const COOKIE = 'r3ds_access';
const BLOCK = 1 << 20;   // rom_worker.js reads the ROM in 1 MiB blocks

async function sha256(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
function isolation(h) {
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  h.set('X-Robots-Tag', 'noindex, nofollow');
  return h;
}
function text(status, body) {
  return new Response(body, { status, headers: isolation(new Headers({ 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' })) });
}
function upstream(env, path) {
  return `https://huggingface.co/datasets/${env.HF_REPO}/resolve/main/${path}`;
}

// follow redirects by hand: the token goes to huggingface.co only, never to
// the storage CDN it redirects to (signed URLs refuse a second credential)
async function fromHub(env, path, range) {
  let url = new URL(upstream(env, path));
  for (let hop = 0; hop < 5; hop++) {
    const h = new Headers();
    if (range) h.set('Range', range);
    if (url.hostname === 'huggingface.co') h.set('Authorization', `Bearer ${env.HF_TOKEN}`);
    const r = await fetch(url, { headers: h, redirect: 'manual' });
    const loc = r.headers.get('Location');
    if (r.status < 300 || r.status >= 400 || !loc) return r;
    url = new URL(loc, url);
  }
  return new Response('too many redirects', { status: 508 });
}

// ---- ROM
// The dataset file is resolved once to its signed CDN URL (reused for a few
// minutes), so a block costs one CDN range request instead of a Hub API call
// plus a redirect. The edge caches 4 MiB superblocks; a miss also prefetches
// the next one, so sequential reads are served from Cloudflare's cache.
const SUPER = 4 << 20;
const MAX_REQ = 8 << 20;
const signed = new Map();      // dataset path -> { url, auth, exp } for this isolate

async function resolveRom(env, force, name = 'rom.3ds') {
  const s0 = signed.get(name);
  if (!force && s0 && s0.exp > Date.now()) return s0;
  let url = new URL(upstream(env, name));
  let out = null;
  for (let hop = 0; hop < 5 && !out; hop++) {
    if (url.hostname !== 'huggingface.co') { out = { url: url.href, auth: false }; break; }
    const r = await fetch(url, { method: 'HEAD', headers: { Authorization: `Bearer ${env.HF_TOKEN}` }, redirect: 'manual' });
    const loc = r.headers.get('Location');
    if (r.status >= 300 && r.status < 400 && loc) { url = new URL(loc, url); continue; }
    if (!r.ok) throw new Error(`resolve: HTTP ${r.status}`);
    out = { url: url.href, auth: true };   // served by the Hub itself
  }
  if (!out) throw new Error('resolve: too many redirects');
  const s1 = { ...out, exp: Date.now() + 10 * 60 * 1000 };
  signed.set(name, s1);
  return s1;
}
async function romRange(env, a, b, name = 'rom.3ds') {
  for (let attempt = 0; attempt < 2; attempt++) {
    const s = await resolveRom(env, attempt > 0, name);
    const h = { Range: `bytes=${a}-${b}` };
    if (s.auth) h.Authorization = `Bearer ${env.HF_TOKEN}`;
    const r = await fetch(s.url, { headers: h });
    if (r.status === 206 || r.status === 200) return r;
    if (attempt === 0 && [400, 401, 403, 404, 410].includes(r.status)) continue;   // signed URL expired
    throw new Error(`rom: HTTP ${r.status}`);
  }
}
function superKey(env, k) { return new Request(`https://rom-cache.invalid/${env.HF_REPO}/s${SUPER}/${k}`); }
async function superblock(env, ctx, k) {
  const hit = await caches.default.match(superKey(env, k));
  if (hit) return { buf: await hit.arrayBuffer(), total: hit.headers.get('X-Total') };
  const r = await romRange(env, k * SUPER, (k + 1) * SUPER - 1);
  const cr = /\/(\d+)$/.exec(r.headers.get('Content-Range') || '');
  const buf = await r.arrayBuffer();
  const total = cr ? cr[1] : String(buf.byteLength);
  ctx.waitUntil(caches.default.put(superKey(env, k),
    new Response(buf.slice(0), { headers: { 'X-Total': total, 'Cache-Control': 'public, max-age=2592000' } })));
  return { buf, total };
}
async function prefetch(env, ctx, k, total) {
  if (total && k * SUPER >= Number(total)) return;
  if (await caches.default.match(superKey(env, k))) return;
  try { await superblock(env, ctx, k); } catch (e) {}
}

async function rom(request, env, ctx) {
  const m = /^bytes=(\d+)-(\d*)$/.exec(request.headers.get('Range') || '');
  if (!m) return text(416, 'byte range required');
  const a = Number(m[1]);
  let b = m[2] === '' ? a + (1 << 20) - 1 : Number(m[2]);
  b = Math.min(b, a + MAX_REQ - 1);
  const k0 = Math.floor(a / SUPER), k1 = Math.floor(b / SUPER);
  let parts;
  try {
    parts = await Promise.all(Array.from({ length: k1 - k0 + 1 }, (_, i) => superblock(env, ctx, k0 + i)));
  } catch (e) {
    return text(502, String(e && e.message || e));
  }
  const total = parts[0].total;
  const end = Math.min(b, Number(total) - 1);
  if (a > end) return new Response(null, { status: 416, headers: isolation(new Headers({ 'Content-Range': `bytes */${total}` })) });
  // read-ahead: the next superblock, while the client consumes this one
  ctx.waitUntil(prefetch(env, ctx, k1 + 1, total));
  const views = [];
  for (let i = 0; i < parts.length; i++) {
    const base = (k0 + i) * SUPER;
    const from = Math.max(a, base) - base, to = Math.min(end + 1, base + parts[i].buf.byteLength) - base;
    if (to > from) views.push(new Uint8Array(parts[i].buf, from, to - from));
  }
  const len = views.reduce((n, v) => n + v.length, 0);
  const body = new ReadableStream({ start(c) { for (const v of views) c.enqueue(v); c.close(); } });
  return new Response(body, {
    status: 206,
    headers: isolation(new Headers({
      'Content-Type': TYPES['3ds'], 'Content-Range': `bytes ${a}-${a + len - 1}/${total}`, 'Accept-Ranges': 'bytes',
      'Content-Length': String(len), 'Cache-Control': 'private, no-store',
    })),
  });
}

// ---- whole-ROM download (tools/pack_rom.py)
// rom.pack.json: { id, size, chunk, parts: [[offset, length, raw length]] };
// /rompack/<i>?v=<id>: part i of rom.pack, a brotli stream, sent with
// Content-Encoding: br so the browser inflates it natively. Parts are cached
// at the edge under the pack id, so a new pack never mixes with an old one.
let packIndex = null;   // { json, exp }
async function packJson(env, force) {
  if (!force && packIndex && packIndex.exp > Date.now()) return packIndex.json;
  const r = await fromHub(env, 'rom.pack.json');
  if (!r.ok) return null;
  const json = await r.text();
  packIndex = { json, exp: Date.now() + 60 * 1000 };
  return json;
}
async function packPart(request, env, ctx, i) {
  const v = new URL(request.url).searchParams.get('v') || '';
  const key = new Request(`https://pack-cache.invalid/${env.HF_REPO}/${v}/${i}`);
  const h = isolation(new Headers({ 'Content-Type': 'application/octet-stream', 'Content-Encoding': 'br', 'Cache-Control': 'private, no-store' }));
  const hit = await caches.default.match(key);
  if (hit) return new Response(hit.body, { status: 200, headers: h, encodeBody: 'manual' });
  let json = await packJson(env, false);
  let idx = json && JSON.parse(json);
  if (idx && idx.id !== v) { json = await packJson(env, true); idx = json && JSON.parse(json); }
  if (!idx) return text(404, 'no rom.pack in the dataset');
  if (idx.id !== v) return text(409, 'the pack changed: reload the page');
  const p = idx.parts[i];
  if (!p) return text(404, 'no such part');
  let r;
  try { r = await romRange(env, p[0], p[0] + p[1] - 1, 'rom.pack'); } catch (e) { return text(502, String(e && e.message || e)); }
  const buf = await r.arrayBuffer();
  if (buf.byteLength !== p[1]) return text(502, `short part ${i}: ${buf.byteLength} of ${p[1]} bytes`);
  ctx.waitUntil(caches.default.put(key, new Response(buf.slice(0), { headers: { 'Cache-Control': 'public, max-age=2592000' } })));
  return new Response(buf, { status: 200, headers: h, encodeBody: 'manual' });
}

async function asset(request, env, ctx, name) {
  const cache = caches.default;
  const key = new Request(`https://asset-cache.invalid/${env.HF_REPO}/${env.BUILD || 'web'}/${name}`);
  const ext = name.split('.').pop();
  const h = isolation(new Headers({ 'Content-Type': TYPES[ext] || 'application/octet-stream', 'Cache-Control': 'private, no-cache' }));
  // the browser revalidates with the ETag: an unchanged 30 MB wasm is a 304
  const hit = await cache.match(key);
  const etag = hit && hit.headers.get('ETag');
  if (etag) h.set('ETag', etag);
  if (etag && request.headers.get('If-None-Match') === etag) return new Response(null, { status: 304, headers: h });
  if (hit) {
    const br = hit.headers.get('X-Enc') === 'br';
    if (br) h.set('Content-Encoding', 'br');
    return new Response(hit.body, { status: 200, headers: h, encodeBody: br ? 'manual' : 'automatic' });
  }
  // the brotli copy (web/<name>.br) when there is one: the wasm is ~5x smaller
  let br = true;
  let u = await fromHub(env, `${env.BUILD || 'web'}/${name}.br`);
  if (u.status === 404) { br = false; u = await fromHub(env, `${env.BUILD || 'web'}/${name}`); }
  if (!u.ok) return text(502, `upstream HTTP ${u.status} for ${name}`);
  if (br) h.set('Content-Encoding', 'br');
  const up = u.headers.get('X-Linked-Etag') || u.headers.get('ETag');
  const tag = up ? (up.startsWith('"') || up.startsWith('W/') ? up : `"${up}"`) : null;
  if (tag) h.set('ETag', tag);
  // streamed, not buffered (the wasm is 30 MB); a copy goes to the edge cache
  // for ten minutes, so a new upload shows up quickly
  const [mine, cached] = u.body.tee();
  const ch = new Headers({ 'Cache-Control': 'public, max-age=600' });
  if (tag) ch.set('ETag', tag);
  if (br) ch.set('X-Enc', 'br');   // stored compressed; the encoding is re-declared when served
  ctx.waitUntil(cache.put(key, new Response(cached, { headers: ch })));
  return new Response(mine, { status: 200, headers: h, encodeBody: br ? 'manual' : 'automatic' });
}

export class MultiplayerRoom {
  constructor(state) {
    this.state = state;
    this.sockets = new Map();
  }

  ids() {
    return [...this.sockets.values()];
  }

  notify() {
    const message = JSON.stringify({ type: 'peers', peers: this.ids() });
    for (const socket of this.sockets.keys()) {
      try { socket.send(message); } catch (e) {}
    }
  }

  async fetch(request) {
    if (request.headers.get('Upgrade') !== 'websocket') return text(426, 'websocket required');
    if (this.sockets.size >= 3) return text(503, 'room full');
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    server.accept();
    const used = new Set(this.sockets.values());
    let id = 1;
    while (used.has(id)) id++;
    this.sockets.set(server, id);
    const remove = () => {
      if (!this.sockets.delete(server)) return;
      try { server.close(); } catch (e) {}
      this.notify();
    };
    server.addEventListener('message', async (event) => {
      if (typeof event.data === 'string') return;
      let payload;
      if (event.data instanceof ArrayBuffer) payload = new Uint8Array(event.data);
      else if (ArrayBuffer.isView(event.data)) payload = new Uint8Array(event.data.buffer, event.data.byteOffset, event.data.byteLength);
      else if (event.data && event.data.arrayBuffer) payload = new Uint8Array(await event.data.arrayBuffer());
      else return;
      if (!payload.length || payload.length > 0x39c) return;
      const packet = new Uint8Array(payload.length + 1);
      packet[0] = id;
      packet.set(payload, 1);
      for (const socket of this.sockets.keys()) if (socket !== server) {
        try { socket.send(packet); } catch (e) {}
      }
    });
    server.addEventListener('close', remove);
    server.addEventListener('error', remove);
    server.send(JSON.stringify({ type: 'welcome', id, peers: this.ids() }));
    this.notify();
    return new Response(null, { status: 101, webSocket: client });
  }
}

export default {
  async fetch(request, env, ctx) {
    if (!env.ACCESS_KEY || !env.HF_TOKEN || !env.HF_REPO) return text(503, 'not configured');
    if (request.method !== 'GET' && request.method !== 'HEAD') return text(405, 'method not allowed');
    const url = new URL(request.url);
    const want = await sha256(env.ACCESS_KEY);
    // ?key=... once per device: remember it in a cookie and drop it from the URL
    const k = url.searchParams.get('key');
    if (k !== null) {
      if ((await sha256(k)) !== want) return text(403, 'wrong key');
      url.searchParams.delete('key');
      return new Response(null, {
        status: 302,
        headers: {
          Location: url.pathname + (url.search || ''),
          'Set-Cookie': `${COOKIE}=${want}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Strict`,
          'Cache-Control': 'no-store',
        },
      });
    }
    const cookie = (request.headers.get('Cookie') || '').split(/;\s*/).find((c) => c.startsWith(COOKIE + '='));
    if (!cookie || cookie.slice(COOKIE.length + 1) !== want) return text(401, 'access key required: open this site once with ?key=<your key>');

    const path = url.pathname.replace(/^\/+/, '') || 'index.html';
    if (path === 'multiplayer') {
      if (request.headers.get('Upgrade') !== 'websocket') return text(426, 'websocket required');
      if (!env.MULTIPLAYER_ROOM) return text(503, 'multiplayer not configured');
      const room = (url.searchParams.get('room') || 'world').slice(0, 64);
      return env.MULTIPLAYER_ROOM.get(env.MULTIPLAYER_ROOM.idFromName(room)).fetch(request);
    }
    if (path === 'rom.3ds') return rom(request, env, ctx);
    if (path === 'rom.pack.json') {
      const j = await packJson(env, true);
      if (!j) return text(404, 'no rom.pack in the dataset');
      return new Response(j, { headers: isolation(new Headers({ 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })) });
    }
    const pm = /^rompack\/(\d+)$/.exec(path);
    if (pm) return packPart(request, env, ctx, Number(pm[1]));
    if (FILES.has(path)) return asset(request, env, ctx, path);
    return text(404, 'not found');
  },
};
