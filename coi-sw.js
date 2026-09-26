// Cross-origin isolation for static hosts (GitHub Pages, S3, ...): the
// recompilation needs SharedArrayBuffer (threads), which browsers only
// enable when the page is served with COOP/COEP headers. This service
// worker re-serves same-origin responses with those headers added.
// Servers that already send them (tools/serve_web.py) do not need it.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.headers.has('range')) return;   // let ROM streaming go straight to the network
  e.respondWith((async () => {
    const r = await fetch(req);
    if (r.status === 0) return r;
    const h = new Headers(r.headers);
    h.set('Cross-Origin-Embedder-Policy', 'require-corp');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    h.set('Cross-Origin-Resource-Policy', 'same-origin');
    return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
  })());
});
