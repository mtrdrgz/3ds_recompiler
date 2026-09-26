#!/usr/bin/env python3
"""Serve the web build locally with the headers it needs.

  python3 tools/serve_web.py [--dir build-web] [--port 8080]
                                     [--rom "game.3ds"] [--host 127.0.0.1]

* sends Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy, so the
  page is cross-origin isolated (SharedArrayBuffer -> threads);
* supports HTTP Range requests, so a ROM can be streamed from it;
* --rom exposes that file at /rom.3ds: open
  http://localhost:8080/?rom=rom.3ds  to stream it without picking a file.
  Only do that on your own machine / network: it serves your ROM.
* --pack DIR serves tools/pack_rom.py output (rom.pack.json, /rompack/<i>
  with Content-Encoding: br) like the Cloudflare Worker, for the page's
  "download all once" mode (?romMode=download).
"""
import argparse, http.server, os, re, socketserver, sys

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument('--dir', default=os.path.join(os.path.dirname(here), 'build-web'))
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--rom', default=None)
ap.add_argument('--pack', default=None)
args = ap.parse_args()
if not os.path.exists(os.path.join(args.dir, 'recomp3ds.wasm')):
    sys.exit('%s has no recomp3ds.wasm: build the web target first (tools/build_web.sh)' % args.dir)
ROM = os.path.abspath(args.rom) if args.rom else None

class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.wasm': 'application/wasm', '.js': 'text/javascript', '.3ds': 'application/octet-stream'}

    def __init__(self, *a, **k):
        super().__init__(*a, directory=args.dir, **k)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def translate_path(self, path):
        if ROM and path.split('?')[0] == '/rom.3ds':
            return ROM
        return super().translate_path(path)

    def send_head(self):
        if args.pack:
            p = self.path.split('?')[0]
            m = re.match(r'/rompack/(\d+)$', p)
            if p == '/rom.pack.json' or m:
                import io, json
                if p == '/rom.pack.json':
                    body = open(os.path.join(args.pack, 'rom.pack.json'), 'rb').read(); enc = None
                else:
                    part = json.load(open(os.path.join(args.pack, 'rom.pack.json')))['parts'][int(m.group(1))]
                    with open(os.path.join(args.pack, 'rom.pack'), 'rb') as f:
                        f.seek(part[0]); body = f.read(part[1])
                    enc = 'br'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json' if not enc else 'application/octet-stream')
                if enc: self.send_header('Content-Encoding', enc)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                return io.BytesIO(body)
        path = self.translate_path(self.path)
        rng = self.headers.get('Range')
        if not rng or os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip())
        size = os.path.getsize(path)
        if not m:
            self.send_error(416); return None
        a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        b = min(b, size - 1)
        if a > b or a >= size:
            self.send_response(416); self.send_header('Content-Range', 'bytes */%d' % size); self.end_headers(); return None
        f = open(path, 'rb'); f.seek(a)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', 'bytes %d-%d/%d' % (a, b, size))
        self.send_header('Content-Length', str(b - a + 1))
        self.end_headers()
        self._remaining = b - a + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, '_remaining', None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            chunk = src.read(min(n, 1 << 20))
            if not chunk: break
            dst.write(chunk); n -= len(chunk)
        self._remaining = None

    def log_message(self, fmt, *a):
        if '/rom.3ds' not in (a[0] if a else ''):
            sys.stderr.write('%s\n' % (fmt % a))

class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

print('serving %s at http://%s:%d/%s' % (args.dir, args.host, args.port, '  (ROM at /rom.3ds -> open ?rom=rom.3ds)' if ROM else ''))
S((args.host, args.port), H).serve_forever()
