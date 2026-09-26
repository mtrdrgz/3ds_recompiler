#!/usr/bin/env python3
"""Bundle a web build into a single offline HTML file.

Reads build-web/recomp3ds.js + recomp3ds.wasm and the web/ page sources, and
emits one .html that carries every payload inline:

  - app.js / compositor.js     -> classic <script> elements
  - recomp3ds.js / .wasm,
    rom_worker.js,
    audio_worklet.js           -> <script type="text/plain" data-r3ds=...>
                                 blocks holding base64; web/app.js decodes
                                 them into blob URLs (see the EMBED helpers)
  - icon.png                   -> data: URIs

The result still needs cross-origin isolation (SharedArrayBuffer for the
threaded runtime), which file:// can never provide — serve it with
tools/serve_web.py or any host that sends COOP/COEP.

The interpreter-only build (the default for this file) contains no
game-derived bytes, so it can be published. If you bundle a build made with
tools/build_web.sh <rom> (R3DS_LIFTED=ON, gen/ present) the wasm embeds code
lifted from that ROM — do not distribute that file.
"""

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def inline_script(src: str, name: str) -> str:
    if '</script' in src.lower():
        sys.exit(f'{name}: contains "</script" — cannot be inlined safely')
    return f'<script>\n{src}\n</script>'


def payload(name: str, data: bytes) -> str:
    return f'<script type="text/plain" data-r3ds="{name}">{b64(data)}</script>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('-b', '--build-dir', default=str(ROOT / 'build-web'),
                    help='directory holding recomp3ds.js/.wasm (default: build-web)')
    ap.add_argument('-o', '--output', default=None,
                    help='output .html path (default: <build-dir>/recomp3ds.html)')
    args = ap.parse_args()

    build = Path(args.build_dir)
    web = ROOT / 'web'
    mod_js = build / 'recomp3ds.js'
    wasm = build / 'recomp3ds.wasm'
    for f in (mod_js, wasm, web / 'index.html', web / 'app.js',
              web / 'compositor.js', web / 'rom_worker.js', web / 'audio_worklet.js'):
        if not f.exists():
            sys.exit(f'missing {f} — build the web target first (tools/build_web.sh, or cmake without gen/ for an interpreter-only build)')

    html = (web / 'index.html').read_text(encoding='utf-8')
    icon = b64((web / 'icon.png').read_bytes()) if (web / 'icon.png').exists() else ''

    # page resources -> inline / data URIs
    html = html.replace('<link rel="manifest" href="manifest.webmanifest" crossorigin="use-credentials">', '')
    if icon:
        html = html.replace('<link rel="apple-touch-icon" href="icon.png">',
                            f'<link rel="apple-touch-icon" href="data:image/png;base64,{icon}">\n'
                            f'<link rel="icon" href="data:image/png;base64,{icon}">')
    html = html.replace('<script src="compositor.js"></script>',
                        inline_script((web / 'compositor.js').read_text(encoding='utf-8'), 'compositor.js'))
    html = html.replace('<script src="app.js"></script>',
                        payload('mod', mod_js.read_bytes()) + '\n' +
                        payload('wasm', wasm.read_bytes()) + '\n' +
                        payload('romWorker', (web / 'rom_worker.js').read_bytes()) + '\n' +
                        payload('audioWorklet', (web / 'audio_worklet.js').read_bytes()) + '\n' +
                        inline_script((web / 'app.js').read_text(encoding='utf-8'), 'app.js'))
    if '<script src=' in html:
        sys.exit('index.html has a <script src=> the bundler does not know')

    out = Path(args.output) if args.output else build / 'recomp3ds.html'
    out.write_text(html, encoding='utf-8')
    size = out.stat().st_size
    print(f'wrote {out} ({size / 1048576:.1f} MiB)')
    print('note: serve it, file:// cannot provide SharedArrayBuffer —')
    print('  python3 tools/serve_web.py   then open http://localhost:8080/recomp3ds.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
