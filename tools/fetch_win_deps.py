#!/usr/bin/env python3
"""Download the Windows (x64, MinGW) build dependencies into winsdk/:
SDL2 (mingw devel), and with --unicorn also Unicorn (DLL + headers, from the
official PyPI wheel; only for -DFL_UNICORN=ON debugging builds).
Pure standard library, works from any Python 3."""
import io, json, os, sys, tarfile, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'winsdk')
UNICORN = '2.1.4'
SDL = '2.30.9'

def get(url):
    print('fetch', url, file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        return r.read()

def main():
    os.makedirs(OUT, exist_ok=True)
    if '--unicorn' in sys.argv:
        fetch_unicorn()
    t = tarfile.open(fileobj=io.BytesIO(get(
        'https://github.com/libsdl-org/SDL/releases/download/release-%s/SDL2-devel-%s-mingw.tar.gz' % (SDL, SDL))))
    t.extractall(OUT)
    print('ok: %s' % OUT, file=sys.stderr)
    print('cmake flag: -DSDL2_DIR=%s' % os.path.join(OUT, 'SDL2-%s' % SDL, 'x86_64-w64-mingw32', 'lib', 'cmake', 'SDL2'))

def fetch_unicorn():
    meta = json.loads(get('https://pypi.org/pypi/unicorn/%s/json' % UNICORN))
    whl = next(u['url'] for u in meta['urls'] if u['filename'].endswith('win_amd64.whl'))
    z = zipfile.ZipFile(io.BytesIO(get(whl)))
    for n in z.namelist():
        if (n.startswith('unicorn/lib/') or n.startswith('unicorn/include/')) and not n.endswith('/'):
            dst = os.path.join(OUT, n)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'wb').write(z.read(n))
    print('unicorn: -DFL_UNICORN=ON -DUNICORN_ROOT=%s' % os.path.join(OUT, 'unicorn'), file=sys.stderr)

if __name__ == '__main__':
    main()
