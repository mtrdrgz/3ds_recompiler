#!/usr/bin/env python3
"""Pack a ROM for the web build's "download the whole game" mode.

  python3 tools/pack_rom.py "game.3ds" deploy/rom

Writes OUTDIR/rom.pack (the ROM in 8 MiB chunks, each an independent brotli
stream at maximum compression) and OUTDIR/rom.pack.json (the index). The
trailing padding after the last NCCH partition is dropped. The Cloudflare
Worker serves chunk i with Content-Encoding: br, so the browser inflates it
natively; the page stores the ROM in its private storage (OPFS) and plays
from there. Output is derived from the ROM: never commit or publish it.

Needs the brotli command-line tool (brew install brotli / apt install brotli)
or the Python brotli module. ~4 minutes on 10 cores.
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('rom')
ap.add_argument('outdir')
ap.add_argument('--chunk', type=int, default=8, help='chunk size in MiB (default 8)')
ap.add_argument('--quality', type=int, default=11, help='brotli quality 0-11 (default 11)')
ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
args = ap.parse_args()

CHUNK = args.chunk << 20
with open(args.rom, 'rb') as f:
    hdr = f.read(0x200)
    f.seek(0, 2)
    file_size = f.tell()
if hdr[0x100:0x104] != b'NCSD':
    sys.exit('not a .3ds (NCSD) image: ' + args.rom)
end = 0
for i in range(8):
    off, ln = struct.unpack_from('<II', hdr, 0x120 + i * 8)
    if ln: end = max(end, (off + ln) * 0x200)
size = min(end or file_size, file_size)
n = (size + CHUNK - 1) // CHUNK
print(f'ROM {file_size / 2**20:.0f} MiB, data {size / 2**20:.1f} MiB -> {n} chunks of {args.chunk} MiB, brotli q{args.quality}', flush=True)

have_cli = shutil.which('brotli') is not None
if not have_cli:
    try:
        import brotli  # noqa
    except ImportError:
        sys.exit('needs the brotli CLI or the Python brotli module')

def compress(i):
    with open(args.rom, 'rb') as f:
        f.seek(i * CHUNK)
        raw = f.read(min(CHUNK, size - i * CHUNK))
    if have_cli:
        out = subprocess.run(['brotli', '-c', '-q', str(args.quality), '-w', '24'], input=raw, stdout=subprocess.PIPE, check=True).stdout
    else:
        import brotli
        out = brotli.compress(raw, quality=args.quality, lgwin=24)
    return len(raw), out

os.makedirs(args.outdir, exist_ok=True)
pack_path = os.path.join(args.outdir, 'rom.pack')
parts, off, done, t0 = [], 0, 0, time.time()
h = hashlib.sha256()
with open(pack_path + '.tmp', 'wb') as out, ThreadPoolExecutor(args.jobs) as ex:
    # map() yields in order, so chunks are written sequentially while the pool keeps working
    for i, (rawlen, comp) in enumerate(ex.map(compress, range(n))):
        out.write(comp); h.update(comp)
        parts.append([off, len(comp), rawlen])
        off += len(comp); done += rawlen
        el = time.time() - t0
        print(f'  chunk {i + 1}/{n}  {done / 2**20:.0f}/{size / 2**20:.0f} MiB  ratio {off / done:.3f}  {el:.0f}s', flush=True)
os.replace(pack_path + '.tmp', pack_path)
index = {'format': 'r3ds-rom-pack-1', 'id': h.hexdigest()[:16], 'size': size, 'chunk': CHUNK, 'parts': parts}
with open(os.path.join(args.outdir, 'rom.pack.json'), 'w') as f:
    json.dump(index, f, separators=(',', ':'))
print(f'wrote {pack_path}: {off / 2**20:.1f} MiB for {size / 2**20:.1f} MiB ({off / size:.3f}), id {index["id"]}', flush=True)
