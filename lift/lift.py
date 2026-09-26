#!/usr/bin/env python3
"""r3ds static recompiler: 3DS code.bin -> C++ (gen/).

  lift.py [--stats] [--max-files N]

Discovery is a recursive descent from the entry point, every function
pointer found in .rodata/.data and in literal pools, jump tables, the
seed list (lift/seeds.txt), and the runtime miss log (addresses the
dispatcher had to hand to the interpreter).

The image layout comes from the ROM's exheader, parsed by
scripts/extract_rom.py into extracted/manifest.json.
"""
import os, sys, json, struct, time, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from armdec import dec_arm, dec_thumb, COND_EXPR, hx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEBIN = os.environ.get('R3DS_CODEBIN') or os.path.join(ROOT, 'extracted', 'code.bin')
MANIFEST = os.environ.get('R3DS_MANIFEST') or os.path.join(ROOT, 'extracted', 'manifest.json')
GEN = os.path.join(ROOT, 'gen')

def load_layout(path):
    """code.bin layout from extracted/manifest.json (exheader CodeSetInfo).

    text/rodata regions are page granular; data's file size is exact and bss
    follows it. The entry point is the .text address.
    """
    m = json.load(open(path))
    t, r, d = m['text'], m['rodata'], m['data']
    text_end = r['addr'] or t['addr'] + t['pages'] * 0x1000
    ro_end = d['addr'] or r['addr'] + r['pages'] * 0x1000
    data_end = (d['addr'] or ro_end) + d['size']
    img_end = (data_end + m['bss_size'] + 0xFFF) & ~0xFFF
    return t['addr'], text_end, ro_end, data_end, img_end

BASE, TEXT_END, RO_END, DATA_END, IMG_END = load_layout(MANIFEST)

code = open(CODEBIN, 'rb').read()
def w32(a): return struct.unpack_from('<I', code, a - BASE)[0]
def w16(a): return struct.unpack_from('<H', code, a - BASE)[0]

def log(*a):
    print('[lift %6.1fs]' % (time.time() - T0), *a, file=sys.stderr, flush=True)
T0 = time.time()

insns = {}          # pc (bit0=thumb) -> Insn
entries = set()     # pcs that must be dispatchable
func_starts = set() # call targets / pointer seeds (preferred chunk starts)

def decode(pc):
    a = pc & ~1
    if a < BASE or a >= TEXT_END: return None
    if pc & 1:
        h = w16(a)
        h2 = w16(a + 2) if a + 2 < TEXT_END else None
        return dec_thumb(a, h, h2)
    if a & 3: return None
    return dec_arm(a, w32(a))

def valid_seed(pc, n=3):
    cur = pc
    for _ in range(n):
        i = decode(cur)
        if i is None or i.bad: return False
        if not i.falls: return True
        cur = (cur & 1) | ((cur & ~1) + i.size)
    return True

def discover(seeds):
    work = list(seeds)
    nbad = 0
    while work:
        pc = work.pop()
        if pc in insns: continue
        cur = pc
        while True:
            if cur in insns: break
            i = decode(cur)
            if i is None: break
            insns[cur] = i
            for t in i.targets:
                entries.add(t)
                if t not in insns: work.append(t)
                if i.call: func_starts.add(t)
            nxt = (cur & 1) | ((cur & ~1) + i.size)
            if i.call:
                entries.add(nxt)
            for lit in i.ptr_loads:
                if BASE <= lit < DATA_END - 3:
                    v = w32(lit)
                    if BASE <= v < TEXT_END and (v & 1 or not v & 3) and v not in insns and valid_seed(v):
                        entries.add(v); func_starts.add(v); work.append(v)
            if i.table:
                handle_table(cur, i, work)
            if i.bad:
                nbad += 1
                break
            if not i.falls: break
            cur = nxt
    return nbad

def handle_table(pc, i, work):
    kind, idx = i.table
    # bound from a preceding cmp rIdx, #n (ARM: cmp encoded cond=AL, opc=0xA, I=1)
    bound = None
    for back in (4, 8, 12):
        p = pc - back
        if p < BASE: break
        w = w32(p)
        if (w & 0x0FF00000) == 0x03500000 and ((w >> 16) & 0xF) == idx:
            from armdec import ror
            bound = ror(w & 0xFF, ((w >> 8) & 0xF) * 2)
            break
    if bound is None or bound > 512: return
    if kind == 'addpc':
        base = (pc & ~1) + 8
        for k in range(bound + 1):
            t = base + 4 * k
            entries.add(t); work.append(t)
        if True:
            t = base - 4   # default case: instruction right after the add (at pc+4)
            entries.add(t); work.append(t)
    else:  # ldr pc, [pc, rX, lsl #2]: table at pc+8
        base = (pc & ~1) + 8
        for k in range(bound + 1):
            v = w32(base + 4 * k)
            if BASE <= v < TEXT_END:
                entries.add(v); work.append(v)

def read_seed_file(path):
    out = []
    if not os.path.exists(path): return out
    for line in open(path):
        line = line.strip().split('#')[0].split()
        if not line: continue
        try:
            v = int(line[0], 16)
        except ValueError:
            continue
        out.append(v)
    return out

def gather_seeds():
    seeds = [BASE]
    # pointer scan over rodata + data
    n = 0
    for a in range(TEXT_END, DATA_END - 3, 4):
        v = w32(a)
        if BASE <= v < TEXT_END and (v & 1 or not v & 3):
            if valid_seed(v):
                seeds.append(v); n += 1
    log('pointer seeds from data: %d' % n)
    extra = 0
    for f in ('miss.log', 'lift/seeds.txt'):
        for v in read_seed_file(os.path.join(ROOT, f)):
            if BASE <= v < TEXT_END and valid_seed(v, 1):
                seeds.append(v); extra += 1
    log('seeds from miss log / seed file: %d' % extra)
    for s in seeds:
        entries.add(s); func_starts.add(s)
    return seeds

# ------------------------------------------------------------------ chunking
MAX_CHUNK = 600
SOFT_CHUNK = 120

def make_chunks():
    pcs = sorted(insns, key=lambda p: (p & ~1, p & 1))
    chunks = []
    cur = []
    prev_end = None; prev_mode = None
    for pc in pcs:
        i = insns[pc]
        a = pc & ~1; mode = pc & 1
        contiguous = (prev_end == a and prev_mode == mode)
        start_new = (not contiguous) or len(cur) >= MAX_CHUNK or (len(cur) >= SOFT_CHUNK and pc in func_starts)
        if start_new and cur:
            chunks.append(cur); cur = []
        cur.append(pc)
        prev_end = a + i.size; prev_mode = mode
    if cur: chunks.append(cur)
    return chunks

# ------------------------------------------------------------------ emission
def lbl(pc):
    return 'L_%s%x' % ('t' if pc & 1 else 'a', pc & ~1)

def lower(body, chunkset, pc, used):
    """replace JMP(x)/EXIT(e)/SVC/TRAP placeholders"""
    out = body
    # JMP(0x...u)
    while 'JMP(' in out:
        k = out.index('JMP(')
        e = out.index(')', k)
        t = int(out[k + 4:e].rstrip('u'), 16)
        if t in chunkset:
            used.add(t)
            if (t & ~1) <= (pc & ~1):
                rep = 'POLL(%s); goto %s;' % (hx(t), lbl(t))
            else:
                rep = 'goto %s;' % lbl(t)
        else:
            rep = 'EXIT(%s);' % hx(t)
        # swallow a following ';'
        end = e + 1
        if end < len(out) and out[end] == ';': end += 1
        out = out[:k] + rep + out[end:]
    return out

def chunk_name(chunk):
    pc = chunk[0]
    return 'CH_%s%x' % ('t' if pc & 1 else 'a', pc & ~1)

def emit_chunk(cid, chunk, f):
    chunkset = set(chunk)
    body_lines = []
    used = set()
    for idx, pc in enumerate(chunk):
        i = insns[pc]
        if i.bad:
            code_s = 'FALLBACK(%s);' % hx(pc)
        else:
            code_s = lower(i.body, chunkset, pc, used)
            if i.cond != 14:
                code_s = 'if (%s) { %s }' % (COND_EXPR[i.cond], code_s)
            else:
                code_s = '{ %s }' % code_s
        body_lines.append((pc, i, code_s))
    ents = [pc for pc in chunk if pc in entries or pc == chunk[0]]
    for e in ents: used.add(e)
    f.write('static u32 %s(Cpu &c, u32 pc) {\n  PROLOGUE\n  switch (pc) {\n' % chunk_name(chunk))
    for e in ents:
        f.write('  case %s: goto %s;\n' % (hx(e), lbl(e)))
    f.write('  default: MISS(pc);\n  }\n')
    for pc, i, code_s in body_lines:
        lab = (lbl(pc) + ': ') if pc in used else ''
        f.write('%s/*%x %s*/ %s\n' % (lab, pc & ~1, i.text, code_s))
    last = chunk[-1]
    nxt = (last & 1) | ((last & ~1) + insns[last].size)
    if insns[last].falls:
        f.write('  EXIT(%s);\n' % hx(nxt))
    f.write('  OUT\n}\n')
    return ents

def write_if_changed(path, text):
    if os.path.exists(path) and open(path).read() == text: return False
    open(path, 'w').write(text)
    return True

def main():
    stats_only = '--stats' in sys.argv
    seeds = gather_seeds()
    log('discovering...')
    nbad = discover(seeds)
    na = sum(1 for p in insns if not p & 1); nt = len(insns) - na
    log('decoded %d instructions (%d ARM, %d Thumb), %d entries, %d bad stops' % (len(insns), na, nt, len(entries), nbad))
    bads = collections.Counter(i.bad for i in insns.values() if i.bad)
    for k, v in bads.most_common(15): log('  bad %6d %s' % (v, k))
    chunks = make_chunks()
    log('%d chunks' % len(chunks))
    if stats_only: return
    os.makedirs(GEN, exist_ok=True)
    # stable file assignment: one file per 16KB of .text (by chunk start)
    BUCKET = 0x4000
    buckets = collections.OrderedDict()
    for ch in chunks:
        b = ((ch[0] & ~1) - BASE) // BUCKET
        buckets.setdefault(b, []).append(ch)
    files = []
    nchanged = 0
    import io
    for b, group in buckets.items():
        name = 'lifted_%04d.cpp' % b
        buf = io.StringIO()
        buf.write('// generated by lift/lift.py -- do not edit\n#include "lift_rt.h"\n\n')
        regs = []
        for ch in group:
            ents = emit_chunk(0, ch, buf)
            regs.append((chunk_name(ch), ents))
        buf.write('void reg_%04d() {\n' % b)
        for nm, ents in regs:
            for e in ents:
                buf.write('  disp_register(%s, %s);\n' % (hx(e), nm))
        buf.write('}\n')
        if write_if_changed(os.path.join(GEN, name), buf.getvalue()): nchanged += 1
        files.append(name)
    nf = len(files)
    ids = list(buckets.keys())
    h = 2166136261
    for b in code:
        h = ((h ^ b) * 16777619) & 0xffffffff
    reg = ['// generated', '#include "lift_rt.h"',
           '// FNV-1a of the code.bin this was lifted from; checked at boot against the ROM',
           'extern const u32 g_lifted_code_hash = %s;' % hx(h), 'extern const u32 g_lifted_code_size = %s;' % hx(len(code))]
    reg += ['void reg_%04d();' % k for k in ids]
    reg += ['void lifted_register_all() {'] + ['  reg_%04d();' % k for k in ids] + ['}', '']
    write_if_changed(os.path.join(GEN, 'lifted_reg.cpp'), '\n'.join(reg))
    cm = ['set(LIFT_SRC ' + ' '.join(['lifted_reg.cpp'] + files) + ')',
          'add_library(r3dslifted STATIC ${LIFT_SRC})',
          'target_include_directories(r3dslifted PRIVATE ${CMAKE_SOURCE_DIR}/rt)',
          'target_compile_options(r3dslifted PRIVATE -O1 -fno-strict-aliasing -w -fwrapv -ffp-contract=off)', '']
    write_if_changed(os.path.join(GEN, 'CMakeLists.txt'), '\n'.join(cm))
    # remove stale files
    for fn in os.listdir(GEN):
        if fn.startswith('lifted_') and fn.endswith('.cpp') and fn not in files and fn != 'lifted_reg.cpp':
            os.unlink(os.path.join(GEN, fn))
    log('wrote %d files (%d changed)' % (nf, nchanged))

if __name__ == '__main__':
    main()
