#!/usr/bin/env python3
"""Generate a differential test of lifted single instructions vs Unicorn.

  difftest.py [per_class]   -> gen_test/tests.cpp
"""
import os, sys, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift
from armdec import COND_EXPR, hx
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
seeds = lift.gather_seeds()
lift.discover(seeds)
mdA = Cs(CS_ARCH_ARM, CS_MODE_ARM); mdT = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
groups = collections.defaultdict(list)
for pc, i in lift.insns.items():
    if i.bad or 'SVC(' in i.body or 'TRAP(' in i.body: continue
    a = pc & ~1
    md = mdT if pc & 1 else mdA
    try:
        d = next(md.disasm(lift.code[a - lift.BASE:a - lift.BASE + i.size], a, 1))
    except StopIteration:
        continue
    m = d.mnemonic
    # strip condition suffix for grouping
    for cc in ('eq','ne','cs','hs','cc','lo','mi','pl','vs','vc','hi','ls','ge','lt','gt','le'):
        if m.endswith(cc) and len(m) > len(cc) + 1 and m not in ('teq', 'vcmpe'):
            m = m[:-2]; break
    key = ('T:' if pc & 1 else 'A:') + m.split('.')[0]
    groups[key].append((pc, d.mnemonic + ' ' + d.op_str))
random.seed(1)
tests = []
for k in sorted(groups):
    g = groups[k]
    random.shuffle(g)
    tests += [(k, pc, txt) for pc, txt in g[:per]]
out = os.path.join(lift.ROOT, 'gen_test')
os.makedirs(out, exist_ok=True)
f = open(os.path.join(out, 'tests.cpp'), 'w')
f.write('#include "lift_rt.h"\n')
for n, (k, pc, txt) in enumerate(tests):
    i = lift.insns[pc]
    body = lift.lower(i.body, set(), pc, set())
    if i.cond != 14: body = 'if (%s) { %s }' % (COND_EXPR[i.cond], body)
    nxt = (pc & 1) | ((pc & ~1) + i.size)
    f.write('static u32 T%d(Cpu &c, u32 pc) { PROLOGUE %s EXIT(%s); OUT }\n' % (n, body, hx(nxt)))
f.write('struct TestEnt { u32 pc; const char *key; const char *txt; ChunkFn fn; };\n')
f.write('TestEnt g_tests[] = {\n')
for n, (k, pc, txt) in enumerate(tests):
    f.write('  {%s, "%s", "%s", T%d},\n' % (hx(pc), k, txt.replace('"', "'"), n))
f.write('};\nint g_ntests = %d;\n' % len(tests))
f.close()
print('wrote %d tests in %d classes' % (len(tests), len(groups)), file=sys.stderr)
