#!/usr/bin/env python3
"""Run the picked instructions one at a time in Unicorn and record the results.

  python3 tests/wasm_diff/oracle.py tests.json [cases_per_test=16] > oracle.json

Input state per case: random registers (a third point into the scratch window,
a third are small, a third random), random NZCV/GE, one shared random scratch
buffer. Cases where Unicorn faults are dropped (random addresses).
"""
import sys, json, random, base64, os
from unicorn import *
from unicorn.arm_const import *

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = json.load(open(sys.argv[1]))
ncases = int(sys.argv[2]) if len(sys.argv) > 2 else 16
L = spec['layout']
code = open(os.path.join(ROOT, 'extracted/code.bin'), 'rb').read()
SCR, SCR_SIZE = 0x08000000, 0x10000
rng = random.Random(4242)
scratch = bytes(rng.getrandbits(8) for _ in range(SCR_SIZE))

REGS = [UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6,
        UC_ARM_REG_R7, UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11, UC_ARM_REG_R12,
        UC_ARM_REG_R13, UC_ARM_REG_R14]

uc = Uc(UC_ARCH_ARM, UC_MODE_ARM)
uc.ctl_set_cpu_model(UC_CPU_ARM_11MPCORE)
base = L['base']
img_size = (len(code) + 0xFFF) & ~0xFFF
uc.mem_map(base, img_size, UC_PROT_READ | UC_PROT_EXEC)
uc.mem_write(base, code)
uc.mem_map(SCR, SCR_SIZE, UC_PROT_ALL)
fpexc = 0x40000000
uc.reg_write(UC_ARM_REG_FPEXC, fpexc)

def rand_reg():
    m = rng.randrange(4)
    if m < 2: return SCR + 0x2000 + (rng.randrange(0xC000) & ~3)
    if m == 2: return rng.randrange(64)
    return rng.getrandbits(32)

out = []
for t in spec['tests']:
    pc = t['pc']
    cases = []
    for _ in range(ncases):
        regs = [rand_reg() for _ in range(15)]
        regs[13] = SCR + 0x8000 + (rng.randrange(0x1000) & ~7)
        n, z, c, v = (rng.getrandbits(1) for _ in range(4))
        ge = rng.getrandbits(4)
        uc.mem_write(SCR, scratch)
        cpsr = (n << 31) | (z << 30) | (c << 29) | (v << 28) | (ge << 16) | 0x10 | (0x20 if pc & 1 else 0)
        uc.reg_write(UC_ARM_REG_CPSR, cpsr)
        for i in range(15): uc.reg_write(REGS[i], regs[i])
        try:
            uc.emu_start(pc, 0xFFFFFFFF, 0, 1)
        except UcError:
            continue
        after = [uc.reg_read(REGS[i]) for i in range(15)]
        pc_after = uc.reg_read(UC_ARM_REG_PC)
        cp = uc.reg_read(UC_ARM_REG_CPSR)
        mem = bytes(uc.mem_read(SCR, SCR_SIZE))
        diff = []
        if mem != scratch:
            diff = [[i, mem[i]] for i in range(SCR_SIZE) if mem[i] != scratch[i]]
        cases.append({'r': regs, 'f': [n, z, c, v], 'ge': ge, 'ro': after, 'fo': [(cp >> 31) & 1, (cp >> 30) & 1, (cp >> 29) & 1, (cp >> 28) & 1],
                      'geo': (cp >> 16) & 15, 'npc': (pc_after | ((cp >> 5) & 1)) & 0xFFFFFFFF, 'mem': diff})
    out.append(cases)
json.dump({'scratch': base64.b64encode(scratch).decode(), 'scr': SCR, 'results': out}, sys.stdout)
print('%d tests, %d cases' % (len(out), sum(len(c) for c in out)), file=sys.stderr)
