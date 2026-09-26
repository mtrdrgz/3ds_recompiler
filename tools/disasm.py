#!/usr/bin/env python3
# quick disassembler: disasm.py addr [count] [t] [code.bin]
#   t -> disassemble as Thumb; code.bin defaults to extracted/code.bin and its
#   load base to the exheader .text address (manifest.json), else 0x100000.
import sys, os, json
from capstone import *
file = sys.argv[4] if len(sys.argv) > 4 else os.environ.get('R3DS_CODEBIN', 'extracted/code.bin')
code = open(file, 'rb').read()
try:
    base = json.load(open(os.path.join(os.path.dirname(file), 'manifest.json')))['text']['addr']
except Exception:
    base = 0x100000
a = int(sys.argv[1], 16); n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
th = len(sys.argv) > 3 or (a & 1)
a &= ~1
md = Cs(CS_ARCH_ARM, CS_MODE_THUMB if th else CS_MODE_ARM)
for i in md.disasm(code[a - base:a - base + n * 4], a):
    if n <= 0: break
    n -= 1
    print(hex(i.address), i.bytes.hex(), i.mnemonic, i.op_str)
