#!/usr/bin/env python3
"""Extract a decrypted .3ds (CCI) -> decrypted code.bin + exheader + ExeFS files.

The dump is already decrypted (NCCH flag[7] bit2 set). Layout:
  NCSD header @0x0 -> partition table -> CXI (NCCH) @ part0 offset
  CXI: NCCH header @+0, exheader @+0x200, ExeFS/RomFS at offsets from NCCH.
"""
import json
import os
import struct
import sys

ROM = sys.argv[1] if len(sys.argv) > 1 else "game.3ds"
OUT = sys.argv[2] if len(sys.argv) > 2 else "extracted"
MU = 0x200  # media unit


def lzss_decompress_code(data: bytes) -> bytes:
    """CTR ExeFS .code decompression (reverse LZSS, footer at end)."""
    comp_size = len(data)
    footer = comp_size - 8
    tb = int.from_bytes(data[footer:footer + 4], "little")
    out_size = comp_size + int.from_bytes(data[comp_size - 4:comp_size], "little")
    out = bytearray(out_size)
    out[:comp_size] = data
    index = comp_size - ((tb >> 24) & 0xFF)
    stop = comp_size - (tb & 0xFFFFFF)
    pos = out_size
    while index > stop:
        control = data[index - 1]; index -= 1
        for _ in range(8):
            if index <= stop or pos <= 0:
                break
            if control & 0x80:
                index -= 2
                seg = data[index] | (data[index + 1] << 8)
                length = ((seg >> 12) & 0xF) + 3
                seg_off = (seg & 0xFFF) + 2
                for _ in range(length):
                    out[pos - 1] = out[pos + seg_off]
                    pos -= 1
            else:
                out[pos - 1] = data[index - 1]
                index -= 1; pos -= 1
            control = (control << 1) & 0xFF
    return bytes(out)


def main():
    d = open(ROM, "rb").read()
    os.makedirs(OUT, exist_ok=True)
    exefs_dir = os.path.join(OUT, "exefs")
    os.makedirs(exefs_dir, exist_ok=True)

    assert d[0x100:0x104] == b"NCSD", "not an NCSD/CCI image"
    part0_off = struct.unpack("<I", d[0x120:0x124])[0] * MU
    cxi = part0_off
    assert d[cxi + 0x100:cxi + 0x104] == b"NCCH", "no NCCH at partition 0"

    ncch = d[cxi:cxi + 0x200]
    flags7 = ncch[0x18F]
    encrypted = not (flags7 & 0x04)
    plain_off, plain_sz = struct.unpack("<II", ncch[0x190:0x198])
    exefs_off, exefs_sz = struct.unpack("<II", ncch[0x1A0:0x1A8])
    romfs_off, romfs_sz = struct.unpack("<II", ncch[0x1B0:0x1B8])
    print(f"CXI @0x{cxi:x}  encrypted={encrypted}  "
          f"exefs=0x{exefs_off * MU:x}+0x{exefs_sz * MU:x}  "
          f"romfs=0x{romfs_off * MU:x}+0x{romfs_sz * MU:x}")

    exheader = d[cxi + 0x200:cxi + 0x200 + 0x800]
    open(os.path.join(OUT, "exheader.bin"), "wb").write(exheader)
    # SCI: title@0x00[8], flag@0x0D bit0=compress, codesets {addr,pages,size}
    code_name = exheader[0x0:0x8].rstrip(b"\0").decode("ascii", "replace")
    prog_id = code_name
    compress_flag = exheader[0x0D] & 1
    text_addr, text_pages, text_size = struct.unpack("<III", exheader[0x10:0x1C])
    stack_size = struct.unpack("<I", exheader[0x1C:0x20])[0]
    ro_addr, ro_pages, ro_size = struct.unpack("<III", exheader[0x20:0x2C])
    data_addr, data_pages, data_size = struct.unpack("<III", exheader[0x30:0x3C])
    bss_size = struct.unpack("<I", exheader[0x3C:0x40])[0]
    print(f"program id={prog_id} name={code_name!r} compress={compress_flag}")
    print(f"  text @0x{text_addr:08x} sz=0x{text_size:x}  "
          f"ro @0x{ro_addr:08x} sz=0x{ro_size:x}  data @0x{data_addr:08x} sz=0x{data_size:x}  "
          f"stack=0x{stack_size:x} bss=0x{bss_size:x}")

    # ExeFS: header = 10 * {name[8], offset u32, size u32}; data @ +0x200
    exefs_base = cxi + exefs_off * MU
    files = []
    for i in range(10):
        e = d[exefs_base + i * 0x10:exefs_base + i * 0x10 + 0x10]
        name = e[0:8].rstrip(b"\0").decode("ascii", "replace")
        off, sz = struct.unpack("<II", e[8:0x10])
        if not name or sz == 0:
            continue
        raw = d[exefs_base + 0x200 + off:exefs_base + 0x200 + off + sz]
        files.append((name, off, sz))
        if name == ".code":
            if compress_flag:
                code = lzss_decompress_code(raw)
                print(f"  .code: LZSS 0x{sz:x} -> 0x{len(code):x}")
            else:
                code = raw
                print(f"  .code: raw 0x{sz:x}")
            open(os.path.join(OUT, "code.bin"), "wb").write(code)
        else:
            open(os.path.join(exefs_dir, name.replace("/", "_")), "wb").write(raw)
        print(f"  exefs[{i}] {name!r} off=0x{off:x} sz=0x{sz:x}")

    manifest = {
        "rom": ROM, "cxi_offset": cxi, "program_id": prog_id,
        "code_name": code_name, "compressed_code": bool(compress_flag),
        "text": {"addr": text_addr, "pages": text_pages, "size": text_size},
        "rodata": {"addr": ro_addr, "pages": ro_pages, "size": ro_size},
        "data": {"addr": data_addr, "pages": data_pages, "size": data_size},
        "stack_size": stack_size, "bss_size": bss_size,
        "exefs": [{"name": n, "offset": o, "size": s} for n, o, s in files],
        "romfs": {"offset": cxi + romfs_off * MU, "size": romfs_sz * MU},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {OUT}/manifest.json")


if __name__ == "__main__":
    main()
