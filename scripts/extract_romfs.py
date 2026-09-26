#!/usr/bin/env python3
"""Extract the IVFC-wrapped RomFS from a decrypted 3DS ROM dump.

Layout (verified empirically against `game.3ds` with SHA-256
hash-chain checks; see docs/ASSETS.md for the evidence):

  RomFS region (NCCH header field, 0x1B0) starts with an IVFC header:

    0x00 u32 magic "IVFC"
    0x04 u32 version
    0x08 u32 master hash size
    0x14 u64 level-1 size          0x1C u32 level-1 block size (log2)
    0x2C u64 level-2 size          0x34 u32 level-2 block size (log2)
    0x44 u64 level-3 size          0x4C u32 level-3 block size (log2)

  Level 3 is the actual filesystem.  Level 2 is a SHA-256 hash table covering
  the level-3 data in 2^log2 byte blocks, level 1 covers level 2 the same way,
  and the master hash (which directly follows the 0x60-byte header) covers
  level 1.

  This image does not store usable absolute offsets in the IVFC header (its
  offset fields do not follow the documented convention), so the regions are
  located from the size fields plus 0x1000 alignment:

    master hash  @ 0x60
    level 3      @ align_up(0x60 + master_hash_size, 0x1000)
    level 1      @ align_up(level3 + level3_size, 0x1000)
    level 2      @ align_up(level1 + level1_size, 0x1000)
    region end   @ align_up(level2 + level2_size, 0x1000)

  For this ROM that yields level3=0x1000, level1=0x34788000,
  level2=0x34796000 and region end 0x34E26000, which matches the NCCH RomFS
  size exactly and reproduces the hash chain.  `--no-verify` skips the hash
  pass, `--verify-sample N` checks N evenly spaced blocks instead of all.

  Level 3 starts with a RomFS header.  Two variants exist and both are handled:
  10 u32 fields (header size 0x28, the variant this image uses) and 10 u64
  fields (header size 0x50).  The fields are, in order:

    header_size, dir_hash_off/size, dir_meta_off/size,
    file_hash_off/size, file_meta_off/size, file_data_off

  Directory entry: {u32 parent, u32 next, u32 first_child, u32 first_file,
                    u32 hash_next, u32 name_size, u16 name[]}
  File entry:      {u32 parent, u32 next, u64 data_offset, u64 data_size,
                    u32 hash_next, u32 name_size, u16 name[]}
  Names are UTF-16LE and padded to 4 bytes.  Metadata offsets are relative to
  the start of their own table; file data offsets are relative to the file
  data region (RomFS base + file_data_offset).

The ROM is opened read-only and never modified.  Extraction is streamed in
fixed-size chunks (never holds a whole file, let alone 887 MB, in memory),
idempotent and resumable: files already on disk with the expected size are
skipped, partial writes go to "<name>.part" and are atomically renamed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import unicodedata

CHUNK = 1 << 20
MU = 0x200
IVFC_HEADER_SIZE = 0x60
ALIGN = 0x1000
MASK32 = 0xFFFFFFFF
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RomFsError(Exception):
    pass


# ---------------------------------------------------------------- ROM access


class RomReader:
    """Read-only, offset-addressed view of a slice of the ROM."""

    def __init__(self, path: str, base: int, size: int):
        self.path = path
        self.base = base
        self.size = size
        self.fh = open(path, "rb")

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise RomFsError(
                f"read out of range: offset=0x{offset:x} len=0x{length:x} "
                f"region=0x{self.size:x}")
        self.fh.seek(self.base + offset)
        data = self.fh.read(length)
        if len(data) != length:
            raise RomFsError(f"short read at 0x{offset:x}: {len(data)}/{length}")
        return data

    def stream(self, offset: int, length: int, chunk: int = CHUNK):
        remaining = length
        pos = offset
        while remaining > 0:
            n = min(chunk, remaining)
            yield self.read(pos, n)
            pos += n
            remaining -= n

    def close(self) -> None:
        self.fh.close()


def locate_romfs(rom_path: str, manifest_path: str | None) -> tuple[int, int]:
    """Return (romfs_offset, romfs_size) from the manifest or the NCSD/NCCH."""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if "romfs" in manifest:
            return int(manifest["romfs"]["offset"]), int(manifest["romfs"]["size"])

    with open(rom_path, "rb") as fh:
        head = fh.read(0x200)
        if head[0x100:0x104] == b"NCSD":
            part0 = struct.unpack_from("<I", head, 0x120)[0] * MU
        else:
            part0 = 0
        fh.seek(part0)
        ncch = fh.read(0x200)
    if ncch[0x100:0x104] != b"NCCH":
        raise RomFsError(f"{rom_path}: no NCCH at partition 0 (0x{part0:x})")
    romfs_off, romfs_sz = struct.unpack_from("<II", ncch, 0x1B0)
    return part0 + romfs_off * MU, romfs_sz * MU


# ------------------------------------------------------------- IVFC / RomFS


def align_up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def read_ivfc_layout(rom: RomReader) -> dict:
    """Parse the IVFC header and derive the level/region offsets."""
    header = rom.read(0, IVFC_HEADER_SIZE)
    if header[0:4] != b"IVFC":
        raise RomFsError(
            f"no IVFC magic at RomFS offset (found {header[0:4]!r}); "
            "the RomFS region is not IVFC-wrapped as expected")
    version, master_hash_size = struct.unpack_from("<II", header, 4)
    level1_size = struct.unpack_from("<Q", header, 0x14)[0]
    level2_size = struct.unpack_from("<Q", header, 0x2C)[0]
    level3_size = struct.unpack_from("<Q", header, 0x44)[0]
    level1_log2 = struct.unpack_from("<I", header, 0x1C)[0]
    level2_log2 = struct.unpack_from("<I", header, 0x34)[0]
    level3_log2 = struct.unpack_from("<I", header, 0x4C)[0]

    master_hash_off = IVFC_HEADER_SIZE
    level3_off = align_up(master_hash_off + master_hash_size)
    level1_off = align_up(level3_off + level3_size)
    level2_off = align_up(level1_off + level1_size)
    region_end = align_up(level2_off + level2_size)

    if not (0 < level3_size <= rom.size and region_end <= rom.size):
        raise RomFsError(
            f"derived IVFC layout does not fit the RomFS region "
            f"(region_end=0x{region_end:x} > size=0x{rom.size:x})")

    return {
        "version": version,
        "master_hash_size": master_hash_size,
        "master_hash_offset": master_hash_off,
        "level1_offset": level1_off,
        "level1_size": level1_size,
        "level1_block_log2": level1_log2,
        "level2_offset": level2_off,
        "level2_size": level2_size,
        "level2_block_log2": level2_log2,
        "level3_offset": level3_off,
        "level3_size": level3_size,
        "level3_block_log2": level3_log2,
        "region_end": region_end,
    }


def _build_header(fields, variant: str, level3_size: int) -> dict | None:
    if fields[0] not in (0x28, 0x50):
        return None
    keys = ("dir_hash", "dir_meta", "file_hash", "file_meta")
    header = {"header_size": fields[0], "variant": variant}
    for i, key in enumerate(keys):
        header[key + "_off"] = fields[1 + 2 * i]
        header[key + "_size"] = fields[2 + 2 * i]
    header["file_data_off"] = fields[9]

    offsets = [header["dir_hash_off"], header["dir_meta_off"],
               header["file_hash_off"], header["file_meta_off"],
               header["file_data_off"]]
    if offsets != sorted(offsets) or offsets[0] < header["header_size"]:
        return None
    if any(o + s > level3_size for o, s in
           ((header[k + "_off"], header[k + "_size"]) for k in keys)):
        return None
    if header["file_data_off"] > level3_size:
        return None
    return header


def read_romfs_header(rom: RomReader, fs_off: int, level3_size: int) -> dict:
    """Parse the RomFS header (32-bit or 64-bit offset variant)."""
    raw = rom.read(fs_off, 0x50)
    for variant, fmt in (("u32", "<10I"), ("u64", "<10Q")):
        header = _build_header(struct.unpack_from(fmt, raw, 0), variant,
                               level3_size)
        if header is not None:
            return header
    raise RomFsError(f"unrecognised RomFS header at 0x{fs_off:x}: "
                     f"{raw[:16].hex()}")


def _decode_name(blob: bytes, offset: int, size: int) -> str:
    raw = blob[offset:offset + size]
    if len(raw) != size:
        raise RomFsError(f"name at 0x{offset:x} runs past its table")
    return raw.decode("utf-16-le", "surrogatepass")


def walk_tree(rom: RomReader, fs_off: int, header: dict) -> list[dict]:
    """Walk the directory/file metadata linked lists into a flat file list."""
    dir_meta = rom.read(fs_off + header["dir_meta_off"], header["dir_meta_size"])
    file_meta = rom.read(fs_off + header["file_meta_off"], header["file_meta_size"])
    entries: list[dict] = []
    seen_dirs: set[int] = set()

    def walk(dir_offset: int, path: str) -> None:
        if dir_offset in seen_dirs:
            raise RomFsError(f"directory loop at metadata offset 0x{dir_offset:x}")
        seen_dirs.add(dir_offset)
        if dir_offset + 0x18 > len(dir_meta):
            raise RomFsError(f"directory offset 0x{dir_offset:x} out of range")
        _parent, _next, child, first_file, _hash, name_size = struct.unpack_from(
            "<6I", dir_meta, dir_offset)
        name = _decode_name(dir_meta, dir_offset + 0x18, name_size)

        offset = first_file
        seen_files: set[int] = set()
        while offset != MASK32:
            if offset in seen_files:
                raise RomFsError(f"file loop in {path or '/'}")
            seen_files.add(offset)
            if offset + 0x20 > len(file_meta):
                raise RomFsError(
                    f"file entry 0x{offset:x} out of range in {path or '/'}")
            _p, nxt, data_off, data_size, _h, fname_size = struct.unpack_from(
                "<IIQQII", file_meta, offset)
            fname = _decode_name(file_meta, offset + 0x20, fname_size)
            entries.append({
                "path": (path + "/" + fname) if path else fname,
                "dir": path,
                "name": fname,
                "data_offset": data_off,
                "size": data_size,
                "meta_offset": offset,
            })
            offset = nxt

        child_offset = child
        seen_children: set[int] = set()
        while child_offset != MASK32:
            if child_offset in seen_children:
                raise RomFsError(f"directory sibling loop in {path or '/'}")
            seen_children.add(child_offset)
            _cparent, cnext, _cc, _cf, _ch, cname_size = struct.unpack_from(
                "<6I", dir_meta, child_offset)
            cname = _decode_name(dir_meta, child_offset + 0x18, cname_size)
            walk(child_offset, (path + "/" + cname) if path else cname)
            child_offset = cnext

    walk(0, "")
    return entries


# --------------------------------------------------------------- extraction


def sanitize_component(name: str) -> str:
    """Make a single path component safe for the host filesystem."""
    name = name.replace("\x00", "")
    cleaned = "".join(
        ch for ch in name
        if ch >= " " and ch not in '/\\:' and unicodedata.category(ch) != "Cc")
    cleaned = cleaned.strip().rstrip(".")
    if cleaned in ("", ".", ".."):
        cleaned = "_"
    return cleaned


class PathAllocator:
    """Allocate unique, filesystem-safe paths for RomFS entries.

    macOS filesystems are case-insensitive and Unicode-normalising, so paths
    that differ only by case or normalisation form must not collide.
    """

    def __init__(self):
        self.taken: dict[str, str] = {}
        self.renamed: list[tuple[str, str]] = []

    @staticmethod
    def _key(path: str) -> str:
        return unicodedata.normalize("NFC", path).casefold()

    def allocate(self, romfs_path: str) -> str:
        parts = [sanitize_component(p) for p in romfs_path.split("/")]
        candidate = "/".join(parts)
        if self._key(candidate) not in self.taken:
            self.taken[self._key(candidate)] = candidate
            return candidate
        stem, ext = os.path.splitext(candidate)
        for n in range(2, 1000):
            alt = f"{stem}~{n}{ext}"
            if self._key(alt) not in self.taken:
                self.taken[self._key(alt)] = alt
                self.renamed.append((romfs_path, alt))
                return alt
        raise RomFsError(f"cannot allocate a unique path for {romfs_path!r}")


def format_size(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def extract_files(rom: RomReader, fs_off: int, header: dict, entries: list[dict],
                  out_dir: str, force: bool, quiet: bool) -> dict:
    data_base = fs_off + header["file_data_off"]
    data_size = header.get("data_region_size", 0)
    allocator = PathAllocator()
    total_bytes = sum(e["size"] for e in entries)
    written = skipped = 0
    written_bytes = 0
    started = time.time()
    last_report = 0.0
    out_of_range: list[dict] = []

    os.makedirs(out_dir, exist_ok=True)

    for index, entry in enumerate(entries, 1):
        rel = allocator.allocate(entry["path"])
        entry["disk_path"] = rel
        dest = os.path.join(out_dir, *rel.split("/"))
        if entry["data_offset"] + entry["size"] > data_size:
            out_of_range.append({
                "path": entry["path"],
                "data_offset": entry["data_offset"],
                "size": entry["size"],
            })
            continue

        if not force and os.path.exists(dest):
            if os.path.getsize(dest) == entry["size"]:
                skipped += 1
                written_bytes += entry["size"]
                continue

        os.makedirs(os.path.dirname(dest) or out_dir, exist_ok=True)
        tmp = dest + ".part"
        remaining = entry["size"]
        pos = data_base + entry["data_offset"]
        with open(tmp, "wb") as fh:
            while remaining > 0:
                n = min(CHUNK, remaining)
                fh.write(rom.read(pos, n))
                pos += n
                remaining -= n
        os.replace(tmp, dest)
        written += 1
        written_bytes += entry["size"]

        now = time.time()
        if not quiet and (now - last_report > 2.0 or index == len(entries)):
            last_report = now
            rate = written_bytes / max(now - started, 1e-6)
            print(f"  [{index}/{len(entries)}] {format_size(written_bytes)} / "
                  f"{format_size(total_bytes)} ({rate / (1 << 20):.0f} MB/s)",
                  file=sys.stderr, flush=True)

    return {
        "files_written": written,
        "files_skipped": skipped,
        "bytes_on_disk": written_bytes,
        "total_bytes": total_bytes,
        "out_of_range": out_of_range,
        "renamed_paths": allocator.renamed,
        "seconds": time.time() - started,
    }


# -------------------------------------------------------------- verification


def verify_hashes(rom: RomReader, layout: dict, sample: int | None,
                  quiet: bool) -> dict:
    """Verify the IVFC SHA-256 hash chain (level3 -> level2 -> level1)."""
    import hashlib

    l1_off, l1_size = layout["level1_offset"], layout["level1_size"]
    l2_off, l2_size = layout["level2_offset"], layout["level2_size"]
    l3_off, l3_size = layout["level3_offset"], layout["level3_size"]
    l1_bs = 1 << layout["level1_block_log2"]
    l2_bs = 1 << layout["level2_block_log2"]
    l3_bs = 1 << layout["level3_block_log2"]
    if l3_bs != l2_bs:
        raise RomFsError("level 2 and level 3 block sizes differ; unsupported")

    l3_blocks = (l3_size + l3_bs - 1) // l3_bs
    l2_blocks = (l2_size + l1_bs - 1) // l1_bs
    if l3_blocks * 32 != l2_size or l2_blocks * 32 != l1_size:
        raise RomFsError(
            "hash table sizes do not match the block counts "
            f"(l3_blocks={l3_blocks} l2_size=0x{l2_size:x}, "
            f"l2_blocks={l2_blocks} l1_size=0x{l1_size:x})")

    if sample is None:
        step_l3 = step_l2 = 1
        checked_l3, checked_l2 = l3_blocks, l2_blocks
    else:
        step_l3 = max(1, l3_blocks // max(sample, 1))
        step_l2 = max(1, l2_blocks // max(sample, 1))
        checked_l3 = len(range(0, l3_blocks, step_l3))
        checked_l2 = len(range(0, l2_blocks, step_l2))

    # The final (partial) block of each level is hashed over the full block
    # size, i.e. including the alignment padding that follows it.
    def read_block(offset: int, size: int) -> bytes:
        return rom.read(offset, min(size, rom.size - offset))

    bad_l3 = []
    for i in range(0, l3_blocks, step_l3):
        block = read_block(l3_off + i * l3_bs, l3_bs)
        expect = rom.read(l2_off + i * 32, 32)
        if hashlib.sha256(block).digest() != expect:
            bad_l3.append(i)
    bad_l2 = []
    for i in range(0, l2_blocks, step_l2):
        block = read_block(l2_off + i * l1_bs, l1_bs)
        expect = rom.read(l1_off + i * 32, 32)
        if hashlib.sha256(block).digest() != expect:
            bad_l2.append(i)

    if not quiet:
        print(f"  hash chain: level2->level3 {checked_l3 - len(bad_l3)}/{checked_l3} "
              f"blocks OK, level1->level2 {checked_l2 - len(bad_l2)}/{checked_l2} OK",
              file=sys.stderr, flush=True)
    return {
        "mode": "full" if sample is None else f"sample(l3 step {step_l3}, l2 step {step_l2})",
        "level3_blocks_checked": checked_l3,
        "level3_blocks_bad": bad_l3[:16],
        "level3_blocks_bad_total": len(bad_l3),
        "level2_blocks_checked": checked_l2,
        "level2_blocks_bad": bad_l2[:16],
        "level2_blocks_bad_total": len(bad_l2),
    }


# -------------------------------------------------------------------- main


def extract(rom: RomReader, args, romfs_off: int, romfs_size: int) -> dict:
    """Walk the RomFS, extract it (unless --list) and return the summary."""
    layout = read_ivfc_layout(rom)
    print(f"IVFC: master hash 0x{layout['master_hash_size']:x} @0x"
          f"{layout['master_hash_offset']:x}, level3 0x{layout['level3_size']:x} "
          f"@0x{layout['level3_offset']:x}, level1 0x{layout['level1_size']:x} "
          f"@0x{layout['level1_offset']:x}, level2 0x{layout['level2_size']:x} "
          f"@0x{layout['level2_offset']:x}, end 0x{layout['region_end']:x}",
          file=sys.stderr)
    if layout["region_end"] != romfs_size:
        print(f"warning: derived region end 0x{layout['region_end']:x} != "
              f"NCCH RomFS size 0x{romfs_size:x}", file=sys.stderr)

    fs_off = layout["level3_offset"]
    header = read_romfs_header(rom, fs_off, layout["level3_size"])
    data_region_size = layout["level3_size"] - header["file_data_off"]
    header["data_region_size"] = data_region_size
    print(f"RomFS header: dir_meta 0x{header['dir_meta_off']:x}+0x"
          f"{header['dir_meta_size']:x}, file_meta 0x{header['file_meta_off']:x}"
          f"+0x{header['file_meta_size']:x}, file_data 0x"
          f"{header['file_data_off']:x} (region 0x{data_region_size:x})",
          file=sys.stderr)

    entries = walk_tree(rom, fs_off, header)
    entries.sort(key=lambda e: e["data_offset"])
    total = sum(e["size"] for e in entries)
    covered = max((e["data_offset"] + e["size"] for e in entries), default=0)
    print(f"tree: {len(entries)} files, {format_size(total)} of file data, "
          f"data region {format_size(data_region_size)}", file=sys.stderr)

    summary: dict = {
        "rom": args.rom,
        "romfs": {"offset": romfs_off, "size": romfs_size},
        "ivfc": layout,
        "romfs_header": header,
        "file_count": len(entries),
        "total_file_bytes": total,
        "data_region_size": data_region_size,
        "data_region_padding": data_region_size - total,
        "data_region_coverage": covered,
    }

    if args.list:
        for e in entries:
            print(f"0x{e['data_offset']:08x} 0x{e['size']:08x}  {e['path']}")
        summary["entries"] = entries
    else:
        summary["extraction"] = extract_files(rom, fs_off, header, entries,
                                              args.out, args.force, args.quiet)

    if not args.no_verify:
        summary["hash_verification"] = verify_hashes(
            rom, layout, args.verify_sample, args.quiet)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the IVFC-wrapped RomFS from game.3ds")
    parser.add_argument("--rom", default=os.path.join(ROOT_DIR, "game.3ds"),
                        help="input .3ds dump (opened read-only)")
    parser.add_argument("--manifest", default=os.path.join(ROOT_DIR, "extracted",
                                                           "manifest.json"),
                        help="manifest with romfs offset/size (optional)")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "extracted", "romfs"),
                        help="output directory")
    parser.add_argument("--force", action="store_true",
                        help="re-extract files that already exist")
    parser.add_argument("--list", action="store_true",
                        help="list the tree instead of extracting")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the IVFC hash-chain verification pass")
    parser.add_argument("--verify-sample", type=int, default=None, metavar="N",
                        help="verify only N evenly spaced blocks per level")
    parser.add_argument("--json", dest="json_out", default=None, metavar="PATH",
                        help="also write the summary as JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress progress")
    args = parser.parse_args(argv)

    if not os.path.exists(args.rom):
        print(f"error: ROM not found: {args.rom}", file=sys.stderr)
        return 1

    romfs_off, romfs_size = locate_romfs(args.rom, args.manifest)
    print(f"RomFS region: offset 0x{romfs_off:x} size 0x{romfs_size:x} "
          f"({format_size(romfs_size)})", file=sys.stderr)
    rom = RomReader(args.rom, romfs_off, romfs_size)
    try:
        try:
            summary = extract(rom, args, romfs_off, romfs_size)
        except RomFsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        rom.close()

    if not args.list:
        stats = summary["extraction"]
        print(f"extracted {stats['files_written']} file(s), skipped "
              f"{stats['files_skipped']} already present, "
              f"{format_size(stats['bytes_on_disk'])} in "
              f"{stats['seconds']:.1f}s -> {args.out}")
        if stats["out_of_range"]:
            print(f"ERROR: {len(stats['out_of_range'])} entries point outside the "
                  "file data region; nothing was written for them", file=sys.stderr)
        if stats["renamed_paths"]:
            print(f"note: {len(stats['renamed_paths'])} path(s) disambiguated for "
                  "the host filesystem", file=sys.stderr)

    pad = summary["data_region_padding"]
    print(f"reconcile: {summary['total_file_bytes']} bytes in {summary['file_count']} "
          f"files vs data region {summary['data_region_size']} "
          f"(delta {pad} bytes of alignment padding, coverage "
          f"{summary['data_region_coverage']}/{summary['data_region_size']})")

    status = 0
    verification = summary.get("hash_verification")
    if verification and (verification["level3_blocks_bad_total"] or
                         verification["level2_blocks_bad_total"]):
        print("ERROR: IVFC hash verification failed", file=sys.stderr)
        status = 2
    if summary.get("extraction", {}).get("out_of_range"):
        status = max(status, 2)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
