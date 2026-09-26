#include "rom.h"
#include "platform.h"
#include <mutex>

static PFile *g_rom;
static u64 g_ncch, g_romfs_off, g_romfs_size;
static const u64 MU = 0x200;

static bool rd(void *b, u64 n, u64 off) { return pf_pread(g_rom, b, n, off) == (s64)n; }
template <class T> static T le(const u8 *p) { T v; memcpy(&v, p, sizeof v); return v; }

u32 code_hash(const u8 *p, size_t n) {
    u32 h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

bool rom_open(const std::string &path) {
    g_rom = pf_open(path.c_str(), false, false);
    if (!g_rom) { LOG("[rom] cannot open '%s'", path.c_str()); return false; }
    u8 hdr[0x200];
    if (!rd(hdr, sizeof hdr, 0)) { LOG("[rom] short read"); return false; }
    if (!memcmp(hdr + 0x100, "NCSD", 4)) g_ncch = (u64)le<u32>(hdr + 0x120) * MU;
    else if (!memcmp(hdr + 0x100, "NCCH", 4)) g_ncch = 0;   // bare CXI
    else { LOG("[rom] '%s' is not a .3ds (NCSD) or .cxi image", path.c_str()); return false; }
    u8 ncch[0x200];
    if (!rd(ncch, sizeof ncch, g_ncch) || memcmp(ncch + 0x100, "NCCH", 4)) { LOG("[rom] no NCCH in partition 0"); return false; }
    if (!(ncch[0x18F] & 0x04)) {
        LOG("[rom] the ROM is encrypted. Decrypt it first (e.g. with GodMode9 \"Decrypt file\" or a 3DS decryptor);"
            " the recompilation needs a decrypted (NoCrypto) image.");
        return false;
    }
    char prod[17]; memcpy(prod, ncch + 0x150, 16); prod[16] = 0;
    INFO("[rom] product code %s", prod);
    u64 ivfc = g_ncch + (u64)le<u32>(ncch + 0x1B0) * MU;
    u8 h[0x60];
    if (!rd(h, sizeof h, ivfc) || memcmp(h, "IVFC", 4)) { LOG("[rom] RomFS IVFC header not found"); return false; }
    u32 master_hash = le<u32>(h + 0x08);
    u64 l3size = le<u64>(h + 0x44);
    u64 bs = 1ull << le<u32>(h + 0x4C);
    g_romfs_off = ivfc + ((0x60 + master_hash + bs - 1) & ~(bs - 1));
    g_romfs_size = l3size;
    INFO("[rom] NCCH at %llx, RomFS level3 at %llx size %llx", (unsigned long long)g_ncch,
         (unsigned long long)g_romfs_off, (unsigned long long)g_romfs_size);
    return true;
}

// Guest image layout, taken from the ROM's exheader in rom_load_code().
// Defaults only let ROM-less tools (rt_test/difftest) run.
u32 IMG_BASE = 0x00100000, TEXT_END = 0x0061D000, RO_END = 0x006A4000,
    IMG_END = 0x007A8000, g_entry = 0x00100000, g_stack_size = 0x40000;

// NCCH exheader, ARM11 system capabilities (SCI):
//   0x10 .text {addr, pages, filesize, stack_size}   (16 bytes)
//   0x20 .rodata {addr, pages, filesize, -}
//   0x30 .data {addr, pages, filesize, bss_size}
// Region end = addr + pages * 0x1000; the image ends at data + filesize + bss,
// page aligned. The entry point is the .text address.
static bool read_layout(const u8 *e) {
    u32 ta = le<u32>(e + 0x10), tp = le<u32>(e + 0x14), ssz = le<u32>(e + 0x1C);
    u32 ra = le<u32>(e + 0x20), rp = le<u32>(e + 0x24);
    u32 da = le<u32>(e + 0x30), dsz = le<u32>(e + 0x38), bss = le<u32>(e + 0x3C);
    // a section ends where the next one begins; absent sections collapse
    u32 text_end = ra ? ra : ta + tp * 0x1000;
    u32 ro_end = da ? da : (ra ? ra + rp * 0x1000 : text_end);
    u32 data_end = da ? da + dsz : ro_end;
    u32 img_end = (data_end + bss + 0xFFF) & ~0xFFFu;
    if (ta < 0x10000 || ta >= 0x10000000 || text_end <= ta ||
        ro_end < text_end || data_end < ro_end || img_end < data_end)
        return false;                       // implausible: keep defaults
    IMG_BASE = g_entry = ta;
    TEXT_END = text_end;
    RO_END = ro_end;
    IMG_END = img_end;
    g_stack_size = ssz && ssz <= 0x800000 ? ssz : 0x40000;
    INFO("[rom] exheader: text %08x..%08x ro %08x..%08x data %08x img_end %08x stack %x",
         ta, TEXT_END, ra, RO_END, da, IMG_END, g_stack_size);
    return true;
}

// CTR ExeFS .code compression: reverse LZSS with a footer at the end.
static bool blz_decompress(std::vector<u8> &d) {
    if (d.size() < 8) return false;
    size_t csz = d.size();
    u32 tb = le<u32>(&d[csz - 8]), extra = le<u32>(&d[csz - 4]);
    std::vector<u8> out(csz + extra);
    memcpy(out.data(), d.data(), csz);
    size_t index = csz - ((tb >> 24) & 0xFF), stop = csz - (tb & 0xFFFFFF), pos = out.size();
    while (index > stop) {
        u8 ctl = d[--index];
        for (int b = 0; b < 8 && index > stop && pos > 0; b++, ctl <<= 1) {
            if (ctl & 0x80) {
                index -= 2;
                u32 seg = d[index] | (d[index + 1] << 8);
                u32 len = ((seg >> 12) & 0xF) + 3, off = (seg & 0xFFF) + 2;
                for (u32 k = 0; k < len && pos > 0; k++) { out[pos - 1] = out[pos + off]; pos--; }
            } else {
                out[--pos] = d[--index];
            }
        }
    }
    d.swap(out);
    return true;
}

bool rom_load_code(std::vector<u8> &code) {
    u8 ncch[0x200], exh[0x400];
    rd(ncch, sizeof ncch, g_ncch);
    rd(exh, sizeof exh, g_ncch + 0x200);
    if (!read_layout(exh))
        LOG("[rom] could not parse the exheader's code layout; keeping built-in defaults");
    bool compressed = exh[0x0D] & 1;
    u64 exefs = g_ncch + (u64)le<u32>(ncch + 0x1A0) * MU;
    u8 eh[0x200];
    if (!rd(eh, sizeof eh, exefs)) return false;
    for (int i = 0; i < 10; i++) {
        const u8 *e = eh + i * 0x10;
        if (memcmp(e, ".code\0\0\0", 8)) continue;
        u32 off = le<u32>(e + 8), sz = le<u32>(e + 12);
        code.resize(sz);
        if (!rd(code.data(), sz, exefs + 0x200 + off)) return false;
        if (compressed && !blz_decompress(code)) return false;
        INFO("[rom] .code %x bytes%s -> %zx", sz, compressed ? " (compressed)" : "", code.size());
        return true;
    }
    LOG("[rom] ExeFS has no .code");
    return false;
}

s64 rom_romfs_read(void *buf, u64 size, u64 off) {
    if (off >= g_romfs_size) return 0;
    if (size > g_romfs_size - off) size = g_romfs_size - off;
    return pf_pread(g_rom, buf, size, g_romfs_off + off);
}
u64 rom_romfs_size() { return g_romfs_size; }
