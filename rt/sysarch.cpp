#include "sysarch.h"
#include <map>
#include <mutex>
#include <string>

namespace {

constexpr u32 INVALID = 0xFFFFFFFFu;

struct File { std::string name; std::vector<u8> data; };

void put16(std::vector<u8> &v, size_t at, u16 x) { memcpy(&v[at], &x, 2); }
void put32(std::vector<u8> &v, size_t at, u32 x) { memcpy(&v[at], &x, 4); }
void put64(std::vector<u8> &v, size_t at, u64 x) { memcpy(&v[at], &x, 8); }
u32 get32(const std::vector<u8> &v, size_t at) { u32 x; memcpy(&x, &v[at], 4); return x; }
size_t align(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

// RomFS name hash table length: smallest count >= n that is odd and not
// divisible by the small primes (what the OS tools emit).
u32 hash_len(u32 n) {
    u32 c = n;
    if (n < 3) c = 3;
    else if (n < 19) c |= 1;
    else while (c % 2 == 0 || c % 3 == 0 || c % 5 == 0 || c % 7 == 0 || c % 11 == 0 || c % 13 == 0 || c % 17 == 0) c++;
    return c;
}

u32 name_hash(const std::vector<u8> &utf16, u32 parent) {
    u32 h = parent ^ 123456789u;
    for (size_t i = 0; i + 1 < utf16.size(); i += 2) {
        h = (h >> 5) | (h << 27);
        h ^= (u32)utf16[i] | ((u32)utf16[i + 1] << 8);
    }
    return h;
}

std::vector<u8> utf16le(const std::string &s) {
    std::vector<u8> o;
    for (unsigned char ch : s) { o.push_back(ch); o.push_back(0); }
    return o;
}

// Level-3 RomFS with every file directly under the root directory.
std::vector<u8> build_romfs(const std::vector<File> &files) {
    const u32 nfiles = (u32)files.size();
    const u32 dh_len = hash_len(1), fh_len = hash_len(nfiles);
    const size_t dhash_off = 0x28, dhash_sz = dh_len * 4;
    const size_t dmeta_off = align(dhash_off + dhash_sz, 4), dmeta_sz = 0x18;
    const size_t fhash_off = align(dmeta_off + dmeta_sz, 4), fhash_sz = fh_len * 4;
    const size_t fmeta_off = align(fhash_off + fhash_sz, 4);
    std::vector<size_t> fentry(nfiles);
    size_t fmeta_sz = 0;
    for (u32 i = 0; i < nfiles; i++) {
        fentry[i] = fmeta_sz;
        fmeta_sz += align(0x20 + files[i].name.size() * 2, 4);
    }
    const size_t data_off = align(fmeta_off + fmeta_sz, 0x10);
    size_t total = data_off;
    std::vector<size_t> fdata(nfiles);
    for (u32 i = 0; i < nfiles; i++) { fdata[i] = total - data_off; total = align(total + files[i].data.size(), 0x10); }

    std::vector<u8> v(total, 0);
    put32(v, 0x00, 0x28);
    put32(v, 0x04, (u32)dhash_off); put32(v, 0x08, (u32)dhash_sz);
    put32(v, 0x0C, (u32)dmeta_off); put32(v, 0x10, (u32)dmeta_sz);
    put32(v, 0x14, (u32)fhash_off); put32(v, 0x18, (u32)fhash_sz);
    put32(v, 0x1C, (u32)fmeta_off); put32(v, 0x20, (u32)fmeta_sz);
    put32(v, 0x24, (u32)data_off);
    for (u32 i = 0; i < dh_len; i++) put32(v, dhash_off + i * 4, INVALID);
    for (u32 i = 0; i < fh_len; i++) put32(v, fhash_off + i * 4, INVALID);
    // root directory (offset 0 in the dir meta table)
    put32(v, dmeta_off + 0x00, 0);
    put32(v, dmeta_off + 0x04, INVALID);
    put32(v, dmeta_off + 0x08, INVALID);
    put32(v, dmeta_off + 0x0C, nfiles ? (u32)fentry[0] : INVALID);
    put32(v, dmeta_off + 0x10, INVALID);
    put32(v, dmeta_off + 0x14, 0);
    put32(v, dhash_off + (name_hash({}, 0) % dh_len) * 4, 0);
    for (u32 i = 0; i < nfiles; i++) {
        size_t e = fmeta_off + fentry[i];
        std::vector<u8> nm = utf16le(files[i].name);
        size_t slot = fhash_off + (name_hash(nm, 0) % fh_len) * 4;
        put32(v, e + 0x00, 0);                                         // parent = root
        put32(v, e + 0x04, i + 1 < nfiles ? (u32)fentry[i + 1] : INVALID);   // next file in dir
        put64(v, e + 0x08, fdata[i]);
        put64(v, e + 0x10, files[i].data.size());
        put32(v, e + 0x18, get32(v, slot));                            // hash chain
        put32(v, e + 0x1C, (u32)nm.size());
        memcpy(&v[e + 0x20], nm.data(), nm.size());
        put32(v, slot, (u32)fentry[i]);
        if (!files[i].data.empty()) memcpy(&v[data_off + fdata[i]], files[i].data.data(), files[i].data.size());
    }
    return v;
}

// Mii component resource (CFL_Res.dat): 20 sections (goatee/hair/face/... models,
// then the texture families), each `count` items. Every item is empty, so the
// library initialises and finds no selectable parts; Miis are simply absent.
std::vector<u8> mii_resource() {
    static const u16 counts[20] = {0x04, 0x108, 0x0C, 0x108, 0x01, 0x108, 0x0C, 0x12, 0x12, 0x84,
                                   0x3E, 0x18, 0x03, 0x0C, 0x0C, 0x09, 0x02, 0x25, 0x06, 0x12};
    std::vector<u8> v(4 + 20 * 4, 0);
    put16(v, 0, 20);
    put16(v, 2, 0x0509);
    for (int s = 0; s < 20; s++) {
        put32(v, 4 + s * 4, (u32)v.size());
        size_t at = v.size();
        v.resize(at + 4 + (counts[s] + 1) * 4, 0);   // count, max item size, offsets[count + 1] (all 0)
        put16(v, at, counts[s]);
    }
    return v;
}

std::vector<u8> build(u64 title_id) {
    switch (title_id) {
    case 0x0004009B00010202ull:   // Mii resources
        return build_romfs({{"CFL_Res.dat", mii_resource()}});
    default:
        return {};
    }
}

}  // namespace

const std::vector<u8> *sysarch_romfs(u64 title_id) {
    static std::mutex mtx;
    static std::map<u64, std::vector<u8>> cache;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find(title_id);
    if (it == cache.end()) it = cache.emplace(title_id, build(title_id)).first;
    return it->second.empty() ? nullptr : &it->second;
}
