// Snapshots of what a GPU command list reads from main memory.
//
// The GPU thread runs behind the game. The real GPU is never a frame late,
// and the game relies on that: it rewrites vertex data it computes on the CPU
// (skinned and animated meshes), and reuses other buffers, right after
// submitting the lists that read them. Executed late, those lists read
// half-rewritten data: geometry and textures broke up more the longer the
// player moved (docs/RECOMP2.md, bug 7). R3DS_SYNC_GPU=1 avoids it by running
// the GPU on the game thread, at a large cost.
//
// Instead, when the game submits a command list, gpu_capture() walks it with
// a mirror of the relevant registers (vertex arrays, index buffers, textures,
// command buffer jumps) and copies the main-memory (FCRAM) ranges it will
// read. The GPU thread executes the list against that copy (gpu_ptr). VRAM is
// never copied: only the GPU writes it (DMA, transfers, draws), in queue order.
#include "gpu.h"
#include "mem.h"
#include "pica.h"
#include "pica_snap.h"
#include <algorithm>

std::atomic<const GpuSnap *> g_gpu_snap{nullptr};

const u8 *gpu_ptr_slow(const GpuSnap *s, u32 va, u32 len) {
    thread_local const GpuSnap *ls = nullptr;
    thread_local size_t li = 0;
    const auto &g = s->segs;
    size_t i;
    if (ls == s && li < g.size() && va - g[li].lo < g[li].hi - g[li].lo) i = li;
    else {
        auto it = std::upper_bound(g.begin(), g.end(), va, [](u32 v, const GpuSnapSeg &x) { return v < x.lo; });
        if (it == g.begin()) return gp(va);
        i = (size_t)(it - g.begin()) - 1;
        if (va >= g[i].hi) return gp(va);
        ls = s; li = i;
    }
    if ((u64)va + len > g[i].hi) return gp(va);
    return s->data.data() + g[i].off + (va - g[i].lo);
}

u32 pica_tex_bytes(u32 w, u32 h, u32 fmt);

static u32 CR[0x400];   // register mirror, in submission order
static const u32 FCRAM_END = 0x28000000;   // FCRAM_PA: mem.h

struct Range { u32 lo, hi; };   // VA
static void add_pa(std::vector<Range> &out, u32 pa, u32 len) {
    if (!len || pa < FCRAM_PA || pa >= FCRAM_END) return;
    len = std::min<u32>(len, FCRAM_END - pa);
    u32 va = pa_to_va(pa);
    if (!va || !mem_is_mapped(va) || !mem_is_mapped(va + len - 1)) return;
    out.push_back({va, va + len});
}
static u32 rd_live(u32 va, u32 n) { return n == 2 ? rd16(va) : n == 1 ? rd8(va) : rd32(va); }

static void capture_draw(std::vector<Range> &out, bool indexed) {
    const u32 base_pa = CR[0x200] << 3;
    const u32 count = CR[0x228], first = CR[0x22A];
    if (!count || count > 0x100000) return;
    u32 mn = first, mx = first + count - 1;
    if (indexed) {
        const u32 icfg = CR[0x227];
        const bool i16 = icfg >> 31;
        const u32 ipa = base_pa + (icfg & 0x0FFFFFFF), isz = i16 ? 2 : 1;
        u32 iva = pa_to_va(ipa);
        if (!iva || !mem_is_mapped(iva) || !mem_is_mapped(iva + count * isz - 1)) return;
        add_pa(out, ipa, count * isz);
        mn = ~0u; mx = 0;
        for (u32 n = 0; n < count; n++) { u32 v = rd_live(iva + n * isz, isz); mn = std::min(mn, v); mx = std::max(mx, v); }
    }
    if (mx - mn > 0x100000) return;
    for (u32 l = 0; l < 12; l++) {
        const u32 off = CR[0x203 + l * 3], hi = CR[0x205 + l * 3];
        const u32 ncomp = hi >> 28, stride = (hi >> 16) & 0xFF;
        if (!ncomp) continue;
        const u32 pa = base_pa + off;
        if (!stride) { add_pa(out, pa, 64); continue; }
        add_pa(out, pa + mn * stride, (mx - mn + 1) * stride);
    }
    // textures: the base level and its mipmaps
    static const u32 unit_reg[3] = {0x81, 0x91, 0x99};
    for (int t = 0; t < 3; t++) {
        if (!((CR[0x80] >> t) & 1)) continue;
        const u32 b = unit_reg[t];
        const u32 h = CR[b + 1] & 0x7FF, w = (CR[b + 1] >> 16) & 0x7FF;
        const u32 fmt = (t == 0 ? CR[0x8E] : CR[b + 5]) & 0xF;
        const u32 levels = ((CR[b + 3] >> 16) & 0xF) + 1;
        u32 bytes = 0;
        for (u32 lv = 0, lw = w, lh = h; lv < levels && lw >= 8 && lh >= 8; lv++, lw >>= 1, lh >>= 1) bytes += pica_tex_bytes(lw, lh, fmt);
        add_pa(out, CR[b + 4] << 3, bytes);
    }
}

std::shared_ptr<GpuSnap> gpu_capture(u32 addr, u32 size, const std::vector<std::pair<u32, u32>> &skip) {
    std::vector<Range> rs;
    if (!mem_is_mapped(addr) || !mem_is_mapped(addr + size - 1)) return nullptr;
    rs.push_back({addr, addr + size});
    u32 p = addr, end = addr + size;
    u32 jump_pa = 0, jump_size = 0;
    int jumps = 0;
    while (p + 8 <= end) {
        if (jump_pa) {
            u32 va = pa_to_va(jump_pa);
            if (!va || ++jumps > 10000 || !mem_is_mapped(va) || !mem_is_mapped(va + jump_size - 1)) break;
            rs.push_back({va, va + jump_size});
            p = va; end = va + jump_size; jump_pa = 0;
            continue;
        }
        u32 param = rd32(p), hdr = rd32(p + 4);
        p += 8;
        u32 reg = hdr & 0xFFFF, mask = (hdr >> 16) & 0xF, extra = (hdr >> 20) & 0xFF;
        bool seq = hdr >> 31;
        const u32 m = (mask & 1 ? 0xFF : 0) | (mask & 2 ? 0xFF00 : 0) | (mask & 4 ? 0xFF0000 : 0) | (mask & 8 ? 0xFF000000 : 0);
        for (u32 i = 0; i <= extra; i++) {
            if (i) { if (seq) reg++; param = rd32(p); p += 4; }
            if (reg >= 0x400) continue;
            CR[reg] = (CR[reg] & ~m) | (param & m);
            switch (reg) {
            case 0x22E: capture_draw(rs, false); break;
            case 0x22F: capture_draw(rs, true); break;
            case 0x23C: jump_pa = CR[0x23A] << 3; jump_size = CR[0x238] << 3; break;
            case 0x23D: jump_pa = CR[0x23B] << 3; jump_size = CR[0x239] << 3; break;
            default: break;
            }
        }
        if (extra & 1) p += 4;
    }
    // merge, then leave out what queued GPU work will still write (a DMA or
    // transfer into main memory): that must be read after it, as before
    std::sort(rs.begin(), rs.end(), [](const Range &a, const Range &b) { return a.lo < b.lo; });
    auto snap = std::make_shared<GpuSnap>();
    std::vector<Range> merged;
    for (const Range &r : rs) {
        if (!merged.empty() && r.lo <= merged.back().hi + 256) merged.back().hi = std::max(merged.back().hi, r.hi);
        else merged.push_back(r);
    }
    size_t total = 0;
    for (Range &r : merged) {
        bool clash = false;
        for (auto &s : skip) if (r.lo < s.second && s.first < r.hi) { clash = true; break; }
        if (clash) { r.hi = r.lo; continue; }
        total += r.hi - r.lo;
    }
    snap->data.resize(total);
    size_t off = 0;
    for (const Range &r : merged) {
        if (r.hi <= r.lo) continue;
        memcpy(snap->data.data() + off, gp(r.lo), r.hi - r.lo);
        snap->segs.push_back({r.lo, r.hi, (u32)off});
        off += r.hi - r.lo;
    }
    return snap;
}
