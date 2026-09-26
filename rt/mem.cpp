#include "mem.h"
#include "platform.h"
#include <map>
#include <string>
#include <mutex>

u8 *g_base;

struct Region { u32 size; u32 perm; std::string tag; };
static std::map<u32, Region> g_regions;   // va -> region
static std::mutex g_mem_mtx;

void uc_sync_map(u32 va, u32 size, u32 perm);   // interp.cpp
void uc_sync_unmap(u32 va, u32 size);
void uc_sync_protect(u32 va, u32 size, u32 perm);

void mem_init() {
    void *p = plat_reserve((size_t)GUEST_SPAN);
    if (!p) fatal("cannot reserve guest address space");
    g_base = (u8 *)p;
}

static int host_prot(u32 perm) { return (perm & MEM_R ? 1 : 0) | (perm & MEM_W ? 2 : 0); }

void mem_map(u32 va, u32 size, u32 perm, const char *tag) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    va &= ~0xFFFu; size = (size + 0xFFF) & ~0xFFFu;
    if (!size) return;
    if ((u64)va + size > GUEST_SPAN) fatal("mem_map %08x+%x is outside the guest span", va, size);
    // replace overlapping pieces
    for (auto it = g_regions.begin(); it != g_regions.end();) {
        u32 a = it->first, e = a + it->second.size;
        if (e <= va || a >= va + size) { ++it; continue; }
        Region r = it->second;
        uc_sync_unmap(a, r.size);
        it = g_regions.erase(it);
        if (a < va) { g_regions[a] = {va - a, r.perm, r.tag}; uc_sync_map(a, va - a, r.perm); }
        if (e > va + size) { g_regions[va + size] = {e - va - size, r.perm, r.tag}; uc_sync_map(va + size, e - va - size, r.perm); }
    }
    void *p = g_base + va;
    if (!plat_commit(p, size)) fatal("mem_map %08x+%x failed", va, size);
    if ((perm & MEM_RW) != MEM_RW) plat_protect(p, size, host_prot(perm));
    g_regions[va] = {size, perm, tag};
    uc_sync_map(va, size, perm);
    DBG("[mem] map %08x-%08x %c%c%c %s", va, va + size, perm & 1 ? 'r' : '-', perm & 2 ? 'w' : '-', perm & 4 ? 'x' : '-', tag);
}

void mem_unmap(u32 va, u32 size) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    va &= ~0xFFFu; size = (size + 0xFFF) & ~0xFFFu;
    for (auto it = g_regions.begin(); it != g_regions.end();) {
        u32 a = it->first, e = a + it->second.size;
        if (e <= va || a >= va + size) { ++it; continue; }
        Region r = it->second;
        uc_sync_unmap(a, r.size);
        it = g_regions.erase(it);
        if (a < va) { g_regions[a] = {va - a, r.perm, r.tag}; uc_sync_map(a, va - a, r.perm); }
        if (e > va + size) { g_regions[va + size] = {e - va - size, r.perm, r.tag}; uc_sync_map(va + size, e - va - size, r.perm); }
    }
    plat_decommit(g_base + va, size);
}

void mem_protect(u32 va, u32 size, u32 perm) {
    // host protection only; region bookkeeping keeps the original
    plat_protect(g_base + (va & ~0xFFFu), (size + 0xFFF) & ~0xFFFu, host_prot(perm));
    uc_sync_protect(va, size, perm);
}

bool mem_is_mapped(u32 va) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    auto it = g_regions.upper_bound(va);
    if (it == g_regions.begin()) return false;
    --it;
    return va < it->first + it->second.size;
}

const char *mem_region_name(u32 va) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    auto it = g_regions.upper_bound(va);
    if (it == g_regions.begin()) return "unmapped";
    --it;
    return va < it->first + it->second.size ? it->second.tag.c_str() : "unmapped";
}

u32 va_to_pa(u32 va) {
    if (va >= LINEAR_BASE && va < LINEAR_BASE + LINEAR_SIZE) return va - LINEAR_BASE + FCRAM_PA;
    if (va >= VRAM_VA && va < VRAM_VA + VRAM_SIZE) return va - VRAM_VA + VRAM_PA;
    return 0;
}
u32 pa_to_va(u32 pa) {
    if (pa >= FCRAM_PA && pa < FCRAM_PA + LINEAR_SIZE) return pa - FCRAM_PA + LINEAR_BASE;
    if (pa >= VRAM_PA && pa < VRAM_PA + VRAM_SIZE) return pa - VRAM_PA + VRAM_VA;
    return 0;
}
u8 *pa_ptr(u32 pa) {
    u32 va = pa_to_va(pa);
    return va ? g_base + va : nullptr;
}

bool mem_query_impl(u32 va, u32 &base, u32 &size, u32 &perm) {
    std::lock_guard<std::mutex> lk(g_mem_mtx);
    auto it = g_regions.upper_bound(va);
    u32 lo = 0;
    if (it != g_regions.begin()) {
        auto p = std::prev(it);
        if (va < p->first + p->second.size) {
            base = p->first; size = p->second.size; perm = p->second.perm;
            return true;
        }
        lo = p->first + p->second.size;
    }
    u32 hi = it == g_regions.end() ? 0xFFFFF000u : it->first;
    base = lo; size = hi - lo; perm = 0;
    return false;
}
