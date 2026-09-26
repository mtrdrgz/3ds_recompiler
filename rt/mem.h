// Flat guest address space: the full 32-bit guest space is reserved at
// g_base on the host; guest VA x lives at g_base + x. Pages are committed
// on demand by mem_map().
#pragma once
#include "common.h"

extern u8 *g_base;

enum : u32 { MEM_R = 1, MEM_W = 2, MEM_X = 4, MEM_RW = 3, MEM_RWX = 7, MEM_RX = 5 };

void mem_init();
void mem_map(u32 va, u32 size, u32 perm, const char *tag);
void mem_unmap(u32 va, u32 size);
void mem_protect(u32 va, u32 size, u32 perm);
bool mem_is_mapped(u32 va);
const char *mem_region_name(u32 va);

// physical address translation (GPU / DMA)
u8 *pa_ptr(u32 pa);
u32 va_to_pa(u32 va);
u32 pa_to_va(u32 pa);

static inline u8 *gp(u32 va) { return g_base + va; }
static inline u8  rd8(u32 a)  { return *(u8 *)(g_base + a); }
static inline u16 rd16(u32 a) { u16 v; memcpy(&v, g_base + a, 2); return v; }
static inline u32 rd32(u32 a) { u32 v; memcpy(&v, g_base + a, 4); return v; }
static inline u64 rd64(u32 a) { u64 v; memcpy(&v, g_base + a, 8); return v; }
static inline void wr8(u32 a, u8 v)   { *(u8 *)(g_base + a) = v; }
static inline void wr16(u32 a, u16 v) { memcpy(g_base + a, &v, 2); }
static inline void wr32(u32 a, u32 v) { memcpy(g_base + a, &v, 4); }
static inline void wr64(u32 a, u64 v) { memcpy(g_base + a, &v, 8); }

// fixed layout
// host bytes reserved for the guest: the full 4GB natively; on wasm32 the
// guest lives in the low 512MB of its VA space (everything the game maps).
#ifdef __EMSCRIPTEN__
constexpr u64 GUEST_SPAN = 0x20000000ull;
#else
constexpr u64 GUEST_SPAN = 0x100000000ull;
#endif
// guest image layout: filled from the ROM's exheader (rt/rom.cpp) at boot.
// The values below are only fallbacks for ROM-less users (rt_test/difftest).
extern u32 IMG_BASE;    // .text address = entry point
extern u32 TEXT_END;    // .text end = .rodata start
extern u32 RO_END;      // .rodata end = .data start
extern u32 IMG_END;     // data + bss, page aligned
extern u32 g_entry;     // process entry point (== IMG_BASE on CTR)
extern u32 g_stack_size; // main thread stack size
constexpr u32 HEAP_BASE    = 0x08000000;
constexpr u32 STACK_TOP    = 0x10000000;
constexpr u32 LINEAR_BASE  = 0x14000000;
constexpr u32 LINEAR_SIZE  = 0x08000000;
constexpr u32 VRAM_VA      = 0x1F000000;
constexpr u32 VRAM_SIZE    = 0x00600000;
constexpr u32 CFGMEM_VA    = 0x1FF80000;
constexpr u32 SHPAGE_VA    = 0x1FF81000;
constexpr u32 TLS_VA       = 0x1FF82000;
constexpr u32 FCRAM_PA     = 0x20000000;
constexpr u32 VRAM_PA      = 0x18000000;
