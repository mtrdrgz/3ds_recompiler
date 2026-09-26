// Guest ARM11 CPU state shared by lifted code, the fallback interpreter and
// the HLE kernel. A "pc" handed around the runtime carries the Thumb state
// in bit 0 (BX semantics): ARM code at 4-aligned addresses, Thumb at pc|1.
#pragma once
#include "common.h"

struct Thread;

struct alignas(16) Cpu {
    u32 r[16];
    u32 n, z, c, v;          // APSR flags (0/1)
    u32 q;                   // sticky saturation
    u32 ge;                  // SIMD GE[3:0]
    u32 fpscr;               // FPSCR (NZCV kept in bits 31..28)
    u32 tpidruro;            // CP15 c13 c0 3 (TLS pointer)
    u32 tpidrurw;            // CP15 c13 c0 2
    u32 excl_valid, excl_addr;
    u32 excl_epoch;          // g_excl_epoch at the last LDREX (see below)
    union {
        float s[32];
        double d[16];
        u32 sw[32];
        u64 dw[16];
    } f;
    Thread *thread;
    u32 thumb;               // current ISA (valid when leaving a block)
};

// Exclusive monitor: the kernel clears every thread's reservation on
// context switches, SVCs, preemption points and HLE writes to shared
// memory by bumping this epoch; STREX succeeds only if it is unchanged
// since the matching LDREX (what CLREX-on-switch gives on hardware).
#include <atomic>
extern std::atomic<u32> g_excl_epoch;
static inline u32 excl_epoch_now() { return g_excl_epoch.load(std::memory_order_relaxed); }
static inline void excl_clear_all() { g_excl_epoch.fetch_add(1, std::memory_order_relaxed); }

// Lifted chunk: runs from `pc` until control leaves the chunk, returns the
// next pc (bit0 = thumb).
typedef u32 (*ChunkFn)(Cpu &, u32 pc);

void disp_register(u32 pc, ChunkFn fn);   // called by generated tables
ChunkFn disp_lookup(u32 pc);
[[noreturn]] void disp_run(Cpu &c, u32 pc);  // per-thread dispatch loop
u32 interp_run(Cpu &c, u32 pc);             // fallback interpreter step
void rt_svc(Cpu &c, u32 imm);               // HLE kernel entry
u32 rt_miss(Cpu &c, u32 pc);                // lifted chunk got an unknown entry
void rt_undef(Cpu &c, u32 pc, const char *what);

static inline u32 cpsr_of(const Cpu &c) {
    return (c.n << 31) | (c.z << 30) | (c.c << 29) | (c.v << 28) | (c.q << 27) |
           (c.ge << 16) | (c.thumb ? 0x20 : 0) | 0x10;
}
static inline void cpsr_set(Cpu &c, u32 v) {
    c.n = v >> 31; c.z = (v >> 30) & 1; c.c = (v >> 29) & 1; c.v = (v >> 28) & 1;
    c.q = (v >> 27) & 1; c.ge = (v >> 16) & 0xF; c.thumb = (v >> 5) & 1;
}
