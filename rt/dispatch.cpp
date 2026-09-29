#include "cpu.h"
#include "mem.h"
#include "gen.h"
#include <vector>
#include <atomic>

// direct-mapped table over .text: index = pc - IMG_BASE (odd = thumb entry)
static ChunkFn *g_table;
static u32 g_table_n;
int g_interp_only = 0;
std::atomic<u64> g_stat_lifted{0}, g_stat_interp{0};

void disp_init() {   // call after rom_load_code(): sizes come from the exheader
    g_table_n = TEXT_END - IMG_BASE;
    g_table = (ChunkFn *)calloc(g_table_n, sizeof(ChunkFn));
}

void disp_register(u32 pc, ChunkFn fn) {
    u32 i = pc - IMG_BASE;
    if (i < g_table_n) g_table[i] = fn;
    gen_mark_host(pc);   // generated code hands this pc back to the runtime
}

static ChunkFn raw_lookup(u32 pc) {
    u32 i = pc - IMG_BASE;
    if (i >= g_table_n) return nullptr;
    return g_table[i];
}

// stands for "there is generated code at this pc" (never called)
static u32 gen_marker(Cpu &, u32 pc) { return pc; }

ChunkFn disp_lookup(u32 pc) {
    if (ChunkFn f = raw_lookup(pc)) return f;
    return gen_entry(pc) > 1 ? gen_marker : nullptr;
}

std::atomic<int> g_irq_pending{0};
void kernel_poll(Cpu &c);

[[noreturn]] void disp_run(Cpu &c, u32 pc) {
    for (;;) {
        if (g_irq_pending.load(std::memory_order_relaxed)) { c.r[15] = pc & ~1u; c.thumb = pc & 1; kernel_poll(c); }
        if (gen_active()) {
            pc = gen_run(c, pc);   // returns on an interrupt, a hook, or code that was never translated
            if (g_irq_pending.load(std::memory_order_relaxed)) continue;
            ChunkFn h = raw_lookup(pc);
            pc = h ? h(c, pc) : interp_run(c, pc);
            continue;
        }
        ChunkFn f = disp_lookup(pc);
        if (f) pc = f(c, pc);
        else pc = interp_run(c, pc);
    }
}

void dump_cpu(const Cpu &c);
u32 rt_miss(Cpu &c, u32 pc) {
    // a chunk was entered at a pc it has no case for: interpret instead
    return interp_run(c, pc);
}

[[noreturn]] void rt_trap(Cpu &c, u32 pc, const char *msg) {
    LOG("[trap] %s at pc=%08x", msg, pc);
    dump_cpu(c);
    fatal("guest trap");
}
