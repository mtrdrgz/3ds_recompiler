// Fallback interpreter: Unicorn executes guest code the lifter did not cover
// (and everything, in R3DS_INTERP=1 mode). It shares guest memory with the
// lifted code via uc_mem_map_ptr and stops at the first block that has a
// lifted entry, or at an SVC, handing control back to the dispatcher.
#ifdef R3DS_USE_UNICORN
#include "cpu.h"
#include "mem.h"
#include "lift_rt.h"
#include <unicorn/unicorn.h>
#include <set>
#include <string>
#include <mutex>
#include <atomic>

static uc_engine *g_uc;
std::atomic<int> g_in_uc{0};
void interp_kick() { if (g_in_uc.load()) uc_emu_stop(g_uc); }
static u32 g_start_pc, g_stop_pc;
static bool g_stopped_at_lift;
static int g_pending_svc = -1;
static int g_intr_other = -1;
extern int g_interp_only;
FILE *g_miss_log;
static std::set<u32> g_miss_seen;
extern std::atomic<u64> g_stat_interp;

static void chk(uc_err e, const char *what) {
    if (e != UC_ERR_OK) fatal("unicorn %s: %s", what, uc_strerror(e));
}

static u32 uc_perm(u32 p) {
    u32 r = 0;
    if (p & MEM_R) r |= UC_PROT_READ;
    if (p & MEM_W) r |= UC_PROT_WRITE;
    if (p & MEM_X) r |= UC_PROT_EXEC;
    return r;
}
void uc_sync_map(u32 va, u32 size, u32 perm) {
    if (!g_uc) return;
    // executable anywhere that is readable: guest can run code from heap too
    uc_err e = uc_mem_map_ptr(g_uc, va, size, uc_perm(perm) | ((perm & MEM_R) ? UC_PROT_EXEC : 0), g_base + va);
    if (e != UC_ERR_OK) LOG("[uc] map %08x+%x: %s", va, size, uc_strerror(e));
}
void uc_sync_unmap(u32 va, u32 size) {
    if (!g_uc) return;
    uc_mem_unmap(g_uc, va, size);
}
void uc_sync_protect(u32 va, u32 size, u32 perm) {
    if (!g_uc) return;
    uc_mem_protect(g_uc, va & ~0xFFFu, (size + 0xFFF) & ~0xFFFu, uc_perm(perm) | ((perm & MEM_R) ? UC_PROT_EXEC : 0));
}

#include <unordered_map>
std::unordered_map<u32, u64> g_prof[64];
u32 thread_id_of(Thread *t);
int g_profiling = 0;
extern Thread *cur_thread();
u32 thread_id_of(Thread *t);
u32 g_trace_pcs[8]; int g_ntrace = 0; int g_trace_budget = 40;
static void hook_block(uc_engine *uc, uint64_t addr, uint32_t size, void *) {
    for (int i = 0; i < g_ntrace; i++) if ((u32)addr == g_trace_pcs[i] && g_trace_budget > 0) {
        g_trace_budget--;
        u32 r[16]; for (int k = 0; k < 13; k++) uc_reg_read(uc, UC_ARM_REG_R0 + k, &r[k]);
        uc_reg_read(uc, UC_ARM_REG_SP, &r[13]); uc_reg_read(uc, UC_ARM_REG_LR, &r[14]);
        LOG("[trace %08x] t%u r0=%08x r1=%08x r2=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x r8=%08x sp=%08x lr=%08x",
            (u32)addr, thread_id_of(cur_thread()), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[13], r[14]);
    }
    if (g_profiling) { u32 id = thread_id_of(cur_thread()); g_prof[id & 63][(u32)addr]++; }
    u32 cpsr; uc_reg_read(uc, UC_ARM_REG_CPSR, &cpsr);
    u32 pc = (u32)addr | ((cpsr >> 5) & 1);
    if (pc != g_start_pc && disp_lookup(pc)) {
        g_stop_pc = pc; g_stopped_at_lift = true;
        uc_emu_stop(uc);
    }
}

static void hook_intr(uc_engine *uc, uint32_t intno, void *) {
    u32 pc, cpsr;
    uc_reg_read(uc, UC_ARM_REG_PC, &pc);
    uc_reg_read(uc, UC_ARM_REG_CPSR, &cpsr);
    if (intno == 2) {
        u32 imm = (cpsr & 0x20) ? (rd16(pc - 2) & 0xFF) : (rd32(pc - 4) & 0xFFFFFF);
        g_pending_svc = (int)imm;
    } else {
        g_intr_other = (int)intno;
    }
    uc_emu_stop(uc);
}

void interp_init() {
    chk(uc_open(UC_ARCH_ARM, UC_MODE_ARM, &g_uc), "open");
    chk(uc_ctl_set_cpu_model(g_uc, UC_CPU_ARM_11MPCORE), "cpu model");
    uc_hook h1, h2;
    chk(uc_hook_add(g_uc, &h1, UC_HOOK_BLOCK, (void *)hook_block, nullptr, 1, 0), "hook block");
    chk(uc_hook_add(g_uc, &h2, UC_HOOK_INTR, (void *)hook_intr, nullptr, 1, 0), "hook intr");
    u32 fpexc = 0x40000000;
    uc_reg_write(g_uc, UC_ARM_REG_FPEXC, &fpexc);
    if (const char *tp = getenv("R3DS_TRACE_PC")) {
        std::string s = tp; size_t p = 0;
        while (p < s.size() && g_ntrace < 8) { g_trace_pcs[g_ntrace++] = (u32)strtoul(s.c_str() + p, nullptr, 16); size_t q = s.find(',', p); if (q == std::string::npos) break; p = q + 1; }
    }
    const char *m = getenv("R3DS_MISS_LOG");
    g_miss_log = fopen(m && *m ? m : "miss.log", "a");
}

static const int REGS[] = {
    UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_R4,
    UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7, UC_ARM_REG_R8, UC_ARM_REG_R9,
    UC_ARM_REG_R10, UC_ARM_REG_R11, UC_ARM_REG_R12, UC_ARM_REG_R13, UC_ARM_REG_R14};

void dump_cpu(const Cpu &c);

static void put_state(const Cpu &c, u32 pc) {
    u32 cpsr = (cpsr_of(c) & ~0x3Fu) | 0x10 | ((pc & 1) ? 0x20 : 0);
    uc_reg_write(g_uc, UC_ARM_REG_CPSR, &cpsr);   // mode first: SP/LR are banked
    for (int i = 0; i < 15; i++) uc_reg_write(g_uc, REGS[i], &c.r[i]);
    u32 fpexc = 0x40000000;
    uc_reg_write(g_uc, UC_ARM_REG_FPEXC, &fpexc);
    for (int i = 0; i < 16; i++) uc_reg_write(g_uc, UC_ARM_REG_D0 + i, &c.f.dw[i]);
    uc_reg_write(g_uc, UC_ARM_REG_FPSCR, &c.fpscr);
    uc_reg_write(g_uc, UC_ARM_REG_C13_C0_3, &c.tpidruro);
    u32 p = pc;
    uc_reg_write(g_uc, UC_ARM_REG_PC, &p);
}

static void get_state(Cpu &c) {
    for (int i = 0; i < 15; i++) uc_reg_read(g_uc, REGS[i], &c.r[i]);
    u32 cpsr; uc_reg_read(g_uc, UC_ARM_REG_CPSR, &cpsr);
    cpsr_set(c, cpsr);
    for (int i = 0; i < 16; i++) uc_reg_read(g_uc, UC_ARM_REG_D0 + i, &c.f.dw[i]);
    uc_reg_read(g_uc, UC_ARM_REG_FPSCR, &c.fpscr);
    uc_reg_read(g_uc, UC_ARM_REG_C13_C0_3, &c.tpidruro);
    uc_reg_read(g_uc, UC_ARM_REG_PC, &c.r[15]);
}

u32 interp_run(Cpu &c, u32 pc) {
    if (!g_interp_only && g_miss_log && g_miss_seen.insert(pc).second) {
        fprintf(g_miss_log, "%08x\n", pc); fflush(g_miss_log);
    }
    g_stat_interp++;
    g_start_pc = pc; g_stopped_at_lift = false; g_pending_svc = -1; g_intr_other = -1;
    c.excl_valid = 0;
    put_state(c, pc);
    g_in_uc = 1;
    uc_err e = uc_emu_start(g_uc, pc, 0xFFFFFFFFull, 0, 0);
    g_in_uc = 0;
    get_state(c);
    host_fpscr(c.fpscr);
    if (e != UC_ERR_OK) {
        LOG("[interp] fault at pc=%08x (%s): %s", c.r[15], c.thumb ? "thumb" : "arm", uc_strerror(e));
        dump_cpu(c);
        fatal("guest fault");
    }
    if (g_pending_svc >= 0) {
        u32 next = c.r[15] | c.thumb;
        rt_svc(c, (u32)g_pending_svc);
        return next;
    }
    if (g_intr_other >= 0) {
        LOG("[interp] exception %d at pc=%08x", g_intr_other, c.r[15]);
        dump_cpu(c);
        fatal("guest exception");
    }
    if (g_stopped_at_lift) return g_stop_pc;
    return c.r[15] | c.thumb;
}
#endif
