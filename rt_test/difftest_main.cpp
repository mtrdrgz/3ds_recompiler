// Differential test: lifted single instructions vs Unicorn single-step.
#include "lift_rt.h"
#include <unicorn/unicorn.h>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <csignal>
#include <csetjmp>
static sigjmp_buf g_jb;
static void on_segv(int) { siglongjmp(g_jb, 1); }

struct TestEnt { u32 pc; const char *key; const char *txt; ChunkFn fn; };
extern TestEnt g_tests[];
extern int g_ntests;

void uc_sync_map(u32, u32, u32) {}
void uc_sync_unmap(u32, u32) {}
void uc_sync_protect(u32, u32, u32) {}
int g_log_level = 0;
void fatal(const char *fmt, ...) { fprintf(stderr, "fatal %s\n", fmt); exit(1); }
std::atomic<int> g_irq_pending{0};
void rt_svc(Cpu &, u32) {}
u32 rt_miss(Cpu &, u32 p) { return p; }
u32 interp_run(Cpu &, u32 p) { return p; }
void rt_trap(Cpu &, u32, const char *) { exit(3); }
void disp_register(u32, ChunkFn) {}
ChunkFn disp_lookup(u32) { return nullptr; }
int g_interp_only = 0;
int g_host_fp_plain = 0;
std::atomic<u32> g_excl_epoch{0};
std::atomic<u64> g_stat_interp{0};
u32 armint_step(Cpu &c, u32 pc, int *svc);
static bool g_use_interp = false;

static const u32 SCR = 0x08000000, SCR_SIZE = 0x100000;
static const int REGS[] = {UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_R4, UC_ARM_REG_R5,
    UC_ARM_REG_R6, UC_ARM_REG_R7, UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11, UC_ARM_REG_R12,
    UC_ARM_REG_R13, UC_ARM_REG_R14};

static bool g_special = false;
int main(int argc, char **argv) {
    g_special = getenv("DT_SPECIAL") != nullptr;
    g_use_interp = getenv("DT_INTERP") != nullptr;
    host_fpscr(0x03C00000);
    mem_init();
    FILE *f = fopen("extracted/code.bin", "rb");
    std::vector<u8> img(IMG_END - IMG_BASE);
    size_t n = fread(img.data(), 1, img.size(), f); fclose(f);
    mem_map(IMG_BASE, IMG_END - IMG_BASE, MEM_RW, "img");
    memcpy(gp(IMG_BASE), img.data(), n);
    mem_map(SCR, SCR_SIZE, MEM_RW, "scratch");
    uc_engine *uc;
    uc_open(UC_ARCH_ARM, UC_MODE_ARM, &uc);
    uc_ctl_set_cpu_model(uc, UC_CPU_ARM_11MPCORE);
    uc_mem_map_ptr(uc, IMG_BASE, IMG_END - IMG_BASE, UC_PROT_ALL, gp(IMG_BASE));
    uc_mem_map_ptr(uc, SCR, SCR_SIZE, UC_PROT_ALL, gp(SCR));
    signal(SIGSEGV, on_segv); signal(SIGBUS, on_segv);
    std::mt19937 rng(argc > 1 ? atoi(argv[1]) : 7);
    std::vector<u8> snap(SCR_SIZE), after_l(SCR_SIZE);
    std::map<std::string, std::pair<int, int>> stats;
    int fails = 0;
    for (int t = 0; t < g_ntests; t++) {
        TestEnt &te = g_tests[t];
        bool bad = false;
        std::string why;
        for (int iter = 0; iter < 24 && !bad; iter++) {
            Cpu c0{};
            for (int i = 0; i < 15; i++) {
                u32 r = rng();
                int mode = rng() % 4;
                if (mode < 2) c0.r[i] = SCR + 0x40000 + (r % 0x80000 & ~3u);
                else if (mode == 2) c0.r[i] = r % 64;
                else c0.r[i] = r;
            }
            c0.r[13] = SCR + 0x80000 + (rng() % 0x1000 & ~7u);
            c0.n = rng() & 1; c0.z = rng() & 1; c0.c = rng() & 1; c0.v = rng() & 1;
            c0.ge = rng() & 15;
            static const u32 specials[] = {0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
                                           0x00000001, 0x80000001, 0x007FFFFF, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0x00800000};
            for (int i = 0; i < 32; i++) {
                if (g_special && rng() % 3 == 0) c0.f.sw[i] = specials[rng() % 13];
                else c0.f.s[i] = (float)((int)(rng() % 2000) - 1000) / (1 + rng() % 16);
            }
            c0.fpscr = 0x03C00000;
            c0.tpidruro = 0x1FF82000;
            // random scratch contents
            for (u32 i = 0; i < SCR_SIZE; i += 4) { u32 v = rng(); memcpy(&snap[i], &v, 4); }
            memcpy(gp(SCR), snap.data(), SCR_SIZE);
            Cpu cl = c0;
            u32 npc_l;
            if (sigsetjmp(g_jb, 1)) { continue; }
            host_fpscr(cl.fpscr);
            if (g_use_interp) { int svc; npc_l = armint_step(cl, te.pc, &svc); }
            else npc_l = te.fn(cl, te.pc);
            memcpy(after_l.data(), gp(SCR), SCR_SIZE);
            // unicorn
            memcpy(gp(SCR), snap.data(), SCR_SIZE);
            bool thumb = te.pc & 1;
            u32 cpsr = (c0.n << 31) | (c0.z << 30) | (c0.c << 29) | (c0.v << 28) | (c0.ge << 16) | 0x10 | (thumb ? 0x20 : 0);
            uc_reg_write(uc, UC_ARM_REG_CPSR, &cpsr);
            for (int i = 0; i < 15; i++) uc_reg_write(uc, REGS[i], &c0.r[i]);
            u32 fpexc = 0x40000000; uc_reg_write(uc, UC_ARM_REG_FPEXC, &fpexc);
            for (int i = 0; i < 16; i++) uc_reg_write(uc, UC_ARM_REG_D0 + i, &c0.f.dw[i]);
            uc_reg_write(uc, UC_ARM_REG_FPSCR, &c0.fpscr);
            uc_reg_write(uc, UC_ARM_REG_C13_C0_3, &c0.tpidruro);
            uc_err e = uc_emu_start(uc, te.pc, 0xFFFFFFFF, 0, 1);
            Cpu cu{};
            for (int i = 0; i < 15; i++) uc_reg_read(uc, REGS[i], &cu.r[i]);
            u32 pcu; uc_reg_read(uc, UC_ARM_REG_PC, &pcu);
            uc_reg_read(uc, UC_ARM_REG_CPSR, &cpsr);
            for (int i = 0; i < 16; i++) uc_reg_read(uc, UC_ARM_REG_D0 + i, &cu.f.dw[i]);
            u32 fpscr_u; uc_reg_read(uc, UC_ARM_REG_FPSCR, &fpscr_u);
            if (e != UC_ERR_OK) { continue; }   // unicorn faulted (random address): skip sample
            u32 npc_u = pcu | ((cpsr >> 5) & 1);
            char buf[512];
            for (int i = 0; i < 15 && !bad; i++)
                if (cu.r[i] != cl.r[i]) { snprintf(buf, sizeof buf, "r%d lifted=%08x uc=%08x (in %08x)", i, cl.r[i], cu.r[i], c0.r[i]); why = buf; bad = true; }
            u32 fl = (cl.n << 3) | (cl.z << 2) | (cl.c << 1) | cl.v, fu = cpsr >> 28;
            if (!bad && fl != fu) { snprintf(buf, sizeof buf, "flags lifted=%x uc=%x (in %x)", fl, fu, (c0.n << 3) | (c0.z << 2) | (c0.c << 1) | c0.v); why = buf; bad = true; }
            if (!bad && ((cpsr >> 16) & 15) != cl.ge) { snprintf(buf, sizeof buf, "ge lifted=%x uc=%x", cl.ge, (cpsr >> 16) & 15); why = buf; bad = true; }
            if (!bad && npc_l != npc_u) { snprintf(buf, sizeof buf, "npc lifted=%08x uc=%08x", npc_l, npc_u); why = buf; bad = true; }
            for (int i = 0; i < 16 && !bad; i++)
                if (cu.f.dw[i] != cl.f.dw[i]) {
                    snprintf(buf, sizeof buf, "d%d lifted=%016llx uc=%016llx (s%d %g/%g) in: d0=%016llx d1=%016llx d2=%016llx fpscr_uc=%08x", i, (unsigned long long)cl.f.dw[i], (unsigned long long)cu.f.dw[i], 2 * i, cl.f.s[2*i], cu.f.s[2*i],
                             (unsigned long long)c0.f.dw[0], (unsigned long long)c0.f.dw[1], (unsigned long long)c0.f.dw[2], fpscr_u);
                    why = buf; bad = true;
                }
            if (!bad && (fpscr_u >> 28) != (cl.fpscr >> 28)) { snprintf(buf, sizeof buf, "fpscr lifted=%08x uc=%08x", cl.fpscr, fpscr_u); why = buf; bad = true; }
            if (!bad && memcmp(after_l.data(), gp(SCR), SCR_SIZE)) {
                for (u32 i = 0; i < SCR_SIZE; i++) if (after_l[i] != rd8(SCR + i)) { snprintf(buf, sizeof buf, "mem @%08x lifted=%02x uc=%02x", SCR + i, after_l[i], rd8(SCR + i)); break; }
                why = buf; bad = true;
            }
        }
        auto &s = stats[te.key];
        s.second++;
        if (bad) {
            s.first++; fails++;
            printf("FAIL %08x %-30s %s\n", te.pc, te.txt, why.c_str());
        }
    }
    printf("\n%d/%d tests failed\n", fails, g_ntests);
    for (auto &kv : stats) if (kv.second.first) printf("  %-16s %d/%d failed\n", kv.first.c_str(), kv.second.first, kv.second.second);
    return 0;
}
