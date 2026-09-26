// Host replacements for guest functions whose original behaviour only
// makes sense on real silicon. Registered in the dispatch table like
// lifted chunks. Which addresses to patch is per game: pass them via the
// environment so one build serves any title.
//
//   R3DS_HOOK_DELAY=0x162514,0x...   pcs of a calibrated delay loop
//   R3DS_NO_HOOKS=1                  disable all hooks
#include "cpu.h"
#include "kernel.h"
#include "mem.h"

// SDK busy-wait loop `subs r0,r0,#2; bgt` burning r0 ticks: the main loop
// would starve lower-priority workers under a single-core scheduler when it
// polls vsync with it, so sleep for real instead. r0 takes the tick count.
static u32 hook_delay_ticks(Cpu &c, u32) {
    u64 ticks = c.r[0];
    u64 ns = ticks * 1000000000ull / 268111856ull;   // CTR CPU clock
    if (ns < 500000) ns = 500000;
    c.r[0] = (u32)ns; c.r[1] = (u32)(ns >> 32);
    rt_svc(c, 0x0A);
    c.r[0] = 0;
    c.n = 0; c.z = 1; c.c = 1; c.v = 0;
    return c.r[14];
}

void hle_hooks_register() {
    if (getenv("R3DS_NO_HOOKS")) return;
    const char *l = getenv("R3DS_HOOK_DELAY");
    while (l && *l) {
        u32 pc = (u32)strtoul(l, (char **)&l, 16);
        disp_register(pc, hook_delay_ticks);   // set bit0 for Thumb entries
        if (*l == ',') l++;
    }
}
