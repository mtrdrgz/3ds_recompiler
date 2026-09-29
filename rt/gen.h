// Ahead-of-boot ARM -> WebAssembly code (web build).
//
// Before the guest starts, the page's translator (web/translator, injected as
// R3T) turns the loaded code image into a WebAssembly module: one function per
// chunk of guest code plus a dispatcher `run(cpu, pc)`. The module is derived
// from the user's own ROM in the user's browser and never leaves it.
#pragma once
#include "cpu.h"

// Translate the code image at IMG_BASE and prepare per-thread instances.
// Returns false (and leaves the interpreter in charge) when unavailable.
bool gen_init();
extern u32 *g_gen_table;    // pc -> chunk entry (see web/translator/translate.js); null when inactive
extern u32 g_gen_span;      // .text size in bytes covered by the table
bool gen_active();
// run generated code from pc until the runtime is needed; returns the pc to continue at
u32 gen_run(Cpu &c, u32 pc);
u32 gen_entry(u32 pc);      // table value for pc (0 none, 1 host hook, else generated chunk)
void gen_mark_host(u32 pc); // route pc to the runtime (hooks)
