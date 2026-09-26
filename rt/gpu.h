// GPU side: GSP command processing, PICA200 register file, display output.
#pragma once
#include "common.h"

struct Pica;
extern Pica *g_pica;

void pica_init();
void pica_run_cmdlist(u32 pa_or_va, u32 size_bytes, bool is_va);
void gpu_memory_fill(u32 start_va, u32 end_va, u32 value, u32 control);
void gpu_display_transfer(u32 in_va, u32 out_va, u32 in_dim, u32 out_dim, u32 flags);
void gpu_texture_copy(u32 in_va, u32 out_va, u32 size, u32 in_wg, u32 out_wg, u32 flags);

// display
struct FbInfo { u32 addr_va; u32 stride; u32 format; bool valid; };
void display_present(const FbInfo &top, const FbInfo &bottom, u64 vblank_ns = 0);
void display_init();
u32 display_frame_count();
