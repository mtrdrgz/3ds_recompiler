// Snapshots of what a GPU command list reads from main memory (pica_snap.cpp).
#pragma once
#include "common.h"
#include "mem.h"
#include <atomic>
#include <memory>
#include <vector>

struct GpuSnapSeg { u32 lo, hi, off; };   // VA range, offset into data
struct GpuSnap { std::vector<GpuSnapSeg> segs; std::vector<u8> data; };

// the snapshot of the command list being executed (GPU thread and its pool jobs)
extern std::atomic<const GpuSnap *> g_gpu_snap;
const u8 *gpu_ptr_slow(const GpuSnap *s, u32 va, u32 len);
// guest memory as the executing command list sees it: at submit time
static inline const u8 *gpu_ptr(u32 va, u32 len = 4) {
    const GpuSnap *s = g_gpu_snap.load(std::memory_order_relaxed);
    return s ? gpu_ptr_slow(s, va, len) : gp(va);
}
static inline u32 gpu_rd32(u32 va) { u32 v; memcpy(&v, gpu_ptr(va, 4), 4); return v; }
static inline u32 gpu_rd16(u32 va) { u16 v; memcpy(&v, gpu_ptr(va, 2), 2); return v; }
static inline u32 gpu_rd8(u32 va) { return *gpu_ptr(va, 1); }
static inline u64 gpu_rd64(u32 va) { u64 v; memcpy(&v, gpu_ptr(va, 8), 8); return v; }

// on the game thread, at submit: copy what the list at addr will read.
// skip: VA ranges queued GPU work still writes (left out of the copy).
std::shared_ptr<GpuSnap> gpu_capture(u32 addr, u32 size, const std::vector<std::pair<u32, u32>> &skip);
