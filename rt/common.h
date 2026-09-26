// r3ds runtime — shared basic types and logging.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using u8 = std::uint8_t;   using s8 = std::int8_t;
using u16 = std::uint16_t; using s16 = std::int16_t;
using u32 = std::uint32_t; using s32 = std::int32_t;
using u64 = std::uint64_t; using s64 = std::int64_t;

extern int g_log_level;
#define LOG(...)  do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } while (0)
#define DBG(...)  do { if (g_log_level >= 2) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
#define INFO(...) do { if (g_log_level >= 1) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
[[noreturn]] void fatal(const char *fmt, ...);

// float -> integer with the exact semantics of x86-64 (the reference the
// renderer was developed against), so every host (incl. WebAssembly, whose
// conversions saturate) computes the same pixels:
//   f2i: cvttss2si (32-bit): NaN / out of range -> INT_MIN
//   f2u: (u32) via the 64-bit cvttss2si: negative values wrap, NaN / out of
//        int64 range -> 0
static inline s32 f2i(float x) {
    return (x != x || x >= 2147483648.0f || x < -2147483648.0f) ? (s32)0x80000000 : (s32)x;
}
static inline u32 f2u(float x) {
    return (x != x || x >= 9.2233720e18f || x < -9.2233720e18f) ? 0u : (u32)(s64)x;
}
