#include "platform.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>

void *plat_reserve(size_t size) { return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS); }
bool plat_commit(void *p, size_t size) {
    VirtualFree(p, size, MEM_DECOMMIT);   // guarantees zero-fill on recommit
    return VirtualAlloc(p, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}
void plat_decommit(void *p, size_t size) { VirtualFree(p, size, MEM_DECOMMIT); }
void plat_protect(void *p, size_t size, int prot) {
    DWORD np = (prot & 2) ? PAGE_READWRITE : (prot & 1) ? PAGE_READONLY : PAGE_NOACCESS, old;
    VirtualProtect(p, size, np, &old);
}
struct PFile { HANDLE h; };
PFile *pf_open(const char *path, bool write, bool create) {
    HANDLE h = CreateFileA(path, GENERIC_READ | (write ? GENERIC_WRITE : 0), FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           create ? OPEN_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return nullptr;
    return new PFile{h};
}
void pf_close(PFile *f) { if (f) { CloseHandle(f->h); delete f; } }
s64 pf_pread(PFile *f, void *buf, size_t n, u64 off) {
    OVERLAPPED ov{}; ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32);
    DWORD got = 0;
    if (!ReadFile(f->h, buf, (DWORD)n, &got, &ov)) return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return got;
}
s64 pf_pwrite(PFile *f, const void *buf, size_t n, u64 off) {
    OVERLAPPED ov{}; ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32);
    DWORD put = 0;
    if (!WriteFile(f->h, buf, (DWORD)n, &put, &ov)) return -1;
    return put;
}
u64 pf_size(PFile *f) { LARGE_INTEGER s; GetFileSizeEx(f->h, &s); return (u64)s.QuadPart; }
bool pf_truncate(PFile *f, u64 size) {
    LARGE_INTEGER p; p.QuadPart = (LONGLONG)size;
    return SetFilePointerEx(f->h, p, nullptr, FILE_BEGIN) && SetEndOfFile(f->h);
}
void plat_mkdir(const char *path) { _mkdir(path); }
void plat_exit(int code) { fflush(stderr); ExitProcess(code); }
#else
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __EMSCRIPTEN__
// wasm32: one flat allocation stands in for the reservation. Fresh wasm
// pages are zero; pages the guest gives back are zeroed on decommit so a
// later commit hands out zeroed memory again. No protection exists.
void *plat_reserve(size_t size) {
    // straight from sbrk so the block is guaranteed to be fresh (zero) pages
    u8 *p = (u8 *)sbrk((intptr_t)(size + 0x10000));
    if (p == (u8 *)-1) return nullptr;
    return (void *)(((uintptr_t)p + 0xFFFF) & ~(uintptr_t)0xFFFF);
}
bool plat_commit(void *, size_t) { return true; }
void plat_decommit(void *p, size_t size) { memset(p, 0, size); }
void plat_protect(void *, size_t, int) {}
// "stream:" paths are the ROM, read through the page's streaming worker
s64 web_stream_read(void *buf, size_t n, u64 off);   // web.cpp
u64 web_stream_size();
#else
#include <sys/mman.h>
#include <algorithm>
#include <cstring>
void *plat_reserve(size_t size) {
    void *p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
// The guest maps 4 KB pages; hosts such as macOS on Apple Silicon use 16 KB
// pages. Whole host pages inside a range are replaced by fresh mappings; the
// host pages it only partly covers are shared with neighbouring guest pages,
// so they are made read/write and the guest part is zeroed instead.
static uintptr_t host_page() { static const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE); return pg; }
static bool remap(uintptr_t a, uintptr_t e, int prot, int extra) {
    return a >= e || mmap((void *)a, e - a, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | extra, -1, 0) != MAP_FAILED;
}
static void partial_zero(uintptr_t a, uintptr_t e) {   // [a, e) lies within one host page
    if (a >= e) return;
    uintptr_t pg = host_page(), h = a & ~(pg - 1);
    mprotect((void *)h, pg, PROT_READ | PROT_WRITE);
    memset((void *)a, 0, e - a);
}
static bool zero_range(void *p, size_t size, int prot, int extra) {
    uintptr_t pg = host_page(), a = (uintptr_t)p, e = a + size;
    uintptr_t ia = (a + pg - 1) & ~(pg - 1), ie = e & ~(pg - 1);
    if (ia >= ie) { partial_zero(a, std::min(e, ia)); partial_zero(std::max(a, ie), e); return true; }
    partial_zero(a, ia);
    partial_zero(ie, e);
    return remap(ia, ie, prot, extra);
}
bool plat_commit(void *p, size_t size) {
    return zero_range(p, size, PROT_READ | PROT_WRITE, 0);
}
void plat_decommit(void *p, size_t size) {
    zero_range(p, size, PROT_NONE, MAP_NORESERVE);
}
void plat_protect(void *p, size_t size, int prot) {
    // only host pages wholly inside the range; shared ones stay read/write
    uintptr_t pg = host_page(), a = ((uintptr_t)p + pg - 1) & ~(pg - 1), e = ((uintptr_t)p + size) & ~(pg - 1);
    if (a >= e) return;
    int np = ((prot & 1) ? PROT_READ : 0) | ((prot & 2) ? PROT_WRITE : 0);
    mprotect((void *)a, e - a, np ? np : PROT_NONE);
}
#endif
struct PFile { int fd; bool stream; };
PFile *pf_open(const char *path, bool write, bool create) {
#ifdef __EMSCRIPTEN__
    if (!strncmp(path, "stream:", 7)) return write ? nullptr : new PFile{-1, true};
#endif
    int fl = write ? O_RDWR : O_RDONLY;
    if (create) fl |= O_CREAT;
    int fd = open(path, fl, 0644);
    if (fd < 0) return nullptr;
    return new PFile{fd, false};
}
void pf_close(PFile *f) { if (f) { if (!f->stream) close(f->fd); delete f; } }
s64 pf_pread(PFile *f, void *buf, size_t n, u64 off) {
#ifdef __EMSCRIPTEN__
    if (f->stream) return web_stream_read(buf, n, off);
#endif
    return pread(f->fd, buf, n, (off_t)off);
}
s64 pf_pwrite(PFile *f, const void *buf, size_t n, u64 off) { return f->stream ? -1 : pwrite(f->fd, buf, n, (off_t)off); }
u64 pf_size(PFile *f) {
#ifdef __EMSCRIPTEN__
    if (f->stream) return web_stream_size();
#endif
    struct stat st; fstat(f->fd, &st); return (u64)st.st_size;
}
bool pf_truncate(PFile *f, u64 size) { return !f->stream && ftruncate(f->fd, (off_t)size) == 0; }
void plat_mkdir(const char *path) { mkdir(path, 0755); }
void plat_exit(int code) { fflush(stderr); _exit(code); }
#endif
