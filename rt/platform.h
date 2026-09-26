// Thin OS layer: virtual memory, positional file I/O, directories, exit.
#pragma once
#include "common.h"

void *plat_reserve(size_t size);                 // address space, no access
bool plat_commit(void *p, size_t size);          // fresh zeroed read/write pages
void plat_decommit(void *p, size_t size);        // back to no-access
void plat_protect(void *p, size_t size, int prot);   // prot: 1 = read, 2 = write

struct PFile;
PFile *pf_open(const char *path, bool write, bool create);
void pf_close(PFile *f);
s64 pf_pread(PFile *f, void *buf, size_t n, u64 off);
s64 pf_pwrite(PFile *f, const void *buf, size_t n, u64 off);
u64 pf_size(PFile *f);
bool pf_truncate(PFile *f, u64 size);
void plat_mkdir(const char *path);
[[noreturn]] void plat_exit(int code);
