// The user's .3ds image (decrypted CCI): code.bin comes out of ExeFS at
// boot and RomFS is streamed from it on demand. Nothing is extracted to
// disk, so a build only needs the ROM file itself.
#pragma once
#include "common.h"
#include <string>
#include <vector>

bool rom_open(const std::string &path);          // false + message on error
bool rom_load_code(std::vector<u8> &code);       // decompressed ExeFS .code
s64 rom_romfs_read(void *buf, u64 size, u64 off); // RomFS level-3 (file data)
u64 rom_romfs_size();
u32 code_hash(const u8 *p, size_t n);            // FNV-1a, matches lift.py
