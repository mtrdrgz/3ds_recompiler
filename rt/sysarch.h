// OS-owned title archives (fs archive 0x2345678A) that a console keeps in NAND
// and a cartridge dump does not carry: Mii component resources, shared fonts,
// country list, NG word list. Games open them during boot and refuse to go on
// when they are missing, so the runtime synthesizes stand-ins in memory: real
// RomFS level-3 images whose files have the documented layout but neutral
// content (no Mii parts, replacement font). Nothing here is Nintendo data.
#pragma once
#include "common.h"
#include <vector>

// Level-3 RomFS image for the OS title `title_id`, or nullptr if unknown.
const std::vector<u8> *sysarch_romfs(u64 title_id);
